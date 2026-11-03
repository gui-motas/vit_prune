"""Command line for vitprune.

Commands for integration, measurement and application:

* ``cost`` -- count operations. No GPU, checkpoint or video needed.
* ``check`` -- verify the dense encoder against the original forward.
* ``calibrate`` -- fit the adaptive budget on calibration videos.
* ``budget-curve`` -- how far each budget moves each clip's embedding from the dense one:
  what the quality-aware budget is calibrated on.
* ``calibrate-quality`` -- fit the quality-aware budget on those records and validate it on
  other drivers. No GPU needed.
* ``region`` -- find the driver region with a person detector, on calibration videos.
* ``latency`` -- time the adaptive budget against the dense path, on real clips.
* ``drift`` -- measure how far the adaptive embedding moves. Needs video, no labels.
* ``run`` -- run on one video and report what came out.
* ``extract`` -- produce embeddings for a whole folder. The expensive stage. Without a
  policy it produces the dense baseline.
* ``evaluate`` -- run a trained downstream model on the embeddings.
* ``benchmark`` -- the experiment, in three steps: ``run`` the variants over videos,
  ``ama`` to run the AMA on each variant's features, ``report`` for the spreadsheet and
  figures, comparing each variant with the dense run. No annotation is needed.

The ordering is not arbitrary. Each command depends on less than the next, which
means results can be produced from the first day, before a dataset is available.

This module only translates arguments and prints. All logic lives in the modules it
calls; when a command starts deciding something, the decision is in the wrong place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vitprune import __version__
from vitprune.core.geometry import ClipGeometry, validate_prune_layer

RULE = "=" * 78
#: Where the three benchmark steps read and write unless told otherwise.
BENCHMARK_OUT = "outputs/benchmark"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (OSError, ValueError, KeyError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def _run_cost(args) -> int:
    from vitprune.bench.cost import budget_cost, corpus_budget, parameter_count, roofline
    from vitprune.bench.hardware import accelerator, custom

    geometry = ClipGeometry.videomae_giant()
    validate_prune_layer(args.prune_layer, geometry.depth)
    tokens = args.tokens
    if args.policy:
        from vitprune.pipeline.calibration import read_policy

        tokens = list(read_policy(args.policy, args.prune_layer).levels)
    device = (
        custom("supplied", args.peak_tflops, args.bandwidth_gbs, args.efficiency)
        if args.peak_tflops and args.bandwidth_gbs
        else accelerator(args.gpu)
    )

    print(RULE)
    print(f"VideoMAE {geometry.describe()}")
    print(f"parameters: {parameter_count(geometry) / 1e9:.2f} billion")
    print(device.describe())
    print(RULE)

    rows = []
    for label, counts, layer in (
        ("dense", [geometry.num_tokens], 0),
        ("adaptive budget", tokens, args.prune_layer),
    ):
        cost = budget_cost(geometry, counts, layer)
        limit = roofline(geometry, cost, device)
        budget = corpus_budget(args.hours, cost, device, args.fps, args.stride)

        print(f"\n{label}: {', '.join(map(str, counts))} tokens per clip")
        print(f"  {cost.describe()}")
        print(f"  {limit.describe()}")
        print(f"  {budget.describe()}")
        if not limit.compute_bound:
            print("  note: weight reading now dominates; pruning further will not pay")

        rows.append(
            {
                "label": label,
                "tokens": counts,
                "prune_layer": layer,
                "gmacs_pruned": cost.gmacs_pruned,
                "predicted_speedup": cost.predicted_speedup,
                "compute_ms": limit.compute_ms,
                "weights_ms": limit.weights_ms,
                "gpu_hours": budget.gpu_hours,
            }
        )

    print("\nThese are estimates. The levels are taken as equally likely, as on the")
    print("calibration clips; an extraction manifest has the counts actually kept. They")
    print("ignore the cost of the statistics, sorting scores and merging tokens, and")
    print("assume every saved operation becomes saved time. Use 'vitprune latency' for")
    print("the measured figure.")
    _maybe_save(args, {"device": device.name, "points": rows})
    return 0


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def _run_latency(args) -> int:
    from vitprune.bench.latency import measure_plans
    from vitprune.core.encoder import PruningPlan
    from vitprune.pipeline.calibration import read_policy
    from vitprune.pipeline.video import gather_clips

    geometry = ClipGeometry.videomae_giant()
    plan = PruningPlan(
        prune_layer=args.prune_layer,
        size_biased_attention=args.size_biased_attention,
        budget=read_policy(args.policy, args.prune_layer, args.frame_step, args.weights,
                           args.precision),
    )
    backbone = _load_backbone(args)
    print(RULE)
    print(f"measuring on {args.device}, precision {args.precision}, batch {args.batch}")
    print(f"{args.warmup} warmup runs, {args.iterations} timed")
    print(RULE)

    points = measure_plans(
        backbone,
        gather_clips(args.videos, geometry, args.stride, args.frame_step, args.batch),
        [PruningPlan.dense(), plan],
        geometry,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        precision=args.precision,
    )

    print("\nRealisation is measured speedup divided by predicted. Below 100%, part of")
    print("the saving went into costs the operation count cannot see. That is a")
    print("finding, not a defect.")
    _maybe_save(args, {"points": [p.to_dict() for p in points]})
    return 0


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------
def _run_drift(args) -> int:
    from vitprune.bench.drift import measure_drift
    from vitprune.pipeline.video import gather_clips

    if args.clips < 1:
        raise ValueError("clips must be positive")
    if not args.policy:
        raise ValueError("drift compares the adaptive budget against the dense path; pass --policy")
    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)
    backbone = _load_backbone(args)
    clips = gather_clips(args.videos, geometry, args.stride, args.frame_step, args.clips)

    print(RULE)
    print(f"{clips.shape[0]} clips | {plan.describe()}")
    print(RULE)

    report = measure_drift(backbone, clips, plan, geometry, args.device, args.precision)
    print(report.describe())
    print("\nDrift is not accuracy loss. An embedding can move a long way and still")
    print("carry what a downstream model needs, or move little and have lost exactly")
    print("what mattered. Use 'vitprune evaluate' to find out.")
    _maybe_save(args, report.to_dict())
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _run_single(args) -> int:
    import time

    import numpy as np
    import torch

    from vitprune.bench.cost import as_dict, budget_cost
    from vitprune.bench.latency import autocast
    from vitprune.bench.report import EmbeddingSummary, RunReport, TimingSummary, summarise_pruning
    from vitprune.core.encoder import PrunedEncoder, PruningPlan
    from vitprune.pipeline.extraction import FEATURE_KEY, encode_clips, extraction_settings
    from vitprune.pipeline.provenance import Manifest, SourceFile
    from vitprune.pipeline.video import iter_clips

    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)
    if args.inspect_clips < 1 or not 0 <= args.top_rows <= geometry.grid_size:
        raise ValueError("inspect-clips must be positive and top-rows within the grid")
    destination = Path(args.out)
    stem = Path(args.video).stem
    for name in (f"{stem}.npz", "report.json", "manifest.json"):
        if (destination / name).exists():
            raise FileExistsError(f"refusing to overwrite {destination / name}")

    backbone = _load_backbone(args)
    encoder = PrunedEncoder(backbone, geometry, plan)
    print(f"[1/3] backbone loaded | {plan.describe()}")

    def stream(spent: list[float], kept: list[torch.Tensor]):
        """The clips of the video, one at a time, timing what reading them costs."""
        clips = iter_clips(args.video, geometry, args.stride, args.frame_step)
        while True:
            started = time.perf_counter()
            item = next(clips, None)
            spent[0] += time.perf_counter() - started
            if item is None:
                return
            if len(kept) < args.inspect_clips:
                kept.append(item[1])
            yield item[1]

    read, sample = [0.0], []
    started = time.perf_counter()
    embeddings, budget = encode_clips(
        encoder, stream(read, sample), args.batch, args.device, args.precision
    )
    read_seconds = read[0]
    encode_seconds = time.perf_counter() - started - read_seconds
    clip_count = int(embeddings.shape[0])
    print(f"[2/3] {clip_count} clips: video read in {read_seconds:.1f}s, "
          f"embeddings in {encode_seconds:.1f}s")

    measured = None
    if args.compare and plan.prunes:
        print("note: exploratory timing without warmup; use 'vitprune latency' for benchmarks")
        dense = PrunedEncoder(backbone, geometry, PruningPlan.dense())
        reread = [0.0]
        started = time.perf_counter()
        encode_clips(dense, stream(reread, []), args.batch, args.device, args.precision)
        measured = (time.perf_counter() - started - reread[0]) / encode_seconds
        print(f"[3/3] dense baseline measured: speedup {measured:.2f}x")
    else:
        print("[3/3] no dense baseline (use --compare to measure the real speedup)")

    pruning = None
    if plan.prunes:
        with autocast(args.device, args.precision):
            pruning = summarise_pruning(
                encoder.select(torch.stack(sample).to(args.device)), geometry, args.top_rows
            )

    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / f"{stem}.npz", **{FEATURE_KEY: embeddings})

    cost = (
        budget_cost(geometry, budget["tokens"], plan.prune_layer)
        if budget is not None
        else budget_cost(geometry, [geometry.num_tokens], 0)
    )
    report = RunReport(
        video=str(Path(args.video).resolve()),
        plan=plan.describe(),
        geometry=geometry.describe(),
        embeddings=EmbeddingSummary.of(embeddings),
        timing=TimingSummary(
            read_seconds=round(read_seconds, 2),
            encode_seconds=round(encode_seconds, 2),
            clips=clip_count,
            predicted_speedup=round(cost.predicted_speedup, 3),
            measured_speedup=None if measured is None else round(measured, 3),
        ),
        pruning=pruning,
        output_files=[f"{stem}.npz", "report.json", "manifest.json"],
    )
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    Manifest(
        settings=extraction_settings(
            encoder, args.stride, args.frame_step, args.batch,
            args.device, args.precision, args.model_source,
        ),
        weights=SourceFile.of(args.weights),
        cost=as_dict(cost),
        outputs={stem: {
            "source": as_dict(SourceFile.of(args.video)),
            "feature_file": as_dict(SourceFile.of(destination / f"{stem}.npz")),
            "clips": clip_count,
            **({"budget": budget} if budget is not None else {}),
        }},
        statistics={"read_seconds": read_seconds, "encode_seconds": encode_seconds},
    ).write(destination)

    print()
    print(report.render())
    print(f"\nwritten to {destination}")
    return 0


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def _run_extract(args) -> int:
    from vitprune.core.encoder import PrunedEncoder
    from vitprune.pipeline.extraction import extract_folder

    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)
    backbone = _load_backbone(args)

    print(RULE)
    print(f"{plan.describe()} | stride {args.stride} | precision {args.precision}")
    print(RULE)

    manifest = extract_folder(
        PrunedEncoder(backbone, geometry, plan),
        args.videos,
        args.out,
        stride=args.stride,
        frame_step=args.frame_step,
        batch=args.batch,
        device=args.device,
        precision=args.precision,
        weights_path=args.weights,
        model_source=args.model_source,
    )
    print(f"run record: {manifest}")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
def _run_calibrate(args) -> int:
    from vitprune.core.encoder import PrunedEncoder, PruningPlan
    from vitprune.pipeline.calibration import calibrate_videos, write_policy

    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite the policy at {args.out}")
    geometry = ClipGeometry.videomae_giant()
    backbone = _load_backbone(args)
    encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=args.prune_layer))

    print(RULE)
    print(f"calibrating levels {args.levels} after block {args.prune_layer}")
    print("calibration videos must not be part of the evaluation")
    print(RULE)

    content = calibrate_videos(
        encoder,
        args.videos,
        args.levels,
        motion_balance=args.motion_balance,
        stride=args.stride,
        frame_step=args.frame_step,
        batch=args.batch,
        device=args.device,
        precision=args.precision,
        per_video=args.clips_per_video,
        weights_path=args.weights,
    )
    path = write_policy(args.out, content)
    policy = content["policy"]
    print(f"\npolicy {policy['version']}: thresholds {policy['thresholds']}")
    print(f"from {content['calibration']['clips']} clips, written to {path}")
    return 0


# ---------------------------------------------------------------------------
# budget-curve
# ---------------------------------------------------------------------------
def _run_budget_curve(args) -> int:
    from vitprune.core.encoder import PrunedEncoder, PruningPlan
    from vitprune.pipeline.quality import SUMMARY, collect_records

    geometry = ClipGeometry.videomae_giant()
    plan = PruningPlan(prune_layer=args.prune_layer, motion_weight=args.motion_weight,
                       size_biased_attention=args.size_biased_attention)
    backbone = _load_backbone(args)

    print(RULE)
    print(f"levels {args.levels} after block {args.prune_layer}, measured against the dense path")
    print("videos used for calibration must not be evaluated")
    print(RULE)

    folder = collect_records(
        PrunedEncoder(backbone, geometry, plan),
        args.videos,
        args.out,
        args.levels,
        clips_per_video=args.clips_per_video,
        depths=args.depths,
        stride=args.stride,
        frame_step=args.frame_step,
        batch=args.batch,
        device=args.device,
        precision=args.precision,
        weights_path=args.weights,
        head=getattr(backbone, "head", None),
    )
    summary = json.loads((folder / SUMMARY).read_text(encoding="utf-8"))
    print(f"\n{summary['clips']} clips of {summary['videos']} videos, cosine to the dense path:")
    for level, found in summary["levels"].items():
        cos, top1 = found["cos"], found["top1"]
        agreement = "" if top1 is None else f" | top-1 as dense {top1:.0%}"
        print(f"  {level:>5} tokens: P10 {cos['p10']:.3f}, P50 {cos['p50']:.3f}, "
              f"P90 {cos['p90']:.3f}{agreement}")
    print(f"\nrecords in {folder}")
    return 0


def _run_calibrate_quality(args) -> int:
    from vitprune.pipeline.calibration import write_policy
    from vitprune.pipeline.quality import describe_validation, fit_quality_policy

    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite the policy at {args.out}")
    print(RULE)
    print(f"fit on the records in {args.records}, validated on {', '.join(args.holdout)}")
    print("the holdout must share no driver with the fit; neither may be evaluated")
    print(RULE)
    content = fit_quality_policy(
        args.records, args.holdout, tolerance=args.tolerance, bins=args.bins,
        min_support=args.min_support, min_videos=args.min_videos, no_probe=args.no_probe,
    )
    path = write_policy(args.out, content)
    print(describe_validation(content["validation"]))
    print(f"\npolicy {content['policy']['version']} written to {path}")
    return 0


# ---------------------------------------------------------------------------
# region
# ---------------------------------------------------------------------------
def _run_region(args) -> int:
    import torch

    from vitprune.pipeline.region import calibrate_region, load_yolo, render_region, write_region

    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite the region at {args.out}")
    geometry = ClipGeometry.videomae_giant()
    detector, record = load_yolo(args.detector, args.device)

    print(RULE)
    print("finding the driver in the middle frame of each clip")
    print("calibration videos must not be part of the evaluation")
    print(RULE)

    content = calibrate_region(
        args.videos, detector, geometry, args.stride, args.frame_step, args.clips_per_video, record
    )
    path = write_region(args.out, content)
    calibration = content["calibration"]
    print(f"\ndriver found in {calibration['detected']} of {calibration['clips']} clips")
    print(f"region: {calibration['area']:.0%} of the frame, written to {path}\n")
    print(render_region(torch.tensor(content["region"]["prior"]), geometry.grid_size))
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _run_evaluate(args) -> int:
    from vitprune.pipeline.evaluation import DownstreamModel, EvaluationRequest, evaluate

    model = DownstreamModel(
        project_dir=args.downstream_project,
        config=args.downstream_config,
        weights=args.downstream_weights,
        entry_point=args.entry_point,
    )
    request = EvaluationRequest(
        features=args.features,
        model=model,
        destination=args.out,
        labels=args.labels,
        stride=args.stride,
        frame_step=args.frame_step,
        extra_arguments=args.extra or [],
    )

    print(RULE)
    print("evaluating with no weight altered, in the backbone or downstream")
    print(RULE)
    evaluate(request)
    return 0


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def _run_benchmark_run(args) -> int:
    from vitprune.pipeline.benchmark.inference import run_benchmark
    from vitprune.pipeline.benchmark.settings import Benchmark

    policies = []
    for item in args.policies or ():
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"--policies takes NAME=PATH, got {item!r}")
        policies.append((name, path))
    bench = Benchmark(
        videos=args.videos,
        out=args.out,
        policy=args.policy,
        region=args.region,
        weights=args.weights,
        model_source=args.model_source,
        evaluate=args.evaluate or (),
        device=args.device,
        precision=args.precision,
        batch=args.batch,
        stride=args.stride,
        frame_step=args.frame_step,
        prune_layer=args.prune_layer,
        fixed_rows=args.fixed_rows,
        fixed_weight=args.fixed_weight,
        show=args.show,
        learnable_positions=args.learnable_positions,
        policies=tuple(policies),
        fixed=tuple(args.fixed or ()),
        proportional=tuple(args.proportional or ()),
    )

    print(RULE)
    if bench.policy_study:
        print(f"the dense backbone and {len(bench.variants) - 1} budgets, no mask, on every clip")
    else:
        print("the dense backbone and three ViT-Prune variants, on every clip")
    print("evaluated videos must not be among the calibration videos")
    print(RULE)
    run_benchmark(bench)
    return 0


def _run_benchmark_ama(args) -> int:
    from vitprune.pipeline.benchmark.ama import run_ama

    print(RULE)
    print("the AMA on the features of each variant, no weight altered")
    print(RULE)
    run_ama(args.out, args.ama_project, args.ama_config, args.ama_weights, args.ama_python,
            labels=args.labels)
    return 0


def _run_benchmark_report(args) -> int:
    try:
        from vitprune.pipeline.benchmark.report import make_report
    except ImportError as error:
        raise RuntimeError(
            "the report needs pandas, openpyxl and matplotlib: pip install 'vitprune[benchmark]'"
        ) from error

    make_report(args.out, args.min_score, args.tiou)
    return 0


def _without_step(parser: argparse.ArgumentParser):
    def handler(args) -> int:
        parser.print_help()
        return 1

    return handler


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------
def _load_backbone(args):
    from vitprune.core import check_backbone
    from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae

    model = load_videomae(
        args.weights,
        args.model_source,
        geometry=ClipGeometry.videomae_giant(),
        device=args.device,
        learnable_positions=args.learnable_positions,
    )
    model = disable_activation_checkpointing(model)
    reference = getattr(model, "forward_features", None)
    if not callable(reference):
        raise ValueError(
            "CLI backbones must expose forward_features returning clip embeddings. "
            "For another reference, integrate through check_backbone in Python."
        )
    report = check_backbone(model, ClipGeometry.videomae_giant(), reference=reference)
    print(report.describe())
    return model


def _run_check(args) -> int:
    _load_backbone(args)
    return 0


def _plan_from(args, geometry: ClipGeometry):
    """The adaptive plan of a policy file, or the dense baseline without one."""
    from vitprune.core.encoder import PruningPlan
    from vitprune.core.priors import top_rows_prior
    from vitprune.pipeline.calibration import read_policy

    if not args.policy:
        options = {
            "--prior-top-rows": args.prior_top_rows,
            "--region": args.region,
            "--motion-weight": args.motion_weight is not None,
            "--size-biased-attention": args.size_biased_attention,
        }
        given = [name for name, value in options.items() if value]
        if given:
            raise ValueError(f"{', '.join(given)} only shape what a policy prunes; pass --policy")
        return PruningPlan.dense()
    if args.region and args.prior_top_rows:
        raise ValueError("choose either --region or --prior-top-rows, not both")
    prior = None
    if args.region:
        from vitprune.pipeline.region import read_region

        prior = read_region(args.region, geometry)
    elif args.prior_top_rows:
        prior = top_rows_prior(geometry.grid_size, args.prior_top_rows, args.prior_weight)
    plan = PruningPlan(
        prune_layer=args.prune_layer,
        motion_weight=0.5 if args.motion_weight is None else args.motion_weight,
        spatial_prior=prior,
        size_biased_attention=args.size_biased_attention,
        budget=read_policy(args.policy, args.prune_layer, args.frame_step, args.weights,
                           args.precision),
    )
    plan.validate_for(geometry)
    return plan


def _maybe_save(args, payload) -> None:
    target = getattr(args, "save", None)
    if not target:
        return
    from dataclasses import asdict

    from vitprune.pipeline.provenance import SourceFile, describe_environment

    payload["environment"] = describe_environment()
    payload["arguments"] = {key: value for key, value in vars(args).items() if key != "handler"}
    if getattr(args, "weights", None):
        payload["weights"] = asdict(SourceFile.of(args.weights))
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {path}")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", required=True, help="backbone checkpoint")
    parser.add_argument(
        "--model-source",
        required=True,
        help="VideoMAE folder containing models/modeling_feature.py",
    )
    parser.add_argument("--device", default="cuda", help="where to run (default: cuda)")
    parser.add_argument(
        "--learnable-positions",
        action="store_true",
        help="use if the checkpoint carries a learned positional embedding",
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        help="adaptive budget written by 'vitprune calibrate'; without it, the dense baseline",
    )
    parser.add_argument(
        "--prune-layer", type=int, default=2, help="prune after this block (default: 2)"
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        help="0 uses uniqueness only, 1 uses motion only (default: 0.5)",
    )
    parser.add_argument(
        "--prior-top-rows",
        type=int,
        default=0,
        help="lower the score of the top N grid rows; check a real frame first",
    )
    parser.add_argument("--prior-weight", type=float, default=0.15)
    parser.add_argument(
        "--region",
        help="driver region written by 'vitprune region'; slots are filled from inside it first",
    )
    parser.add_argument(
        "--size-biased-attention",
        action="store_true",
        help="experimental: weight attention by merged token size",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stride", type=int, default=16, help="frames between clips (default: 16)")
    parser.add_argument(
        "--frame-step", type=int, default=2, help="take every Nth frame within a clip (default: 2)"
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "off"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vitprune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"vitprune {__version__}")
    commands = parser.add_subparsers(dest="command")

    check = commands.add_parser("check", help="verify dense equivalence to the original backbone")
    _add_model_arguments(check)
    check.set_defaults(handler=_run_check)

    calibrate = commands.add_parser("calibrate", help="fit the adaptive budget on videos")
    _add_model_arguments(calibrate)
    _add_run_arguments(calibrate)
    calibrate.add_argument("--videos", required=True, help="calibration videos, not evaluated")
    calibrate.add_argument("--out", required=True, help="policy file to write")
    calibrate.add_argument(
        "--levels", type=int, nargs="+", default=[256, 512, 768], help="tokens per regime"
    )
    calibrate.add_argument("--motion-balance", type=float, default=0.5)
    calibrate.add_argument("--prune-layer", type=int, default=2)
    calibrate.add_argument(
        "--clips-per-video", type=int,
        help="clips taken from each video, spread over it (default: every clip)",
    )
    calibrate.set_defaults(handler=_run_calibrate)

    curve = commands.add_parser(
        "budget-curve", help="per-clip drift of each budget against the dense path"
    )
    _add_model_arguments(curve)
    _add_run_arguments(curve)
    curve.set_defaults(precision="off")
    curve.add_argument("--videos", required=True, nargs="+", help="video files or folders")
    curve.add_argument("--out", required=True, help="records folder; resumed video by video")
    curve.add_argument(
        "--levels", type=int, nargs="+", default=[512, 1024, 1536, 2048],
        help="tokens kept per clip; the full level is always measured (default: %(default)s)",
    )
    curve.add_argument(
        "--depths", type=int, nargs="*", default=[],
        help="blocks after which the drift of the token mean is also recorded",
    )
    curve.add_argument(
        "--clips-per-video", type=int, default=48,
        help="clips spread over each video (default: %(default)s)",
    )
    curve.add_argument("--prune-layer", type=int, default=2)
    curve.add_argument("--motion-weight", type=float, default=0.5)
    curve.add_argument(
        "--size-biased-attention", action="store_true",
        help="merged tokens weigh in attention by their size (experimental)",
    )
    curve.set_defaults(handler=_run_budget_curve)

    quality = commands.add_parser(
        "calibrate-quality", help="fit and validate the quality-aware budget; no GPU needed"
    )
    quality.add_argument("--records", required=True, help="written by 'vitprune budget-curve'")
    quality.add_argument(
        "--holdout", required=True, nargs="+", metavar="STEM",
        help="videos kept out of the fit to validate on; no driver may be on both sides",
    )
    quality.add_argument("--out", required=True, help="policy file to write; never replaced")
    quality.add_argument(
        "--tolerance", type=float, default=0.10,
        help="largest drift 1 - cos accepted at the risk quantile (default: %(default)s)",
    )
    quality.add_argument("--bins", type=int, default=6)
    quality.add_argument("--min-support", type=int, default=30, help="clips per bin to trust it")
    quality.add_argument("--min-videos", type=int, default=3, help="videos per bin to trust it")
    quality.add_argument(
        "--no-probe", action="store_true",
        help="write the ablation: the same risk table without the probe",
    )
    quality.set_defaults(handler=_run_calibrate_quality)

    region = commands.add_parser("region", help="find the driver region with a person detector")
    region.add_argument("--videos", required=True, help="calibration videos, not evaluated")
    region.add_argument(
        "--detector", required=True,
        help="weights of a COCO detector Ultralytics loads, such as yolo26s.pt",
    )
    region.add_argument("--out", required=True, help="region file to write")
    region.add_argument("--stride", type=int, default=16)
    region.add_argument("--frame-step", type=int, default=2)
    region.add_argument(
        "--clips-per-video", type=int,
        help="clips taken from each video, spread over it (default: every clip)",
    )
    region.add_argument("--device", default="cuda", help="where the detector runs (default: cuda)")
    region.set_defaults(handler=_run_region)

    cost = commands.add_parser("cost", help="estimate cost; needs no GPU or checkpoint")
    cost.add_argument(
        "--tokens", type=int, nargs="+", default=[256, 512, 768],
        help="tokens kept per clip in each regime (default: 256 512 768)",
    )
    cost.add_argument("--policy", help="take the levels from this policy file instead")
    cost.add_argument("--prune-layer", type=int, default=2)
    cost.add_argument("--gpu", default="h100-sxm", help="accelerator name (default: h100-sxm)")
    cost.add_argument("--peak-tflops", type=float, help="describe a card outside the catalogue")
    cost.add_argument("--bandwidth-gbs", type=float)
    cost.add_argument("--efficiency", type=float, default=0.5)
    cost.add_argument("--hours", type=float, default=7.2, help="hours of video in the corpus")
    cost.add_argument("--fps", type=int, default=30)
    cost.add_argument("--stride", type=int, default=16)
    cost.add_argument("--save", help="write the result as JSON")
    cost.set_defaults(handler=_run_cost)

    latency = commands.add_parser("latency", help="measure wall-clock against the dense path")
    _add_model_arguments(latency)
    latency.add_argument("--policy", required=True, help="the adaptive budget to time")
    latency.add_argument("--videos", required=True, help="clips to time; the budget needs content")
    latency.add_argument("--stride", type=int, default=16)
    latency.add_argument("--frame-step", type=int, default=2)
    latency.add_argument("--prune-layer", type=int, default=2)
    latency.add_argument("--batch", type=int, default=8)
    latency.add_argument("--warmup", type=int, default=5)
    latency.add_argument("--iterations", type=int, default=20)
    latency.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "off"])
    latency.add_argument("--size-biased-attention", action="store_true")
    latency.add_argument("--save")
    latency.set_defaults(handler=_run_latency)

    drift = commands.add_parser("drift", help="measure embedding drift; needs no labels")
    _add_model_arguments(drift)
    _add_plan_arguments(drift)
    drift.add_argument("--videos", required=True)
    drift.add_argument("--clips", type=int, default=200, help="how many clips to use")
    drift.add_argument("--stride", type=int, default=16)
    drift.add_argument("--frame-step", type=int, default=2)
    drift.add_argument("--precision", default="off", choices=["fp16", "bf16", "off"])
    drift.add_argument("--save")
    drift.set_defaults(handler=_run_drift)

    run = commands.add_parser("run", help="run on one video and report what came out")
    _add_model_arguments(run)
    _add_plan_arguments(run)
    _add_run_arguments(run)
    run.add_argument("--video", required=True, help="a single video file")
    run.add_argument("--out", default="outputs/run")
    run.add_argument(
        "--compare",
        action="store_true",
        help="also run the dense baseline to measure the real speedup (doubles the time)",
    )
    run.add_argument(
        "--inspect-clips", type=int, default=4, help="clips to inspect for the summary"
    )
    run.add_argument("--top-rows", type=int, default=4, help="top rows counted in the budget share")
    run.set_defaults(handler=_run_single)

    extract = commands.add_parser("extract", help="produce embeddings; the expensive stage")
    _add_model_arguments(extract)
    _add_plan_arguments(extract)
    _add_run_arguments(extract)
    extract.add_argument("--videos", required=True)
    extract.add_argument("--out", required=True)
    extract.set_defaults(handler=_run_extract)

    evaluate = commands.add_parser("evaluate", help="run a trained downstream model")
    evaluate.add_argument("--features", required=True, help="folder of embeddings")
    evaluate.add_argument("--downstream-project", required=True, help="folder holding the model")
    evaluate.add_argument("--downstream-config", required=True)
    evaluate.add_argument("--downstream-weights", required=True)
    evaluate.add_argument("--entry-point", default="eval.py")
    evaluate.add_argument("--labels", help="annotation file for the evaluation split")
    evaluate.add_argument("--frame-step", type=int, help="assert the extraction frame step")
    evaluate.add_argument(
        "--stride", type=int, help="only if no manifest exists; a mismatch aborts"
    )
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--extra", nargs=argparse.REMAINDER)
    evaluate.set_defaults(handler=_run_evaluate)

    benchmark = commands.add_parser(
        "benchmark",
        help="the experiment: the variants on videos, the AMA, the report against the dense run",
    )
    benchmark.set_defaults(handler=_without_step(benchmark))
    steps = benchmark.add_subparsers(dest="step")

    bench_run = steps.add_parser("run", help="every clip through the dense and pruned variants")
    bench_run.add_argument(
        "--videos", required=True,
        help="the videos under evaluation (a folder or a file), not the calibration videos",
    )
    bench_run.add_argument(
        "--evaluate", nargs="+", metavar="STEM",
        help="names of the videos to evaluate, without extension (default: every video)",
    )
    bench_run.add_argument(
        "--policy",
        help="written by 'vitprune calibrate'; the mask study needs it, the policy study runs "
             "it as legacy_original",
    )
    bench_run.add_argument(
        "--policies", nargs="+", metavar="NAME=PATH",
        help="policy study: policy files to compare, legacy or quality-aware, as variants NAME",
    )
    bench_run.add_argument(
        "--fixed", nargs="+", type=int, metavar="TOKENS",
        help="policy study: fixed budgets to compare, as variants fixo_TOKENS",
    )
    bench_run.add_argument(
        "--proportional", nargs="+", type=int, metavar="TOKENS",
        help="policy study: fixed budgets with size-biased attention, as variants fixo_TOKENS_pa",
    )
    bench_run.add_argument("--region", required=True, help="written by 'vitprune region'")
    _add_model_arguments(bench_run)
    bench_run.add_argument(
        "--out", default=BENCHMARK_OUT, help="results folder (default: %(default)s)"
    )
    bench_run.add_argument(
        "--stride", type=int, default=16, help="frames between clips (default: 16)"
    )
    bench_run.add_argument(
        "--frame-step", type=int, default=2, help="take every Nth frame within a clip (default: 2)"
    )
    bench_run.add_argument("--batch", type=int, default=8)
    bench_run.add_argument(
        "--precision", default="off", choices=["fp16", "bf16", "off"],
        help="off is fp32, as the AI City extraction ran; calibrate with the same (default: off)",
    )
    bench_run.add_argument(
        "--prune-layer", type=int, default=2, help="prune after this block (default: 2)"
    )
    bench_run.add_argument(
        "--fixed-rows", type=int, default=4,
        help="rows from the top the fixed mask lowers (default: 4)",
    )
    bench_run.add_argument(
        "--fixed-weight", type=float, default=0.15,
        help="multiplier the fixed mask applies to those rows (default: 0.15)",
    )
    bench_run.add_argument(
        "--show", type=int, default=6,
        help="clips per video kept for the figures, spread over it (default: 6)",
    )
    bench_run.set_defaults(handler=_run_benchmark_run)

    bench_ama = steps.add_parser("ama", help="the AMA on each variant's features")
    bench_ama.add_argument(
        "--out", default=BENCHMARK_OUT, help="results folder of 'run' (default: %(default)s)"
    )
    bench_ama.add_argument("--ama-project", required=True, help="folder holding the AMA")
    bench_ama.add_argument("--ama-config", required=True, help="the AMA's configuration")
    bench_ama.add_argument("--ama-weights", required=True, help="the AMA's trained weights")
    bench_ama.add_argument(
        "--ama-python",
        help="interpreter of the AMA's environment, if not the one running vitprune",
    )
    bench_ama.add_argument(
        "--labels",
        help="annotations of the evaluated videos (ActivityNet JSON, as the AMA's); with them, "
             "the AMA's mAP per tIoU against them, per variant",
    )
    bench_ama.set_defaults(handler=_run_benchmark_ama)

    bench_report = steps.add_parser("report", help="the spreadsheet and the figures")
    bench_report.add_argument(
        "--out", default=BENCHMARK_OUT, help="results folder of 'run' (default: %(default)s)"
    )
    bench_report.add_argument(
        "--min-score", type=float, default=0.3,
        help="detections below this confidence are left out, in the dense reference and in "
             "each variant (default: 0.3)",
    )
    bench_report.add_argument(
        "--tiou", type=float, default=0.5,
        help="overlap at which a variant's detection matches one of the dense (default: 0.5)",
    )
    bench_report.set_defaults(handler=_run_benchmark_report)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
