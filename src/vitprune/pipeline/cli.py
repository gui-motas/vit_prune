"""Command line for vitprune.

Six commands, ordered from cheapest to most expensive:

* ``cost`` -- count operations. No GPU, checkpoint or video needed.
* ``latency`` -- measure wall-clock. Needs a GPU and a checkpoint, but no video.
* ``drift`` -- measure embedding drift. Needs video, but no labels.
* ``run`` -- run on one video and report what came out.
* ``extract`` -- produce embeddings for a whole folder. The expensive stage.
* ``evaluate`` -- run a trained downstream model on the embeddings.

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
from vitprune.core.geometry import ClipGeometry

RULE = "=" * 78


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, KeyError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def _run_cost(args) -> int:
    from vitprune.bench.cost import clip_cost, corpus_budget, parameter_count, roofline
    from vitprune.bench.hardware import accelerator, custom

    geometry = ClipGeometry.videomae_giant()
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
    for retention in args.retention:
        layer = 0 if retention >= 1.0 else args.prune_layer
        cost = clip_cost(geometry, retention, layer)
        limit = roofline(geometry, cost, device)
        budget = corpus_budget(args.hours, cost, device, args.fps, args.stride)

        print(f"\nretention {retention:.0%}")
        print(f"  {cost.describe()}")
        print(f"  {limit.describe()}")
        print(f"  {budget.describe()}")
        if not limit.compute_bound:
            print("  note: weight reading now dominates; pruning further will not pay")

        rows.append(
            {
                "retention": retention,
                "prune_layer": layer,
                "gmacs_pruned": cost.gmacs_pruned,
                "predicted_speedup": cost.predicted_speedup,
                "compute_ms": limit.compute_ms,
                "weights_ms": limit.weights_ms,
                "gpu_hours": budget.gpu_hours,
            }
        )

    print("\nThese are estimates. They ignore the cost of sorting scores and merging")
    print("tokens, and assume every saved operation becomes saved time. Use")
    print("'vitprune latency' for the measured figure.")
    _maybe_save(args, {"device": device.name, "points": rows})
    return 0


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def _run_latency(args) -> int:
    from vitprune.bench.latency import measure_latency

    backbone = _load_backbone(args)
    print(RULE)
    print(f"measuring on {args.device}, precision {args.precision}, batch {args.batch}")
    print(f"{args.warmup} warmup runs, {args.iterations} timed")
    print(RULE)

    points = measure_latency(
        backbone,
        args.retention,
        geometry=ClipGeometry.videomae_giant(),
        prune_layer=args.prune_layer,
        batch=args.batch,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        precision=args.precision,
        size_biased_attention=args.size_biased_attention,
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
    import torch

    from vitprune.bench.drift import measure_drift
    from vitprune.pipeline.video import find_videos, read_video, to_clips

    backbone = _load_backbone(args)
    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)

    videos = find_videos(args.videos)
    if not videos:
        raise FileNotFoundError(f"no video found in {args.videos}")

    gathered, total = [], 0
    for video in videos:
        gathered.append(to_clips(read_video(video), geometry, args.stride, args.frame_step))
        total += gathered[-1].shape[0]
        if total >= args.clips:
            break
    clips = torch.cat(gathered)[: args.clips]

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

    from vitprune.bench.cost import clip_cost
    from vitprune.bench.report import EmbeddingSummary, RunReport, TimingSummary, summarise_pruning
    from vitprune.core.encoder import PrunedEncoder, PruningPlan
    from vitprune.pipeline.extraction import FEATURE_KEY, encode_clips
    from vitprune.pipeline.video import read_video, to_clips

    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)

    started = time.perf_counter()
    frames = read_video(args.video)
    clips = to_clips(frames, geometry, args.stride, args.frame_step)
    read_seconds = time.perf_counter() - started
    print(f"[1/4] video read: {frames.shape[0]} frames -> {clips.shape[0]} clips")

    backbone = _load_backbone(args)
    encoder = PrunedEncoder(backbone, geometry, plan)
    print(f"[2/4] backbone loaded | {plan.describe()}")

    started = time.perf_counter()
    embeddings = encode_clips(encoder, clips, args.batch, args.device, args.precision)
    encode_seconds = time.perf_counter() - started
    print(f"[3/4] embeddings produced in {encode_seconds:.1f}s")

    measured = None
    if args.compare and plan.prunes:
        dense = PrunedEncoder(backbone, geometry, PruningPlan.dense())
        started = time.perf_counter()
        encode_clips(dense, clips, args.batch, args.device, args.precision)
        measured = (time.perf_counter() - started) / encode_seconds
        print(f"[4/4] dense baseline measured: speedup {measured:.2f}x")
    else:
        print("[4/4] no dense baseline (use --compare to measure the real speedup)")

    pruning = None
    if plan.prunes:
        sample = clips[: min(args.inspect_clips, clips.shape[0])].to(args.device)
        pruning = summarise_pruning(encoder.select(sample), geometry, args.top_rows)

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    np.savez(destination / f"{stem}.npz", **{FEATURE_KEY: embeddings})

    cost = clip_cost(geometry, plan.retention, plan.prune_layer)
    report = RunReport(
        video=str(Path(args.video).resolve()),
        plan=plan.describe(),
        geometry=geometry.describe(),
        embeddings=EmbeddingSummary.of(embeddings),
        timing=TimingSummary(
            read_seconds=round(read_seconds, 2),
            encode_seconds=round(encode_seconds, 2),
            clips=int(clips.shape[0]),
            predicted_speedup=round(cost.predicted_speedup, 3),
            measured_speedup=None if measured is None else round(measured, 3),
        ),
        pruning=pruning,
        output_files=[f"{stem}.npz", "report.json"],
    )
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

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

    backbone = _load_backbone(args)
    geometry = ClipGeometry.videomae_giant()
    plan = _plan_from(args, geometry)

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
    )
    print(f"run record: {manifest}")
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
        extra_arguments=args.extra or [],
    )

    print(RULE)
    print("evaluating with no weight altered, in the backbone or downstream")
    print(RULE)
    evaluate(request)
    return 0


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------
def _load_backbone(args):
    from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae

    model = load_videomae(
        args.weights,
        args.model_source,
        geometry=ClipGeometry.videomae_giant(),
        device=args.device,
        learnable_positions=args.learnable_positions,
    )
    return disable_activation_checkpointing(model)


def _plan_from(args, geometry: ClipGeometry):
    from vitprune.core.encoder import PruningPlan
    from vitprune.core.priors import top_rows_prior

    prior = (
        top_rows_prior(geometry.grid_size, args.prior_top_rows, args.prior_weight)
        if args.prior_top_rows > 0
        else None
    )
    if args.retention >= 1.0:
        return PruningPlan.dense()
    return PruningPlan(
        retention=args.retention,
        prune_layer=args.prune_layer,
        motion_weight=args.motion_weight,
        spatial_prior=prior,
        size_biased_attention=args.size_biased_attention,
    )


def _maybe_save(args, payload) -> None:
    target = getattr(args, "save", None)
    if not target:
        return
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
        "--retention", type=float, default=0.25, help="retention ratio (default: 0.25)"
    )
    parser.add_argument(
        "--prune-layer", type=int, default=2, help="prune after this block (default: 2)"
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.5,
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

    cost = commands.add_parser("cost", help="estimate cost; needs no GPU or checkpoint")
    cost.add_argument("--retention", type=float, nargs="+", default=[1.0, 0.5, 0.35, 0.25, 0.15])
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

    latency = commands.add_parser("latency", help="measure wall-clock; needs no video")
    _add_model_arguments(latency)
    latency.add_argument("--retention", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.15])
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
    evaluate.add_argument(
        "--stride", type=int, help="only if no manifest exists; a mismatch aborts"
    )
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--extra", nargs=argparse.REMAINDER)
    evaluate.set_defaults(handler=_run_evaluate)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
