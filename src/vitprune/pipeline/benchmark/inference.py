"""Run the benchmark: every clip of the evaluated videos through every variant.

The mask study runs the dense VideoMAE and three ViT-Prune variants under one policy. The
policy study runs the dense VideoMAE and budgets -- policy files, fixed budgets, fixed
budgets with size-biased attention and fixed budgets selecting inside the driver region, the
only ones with a mask -- turning the order in which they are
timed from batch to batch, so that none is always timed first. A quality-aware policy
calibrated on an evaluated video or driver is refused; a legacy one is only reported.

The policy and the region come from the command line, calibrated on videos that are not
evaluated here::

    vitprune calibrate --videos <calibration> ... --clips-per-video 160 --out policy.json
    vitprune region --videos <calibration> --detector yolo26s.pt ... --out region.json

Writes, under the results folder:

* ``features/<variant>/<video>.npz`` -- what the AMA reads: ``feats`` (one row per clip)
  and ``a_prob`` (the fine-tuned head's class probabilities), with a manifest per
  variant;
* ``clips.jsonl`` -- one line per clip and variant: the tokens kept, the budget regime,
  the share of the kept tokens inside the driver region (mask study), the cosine and the
  relative L2 distance to the dense embedding, the risk a quality-aware policy predicted
  and why it chose, and the seconds taken;
* ``selection/`` -- the tokens each variant kept in the ``--show`` clips spread over each
  video, for the figures;
* ``frames/`` -- the middle frame of those clips;
* ``run.json`` -- the inputs and settings of the run.

A video is written whole or redone, so an interrupted run resumes video by video.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.bench.latency import autocast
from vitprune.bench.quality import time_probe
from vitprune.core import FixedBudget, PrunedEncoder, PruningPlan, check_backbone
from vitprune.core.policy import QualityAwareBudgetPolicy
from vitprune.core.priors import top_rows_prior
from vitprune.pipeline.benchmark.settings import GEOMETRY, Benchmark
from vitprune.pipeline.calibration import read_policy
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.provenance import Manifest, SourceFile, describe_environment
from vitprune.pipeline.quality import driver_of
from vitprune.pipeline.region import read_region
from vitprune.pipeline.video import _opencv, _spread, iter_clips


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def load_model(bench: Benchmark):
    started = time.time()
    model = load_videomae(bench.weights, bench.model_source, GEOMETRY, device=bench.device,
                          learnable_positions=bench.learnable_positions)
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)
    return disable_activation_checkpointing(model)


def video_info(video: Path) -> dict:
    """The frame rate and the frame count the container reports."""
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(video))
    info = {"fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    capture.release()
    return info


def plan(bench: Benchmark, video: Path, info: dict) -> list[dict]:
    """Every clip of a video. ``bench.show`` of them, spread evenly, are kept for the
    figures."""
    starts = bench.clip_starts(info["frames"])
    shown = set(_spread(starts, bench.show))
    return [{"video": video.stem, "start": start, "fps": info["fps"], "show": start in shown}
            for start in starts]


def save_frames(bench: Benchmark, video: Path, clips: list[dict]) -> None:
    """The middle frame of each shown clip, for the figures."""
    cv2 = _opencv()
    folder = bench.frames / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    for clip in clips:
        frame = folder / f"{clip['start']:06d}.jpg"
        if clip["show"] and not frame.exists():
            capture.set(cv2.CAP_PROP_POS_FRAMES, clip["start"] + bench.span // 2)
            ok, image = capture.read()
            if ok:
                cv2.imwrite(str(frame), image)
    capture.release()


def encoders(bench: Benchmark, model, policies: dict, region) -> dict[str, PrunedEncoder]:
    """One encoder per variant, the dense one first. ``policies`` maps a variant's name to
    its budget; the mask study's policy is ``legacy_original``."""
    layer, plans = bench.prune_layer, {"denso": PruningPlan.dense(attention=bench.attention)}
    if bench.policy_study:
        plans |= {name: PruningPlan(prune_layer=layer, budget=budget, attention=bench.attention)
                  for name, budget in policies.items()}
        plans |= {f"fixo_{k}": PruningPlan(prune_layer=layer, budget=FixedBudget(k),
                                          attention=bench.attention)
                  for k in bench.fixed}
        plans |= {f"fixo_{k}_pa": PruningPlan(prune_layer=layer, budget=FixedBudget(k),
                                              size_biased_attention=True, attention=bench.attention)
                  for k in bench.proportional}
        plans |= {f"fixo_{k}_mascara": PruningPlan(prune_layer=layer, budget=FixedBudget(k),
                                                   spatial_prior=region, attention=bench.attention)
                  for k in bench.masked}
        if bench.progressive:
            plans[bench.progressive_name] = PruningPlan(
                prune_layer=layer, budget=FixedBudget(bench.progressive[0][1]),
                spatial_prior=region, stages=bench.progressive[1:], attention=bench.attention)
    else:
        fixed = top_rows_prior(GEOMETRY.grid_size, bench.fixed_rows, bench.fixed_weight)
        priors = {"vitprune": None, "vitprune_mascara_fixa": fixed,
                  "vitprune_mascara_inteligente": region}
        plans |= {name: PruningPlan(prune_layer=layer, budget=policies["legacy_original"],
                                    spatial_prior=prior, attention=bench.attention)
                  for name, prior in priors.items()}
    return {name: PrunedEncoder(model, GEOMETRY, plans[name]) for name in bench.variants}


def inside_share(kept: torch.Tensor, region: torch.Tensor) -> list[float]:
    """Share of each clip's kept tokens that sit inside the region.

    ``kept`` holds grid positions, ``[clips, slices, kept]``, padded with -1 for a clip
    that kept fewer tokens than the widest in its batch.
    """
    valid = kept >= 0
    hits = (region.to(kept.device) > 0)[kept.clamp(min=0)] & valid
    return (hits.sum(dim=(1, 2)) / valid.sum(dim=(1, 2))).tolist()


@torch.inference_mode()
def evaluate_batch(bench: Benchmark, model, clips: list[dict], tensor: torch.Tensor,
                   runners: dict[str, PrunedEncoder], region: torch.Tensor, turn: int = 0):
    """Every variant on a batch of clips: the rows, and per variant the embeddings and
    class probabilities. In the policy study, the variant timed first is the ``turn``-th."""
    tensor = tensor.to(bench.device)
    shown = {i: {} for i, clip in enumerate(clips) if clip["show"]}
    names = list(runners)
    if bench.policy_study:
        turn %= len(names)
        names = names[turn:] + names[:turn]
    count, found = len(clips), {}
    for name in names:
        encoder = runners[name]
        _sync(bench.device)
        started = time.perf_counter()
        decision = None
        with autocast(bench.device, bench.precision):
            if encoder.plan.prunes:
                embedding, budget = encoder.encode_with_budget(tensor)
                tokens, regimes = budget.tokens.tolist(), budget.regimes.tolist()
                decision = budget.decision
            else:
                embedding = encoder.encode(tensor)
                tokens, regimes = [GEOMETRY.num_tokens] * count, [None] * count
            probabilities = model.head(embedding).float().softmax(dim=-1)
        _sync(bench.device)
        seconds = (time.perf_counter() - started) / count
        shares = [None] * count
        if encoder.plan.prunes and not bench.policy_study:
            # Outside the timing: the same decision again, to see where the tokens went.
            with autocast(bench.device, bench.precision):
                selection = encoder.select(tensor)
            shares = inside_share(selection.kept, region)
            for i, arrays in shown.items():
                arrays[f"{name}_kept"] = np.stack([
                    selection.keep_mask(i, s).flatten() for s in range(selection.temporal_slices)
                ])
                if encoder.plan.spatial_prior is not None:
                    arrays[f"{name}_prior"] = encoder.plan.spatial_prior.cpu().numpy()
        found[name] = (embedding.float(), probabilities, tokens, regimes, decision, shares,
                       seconds)

    dense = found["denso"][0]
    rows, outputs = [], {}
    for name in names:
        embedding, probabilities, tokens, regimes, decision, shares, seconds = found[name]
        cosines = F.cosine_similarity(embedding, dense, dim=-1).tolist()
        relative = ((embedding - dense).norm(dim=-1) / dense.norm(dim=-1)).tolist()
        risks = [None] * count if decision is None else decision.predicted_risk.tolist()
        reasons = [None] * count if decision is None else list(decision.reasons)
        outputs[name] = (embedding.cpu().numpy(), probabilities.cpu().numpy())
        rows += [{
            "video": clip["video"], "start_frame": clip["start"],
            "start_s": round(clip["start"] / clip["fps"], 3),
            "variant": name, "tokens": int(tokens[i]), "regime": regimes[i],
            "inside_region": shares[i], "cosine_to_dense": cosines[i],
            "rel_l2_to_dense": relative[i], "predicted_risk": risks[i], "reason": reasons[i],
            "seconds": round(seconds, 4),
        } for i, clip in enumerate(clips)]
    for i, arrays in shown.items():
        bench.selections.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bench.selections / f"{clips[i]['video']}_{clips[i]['start']:06d}.npz",
                            **arrays)
    return rows, outputs


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append JSON lines and make sure they are on disk before anything is written after."""
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_npz(path: Path, **arrays) -> None:
    """Write an ``.npz`` whole or not at all: a write cut short leaves no file under the name
    a resume takes for done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def keep_rows(path: Path, keep: Callable[[dict], bool]) -> None:
    """Rewrite a JSON-lines file without the rows ``keep`` refuses and without a row an
    interruption cut short, which is the only one that can lack its newline."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if line.endswith("\n") and keep(json.loads(line))]
    if len(kept) != len(lines):
        partial = path.with_name(path.name + ".partial")
        partial.write_text("".join(kept), encoding="utf-8")
        os.replace(partial, path)


def run(bench: Benchmark, model, videos: list[Path], schedules: dict,
        runners: dict, region: torch.Tensor) -> None:
    """Every clip of every video, in batches. A video's rows are written before its features
    and each feature file whole, so a video is done once all its features exist; the rows
    an interruption left of any other video are dropped and the video is redone."""
    targets = {video.stem: {name: bench.features / name / f"{video.stem}.npz" for name in runners}
               for video in videos}
    unfinished = {stem for stem, paths in targets.items()
                  if not all(path.exists() for path in paths.values())}
    keep_rows(bench.clips, lambda row: row["video"] not in unfinished)
    for video in videos:
        if video.stem not in unfinished:
            print(f"{video.stem}: already done", flush=True)
            continue
        rows, outputs = run_video(bench, model, video, schedules[video.stem], runners, region)
        append_rows(bench.clips, rows)
        for name, (embeddings, probabilities) in outputs.items():
            save_npz(targets[video.stem][name], feats=np.concatenate(embeddings),
                     a_prob=np.concatenate(probabilities))


def run_video(bench: Benchmark, model, video: Path, clips: list[dict],
              runners: dict[str, PrunedEncoder], region: torch.Tensor):
    by_start = {c["start"]: c for c in clips}
    rows, outputs = [], {name: ([], []) for name in runners}
    batch, tensors, started, turns = [], [], time.time(), [0]

    def flush() -> None:
        found, produced = evaluate_batch(bench, model, batch, torch.stack(tensors), runners,
                                         region, turns[0])
        turns[0] += 1
        rows.extend(found)
        for name, (embedding, probability) in produced.items():
            outputs[name][0].append(embedding)
            outputs[name][1].append(probability)
        print(f"  {video.stem}: {len(rows) // len(runners)}/{len(by_start)} clips "
              f"({time.time() - started:.0f}s)", flush=True)
        batch.clear()
        tensors.clear()

    for start, clip in iter_clips(video, GEOMETRY, bench.stride, bench.frame_step, by_start):
        batch.append(by_start[start])
        tensors.append(clip)
        if len(batch) == bench.batch:
            flush()
    if batch:
        flush()
    return rows, outputs


def write_manifests(bench: Benchmark, videos: list[Path], runners: dict) -> None:
    """The record ``vitprune evaluate`` and the AMA step read next to the features."""
    rows = [json.loads(line) for line in bench.clips.read_text(encoding="utf-8").splitlines()]
    for name, encoder in runners.items():
        plan = encoder.plan
        outputs = {}
        for video in videos:
            mine = sorted((r for r in rows if r["video"] == video.stem and r["variant"] == name),
                          key=lambda r: r["start_frame"])
            outputs[video.stem] = {
                "feature_file": asdict(SourceFile.of(bench.features / name / f"{video.stem}.npz")),
                "clips": len(mine),
                "tokens": [r["tokens"] for r in mine],
            }
        Manifest(
            settings={"variant": name, "stride": bench.stride, "frame_step": bench.frame_step,
                      "prune_layer": plan.prune_layer if plan.prunes else 0,
                      "policy": plan.budget.version if plan.prunes else None,
                      "size_biased_attention": plan.size_biased_attention,
                      "attention": plan.attention, "stages": [list(stage) for stage in plan.stages],
                      "precision": bench.precision, "device": bench.device,
                      "geometry": asdict(GEOMETRY),
                      "fixed_mask": bench.fixed_mask if name == "vitprune_mascara_fixa" else None,
                      "region": (str(bench.region) if plan.spatial_prior is not None
                                 and name != "vitprune_mascara_fixa" else None)},
            outputs=outputs,
        ).write(bench.features / name)


def refuse_other_runs(record_path: Path, settings: dict, policy: Path | None,
                      region: Path) -> None:
    """Stop when the results folder holds a run of another policy, region or settings:
    resuming next to it would mix two experiments."""
    if not record_path.exists():
        return
    before = json.loads(record_path.read_text(encoding="utf-8"))
    policy = None if policy is None else SourceFile.of(policy).sha256
    if (before.get("settings") != settings
            or (before.get("policy") or {}).get("sha256") != policy
            or before.get("region", {}).get("sha256") != SourceFile.of(region).sha256):
        raise ValueError(f"{record_path.parent} holds results of another configuration; "
                         "move them out or choose another --out")


def calibration_overlap(path: Path, evaluated: dict[str, str]) -> dict:
    """The evaluated videos (``stem -> sha256``) and drivers a policy or region file was
    calibrated on, matched by content or by name. Its videos are a list of entries, as the
    calibrations write them, or ``{stem: box}``, as a region derived from another keeps them."""
    calibration = json.loads(Path(path).read_text(encoding="utf-8")).get("calibration", {})
    videos = calibration.get("videos", [])
    hashes, stems, drivers = set(), set(), set()
    for entry in [{"stem": stem} for stem in videos] if isinstance(videos, dict) else videos:
        source = entry.get("source") or {}
        stem = entry.get("stem") or Path(source.get("path", "")).stem
        hashes.add(source.get("sha256"))
        stems.add(stem)
        drivers.add(entry.get("driver") or driver_of(stem))
    return {"videos": sorted(stem for stem, sha in evaluated.items()
                             if sha in hashes or stem in stems),
            "drivers": sorted({driver_of(stem) for stem in evaluated} & drivers)}


def run_benchmark(bench: Benchmark) -> None:
    """Run the variants over the evaluated videos, resuming a run of the same configuration."""
    bench.out.mkdir(parents=True, exist_ok=True)
    files = ({"legacy_original": bench.policy} if bench.policy is not None else {}) | dict(
        bench.policies)
    policies = {name: read_policy(path, bench.prune_layer, bench.frame_step, bench.weights,
                                  bench.precision) for name, path in files.items()}
    region = read_region(bench.region, GEOMETRY)
    videos = bench.evaluated()
    evaluated = {video.stem: SourceFile.of(video).sha256 for video in videos}
    overlap = {name: calibration_overlap(path, evaluated) for name, path in files.items()}
    if not bench.policy_study or bench.masked or bench.progressive:
        overlap["--region"] = calibration_overlap(bench.region, evaluated)
    tainted = {name: found for name, found in overlap.items()
               if (name == "--region" or isinstance(policies[name], QualityAwareBudgetPolicy))
               and (found["videos"] or found["drivers"])}
    if bench.allow_region_overlap and tainted.pop("--region", None):
        print("warning: the region overlaps the evaluated footage; run as asked by "
              "--allow-region-overlap and recorded in run.json", flush=True)
    if tainted:
        raise ValueError(f"a quality-aware policy or the region calibrated on an evaluated "
                         f"video or driver: {tainted}; calibrate it on other drivers")
    for name, found in overlap.items():
        if found["videos"] or found["drivers"]:
            print(f"warning: {name} was calibrated on evaluated videos {found['videos']} "
                  f"or drivers {found['drivers']}", flush=True)
    infos = {video.stem: video_info(video) for video in videos}
    schedules = {video.stem: plan(bench, video, infos[video.stem]) for video in videos}
    for video in videos:
        save_frames(bench, video, schedules[video.stem])
    settings = {"stride": bench.stride, "frame_step": bench.frame_step,
                "prune_layer": bench.prune_layer, "fixed_mask": bench.fixed_mask,
                "variants": list(bench.variants), "device": bench.device, "batch": bench.batch,
                "precision": bench.precision, "evaluate": [video.stem for video in videos],
                "show": bench.show, "learnable_positions": bench.learnable_positions}
    if bench.policy_study:
        settings |= {"policies": {name: SourceFile.of(path).sha256 for name, path in files.items()},
                     "fixed": list(bench.fixed), "proportional": list(bench.proportional)}
        if bench.masked:  # only when used, so the runs made before it still resume
            settings["masked"] = list(bench.masked)
        if bench.progressive:
            settings["progressive"] = [list(stage) for stage in bench.progressive]
    if bench.attention != "model":
        settings["attention"] = bench.attention
    if bench.allow_region_overlap:
        settings["allow_region_overlap"] = True
    refuse_other_runs(bench.record, settings, bench.policy, bench.region)
    model = load_model(bench)
    if not bench.record.exists():
        report = check_backbone(model, GEOMETRY, reference=model.forward_features)
        print(report.describe(), flush=True)
        record = {
            "check": report.describe(),
            "weights": asdict(SourceFile.of(bench.weights)),
            "model_source": str(bench.model_source),
            "policy": None if bench.policy is None else asdict(SourceFile.of(bench.policy)),
            "region": asdict(SourceFile.of(bench.region)),
            "videos": str(bench.videos),
            "environment": describe_environment(),
            "threads": torch.get_num_threads(),
            "settings": settings,
            "schedules": {stem: {**infos[stem], "clips": len(clips)}
                          for stem, clips in schedules.items()},
            "calibration_overlap": overlap,
        }
        if bench.policy_study:
            record["policies"] = {name: asdict(SourceFile.of(path)) for name, path in files.items()}
            probed = [p for p in policies.values()
                      if isinstance(p, QualityAwareBudgetPolicy) and p.uses_probe]
            if probed:
                record["probe_timing"] = time_probe(GEOMETRY, bench.batch, bench.device,
                                                    cells=probed[0].probe_cells,
                                                    top=probed[0].probe_top)
        bench.record.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if bench.policy_study:
        print(f"variants {list(bench.variants)}", flush=True)
    else:
        policy = policies["legacy_original"]
        print(f"policy {policy.version}, levels {list(policy.levels)}; region "
              f"{float(region.mean()):.0%} of the frame", flush=True)
    runners = encoders(bench, model, {name: policies[name] for name in files}, region)
    run(bench, model, videos, schedules, runners, region)
    write_manifests(bench, videos, runners)
    print("done", flush=True)
