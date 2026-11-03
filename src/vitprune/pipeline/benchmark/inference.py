"""Run the benchmark: every clip of the evaluated videos through the four variants.

The policy and the region come from the command line, calibrated on videos that are not
evaluated here::

    vitprune calibrate --videos <calibration> ... --clips-per-video 160 --out policy.json
    vitprune region --videos <calibration> --detector yolo26s.pt ... --out region.json

Writes, under the results folder:

* ``features/<variant>/<video>.npz`` -- what the AMA reads: ``feats`` (one row per clip)
  and ``a_prob`` (the fine-tuned head's class probabilities), with a manifest per
  variant;
* ``clips.jsonl`` -- one line per clip and variant: the tokens kept, the budget regime,
  the share of the kept tokens inside the driver region, the cosine to the dense
  embedding and the seconds taken;
* ``selection/`` -- the tokens each variant kept in the ``--show`` clips spread over each
  video, for the figures;
* ``frames/`` -- the middle frame of those clips;
* ``run.json`` -- the inputs and settings of the run.

A video is written whole or redone, so an interrupted run resumes video by video.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.bench.latency import autocast
from vitprune.core import PrunedEncoder, PruningPlan, check_backbone
from vitprune.core.priors import top_rows_prior
from vitprune.pipeline.benchmark.settings import GEOMETRY, VARIANTS, Benchmark
from vitprune.pipeline.calibration import read_policy
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.provenance import Manifest, SourceFile, describe_environment
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


def encoders(bench: Benchmark, model, policy, region) -> dict[str, PrunedEncoder]:
    fixed = top_rows_prior(GEOMETRY.grid_size, bench.fixed_rows, bench.fixed_weight)
    priors = {"vitprune": None, "vitprune_mascara_fixa": fixed,
              "vitprune_mascara_inteligente": region}
    plans = {"denso": PruningPlan.dense()} | {
        name: PruningPlan(prune_layer=bench.prune_layer, budget=policy, spatial_prior=prior)
        for name, prior in priors.items()
    }
    return {name: PrunedEncoder(model, GEOMETRY, plans[name]) for name in VARIANTS}


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
                   runners: dict[str, PrunedEncoder], region: torch.Tensor):
    """Every variant on a batch of clips: the rows, and per variant the embeddings and
    class probabilities."""
    tensor = tensor.to(bench.device)
    shown = {i: {} for i, clip in enumerate(clips) if clip["show"]}
    rows, outputs, dense = [], {}, None
    for name, encoder in runners.items():
        _sync(bench.device)
        started = time.perf_counter()
        with autocast(bench.device, bench.precision):
            if encoder.plan.prunes:
                embedding, budget = encoder.encode_with_budget(tensor)
                tokens, regimes = budget.tokens.tolist(), budget.regimes.tolist()
            else:
                embedding = encoder.encode(tensor)
                tokens, regimes = [GEOMETRY.num_tokens] * len(clips), [None] * len(clips)
            probabilities = model.head(embedding).float().softmax(dim=-1)
        _sync(bench.device)
        seconds = (time.perf_counter() - started) / len(clips)
        shares = [None] * len(clips)
        if encoder.plan.prunes:
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
        embedding = embedding.float()
        dense = embedding if dense is None else dense
        cosines = F.cosine_similarity(embedding, dense, dim=-1).tolist()
        outputs[name] = (embedding.cpu().numpy(), probabilities.cpu().numpy())
        rows += [{
            "video": clip["video"], "start_frame": clip["start"],
            "start_s": round(clip["start"] / clip["fps"], 3),
            "variant": name, "tokens": int(tokens[i]), "regime": regimes[i],
            "inside_region": shares[i], "cosine_to_dense": cosines[i],
            "seconds": round(seconds, 4),
        } for i, clip in enumerate(clips)]
    for i, arrays in shown.items():
        bench.selections.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bench.selections / f"{clips[i]['video']}_{clips[i]['start']:06d}.npz",
                            **arrays)
    return rows, outputs


def _append(bench: Benchmark, rows: list[dict]) -> None:
    with bench.clips.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def run(bench: Benchmark, model, videos: list[Path], schedules: dict,
        runners: dict, region: torch.Tensor) -> None:
    """Every clip of every video, in batches; a video is written whole or redone."""
    for video in videos:
        targets = {name: bench.features / name / f"{video.stem}.npz" for name in runners}
        if all(path.exists() for path in targets.values()):
            print(f"{video.stem}: already done", flush=True)
            continue
        rows, outputs = run_video(bench, model, video, schedules[video.stem], runners, region)
        for name, (embeddings, probabilities) in outputs.items():
            targets[name].parent.mkdir(parents=True, exist_ok=True)
            np.savez(targets[name], feats=np.concatenate(embeddings),
                     a_prob=np.concatenate(probabilities))
        _append(bench, rows)


def run_video(bench: Benchmark, model, video: Path, clips: list[dict],
              runners: dict[str, PrunedEncoder], region: torch.Tensor):
    by_start = {c["start"]: c for c in clips}
    rows, outputs = [], {name: ([], []) for name in runners}
    batch, tensors, started = [], [], time.time()

    def flush() -> None:
        found, produced = evaluate_batch(bench, model, batch, torch.stack(tensors), runners,
                                         region)
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


def write_manifests(bench: Benchmark, videos: list[Path], policy, runners: dict) -> None:
    """The record ``vitprune evaluate`` and the AMA step read next to the features."""
    rows = [json.loads(line) for line in bench.clips.read_text(encoding="utf-8").splitlines()]
    for name in runners:
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
                      "prune_layer": 0 if name == "denso" else bench.prune_layer,
                      "policy": None if name == "denso" else policy.version,
                      "precision": bench.precision, "device": bench.device,
                      "geometry": asdict(GEOMETRY),
                      "fixed_mask": bench.fixed_mask if name == "vitprune_mascara_fixa" else None,
                      "region": str(bench.region) if name == "vitprune_mascara_inteligente"
                      else None},
            outputs=outputs,
        ).write(bench.features / name)


def refuse_other_runs(record_path: Path, settings: dict, policy: Path, region: Path) -> None:
    """Stop when the results folder holds a run of another policy, region or settings:
    resuming next to it would mix two experiments."""
    if not record_path.exists():
        return
    before = json.loads(record_path.read_text(encoding="utf-8"))
    if (before.get("settings") != settings
            or before.get("policy", {}).get("sha256") != SourceFile.of(policy).sha256
            or before.get("region", {}).get("sha256") != SourceFile.of(region).sha256):
        raise ValueError(f"{record_path.parent} holds results of another configuration; "
                         "move them out or choose another --out")


def run_benchmark(bench: Benchmark) -> None:
    """Run the variants over the evaluated videos, resuming a run of the same configuration."""
    bench.out.mkdir(parents=True, exist_ok=True)
    policy = read_policy(bench.policy, bench.prune_layer, bench.frame_step, bench.weights)
    region = read_region(bench.region, GEOMETRY)
    videos = bench.evaluated()
    infos = {video.stem: video_info(video) for video in videos}
    schedules = {video.stem: plan(bench, video, infos[video.stem]) for video in videos}
    for video in videos:
        save_frames(bench, video, schedules[video.stem])
    settings = {"stride": bench.stride, "frame_step": bench.frame_step,
                "prune_layer": bench.prune_layer, "fixed_mask": bench.fixed_mask,
                "variants": list(VARIANTS), "device": bench.device, "batch": bench.batch,
                "precision": bench.precision, "evaluate": [video.stem for video in videos],
                "show": bench.show, "learnable_positions": bench.learnable_positions}
    refuse_other_runs(bench.record, settings, bench.policy, bench.region)
    model = load_model(bench)
    if not bench.record.exists():
        report = check_backbone(model, GEOMETRY, reference=model.forward_features)
        print(report.describe(), flush=True)
        bench.record.write_text(json.dumps({
            "check": report.describe(),
            "weights": asdict(SourceFile.of(bench.weights)),
            "model_source": str(bench.model_source),
            "policy": asdict(SourceFile.of(bench.policy)),
            "region": asdict(SourceFile.of(bench.region)),
            "videos": str(bench.videos),
            "environment": describe_environment(),
            "threads": torch.get_num_threads(),
            "settings": settings,
            "schedules": {stem: {**infos[stem], "clips": len(clips)}
                          for stem, clips in schedules.items()},
        }, indent=2), encoding="utf-8")

    print(f"policy {policy.version}, levels {list(policy.levels)}; region "
          f"{float(region.mean()):.0%} of the frame", flush=True)
    runners = encoders(bench, model, policy, region)
    run(bench, model, videos, schedules, runners, region)
    write_manifests(bench, videos, policy, runners)
    print("done", flush=True)
