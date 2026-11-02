"""Run the benchmark: every clip of the evaluated videos through the four variants.

The policy and the region come from the command line, calibrated on videos that are
not evaluated here::

    vitprune calibrate --videos <calibration> ... --clips-per-video 160 --out policy.json
    vitprune region --videos <calibration> --detector yolo26s.pt ... --out region.json

and their paths go in the settings (``policy``, ``region``; see ``common.py``).

Writes, under ``results``:

* ``features/<variant>/<video>.npz`` -- what the AMA reads: ``feats`` (one row per clip)
  and ``a_prob`` (the fine-tuned head's class probabilities), with a manifest per
  variant;
* ``clips.jsonl`` -- one line per clip and variant: the tokens kept, the budget regime,
  the share of the kept tokens inside the driver region, the cosine to the dense
  embedding and the seconds taken;
* ``selection/`` -- the tokens each variant kept in the middle clip of each action, for
  the figures.

A video is written whole or redone, so an interrupted run resumes video by video.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import (
    CACHE,
    DEVICE,
    FRAME_STEP,
    GEOMETRY,
    MODEL_SOURCE,
    RESULTS,
    SETTINGS,
    SPAN,
    STRIDE,
    VARIANTS,
    WEIGHTS,
    clip_starts,
    segment_of,
    segments,
    videos,
)

from vitprune.bench.latency import autocast
from vitprune.core import PrunedEncoder, PruningPlan, check_backbone
from vitprune.core.priors import top_rows_prior
from vitprune.pipeline.calibration import read_policy
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.provenance import Manifest, SourceFile, describe_environment
from vitprune.pipeline.region import read_region
from vitprune.pipeline.video import iter_clips

PRUNE_LAYER = 2
FIXED_MASK = {"rows": 4, "weight": 0.15}
BATCH, PRECISION = SETTINGS["batch"], SETTINGS["precision"]
POLICY, REGION = Path(SETTINGS["policy"]), Path(SETTINGS["region"])
CLIPS = RESULTS / "clips.jsonl"
SELECTIONS = RESULTS / "selection"
FEATURES = RESULTS / "features"


def _sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def load_model():
    started = time.time()
    model = load_videomae(WEIGHTS, MODEL_SOURCE, GEOMETRY, device=DEVICE)
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)
    return disable_activation_checkpointing(model)


def plan(video) -> list[dict]:
    """Every clip of a video, with the action holding it whole, if any. The middle clip
    of each action is the one shown in the figures."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    annotated = segments(video.stem)
    clips = []
    for start in clip_starts(total):
        segment = segment_of(start, fps, annotated)
        clips.append({"video": video.stem, "start": start, "fps": fps, "show": False,
                      "segment": None if segment is None else annotated.index(segment)})
    for index in range(len(annotated)):
        inside = [clip for clip in clips if clip["segment"] == index]
        if inside:
            inside[len(inside) // 2]["show"] = True
    return clips


def save_frames(video, clips: list[dict]) -> None:
    """The middle frame of each shown clip, for the figures."""
    folder = CACHE / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    for clip in clips:
        frame = folder / f"{clip['start']:06d}.jpg"
        if clip["show"] and not frame.exists():
            capture.set(cv2.CAP_PROP_POS_FRAMES, clip["start"] + SPAN // 2)
            ok, image = capture.read()
            if ok:
                cv2.imwrite(str(frame), image)
    capture.release()


def encoders(model, policy, region) -> dict[str, PrunedEncoder]:
    fixed = top_rows_prior(GEOMETRY.grid_size, FIXED_MASK["rows"], FIXED_MASK["weight"])
    priors = {"vitprune": None, "vitprune_mascara_fixa": fixed,
              "vitprune_mascara_inteligente": region}
    plans = {"denso": PruningPlan.dense()} | {
        name: PruningPlan(prune_layer=PRUNE_LAYER, budget=policy, spatial_prior=prior)
        for name, prior in priors.items()
    }
    return {name: PrunedEncoder(model, GEOMETRY, plans[name]) for name in VARIANTS}


@torch.inference_mode()
def evaluate_batch(model, clips: list[dict], tensor: torch.Tensor,
                   runners: dict[str, PrunedEncoder], region: torch.Tensor):
    """Every variant on a batch of clips: the rows, and per variant the embeddings and
    class probabilities."""
    tensor = tensor.to(DEVICE)
    inside = region.to(DEVICE) > 0
    shown = {i: {} for i, clip in enumerate(clips) if clip["show"]}
    rows, outputs, dense = [], {}, None
    for name, encoder in runners.items():
        _sync()
        started = time.perf_counter()
        with autocast(DEVICE, PRECISION):
            if encoder.plan.prunes:
                embedding, budget = encoder.encode_with_budget(tensor)
                tokens, regimes = budget.tokens.tolist(), budget.regimes.tolist()
            else:
                embedding = encoder.encode(tensor)
                tokens, regimes = [GEOMETRY.num_tokens] * len(clips), [None] * len(clips)
            probabilities = model.head(embedding).float().softmax(dim=-1)
        _sync()
        seconds = (time.perf_counter() - started) / len(clips)
        shares = [None] * len(clips)
        if encoder.plan.prunes:
            # Outside the timing: the same decision again, to see where the tokens went.
            with autocast(DEVICE, PRECISION):
                selection = encoder.select(tensor)
            valid = selection.kept >= 0
            hits = inside[selection.kept.clamp(min=0)] & valid
            shares = (hits.sum(dim=(1, 2)) / valid.sum(dim=(1, 2))).tolist()
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
            "start_s": round(clip["start"] / clip["fps"], 3), "segment": clip["segment"],
            "variant": name, "tokens": int(tokens[i]), "regime": regimes[i],
            "inside_region": shares[i], "cosine_to_dense": cosines[i],
            "seconds": round(seconds, 4),
        } for i, clip in enumerate(clips)]
    for i, arrays in shown.items():
        SELECTIONS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SELECTIONS / f"{clips[i]['video']}_{clips[i]['start']:06d}.npz",
                            **arrays)
    return rows, outputs


def _append(rows: list[dict]) -> None:
    with CLIPS.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def run(model, schedules: dict, runners: dict, region: torch.Tensor) -> None:
    """Every clip of every video, in batches; a video is written whole or redone."""
    for video in videos():
        targets = {name: FEATURES / name / f"{video.stem}.npz" for name in runners}
        if all(path.exists() for path in targets.values()):
            print(f"{video.stem}: already done", flush=True)
            continue
        rows, outputs = run_video(model, video, schedules[video.stem], runners, region)
        for name, (embeddings, probabilities) in outputs.items():
            targets[name].parent.mkdir(parents=True, exist_ok=True)
            np.savez(targets[name], feats=np.concatenate(embeddings),
                     a_prob=np.concatenate(probabilities))
        _append(rows)


def run_video(model, video, clips: list[dict], runners: dict[str, PrunedEncoder],
              region: torch.Tensor):
    by_start = {c["start"]: c for c in clips}
    rows, outputs = [], {name: ([], []) for name in runners}
    batch, tensors, started = [], [], time.time()

    def flush() -> None:
        found, produced = evaluate_batch(model, batch, torch.stack(tensors), runners, region)
        rows.extend(found)
        for name, (embedding, probability) in produced.items():
            outputs[name][0].append(embedding)
            outputs[name][1].append(probability)
        print(f"  {video.stem}: {len(rows) // len(runners)}/{len(by_start)} clips "
              f"({time.time() - started:.0f}s)", flush=True)
        batch.clear()
        tensors.clear()

    for start, clip in iter_clips(video, GEOMETRY, STRIDE, FRAME_STEP, by_start):
        batch.append(by_start[start])
        tensors.append(clip)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    return rows, outputs


def write_manifests(policy, runners: dict) -> None:
    """The record ``vitprune evaluate`` and the AMA step read next to the features."""
    rows = [json.loads(line) for line in CLIPS.read_text(encoding="utf-8").splitlines()]
    for name in runners:
        outputs = {}
        for video in videos():
            mine = sorted((r for r in rows if r["video"] == video.stem and r["variant"] == name),
                          key=lambda r: r["start_frame"])
            outputs[video.stem] = {
                "feature_file": asdict(SourceFile.of(FEATURES / name / f"{video.stem}.npz")),
                "clips": len(mine),
                "tokens": [r["tokens"] for r in mine],
            }
        Manifest(
            settings={"variant": name, "stride": STRIDE, "frame_step": FRAME_STEP,
                      "prune_layer": 0 if name == "denso" else PRUNE_LAYER,
                      "policy": None if name == "denso" else policy.version,
                      "precision": PRECISION, "device": DEVICE, "geometry": asdict(GEOMETRY),
                      "fixed_mask": FIXED_MASK if name == "vitprune_mascara_fixa" else None,
                      "region": str(REGION) if name == "vitprune_mascara_inteligente" else None},
            outputs=outputs,
        ).write(FEATURES / name)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    policy = read_policy(POLICY, PRUNE_LAYER, FRAME_STEP, WEIGHTS)
    region = read_region(REGION, GEOMETRY)
    schedules = {video.stem: plan(video) for video in videos()}
    for video in videos():
        save_frames(video, schedules[video.stem])
    settings = {"stride": STRIDE, "frame_step": FRAME_STEP, "prune_layer": PRUNE_LAYER,
                "fixed_mask": FIXED_MASK, "variants": list(VARIANTS), "device": DEVICE,
                "batch": BATCH, "precision": PRECISION}
    record_path = RESULTS / "run.json"
    if record_path.exists():
        # Resuming next to results of another policy or region would mix two experiments.
        before = json.loads(record_path.read_text(encoding="utf-8"))
        if (before.get("settings") != settings
                or before.get("policy", {}).get("sha256") != SourceFile.of(POLICY).sha256
                or before.get("region", {}).get("sha256") != SourceFile.of(REGION).sha256):
            raise SystemExit(f"{RESULTS} holds results of another configuration; move them "
                             "out before running this one")
    model = load_model()
    if not record_path.exists():
        report = check_backbone(model, GEOMETRY, reference=model.forward_features)
        print(report.describe(), flush=True)
        record_path.write_text(json.dumps({
            "check": report.describe(),
            "weights": asdict(SourceFile.of(WEIGHTS)),
            "model_source": str(MODEL_SOURCE),
            "policy": asdict(SourceFile.of(POLICY)),
            "region": asdict(SourceFile.of(REGION)),
            "environment": describe_environment(),
            "threads": torch.get_num_threads(),
            "settings": settings,
            "schedules": {stem: len(clips) for stem, clips in schedules.items()},
        }, indent=2), encoding="utf-8")

    print(f"policy {policy.version}, levels {list(policy.levels)}; region "
          f"{float(region.mean()):.0%} of the frame", flush=True)
    runners = encoders(model, policy, region)
    run(model, schedules, runners, region)
    write_manifests(policy, runners)
    print("done", flush=True)


if __name__ == "__main__":
    main()
