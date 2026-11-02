"""Run the benchmark: dense VideoMAE against ViT-Prune without a mask, with the old
top-rows mask and with the driver region.

The policy and the region come from the command line, calibrated on videos that are
not evaluated here::

    vitprune calibrate --videos <calibration> ... --clips-per-video 160 --out policy.json
    vitprune region --videos <calibration> ... --clips-per-video 160 --out region.json

and their paths go in the settings (``policy``, ``region``; see ``common.py``).

Two modes, chosen by ``clips`` in the settings:

* ``sample`` -- four evenly spaced clips per annotated action, each wholly inside it.
  What a CPU finishes in a few hours.
* ``all`` -- every clip of every video, in batches. Also writes, per variant, the
  embeddings the AMA reads: ``results/features/<variant>/<video>.npz`` with ``feats``
  (one row per clip) and ``a_prob`` (the head's class probabilities).

Per clip and variant, one JSON line goes to ``results/clips.jsonl``: the class the
fine-tuned head predicts, its confidence, the cosine to the dense embedding, the tokens
kept and the seconds taken. Clips outside every action carry no label. One clip per
action also gets its token selection saved for the figures. An interrupted run resumes,
clip by clip in ``sample`` mode and video by video in ``all`` mode.
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
    WEIGHTS,
    clip_starts,
    segment_of,
    segments,
    videos,
)

from vitprune.core import PrunedEncoder, PruningPlan, check_backbone
from vitprune.core.priors import top_rows_prior
from vitprune.pipeline.calibration import read_policy
from vitprune.pipeline.checkpoint import disable_activation_checkpointing, load_videomae
from vitprune.pipeline.provenance import Manifest, SourceFile, describe_environment
from vitprune.pipeline.region import read_region
from vitprune.pipeline.video import iter_clips

PER_SEGMENT = 4
PRUNE_LAYER = 2
OLD_MASK = {"rows": 4, "weight": 0.15}
MODE, BATCH, PRECISION = SETTINGS["clips"], SETTINGS["batch"], SETTINGS["precision"]
POLICY, REGION = Path(SETTINGS["policy"]), Path(SETTINGS["region"])
CLIPS = RESULTS / "clips.jsonl"
SELECTIONS = RESULTS / "selection"
FEATURES = RESULTS / "features"


def _autocast():
    dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[PRECISION]
    return torch.autocast("cuda", dtype=dtype or torch.float16,
                          enabled=DEVICE.startswith("cuda") and dtype is not None)


def _sync() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def load_model():
    started = time.time()
    model = load_videomae(WEIGHTS, MODEL_SOURCE, GEOMETRY, device=DEVICE)
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)
    return disable_activation_checkpointing(model)


def _clip(stem, start, fps, index, segment, order, show) -> dict:
    return {
        "video": stem, "round": order, "segment": index, "start": start, "fps": fps,
        "label_id": segment["label_id"] if segment else None,
        "label": segment["label"] if segment else None, "show": show,
    }


def plan(video) -> list[dict]:
    """The clips to evaluate in one video."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    starts = clip_starts(total)
    annotated = segments(video.stem)
    chosen = []
    for index, segment in enumerate(annotated):
        inside = [start for start in starts if segment_of(start, fps, [segment])]
        if not inside:
            continue
        picks = sorted({round(i) for i in np.linspace(0, len(inside) - 1, PER_SEGMENT)})
        for order, pick in enumerate(picks):
            chosen.append(_clip(video.stem, inside[pick], fps, index, segment, order,
                                order == len(picks) // 2))
    if MODE == "sample":
        return chosen
    shown = {c["start"] for c in chosen if c["show"]}
    every = []
    for start in starts:
        segment = segment_of(start, fps, annotated)
        index = annotated.index(segment) if segment else None
        every.append(_clip(video.stem, start, fps, index, segment, 0, start in shown))
    return every


def cached(stem: str, start: int) -> Path:
    return CACHE / stem / f"{start:06d}.pt"


def prepare(video, clips: list[dict]) -> None:
    """Save the frames shown in the figures and, in ``sample`` mode, cache the clips."""
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
    if MODE != "sample":
        return
    missing = [c["start"] for c in clips if not cached(video.stem, c["start"]).exists()]
    for start, clip in iter_clips(video, GEOMETRY, STRIDE, FRAME_STEP, missing):
        torch.save(clip.clone(), cached(video.stem, start))


def encoders(model, policy, region) -> dict[str, PrunedEncoder]:
    old = top_rows_prior(GEOMETRY.grid_size, OLD_MASK["rows"], OLD_MASK["weight"])
    plans = {
        "denso": PruningPlan.dense(),
        "vitprune": PruningPlan(prune_layer=PRUNE_LAYER, budget=policy),
        "vitprune_mascara_antiga": PruningPlan(prune_layer=PRUNE_LAYER, budget=policy,
                                               spatial_prior=old),
        "vitprune_regiao": PruningPlan(prune_layer=PRUNE_LAYER, budget=policy,
                                       spatial_prior=region),
    }
    return {name: PrunedEncoder(model, GEOMETRY, plan) for name, plan in plans.items()}


@torch.inference_mode()
def evaluate_batch(model, clips: list[dict], tensor: torch.Tensor,
                   runners: dict[str, PrunedEncoder]) -> tuple[list[dict], dict]:
    """Every variant on a batch of clips: the rows, and per variant the embeddings and
    class probabilities."""
    tensor = tensor.to(DEVICE)
    rows, outputs, dense = [], {}, None
    for name, encoder in runners.items():
        _sync()
        started = time.perf_counter()
        with _autocast():
            if encoder.plan.prunes:
                embedding, budget = encoder.encode_with_budget(tensor)
                tokens, regimes = budget.tokens.tolist(), budget.regimes.tolist()
            else:
                embedding = encoder.encode(tensor)
                tokens, regimes = [GEOMETRY.num_tokens] * len(clips), [None] * len(clips)
            probabilities = model.head(embedding).float().softmax(dim=-1)
        _sync()
        seconds = (time.perf_counter() - started) / len(clips)
        embedding = embedding.float()
        dense = embedding if dense is None else dense
        cosines = F.cosine_similarity(embedding, dense, dim=-1).tolist()
        outputs[name] = (embedding.cpu().numpy(), probabilities.cpu().numpy())
        for i, clip in enumerate(clips):
            top = probabilities[i].topk(3).indices.tolist()
            label = clip["label_id"]
            rows.append({
                "video": clip["video"], "start_frame": clip["start"],
                "start_s": round(clip["start"] / clip["fps"], 3),
                "end_s": round((clip["start"] + SPAN) / clip["fps"], 3),
                "segment": clip["segment"], "label_id": label, "label": clip["label"],
                "variant": name, "predicted_id": top[0],
                "confidence": float(probabilities[i, top[0]]),
                "true_probability": None if label is None else float(probabilities[i, label]),
                "correct": None if label is None else top[0] == label,
                "top3": None if label is None else label in top,
                "cosine_to_dense": cosines[i], "tokens": int(tokens[i]), "regime": regimes[i],
                "seconds": round(seconds, 4),
            })
    for i, clip in enumerate(clips):
        if clip["show"]:
            save_selection(clip, tensor[i : i + 1], runners)
    return rows, outputs


def save_selection(clip: dict, tensor: torch.Tensor, runners: dict[str, PrunedEncoder]) -> None:
    arrays = {}
    for name, encoder in runners.items():
        if encoder.plan.prunes:
            with _autocast():
                selection = encoder.select(tensor)
            arrays[f"{name}_kept"] = np.stack([
                selection.keep_mask(0, s).flatten() for s in range(selection.temporal_slices)
            ])
            arrays[f"{name}_scores"] = selection.scores[0].float().cpu().numpy()
            if encoder.plan.spatial_prior is not None:
                arrays[f"{name}_prior"] = encoder.plan.spatial_prior.cpu().numpy()
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SELECTIONS / f"{clip['video']}_{clip['start']:06d}.npz", **arrays)


def _append(rows: list[dict]) -> None:
    with CLIPS.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def run_sampled(model, schedules: dict, runners: dict) -> None:
    done = set()
    if CLIPS.exists():
        for line in CLIPS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["video"], row["start_frame"]))
    queue = sorted(
        (c for clips in schedules.values() for c in clips
         if (c["video"], c["start"]) not in done),
        key=lambda c: (c["round"], c["segment"], c["video"]),
    )
    started, total = time.time(), len(queue)
    for position, clip in enumerate(queue, start=1):
        tensor = torch.load(cached(clip["video"], clip["start"])).unsqueeze(0)
        rows, _ = evaluate_batch(model, [clip], tensor, runners)
        _append(rows)
        pace = (time.time() - started) / position
        verdict = " ".join(f"{r['variant']}={'ok' if r['correct'] else 'x'}({r['seconds']:.0f}s)"
                           for r in rows)
        print(f"[{position}/{total}] {clip['video'][-6:]} {clip['label']}: {verdict} | "
              f"eta {pace * (total - position) / 60:.0f} min", flush=True)


def run_all(model, schedules: dict, runners: dict) -> None:
    """Every clip of every video, in batches; a video is written whole or redone."""
    for video in videos():
        targets = {name: FEATURES / name / f"{video.stem}.npz" for name in runners}
        if all(path.exists() for path in targets.values()):
            print(f"{video.stem}: already done", flush=True)
            continue
        rows, outputs = run_video(model, video, schedules[video.stem], runners)
        for name, (embeddings, probabilities) in outputs.items():
            targets[name].parent.mkdir(parents=True, exist_ok=True)
            np.savez(targets[name], feats=np.concatenate(embeddings),
                     a_prob=np.concatenate(probabilities))
        _append(rows)


def run_video(model, video, clips: list[dict], runners: dict[str, PrunedEncoder]):
    by_start = {c["start"]: c for c in clips}
    rows, outputs = [], {name: ([], []) for name in runners}
    batch, tensors, started = [], [], time.time()

    def flush() -> None:
        found, produced = evaluate_batch(model, batch, torch.stack(tensors), runners)
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
                      "old_mask": OLD_MASK if name == "vitprune_mascara_antiga" else None,
                      "region": str(REGION) if name == "vitprune_regiao" else None},
            outputs=outputs,
        ).write(FEATURES / name)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    policy = read_policy(POLICY, PRUNE_LAYER, FRAME_STEP)
    region = read_region(REGION, GEOMETRY)
    schedules = {video.stem: plan(video) for video in videos()}
    for video in videos():
        prepare(video, schedules[video.stem])
    model = load_model()
    record_path = RESULTS / "run.json"
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
            "settings": {"stride": STRIDE, "frame_step": FRAME_STEP,
                         "prune_layer": PRUNE_LAYER, "per_segment": PER_SEGMENT,
                         "old_mask": OLD_MASK, "clips": MODE, "device": DEVICE,
                         "batch": BATCH, "precision": PRECISION},
            "schedules": {stem: len(clips) for stem, clips in schedules.items()},
        }, indent=2), encoding="utf-8")

    print(f"policy {policy.version}, levels {list(policy.levels)}; region "
          f"{float(region.mean()):.0%} of the frame", flush=True)
    runners = encoders(model, policy, region)
    if MODE == "sample":
        run_sampled(model, schedules, runners)
    else:
        run_all(model, schedules, runners)
        write_manifests(policy, runners)
    print("done", flush=True)


if __name__ == "__main__":
    main()
