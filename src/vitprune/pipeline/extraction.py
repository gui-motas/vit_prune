"""Walking a folder of videos and writing one embedding per clip.

This is the expensive stage: it is where the GPU works. What comes out are embedding
files, one per video, ready to be consumed by whatever model decides which action
occurs and when.

The output format was not chosen freely. It is what such models already read: a
compressed file with a field named ``feats`` holding one row per clip. Keeping that
format is what allows swapping dense embeddings for pruned ones without touching
anything downstream, which is the premise of the whole experiment.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vitprune.bench.cost import as_dict, budget_cost
from vitprune.bench.latency import PRECISIONS, autocast
from vitprune.core.encoder import PrunedEncoder
from vitprune.pipeline.provenance import MANIFEST_NAME, Manifest, SourceFile
from vitprune.pipeline.video import find_videos, iter_clips

#: Field name inside the compressed file. Downstream readers look for exactly this.
FEATURE_KEY = "feats"
#: The record of an extraction still under way, replaced by the manifest at the end.
PARTIAL_NAME = "manifest.partial.json"


def extract_folder(
    encoder: PrunedEncoder,
    videos: str | Path,
    destination: str | Path,
    stride: int = 16,
    frame_step: int = 2,
    batch: int = 8,
    device: str = "cuda",
    precision: str = "fp16",
    weights_path: str | Path | None = None,
    progress=print,
    model_source: str | Path | None = None,
) -> Path:
    """Process every video in a folder and write the run record.

    Args:
        encoder: the assembled encoder, with its pruning plan already chosen.
        videos: a folder or a single video file.
        destination: where to write embeddings and the record.
        stride: frames between the start of consecutive clips.
        frame_step: take every ``frame_step`` frames within a clip.
        batch: clips per forward pass.
        device: where to run.
        precision: ``fp16``, ``bf16`` or ``off``. Must match whatever was used for
            timing, or the two numbers describe different code paths.
        weights_path: the checkpoint, recorded in the manifest for traceability.
        progress: where to send progress output.
        model_source: optional external source folder, fingerprinted for provenance.

    Returns:
        The path of the written record.

    The record of the videos done so far is kept in ``manifest.partial.json``. Run the
    same extraction again after an interruption and it carries on from there; the
    finished ``manifest.json`` is written only once every video is done.
    """
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")
    duplicates = [stem for stem, count in Counter(p.stem.casefold() for p in found).items()
                  if count > 1]
    if duplicates:
        raise ValueError(f"video stems must be unique; duplicate output names: {duplicates}")
    _validate_encoding(batch, precision)

    destination = Path(destination)
    settings = extraction_settings(
        encoder, stride, frame_step, batch, device, precision, model_source
    )
    weights = SourceFile.of(weights_path) if weights_path else None
    per_video = _resumable(destination, found, settings, weights)
    destination.mkdir(parents=True, exist_ok=True)

    for position, video in enumerate(found, start=1):
        target = destination / f"{video.stem}.npz"
        if video.stem in per_video and target.exists():
            progress(f"[{position}/{len(found)}] {video.name}: done before")
            continue
        moment = time.time()
        clips = (clip for _, clip in iter_clips(video, encoder.geometry, stride, frame_step))
        embeddings, budget = encode_clips(encoder, clips, batch, device, precision)
        np.savez(target, **{FEATURE_KEY: embeddings})

        elapsed = time.time() - moment
        per_video[video.stem] = {
            "path": str(video),
            "source": asdict(SourceFile.of(video)),
            "feature_file": asdict(SourceFile.of(target)),
            "clips": int(embeddings.shape[0]),
            "embed_dim": int(embeddings.shape[1]),
            "seconds": round(elapsed, 2),
        }
        if budget is not None:
            per_video[video.stem]["budget"] = budget
        _build_manifest(encoder, settings, weights, per_video).write(destination, PARTIAL_NAME)
        progress(f"[{position}/{len(found)}] {video.name}: {embeddings.shape} in {elapsed:.1f}s")

    manifest = _build_manifest(encoder, settings, weights, per_video)
    written = manifest.write(destination)
    (destination / PARTIAL_NAME).unlink(missing_ok=True)
    statistics = manifest.statistics
    progress(
        f"[done] {statistics['clips']} clips from {len(found)} videos in "
        f"{statistics['seconds']:.1f}s -> {destination}"
    )
    return written


def _resumable(destination: Path, found, settings: dict, weights) -> dict[str, dict]:
    """The videos an interrupted run of this same extraction already finished.

    Refuses a folder holding a finished extraction, an unfinished one with other
    settings or checkpoint, or features no record accounts for.
    """
    if (destination / MANIFEST_NAME).exists():
        raise FileExistsError(f"destination already holds a finished extraction: {destination}")
    partial = destination / PARTIAL_NAME
    if partial.exists():
        before = json.loads(partial.read_text(encoding="utf-8"))
        same_weights = (before.get("weights") or {}).get("sha256") == (
            weights.sha256 if weights else None
        )
        if before["settings"] != json.loads(json.dumps(settings)) or not same_weights:
            raise FileExistsError(
                f"{destination} holds an unfinished extraction with other settings or "
                "another checkpoint; use another folder"
            )
        return before["outputs"]
    if any((destination / f"{video.stem}.npz").exists() for video in found):
        raise FileExistsError(f"destination already contains extraction outputs: {destination}")
    return {}


@torch.no_grad()
def encode_clips(
    encoder: PrunedEncoder,
    clips: torch.Tensor | Iterable[torch.Tensor],
    batch: int = 8,
    device: str = "cuda",
    precision: str = "fp16",
) -> tuple[np.ndarray, dict[str, list] | None]:
    """Run the clips through the encoder in batches.

    ``clips`` is a tensor of clips or any iterable of single clips, such as
    :func:`vitprune.pipeline.video.iter_clips`, so a long video never has to be in
    memory at once.

    Returns:
        The embeddings, ``[clips, embed_dim]`` in single precision, and what the
        adaptive budget decided: lists of tokens, regime, redundancy and motion, one
        entry per clip in order; ``None`` for a plan that prunes nothing.
    """
    _validate_encoding(batch, precision)
    adaptive = encoder.plan.prunes
    pieces = []
    record: dict[str, list] = {"tokens": [], "regimes": [], "redundancy": [], "motion": []}

    for chunk in _batches(clips, batch):
        chunk = chunk.to(device, non_blocking=True)
        with autocast(device, precision):
            if adaptive:
                embeddings, budget = encoder.encode_with_budget(chunk)
                record["tokens"] += budget.tokens.tolist()
                record["regimes"] += budget.regimes.tolist()
                record["redundancy"] += budget.statistics.redundancy.tolist()
                record["motion"] += budget.statistics.motion.tolist()
            else:
                embeddings = encoder.encode(chunk)
            embeddings = embeddings.float().cpu()
            if not torch.isfinite(embeddings).all():
                raise ValueError("encoder produced non-finite embeddings")
            pieces.append(embeddings)

    if not pieces:
        raise ValueError("at least one clip is needed to encode")
    return torch.cat(pieces).numpy().astype(np.float32), record if adaptive else None


def _batches(clips, size: int):
    if isinstance(clips, torch.Tensor):
        for start in range(0, clips.shape[0], size):
            yield clips[start : start + size]
        return
    pending = []
    for clip in clips:
        pending.append(clip)
        if len(pending) == size:
            yield torch.stack(pending)
            pending = []
    if pending:
        yield torch.stack(pending)


def _build_manifest(encoder, settings, weights, per_video) -> Manifest:
    geometry, plan = encoder.geometry, encoder.plan
    tokens = [t for entry in per_video.values() for t in entry.get("budget", {}).get("tokens", [])]
    cost = (
        budget_cost(geometry, tokens, plan.prune_layer)
        if tokens
        else budget_cost(geometry, [geometry.num_tokens], 0)
    )
    clips = sum(entry["clips"] for entry in per_video.values())
    seconds = sum(entry["seconds"] for entry in per_video.values())
    return Manifest(
        settings=settings,
        weights=weights,
        cost=as_dict(cost),
        outputs=per_video,
        statistics={
            "videos": len(per_video),
            "clips": clips,
            "seconds": round(seconds, 1),
            "clips_per_second": round(clips / max(seconds, 1e-9), 2),
        },
    )


def extraction_settings(
    encoder: PrunedEncoder,
    stride: int,
    frame_step: int,
    batch: int,
    device: str,
    precision: str,
    model_source: str | Path | None = None,
) -> dict:
    """Describe the same extraction protocol for folder and single-video runs."""
    geometry, plan = encoder.geometry, encoder.plan
    return {
        "prune_layer": plan.prune_layer,
        "motion_weight": plan.motion_weight,
        "size_biased_attention": plan.size_biased_attention,
        "spatial_prior": (
            None if plan.spatial_prior is None else plan.spatial_prior.detach().cpu().tolist()
        ),
        "budget": None if plan.budget is None else plan.budget.to_dict(),
        "stride": stride,
        "frame_step": frame_step,
        "batch": batch,
        "precision": precision,
        "effective_precision": precision if str(device).startswith("cuda") else "off",
        "device": str(device),
        "geometry": asdict(geometry),
        "backbone_class": (
            f"{type(encoder.backbone).__module__}.{type(encoder.backbone).__qualname__}"
        ),
        "model_definition": (
            asdict(SourceFile.of(Path(model_source) / "models" / "modeling_feature.py"))
            if model_source is not None else None
        ),
    }


def _validate_encoding(batch: int, precision: str) -> None:
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision '{precision}'. Use: {', '.join(PRECISIONS)}")
