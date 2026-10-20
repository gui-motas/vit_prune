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

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vitprune.bench.cost import as_dict, clip_cost
from vitprune.core.encoder import PrunedEncoder
from vitprune.pipeline.provenance import Manifest, SourceFile
from vitprune.pipeline.video import find_videos, read_video, to_clips

#: Field name inside the compressed file. Downstream readers look for exactly this.
FEATURE_KEY = "feats"


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
    existing = [p for p in (destination / f"{video.stem}.npz" for video in found) if p.exists()]
    if existing or (destination / "manifest.json").exists():
        raise FileExistsError(f"destination already contains extraction outputs: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    started = time.time()
    per_video: dict[str, dict] = {}
    total_clips = 0

    for position, video in enumerate(found, start=1):
        moment = time.time()
        embeddings = extract_video(encoder, video, stride, frame_step, batch, device, precision)
        np.savez(destination / f"{video.stem}.npz", **{FEATURE_KEY: embeddings})

        elapsed = time.time() - moment
        per_video[video.stem] = {
            "path": str(video),
            "source": asdict(SourceFile.of(video)),
            "feature_file": asdict(SourceFile.of(destination / f"{video.stem}.npz")),
            "clips": int(embeddings.shape[0]),
            "embed_dim": int(embeddings.shape[1]),
            "seconds": round(elapsed, 2),
        }
        total_clips += int(embeddings.shape[0])
        progress(f"[{position}/{len(found)}] {video.name}: {embeddings.shape} in {elapsed:.1f}s")

    total_seconds = time.time() - started
    manifest = _build_manifest(
        encoder,
        found,
        per_video,
        total_clips,
        total_seconds,
        stride,
        frame_step,
        batch,
        device,
        precision,
        weights_path,
        model_source,
    )
    written = manifest.write(destination)
    progress(
        f"[done] {total_clips} clips from {len(found)} videos in {total_seconds:.1f}s "
        f"-> {destination}"
    )
    return written


@torch.no_grad()
def extract_video(
    encoder: PrunedEncoder,
    video: str | Path,
    stride: int = 16,
    frame_step: int = 2,
    batch: int = 8,
    device: str = "cuda",
    precision: str = "fp16",
) -> np.ndarray:
    """Return one embedding per clip for a single video.

    Returns:
        ``[clips, embed_dim]`` in single precision. Storing at single precision is
        deliberate: halving the file size saves little and risks introducing a
        difference that would later be mistaken for an effect of the method.
    """
    clips = to_clips(read_video(video), encoder.geometry, stride, frame_step)
    return encode_clips(encoder, clips, batch, device, precision)


@torch.no_grad()
def encode_clips(
    encoder: PrunedEncoder,
    clips: torch.Tensor,
    batch: int = 8,
    device: str = "cuda",
    precision: str = "fp16",
) -> np.ndarray:
    """Run the clips through the encoder in batches."""
    _validate_encoding(batch, precision)
    if clips.shape[0] < 1:
        raise ValueError("at least one clip is needed to encode")
    dtype = _precision(precision)
    on_gpu = str(device).startswith("cuda")
    pieces = []

    for start in range(0, clips.shape[0], batch):
        chunk = clips[start : start + batch].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=on_gpu and dtype is not None):
            embeddings = encoder.encode(chunk).float().cpu()
            if not torch.isfinite(embeddings).all():
                raise ValueError("encoder produced non-finite embeddings")
            pieces.append(embeddings)

    return torch.cat(pieces).numpy().astype(np.float32)


def _build_manifest(
    encoder,
    found,
    per_video,
    total_clips,
    total_seconds,
    stride,
    frame_step,
    batch,
    device,
    precision,
    weights_path,
    model_source=None,
) -> Manifest:
    geometry = encoder.geometry
    plan = encoder.plan
    return Manifest(
        settings=extraction_settings(
            encoder, stride, frame_step, batch, device, precision, model_source
        ),
        weights=SourceFile.of(weights_path) if weights_path else None,
        cost=as_dict(clip_cost(geometry, plan.retention, plan.prune_layer)),
        outputs=per_video,
        statistics={
            "videos": len(found),
            "clips": total_clips,
            "seconds": round(total_seconds, 1),
            "clips_per_second": round(total_clips / max(total_seconds, 1e-9), 2),
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
        "retention": plan.retention,
        "prune_layer": plan.prune_layer,
        "motion_weight": plan.motion_weight,
        "size_biased_attention": plan.size_biased_attention,
        "spatial_prior": (
            None if plan.spatial_prior is None else plan.spatial_prior.detach().cpu().tolist()
        ),
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


def _precision(name: str):
    table = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}
    if name not in table:
        raise ValueError(f"unknown precision '{name}'. Use: {', '.join(table)}")
    return table[name]


def _validate_encoding(batch: int, precision: str) -> None:
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    _precision(precision)
