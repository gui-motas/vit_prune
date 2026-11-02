"""Calibrating the adaptive budget on a set of videos.

The policy in :mod:`vitprune.core.policy` needs reference values: what counts as much
or little redundancy and motion. They come from calibration videos, which must not be
part of the evaluation, or the policy is tuned on the data it is judged on.

Only the blocks up to the cut run, so calibrating costs a fraction of an extraction.
The result is a policy file holding the policy and the conditions it was measured
under. The statistics are taken at the pruning layer and depend on how clips are
sampled, so a policy is refused when used with a different cut or frame step.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vitprune.bench.latency import autocast
from vitprune.core.encoder import PrunedEncoder
from vitprune.core.policy import BudgetPolicy, calibrate_policy
from vitprune.core.statistics import ClipStatistics
from vitprune.pipeline.extraction import _validate_encoding
from vitprune.pipeline.provenance import SourceFile
from vitprune.pipeline.video import find_videos, frame_count, iter_clips, spread_starts


def calibrate_videos(
    encoder: PrunedEncoder,
    videos: str | Path,
    levels: list[int],
    motion_balance: float = 0.5,
    stride: int = 16,
    frame_step: int = 2,
    batch: int = 8,
    device: str = "cuda",
    precision: str = "fp16",
    per_video: int | None = None,
    weights_path: str | Path | None = None,
    progress=print,
) -> dict[str, Any]:
    """Measure the statistics of calibration videos and fit a policy to them.

    Args:
        encoder: an encoder whose plan sets the pruning layer the policy will serve.
        videos: a folder or a single video file.
        levels: tokens kept per clip in each regime.
        motion_balance: weight of motion against redundancy in the budget.
        stride, frame_step: how clips are sampled; use the extraction values.
        batch, device, precision: how the blocks up to the cut run.
        per_video: clips taken from each video, spread evenly over it; every clip
            when ``None``. The first clips of a video are not a sample of it.
        weights_path: the checkpoint, recorded for traceability.
        progress: where to send progress output.

    Returns:
        The content of a policy file, ready for :func:`write_policy`.

    Videos are read a frame at a time, so their length is not limited by memory.
    """
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")
    if per_video is not None and per_video < 1:
        raise ValueError("clips per video must be positive")
    _validate_encoding(batch, precision)

    def measure(clips: list[torch.Tensor]) -> None:
        chunk = torch.stack(clips).to(device)
        with autocast(device, precision):
            statistics = encoder.statistics(chunk)
        redundancy.append(statistics.redundancy.cpu())
        motion.append(statistics.motion.cpu())

    redundancy, motion, sources, total = [], [], [], 0
    for video in found:
        starts = None
        if per_video is not None:
            starts = spread_starts(frame_count(video), per_video, encoder.geometry, stride,
                                   frame_step)
        pending, clips = [], 0
        for _, clip in iter_clips(video, encoder.geometry, stride, frame_step, starts):
            pending.append(clip)
            clips += 1
            if len(pending) == batch:
                measure(pending)
                pending = []
        if pending:
            measure(pending)
        total += clips
        sources.append({"source": asdict(SourceFile.of(video)), "clips": clips})
        progress(f"[calibrate] {video.name}: {clips} clips")

    policy = calibrate_policy(
        ClipStatistics(torch.cat(redundancy), torch.cat(motion)), levels, motion_balance
    )
    return {
        "policy": policy.to_dict(),
        "calibration": {
            "clips": total,
            "clips_per_video": per_video,
            "videos": sources,
            "prune_layer": encoder.plan.prune_layer,
            "stride": stride,
            "frame_step": frame_step,
            "precision": precision,
            "geometry": asdict(encoder.geometry),
            "weights": asdict(SourceFile.of(weights_path)) if weights_path else None,
        },
    }


def write_policy(path: str | Path, content: dict[str, Any]) -> Path:
    """Write a policy file, refusing to replace one: a policy is frozen once used."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite the policy at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return path


def read_policy(
    path: str | Path,
    prune_layer: int,
    frame_step: int | None = None,
    weights: str | Path | None = None,
) -> BudgetPolicy:
    """Read a policy file and check it was calibrated for the same cut, sampling and,
    when ``weights`` is given, the same checkpoint."""
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    policy = BudgetPolicy.from_dict(content["policy"])
    calibration = content.get("calibration", {})
    checks = {"prune_layer": prune_layer, "frame_step": frame_step}
    for name, value in checks.items():
        recorded = calibration.get(name)
        if value is not None and recorded is not None and recorded != value:
            raise ValueError(
                f"the policy was calibrated with {name} {recorded} but is used with {value}; "
                "its statistics are not comparable. Calibrate again for this setting."
            )
    recorded = (calibration.get("weights") or {}).get("sha256")
    if weights is not None and recorded and recorded != SourceFile.of(weights).sha256:
        raise ValueError(
            "the policy was calibrated on another checkpoint; its statistics are on another "
            "scale. Calibrate again with these weights."
        )
    return policy
