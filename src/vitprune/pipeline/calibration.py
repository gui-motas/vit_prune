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

from vitprune.core.encoder import PrunedEncoder
from vitprune.core.policy import BudgetPolicy, calibrate_policy
from vitprune.core.statistics import ClipStatistics
from vitprune.pipeline.extraction import _precision, _validate_encoding
from vitprune.pipeline.provenance import SourceFile
from vitprune.pipeline.video import find_videos, read_video, to_clips


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
    limit: int | None = None,
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
        batch, device, precision: how the blocks up to the cut are run.
        limit: stop after this many clips, or use every clip when ``None``.
        weights_path: the checkpoint, recorded for traceability.
        progress: where to send progress output.

    Returns:
        The content of a policy file, ready for :func:`write_policy`.
    """
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")
    if limit is not None and limit < 1:
        raise ValueError("the clip limit must be positive")
    _validate_encoding(batch, precision)
    dtype = _precision(precision)
    on_gpu = str(device).startswith("cuda")

    redundancy, motion, sources, total = [], [], [], 0
    for video in found:
        clips = to_clips(read_video(video), encoder.geometry, stride, frame_step)
        if limit is not None:
            clips = clips[: limit - total]
        for start in range(0, clips.shape[0], batch):
            chunk = clips[start : start + batch].to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=on_gpu and dtype is not None):
                statistics = encoder.statistics(chunk)
            redundancy.append(statistics.redundancy.cpu())
            motion.append(statistics.motion.cpu())
        total += clips.shape[0]
        sources.append({"source": asdict(SourceFile.of(video)), "clips": int(clips.shape[0])})
        progress(f"[calibrate] {video.name}: {clips.shape[0]} clips")
        if limit is not None and total >= limit:
            break

    policy = calibrate_policy(
        ClipStatistics(torch.cat(redundancy), torch.cat(motion)), levels, motion_balance
    )
    return {
        "policy": policy.to_dict(),
        "calibration": {
            "clips": total,
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
    path: str | Path, prune_layer: int, frame_step: int | None = None
) -> BudgetPolicy:
    """Read a policy file and check it was calibrated for the same cut and sampling."""
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
    return policy
