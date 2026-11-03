"""The quality-aware budget, offline: per-clip drift of each budget, measured on videos.

``vitprune budget-curve`` runs every clip at every level from one cut
(:meth:`vitprune.core.encoder.PrunedEncoder.encode_levels`) and writes one record per clip:
its features at the cut and, per level, how far its final embedding moved from the dense
one. On calibration videos those records are what the policy is fitted on; on a few clips
they are the P0 diagnosis. Under the output folder:

* ``<video>.jsonl`` -- one line per clip, written whole when the video ends;
* ``sources.json`` -- each video's file, driver and clip count;
* ``settings.json`` -- what the records were measured under; a folder of other settings
  is refused, so an interrupted collection resumes video by video;
* ``summary.json`` -- the fidelity of each level over all the clips.

The driver of a video is the number after ``user_id_`` in its name, or the name itself;
splits between calibration and validation are by driver, never by clip.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vitprune.bench.latency import autocast
from vitprune.bench.quality import clip_records
from vitprune.core.compressibility import PROBE_CELLS, PROBE_TOP, fixed_anchors
from vitprune.core.encoder import PrunedEncoder
from vitprune.pipeline.extraction import _validate_encoding
from vitprune.pipeline.provenance import SourceFile
from vitprune.pipeline.video import find_videos, frame_count, iter_clips, spread_starts

#: G1: the full level must reproduce the dense path to within this, per embedding value.
IDENTITY_TOLERANCE = 1e-4
#: G2: below this median cosine at this level, a policy alone is unlikely to recover
#: fidelity; the merge and the cut need auditing first.
G2_LEVEL = 1536
G2_MEDIAN = 0.90

SETTINGS = "settings.json"
SOURCES = "sources.json"
SUMMARY = "summary.json"


def driver_of(stem: str) -> str:
    """The driver a video shows: the digits after ``user_id_``, or the whole name."""
    found = re.search(r"user_id_(\d+)", stem)
    return found.group(1) if found else stem


def collect_records(
    encoder: PrunedEncoder,
    videos: Sequence[str | Path],
    out: str | Path,
    levels: Sequence[int],
    clips_per_video: int = 48,
    depths: Sequence[int] = (),
    stride: int = 16,
    frame_step: int = 2,
    batch: int = 8,
    device: str = "cuda",
    precision: str = "off",
    weights_path: str | Path | None = None,
    head: Callable | None = None,
    progress=print,
) -> Path:
    """Measure every level on clips spread over each video and write one record per clip.

    Args:
        encoder: its plan sets the cut and the selection; it needs no budget.
        videos: files or folders.
        out: the records folder.
        levels: tokens kept per clip; the full level, the dense path, is always added.
        clips_per_video: clips spread evenly over each video; all of them if it has fewer.
        depths: blocks after which the drift of the token mean is also recorded.
        stride, frame_step: how clips are sampled.
        batch, device, precision: how the model runs.
        weights_path: the checkpoint, recorded.
        head: a classification head, for the top-1 agreement with the dense path.
        progress: where to send progress output.

    Returns:
        The records folder.
    """
    found = [path for item in videos for path in find_videos(item)]
    if not found:
        raise FileNotFoundError(f"no video found in {list(map(str, videos))}")
    stems = [path.stem for path in found]
    if len(set(stems)) != len(stems):
        raise ValueError(f"video names must be unique, got {sorted(stems)}")
    if clips_per_video < 1:
        raise ValueError("clips per video must be positive")
    _validate_encoding(batch, precision)

    geometry, plan = encoder.geometry, encoder.plan
    levels = sorted({*levels, geometry.num_tokens})
    settings = {
        "levels": levels,
        "depths": sorted(set(depths)),
        "clips_per_video": clips_per_video,
        "stride": stride,
        "frame_step": frame_step,
        "batch": batch,
        "precision": precision,
        "device": str(device),
        "prune_layer": plan.prune_layer,
        "motion_weight": plan.motion_weight,
        "spatial_prior": plan.spatial_prior is not None,
        "size_biased_attention": plan.size_biased_attention,
        "probe": {"cells": list(PROBE_CELLS), "top": PROBE_TOP,
                  "fixed": fixed_anchors(geometry.grid_size).tolist()},
        "geometry": asdict(geometry),
        "weights": SourceFile.of(weights_path).sha256 if weights_path else None,
    }
    out = Path(out)
    _settle(out, settings)
    sources = _read_json(out / SOURCES, {})

    for video in found:
        target = out / f"{video.stem}.jsonl"
        if target.exists():
            progress(f"[budget-curve] {video.name}: done before")
            continue
        driver = driver_of(video.stem)
        starts = spread_starts(frame_count(video), clips_per_video, geometry, stride, frame_step)
        lines, pending = [], []
        for item in iter_clips(video, geometry, stride, frame_step, starts):
            pending.append(item)
            if len(pending) == batch:
                lines += _measure(encoder, pending, levels, settings["depths"], device, precision,
                                  head)
                pending = []
        if pending:
            lines += _measure(encoder, pending, levels, settings["depths"], device, precision, head)
        lines = [{"video": video.stem, "driver": driver, **line} for line in lines]

        partial = target.with_suffix(".partial")
        partial.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        partial.replace(target)
        sources[video.stem] = {"source": asdict(SourceFile.of(video)), "driver": driver,
                               "clips": len(lines), "requested": clips_per_video}
        (out / SOURCES).write_text(json.dumps(sources, indent=2), encoding="utf-8")
        progress(f"[budget-curve] {video.name}: {len(lines)} clips")

    _, _, records = read_records(out)
    summary = summarise_records(records, levels)
    (out / SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _measure(encoder, pending, levels, depths, device, precision, head) -> list[dict]:
    """The records of a batch of ``(start, clip)``, each with its start."""
    clips = torch.stack([clip for _, clip in pending]).to(device)
    with autocast(device, precision):
        records = clip_records(encoder.encode_levels(clips, levels, depths), head)
    return [{"start": start, **record} for (start, _), record in zip(pending, records, strict=True)]


def read_records(folder: str | Path) -> tuple[dict, dict, list[dict]]:
    """The settings, the sources and every record of a folder, by video and start."""
    folder = Path(folder)
    settings = _read_json(folder / SETTINGS, None)
    if settings is None:
        raise FileNotFoundError(f"no records in {folder}: run 'vitprune budget-curve' first")
    sources = _read_json(folder / SOURCES, {})
    records = []
    for stem in sources:
        lines = (folder / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
        records += [json.loads(line) for line in lines]
    return settings, sources, sorted(records, key=lambda r: (r["video"], r["start"]))


def summarise_records(records: Sequence[dict], levels: Sequence[int]) -> dict:
    """The fidelity of each level below the full one, over all the clips."""
    summary = {"clips": len(records), "videos": len({r["video"] for r in records}),
               "drivers": len({r["driver"] for r in records}), "levels": {}}
    for level in sorted(levels)[:-1]:
        rows = [record["levels"][str(level)] for record in records]
        cos = 1.0 - np.array([row["d_cos"] for row in rows])
        relative = np.array([row["d_rel"] for row in rows])
        agree = [row["top1"] for row in rows]
        stages = {}
        for depth in rows[0]["stages"] if rows else ():
            ratio = [row["stages"][depth]["norm"] / record["dense"]["stages"][depth]["norm"]
                     for row, record in zip(rows, records, strict=True)]
            stages[depth] = {
                "cos_p50": float(np.median([row["stages"][depth]["cos"] for row in rows])),
                "rel_l2_p50": float(np.median([row["stages"][depth]["rel_l2"] for row in rows])),
                "norm_ratio_p50": float(np.median(ratio)),
            }
        summary["levels"][str(level)] = {
            "n": len(rows),
            "cos": {"p10": float(np.percentile(cos, 10)), "p50": float(np.percentile(cos, 50)),
                    "p90": float(np.percentile(cos, 90)), "mean": float(cos.mean())},
            "d_rel": {"mean": float(relative.mean()), "p90": float(np.percentile(relative, 90))},
            "top1": None if None in agree else float(np.mean(agree)),
            "merge_error": {
                "mean": float(np.mean([row["merge_error"]["mean"] for row in rows])),
                "p90": float(np.mean([row["merge_error"]["p90"] for row in rows])),
                "max": float(np.max([row["merge_error"]["max"] for row in rows])),
            },
            "norm_ratio_p50": float(np.median([row["norm"] / record["dense"]["norm"]
                                               for row, record in zip(rows, records,
                                                                      strict=True)])),
            "stages": stages,
        }
    return summary


def p0_verdict(summary: dict, identity_max_diff: float, backbone_matches: bool) -> dict:
    """The P0 gates, before any policy is built.

    G1: the backbone check passed and the full level reproduced the dense path. G2: the
    median cosine at 1536 tokens is at least 0.90. A failed G2 is not proof the policy is
    unviable; it may call for more conservative budgets or for auditing the merge.
    """
    median = summary["levels"][str(G2_LEVEL)]["cos"]["p50"]
    first = bool(backbone_matches) and identity_max_diff <= IDENTITY_TOLERANCE
    second = median >= G2_MEDIAN
    return {"G1": first, "G2": second, "passes": first and second,
            "identity_max_diff": identity_max_diff, "cos_p50_at_1536": median}


def _settle(out: Path, settings: dict) -> None:
    """Write the settings of a new folder, or refuse a folder measured under others."""
    before = _read_json(out / SETTINGS, None)
    if before is None:
        out.mkdir(parents=True, exist_ok=True)
        (out / SETTINGS).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    elif before != json.loads(json.dumps(settings)):
        raise FileExistsError(f"{out} holds records of other settings; use another folder")


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
