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

``vitprune calibrate-quality`` then fits the policy on the records of some drivers and
validates it on the others (:func:`fit_quality_policy`). The criteria are registered here,
before any result: V0, the policy compresses at all; V1, the probe tracks the drift; V2,
the clips it compresses stay within the tolerance; V4, the probe earns its cost against the
same table without it. Gate G3 passes only when all four do.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from vitprune.bench.cost import budget_cost
from vitprune.bench.latency import autocast
from vitprune.bench.quality import auc, clip_records, permutation_p, spearman
from vitprune.core.compressibility import (
    PROBE_CELLS,
    PROBE_TOP,
    CompressibilityFeatures,
    fixed_anchors,
)
from vitprune.core.encoder import PrunedEncoder
from vitprune.core.geometry import ClipGeometry
from vitprune.core.policy import REASONS, QualityAwareBudgetPolicy, calibrate_quality_aware
from vitprune.pipeline.extraction import _validate_encoding
from vitprune.pipeline.provenance import SourceFile
from vitprune.pipeline.video import find_videos, frame_count, iter_clips, spread_starts

#: G1: the full level must reproduce the dense path to within this, per embedding value.
IDENTITY_TOLERANCE = 1e-4
#: G2: below this median cosine at this level, a policy alone is unlikely to recover
#: fidelity; the merge and the cut need auditing first.
G2_LEVEL = 1536
G2_MEDIAN = 0.90

#: The schema of a quality-aware policy file.
SCHEMA = "vitprune.quality_policy/1"
#: Tolerances of the exploratory curve; it informs, it never chooses.
TOLERANCE_CURVE = (0.10, 0.15, 0.20, 0.30)
#: V0: at least this share of holdout clips gets fewer tokens than the dense path.
COMPRESSED_SHARE = 0.20
#: V1: a level "the policy uses" is chosen for at least this share of holdout clips, and the
#: probe's rank correlation with its drift is at least this, and positive for every driver.
USED_LEVEL_SHARE = 0.10
SIGNAL_RHO = 0.3
#: V2: at least this share of compressed clips stays within the tolerance; the nominal
#: share of a 90th percentile risk is reported apart.
COVERAGE_GATE = 0.85
NOMINAL_COVERAGE = 0.90
#: V4: the probe earns its cost with 2% less cost, 2 points more coverage, or an AUC 0.05
#: higher, without losing on the other two.
COST_MARGIN = 0.02
COVERAGE_MARGIN = 0.02
AUC_MARGIN = 0.05

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


def fit_quality_policy(
    folder: str | Path,
    holdout: Sequence[str],
    tolerance: float = 0.10,
    bins: int = 6,
    min_support: int = 30,
    min_videos: int = 3,
    no_probe: bool = False,
) -> dict:
    """Fit the quality-aware policy on some videos' records and validate it on others.

    Args:
        folder: records written by ``vitprune budget-curve`` on calibration videos.
        holdout: the videos kept out of the fit, by name; no driver may be on both sides.
        tolerance, bins, min_support, min_videos: see
            :class:`vitprune.core.policy.QualityAwareBudgetPolicy`.
        no_probe: write the ablation, the same table without the probe, instead.

    Returns:
        The content of a policy file: the policy, what it was calibrated on, and its
        validation on the holdout, both with and without the probe.
    """
    settings, sources, records = read_records(folder)
    if settings["spatial_prior"] or settings["size_biased_attention"]:
        raise ValueError("records measured with a spatial prior or size-biased attention "
                         "cannot calibrate the quality-aware policy")
    held_out = set(holdout)
    unknown = held_out - set(sources)
    if unknown:
        raise ValueError(f"holdout videos without records: {sorted(unknown)}")
    fit = [record for record in records if record["video"] not in held_out]
    held = [record for record in records if record["video"] in held_out]
    if not fit or not held:
        raise ValueError("the fit and the holdout both need records")
    shared = {r["driver"] for r in fit} & {r["driver"] for r in held}
    if shared:
        raise ValueError(f"drivers on both sides of the split: {sorted(shared)}")

    levels = settings["levels"]
    features, drift = _arrays(fit, levels)
    options = {"levels": levels, "tolerance": tolerance, "bins": bins,
               "min_support": min_support, "min_videos": min_videos,
               "motion_weight": settings["motion_weight"],
               "probe_cells": tuple(settings["probe"]["cells"]),
               "probe_top": settings["probe"]["top"]}
    names, drivers = [r["video"] for r in fit], [r["driver"] for r in fit]
    with_probe = calibrate_quality_aware(features, drift, names, drivers, alpha=0.5, **options)
    without = calibrate_quality_aware(features, drift, names, drivers, alpha=1.0, **options)
    geometry = ClipGeometry(**settings["geometry"])
    validation = validate_policies(with_probe, without, held, levels, tolerance, geometry,
                                   settings["prune_layer"])

    videos = [{"stem": stem, "driver": source["driver"],
               "split": "holdout" if stem in held_out else "fit",
               "source": source["source"], "clips": source["clips"]}
              for stem, source in sorted(sources.items())]
    return {
        "policy": (without if no_probe else with_probe).to_dict(),
        "calibration": {
            "schema": SCHEMA, "records": str(Path(folder).resolve()),
            "prune_layer": settings["prune_layer"], "stride": settings["stride"],
            "frame_step": settings["frame_step"], "precision": settings["precision"],
            "motion_weight": settings["motion_weight"], "probe": settings["probe"],
            "geometry": settings["geometry"], "weights": {"sha256": settings["weights"]},
            "videos": videos, "holdout": sorted(held_out),
            "clips": {"fit": len(fit), "holdout": len(held)},
        },
        "validation": validation,
    }


def validate_policies(
    primary: QualityAwareBudgetPolicy,
    ablation: QualityAwareBudgetPolicy,
    held: Sequence[dict],
    levels: Sequence[int],
    tolerance: float,
    geometry: ClipGeometry,
    prune_layer: int,
) -> dict:
    """The registered criteria on the holdout: V0, V1, V2 and V4, and gate G3.

    ``primary`` uses the probe and ``ablation`` does not. Every share comes with its count.
    """
    features, drift = _arrays(held, levels)
    lower = list(levels[:-1])
    videos, drivers = [r["video"] for r in held], [r["driver"] for r in held]
    probe_gmacs = (geometry.temporal_slices * geometry.tokens_per_slice * geometry.embed_dim
                   * (primary.probe_cells[0] * primary.probe_cells[1] + primary.probe_top)) / 1e9

    def operating(policy: QualityAwareBudgetPolicy, eps: float) -> dict:
        decision = replace(policy, tolerance=eps).choose_budget(features)
        tokens = decision.tokens.tolist()
        compressed = [t != levels[-1] for t in tokens]
        observed = [float(drift[i, lower.index(t)]) if c else 0.0
                    for i, (t, c) in enumerate(zip(tokens, compressed, strict=True))]
        within = [c and o <= eps for c, o in zip(compressed, observed, strict=True)]
        cost = budget_cost(geometry, tokens, prune_layer)
        gmacs = cost.gmacs_pruned + (probe_gmacs if policy.uses_probe else 0.0)
        by_driver = {}
        for driver in sorted(set(drivers)):
            mine = [i for i, d in enumerate(drivers) if d == driver]
            squeezed = sum(compressed[i] for i in mine)
            kept = sum(within[i] for i in mine)
            by_driver[driver] = {"clips": len(mine), "compressed": squeezed, "within": kept,
                                 "coverage": kept / squeezed if squeezed else None}
        squeezed = sum(compressed)
        return {
            "tolerance": eps, "clips": len(tokens),
            "levels": {str(level): tokens.count(level) for level in levels},
            "reasons": {reason: decision.reasons.count(reason) for reason in REASONS},
            "compressed": squeezed, "compressed_share": squeezed / len(tokens),
            "within": sum(within), "coverage": sum(within) / squeezed if squeezed else None,
            "coverage_by_driver": by_driver,
            "fallback_share": float(decision.used_fallback.double().mean()),
            "mean_tokens": float(np.mean(tokens)),
            "d_cos": {"mean": float(np.mean(observed)),
                      "p50": float(np.percentile(observed, 50)),
                      "p90": float(np.percentile(observed, 90))},
            "gmacs": gmacs, "predicted_speedup": cost.gmacs_dense / gmacs,
        }

    a, b = operating(primary, tolerance), operating(ablation, tolerance)
    v0 = a["compressed_share"] >= COMPRESSED_SHARE
    probe = features.probe_p90.numpy()
    index_a = primary.index(features).numpy()
    index_b = ablation.index(features).numpy()
    signal = {}
    for j, level in enumerate(lower):
        column = drift[:, j].numpy()
        signal[str(level)] = {
            "n": len(column),
            "chosen": a["levels"][str(level)],
            "rho_probe": spearman(probe, column),
            "p_probe_within_videos": permutation_p(probe, column, videos),
            "rho_probe_by_driver": {d: spearman(probe[_rows(drivers, d)], column[_rows(drivers, d)])
                                    for d in sorted(set(drivers))},
            "rho_probe_by_video": {v: spearman(probe[_rows(videos, v)], column[_rows(videos, v)])
                                   for v in sorted(set(videos))},
            "rho_g": spearman(index_b, column),
            "rho_h": spearman(index_a, column),
            "auc_with_probe": auc(_predicted(primary, features, j), column > tolerance),
            "auc_without_probe": auc(_predicted(ablation, features, j), column > tolerance),
        }
    used = [str(level) for level in lower
            if a["levels"][str(level)] / len(held) >= USED_LEVEL_SHARE]
    v1 = v2 = v4 = None
    if v0:
        v1 = bool(used) and all(
            signal[level]["rho_probe"] >= SIGNAL_RHO
            and all(rho > 0 for rho in signal[level]["rho_probe_by_driver"].values())
            for level in used
        )
        v2 = a["coverage"] >= COVERAGE_GATE
        v4 = _probe_pays(a, b, signal, used, v2)

    return {
        "tolerance": tolerance,
        "with_probe": a,
        "without_probe": b,
        "V0": {"passes": v0, "compressed": a["compressed"], "clips": a["clips"],
               "share": a["compressed_share"], "threshold": COMPRESSED_SHARE},
        "V1": {"passes": v1, "used_levels": used, "threshold": SIGNAL_RHO, "by_level": signal},
        "V2": {"passes": v2, "coverage": a["coverage"], "compressed": a["compressed"],
               "within": a["within"], "threshold": COVERAGE_GATE,
               "nominal": NOMINAL_COVERAGE,
               "nominal_reached": None if a["coverage"] is None
               else a["coverage"] >= NOMINAL_COVERAGE,
               "by_driver": a["coverage_by_driver"]},
        "V4": {"passes": v4, "cost_with_probe": a["gmacs"], "cost_without_probe": b["gmacs"],
               "coverage_with_probe": a["coverage"], "coverage_without_probe": b["coverage"]},
        "G3": {"V0": v0, "V1": v1, "V2": v2, "V4": v4,
               "passes": bool(v0 and v1 and v2 and v4)},
        "bins": _bins(primary, features, drift, lower),
        "curve": {"with_probe": [operating(primary, eps) for eps in TOLERANCE_CURVE],
                  "without_probe": [operating(ablation, eps) for eps in TOLERANCE_CURVE],
                  "fixed": {str(level): _fixed(drift[:, j].numpy(), level, geometry, prune_layer)
                            for j, level in enumerate(lower)}},
    }


def describe_validation(validation: dict) -> str:
    """A few lines a person reads to decide gate G3."""
    lines = []
    for name, found in (("with the probe", validation["with_probe"]),
                        ("without the probe", validation["without_probe"])):
        coverage = found["coverage"]
        lines.append(
            f"{name}: {found['compressed']}/{found['clips']} clips below the dense path, "
            f"levels {found['levels']}, coverage "
            f"{'n/a' if coverage is None else f'{coverage:.1%}'}, mean tokens "
            f"{found['mean_tokens']:.0f}, predicted speedup {found['predicted_speedup']:.2f}x, "
            f"fallback {found['fallback_share']:.0%}"
        )
    for level, found in validation["V1"]["by_level"].items():
        lines.append(f"  {level} tokens: probe rho {found['rho_probe']:.3f} "
                     f"(p {found['p_probe_within_videos']:.4f}, n {found['n']}), "
                     f"g rho {found['rho_g']:.3f}, AUC {found['auc_with_probe']:.3f} with the "
                     f"probe against {found['auc_without_probe']:.3f} without")
    gate = validation["G3"]
    failed = [name for name in ("V0", "V1", "V2", "V4") if not gate[name]]
    lines.append("G3: passed" if gate["passes"] else f"G3: failed ({', '.join(failed)})")
    return "\n".join(lines)


def _probe_pays(a: dict, b: dict, signal: dict, used: list[str], v2: bool) -> bool:
    if b["compressed_share"] < COMPRESSED_SHARE:
        return bool(v2)  # without the probe nothing compresses; with it, within coverage
    coverage_a, coverage_b = a["coverage"], b["coverage"]
    if coverage_a is None or coverage_b is None:
        return False
    cheaper = a["gmacs"] <= (1 - COST_MARGIN) * b["gmacs"]
    if coverage_a >= coverage_b and cheaper:
        return True
    if a["gmacs"] <= b["gmacs"] and coverage_a >= coverage_b + COVERAGE_MARGIN:
        return True
    if used and coverage_a >= coverage_b and a["gmacs"] <= b["gmacs"]:
        top = max(used, key=lambda level: a["levels"][level])
        gain = signal[top]["auc_with_probe"] - signal[top]["auc_without_probe"]
        return bool(gain >= AUC_MARGIN)
    return False


def _predicted(policy: QualityAwareBudgetPolicy, features, column: int) -> np.ndarray:
    """The calibrated risk at one level for each clip, read from its bin."""
    safe = CompressibilityFeatures(*(torch.nan_to_num(v, 0.0, 0.0, 0.0) for v in features))
    bins = torch.searchsorted(torch.tensor(policy.bin_edges, dtype=torch.float64),
                              policy.index(safe).contiguous(), right=True)
    return np.asarray(policy.risk, dtype=float)[bins.numpy(), column]


def _bins(policy: QualityAwareBudgetPolicy, features, drift, lower) -> list[dict]:
    """Per bin: the risk calibrated on the fit against the 90th percentile on the holdout."""
    safe = CompressibilityFeatures(*(torch.nan_to_num(v, 0.0, 0.0, 0.0) for v in features))
    where = torch.searchsorted(torch.tensor(policy.bin_edges, dtype=torch.float64),
                               policy.index(safe).contiguous(), right=True).numpy()
    found = []
    for b, row in enumerate(policy.risk):
        members = np.flatnonzero(where == b)
        found.append({
            "fit_support": policy.support[b], "fit_videos": policy.support_videos[b],
            "fit_drivers": policy.support_drivers[b], "holdout": int(members.size),
            "levels": {str(level): {
                "predicted": row[j],
                "observed_p90": float(np.quantile(drift[members, j].numpy(), 0.9,
                                                  method="higher")) if members.size else None,
            } for j, level in enumerate(lower)},
        })
    return found


def _fixed(column: np.ndarray, level: int, geometry: ClipGeometry, prune_layer: int) -> dict:
    """A fixed level on the holdout, at each tolerance of the curve."""
    return {
        "mean_d_cos": float(column.mean()), "p90_d_cos": float(np.percentile(column, 90)),
        "within": {str(eps): float((column <= eps).mean()) for eps in TOLERANCE_CURVE},
        "predicted_speedup": budget_cost(geometry, [level], prune_layer).predicted_speedup,
    }


def _rows(labels: Sequence[str], label: str) -> np.ndarray:
    return np.array([found == label for found in labels])


def _arrays(records: Sequence[dict], levels: Sequence[int]):
    """Features ``[clips]`` and drift ``[clips, levels - 1]``, refusing a gap by name."""
    features, drift = [], []
    for record in records:
        name = f"{record['video']} at frame {record['start']}"
        try:
            row = [record["levels"][str(level)]["d_cos"] for level in levels[:-1]]
        except KeyError as error:
            raise ValueError(f"{name} has no record of level {error}") from error
        values = [record["motion"], record["redundancy"], record["probe_mean"],
                  record["probe_p90"]]
        if not all(isinstance(v, (int, float)) and isfinite(v) for v in values + row):
            raise ValueError(f"{name} holds a value that is not finite")
        features.append(values)
        drift.append(row)
    columns = torch.tensor(features, dtype=torch.float64).T
    return CompressibilityFeatures(*columns), torch.tensor(drift, dtype=torch.float64)


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
