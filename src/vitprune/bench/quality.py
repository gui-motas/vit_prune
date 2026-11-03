"""How far each budget moves a clip's embedding from the dense one, clip by clip.

:meth:`vitprune.core.encoder.PrunedEncoder.encode_levels` runs every level of the same
clips from one cut. This module turns that into one record per clip, the material a
quality-aware policy is calibrated on and the P0 diagnosis reads, and holds the few
statistics their validation rests on.

Drift is measured against the dense path of the same clip:

* ``d_cos = 1 - cos(E_dense, E_k)``, the fidelity the policy's tolerance is set on;
* ``d_rel = |E_dense - E_k| / |E_dense|``, which also sees a change of norm;
* ``top1``, whether a classification head gives the same class.

None of them is accuracy: they say how much pruning changed the embedding, not whether
the answer is right.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.core.compressibility import PROBE_CELLS, PROBE_TOP, compressibility_features
from vitprune.core.encoder import LevelEncoding
from vitprune.core.geometry import ClipGeometry
from vitprune.core.statistics import clip_statistics


@torch.no_grad()
def clip_records(encoding: LevelEncoding, head: Callable | None = None) -> list[dict]:
    """One record per clip: its features at the cut and, per level below the full one, its
    drift from the dense embedding, its merge error and the drift after each recorded block.

    Args:
        encoding: what ``encode_levels`` returned.
        head: a classification head; with one, ``top1`` says whether each level gives the
            dense path's class.
    """
    full = max(encoding.embeddings)
    reference = encoding.embeddings[full].double().cpu()
    dense_stages = {depth: vector.double().cpu() for depth, vector in encoding.stages[full].items()}
    classes = None if head is None else head(encoding.embeddings[full]).argmax(dim=-1).cpu()

    measured = {}
    for level, embedding in encoding.embeddings.items():
        if level == full:
            continue
        pruned = embedding.double().cpu()
        errors = encoding.merge_error[level].double().cpu().flatten(1)
        measured[level] = {
            "d_cos": 1.0 - F.cosine_similarity(reference, pruned, dim=-1),
            "d_rel": (reference - pruned).norm(dim=-1) / reference.norm(dim=-1),
            "norm": pruned.norm(dim=-1),
            "top1": None if classes is None
            else head(embedding).argmax(dim=-1).cpu() == classes,
            "merge_error": {"mean": errors.mean(dim=1),
                            "p90": torch.quantile(errors, 0.9, dim=1),
                            "max": errors.amax(dim=1)},
            "stages": {depth: _compare(dense_stages[depth], vector.double().cpu())
                       for depth, vector in encoding.stages[level].items()},
        }

    features = encoding.features
    records = []
    for clip in range(reference.shape[0]):
        records.append({
            "motion": float(features.motion[clip]),
            "redundancy": float(features.redundancy[clip]),
            "probe_mean": float(features.probe_mean[clip]),
            "probe_p90": float(features.probe_p90[clip]),
            "dense": {"norm": float(reference[clip].norm()),
                      "stages": {str(depth): {"norm": float(vector[clip].norm())}
                                 for depth, vector in sorted(dense_stages.items())}},
            "levels": {str(level): {
                "d_cos": float(found["d_cos"][clip]),
                "d_rel": float(found["d_rel"][clip]),
                "norm": float(found["norm"][clip]),
                "top1": None if found["top1"] is None else bool(found["top1"][clip]),
                "merge_error": {name: float(value[clip])
                                for name, value in found["merge_error"].items()},
                "stages": {str(depth): {name: float(value[clip]) for name, value in stage.items()}
                           for depth, stage in sorted(found["stages"].items())},
            } for level, found in sorted(measured.items())},
        })
    return records


def _compare(dense: torch.Tensor, pruned: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"cos": F.cosine_similarity(dense, pruned, dim=-1),
            "rel_l2": (dense - pruned).norm(dim=-1) / dense.norm(dim=-1),
            "norm": pruned.norm(dim=-1)}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation, ties at their average rank; NaN when a side is constant."""
    return _pearson(_ranks(x), _ranks(y))


def partial_spearman(x: Sequence[float], y: Sequence[float], control: Sequence[float]) -> float:
    """Spearman's correlation of ``x`` and ``y`` once a straight line in the ranks of
    ``control`` is taken out of both: what ``x`` tells about ``y`` beyond ``control``."""
    rc = _ranks(control)
    design = np.c_[np.ones_like(rc), rc]

    def residual(values: Sequence[float]) -> np.ndarray:
        ranks = _ranks(values)
        return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]

    return _pearson(residual(x), residual(y))


def permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence | None = None,
    permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """One-sided p-value of Spearman's correlation being this high by chance.

    ``y`` is shuffled within each group only, so a correlation that comes from differences
    between groups -- one video with more of both -- is not taken for a relation inside
    them. Clips of a video are not independent: read the result as exploratory.
    """
    rx, ry = _ranks(x), _ranks(y)
    observed = _pearson(rx, ry)
    if np.isnan(observed):
        return float("nan")
    labels = np.zeros(len(ry)) if groups is None else np.asarray(groups)
    members = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(seed)
    shuffled, hits = ry.copy(), 0
    for _ in range(permutations):
        for indices in members:
            shuffled[indices] = ry[rng.permutation(indices)]
        hits += _pearson(rx, shuffled) >= observed - 1e-12
    return (1 + hits) / (1 + permutations)


def auc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """The chance that a positive scores above a negative, ties counting half; NaN when one
    of the classes is empty."""
    scores, positives = np.asarray(scores, dtype=float), np.asarray(positives, dtype=bool)
    hits, misses = int(positives.sum()), int((~positives).sum())
    if not hits or not misses:
        return float("nan")
    ranks = _ranks(scores)
    return float((ranks[positives].sum() - hits * (hits + 1) / 2) / (hits * misses))


def time_probe(
    geometry: ClipGeometry,
    batch: int = 8,
    device: str = "cuda",
    warmup: int = 3,
    iterations: int = 20,
    cells: tuple[int, int] = PROBE_CELLS,
    top: int = PROBE_TOP,
) -> dict:
    """Milliseconds per clip of the probe and of the clip statistics, median of the runs.

    Random tokens of the real shape: both cost the same whatever the content.
    """
    generator = torch.Generator().manual_seed(0)
    shape = (batch, geometry.temporal_slices, geometry.tokens_per_slice, geometry.embed_dim)
    tokens = torch.randn(shape, generator=generator).to(device)
    scores = torch.rand(shape[:3], generator=generator).to(device)
    statistics = clip_statistics(tokens)

    def per_clip(run) -> float:
        for _ in range(warmup):
            run()
        times = []
        for _ in range(iterations):
            _synchronise(device)
            started = time.perf_counter()
            run()
            _synchronise(device)
            times.append(time.perf_counter() - started)
        return float(np.median(times)) / batch * 1e3

    return {
        "probe_ms_per_clip": per_clip(
            lambda: compressibility_features(tokens, scores, statistics, cells, top)),
        "statistics_ms_per_clip": per_clip(lambda: clip_statistics(tokens)),
        "batch": batch,
        "device": str(device),
    }


def _synchronise(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values))
    ranks[values.argsort(kind="mergesort")] = np.arange(1, len(values) + 1)
    _, tie, counts = np.unique(values, return_inverse=True, return_counts=True)
    return np.bincount(tie, weights=ranks)[tie] / counts[tie]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / scale) if scale > 0 else float("nan")
