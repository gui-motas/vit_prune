"""Measures how far the pruned embedding moves from the dense one.

The merge described in :mod:`vitprune.core.merging` guarantees an exact identity at
the moment it happens: size-weighted pooling over the survivors equals mean pooling
over all tokens. But the clip still has dozens of blocks ahead of it, and along the
way the two paths separate. The merged tokens are not the tokens that would have
existed without merging, so the remaining blocks produce different results.

How large that separation grows cannot be derived on paper. It depends on the
weights, on the content, and on how much was pruned. Only measurement answers it.

This does not replace an accuracy measurement. An embedding can move a long way and
still carry what a downstream model needs, or move very little and have lost exactly
what mattered. Drift is a cheap warning sign, not an answer: it says something
changed, never that something got worse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.bench.latency import autocast
from vitprune.core.encoder import PrunedEncoder, PruningPlan


@dataclass(frozen=True)
class DriftReport:
    """Drift measured over a set of clips.

    Attributes:
        clips: how many clips went into the figures.
        cosine_mean: mean cosine similarity against the dense embedding. Near 1
            means the pruned embedding points almost the same way.
        cosine_p5: the value below which the worst 5% of clips fall. More
            informative than the mean, which hides the bad cases.
        cosine_min: the worst clip of all.
        relative_l2_mean: mean L2 distance as a fraction of the dense norm.
        relative_l2_p95: that distance for the worst 5%.
    """

    clips: int
    cosine_mean: float
    cosine_p5: float
    cosine_min: float
    relative_l2_mean: float
    relative_l2_p95: float

    def describe(self) -> str:
        return (
            f"{self.clips} clips | cosine: mean {self.cosine_mean:.5f}, "
            f"worst 5% below {self.cosine_p5:.5f}, worst {self.cosine_min:.5f} | "
            f"relative L2: mean {self.relative_l2_mean:.5f}, "
            f"worst 5% above {self.relative_l2_p95:.5f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def measure_drift(
    backbone: torch.nn.Module,
    clips: torch.Tensor,
    plan: PruningPlan,
    geometry=None,
    device: str = "cpu",
    precision: str = "off",
    batch: int = 8,
) -> DriftReport:
    """Compare the pruned embedding against the dense one, clip by clip.

    Args:
        backbone: the frozen video backbone.
        clips: ``[clips, channels, frames, height, width]``.
        plan: the pruning plan to evaluate.
        geometry: the clip geometry; defaults to the Giant.
        device: where to run.
        precision: ``fp16``, ``bf16`` or ``off``.
        batch: clips per forward pass; each clip's figures do not depend on it.

    Returns:
        A :class:`DriftReport`.

    Both paths use the same backbone with the same weights. Pruning is the only
    difference between them, which is what isolates the effect being measured.
    """
    if clips.shape[0] == 0:
        raise ValueError("at least one clip is needed to measure drift")
    if batch < 1:
        raise ValueError("batch must be positive")

    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense())
    pruned = PrunedEncoder(backbone, geometry, plan)

    cosines: list[float] = []
    distances: list[float] = []

    for start in range(0, clips.shape[0], batch):
        chunk = clips[start : start + batch].to(device)
        with autocast(device, precision):
            a = dense.encode(chunk).float()
            b = pruned.encode(chunk).float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("cannot measure drift of non-finite embeddings")
        norms = a.norm(dim=-1)
        if (norms == 0).any() or (b.norm(dim=-1) == 0).any():
            raise ValueError("cannot measure drift of zero-norm embeddings")
        cosines += F.cosine_similarity(a, b, dim=-1).tolist()
        distances += ((a - b).norm(dim=-1) / norms).tolist()

    return _summarise(cosines, distances)


def _summarise(cosines: list[float], distances: list[float]) -> DriftReport:
    cosine = np.asarray(cosines, dtype=np.float64)
    apart = np.asarray(distances, dtype=np.float64)
    return DriftReport(
        clips=cosine.size,
        cosine_mean=float(cosine.mean()),
        cosine_p5=float(np.percentile(cosine, 5)),
        cosine_min=float(cosine.min()),
        relative_l2_mean=float(apart.mean()),
        relative_l2_p95=float(np.percentile(apart, 95)),
    )
