"""What pruning should save, according to an operation count.

Nothing runs here. This module counts multiply-accumulate operations. It works with
no GPU, no checkpoint and without ``torch`` being importable, so that an estimate is
available before any machine time is booked.

Two caveats worth more than the numbers produced here.

**The count ignores everything that is not a multiply.** Sorting the scores,
selecting the survivors and scattering the merged ones cost real time that appears on
no line of this file.

**Not every saved operation becomes saved time.** There is a floor: the weights have
to be read from memory for every clip, and that cost does not shrink when tokens are
pruned. The more aggressive the pruning, the closer to that floor, and the less of
the predicted saving materialises. :func:`roofline` estimates where the floor sits.

The gap between what this predicts and what a clock measures is the subject of study.
Treating these numbers as a result would sidestep the question worth asking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vitprune.bench.hardware import Accelerator
from vitprune.core.geometry import ClipGeometry

#: One multiply-accumulate is two floating point operations.
FLOPS_PER_MAC = 2

#: Reduced precision: two bytes per weight.
BYTES_PER_WEIGHT = 2

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class ClipCost:
    """The cost of processing a single clip."""

    tokens_dense: int
    tokens_pruned: int
    gmacs_dense: float
    gmacs_pruned: float

    @property
    def predicted_speedup(self) -> float:
        return self.gmacs_dense / self.gmacs_pruned

    def describe(self) -> str:
        return (
            f"{self.tokens_dense} -> {self.tokens_pruned} tokens | "
            f"{self.gmacs_dense:.0f} -> {self.gmacs_pruned:.0f} GMAC/clip | "
            f"predicted speedup {self.predicted_speedup:.2f}x"
        )


@dataclass(frozen=True)
class Roofline:
    """Where the weight-reading floor sits, and how far above it we still are."""

    compute_ms: float
    weights_ms: float

    @property
    def ratio(self) -> float:
        return self.compute_ms / self.weights_ms

    @property
    def compute_bound(self) -> bool:
        """When false, pruning further has stopped paying off."""
        return self.ratio > 1.0

    def describe(self) -> str:
        regime = "compute bound" if self.compute_bound else "memory bound"
        return (
            f"compute {self.compute_ms:.1f} ms, weight read {self.weights_ms:.1f} ms, "
            f"ratio {self.ratio:.1f} ({regime})"
        )


@dataclass(frozen=True)
class CorpusBudget:
    """How much machine time a whole corpus costs."""

    hours_of_video: float
    clips: int
    gpu_hours: float

    def describe(self) -> str:
        return (
            f"{self.hours_of_video:.1f} h of video = {self.clips} clips "
            f"= {self.gpu_hours:.2f} GPU hours"
        )


def clip_cost(geometry: ClipGeometry, retention: float, prune_layer: int) -> ClipCost:
    """Count the operations in one clip, with and without pruning.

    The per-block count sums three parts: producing query, key and value; comparing
    every token against every other; and the feed-forward layer. Only the second
    grows with the square of the token count, and most of the saving comes from it.
    """
    dense = _accumulate(geometry, retention=1.0, prune_layer=0)
    pruned = _accumulate(geometry, retention, prune_layer)
    prunes = prune_layer > 0 and retention < 1.0
    return ClipCost(
        tokens_dense=geometry.num_tokens,
        tokens_pruned=geometry.kept_tokens(retention) if prunes else geometry.num_tokens,
        gmacs_dense=dense,
        gmacs_pruned=pruned,
    )


def parameter_count(geometry: ClipGeometry) -> int:
    """How many weights the backbone holds, derived from the geometry.

    Each block carries the three attention projections, the output projection and the
    two feed-forward layers. With the Giant values that comes to roughly a billion,
    which matches the published size.
    """
    dim = geometry.embed_dim
    per_block = (4 + 2 * geometry.mlp_ratio) * dim * dim
    return int(per_block * geometry.depth)


def roofline(geometry: ClipGeometry, cost: ClipCost, device: Accelerator) -> Roofline:
    """Compare time spent computing against time spent merely reading the weights.

    While compute dominates, pruning tokens pays off. As the two converge, pruning
    stops helping: the card waits on memory instead.
    """
    compute_seconds = cost.gmacs_pruned * 1e9 * FLOPS_PER_MAC / device.usable_flops
    weights_seconds = parameter_count(geometry) * BYTES_PER_WEIGHT / device.bandwidth
    return Roofline(compute_seconds * 1e3, weights_seconds * 1e3)


def corpus_budget(
    hours_of_video: float,
    cost: ClipCost,
    device: Accelerator,
    fps: int = 30,
    stride: int = 16,
) -> CorpusBudget:
    """Estimate the machine time needed to process a corpus.

    Args:
        hours_of_video: total duration of the corpus.
        cost: the output of :func:`clip_cost`.
        device: the accelerator considered.
        fps: frames per second of the source video.
        stride: frames between the start of consecutive clips.
    """
    if hours_of_video < 0:
        raise ValueError("corpus duration cannot be negative")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    frames = hours_of_video * SECONDS_PER_HOUR * fps
    clips = int(frames / stride)
    seconds = clips * cost.gmacs_pruned * 1e9 * FLOPS_PER_MAC / device.usable_flops
    return CorpusBudget(hours_of_video, clips, seconds / SECONDS_PER_HOUR)


def as_dict(item) -> dict:
    """Turn any result from this module into a dictionary, for writing to disk."""
    data = asdict(item)
    for name in ("predicted_speedup", "ratio", "compute_bound"):
        if hasattr(item, name):
            data[name] = getattr(item, name)
    return data


def _accumulate(geometry: ClipGeometry, retention: float, prune_layer: int) -> float:
    tokens = geometry.num_tokens
    total = 0.0
    for index in range(geometry.depth):
        total += _block_macs(tokens, geometry)
        if prune_layer > 0 and (index + 1) == prune_layer and retention < 1.0:
            tokens = geometry.kept_tokens(retention)
    return total / 1e9


def _block_macs(tokens: int, geometry: ClipGeometry) -> float:
    dim = geometry.embed_dim
    projections = 4 * tokens * dim * dim
    comparisons = 2 * tokens * tokens * dim
    feed_forward = 2 * geometry.mlp_ratio * tokens * dim * dim
    return projections + comparisons + feed_forward
