"""Measures how long the backbone actually takes, against what was predicted.

This is the central measurement. The cost model in :mod:`vitprune.bench.cost` says
how many operations were saved; here we measure how much of that became time off the
clock. The ratio between the two is what this module calls **realisation**, and it is
the main result the project is after.

A realisation well below 100% is a finding, not a failure. It means the operation
count does not describe what the machine does, and the causes are investigable:
sorting the scores and scattering the merge cost time the count ignores; reading the
weights has a floor that pruning does not lower; and an operation that is cheap in
multiplies may simply have no fast kernel.

Three precautions decide whether the number is valid:

* **Warm up before timing.** The first runs pay one-off setup costs that never recur.
* **Synchronise with the device.** GPUs work asynchronously; without synchronising,
  the clock measures the time to queue the work, not to do it.
* **Use the same precision as extraction.** Different precisions take different code
  paths, and the two numbers then describe different things.

The input can be random: the operation count depends only on the shapes, which are
static. That means this measurement needs neither video nor a dataset, and can be the
first result obtained.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from vitprune.bench.cost import clip_cost
from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.geometry import ClipGeometry

DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20


@dataclass(frozen=True)
class LatencyPoint:
    """Time measured for one pruning configuration."""

    retention: float
    prune_layer: int
    ms_per_clip: float
    clips_per_second: float
    peak_memory_gb: float
    predicted_speedup: float
    measured_speedup: float

    @property
    def realisation(self) -> float:
        """What fraction of the predicted saving became actual time.

        1.0 means the operation count held up. Below that, part of the saving went
        into costs the count does not see.
        """
        return self.measured_speedup / self.predicted_speedup

    def describe(self) -> str:
        return (
            f"retention {self.retention:.0%}: {self.ms_per_clip:.2f} ms/clip, "
            f"{self.clips_per_second:.1f} clips/s, peak {self.peak_memory_gb:.2f} GB | "
            f"predicted {self.predicted_speedup:.2f}x, measured {self.measured_speedup:.2f}x "
            f"({self.realisation:.0%} realised)"
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["realisation"] = self.realisation
        return data


def measure_latency(
    backbone: torch.nn.Module,
    retentions: list[float],
    geometry: ClipGeometry | None = None,
    prune_layer: int = 2,
    batch: int = 8,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    device: str = "cuda",
    precision: str = "fp16",
    size_biased_attention: bool = False,
    progress=print,
) -> list[LatencyPoint]:
    """Time the backbone at each requested retention ratio.

    Args:
        backbone: the frozen video backbone.
        retentions: ratios to measure. The first is the reference for the speedups,
            so it is usually 1.0.
        geometry: the clip geometry.
        prune_layer: which block to prune after.
        batch: clips per forward pass.
        warmup: runs discarded before timing starts.
        iterations: timed runs.
        device: where to run.
        precision: ``fp16``, ``bf16`` or ``off``.
        size_biased_attention: enable the experimental path.
        progress: where to send progress output.

    Returns:
        One :class:`LatencyPoint` per ratio, in the order requested.
    """
    if not retentions:
        raise ValueError("provide at least one retention ratio")

    geometry = geometry or ClipGeometry.videomae_giant()
    sample = torch.randn(*geometry.input_shape(batch), device=device)
    dtype = _precision(precision)

    points: list[LatencyPoint] = []
    baseline_ms: float | None = None

    for retention in retentions:
        plan = _plan_for(retention, prune_layer, size_biased_attention)
        encoder = PrunedEncoder(backbone, geometry, plan)

        ms, peak = _time_one(encoder, sample, warmup, iterations, device, dtype, batch)
        if baseline_ms is None:
            baseline_ms = ms

        cost = clip_cost(geometry, plan.retention, plan.prune_layer)
        point = LatencyPoint(
            retention=retention,
            prune_layer=plan.prune_layer,
            ms_per_clip=round(ms, 3),
            clips_per_second=round(1000.0 / ms, 2),
            peak_memory_gb=round(peak, 3),
            predicted_speedup=round(cost.predicted_speedup, 3),
            measured_speedup=round(baseline_ms / ms, 3),
        )
        points.append(point)
        progress("  " + point.describe())

    return points


def _time_one(encoder, sample, warmup, iterations, device, dtype, batch):
    on_gpu = str(device).startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=dtype, enabled=on_gpu and dtype is not None):
                encoder.encode(sample)

        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        for _ in range(iterations):
            with torch.autocast("cuda", dtype=dtype, enabled=on_gpu and dtype is not None):
                encoder.encode(sample)
        if on_gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated() / 2**30 if on_gpu else 0.0
    return elapsed / iterations / batch * 1e3, peak


def _plan_for(retention: float, prune_layer: int, biased: bool) -> PruningPlan:
    if retention >= 1.0:
        return PruningPlan.dense()
    return PruningPlan(retention=retention, prune_layer=prune_layer, size_biased_attention=biased)


def _precision(name: str):
    table = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}
    if name not in table:
        raise ValueError(f"unknown precision '{name}'. Use: {', '.join(table)}")
    return table[name]
