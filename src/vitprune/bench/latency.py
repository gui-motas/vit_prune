"""Measures how long the backbone actually takes, against what was predicted.

This is the central measurement. The cost model in :mod:`vitprune.bench.cost` says
how many operations were saved; here we measure how much of that became time off the
clock. The ratio between the two is what this module calls **realisation**, and it is
the main result the project is after.

A realisation well below 100% is a finding, not a failure. It means the operation
count does not describe what the machine does, and the causes are investigable:
computing the clip statistics, sorting the scores, scattering the merge and running
one sub-batch per budget level cost time the count ignores; reading the weights has a
floor that pruning does not lower; and an operation that is cheap in multiplies may
simply have no fast kernel.

Three precautions decide whether the number is valid:

* **Warm up before timing.** The first runs pay one-off setup costs that never recur.
* **Synchronise with the device.** GPUs work asynchronously; without synchronising,
  the clock measures the time to queue the work, not to do it.
* **Use the same precision as extraction.** Different precisions take different code
  paths, and the two numbers then describe different things.

Under an adaptive budget the cost depends on the content, which decides how many
tokens each clip keeps. Measure it on real clips, not random input.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from vitprune.bench.cost import budget_cost
from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.geometry import ClipGeometry

DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20


@dataclass(frozen=True)
class LatencyPoint:
    """Time measured for one plan.

    Attributes:
        plan: the plan's description.
        mean_tokens: tokens kept after the cut, averaged over the clips.
        ms_per_clip: mean time per clip over the timed iterations.
        p50_ms, p95_ms: median and 95th percentile of the per-iteration time per clip.
    """

    plan: str
    mean_tokens: float
    ms_per_clip: float
    p50_ms: float
    p95_ms: float
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
            f"{self.plan}: {self.ms_per_clip:.2f} ms/clip (p50 {self.p50_ms:.2f}, "
            f"p95 {self.p95_ms:.2f}), {self.clips_per_second:.1f} clips/s, "
            f"{self.mean_tokens:.0f} tokens, peak {self.peak_memory_gb:.2f} GB | "
            f"predicted {self.predicted_speedup:.2f}x, measured {self.measured_speedup:.2f}x "
            f"({self.realisation:.0%} realised)"
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["realisation"] = self.realisation
        return data


def measure_plans(
    backbone: torch.nn.Module,
    clips: torch.Tensor,
    plans: list[PruningPlan],
    geometry: ClipGeometry | None = None,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    device: str = "cuda",
    precision: str = "fp16",
    progress=print,
) -> list[LatencyPoint]:
    """Time each plan on the same clips, as one batch.

    Args:
        backbone: the frozen video backbone.
        clips: ``[batch, channels, frames, height, width]``, timed as one batch.
        plans: plans to time. The first must be the dense baseline, the denominator
            of every speedup.
        geometry: the clip geometry.
        warmup: runs discarded before timing starts.
        iterations: timed runs.
        device: where to run.
        precision: ``fp16``, ``bf16`` or ``off``.
        progress: where to send progress output.

    Returns:
        One :class:`LatencyPoint` per plan, in the order given. The predicted speedup
        of an adaptive plan comes from the token counts its clips actually kept.
    """
    if not plans:
        raise ValueError("provide at least one plan")
    if plans[0].prunes:
        raise ValueError("the first plan must be the dense baseline")
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    if clips.shape[0] < 1:
        raise ValueError("at least one clip is needed to measure latency")

    geometry = geometry or ClipGeometry.videomae_giant()
    dtype = _precision(precision)
    encoders = [PrunedEncoder(backbone, geometry, plan) for plan in plans]
    sample = clips.to(device)

    points: list[LatencyPoint] = []
    baseline_ms: float | None = None
    for plan, encoder in zip(plans, encoders, strict=True):
        tokens = _tokens_kept(encoder, sample, device, dtype)
        times, peak = _time(encoder, sample, warmup, iterations, device, dtype)
        ms = float(np.mean(times))
        if baseline_ms is None:
            baseline_ms = ms

        cost = budget_cost(geometry, tokens, plan.prune_layer if plan.prunes else 0)
        point = LatencyPoint(
            plan=plan.describe(),
            mean_tokens=float(np.mean(tokens)),
            ms_per_clip=round(ms, 3),
            p50_ms=round(float(np.percentile(times, 50)), 3),
            p95_ms=round(float(np.percentile(times, 95)), 3),
            clips_per_second=round(1000.0 / ms, 2),
            peak_memory_gb=round(peak, 3),
            predicted_speedup=round(cost.predicted_speedup, 3),
            measured_speedup=round(baseline_ms / ms, 3),
        )
        points.append(point)
        progress("  " + point.describe())

    return points


def _tokens_kept(encoder: PrunedEncoder, sample, device, dtype) -> list[int]:
    """Tokens each clip keeps after the cut. Runs once, outside the timed loop."""
    if not encoder.plan.prunes:
        return [encoder.geometry.num_tokens] * sample.shape[0]
    with torch.no_grad(), _autocast(device, dtype):
        _, budget = encoder.encode_with_budget(sample)
    return budget.tokens.tolist()


def _time(encoder, sample, warmup, iterations, device, dtype) -> tuple[list[float], float]:
    """Milliseconds per clip for each timed iteration, and the peak memory in GB."""
    on_gpu = str(device).startswith("cuda")
    times = []

    with torch.no_grad():
        for _ in range(warmup):
            with _autocast(device, dtype):
                encoder.encode(sample)

        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(iterations):
            started = time.perf_counter()
            with _autocast(device, dtype):
                encoder.encode(sample)
            if on_gpu:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) / sample.shape[0] * 1e3)

    peak = torch.cuda.max_memory_allocated() / 2**30 if on_gpu else 0.0
    return times, peak


def _autocast(device, dtype):
    return torch.autocast(
        "cuda", dtype=dtype, enabled=str(device).startswith("cuda") and dtype is not None
    )


def _precision(name: str):
    table = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}
    if name not in table:
        raise ValueError(f"unknown precision '{name}'. Use: {', '.join(table)}")
    return table[name]
