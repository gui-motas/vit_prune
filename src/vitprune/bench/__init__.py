"""Measurement: the analytic cost model, drift, wall-clock latency and reports.

Separate from :mod:`vitprune.core` because measuring is not part of the method. The
core can be installed and used without any of this.
"""

from vitprune.bench.cost import ClipCost, CorpusBudget, Roofline, clip_cost, corpus_budget, roofline
from vitprune.bench.drift import DriftReport, measure_drift
from vitprune.bench.hardware import Accelerator, accelerator, custom
from vitprune.bench.latency import LatencyPoint, measure_latency
from vitprune.bench.report import (
    EmbeddingSummary,
    PruningSummary,
    RunReport,
    TimingSummary,
    summarise_pruning,
)

__all__ = [
    "Accelerator",
    "ClipCost",
    "CorpusBudget",
    "DriftReport",
    "EmbeddingSummary",
    "LatencyPoint",
    "PruningSummary",
    "Roofline",
    "RunReport",
    "TimingSummary",
    "accelerator",
    "clip_cost",
    "corpus_budget",
    "custom",
    "measure_drift",
    "measure_latency",
    "roofline",
    "summarise_pruning",
]
