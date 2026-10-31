"""The method: token scoring, merging, and the encoder that drives a backbone.

Everything here depends only on ``numpy`` and ``torch``. Reading video files and
driving an external temporal model live in :mod:`vitprune.pipeline`; measurement
lives in :mod:`vitprune.bench`.

Typical use::

    from vitprune.core import (
        ClipGeometry, PrunedEncoder, PruningPlan, calibrate_policy, check_backbone
    )

    geometry = ClipGeometry.videomae_giant()
    check_backbone(backbone, geometry, reference=backbone.forward_features)

    probe = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=2))
    policy = calibrate_policy(probe.statistics(calibration_clips), [256, 512, 768])

    encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=2, budget=policy))
    features = encoder.encode(clips)
"""

from vitprune.core.compat import CompatibilityReport, check_backbone
from vitprune.core.encoder import ClipBudget, PrunedEncoder, PruningPlan
from vitprune.core.geometry import ClipGeometry
from vitprune.core.inspection import TokenSelection
from vitprune.core.merging import MergedTokens, initial_sizes, merge_tokens, size_weighted_mean
from vitprune.core.policy import BudgetPolicy, calibrate_policy
from vitprune.core.priors import box_prior, top_rows_prior
from vitprune.core.scoring import score_tokens
from vitprune.core.statistics import ClipStatistics, clip_statistics

__all__ = [
    "BudgetPolicy",
    "ClipBudget",
    "ClipGeometry",
    "ClipStatistics",
    "CompatibilityReport",
    "MergedTokens",
    "PrunedEncoder",
    "PruningPlan",
    "TokenSelection",
    "box_prior",
    "calibrate_policy",
    "check_backbone",
    "clip_statistics",
    "initial_sizes",
    "merge_tokens",
    "score_tokens",
    "size_weighted_mean",
    "top_rows_prior",
]
