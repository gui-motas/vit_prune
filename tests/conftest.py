"""Shared test helpers."""

import pytest

from vitprune.core.policy import BudgetPolicy


@pytest.fixture
def uniform_budget():
    """Build a budget giving every clip the same token count.

    Test-only: a calibrated policy decides counts from the content, which makes exact
    counts awkward to assert. With the threshold at zero every clip lands in the top
    level, so tests about merging, selection and shapes get a predictable cut.
    """

    def build(geometry, tokens: int) -> BudgetPolicy:
        return BudgetPolicy(
            redundancy_reference=[0.0, 1.0],
            motion_reference=[0.0, 1.0],
            thresholds=[0.0],
            levels=[geometry.temporal_slices, tokens],
        )

    return build


@pytest.fixture
def quality_budget():
    """Build a quality-aware policy calibrated on the given clips themselves.

    Test-only: a real policy is calibrated on other videos than the ones it serves. Each clip
    counts as its own video and driver, and the probe uses 4 + 2 anchors, which a 4x4 grid
    can hold.
    """
    import torch
    import torch.nn.functional as F

    from vitprune.core.encoder import PrunedEncoder, PruningPlan
    from vitprune.core.policy import calibrate_quality_aware

    def build(backbone, geometry, clips, levels, prune_layer=1, **options):
        encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=prune_layer))
        found = encoder.encode_levels(clips, levels[:-1], cells=(2, 2), top=2)
        dense = found.embeddings[levels[-1]]
        drift = torch.stack([1 - F.cosine_similarity(dense, found.embeddings[level], dim=-1)
                             for level in levels[:-1]], dim=1)
        names = [str(clip) for clip in range(clips.shape[0])]
        options = {"bins": 2, "min_support": 1, "min_videos": 1, "probe_cells": (2, 2),
                   "probe_top": 2} | options
        return calibrate_quality_aware(found.features, drift, names, names, levels, **options)

    return build
