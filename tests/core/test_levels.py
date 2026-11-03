"""Every level of a clip measured from one cut: the calibration and the diagnosis read
these numbers, so each level must be exactly what inference would produce."""

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.policy import FixedBudget
from vitprune.core.testing import reference_backbone, reference_geometry

LEVELS = (16, 32, 48)
PROBE = {"cells": (2, 2), "top": 2}


@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def backbone(geometry):
    return reference_backbone(geometry).double()


@pytest.fixture
def clips(geometry):
    torch.manual_seed(13)
    return torch.randn(*geometry.input_shape(batch=4), dtype=torch.float64)


def _levels(backbone, geometry, clips, depths=(1,)):
    encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1))
    return encoder.encode_levels(clips, LEVELS, depths=depths, **PROBE)


def test_each_level_matches_the_online_path_with_a_fixed_budget(backbone, geometry, clips):
    found = _levels(backbone, geometry, clips)
    for level in LEVELS:
        online = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1,
                                                               budget=FixedBudget(level)))
        assert torch.allclose(found.embeddings[level], online.encode(clips), atol=1e-10)
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    assert torch.allclose(found.embeddings[geometry.num_tokens], dense, atol=1e-10)


def test_the_merge_preserves_the_mean_at_the_cut(backbone, geometry, clips):
    """The trap: right after the merge the means agree by construction, whatever was lost."""
    found = _levels(backbone, geometry, clips)
    for level in LEVELS:
        assert torch.allclose(found.stages[level][1], found.stages[geometry.num_tokens][1],
                              atol=1e-10)


def test_the_full_level_has_no_merge_error(backbone, geometry, clips):
    found = _levels(backbone, geometry, clips)
    assert torch.equal(found.merge_error[geometry.num_tokens],
                       torch.zeros_like(found.merge_error[geometry.num_tokens]))
    assert found.merge_error[16].max() > 0


def test_the_features_are_measured_before_any_merge(backbone, geometry, clips):
    found = _levels(backbone, geometry, clips)
    statistics = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1)).statistics(clips)
    assert torch.allclose(found.features.motion, statistics.motion)


@pytest.mark.parametrize("levels, depths, dense", [
    ((18,), (), False), (LEVELS, (0,), False), (LEVELS, (4,), False), (LEVELS, (), True),
])
def test_levels_and_depths_are_validated(backbone, geometry, clips, levels, depths, dense):
    plan = PruningPlan.dense() if dense else PruningPlan(prune_layer=1)
    with pytest.raises(ValueError):
        PrunedEncoder(backbone, geometry, plan).encode_levels(clips, levels, depths=depths,
                                                              **PROBE)
