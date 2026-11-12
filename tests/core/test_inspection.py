"""Inspection has to reflect the real decision. If it diverged from what encode()
does, a figure would show one selection while the embedding came from another.
"""

import numpy as np
import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.priors import top_rows_prior
from vitprune.core.testing import reference_backbone, reference_geometry


@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def encoder(geometry, uniform_budget):
    plan = PruningPlan(prune_layer=1, budget=uniform_budget(geometry, 16))
    return PrunedEncoder(reference_backbone(geometry), geometry, plan)


@pytest.fixture
def clips(geometry):
    torch.manual_seed(11)
    return torch.randn(*geometry.input_shape(batch=2))


def test_the_selection_covers_every_clip_and_slice(encoder, geometry, clips):
    s = encoder.select(clips)
    assert s.clips == 2
    assert s.temporal_slices == geometry.temporal_slices
    assert s.tokens_per_slice == geometry.tokens_per_slice


def test_it_keeps_the_budgeted_count(encoder, geometry):
    s = encoder.select(torch.randn(*geometry.input_shape(batch=1)))
    assert s.kept_per_slice == s.kept_in(0) == 16 // geometry.temporal_slices


def test_the_mask_marks_exactly_the_kept_tokens(encoder, clips):
    s = encoder.select(clips)
    mask = s.keep_mask(clip=0, slice_index=0)
    assert mask.shape == (s.grid_size, s.grid_size)
    assert mask.sum() == s.kept_per_slice


def test_the_kept_tokens_are_the_highest_scoring(encoder, clips):
    """The figure has to reflect the criterion, not an approximation of it."""
    s = encoder.select(clips)
    scores = s.score_grid(0, 0).ravel()
    kept = s.keep_mask(0, 0).ravel()

    assert scores[kept].min() >= scores[~kept].max()


def test_the_sizes_sum_to_the_whole_slice(encoder, clips):
    """Every pruned token was absorbed by someone; none vanished."""
    s = encoder.select(clips)
    assert s.size_grid(0, 0).sum() == pytest.approx(s.tokens_per_slice)


def test_pruned_positions_have_size_zero(encoder, clips):
    s = encoder.select(clips)
    assert np.all(s.size_grid(0, 0)[~s.keep_mask(0, 0)] == 0.0)


def test_every_kept_token_stands_for_at_least_itself(encoder, clips):
    s = encoder.select(clips)
    assert np.all(s.size_grid(0, 0)[s.keep_mask(0, 0)] >= 1.0)


def test_the_selection_is_reproducible(encoder, clips):
    first = encoder.select(clips)
    second = encoder.select(clips)
    assert torch.equal(first.kept, second.kept)
    assert torch.allclose(first.scores, second.scores)


def test_the_spatial_prior_moves_the_selection(geometry, clips, uniform_budget):
    """With the top rows zeroed, no token from there should survive."""
    plan = PruningPlan(
        prune_layer=1,
        spatial_prior=top_rows_prior(geometry.grid_size, rows=2, weight=0.0),
        budget=uniform_budget(geometry, 16),
    )
    s = PrunedEncoder(reference_backbone(geometry), geometry, plan).select(clips)
    assert s.rows_share(0, 0, rows=2) == 0.0


def test_the_top_rows_share_is_a_fraction(encoder, clips):
    assert 0.0 <= encoder.select(clips).rows_share(0, 0, rows=2) <= 1.0


def test_inspecting_a_dense_plan_is_rejected(geometry, clips):
    """With no pruning there is no selection to show."""
    encoder = PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan.dense())
    with pytest.raises(ValueError, match="prunes nothing"):
        encoder.select(clips)


def test_inspecting_a_cut_without_a_budget_is_rejected(geometry, clips):
    encoder = PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan(prune_layer=1))
    with pytest.raises(ValueError, match="without a budget"):
        encoder.select(clips)


def test_the_selection_describes_itself(encoder, clips):
    assert "temporal slices" in encoder.select(clips).describe()


@pytest.mark.parametrize("autocast", [False, True])
def test_inspection_uses_the_actual_encode_decision(encoder, clips, monkeypatch, autocast):
    from vitprune.core import encoder as implementation

    original = implementation.merge_tokens
    decisions = []

    def observe(tokens, sizes, scores, keep, groups=None):
        result = original(tokens, sizes, scores, keep, groups)
        decisions.append((scores.clone(), result.sizes.clone()))
        return result

    monkeypatch.setattr(implementation, "merge_tokens", observe)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        encoder.encode(clips)
        selection = encoder.select(clips)
    assert torch.equal(decisions[0][0], selection.scores)
    assert torch.equal(decisions[0][1].squeeze(-1), selection.sizes)
