"""Clip statistics feed the per-clip token budget. What matters is that each clip's
values depend on that clip alone and keep the scale differences between clips.
"""

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.statistics import clip_statistics
from vitprune.core.testing import reference_backbone, reference_geometry


def _tokens(batch=3, slices=4, count=16, dim=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, slices, count, dim)


def test_one_value_per_clip():
    stats = clip_statistics(_tokens(batch=3))
    assert stats.redundancy.shape == stats.motion.shape == (3,)
    assert stats.redundancy.dtype == stats.motion.dtype == torch.float32


def test_a_uniform_clip_is_maximally_redundant():
    assert torch.allclose(clip_statistics(torch.ones(1, 4, 16, 8)).redundancy, torch.ones(1))


def test_varied_content_is_less_redundant_than_uniform_content():
    uniform = clip_statistics(torch.ones(1, 4, 16, 8)).redundancy
    assert clip_statistics(_tokens(batch=1)).redundancy < uniform


def test_a_still_clip_has_no_motion():
    still = _tokens(batch=1, slices=1).expand(1, 4, 16, 8)
    assert clip_statistics(still).motion.item() == 0.0


def test_motion_grows_with_the_change_between_slices():
    base, step = _tokens(batch=1, slices=1, seed=1), _tokens(batch=1, slices=1, seed=2)
    ramp = torch.arange(4.0).view(1, 4, 1, 1)
    slow = clip_statistics(base + ramp * step).motion
    fast = clip_statistics(base + ramp * 2 * step).motion
    assert torch.allclose(fast, 2 * slow)


def test_a_single_slice_has_no_motion():
    """The image geometry case: there are no consecutive slices to compare."""
    stats = clip_statistics(_tokens(slices=1))
    assert torch.all(stats.motion == 0.0)
    assert torch.isfinite(stats.redundancy).all()


def test_scale_differences_between_clips_are_kept():
    """Per-slice normalisation would erase this, which is why it is not applied."""
    clip = _tokens(batch=1)
    stats = clip_statistics(torch.cat([clip, 10 * clip]))
    assert torch.allclose(stats.motion[1], 10 * stats.motion[0])
    assert torch.allclose(stats.redundancy[1], stats.redundancy[0])


def test_each_clip_ignores_the_rest_of_the_batch():
    tokens = _tokens(batch=3)
    together = clip_statistics(tokens)
    alone = [clip_statistics(tokens[i : i + 1]) for i in range(3)]
    assert torch.allclose(together.redundancy, torch.cat([s.redundancy for s in alone]))
    assert torch.allclose(together.motion, torch.cat([s.motion for s in alone]))


def test_a_wrong_input_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        clip_statistics(torch.randn(4, 16, 8))


# ---------------------------------------------------------------------------
# Through the encoder
# ---------------------------------------------------------------------------
@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def encoder(geometry):
    plan = PruningPlan(retention=0.25, prune_layer=1)
    return PrunedEncoder(reference_backbone(geometry), geometry, plan)


@pytest.fixture
def clips(geometry):
    torch.manual_seed(5)
    return torch.randn(*geometry.input_shape(batch=3))


def test_the_encoder_measures_the_grid_at_the_pruning_layer(encoder, geometry, clips):
    backbone = encoder.backbone
    tokens = backbone.patch_embed(clips).flatten(2).transpose(1, 2) + backbone.pos_embed
    tokens = backbone.blocks[0](tokens)
    grid = tokens.reshape(clips.shape[0], geometry.temporal_slices, geometry.tokens_per_slice, -1)

    expected, actual = clip_statistics(grid), encoder.statistics(clips)

    assert torch.allclose(actual.redundancy, expected.redundancy)
    assert torch.allclose(actual.motion, expected.motion)


def test_the_encoder_statistics_ignore_the_rest_of_the_batch(encoder, clips):
    together = encoder.statistics(clips)
    alone = [encoder.statistics(clips[i : i + 1]) for i in range(clips.shape[0])]
    for name in ("redundancy", "motion"):
        expected = torch.cat([getattr(s, name) for s in alone])
        assert torch.allclose(getattr(together, name), expected, rtol=1e-4, atol=1e-6)


def test_statistics_need_a_pruning_layer(geometry, clips):
    encoder = PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan.dense())
    with pytest.raises(ValueError, match="prunes nothing"):
        encoder.statistics(clips)
