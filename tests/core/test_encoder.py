"""The test that anchors everything: with a dense plan the encoder has to reproduce
the backbone's own forward pass exactly.
"""

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.geometry import ClipGeometry
from vitprune.core.priors import top_rows_prior
from vitprune.core.testing import reference_backbone, reference_encode, reference_geometry


@pytest.fixture
def geometry():
    return reference_geometry(depth=4)


@pytest.fixture
def backbone(geometry):
    return reference_backbone(geometry).double()


@pytest.fixture
def clips(geometry):
    torch.manual_seed(7)
    return torch.randn(*geometry.input_shape(batch=2), dtype=torch.float64)


def test_the_dense_plan_reproduces_the_original_forward(backbone, geometry, clips):
    expected = reference_encode(backbone, clips)
    actual = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    assert torch.allclose(expected, actual, atol=1e-12)


def test_full_retention_also_reproduces_the_original(backbone, geometry, clips):
    """Keeping 100% must be indistinguishable from not pruning, even with a cut
    requested."""
    plan = PruningPlan(retention=1.0, prune_layer=2)
    expected = reference_encode(backbone, clips)
    assert torch.allclose(expected, PrunedEncoder(backbone, geometry, plan).encode(clips))


def test_pruning_returns_one_embedding_per_clip(backbone, geometry, clips):
    plan = PruningPlan(retention=0.25, prune_layer=1)
    assert PrunedEncoder(backbone, geometry, plan).encode(clips).shape == (2, geometry.embed_dim)


def test_pruning_changes_the_embedding(backbone, geometry, clips):
    """If nothing changed, pruning would not be happening."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    plan = PruningPlan(retention=0.25, prune_layer=1)
    pruned = PrunedEncoder(backbone, geometry, plan).encode(clips)
    assert not torch.allclose(dense, pruned, atol=1e-6)


def test_pruning_after_the_last_block_leaves_the_embedding_almost_unchanged(
    backbone, geometry, clips
):
    """With no blocks left after the merge, the pooling identity shows through
    end to end."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    plan = PruningPlan(retention=0.25, prune_layer=geometry.depth)
    late = PrunedEncoder(backbone, geometry, plan).encode(clips)
    assert torch.allclose(dense, late, atol=1e-10)


def test_pruning_early_drifts_more_than_pruning_late(backbone, geometry, clips):
    """The more blocks run after the merge, the further the result moves."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)

    def drift(layer):
        plan = PruningPlan(retention=0.25, prune_layer=layer)
        return (PrunedEncoder(backbone, geometry, plan).encode(clips) - dense).norm().item()

    assert drift(1) > drift(geometry.depth)


@pytest.mark.parametrize("retention", [0.75, 0.5, 0.25, 0.125])
def test_several_retention_ratios_produce_valid_embeddings(backbone, geometry, clips, retention):
    plan = PruningPlan(retention=retention, prune_layer=2)
    out = PrunedEncoder(backbone, geometry, plan).encode(clips)
    assert out.shape == (2, geometry.embed_dim)
    assert torch.isfinite(out).all()


def test_the_spatial_prior_changes_the_selection(backbone, geometry, clips):
    without = PrunedEncoder(backbone, geometry, PruningPlan(retention=0.25, prune_layer=1)).encode(
        clips
    )
    with_prior = PrunedEncoder(
        backbone,
        geometry,
        PruningPlan(
            retention=0.25,
            prune_layer=1,
            spatial_prior=top_rows_prior(geometry.grid_size, rows=2, weight=0.0),
        ),
    ).encode(clips)
    assert not torch.allclose(without, with_prior, atol=1e-6)


def test_the_size_bias_does_not_affect_the_dense_path(backbone, geometry, clips):
    """Without pruning every size is 1, so enabling the bias changes nothing."""
    plan = PruningPlan(retention=1.0, prune_layer=0, size_biased_attention=True)
    expected = reference_encode(backbone, clips)
    assert torch.allclose(expected, PrunedEncoder(backbone, geometry, plan).encode(clips))


def test_the_size_bias_affects_the_pruned_path(backbone, geometry, clips):
    plain = PruningPlan(retention=0.25, prune_layer=1)
    biased = PruningPlan(retention=0.25, prune_layer=1, size_biased_attention=True)
    a = PrunedEncoder(backbone, geometry, plain).encode(clips)
    b = PrunedEncoder(backbone, geometry, biased).encode(clips)
    assert not torch.allclose(a, b, atol=1e-8)


def test_an_incompatible_geometry_fails_with_a_useful_message(backbone, clips):
    wrong = reference_geometry(depth=4)
    object.__setattr__(wrong, "patch_size", 4)
    with pytest.raises(ValueError, match="tokens"):
        PrunedEncoder(backbone, wrong, PruningPlan.dense()).encode(clips)


def test_the_backbone_is_frozen(backbone, geometry):
    PrunedEncoder(backbone, geometry)
    assert all(not p.requires_grad for p in backbone.parameters())


def test_the_default_geometry_is_the_giant():
    assert PrunedEncoder(reference_backbone()).geometry.num_tokens == ClipGeometry().num_tokens


def test_the_dense_plan_describes_itself_as_the_baseline():
    assert "baseline" in PruningPlan.dense().describe()
    assert not PruningPlan.dense().prunes


def test_setting_retention_to_one_yields_the_dense_plan():
    plan = PruningPlan(retention=0.25, prune_layer=3)
    assert not plan.with_retention(1.0).prunes
    assert plan.with_retention(0.5).prune_layer == 3
