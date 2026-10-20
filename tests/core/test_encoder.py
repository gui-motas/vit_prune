"""The test that anchors everything: with a dense plan the encoder has to reproduce
the backbone's own forward pass exactly.
"""

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
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
    """With an identity token norm and no remaining blocks, the identity holds."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    plan = PruningPlan(retention=0.25, prune_layer=geometry.depth)
    late = PrunedEncoder(backbone, geometry, plan).encode(clips)
    assert torch.allclose(dense, late, atol=1e-10)


def test_nonlinear_token_norm_can_break_final_equivalence_even_after_the_last_block(
    backbone, geometry, clips
):
    backbone.norm = torch.nn.LayerNorm(geometry.embed_dim).double()
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    late = PrunedEncoder(
        backbone, geometry, PruningPlan(prune_layer=geometry.depth)
    ).encode(clips)
    assert not torch.allclose(dense, late, atol=1e-8)


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


def test_default_giant_geometry_does_not_silently_accept_a_toy_backbone():
    with pytest.raises(ValueError, match="depth"):
        PrunedEncoder(reference_backbone())


def test_the_dense_plan_describes_itself_as_the_baseline():
    assert "baseline" in PruningPlan.dense().describe()
    assert not PruningPlan.dense().prunes


def test_setting_retention_to_one_yields_the_dense_plan():
    plan = PruningPlan(retention=0.25, prune_layer=3)
    assert not plan.with_retention(1.0).prunes
    assert plan.with_retention(0.5).prune_layer == 3


@pytest.mark.parametrize("retention", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_retention_is_rejected_at_construction(retention):
    with pytest.raises(ValueError, match="retention"):
        PruningPlan(retention=retention)
    with pytest.raises(ValueError, match="retention"):
        PruningPlan().with_retention(retention)


@pytest.mark.parametrize("layer", [-1, 1.5, True])
def test_invalid_layer_is_rejected(layer):
    with pytest.raises(ValueError, match="prune_layer"):
        PruningPlan(prune_layer=layer)


@pytest.mark.parametrize("retention", [0.25, 1.0])
def test_cut_past_the_backbone_is_rejected_even_when_dense(backbone, geometry, retention):
    with pytest.raises(ValueError, match="exceeds"):
        PrunedEncoder(backbone, geometry, PruningPlan(retention=retention, prune_layer=100))


def test_encode_and_select_reject_a_replaced_invalid_plan(backbone, geometry, clips):
    encoder = PrunedEncoder(backbone, geometry)
    encoder.plan = PruningPlan(prune_layer=100)
    for operation in (encoder.encode, encoder.select):
        with pytest.raises(ValueError, match="exceeds"):
            operation(clips)


def test_construction_sets_eval_without_changing_weight_values(backbone, geometry):
    backbone.train()
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    encoder = PrunedEncoder(backbone, geometry)
    assert not encoder.training
    assert all(not module.training for module in backbone.modules())
    assert all(torch.equal(before[name], value) for name, value in backbone.state_dict().items())


def test_training_mode_is_rejected(backbone, geometry, clips):
    encoder = PrunedEncoder(backbone, geometry).train()
    for operation in (encoder.encode, encoder.select):
        with pytest.raises(ValueError, match="inference-only"):
            operation(clips)


def test_patch_embedding_width_is_checked(backbone, geometry, clips):
    backbone.patch_embed = torch.nn.Identity()
    with pytest.raises(ValueError, match="width"):
        PrunedEncoder(backbone, geometry).encode(clips)


def test_input_shape_is_checked(backbone, geometry, clips):
    with pytest.raises(ValueError, match="shape"):
        PrunedEncoder(backbone, geometry).encode(clips[:, :, 0])


def test_size_biased_attention_requires_a_videomae_block(geometry):
    backbone = reference_backbone(geometry)
    backbone.blocks[-1] = torch.nn.Identity()
    with pytest.raises(ValueError, match="size-biased attention"):
        PrunedEncoder(backbone, geometry, PruningPlan(size_biased_attention=True))


@pytest.mark.parametrize("prior", [torch.ones(2, 2), torch.tensor([float("nan")]),
                                   torch.tensor([-0.1])])
def test_invalid_prior_is_rejected_at_construction(prior):
    with pytest.raises(ValueError, match="spatial_prior"):
        PruningPlan(spatial_prior=prior)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_invalid_motion_weight_is_rejected_at_construction(weight):
    with pytest.raises(ValueError, match="motion_weight"):
        PruningPlan(motion_weight=weight)
