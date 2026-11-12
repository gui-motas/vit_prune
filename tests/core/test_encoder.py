"""The test that anchors everything: with a dense plan the encoder has to reproduce
the backbone's own forward pass exactly.
"""

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.policy import FixedBudget
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


@pytest.fixture
def pruned(geometry, uniform_budget):
    """A plan keeping the same number of tokens in every clip, 16 of 64 by default."""

    def plan(tokens=16, **settings):
        settings.setdefault("prune_layer", 1)
        return PruningPlan(budget=uniform_budget(geometry, tokens), **settings)

    return plan


def test_the_dense_plan_reproduces_the_original_forward(backbone, geometry, clips):
    expected = reference_encode(backbone, clips)
    actual = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    assert torch.allclose(expected, actual, atol=1e-12)


def test_a_cut_without_a_budget_prunes_nothing(backbone, geometry, clips):
    """The cut only marks where statistics are measured; pruning needs a budget."""
    plan = PruningPlan(prune_layer=2)
    assert not plan.prunes
    expected = reference_encode(backbone, clips)
    assert torch.allclose(expected, PrunedEncoder(backbone, geometry, plan).encode(clips))


def test_a_budget_of_every_token_reproduces_the_original(backbone, geometry, clips, pruned):
    plan = pruned(geometry.num_tokens, prune_layer=2)
    expected = reference_encode(backbone, clips)
    assert torch.allclose(expected, PrunedEncoder(backbone, geometry, plan).encode(clips))


def test_pruning_returns_one_embedding_per_clip(backbone, geometry, clips, pruned):
    out = PrunedEncoder(backbone, geometry, pruned()).encode(clips)
    assert out.shape == (2, geometry.embed_dim)


def test_pruning_changes_the_embedding(backbone, geometry, clips, pruned):
    """If nothing changed, pruning would not be happening."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    out = PrunedEncoder(backbone, geometry, pruned()).encode(clips)
    assert not torch.allclose(dense, out, atol=1e-6)


def test_pruning_after_the_last_block_leaves_the_embedding_almost_unchanged(
    backbone, geometry, clips, pruned
):
    """With an identity token norm and no remaining blocks, the identity holds."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    late = PrunedEncoder(backbone, geometry, pruned(prune_layer=geometry.depth)).encode(clips)
    assert torch.allclose(dense, late, atol=1e-10)


def test_nonlinear_token_norm_can_break_final_equivalence_even_after_the_last_block(
    backbone, geometry, clips, pruned
):
    backbone.norm = torch.nn.LayerNorm(geometry.embed_dim).double()
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    late = PrunedEncoder(backbone, geometry, pruned(prune_layer=geometry.depth)).encode(clips)
    assert not torch.allclose(dense, late, atol=1e-8)


def test_pruning_early_drifts_more_than_pruning_late(backbone, geometry, clips, pruned):
    """The more blocks run after the merge, the further the result moves."""
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)

    def drift(layer):
        out = PrunedEncoder(backbone, geometry, pruned(prune_layer=layer)).encode(clips)
        return (out - dense).norm().item()

    assert drift(1) > drift(geometry.depth)


@pytest.mark.parametrize("tokens", [48, 32, 16, 8])
def test_several_budgets_produce_valid_embeddings(backbone, geometry, clips, pruned, tokens):
    out = PrunedEncoder(backbone, geometry, pruned(tokens, prune_layer=2)).encode(clips)
    assert out.shape == (2, geometry.embed_dim)
    assert torch.isfinite(out).all()


def test_the_spatial_prior_changes_the_selection(backbone, geometry, clips, pruned):
    without = PrunedEncoder(backbone, geometry, pruned()).encode(clips)
    prior = top_rows_prior(geometry.grid_size, rows=2, weight=0.0)
    with_prior = PrunedEncoder(backbone, geometry, pruned(spatial_prior=prior)).encode(clips)
    assert not torch.allclose(without, with_prior, atol=1e-6)


def test_the_size_bias_does_not_affect_the_dense_path(backbone, geometry, clips):
    """Without pruning every size is 1, so enabling the bias changes nothing."""
    plan = PruningPlan(prune_layer=0, size_biased_attention=True)
    expected = reference_encode(backbone, clips)
    assert torch.allclose(expected, PrunedEncoder(backbone, geometry, plan).encode(clips))


def test_the_size_bias_affects_the_pruned_path(backbone, geometry, clips, pruned):
    a = PrunedEncoder(backbone, geometry, pruned()).encode(clips)
    b = PrunedEncoder(backbone, geometry, pruned(size_biased_attention=True)).encode(clips)
    assert not torch.allclose(a, b, atol=1e-8)


@pytest.mark.parametrize("budget", [None, FixedBudget(16)])
def test_sdpa_attention_matches_model_attention(backbone, geometry, clips, budget, monkeypatch):
    model = PruningPlan(prune_layer=2, budget=budget)
    sdpa = PruningPlan(prune_layer=2, budget=budget, attention="sdpa")
    expected = PrunedEncoder(backbone, geometry, model).encode(clips)
    actual_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def observe(*args, **kwargs):
        calls.append(args[0].shape[-2])
        return actual_sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    actual = PrunedEncoder(backbone, geometry, sdpa).encode(clips)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert calls == ([64, 64, 64, 64] if budget is None else [64, 64, 16, 16])
    assert "SDPA attention" in sdpa.describe()


def test_attention_rejects_unknown_value():
    with pytest.raises(ValueError, match="attention"):
        PruningPlan(attention="flash")


def test_dense_plan_accepts_attention():
    plan = PruningPlan.dense(attention="sdpa")
    assert plan.attention == "sdpa"
    assert not plan.prunes
    assert "SDPA attention" in plan.describe()


@pytest.mark.parametrize("index", [0, -1])
def test_sdpa_requires_block_attributes(geometry, index):
    backbone = reference_backbone(geometry)
    backbone.blocks[index].attn.qkv = None
    with pytest.raises(ValueError, match="attention.qkv"):
        PrunedEncoder(backbone, geometry, PruningPlan.dense(attention="sdpa"))


def test_stages_empty_is_unchanged(backbone, geometry, clips):
    default = PruningPlan(budget=FixedBudget(16))
    empty = PruningPlan(budget=FixedBudget(16), stages=())
    actual = PrunedEncoder(backbone, geometry, empty).encode(clips)
    expected = PrunedEncoder(backbone, geometry, default).encode(clips)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("attention", ["model", "sdpa"])
def test_stages_shrink_tokens_after_block(attention):
    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    clips = torch.randn(*geometry.input_shape(batch=2))
    seen = []
    for block in backbone.blocks:
        block.attn.proj.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape[1])
        )
    plan = PruningPlan(budget=FixedBudget(32), stages=((3, 24), (5, 12)), attention=attention)
    encoder = PrunedEncoder(backbone, geometry, plan)
    result = encoder.encode(clips)
    assert seen == [64, 64, 32, 24, 24, 12]
    assert result.shape == (2, geometry.embed_dim)
    assert torch.isfinite(result).all()
    assert "stages 3:24 5:12" in plan.describe()


def test_bfloat16_encoder_preserves_mass_through_stages_and_pool(monkeypatch):
    import vitprune.core.encoder as module

    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry).bfloat16()
    encoder = PrunedEncoder(backbone, geometry,
                            PruningPlan(budget=FixedBudget(32), stages=((3, 24), (5, 12))))
    original_merge, original_pool = module.bipartite_merge, encoder._pool
    masses = []

    def merge(*args):
        tokens, sizes, groups = original_merge(*args)
        masses.append(sizes)
        return tokens, sizes, groups

    def pool(tokens, sizes):
        masses.append(sizes)
        return original_pool(tokens, sizes)

    monkeypatch.setattr(module, "bipartite_merge", merge)
    monkeypatch.setattr(encoder, "_pool", pool)
    encoder.encode(torch.randn(*geometry.input_shape()).bfloat16())
    assert len(masses) == 3
    assert all(mass.dtype == torch.float32 and mass.sum().item() == 64 for mass in masses)


@pytest.mark.parametrize("stages", [
    ((3, 24), (3, 16)), ((4, 24), (3, 16)), ((2, 24),), ((6, 24),),
    ((3, 32),), ((3, 24), (4, 24)), ((3, 24), (4, 25)), ((3, 15),),
    ((3, 0),), ((True, 24),), ((3, 24.5),), ((3,),),
])
def test_stages_validation(stages):
    geometry = reference_geometry(depth=6)
    with pytest.raises(ValueError, match="stage"):
        plan = PruningPlan(budget=FixedBudget(32), stages=stages)
        PrunedEncoder(reference_backbone(geometry), geometry, plan)


def test_stages_require_fixed_budget(backbone, geometry, clips, uniform_budget, quality_budget):
    quality = quality_budget(backbone, geometry, clips, [16, 64])
    for budget in (None, uniform_budget(geometry, 32), quality):
        with pytest.raises(ValueError, match="FixedBudget"):
            PruningPlan(budget=budget, stages=((3, 16),))


def test_stages_refused_by_select_and_encode_levels():
    geometry = reference_geometry(depth=6)
    encoder = PrunedEncoder(reference_backbone(geometry), geometry,
                            PruningPlan(budget=FixedBudget(32), stages=((3, 24),)))
    clips = torch.randn(*geometry.input_shape())
    with pytest.raises(ValueError, match="stages"):
        encoder.select(clips)
    with pytest.raises(ValueError, match="stages"):
        encoder.encode_levels(clips, [32])


@pytest.mark.parametrize("biased", [False, True])
def test_stage_over_dense_group(biased):
    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    seen = []
    backbone.blocks[3].attn.proj.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape[1])
    )
    encoder = PrunedEncoder(backbone, geometry,
                            PruningPlan(budget=FixedBudget(64), stages=((3, 48),),
                                        size_biased_attention=biased))
    result = encoder.encode(torch.randn(*geometry.input_shape()))
    assert seen == [48]
    assert torch.isfinite(result).all()


def test_stage_requires_next_block_attributes():
    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    backbone.blocks[3].attn.qkv = None
    with pytest.raises(ValueError, match="attention.qkv"):
        PrunedEncoder(backbone, geometry, PruningPlan(budget=FixedBudget(32), stages=((3, 24),)))


def test_stage_uses_next_block_for_keys(monkeypatch):
    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    encoder = PrunedEncoder(backbone, geometry,
                            PruningPlan(budget=FixedBudget(32), stages=((3, 24),)))
    seen = []
    original = encoder._stage_keys

    def observe(block, tokens):
        seen.append(block)
        return original(block, tokens)

    monkeypatch.setattr(encoder, "_stage_keys", observe)
    encoder.encode(torch.randn(*geometry.input_shape()))
    assert len(seen) == 1 and seen[0] is backbone.blocks[3]


@pytest.mark.parametrize("split_bias", [True, False])
def test_stages_allow_qkv_without_a_bias_attribute(split_bias):
    from vitprune.core.testing import ReferenceAttention

    class BiaslessQKV(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.detach().clone())

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    attention = ReferenceAttention(geometry.embed_dim, geometry.heads, split_bias)
    backbone.blocks[3].attn = attention
    attention.qkv = BiaslessQKV(attention.qkv.weight)
    encoder = PrunedEncoder(backbone, geometry,
                            PruningPlan(budget=FixedBudget(32), stages=((3, 24),)))
    assert torch.isfinite(encoder.encode(torch.randn(*geometry.input_shape()))).all()


@pytest.mark.parametrize("split_bias", [True, False])
def test_stage_keys_use_next_block_key_projection(split_bias):
    from vitprune.core.testing import ReferenceAttention

    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    block = backbone.blocks[3]
    block.attn = ReferenceAttention(geometry.embed_dim, geometry.heads, split_bias)
    if split_bias:
        block.attn.q_bias.data.fill_(2)
    else:
        block.attn.qkv.bias.data[geometry.embed_dim:2 * geometry.embed_dim].fill_(2)
    encoder = PrunedEncoder(backbone, geometry, PruningPlan(budget=FixedBudget(32),
                                                           stages=((3, 24),)))
    x = torch.randn(2, 24, geometry.embed_dim)
    qkv = block.attn._qkv(block.norm1(x))
    expected = qkv[1].transpose(1, 2).mean(dim=2)
    torch.testing.assert_close(encoder._stage_keys(block, x), expected)


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


def test_a_budget_needs_a_pruning_layer(geometry, uniform_budget):
    with pytest.raises(ValueError, match="pruning layer"):
        PruningPlan(prune_layer=0, budget=uniform_budget(geometry, 16))


@pytest.mark.parametrize("layer", [-1, 1.5, True])
def test_invalid_layer_is_rejected(layer):
    with pytest.raises(ValueError, match="prune_layer"):
        PruningPlan(prune_layer=layer)


@pytest.mark.parametrize("with_budget", [False, True])
def test_cut_past_the_backbone_is_rejected(backbone, geometry, uniform_budget, with_budget):
    budget = uniform_budget(geometry, 16) if with_budget else None
    with pytest.raises(ValueError, match="exceeds"):
        PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=100, budget=budget))


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


def test_size_biased_attention_requires_a_videomae_block(geometry, pruned):
    backbone = reference_backbone(geometry)
    backbone.blocks[-1] = torch.nn.Identity()
    with pytest.raises(ValueError, match="size-biased attention"):
        PrunedEncoder(backbone, geometry, pruned(prune_layer=2, size_biased_attention=True))


@pytest.mark.parametrize("prior", [torch.ones(2, 2), torch.tensor([float("nan")]),
                                   torch.tensor([-0.1])])
def test_invalid_prior_is_rejected_at_construction(prior):
    with pytest.raises(ValueError, match="spatial_prior"):
        PruningPlan(spatial_prior=prior)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_invalid_motion_weight_is_rejected_at_construction(weight):
    with pytest.raises(ValueError, match="motion_weight"):
        PruningPlan(motion_weight=weight)


def test_motion_alone_is_refused_because_the_first_slice_has_none():
    # The first slice has no earlier one: under motion alone all its tokens tie and would be
    # kept by grid position, whatever the clip shows.
    with pytest.raises(ValueError, match="motion_weight"):
        PruningPlan(motion_weight=1.0)
    assert PruningPlan(motion_weight=0.99).motion_weight == 0.99
