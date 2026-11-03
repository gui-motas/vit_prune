"""The adaptive budget inside the encoder. The guarantee that matters: splitting a
batch by budget and putting it back together must not change any embedding."""

from dataclasses import replace

import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.policy import BudgetPolicy, calibrate_policy
from vitprune.core.testing import reference_backbone, reference_geometry

LEVELS = (16, 32, 48)


@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def backbone(geometry):
    return reference_backbone(geometry).double()


@pytest.fixture
def clips(geometry):
    torch.manual_seed(13)
    return torch.randn(*geometry.input_shape(batch=6), dtype=torch.float64)


@pytest.fixture
def policy(backbone, geometry, clips):
    probe = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1))
    return calibrate_policy(probe.statistics(clips), LEVELS)


def _encoder(backbone, geometry, policy, **plan):
    return PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1, budget=policy, **plan))


def test_clips_receive_different_budgets(backbone, geometry, policy, clips):
    _, budget = _encoder(backbone, geometry, policy).encode_with_budget(clips)
    assert set(budget.tokens.tolist()) <= set(LEVELS)
    assert len(set(budget.tokens.tolist())) > 1


def test_the_record_follows_the_policy(backbone, geometry, policy, clips):
    encoder = _encoder(backbone, geometry, policy)
    _, budget = encoder.encode_with_budget(clips)
    assert torch.equal(budget.tokens, policy.tokens(budget.statistics))
    assert torch.equal(budget.regimes, policy.regimes(budget.statistics))
    assert torch.allclose(budget.statistics.motion, encoder.statistics(clips).motion)


@pytest.mark.parametrize("biased", [False, True])
def test_grouping_the_batch_changes_no_embedding(backbone, geometry, policy, clips, biased):
    """Encoding the whole batch must equal encoding each clip on its own."""
    encoder = _encoder(backbone, geometry, policy, size_biased_attention=biased)
    together = encoder.encode(clips)
    alone = torch.cat([encoder.encode(clips[i : i + 1]) for i in range(clips.shape[0])])
    assert torch.allclose(together, alone, atol=1e-10)


def test_each_level_runs_as_one_sub_batch(backbone, geometry, policy, clips):
    encoder = _encoder(backbone, geometry, policy)
    _, budget = encoder.encode_with_budget(clips)
    levels, counts = budget.tokens.unique(return_counts=True)
    expected = sorted(
        (count, level, geometry.embed_dim)
        for level, count in zip(levels.tolist(), counts.tolist(), strict=True)
    )

    shapes = []
    hook = backbone.blocks[-1].register_forward_hook(
        lambda module, args, output: shapes.append(tuple(args[0].shape))
    )
    encoder.encode(clips)
    hook.remove()
    assert sorted(shapes) == expected


def test_a_budget_of_every_token_reproduces_the_dense_path(backbone, geometry, clips):
    everything = BudgetPolicy(
        redundancy_reference=[0.0, 1.0],
        motion_reference=[0.0, 1.0],
        thresholds=[0.0],
        levels=[geometry.num_tokens // 2, geometry.num_tokens],
    )
    adaptive = _encoder(backbone, geometry, everything).encode(clips)
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    assert torch.allclose(adaptive, dense, atol=1e-10)


def test_selection_reports_each_clip_budget(backbone, geometry, policy, clips):
    encoder = _encoder(backbone, geometry, policy)
    _, budget = encoder.encode_with_budget(clips)
    selection = encoder.select(clips)

    for clip in range(clips.shape[0]):
        kept = budget.tokens[clip].item() // geometry.temporal_slices
        assert selection.kept_in(clip) == kept
        assert selection.keep_mask(clip, 0).sum() == kept
        assert selection.size_grid(clip, 0).sum() == pytest.approx(geometry.tokens_per_slice)
    smallest, largest = (int(v) // geometry.temporal_slices for v in budget.tokens.aminmax())
    assert f"{smallest} to {largest}" in selection.describe()


def test_a_plan_without_a_budget_has_none_to_report(backbone, geometry, clips):
    with pytest.raises(ValueError, match="adaptive"):
        PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1)).encode_with_budget(clips)


def test_levels_that_do_not_fit_the_grid_are_rejected(backbone, geometry, policy):
    with pytest.raises(ValueError, match="at most"):
        _encoder(backbone, geometry, replace(policy, levels=(16, 32, 128)))


def test_the_budget_must_be_a_policy():
    with pytest.raises(ValueError, match="BudgetPolicy"):
        PruningPlan(budget={"levels": [16, 32]})


def test_the_plan_names_its_policy(policy):
    assert policy.version in PruningPlan(budget=policy).describe()


def test_a_fixed_budget_keeps_the_same_tokens_in_every_clip(backbone, geometry, clips):
    from vitprune.core.policy import FixedBudget

    _, budget = _encoder(backbone, geometry, FixedBudget(32)).encode_with_budget(clips)
    assert budget.tokens.tolist() == [32] * clips.shape[0]
    assert budget.regimes.tolist() == [0] * clips.shape[0]


def test_a_fixed_budget_of_every_token_is_the_dense_path_exactly(
    backbone, geometry, clips, monkeypatch
):
    """The full level is the dense path itself, not 2048 tokens selected and regrouped."""
    from vitprune.core import encoder as module
    from vitprune.core.policy import FixedBudget

    def never(*args, **kwargs):
        raise AssertionError("the full level must not merge")

    monkeypatch.setattr(module, "merge_tokens", never)
    full = _encoder(backbone, geometry, FixedBudget(geometry.num_tokens)).encode(clips)
    dense = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(clips)
    assert torch.equal(full, dense)


@pytest.mark.parametrize("tokens", [18, 0])
def test_a_fixed_budget_must_fit_the_grid(backbone, geometry, tokens):
    from vitprune.core.policy import FixedBudget

    with pytest.raises(ValueError):
        _encoder(backbone, geometry, FixedBudget(tokens))


def test_the_plan_names_a_fixed_budget():
    from vitprune.core.policy import FixedBudget

    assert "fixed budget of 32 tokens" in PruningPlan(budget=FixedBudget(32)).describe()
