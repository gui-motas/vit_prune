import pytest
import torch

from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.merging import bipartite_merge, initial_sizes, merge_tokens
from vitprune.core.policy import FixedBudget
from vitprune.core.testing import reference_backbone, reference_geometry


def test_cut_prefers_same_region_and_falls_back_when_absent():
    tokens = torch.tensor([[[[1., 0.], [-1., 0.], [1., 0.]]]])
    scores = torch.tensor([[[3., 2., 1.]]])
    groups = torch.tensor([False, True, True])
    separated = merge_tokens(tokens, initial_sizes(tokens), scores, 2, groups)
    plain = merge_tokens(tokens, initial_sizes(tokens), scores, 2)
    assert separated.sizes.flatten().tolist() == [1, 2]
    assert plain.sizes.flatten().tolist() == [2, 1]
    fallback = merge_tokens(tokens, initial_sizes(tokens), scores, 1, groups)
    assert fallback.sizes.item() == 3


def test_bipartite_groups_follow_unmerged_a_then_b_and_forced_fallback():
    tokens = torch.tensor([[[2.], [4.], [8.], [16.]]])
    metric = torch.tensor([[[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]]])
    groups = torch.tensor([[False, True, True, False]])
    out, sizes, result_groups = bipartite_merge(tokens, torch.ones_like(tokens), metric, 3, groups)
    assert result_groups.tolist() == [[True, True, False]]
    assert sizes.flatten().tolist() == [1, 1, 2]
    assert out.flatten().tolist() == [8, 4, 9]
    _, sizes, result_groups = bipartite_merge(tokens, torch.ones_like(tokens), metric, 2,
                                               torch.tensor([[False, True, False, True]]))
    assert result_groups.tolist() == [[True, True]]
    assert sizes.sum().item() == 4


def test_bipartite_group_matching_stays_float32_under_autocast():
    tokens = torch.tensor([[[2.], [4.], [8.], [16.]]], dtype=torch.bfloat16)
    metric = torch.tensor([[[1., .001], [1., 0.], [0., 1.], [1., .001]]])
    groups = torch.tensor([[True, True, False, True]])
    expected = bipartite_merge(tokens, torch.ones_like(tokens), metric, 3, groups)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = bipartite_merge(tokens, torch.ones_like(tokens), metric, 3, groups)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert actual[1].dtype == torch.float32


def test_cut_group_matching_stays_float32_under_bfloat16_autocast():
    tokens = torch.tensor([[[[1., 0.], [1., .001], [1., .001]]]], dtype=torch.bfloat16)
    scores = torch.tensor([[[3., 2., 1.]]])
    groups = torch.ones(3, dtype=torch.bool)
    expected = merge_tokens(tokens, initial_sizes(tokens), scores, 2, groups)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = merge_tokens(tokens, initial_sizes(tokens), scores, 2, groups)
    assert torch.equal(actual.sizes, expected.sizes)
    assert actual.sizes.flatten().tolist() == [1, 2]


def test_separated_plan_validation_and_default_identity():
    with pytest.raises(ValueError, match='spatial_prior'):
        PruningPlan(separate_region=True)
    with pytest.raises(ValueError, match='separate_region'):
        PruningPlan(spatial_prior=torch.ones(16), separate_region=1)
    geometry = reference_geometry(depth=6)
    backbone = reference_backbone(geometry)
    prior = torch.tensor([0.] * 8 + [1.] * 8)
    options = dict(budget=FixedBudget(32), stages=((3, 24), (5, 12)), spatial_prior=prior)
    clips = torch.randn(*geometry.input_shape(batch=2))
    default = PrunedEncoder(backbone, geometry, PruningPlan(**options)).encode(clips)
    explicit = PrunedEncoder(backbone, geometry, PruningPlan(**options, separate_region=False))
    assert torch.equal(default, explicit.encode(clips))
    separated = PruningPlan(**options, separate_region=True)
    assert 'separate' in separated.describe()
    assert torch.isfinite(PrunedEncoder(backbone, geometry, separated).encode(clips)).all()


def test_invalid_group_shapes_are_rejected():
    x = torch.ones(1, 1, 4, 2)
    with pytest.raises(ValueError, match='groups'):
        merge_tokens(x, initial_sizes(x), torch.ones(1, 1, 4), 2, torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match='groups'):
        bipartite_merge(x[:, 0], torch.ones(1, 4, 1), x[:, 0], 2,
                        torch.ones(1, 4))


@pytest.mark.parametrize("initial", [32, 64])
def test_encoder_propagates_groups_in_kept_and_stage_output_order(monkeypatch, initial):
    import vitprune.core.encoder as module

    geometry = reference_geometry(depth=6)
    prior = torch.tensor([0.] * 8 + [1.] * 8)
    plan = PruningPlan(budget=FixedBudget(initial), stages=((3, initial * 3 // 4),),
                       spatial_prior=prior, separate_region=True)
    encoder = PrunedEncoder(reference_backbone(geometry), geometry, plan)
    original_cut, original_stage = module.merge_tokens, module.bipartite_merge
    tracked = []

    def cut(tokens, sizes, scores, keep, groups):
        indices = scores.argsort(-1, descending=True)[..., :keep].sort(-1).values
        tracked.append(groups[indices].reshape(tokens.shape[0], -1))
        return original_cut(tokens, sizes, scores, keep, groups)

    def stage(tokens, sizes, metric, keep, groups):
        if initial == geometry.num_tokens:
            tracked.append((prior > 0).repeat(geometry.temporal_slices)
                           .expand(tokens.shape[0], -1))
        assert torch.equal(groups, tracked[0])
        result = original_stage(tokens, sizes, metric, keep, groups)
        assert result[2].shape == (tokens.shape[0], keep)
        return result

    monkeypatch.setattr(module, "merge_tokens", cut)
    monkeypatch.setattr(module, "bipartite_merge", stage)
    assert torch.isfinite(encoder.encode(torch.randn(*geometry.input_shape(batch=2)))).all()


def test_separated_select_matches_encoding_merge_and_levels(monkeypatch):
    import vitprune.core.encoder as module

    geometry = reference_geometry(depth=4)
    prior = torch.tensor([0.] * 12 + [1.] * 4)
    encoder = PrunedEncoder(reference_backbone(geometry), geometry,
                            PruningPlan(budget=FixedBudget(16), spatial_prior=prior,
                                        separate_region=True))
    clips = torch.randn(*geometry.input_shape())
    original = module.merge_tokens
    seen = []

    def capture(*args):
        result = original(*args)
        seen.append(result.sizes)
        return result

    monkeypatch.setattr(module, "merge_tokens", capture)
    selection = encoder.select(clips)
    encoder.encode(clips)
    encoder.encode_levels(clips, [16], cells=(2, 2), top=2)
    assert torch.equal(selection.sizes, seen[1].squeeze(-1))
    assert torch.equal(seen[0], seen[1]) and torch.equal(seen[1], seen[2])
