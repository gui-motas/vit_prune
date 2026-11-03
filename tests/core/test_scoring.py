import pytest
import torch

from vitprune.core.priors import box_prior, top_rows_prior
from vitprune.core.scoring import score_tokens


def _tokens(batch=2, slices=4, count=16, dim=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, slices, count, dim)


def test_one_score_per_token():
    t = _tokens()
    assert score_tokens(t).shape == t.shape[:3]


def test_scores_are_never_negative():
    assert score_tokens(_tokens()).min() >= 0.0


def test_weight_zero_ignores_motion_entirely():
    """With weight 0, reversing the temporal order leaves the scores unchanged
    once they are reversed back."""
    t = _tokens(slices=3)
    reversed_order = t.flip(dims=[1])

    forward = score_tokens(t, motion_weight=0.0)
    backward = score_tokens(reversed_order, motion_weight=0.0)

    assert torch.allclose(forward.flip(dims=[1]), backward, atol=1e-5)


def test_weight_one_scores_a_static_token_lowest():
    t = _tokens(batch=1, slices=3, count=6)
    t[:, :, 0, :] = 1.0  # position 0 is frozen across slices

    scores = score_tokens(t, motion_weight=1.0)[0, 1]
    assert scores[0] == scores.min()


def test_the_first_slice_has_nothing_to_compare_against():
    """With no earlier slice every motion value is zero, so the scores tie."""
    t = _tokens(batch=1, slices=3)
    scores = score_tokens(t, motion_weight=1.0)[0, 0]
    assert torch.allclose(scores, scores[0].expand_as(scores))


def test_a_single_temporal_slice_does_not_break():
    """The image geometry case: one slice, no motion term available."""
    t = _tokens(batch=1, slices=1, count=10)
    assert score_tokens(t, motion_weight=0.0).shape == (1, 1, 10)


def test_uniqueness_favours_the_odd_token_out():
    t = torch.zeros(1, 1, 8, 4)
    t[0, 0, :, 0] = 1.0
    t[0, 0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # the only different one

    scores = score_tokens(t, motion_weight=0.0)[0, 0]
    assert scores.argmax().item() == 3


def test_zero_weight_positions_rank_last_in_their_own_order():
    """A zero weight sends the position behind every weighted one; among themselves
    the positions keep the order of their own score instead of tying at zero."""
    t = _tokens(batch=1, slices=2, count=16)
    prior = top_rows_prior(grid_size=4, rows=1, weight=0.0)

    without = score_tokens(t, motion_weight=0.0)
    with_prior = score_tokens(t, motion_weight=0.0, spatial_prior=prior)

    assert with_prior[..., :4].max() < with_prior[..., 4:].min()
    assert torch.equal(with_prior[..., :4].argsort(dim=-1), without[..., :4].argsort(dim=-1))
    assert torch.allclose(with_prior[..., 4:], without[..., 4:])


def test_a_neutral_prior_changes_nothing():
    t = _tokens(batch=1, slices=2, count=16)
    assert torch.allclose(score_tokens(t), score_tokens(t, spatial_prior=torch.ones(16)))


def test_a_prior_of_the_wrong_size_is_rejected():
    with pytest.raises(ValueError, match="positions"):
        score_tokens(_tokens(count=16), spatial_prior=torch.ones(9))


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_a_weight_outside_the_range_is_rejected(weight):
    with pytest.raises(ValueError, match="motion_weight"):
        score_tokens(_tokens(), motion_weight=weight)


def test_a_wrong_input_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        score_tokens(torch.randn(4, 16, 8))


def test_top_rows_prior_lowers_only_the_top():
    prior = top_rows_prior(grid_size=4, rows=2, weight=0.25)
    assert prior.shape == (16,)
    assert torch.all(prior[:8] == 0.25)
    assert torch.all(prior[8:] == 1.0)


@pytest.mark.parametrize("rows,weight", [(-1, 0.5), (5, 0.5), (1, -0.1), (1, 2.0)])
def test_invalid_prior_parameters_are_rejected(rows, weight):
    with pytest.raises(ValueError):
        top_rows_prior(grid_size=4, rows=rows, weight=weight)


def test_box_prior_marks_every_cell_a_box_touches():
    """A box grazing a cell takes all of it; a box ending on a boundary does not
    spill into the next cell."""
    prior = box_prior(grid_size=4, boxes=[(0.30, 0.0, 0.50, 0.26)]).reshape(4, 4)
    assert prior.tolist() == [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_box_prior_joins_the_boxes():
    prior = box_prior(grid_size=4, boxes=[(0.0, 0.0, 0.25, 0.25), (0.75, 0.75, 1.0, 1.0)])
    assert prior.nonzero().flatten().tolist() == [0, 15]


def test_box_prior_without_boxes_marks_nothing():
    assert torch.all(box_prior(grid_size=4, boxes=[]) == 0.0)


@pytest.mark.parametrize("box", [(0.5, 0.0, 0.5, 1.0), (0.0, 0.0, 1.2, 1.0), (-0.1, 0.0, 1.0, 1.0)])
def test_box_prior_rejects_boxes_outside_the_frame_or_empty(box):
    with pytest.raises(ValueError, match="box"):
        box_prior(grid_size=4, boxes=[box])


def test_the_prior_applied_afterwards_gives_the_same_scores():
    from vitprune.core.scoring import apply_prior

    t = _tokens()
    prior = torch.linspace(0.0, 1.0, t.shape[2])  # position 0 has weight zero
    assert torch.equal(apply_prior(score_tokens(t, 0.5), prior), score_tokens(t, 0.5, prior))
