"""Cost formulas need no GPU or weights. The package still requires torch through
its base imports; these tests do not directly execute torch operations.
"""

import pytest

from vitprune.bench.cost import budget_cost, clip_cost, corpus_budget, parameter_count, roofline
from vitprune.bench.hardware import CATALOG, accelerator, custom
from vitprune.core.geometry import ClipGeometry


@pytest.fixture
def geometry():
    return ClipGeometry.videomae_giant()


def test_the_parameter_count_matches_the_published_model_size(geometry):
    """The Giant holds roughly a billion weights."""
    assert 0.95e9 < parameter_count(geometry) < 1.10e9


def test_without_pruning_both_costs_coincide(geometry):
    cost = clip_cost(geometry, retention=1.0, prune_layer=0)
    assert cost.gmacs_dense == cost.gmacs_pruned
    assert cost.predicted_speedup == 1.0


def test_quarter_retention_predicts_close_to_fourfold(geometry):
    assert 3.5 < clip_cost(geometry, 0.25, 2).predicted_speedup < 4.5


def test_pruning_more_predicts_more(geometry):
    assert (
        clip_cost(geometry, 0.15, 2).predicted_speedup
        > clip_cost(geometry, 0.50, 2).predicted_speedup
    )


def test_pruning_earlier_predicts_more(geometry):
    assert (
        clip_cost(geometry, 0.25, 2).predicted_speedup
        > clip_cost(geometry, 0.25, 20).predicted_speedup
    )


def test_pruning_at_the_last_block_barely_helps(geometry):
    assert clip_cost(geometry, 0.25, geometry.depth).predicted_speedup < 1.05


def test_the_token_counts_appear_in_the_result(geometry):
    cost = clip_cost(geometry, retention=0.25, prune_layer=2)
    assert cost.tokens_dense == 2048
    assert cost.tokens_pruned == 512


def test_the_weight_reading_floor_does_not_move_with_pruning(geometry):
    """Reading the weights costs the same whether tokens are pruned or not. That
    floor is what caps the achievable speedup."""
    card = accelerator("a40")
    dense = roofline(geometry, clip_cost(geometry, 1.0, 0), card)
    pruned = roofline(geometry, clip_cost(geometry, 0.15, 2), card)
    assert dense.weights_ms == pytest.approx(pruned.weights_ms)
    assert pruned.compute_ms < dense.compute_ms


def test_the_margin_above_the_floor_shrinks_with_pruning(geometry):
    card = accelerator("a40")
    dense = roofline(geometry, clip_cost(geometry, 1.0, 0), card)
    pruned = roofline(geometry, clip_cost(geometry, 0.15, 2), card)
    assert pruned.ratio < dense.ratio


def test_more_bandwidth_pushes_the_floor_further_away(geometry):
    """On a card with faster memory, pruning should convert better."""
    cost = clip_cost(geometry, 0.25, 2)
    assert (
        roofline(geometry, cost, accelerator("h100-sxm")).weights_ms
        < roofline(geometry, cost, accelerator("a40")).weights_ms
    )


def test_doubling_the_stride_halves_the_budget(geometry):
    cost = clip_cost(geometry, 0.25, 2)
    card = accelerator("h100-sxm")
    short = corpus_budget(7.2, cost, card, stride=8)
    long = corpus_budget(7.2, cost, card, stride=16)
    assert short.gpu_hours == pytest.approx(long.gpu_hours * 2, rel=0.01)


def test_pruning_cuts_the_budget_in_proportion_to_the_speedup(geometry):
    card = accelerator("h100-sxm")
    cost = clip_cost(geometry, 0.25, 2)
    dense = corpus_budget(7.2, clip_cost(geometry, 1.0, 0), card)
    pruned = corpus_budget(7.2, cost, card)
    assert dense.gpu_hours / pruned.gpu_hours == pytest.approx(cost.predicted_speedup, rel=0.01)


def test_an_empty_corpus_costs_nothing(geometry):
    assert corpus_budget(0.0, clip_cost(geometry, 1.0, 0), accelerator("a40")).gpu_hours == 0.0


def test_a_negative_duration_is_rejected(geometry):
    with pytest.raises(ValueError, match="negative"):
        corpus_budget(-1.0, clip_cost(geometry, 1.0, 0), accelerator("a40"))


def test_an_unknown_accelerator_lists_the_known_ones():
    with pytest.raises(KeyError, match="Known"):
        accelerator("no-such-card")


def test_a_card_can_be_described_without_touching_the_code():
    mine = custom("test", peak_tflops=500, bandwidth_gbs=2000, efficiency=0.6)
    assert mine.usable_flops == pytest.approx(500e12 * 0.6)


@pytest.mark.parametrize("peak,bandwidth,efficiency", [(0, 100, 0.5), (100, 0, 0.5), (100, 100, 0)])
def test_invalid_card_parameters_are_rejected(peak, bandwidth, efficiency):
    with pytest.raises(ValueError):
        custom("bad", peak, bandwidth, efficiency)


def test_every_catalogued_card_describes_itself():
    for name in CATALOG:
        assert "TFLOP/s" in accelerator(name).describe()


@pytest.mark.parametrize("retention,layer", [(1.5, 0), (float("nan"), 0), (0.25, 41),
                                           (1.0, -1), (1.0, 41)])
def test_invalid_plans_are_not_reported_as_dense_costs(geometry, retention, layer):
    with pytest.raises(ValueError):
        clip_cost(geometry, retention, layer)


# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------
def test_a_budget_of_every_token_costs_the_dense_path(geometry):
    cost = budget_cost(geometry, [geometry.num_tokens] * 3, 2)
    assert cost.predicted_speedup == pytest.approx(1.0)


def test_the_mean_of_the_costs_exceeds_the_cost_of_the_mean(geometry):
    """Attention is quadratic: 256 and 768 tokens cost more than 512 and 512."""
    spread = budget_cost(geometry, [256, 768], 2)
    even = budget_cost(geometry, [512, 512], 2)
    assert spread.tokens_pruned == even.tokens_pruned == 512
    assert spread.gmacs_pruned > even.gmacs_pruned


def test_the_expected_cost_averages_the_clips(geometry):
    low, high = budget_cost(geometry, [256], 2), budget_cost(geometry, [768], 2)
    mixed = budget_cost(geometry, [256, 768, 768], 2)
    assert mixed.gmacs_pruned == pytest.approx((low.gmacs_pruned + 2 * high.gmacs_pruned) / 3)


def test_fewer_tokens_cost_less(geometry):
    fewer, more = (budget_cost(geometry, [count], 2).gmacs_pruned for count in (256, 768))
    assert fewer < more


@pytest.mark.parametrize("tokens", [[], [0], [4096], [512.0], [True]])
def test_invalid_token_counts_are_rejected(geometry, tokens):
    with pytest.raises(ValueError):
        budget_cost(geometry, tokens, 2)
