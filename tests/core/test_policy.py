"""The budget policy turns clip statistics into a token count. It has to be monotonic,
bounded to its levels, and identifiable once written to disk."""

import pytest
import torch

from vitprune.core.geometry import ClipGeometry
from vitprune.core.policy import BudgetPolicy
from vitprune.core.statistics import ClipStatistics


def _policy(**changes):
    settings = {
        "redundancy_reference": [0.1, 0.2, 0.3, 0.4, 0.5],
        "motion_reference": [1.0, 2.0, 3.0, 4.0, 5.0],
        "thresholds": [1 / 3, 2 / 3],
        "levels": [256, 512, 768],
        "motion_balance": 0.5,
    }
    settings.update(changes)
    return BudgetPolicy(**settings)


def _stats(redundancy, motion):
    return ClipStatistics(torch.tensor(redundancy), torch.tensor(motion))


def test_more_motion_asks_for_more_tokens():
    g = _policy().score(_stats([0.3, 0.3, 0.3], [0.5, 3.0, 9.0]))
    assert g[0] < g[1] < g[2]


def test_more_redundancy_asks_for_fewer_tokens():
    g = _policy().score(_stats([0.0, 0.3, 0.9], [3.0, 3.0, 3.0]))
    assert g[0] > g[1] > g[2]


def test_the_score_stays_between_zero_and_one():
    g = _policy().score(_stats([-5.0, 0.3, 5.0], [-1.0, 3.0, 99.0]))
    assert g.min() >= 0.0 and g.max() <= 1.0


@pytest.mark.parametrize("balance,varies", [(0.0, "motion"), (1.0, "redundancy")])
def test_the_balance_can_silence_either_statistic(balance, varies):
    """β = 0 uses redundancy alone, β = 1 motion alone."""
    stats = (
        _stats([0.3, 0.3], [0.5, 9.0]) if varies == "motion" else _stats([0.0, 0.9], [3.0, 3.0])
    )
    g = _policy(motion_balance=balance).score(stats)
    assert g[0] == g[1]


def test_a_value_tied_with_every_reference_ranks_in_the_middle():
    """A statistic that never varies, such as motion in a single-slice grid, has no say."""
    policy = _policy(motion_reference=[0.0, 0.0, 0.0], motion_balance=1.0)
    assert policy.score(_stats([0.3], [0.0])).item() == pytest.approx(0.5)


def test_thresholds_cut_the_score_into_regimes():
    policy = _policy()
    stats = _stats([0.5, 0.3, 0.1], [1.0, 3.0, 5.0])
    assert policy.regimes(stats).tolist() == [0, 1, 2]
    assert policy.tokens(stats).tolist() == [256, 512, 768]


def test_tokens_only_take_the_declared_levels():
    torch.manual_seed(0)
    stats = ClipStatistics(torch.rand(200), torch.rand(200) * 6)
    assert set(_policy().tokens(stats).tolist()) <= {256, 512, 768}


def test_non_finite_statistics_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        _policy().score(_stats([float("nan")], [1.0]))


def test_levels_must_fit_the_grid():
    giant = ClipGeometry.videomae_giant()
    _policy().validate_for(giant)
    with pytest.raises(ValueError, match="multiple"):
        _policy(levels=[256, 500, 768]).validate_for(giant)
    with pytest.raises(ValueError, match="at most"):
        _policy(levels=[256, 512, 4096]).validate_for(giant)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"redundancy_reference": []}, "non-empty"),
        ({"motion_reference": [1.0, float("nan")]}, "finite"),
        ({"motion_reference": [2.0, 1.0]}, "ascending"),
        ({"levels": [512], "thresholds": []}, "two levels"),
        ({"levels": [256, 256, 768]}, "strictly ascending"),
        ({"levels": [0, 512, 768]}, "positive"),
        ({"levels": [256, 512.0, 768]}, "integers"),
        ({"thresholds": [0.5]}, "one threshold fewer"),
        ({"thresholds": [0.6, 0.4]}, "strictly ascending"),
        ({"thresholds": [0.2, 1.5]}, r"\[0, 1\]"),
        ({"motion_balance": 1.5}, "motion_balance"),
        ({"motion_balance": True}, "motion_balance"),
    ],
)
def test_invalid_policies_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        _policy(**changes)


def test_a_policy_survives_a_round_trip_through_a_dictionary():
    policy = _policy()
    restored = BudgetPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.version == policy.version


def test_the_version_follows_the_content():
    assert _policy().version == _policy().version
    assert _policy().version != _policy(levels=[128, 512, 768]).version


def test_an_edited_policy_no_longer_matches_its_version():
    data = _policy().to_dict()
    data["levels"] = [128, 512, 768]
    with pytest.raises(ValueError, match="version"):
        BudgetPolicy.from_dict(data)
