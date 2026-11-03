"""The probe of compressibility: how well a handful of anchor tokens stand for a clip.

It feeds the budget, so its failures are silent: a probe that ignored a token, or mixed
clips of a batch, would still return plausible numbers.
"""

import pytest
import torch

from vitprune.core import compressibility
from vitprune.core.compressibility import compressibility_features, fixed_anchors, probe_errors
from vitprune.core.scoring import score_tokens
from vitprune.core.statistics import clip_statistics

# The reference geometry: 4 slices of a 4x4 grid. Four fixed anchors and two by score.
SMALL = {"cells": (2, 2), "top": 2}


def _tokens(batch=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, 4, 16, 8)


def test_the_fixed_anchors_sit_at_the_centres_of_a_two_by_four_grid():
    assert fixed_anchors(16).tolist() == [66, 70, 74, 78, 194, 198, 202, 206]
    assert fixed_anchors(4, (2, 2)).tolist() == [5, 7, 13, 15]


def test_the_fixed_anchors_need_a_grid_at_least_as_fine_as_the_cells():
    with pytest.raises(ValueError):
        fixed_anchors(1)


def test_the_probe_is_deterministic_and_gives_one_value_per_clip():
    tokens = _tokens(batch=3)
    scores = score_tokens(tokens)
    first = compressibility_features(tokens, scores, **SMALL)
    second = compressibility_features(tokens, scores, **SMALL)

    for name, value in first._asdict().items():
        assert value.shape == (3,), name
        assert torch.isfinite(value).all(), name
        assert torch.equal(value, getattr(second, name)), name


def test_repeated_tokens_have_no_error():
    tokens = torch.ones(1, 4, 16, 8)
    found = compressibility_features(tokens, score_tokens(tokens), **SMALL)
    assert found.probe_mean.item() == pytest.approx(0.0, abs=1e-6)
    assert found.probe_p90.item() == pytest.approx(0.0, abs=1e-6)


def _two_directions():
    """Even positions along e1, odd ones along e2. The fixed anchors (5, 7, 13, 15) are
    odd; positions 0 and 2 score highest, so the anchors cover both directions."""
    tokens = torch.zeros(1, 4, 16, 8)
    tokens[..., 0::2, 0] = 1.0
    tokens[..., 1::2, 1] = 1.0
    scores = torch.ones(1, 4, 16)
    scores[..., [0, 2]] = 100.0
    return tokens, scores


def test_a_discrepant_token_raises_the_error_of_its_clip():
    tokens, scores = _two_directions()
    before = compressibility_features(tokens, scores, **SMALL).probe_mean
    tokens[0, 1, 3] = 0.0
    tokens[0, 1, 3, 2] = 1.0  # along e3, which no anchor shares
    scores[0, 1, 3] = 0.0

    assert probe_errors(tokens, scores, **SMALL)[0, 1, 3] >= 0.99
    assert compressibility_features(tokens, scores, **SMALL).probe_mean > before


def test_zero_tokens_count_as_the_largest_error():
    tokens = torch.zeros(1, 4, 16, 8)
    found = compressibility_features(tokens, torch.ones(1, 4, 16), **SMALL)
    assert found.probe_mean.item() == pytest.approx(1.0)
    assert all(torch.isfinite(value).all() for value in found)


def test_non_finite_tokens_give_non_finite_features():
    tokens = _tokens(batch=1)
    tokens[0, 2, 9] = float("nan")
    found = compressibility_features(tokens, torch.ones(1, 4, 16), **SMALL)
    assert torch.isnan(found.probe_p90).all()


def test_each_clip_is_probed_on_its_own():
    tokens = _tokens(batch=3)
    scores = score_tokens(tokens)
    together = compressibility_features(tokens, scores, **SMALL)
    for clip in range(3):
        alone = compressibility_features(tokens[clip : clip + 1], scores[clip : clip + 1], **SMALL)
        for name, value in alone._asdict().items():
            assert torch.allclose(value, getattr(together, name)[clip : clip + 1], atol=1e-6)


def test_without_the_probe_only_the_statistics_are_measured(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("the probe must not run")

    monkeypatch.setattr(compressibility, "probe_errors", never)
    tokens = _tokens()
    found = compressibility_features(tokens, score_tokens(tokens), probe=False, **SMALL)
    statistics = clip_statistics(tokens)

    assert torch.isnan(found.probe_mean).all() and torch.isnan(found.probe_p90).all()
    assert torch.equal(found.motion, statistics.motion)
    assert torch.equal(found.redundancy, statistics.redundancy)
