"""The quality-aware budget: a risk table calibrated on the drift of each level, read by
the features of a clip. Its errors are silent -- a wrong bin or a misread risk still
returns a legal budget -- so each rule of the decision is pinned here."""

import json
import math

import numpy as np
import pytest
import torch

from vitprune.core.compressibility import CompressibilityFeatures
from vitprune.core.policy import QualityAwareBudgetPolicy, calibrate_quality_aware
from vitprune.core.testing import reference_geometry

LEVELS = (16, 32, 48, 64)


def _calibration(n=120, seed=0, videos=6, drivers=3):
    """Drift grows with the probe error ``c`` and with fewer tokens: ``0.4 c`` at 16 tokens,
    ``0.27 c`` at 32 and ``0.13 c`` at 48, plus a little noise. Motion and redundancy are
    unrelated to it."""
    generator = torch.Generator().manual_seed(seed)
    motion = torch.rand(n, generator=generator) * 10 + 30
    redundancy = torch.rand(n, generator=generator) * 0.05 + 0.73
    probe = torch.rand(n, generator=generator)
    drift = probe[:, None] * torch.tensor([48.0, 32.0, 16.0]) / 48 * 0.4
    drift = drift + torch.rand(n, 3, generator=generator) * 0.01
    features = CompressibilityFeatures(motion, redundancy, torch.rand(n, generator=generator),
                                       probe)
    return (features, drift.double(), [f"v{i % videos}" for i in range(n)],
            [f"d{i % drivers}" for i in range(n)])


def _policy(**options):
    features, drift, videos, drivers = _calibration()
    options = {"alpha": 0.0, "bins": 4, "min_support": 10, "min_videos": 3} | options
    return calibrate_quality_aware(features, drift, videos, drivers, LEVELS, **options)


def _bins_by_probe(features, bins=4):
    """With ``alpha = 0`` the bins are equal slices of the clips sorted by probe error."""
    order = np.argsort(features.probe_p90.numpy())
    return np.array_split(order, bins)


def test_a_clip_easy_to_compress_gets_fewer_tokens_than_a_hard_one():
    features, *_ = _calibration()
    decision = _policy().choose_budget(features)
    easy, hard = int(features.probe_p90.argmin()), int(features.probe_p90.argmax())

    assert decision.tokens[easy] < decision.tokens[hard]
    assert decision.reasons[easy] == "ok"


def test_the_risk_is_a_high_quantile_of_the_drift_in_each_bin():
    features, drift, _, _ = _calibration()
    policy = _policy()
    for row, members in zip(policy.risk, _bins_by_probe(features), strict=True):
        expected = np.quantile(drift.numpy()[members], 0.9, axis=0, method="higher")
        assert np.allclose(row, expected)


def test_support_counts_clips_videos_and_drivers_per_bin():
    features, _, videos, drivers = _calibration()
    policy = _policy()
    members = _bins_by_probe(features)

    assert policy.support == tuple(len(m) for m in members)
    assert policy.support_videos == tuple(len({videos[i] for i in m}) for m in members)
    assert policy.support_drivers == tuple(len({drivers[i] for i in m}) for m in members)


def test_no_level_within_tolerance_means_the_dense_path():
    features, *_ = _calibration()
    decision = _policy(tolerance=1e-9).choose_budget(features)

    assert decision.tokens.tolist() == [64] * 120
    assert set(decision.reasons) == {"no_level_within_tolerance"}
    assert not decision.used_fallback.any()
    assert (decision.predicted_risk == 0).all()


def test_a_bin_with_too_few_videos_falls_back_to_dense():
    features, *_ = _calibration()
    decision = _policy(min_videos=7).choose_budget(features)

    assert set(decision.reasons) == {"low_support"}
    assert decision.used_fallback.all() and decision.tokens.tolist() == [64] * 120


def _one(features, clip, **changes):
    values = {name: value[clip : clip + 1].clone() for name, value in features._asdict().items()}
    for name, value in changes.items():
        values[name][0] = value
    return CompressibilityFeatures(**values)


def test_a_clip_outside_the_calibrated_range_falls_back_to_dense():
    features, *_ = _calibration()
    beyond = _one(features, 0, motion=float(features.motion.max()) + 1)
    decision = _policy().choose_budget(beyond)

    assert decision.reasons == ("out_of_range",)
    assert decision.used_fallback.item() and decision.tokens.item() == 64


def test_the_calibration_clips_are_inside_their_own_range():
    features, *_ = _calibration()
    assert "out_of_range" not in _policy().choose_budget(features).reasons


def test_a_non_finite_feature_falls_back_to_dense():
    features, *_ = _calibration()
    decision = _policy().choose_budget(_one(features, 0, motion=float("nan")))

    assert decision.reasons == ("non_finite",)
    assert decision.tokens.item() == 64 and math.isnan(decision.index.item())


def test_without_the_probe_the_probe_feature_is_ignored():
    features, drift, videos, drivers = _calibration()
    blind = features._replace(probe_p90=torch.full_like(features.probe_p90, float("nan")))
    policy = calibrate_quality_aware(blind, drift, videos, drivers, LEVELS, alpha=1.0, bins=4,
                                     min_support=10, min_videos=3)

    assert policy.probe_reference == () and len(policy.feature_range) == 2
    assert not policy.uses_probe
    assert "non_finite" not in policy.choose_budget(blind).reasons


def test_decisions_are_levels_and_do_not_depend_on_the_batch_order():
    features, *_ = _calibration()
    policy = _policy()
    order = torch.randperm(120, generator=torch.Generator().manual_seed(3))
    shuffled = CompressibilityFeatures(*(value[order] for value in features))

    together, reordered = policy.choose_budget(features), policy.choose_budget(shuffled)
    assert set(together.tokens.tolist()) <= set(LEVELS)
    assert torch.equal(together.tokens[order], reordered.tokens)
    assert [together.reasons[i] for i in order.tolist()] == list(reordered.reasons)


def test_the_policy_round_trips_and_refuses_an_edit():
    policy = _policy()
    data = json.loads(json.dumps(policy.to_dict()))
    assert QualityAwareBudgetPolicy.from_dict(data) == policy

    data["risk"][0][0] += 0.01
    with pytest.raises(ValueError, match="version"):
        QualityAwareBudgetPolicy.from_dict(data)


def test_another_policy_type_is_refused():
    data = _policy().to_dict()
    data["policy_type"] = "something_else"
    with pytest.raises(ValueError, match="policy type"):
        QualityAwareBudgetPolicy.from_dict(data)


def test_levels_must_end_at_the_dense_path():
    features, drift, videos, drivers = _calibration()
    policy = calibrate_quality_aware(features, drift[:, :2], videos, drivers, (16, 32, 48),
                                     alpha=0.0, bins=4, min_support=10, min_videos=3)
    with pytest.raises(ValueError, match="dense"):
        policy.validate_for(reference_geometry())


def test_calibration_needs_bins_that_separate():
    features, drift, videos, drivers = _calibration()
    flat = CompressibilityFeatures(*(torch.ones_like(value) for value in features))
    with pytest.raises(ValueError, match="vary too little"):
        calibrate_quality_aware(flat, drift, videos, drivers, LEVELS, alpha=0.0, bins=4)
