"""Drift and latency are measurements, not derivable facts. These tests verify that
the measurements behave as expected -- not that the values are good, which only real
hardware can say.
"""

import numpy as np
import pytest
import torch

from vitprune.bench.cost import budget_cost
from vitprune.bench.drift import measure_drift
from vitprune.bench.latency import LatencyPoint, measure_plans
from vitprune.bench.report import (
    EmbeddingSummary,
    PruningSummary,
    RunReport,
    TimingSummary,
    summarise_pruning,
)
from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.policy import calibrate_policy
from vitprune.core.testing import reference_backbone, reference_geometry


@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def backbone(geometry):
    return reference_backbone(geometry)


@pytest.fixture
def clips(geometry):
    torch.manual_seed(3)
    return torch.randn(*geometry.input_shape(batch=6))


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def test_without_pruning_there_is_no_drift(backbone, geometry, clips):
    """The dense path compared against itself has to coincide."""
    report = measure_drift(backbone, clips, PruningPlan.dense(), geometry)
    assert report.cosine_mean == pytest.approx(1.0, abs=1e-6)
    assert report.relative_l2_mean == pytest.approx(0.0, abs=1e-6)


@pytest.fixture
def pruned(geometry, uniform_budget):
    """A plan keeping the same number of tokens in every clip, 16 of 64 by default."""
    return lambda tokens=16: PruningPlan(prune_layer=1, budget=uniform_budget(geometry, tokens))


def test_pruning_introduces_drift(backbone, geometry, clips, pruned):
    assert measure_drift(backbone, clips, pruned(), geometry).relative_l2_mean > 0.0


def test_pruning_more_drifts_more(backbone, geometry, clips, pruned):
    mild = measure_drift(backbone, clips, pruned(48), geometry)
    harsh = measure_drift(backbone, clips, pruned(8), geometry)
    assert harsh.relative_l2_mean > mild.relative_l2_mean


def test_the_report_counts_the_clips(backbone, geometry, clips):
    assert measure_drift(backbone, clips, PruningPlan.dense(), geometry).clips == 6


def test_batching_the_clips_changes_no_figure(backbone, geometry, clips, pruned):
    """Each clip's embedding depends on that clip alone, whatever else shares its batch."""
    one = measure_drift(backbone, clips, pruned(), geometry, batch=1)
    four = measure_drift(backbone, clips, pruned(), geometry, batch=4)
    for name, value in vars(one).items():
        assert getattr(four, name) == pytest.approx(value, abs=1e-5)


def test_the_worst_case_is_never_better_than_the_mean(backbone, geometry, clips, pruned):
    """The mean hides the bad cases, which is why the extremes are reported."""
    r = measure_drift(backbone, clips, pruned(), geometry)
    assert r.cosine_min <= r.cosine_p5 <= r.cosine_mean + 1e-9
    assert r.relative_l2_p95 >= r.relative_l2_mean - 1e-9


def test_measuring_with_no_clips_is_rejected(backbone, geometry):
    empty = torch.empty(0, 3, geometry.frames_per_clip, geometry.image_size, geometry.image_size)
    with pytest.raises(ValueError, match="at least one clip"):
        measure_drift(backbone, empty, PruningPlan.dense(), geometry)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def _measure(backbone, geometry, clips, plans, **options):
    settings = {"warmup": 1, "iterations": 2, "device": "cpu", "precision": "off"}
    settings.update(options)
    return measure_plans(backbone, clips, plans, geometry, progress=lambda _: None, **settings)


def test_one_point_per_plan(backbone, geometry, clips, pruned):
    points = _measure(backbone, geometry, clips, [PruningPlan.dense(), pruned(32), pruned(16)])
    assert [p.mean_tokens for p in points] == [64, 32, 16]


def test_the_dense_plan_is_the_reference(backbone, geometry, clips, pruned):
    dense, _ = _measure(backbone, geometry, clips, [PruningPlan.dense(), pruned()])
    assert dense.measured_speedup == pytest.approx(1.0, abs=1e-6)
    assert dense.predicted_speedup == 1.0
    assert dense.mean_tokens == geometry.num_tokens


def test_percentiles_bracket_the_typical_time(backbone, geometry, clips):
    point = _measure(backbone, geometry, clips, [PruningPlan.dense()])[0]
    assert 0.0 < point.p50_ms <= point.p95_ms


def test_realisation_relates_measured_to_predicted():
    point = LatencyPoint(
        plan="adaptive budget of 256/512/768 tokens after block 2",
        mean_tokens=512.0,
        ms_per_clip=10.0,
        p50_ms=9.0,
        p95_ms=12.0,
        clips_per_second=100.0,
        peak_memory_gb=1.0,
        predicted_speedup=4.0,
        measured_speedup=2.0,
    )
    assert point.realisation == pytest.approx(0.5)
    assert "50% realised" in point.describe()


def test_an_adaptive_plan_is_timed_on_real_clips(backbone, geometry, clips):
    probe = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1))
    plan = PruningPlan(
        prune_layer=1, budget=calibrate_policy(probe.statistics(clips), [16, 32, 48])
    )
    dense, adaptive = _measure(backbone, geometry, clips, [PruningPlan.dense(), plan])
    _, budget = PrunedEncoder(backbone, geometry, plan).encode_with_budget(clips)
    expected = budget_cost(geometry, budget.tokens.tolist(), 1)
    assert adaptive.mean_tokens == pytest.approx(budget.tokens.float().mean().item())
    assert adaptive.predicted_speedup == pytest.approx(expected.predicted_speedup, abs=1e-3)
    assert dense.measured_speedup == pytest.approx(1.0)


def test_plans_must_start_with_the_dense_baseline(backbone, geometry, clips, pruned):
    with pytest.raises(ValueError, match="dense baseline"):
        _measure(backbone, geometry, clips, [pruned()])


def test_drift_is_measured_for_an_adaptive_plan(backbone, geometry, clips):
    probe = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1))
    plan = PruningPlan(
        prune_layer=1, budget=calibrate_policy(probe.statistics(clips), [16, 32, 48])
    )
    assert measure_drift(backbone, clips, plan, geometry).relative_l2_mean > 0.0


def test_an_empty_list_of_plans_is_rejected(backbone, geometry, clips):
    with pytest.raises(ValueError, match="at least one"):
        _measure(backbone, geometry, clips, [])


@pytest.mark.parametrize("options", [{"iterations": 0}, {"warmup": -1}])
def test_invalid_timing_configuration_fails_before_running(backbone, geometry, clips, options):
    with pytest.raises(ValueError):
        _measure(backbone, geometry, clips, [PruningPlan.dense()], **options)


def test_an_unknown_precision_is_rejected(backbone, geometry, clips):
    with pytest.raises(ValueError, match="precision"):
        _measure(backbone, geometry, clips, [PruningPlan.dense()], precision="fp8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def test_repeated_embeddings_raise_a_warning():
    """Repeated vectors warrant inspection; norms alone do not prove a defect."""
    assert "WARNING" in "\n".join(EmbeddingSummary.of(np.tile(np.ones(8), (5, 1))).describe())


def test_distinct_embeddings_with_equal_norms_do_not_raise_a_warning():
    assert "WARNING" not in "\n".join(EmbeddingSummary.of(np.eye(8)).describe())


def test_non_finite_values_raise_a_warning():
    broken = np.ones((4, 8))
    broken[2, 3] = np.nan
    summary = EmbeddingSummary.of(broken)
    assert not summary.all_finite
    assert "non-finite" in "\n".join(summary.describe())


def test_healthy_embeddings_raise_no_warning():
    healthy = np.random.default_rng(1).normal(size=(10, 32))
    assert "WARNING" not in "\n".join(EmbeddingSummary.of(healthy).describe())


def test_repeated_clips_have_maximum_similarity():
    repeated = np.tile(np.arange(8.0), (5, 1))
    assert EmbeddingSummary.of(repeated).consecutive_similarity_mean == pytest.approx(1.0)


def test_a_single_clip_does_not_break_the_similarity():
    assert EmbeddingSummary.of(np.ones((1, 8))).consecutive_similarity_mean == 1.0


def test_the_pruning_summary_reports_the_reduction(backbone, geometry, clips, pruned):
    selection = PrunedEncoder(backbone, geometry, pruned()).select(clips[:2])
    summary = summarise_pruning(selection, geometry, top_rows=2)

    assert summary.tokens_before == geometry.num_tokens
    assert summary.tokens_after == 16
    assert summary.token_reduction == pytest.approx(4.0, rel=0.05)
    assert summary.mean_merge_size == pytest.approx(4.0, rel=0.01)


def test_budget_concentrated_on_the_top_is_flagged():
    summary = PruningSummary(
        tokens_before=2048,
        tokens_after=512,
        kept_per_slice=64,
        tokens_per_slice=256,
        mean_merge_size=4.0,
        largest_merge_size=40,
        top_rows_budget_share=0.60,
        top_rows_screen_share=0.25,
    )
    assert "favouring the top" in "\n".join(summary.describe())


def test_budget_avoiding_the_top_is_flagged():
    summary = PruningSummary(
        tokens_before=2048,
        tokens_after=512,
        kept_per_slice=64,
        tokens_per_slice=256,
        mean_merge_size=4.0,
        largest_merge_size=64,
        top_rows_budget_share=0.02,
        top_rows_screen_share=0.25,
    )
    assert "avoiding the top" in "\n".join(summary.describe())


def _report(pruning=None):
    return RunReport(
        video="/tmp/x.mp4",
        plan="adaptive budget of 256/512/768 tokens after block 2",
        geometry="patch 14px",
        embeddings=EmbeddingSummary.of(np.random.default_rng(2).normal(size=(6, 16))),
        timing=TimingSummary(read_seconds=1.0, encode_seconds=8.0, clips=6, predicted_speedup=3.9),
        pruning=pruning,
        output_files=["x.npz"],
    )


def test_without_a_baseline_realisation_is_unknown():
    timing = TimingSummary(read_seconds=1.0, encode_seconds=10.0, clips=5, predicted_speedup=4.0)
    assert timing.realisation is None
    assert "not measured" in "\n".join(timing.describe())


def test_the_report_states_that_it_does_not_recognise_actions():
    """Without that note, a reader would expect labels and intervals in the output."""
    assert "temporal model" in _report().render()


def test_the_report_omits_the_pruning_section_on_the_dense_path():
    assert "Pruning" not in _report().render()


def test_the_report_becomes_a_dictionary_with_derived_fields(backbone, geometry, clips, pruned):
    selection = PrunedEncoder(backbone, geometry, pruned()).select(clips[:2])
    data = _report(summarise_pruning(selection, geometry, 2)).to_dict()

    assert "seconds_per_clip" in data["timing"]
    assert "realisation" in data["timing"]
    assert "token_reduction" in data["pruning"]
