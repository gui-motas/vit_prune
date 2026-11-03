"""Turning the levels of each clip into records, and the statistics the validation of a
policy rests on. A wrong rank or a permutation that crosses groups would make a weak
signal look strong."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from vitprune.bench.quality import auc, clip_records, permutation_p, spearman, time_probe
from vitprune.core.encoder import PrunedEncoder, PruningPlan
from vitprune.core.testing import reference_backbone, reference_geometry


@pytest.fixture
def encoding():
    geometry = reference_geometry(depth=3)
    backbone = reference_backbone(geometry).double()
    torch.manual_seed(13)
    clips = torch.randn(*geometry.input_shape(batch=3), dtype=torch.float64)
    encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=1))
    return encoder.encode_levels(clips, (16, 32, 48), depths=(1, 3), cells=(2, 2), top=2)


def test_records_hold_the_drift_of_every_level_against_the_dense(encoding):
    head = nn.Linear(32, 3).double()
    with_head, without = clip_records(encoding, head), clip_records(encoding)

    assert len(with_head) == 3
    record = with_head[0]
    assert set(record["levels"]) == {"16", "32", "48"}
    assert set(record["dense"]["stages"]) == {"1", "3"}
    for level in record["levels"].values():
        assert level["d_cos"] >= -1e-12 and level["d_rel"] >= 0
        assert isinstance(level["top1"], bool)
        assert set(level["stages"]) == {"1", "3"}
        assert set(level["merge_error"]) == {"mean", "p90", "max"}
    assert without[0]["levels"]["16"]["top1"] is None


def test_spearman_ranks_monotone_relations_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_permutation_p_is_small_for_a_strong_relation_and_large_for_noise():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    assert permutation_p(x, x + 0.05 * rng.random(40), permutations=2000) < 0.01
    assert permutation_p(x, rng.random(40), permutations=2000) > 0.2


def test_permutation_within_groups_ignores_differences_between_groups():
    """Two videos, one with more drift and more probe error than the other, and no
    relation inside either: the pooled correlation is an artefact of the video."""
    rng = np.random.default_rng(1)
    groups = ["a"] * 20 + ["b"] * 20
    x = np.r_[rng.random(20), 10 + rng.random(20)]
    y = np.r_[rng.random(20), 10 + rng.random(20)]
    assert spearman(x, y) > 0.5
    assert permutation_p(x, y, groups, permutations=2000) > 0.2


def test_auc_of_a_perfect_ranking_is_one_and_of_its_reverse_zero():
    assert auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert math.isnan(auc([0.1, 0.2], [True, True]))


def test_the_probe_is_timed_per_clip():
    found = time_probe(reference_geometry(), batch=2, device="cpu", warmup=0, iterations=2,
                       cells=(2, 2), top=2)
    assert found["probe_ms_per_clip"] > 0 and found["statistics_ms_per_clip"] > 0
    assert found["batch"] == 2 and found["device"] == "cpu"


def test_partial_spearman_removes_what_the_control_explains():
    from vitprune.bench.quality import partial_spearman

    rng = np.random.default_rng(2)
    control = rng.random(200)
    x = control + 0.01 * rng.random(200)
    y = control + 0.01 * rng.random(200)  # both follow the control, not each other
    assert spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, control)) < 0.3
    extra = rng.random(200)
    assert partial_spearman(control + extra, control + extra, control) > 0.9


def test_partial_spearman_takes_out_every_control():
    from vitprune.bench.quality import partial_spearman

    rng = np.random.default_rng(3)
    first, second = rng.random(200), rng.random(200)
    for shared, other in ((first, second), (second, first)):
        x = shared + 0.01 * rng.random(200)
        y = shared + 0.01 * rng.random(200)
        assert partial_spearman(x, y, other) > 0.9  # the other control explains nothing
        assert abs(partial_spearman(x, y, first, second)) < 0.3
