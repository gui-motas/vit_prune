"""The proxy study's measures: which clips a class can be read from, and what the pruned head
lost against it. A window across two actions, or a gain counted as a loss, would make any
signal look predictive."""

import math

import numpy as np
import pytest

from vitprune.pipeline.benchmark.proxy import clip_labels, functional_losses


def test_a_clip_takes_the_class_of_the_one_action_holding_its_window():
    actions = [(0.0, 10.0, 1), (8.0, 20.0, 2)]
    # inside the first only; across both; inside the second only; outside any action
    assert clip_labels([1.0, 7.0, 12.0, 25.0], 4.0, actions) == [1, None, 2, None]


def test_the_functional_loss_counts_only_what_the_pruned_head_lost():
    dense = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    pruned = np.array([[0.4, 0.6], [0.1, 0.9], [0.6, 0.4]])

    found = functional_losses(dense, pruned, [0, 1, 0])

    assert found["p_true"] == pytest.approx([0.4, 0.9, 0.6])
    assert found["delta_p"] == pytest.approx([0.3, 0.0, 0.0])  # the second improved
    assert found["delta_ce"] == pytest.approx([math.log(0.7 / 0.4), 0.0, 0.0])
    assert found["flip"].tolist() == [True, False, False]  # right in dense, wrong pruned
