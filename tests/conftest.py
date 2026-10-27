"""Shared test helpers."""

import pytest

from vitprune.core.policy import BudgetPolicy


@pytest.fixture
def uniform_budget():
    """Build a budget giving every clip the same token count.

    Test-only: a calibrated policy decides counts from the content, which makes exact
    counts awkward to assert. With the threshold at zero every clip lands in the top
    level, so tests about merging, selection and shapes get a predictable cut.
    """

    def build(geometry, tokens: int) -> BudgetPolicy:
        return BudgetPolicy(
            redundancy_reference=[0.0, 1.0],
            motion_reference=[0.0, 1.0],
            thresholds=[0.0],
            levels=[geometry.temporal_slices, tokens],
        )

    return build
