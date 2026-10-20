"""check_backbone is what makes integration safe.

An incompatible backbone usually runs and produces embeddings of the right shape,
surfacing only later as degraded accuracy. By then it is indistinguishable from the
effect of pruning, which is the thing being measured.
"""

import pytest
import torch
from torch import nn

from vitprune.core.compat import check_backbone
from vitprune.core.protocols import missing_attributes, present_optional
from vitprune.core.testing import reference_backbone, reference_geometry


@pytest.fixture
def geometry():
    return reference_geometry(depth=3)


@pytest.fixture
def backbone(geometry):
    return reference_backbone(geometry)


def test_a_conforming_backbone_passes(backbone, geometry):
    report = check_backbone(backbone, geometry, reference=backbone.forward_features)
    assert report.ok
    assert report.structure_ok
    assert report.missing == ()


def test_the_dense_path_matches_the_reference_exactly(backbone, geometry):
    report = check_backbone(backbone, geometry, reference=backbone.forward_features)
    assert report.dense_matches_reference
    assert report.max_difference == pytest.approx(0.0, abs=1e-6)


def test_the_token_grid_is_compared_against_the_geometry(backbone, geometry):
    report = check_backbone(backbone, geometry, reference=backbone.forward_features)
    assert report.tokens_produced == geometry.num_tokens


def test_a_mismatched_geometry_is_caught(backbone):
    wrong = reference_geometry(depth=3)
    object.__setattr__(wrong, "patch_size", 4)
    report = check_backbone(backbone, wrong, raise_on_failure=False)
    assert not report.ok
    assert "MISMATCH" in report.describe()


def test_a_backbone_without_blocks_is_caught():
    class OnlyForward(nn.Module):
        def forward(self, x):
            return x

    report = check_backbone(OnlyForward(), raise_on_failure=False)
    assert not report.ok
    assert "patch_embed" in report.describe()
    assert "blocks" in report.describe()


def test_failure_raises_by_default():
    class OnlyForward(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="not compatible"):
        check_backbone(OnlyForward())


def test_a_reference_that_disagrees_is_reported(backbone, geometry):
    """Stands in for a backbone that summarises through a class token: the shape is
    right, the values are not, and only the behaviour check notices."""

    def wrong_reference(clips):
        return backbone.forward_features(clips) + 1.0

    report = check_backbone(backbone, geometry, reference=wrong_reference, raise_on_failure=False)
    assert not report.ok
    assert "DIFFERS" in report.describe()


def test_without_a_reference_the_behaviour_check_is_skipped(backbone, geometry):
    report = check_backbone(backbone, geometry, raise_on_failure=False)
    assert not report.ok
    assert report.dense_matches_reference is None
    assert "not checked" in report.describe()


def test_the_probe_follows_the_backbone_dtype(geometry):
    """Feeding single-precision input to a double-precision model fails with a
    message that points nowhere near the cause."""
    backbone = reference_backbone(geometry).double()
    assert check_backbone(backbone, geometry, reference=backbone.forward_features).ok


def test_optional_attributes_in_use_are_reported(backbone, geometry):
    report = check_backbone(backbone, geometry, reference=backbone.forward_features)
    assert "pos_embed" in report.optional_present


def test_missing_attributes_are_listed():
    assert missing_attributes(object()) == ("patch_embed", "blocks")


def test_present_optional_ignores_absent_ones(backbone):
    present = present_optional(backbone)
    assert "pos_embed" in present
    assert "pos_drop" not in present


def test_a_reference_returning_a_tuple_is_accepted(backbone, geometry):
    """Some backbones return (features, logits) from their forward pass."""

    def tuple_reference(clips):
        return backbone.forward_features(clips), torch.zeros(1)

    assert check_backbone(backbone, geometry, reference=tuple_reference).ok


def test_a_reference_is_required_for_full_approval(backbone, geometry):
    with pytest.raises(ValueError, match="not checked"):
        check_backbone(backbone, geometry)


@pytest.mark.parametrize("failure", [False, True])
def test_check_restores_training_gradient_and_rng_state(backbone, geometry, failure):
    backbone.train()
    backbone.blocks[0].eval()
    next(backbone.parameters()).requires_grad_(False)
    modes = [module.training for module in backbone.modules()]
    gradients = [parameter.requires_grad for parameter in backbone.parameters()]
    rng = torch.random.get_rng_state().clone()

    def reference(clips):
        assert not any(module.training for module in backbone.modules())
        return backbone.forward_features(clips) + (1 if failure else 0)

    report = check_backbone(backbone, geometry, reference=reference, raise_on_failure=False)
    assert report.ok is not failure
    assert modes == [module.training for module in backbone.modules()]
    assert gradients == [parameter.requires_grad for parameter in backbone.parameters()]
    assert torch.equal(rng, torch.random.get_rng_state())


def test_reference_shapes_cannot_pass_via_broadcasting(backbone, geometry):
    report = check_backbone(
        backbone, geometry, reference=lambda clips: torch.zeros(geometry.embed_dim),
        raise_on_failure=False,
    )
    assert not report.ok
    assert "shaped" in report.describe()


def test_non_finite_reference_is_not_compatible(backbone, geometry):
    with pytest.raises(ValueError, match="DIFFERS"):
        check_backbone(
            backbone, geometry,
            reference=lambda clips: backbone.forward_features(clips) * float("nan"),
        )
