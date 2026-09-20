import pytest

from vitprune.core.geometry import ClipGeometry


def test_giant_defaults_match_the_published_grid():
    g = ClipGeometry.videomae_giant()
    assert g.grid_size == 16
    assert g.tokens_per_slice == 256
    assert g.temporal_slices == 8
    assert g.num_tokens == 2048
    assert g.head_dim == 88


def test_large_has_a_coarser_grid_than_giant():
    assert ClipGeometry.videomae_large().num_tokens < ClipGeometry.videomae_giant().num_tokens


def test_image_geometry_has_a_single_temporal_slice():
    g = ClipGeometry.image_vit(224, 16, 768, 12, 12)
    assert g.temporal_slices == 1
    assert not g.has_temporal_axis


def test_video_geometry_reports_a_temporal_axis():
    assert ClipGeometry.videomae_giant().has_temporal_axis


def test_quarter_retention_keeps_sixty_four_per_slice():
    g = ClipGeometry.videomae_giant()
    assert g.kept_per_slice(0.25) == 64
    assert g.kept_tokens(0.25) == 512


def test_full_retention_keeps_everything():
    g = ClipGeometry.videomae_giant()
    assert g.kept_tokens(1.0) == g.num_tokens


def test_tiny_retention_still_keeps_one_token_per_slice():
    assert ClipGeometry.videomae_giant().kept_per_slice(0.0001) == 1


@pytest.mark.parametrize("retention", [0.0, -0.1, 1.5])
def test_retention_outside_the_range_is_rejected(retention):
    with pytest.raises(ValueError, match="retention"):
        ClipGeometry.videomae_giant().kept_per_slice(retention)


def test_input_shape_follows_the_requested_batch():
    assert ClipGeometry.videomae_giant().input_shape(batch=4) == (4, 3, 16, 224, 224)


def test_image_size_not_divisible_by_patch_size_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ClipGeometry(image_size=225, patch_size=14)


def test_frames_not_divisible_by_tubelet_are_rejected():
    with pytest.raises(ValueError, match="tubelet_size"):
        ClipGeometry(frames_per_clip=15, tubelet_size=2)


def test_embed_dim_not_divisible_by_heads_is_rejected():
    with pytest.raises(ValueError, match="heads"):
        ClipGeometry(embed_dim=1000, heads=16)


def test_geometry_is_immutable():
    """Mutating it after construction would leave the derived counts out of step
    with the loaded checkpoint."""
    g = ClipGeometry.videomae_giant()
    with pytest.raises(AttributeError):
        g.image_size = 112


def test_clip_length_can_be_overridden_on_the_named_constructor():
    assert ClipGeometry.videomae_giant(frames_per_clip=32).temporal_slices == 16
