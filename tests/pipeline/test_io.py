import numpy as np
import pytest
import torch
from torch import nn

from vitprune.core.geometry import ClipGeometry
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline.checkpoint import (
    LoadReport,
    disable_activation_checkpointing,
    install_weights,
    read_weights,
)
from vitprune.pipeline.video import clip_span, count_clips, find_videos, to_clips


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)


def test_wrapped_weights_are_unwrapped(tmp_path):
    model = Small()
    path = tmp_path / "weights.pth"
    torch.save({"module": model.state_dict(), "optimizer": {"junk": 1}}, path)

    assert set(read_weights(path)) == set(model.state_dict())


def test_name_prefixes_are_stripped(tmp_path):
    model = Small()
    path = tmp_path / "weights.pth"
    torch.save({f"module.{k}": v for k, v in model.state_dict().items()}, path)

    assert set(read_weights(path)) == set(model.state_dict())


def test_a_file_without_recognisable_weights_is_rejected(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError, match="weight dictionary"):
        read_weights(path)


def test_a_clean_install_reports_nothing_amiss():
    model = Small()
    report = install_weights(model, model.state_dict())
    assert report.clean
    assert "nothing missing" in report.describe()


def test_a_missing_layer_appears_in_the_report():
    model = Small()
    partial = {k: v for k, v in model.state_dict().items() if not k.startswith("b.")}
    report = install_weights(model, partial)
    assert not report.clean
    assert "missing" in report.describe()


def test_a_wrong_critical_shape_fails_before_installing():
    """A partial load would produce a model that looks functional and is wrong."""
    model = reference_backbone(reference_geometry(depth=1))
    weights = dict(model.state_dict())
    weights["patch_embed.proj.weight"] = torch.randn(3, 3, 3, 3, 3)

    with pytest.raises(ValueError, match="shape mismatch at patch_embed"):
        install_weights(model, weights)


def test_the_report_mentions_unused_keys():
    assert "unused" in LoadReport(missing=(), unexpected=("x",)).describe()


def test_activation_checkpointing_is_turned_off():
    """Recomputation only makes sense while training; at inference it runs every
    block twice and would distort timing."""

    class WithRecompute(nn.Module):
        def __init__(self):
            super().__init__()
            self.with_cp = True
            self.blocks = nn.ModuleList([nn.Identity()])
            self.blocks[0].with_cp = True

    model = disable_activation_checkpointing(WithRecompute())
    assert model.with_cp is False
    assert model.blocks[0].with_cp is False


def test_disabling_recomputation_tolerates_a_model_without_it():
    assert disable_activation_checkpointing(Small()) is not None


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def _frames(total: int, side: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(total, side, side, 3), dtype=np.uint8)


def test_clips_come_out_in_the_shape_the_backbone_expects():
    g = reference_geometry()
    clips = to_clips(_frames(64), g, stride=8)
    assert clips.shape[1:] == (3, g.frames_per_clip, g.image_size, g.image_size)


def test_a_larger_stride_yields_fewer_clips():
    g = reference_geometry()
    assert (
        to_clips(_frames(128), g, stride=16).shape[0] < to_clips(_frames(128), g, stride=8).shape[0]
    )


def test_doubling_the_stride_roughly_halves_the_clip_count():
    g = reference_geometry()
    assert abs(count_clips(1000, g, stride=8) / count_clips(1000, g, stride=16) - 2.0) < 0.05


def test_a_clip_spans_twice_the_frames_with_step_two():
    g = ClipGeometry.videomae_giant()
    assert clip_span(g, frame_step=1) == 16
    assert clip_span(g, frame_step=2) == 32


def test_the_clip_still_holds_the_same_frame_count_at_any_step():
    """Covering more time costs nothing extra: the backbone always gets the same
    number of frames."""
    g = reference_geometry()
    consecutive = to_clips(_frames(64), g, stride=8, frame_step=1)
    spaced = to_clips(_frames(64), g, stride=8, frame_step=2)
    assert consecutive.shape[2] == spaced.shape[2] == g.frames_per_clip


def test_a_larger_frame_step_yields_fewer_clips():
    g = reference_geometry()
    assert count_clips(200, g, stride=8, frame_step=2) < count_clips(200, g, stride=8, frame_step=1)


def test_the_frame_step_takes_every_other_frame():
    g = reference_geometry()
    frames = np.zeros((32, 40, 40, 3), dtype=np.uint8)
    for i in range(32):
        frames[i] = i * 8

    clip = to_clips(frames, g, stride=8, frame_step=2)[0]
    brightness = clip[0, :, 0, 0]

    steps = (brightness[1:] - brightness[:-1]).abs()
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-4)


def test_a_video_shorter_than_one_clip_still_yields_one():
    """Short videos tend to be the hard cases; dropping them hides the problem."""
    assert to_clips(_frames(3), reference_geometry(), stride=8).shape[0] == 1


def test_a_video_with_no_frames_is_rejected():
    with pytest.raises(ValueError, match="no frames"):
        to_clips(np.zeros((0, 10, 10, 3), dtype=np.uint8), reference_geometry())


def test_a_wrong_input_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        to_clips(np.zeros((4, 10, 10), dtype=np.uint8), reference_geometry())


@pytest.mark.parametrize("stride,step", [(0, 2), (1, 0)])
def test_invalid_stride_or_step_is_rejected(stride, step):
    with pytest.raises(ValueError):
        to_clips(_frames(64), reference_geometry(), stride=stride, frame_step=step)


def test_normalisation_shifts_the_value_range():
    """Raw 0-255 values are not what the backbone was trained on."""
    clips = to_clips(_frames(32), reference_geometry(), stride=8)
    assert clips.min() < 0.0
    assert clips.max() < 10.0


def test_the_clip_count_matches_what_extraction_produces():
    g = reference_geometry()
    frames = _frames(120)
    for step in (1, 2):
        assert (
            count_clips(120, g, stride=5, frame_step=step)
            == to_clips(frames, g, stride=5, frame_step=step).shape[0]
        )


def test_finding_videos_ignores_other_files(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.avi").write_bytes(b"")

    assert [p.name for p in find_videos(tmp_path)] == ["a.mp4", "c.avi"]


def test_finding_videos_accepts_a_single_file(tmp_path):
    one = tmp_path / "a.mp4"
    one.write_bytes(b"")
    assert find_videos(one) == [one]


def test_cached_model_from_another_source_cannot_falsify_provenance(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from vitprune.pipeline.checkpoint import load_videomae

    source = tmp_path / "requested"
    (source / "models").mkdir(parents=True)
    (source / "models" / "modeling_feature.py").write_text("", encoding="utf-8")
    models = ModuleType("models")
    definitions = ModuleType("models.modeling_feature")
    definitions.__file__ = str(tmp_path / "other" / "models" / "modeling_feature.py")
    models.modeling_feature = definitions
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "models.modeling_feature", definitions)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ValueError, match="different source"):
        load_videomae(tmp_path / "weights.pth", source)
