import numpy as np
import pytest
import torch
from torch import nn

from vitprune.core.geometry import ClipGeometry
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline import video
from vitprune.pipeline.checkpoint import (
    LoadReport,
    disable_activation_checkpointing,
    install_weights,
    read_weights,
)
from vitprune.pipeline.video import (
    clip_span,
    find_videos,
    gather_clips,
    iter_clips,
    spread_starts,
)


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


def _clips(monkeypatch, frames, geometry=None, **sampling) -> torch.Tensor:
    """Every clip ``iter_clips`` gives for these frames, stacked."""
    monkeypatch.setattr(video, "read_frames", _reader(frames))
    clips = iter_clips("x.mp4", geometry or reference_geometry(), **sampling)
    return torch.stack([clip for _, clip in clips])


def _whole_video_clips(frames, g, stride, step) -> torch.Tensor:
    """The clips built the obvious way, from every frame at once."""
    span, total = clip_span(g, step), len(frames)
    normalised = video._normalise(frames, g.image_size)
    starts = range(0, total - span + 1, stride) if total >= span else [0]
    return torch.stack([
        normalised[[min(s + step * k, total - 1) for k in range(g.frames_per_clip)]]
        .permute(1, 0, 2, 3)
        for s in starts
    ])


def test_clips_come_out_in_the_shape_the_backbone_expects(monkeypatch):
    g = reference_geometry()
    clips = _clips(monkeypatch, _frames(64), g, stride=8)
    assert clips.shape[1:] == (3, g.frames_per_clip, g.image_size, g.image_size)


def test_a_larger_stride_yields_fewer_clips(monkeypatch):
    frames = _frames(128)
    assert len(_clips(monkeypatch, frames, stride=16)) < len(_clips(monkeypatch, frames, stride=8))


def test_a_clip_spans_twice_the_frames_with_step_two():
    g = ClipGeometry.videomae_giant()
    assert clip_span(g, frame_step=1) == 16
    assert clip_span(g, frame_step=2) == 32


def test_the_clip_still_holds_the_same_frame_count_at_any_step(monkeypatch):
    """Covering more time costs nothing extra: the backbone always gets the same
    number of frames."""
    g, frames = reference_geometry(), _frames(64)
    consecutive = _clips(monkeypatch, frames, g, stride=8, frame_step=1)
    spaced = _clips(monkeypatch, frames, g, stride=8, frame_step=2)
    assert consecutive.shape[2] == spaced.shape[2] == g.frames_per_clip


def test_a_larger_frame_step_yields_fewer_clips(monkeypatch):
    frames = _frames(200)
    spaced = _clips(monkeypatch, frames, stride=8, frame_step=2)
    assert len(spaced) < len(_clips(monkeypatch, frames, stride=8, frame_step=1))


def test_the_frame_step_takes_every_other_frame(monkeypatch):
    frames = np.zeros((32, 40, 40, 3), dtype=np.uint8)
    for i in range(32):
        frames[i] = i * 8

    clip = _clips(monkeypatch, frames, stride=8, frame_step=2)[0]
    brightness = clip[0, :, 0, 0]

    steps = (brightness[1:] - brightness[:-1]).abs()
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-4)


def test_a_video_shorter_than_one_clip_still_yields_one(monkeypatch):
    """Short videos tend to be the hard cases; dropping them hides the problem."""
    assert len(_clips(monkeypatch, _frames(3), stride=8)) == 1


@pytest.mark.parametrize("stride,step", [(0, 2), (1, 0)])
def test_invalid_stride_or_step_is_rejected(monkeypatch, stride, step):
    with pytest.raises(ValueError):
        _clips(monkeypatch, _frames(64), stride=stride, frame_step=step)


def test_normalisation_shifts_the_value_range(monkeypatch):
    """Raw 0-255 values are not what the backbone was trained on."""
    clips = _clips(monkeypatch, _frames(32), stride=8)
    assert clips.min() < 0.0
    assert clips.max() < 10.0


@pytest.mark.parametrize("total", [3, 15, 16, 23, 24, 64, 71])
@pytest.mark.parametrize("stride,step", [(8, 2), (5, 1)])
def test_streamed_clips_equal_the_clips_of_the_whole_video(monkeypatch, total, stride, step):
    """Reading a frame at a time must not change a single value, including the padding
    of a video shorter than one clip."""
    g, frames = reference_geometry(), _frames(total)
    monkeypatch.setattr(video, "read_frames", _reader(frames))

    streamed = list(iter_clips("x.mp4", g, stride=stride, frame_step=step))
    expected = _whole_video_clips(frames, g, stride, step)

    assert [start for start, _ in streamed] == list(range(0, len(streamed) * stride, stride))
    assert torch.equal(torch.stack([clip for _, clip in streamed]), expected)


def _reader(frames, calls=None):
    """A stand-in for ``read_frames`` that records where each read started."""
    def read(path, start=0):
        if calls is not None:
            calls.append(start)
        return iter(frames[start:])
    return read


def test_streaming_can_keep_only_some_clips(monkeypatch):
    g, frames, calls = reference_geometry(), _frames(64), []
    monkeypatch.setattr(video, "read_frames", _reader(frames, calls))

    streamed = dict(iter_clips("x.mp4", g, stride=8, starts=[8, 40, 41, 999]))

    assert sorted(streamed) == [8, 40]
    assert calls == [8, 40], "the frames between the two clips are skipped, not decoded"
    monkeypatch.setattr(video, "read_frames", _reader(frames))
    assert torch.equal(streamed[40], dict(iter_clips("x.mp4", g, stride=8))[40])


@pytest.mark.parametrize("starts", [[0], [0, 8, 40], [16, 24, 32], [8, 48], [48], []])
@pytest.mark.parametrize("total", [3, 64])
def test_jumping_to_some_clips_changes_no_value(monkeypatch, starts, total):
    """Overlapping, adjacent and distant clips, and the padded clip of a short video."""
    g, frames = reference_geometry(), _frames(total)
    monkeypatch.setattr(video, "read_frames", _reader(frames))

    every = dict(iter_clips("x.mp4", g, stride=8))
    picked = dict(iter_clips("x.mp4", g, stride=8, starts=starts))

    assert sorted(picked) == [s for s in starts if s in every]
    assert all(torch.equal(clip, every[start]) for start, clip in picked.items())


def test_seeking_in_a_real_file_lands_on_the_same_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    for frame in _frames(80, side=64)[:, :48]:
        writer.write(np.ascontiguousarray(frame))
    writer.release()
    g = reference_geometry()

    every = dict(iter_clips(path, g, stride=8))
    picked = dict(iter_clips(path, g, stride=8, starts=[8, 40, 64]))

    assert sorted(picked) == [8, 40, 64]
    assert all(torch.equal(clip, every[start]) for start, clip in picked.items())


def test_a_streamed_video_with_no_frames_is_rejected(monkeypatch):
    monkeypatch.setattr(video, "read_frames", _reader([]))
    with pytest.raises(ValueError, match="no frames"):
        list(iter_clips("x.mp4", reference_geometry()))


def test_spread_starts_cover_the_whole_video():
    g = reference_geometry()
    assert spread_starts(64, 3, g, stride=8) == [0, 24, 48]
    assert spread_starts(64, 100, g, stride=8) == list(range(0, 49, 8))


def test_gathered_clips_are_spread_over_every_video(tmp_path, monkeypatch):
    """The first clips of a folder are a few seconds of one video, not a sample of it."""
    g = reference_geometry()
    frames = {"a.mp4": _frames(48), "b.mp4": _frames(48)[::-1].copy()}
    for name in frames:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(video, "read_frames", lambda path, start=0: iter(frames[path.name][start:]))
    monkeypatch.setattr(video, "frame_count", lambda path: 48)

    gathered = gather_clips(tmp_path, g, stride=8, frame_step=2, count=4)

    picks = [("a.mp4", 0), ("a.mp4", 24), ("b.mp4", 8), ("b.mp4", 32)]
    expected = [dict(iter_clips(tmp_path / name, g, stride=8))[start] for name, start in picks]
    assert torch.equal(gathered, torch.stack(expected))


def test_weights_load_from_the_legacy_format_too(tmp_path):
    """Memory-mapping needs the zip format; older files fall back to a plain read."""
    model = Small()
    path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)

    loaded = read_weights(path)

    assert all(torch.equal(loaded[k], v) for k, v in model.state_dict().items())


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
