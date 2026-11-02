"""The driver region: a detector run once on calibration footage, turned into a prior
that fills each slice's slots from inside the region first."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vitprune.core import ClipGeometry
from vitprune.core.priors import box_prior
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline import calibration, region
from vitprune.pipeline import video as reader
from vitprune.pipeline.cli import main
from vitprune.pipeline.region import (
    calibrate_region,
    largest_person,
    read_region,
    render_region,
    write_region,
)

LEVELS = [16, 32, 48]


def _frames(path):
    """48 noisy 40x40 frames: the first pixel holds the frame's index, the second the
    initial of the video's name."""
    rng = np.random.default_rng(sum(map(ord, Path(path).name)))
    frames = rng.integers(0, 256, size=(48, 40, 40, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(48)
    frames[:, 0, 0, 1] = ord(Path(path).name[0])
    return frames


def _detector(seen):
    """Finds the driver small at the top left in frame 8, nobody confident in 24, and
    larger at the bottom right in 40. In video ``c`` the driver fills frame 40."""
    found = {
        8: [[0, 0, 10, 10, 0.9, 0], [0, 0, 40, 40, 0.9, 2]],
        24: [[0, 0, 40, 40, 0.4, 0]],
        40: [[20, 20, 40, 40, 0.8, 0]],
    }

    def detect(frame):
        index = int(frame[0, 0, 0])
        seen.append(index)
        if chr(int(frame[0, 0, 1])) == "c" and index == 40:
            return np.array([[0, 0, 40, 40, 0.9, 0]], dtype=np.float32)
        return np.array(found.get(index, []), dtype=np.float32).reshape(-1, 6)

    return detect


@pytest.fixture
def videos(tmp_path, monkeypatch):
    folder = tmp_path / "calibration"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (folder / name).write_bytes(name.encode())
    for module in (reader, region):
        monkeypatch.setattr(
            module, "read_frames", lambda path, start=0: iter(_frames(path)[start:])
        )
    for module in (region, calibration):
        monkeypatch.setattr(module, "frame_count", lambda path: len(_frames(path)))
    return folder


def _region_file(tmp_path, prior, grid_size=4):
    content = {"region": {"grid_size": grid_size, "prior": prior.tolist()}, "calibration": {}}
    return write_region(tmp_path / "region.json", content)


def test_the_largest_confident_person_is_kept_as_fractions_of_the_frame():
    detections = np.array([
        [0, 0, 100, 100, 0.9, 2],  # a car, larger than anyone
        [0, 0, 20, 10, 0.9, 0],  # a small person
        [0, 0, 90, 50, 0.5, 0],  # larger, but not above the threshold
        [10, 20, 60, 70, 0.6, 0],  # the driver
    ])
    assert largest_person(detections, width=100, height=200) == (0.1, 0.1, 0.6, 0.35)


@pytest.mark.parametrize("detections", [np.zeros((0, 6)), np.array([[0, 0, 9, 9, 0.3, 0]])])
def test_without_a_confident_person_there_is_no_box(detections):
    assert largest_person(detections, width=10, height=10) is None


def test_a_box_is_clipped_to_the_frame():
    detections = np.array([[-5, -5, 120, 50, 0.9, 0]])
    assert largest_person(detections, width=100, height=100) == (0.0, 0.0, 1.0, 0.5)


def test_the_region_is_the_largest_crop_of_each_video_not_every_box(videos):
    seen = []
    content = calibrate_region(
        videos, _detector(seen), reference_geometry(), progress=lambda _: None
    )
    record = content["calibration"]

    assert seen == [8, 24, 40] * 2
    # The bottom-right quarter only: joining every box would add the top-left cell.
    assert torch.tensor(content["region"]["prior"]).nonzero().flatten().tolist() == [
        10, 11, 14, 15
    ]
    assert (record["clips"], record["detected"], record["area"]) == (6, 4, 4 / 16)
    assert record["box"] == [0.5, 0.5, 1.0, 1.0]
    assert [video["detected"] for video in record["videos"]] == [2, 2]
    assert [video["crop"] for video in record["videos"]] == [[0.5, 0.5, 1.0, 1.0]] * 2


def test_a_video_unlike_the_rest_does_not_stretch_the_region(videos):
    (videos / "c.mp4").write_bytes(b"c")
    content = calibrate_region(videos, _detector([]), reference_geometry(), progress=lambda _: None)

    assert [video["crop"] for video in content["calibration"]["videos"]][-1] == [0, 0, 1, 1]
    assert content["calibration"]["box"] == [0.5, 0.5, 1.0, 1.0]
    assert content["calibration"]["area"] == 4 / 16


def test_region_detection_spreads_its_clips_over_each_video(videos):
    seen = []
    content = calibrate_region(
        videos, _detector(seen), reference_geometry(), per_video=2, progress=lambda _: None
    )
    assert seen == [8, 40, 8, 40]
    assert content["calibration"]["clips"] == 4


def test_without_any_driver_there_is_no_region(videos):
    nobody = lambda frame: np.zeros((0, 6))  # noqa: E731
    with pytest.raises(ValueError, match="no person"):
        calibrate_region(videos, nobody, reference_geometry(), progress=lambda _: None)


def test_a_region_is_read_back_as_a_prior_and_never_overwritten(tmp_path):
    prior = box_prior(4, [(0.0, 0.0, 0.5, 0.5)])
    path = _region_file(tmp_path, prior)

    assert torch.equal(read_region(path, reference_geometry()), prior)
    with pytest.raises(FileExistsError):
        write_region(path, {})


@pytest.mark.parametrize(
    "prior,grid_size,match",
    [
        (torch.ones(16), 8, "grid"),
        (torch.zeros(16), 4, "at least one"),
        (torch.full((16,), 0.5), 4, "0 or 1"),
    ],
)
def test_a_region_that_does_not_fit_the_backbone_is_refused(tmp_path, prior, grid_size, match):
    path = _region_file(tmp_path, prior, grid_size)
    with pytest.raises(ValueError, match=match):
        read_region(path, reference_geometry())


def test_the_region_renders_one_grid_row_per_line():
    prior = box_prior(4, [(0.0, 0.0, 0.25, 0.25), (0.75, 0.75, 1.0, 1.0)])
    assert render_region(prior, 4) == "#...\n....\n....\n...#"


# ---------------------------------------------------------------------------
# Through the command line
# ---------------------------------------------------------------------------
@pytest.fixture
def model(tmp_path, monkeypatch):
    from vitprune.pipeline import checkpoint

    geometry = reference_geometry()
    backbone = reference_backbone(geometry)
    monkeypatch.setattr(ClipGeometry, "videomae_giant", lambda: geometry)
    monkeypatch.setattr(checkpoint, "load_videomae", lambda *args, **kwargs: backbone)

    source = tmp_path / "videomae"
    (source / "models").mkdir(parents=True)
    (source / "models" / "modeling_feature.py").write_text("# model", encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    return ["--weights", str(weights), "--model-source", str(source), "--device", "cpu"]


def test_the_region_command_writes_the_region_and_draws_it(
    videos, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(ClipGeometry, "videomae_giant", reference_geometry)
    monkeypatch.setattr(region, "load_yolo", lambda *args: (_detector([]), {"name": "fake"}))
    out = tmp_path / "region.json"
    arguments = ["region", "--videos", str(videos), "--detector", "yolo26s.pt", "--out", str(out)]

    assert main(arguments) == 0
    assert "....\n....\n..##\n..##" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["calibration"]["detector"] == {
        "name": "fake"
    }
    assert main(arguments) == 2


def test_the_region_changes_which_tokens_are_kept_not_how_many(model, videos, tmp_path):
    """Isolating the variable: with the same policy, the counts per clip match and only
    the prior recorded in the manifest differs."""
    policy = tmp_path / "policy.json"
    common = [*model, "--videos", str(videos), "--precision", "off", "--prune-layer", "1"]
    levels = ["--levels", *map(str, LEVELS)]
    assert main(["calibrate", *common, *levels, "--out", str(policy)]) == 0
    prior = box_prior(4, [(0.0, 0.0, 0.5, 1.0)])
    path = _region_file(tmp_path, prior)

    adaptive = [*common, "--policy", str(policy)]
    assert main(["extract", *adaptive, "--out", str(tmp_path / "b")]) == 0
    assert main(["extract", *adaptive, "--region", str(path), "--out", str(tmp_path / "r")]) == 0

    without, within = (
        json.loads((tmp_path / name / "manifest.json").read_text(encoding="utf-8"))
        for name in ("b", "r")
    )
    assert within["settings"]["spatial_prior"] == prior.tolist()
    assert without["settings"]["spatial_prior"] is None
    for name, video in within["outputs"].items():
        assert video["budget"]["tokens"] == without["outputs"][name]["budget"]["tokens"]


@pytest.mark.parametrize(
    "extra", [[], ["--policy", "policy.json", "--prior-top-rows", "1"]]
)
def test_a_region_needs_a_policy_and_excludes_the_top_rows_prior(
    model, videos, tmp_path, extra, capsys
):
    path = _region_file(tmp_path, box_prior(4, [(0.0, 0.0, 0.5, 1.0)]))
    arguments = [*model, "--videos", str(videos), "--precision", "off", "--prune-layer", "1"]
    out = tmp_path / "features"

    assert main(["extract", *arguments, "--region", str(path), *extra, "--out", str(out)]) == 2
    assert "region" in capsys.readouterr().err
    assert not out.exists()
