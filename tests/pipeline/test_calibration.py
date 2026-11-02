"""Calibration turns videos into a frozen policy file, and the pipeline refuses to use
that policy under conditions its statistics were not measured in."""

import json
from pathlib import Path

import numpy as np
import pytest

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan
from vitprune.core.policy import BudgetPolicy
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline import calibration, video
from vitprune.pipeline.calibration import calibrate_videos, read_policy, write_policy
from vitprune.pipeline.cli import main

LEVELS = [16, 32, 48]


def _frames(path):
    # Seeded by the file name, not the full path: pytest numbers its temporary folders
    # per session, and some seeds give clips too alike to separate three regimes.
    rng = np.random.default_rng(sum(map(ord, Path(path).name)))
    return rng.integers(0, 256, size=(48, 40, 40, 3), dtype=np.uint8)


@pytest.fixture
def videos(tmp_path, monkeypatch):
    folder = tmp_path / "calibration"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (folder / name).write_bytes(name.encode())
    monkeypatch.setattr(video, "read_frames", lambda path: iter(_frames(path)))
    monkeypatch.setattr(calibration, "frame_count", lambda path: len(_frames(path)))
    return folder


@pytest.fixture
def encoder():
    geometry = reference_geometry()
    return PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan(prune_layer=1))


def _calibrate(encoder, videos, **options):
    return calibrate_videos(
        encoder, videos, LEVELS, device="cpu", precision="off", progress=lambda _: None, **options
    )


def test_calibration_records_the_policy_and_its_conditions(encoder, videos):
    content = _calibrate(encoder, videos)
    policy = BudgetPolicy.from_dict(content["policy"])
    record = content["calibration"]

    assert policy.levels == tuple(LEVELS)
    assert record["clips"] == 6
    assert record["prune_layer"] == 1
    assert [len(v["source"]["sha256"]) for v in record["videos"]] == [64, 64]


def test_calibration_spreads_its_clips_over_each_video(encoder, videos):
    """The first clips of a video are not a sample of it."""
    record = _calibrate(encoder, videos, per_video=2)["calibration"]
    assert record["clips"] == 4
    assert [v["clips"] for v in record["videos"]] == [2, 2]


def test_a_policy_file_is_never_overwritten(encoder, videos, tmp_path):
    path = write_policy(tmp_path / "policy.json", _calibrate(encoder, videos))
    with pytest.raises(FileExistsError):
        write_policy(path, {})


def test_a_policy_is_read_back_for_the_same_setting(encoder, videos, tmp_path):
    content = _calibrate(encoder, videos)
    path = write_policy(tmp_path / "policy.json", content)
    assert read_policy(path, prune_layer=1, frame_step=2).version == content["policy"]["version"]


@pytest.mark.parametrize("setting", [{"prune_layer": 2}, {"prune_layer": 1, "frame_step": 1}])
def test_a_policy_is_refused_under_another_setting(encoder, videos, tmp_path, setting):
    path = write_policy(tmp_path / "policy.json", _calibrate(encoder, videos))
    with pytest.raises(ValueError, match="calibrated with"):
        read_policy(path, **setting)


def test_an_edited_policy_file_is_refused(encoder, videos, tmp_path):
    content = _calibrate(encoder, videos)
    content["policy"]["levels"] = [16, 32, 64]
    path = write_policy(tmp_path / "policy.json", content)
    with pytest.raises(ValueError, match="version"):
        read_policy(path, prune_layer=1)


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


def test_calibrate_then_extract_with_the_policy(model, videos, tmp_path, capsys):
    policy, out = tmp_path / "policy.json", tmp_path / "features"
    common = [*model, "--videos", str(videos), "--precision", "off", "--prune-layer", "1"]
    levels = ["--levels", *map(str, LEVELS)]

    assert main(["calibrate", *common, *levels, "--out", str(policy)]) == 0
    assert main(["extract", *common, "--policy", str(policy), "--out", str(out)]) == 0

    version = json.loads(policy.read_text(encoding="utf-8"))["policy"]["version"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    tokens = [t for video in manifest["outputs"].values() for t in video["budget"]["tokens"]]
    assert manifest["settings"]["budget"]["version"] == version
    assert len(tokens) == 6 and set(tokens) <= set(LEVELS)
    assert manifest["cost"]["tokens_pruned"] == pytest.approx(np.mean(tokens))
    with np.load(out / "a.npz") as features:
        assert features["feats"].shape[0] == 3

    capsys.readouterr()
    assert main(["cost", "--policy", str(policy), "--prune-layer", "1"]) == 0
    assert "adaptive budget: 16, 32, 48 tokens per clip" in capsys.readouterr().out


def test_extraction_without_a_policy_is_the_dense_baseline(model, videos, tmp_path):
    out = tmp_path / "dense"
    arguments = [*model, "--videos", str(videos), "--precision", "off", "--out", str(out)]
    assert main(["extract", *arguments]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["budget"] is None
    assert manifest["cost"]["predicted_speedup"] == 1.0
    assert all("budget" not in video for video in manifest["outputs"].values())


def test_run_reads_one_video_as_a_stream(model, videos, tmp_path):
    policy, out = tmp_path / "policy.json", tmp_path / "run"
    common = [*model, "--precision", "off", "--prune-layer", "1"]
    levels = ["--levels", *map(str, LEVELS)]
    assert main(["calibrate", *common, "--videos", str(videos), *levels, "--out", str(policy)]) == 0

    single = ["--video", str(videos / "a.mp4"), "--policy", str(policy), "--compare"]
    assert main(["run", *common, *single, "--inspect-clips", "2", "--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["timing"]["clips"] == 3
    assert report["timing"]["measured_speedup"] is not None


def test_extraction_refuses_a_policy_from_another_cut(model, videos, tmp_path):
    policy = tmp_path / "policy.json"
    common = [*model, "--videos", str(videos), "--precision", "off"]
    levels = ["--levels", *map(str, LEVELS)]
    assert main(["calibrate", *common, *levels, "--prune-layer", "1", "--out", str(policy)]) == 0

    out = tmp_path / "features"
    elsewhere = ["--policy", str(policy), "--prune-layer", "2", "--out", str(out)]
    assert main(["extract", *common, *elsewhere]) == 2
    assert not out.exists()
