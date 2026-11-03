"""Collecting the per-clip drift of each budget from videos: the material the quality-aware
policy is calibrated on and the P0 diagnosis reads. A record lost, mixed between videos or
measured under other settings would bias both without any visible failure."""

import json
from pathlib import Path

import numpy as np
import pytest

from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan
from vitprune.core.testing import reference_backbone, reference_geometry
from vitprune.pipeline import quality, video
from vitprune.pipeline.cli import main
from vitprune.pipeline.quality import (
    collect_records,
    driver_of,
    read_records,
    summarise_records,
)

LEVELS = [16, 32, 48, 64]


def _frames(path):
    rng = np.random.default_rng(sum(map(ord, Path(path).name)))
    return rng.integers(0, 256, size=(48, 40, 40, 3), dtype=np.uint8)


@pytest.fixture
def videos(tmp_path, monkeypatch):
    folder = tmp_path / "videos"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (folder / name).write_bytes(name.encode())
    monkeypatch.setattr(video, "read_frames", lambda path, start=0: iter(_frames(path)[start:]))
    monkeypatch.setattr(quality, "frame_count", lambda path: len(_frames(path)))
    return folder


@pytest.fixture
def encoder():
    geometry = reference_geometry()
    return PrunedEncoder(reference_backbone(geometry), geometry, PruningPlan(prune_layer=1))


def _collect(encoder, videos, out, **options):
    options = {"clips_per_video": 2, "device": "cpu", "progress": lambda _: None} | options
    return collect_records(encoder, [videos], out, options.pop("levels", LEVELS), **options)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_driver_is_the_user_id_of_the_video():
    assert driver_of("Dashboard_user_id_38159_NoAudio_7") == "38159"
    assert driver_of("a") == "a"


def test_the_records_hold_clips_spread_over_each_video(encoder, videos, tmp_path):
    out = _collect(encoder, videos, tmp_path / "records")

    for stem in ("a", "b"):
        lines = _lines(out / f"{stem}.jsonl")
        assert [line["start"] for line in lines] == [0, 32]  # 3 clips; first and last kept
        assert all(line["video"] == stem and line["driver"] == stem for line in lines)
        assert set(lines[0]["levels"]) == {"16", "32", "48"}
    sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert [len(sources[stem]["source"]["sha256"]) for stem in ("a", "b")] == [64, 64]


def test_a_video_with_fewer_clips_than_asked_records_what_it_has(encoder, videos, tmp_path):
    out = _collect(encoder, videos, tmp_path / "records", clips_per_video=99)

    sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert len(_lines(out / "a.jsonl")) == 3
    assert sources["a"]["clips"] == 3 and sources["a"]["requested"] == 99


def test_an_interrupted_collection_resumes_video_by_video(encoder, videos, tmp_path, monkeypatch):
    out = _collect(encoder, videos, tmp_path / "records")
    (out / "b.jsonl").unlink()
    measured = []
    original = PrunedEncoder.encode_levels

    def counting(self, clips, *args, **kwargs):
        measured.append(clips.shape[0])
        return original(self, clips, *args, **kwargs)

    monkeypatch.setattr(PrunedEncoder, "encode_levels", counting)
    _collect(encoder, videos, tmp_path / "records")

    assert measured == [2]
    assert len(_lines(out / "b.jsonl")) == 2
    _, _, records = read_records(out)
    assert [(r["video"], r["start"]) for r in records] == [("a", 0), ("a", 32), ("b", 0),
                                                           ("b", 32)]


def test_a_folder_of_other_settings_is_refused(encoder, videos, tmp_path):
    _collect(encoder, videos, tmp_path / "records", levels=[16, 32, 64])
    with pytest.raises(FileExistsError, match="other settings"):
        _collect(encoder, videos, tmp_path / "records", levels=[16, 48, 64])


def test_the_summary_gives_the_percentiles_of_the_cosine_per_level():
    def record(drift):
        stage = {"1": {"cos": 1 - drift, "rel_l2": drift, "norm": 2.0}}
        return {"video": "v", "driver": "1", "dense": {"norm": 1.0, "stages": {"1": {"norm": 4.0}}},
                "levels": {"16": {"d_cos": drift, "d_rel": drift, "norm": 1.0, "top1": drift < 0.15,
                                  "merge_error": {"mean": 0.1, "p90": 0.2, "max": 0.5},
                                  "stages": stage}}}

    found = summarise_records([record(d) for d in (0.0, 0.1, 0.2, 0.3)], [16, 64])["levels"]["16"]
    assert found["n"] == 4
    assert found["cos"]["p10"] == pytest.approx(0.73)
    assert found["cos"]["p50"] == pytest.approx(0.85)
    assert found["cos"]["p90"] == pytest.approx(0.97)
    assert found["top1"] == pytest.approx(0.5)
    assert found["stages"]["1"]["norm_ratio_p50"] == pytest.approx(0.5)


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


def test_budget_curve_writes_the_records_and_prints_the_summary(model, videos, tmp_path, capsys):
    out = tmp_path / "records"
    arguments = [*model, "--videos", str(videos / "a.mp4"), str(videos / "b.mp4"),
                 "--levels", *map(str, LEVELS), "--prune-layer", "1", "--depths", "1", "4",
                 "--clips-per-video", "2", "--out", str(out)]
    assert main(["budget-curve", *arguments]) == 0

    printed = capsys.readouterr().out
    assert all(f"{level} tokens" in printed for level in (16, 32, 48))
    settings, sources, records = read_records(out)
    assert settings["depths"] == [1, 4] and settings["levels"] == LEVELS
    assert len(records) == 4 and set(sources) == {"a", "b"}
    assert (out / "summary.json").exists()
