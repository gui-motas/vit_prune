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


@pytest.mark.parametrize("difference, matches, median, expected", [
    (0.0, True, 0.95, {"G1": True, "G2": True, "passes": True}),
    (2e-4, True, 0.95, {"G1": False, "G2": True, "passes": False}),
    (0.0, False, 0.95, {"G1": False, "G2": True, "passes": False}),
    (0.0, True, 0.899, {"G1": True, "G2": False, "passes": False}),
    (1e-4, True, 0.90, {"G1": True, "G2": True, "passes": True}),
])
def test_the_p0_gates_follow_the_registered_thresholds(difference, matches, median, expected):
    from vitprune.pipeline.quality import p0_verdict

    summary = {"levels": {"1536": {"cos": {"p50": median}}}}
    found = p0_verdict(summary, difference, matches)
    assert {gate: found[gate] for gate in expected} == expected
    assert found["cos_p50_at_1536"] == median and found["identity_max_diff"] == difference


# ---------------------------------------------------------------------------
# Fitting and validating the policy, on records written by hand
# ---------------------------------------------------------------------------
def _write_records(folder, videos, clips=40, seed=0, outside=()):
    """Records of ``videos`` (stem -> driver). The drift grows with the probe error ``c``
    and with fewer tokens; motion and redundancy are unrelated to it. The clips of the
    videos in ``outside`` have a motion no fit video has."""
    from dataclasses import asdict

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    geometry = reference_geometry()
    settings = {"levels": LEVELS, "depths": [], "clips_per_video": clips, "stride": 16,
                "frame_step": 2, "batch": 8, "precision": "off", "device": "cpu",
                "prune_layer": 2, "motion_weight": 0.5, "spatial_prior": False,
                "size_biased_attention": False,
                "probe": {"cells": [2, 2], "top": 2, "fixed": [5, 7, 13, 15]},
                "geometry": asdict(geometry), "weights": "abc"}
    (folder / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    sources = {}
    for stem, driver in videos.items():
        lines = []
        for clip in range(clips):
            c = float(rng.random())
            drift = {str(level): {"d_cos": c * (64 - level) / 48 * 0.4 + 0.01 * float(rng.random())}
                     for level in LEVELS[:-1]}
            lines.append({"video": stem, "driver": driver, "start": 16 * clip,
                          "motion": 1000.0 if stem in outside else 30 + 10 * float(rng.random()),
                          "redundancy": 0.73 + 0.05 * float(rng.random()),
                          "probe_mean": c / 2, "probe_p90": c, "levels": drift})
        (folder / f"{stem}.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        sources[stem] = {"source": {"sha256": stem * 4}, "driver": driver, "clips": clips,
                         "requested": clips}
    (folder / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    return folder


FIT = {"a1": "1", "a2": "1", "b1": "2", "b2": "2", "c1": "3", "c2": "3"}
HOLDOUT = {"h8": "8", "h9": "9"}


def _fit(tmp_path, **options):
    from vitprune.pipeline.quality import fit_quality_policy

    outside = options.pop("outside", ())
    folder = _write_records(tmp_path / "records", FIT | HOLDOUT, outside=outside)
    return fit_quality_policy(folder, list(HOLDOUT), **options)


def test_the_fit_uses_only_the_fit_videos_and_validates_on_the_holdout(tmp_path):
    from vitprune.core.policy import QualityAwareBudgetPolicy

    content = _fit(tmp_path)
    policy = QualityAwareBudgetPolicy.from_dict(content["policy"])

    assert sum(policy.support) == 6 * 40
    splits = {video["stem"]: video["split"] for video in content["calibration"]["videos"]}
    assert splits == {stem: "fit" for stem in FIT} | {stem: "holdout" for stem in HOLDOUT}
    assert content["validation"]["with_probe"]["clips"] == 80


def test_a_driver_on_both_sides_of_the_split_is_refused(tmp_path):
    from vitprune.pipeline.quality import fit_quality_policy

    folder = _write_records(tmp_path / "records", FIT | HOLDOUT)
    with pytest.raises(ValueError, match="both sides"):
        fit_quality_policy(folder, ["a2", "h9"])


def test_an_unknown_holdout_video_is_refused(tmp_path):
    from vitprune.pipeline.quality import fit_quality_policy

    folder = _write_records(tmp_path / "records", FIT | HOLDOUT)
    with pytest.raises(ValueError, match="zz"):
        fit_quality_policy(folder, ["zz"])


def test_a_record_with_a_non_finite_drift_is_refused_by_name(tmp_path):
    from vitprune.pipeline.quality import fit_quality_policy

    folder = _write_records(tmp_path / "records", FIT | HOLDOUT)
    lines = (folder / "b1.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[3])
    record["levels"]["32"]["d_cos"] = float("nan")
    lines[3] = json.dumps(record)
    (folder / "b1.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="b1 at frame 48"):
        fit_quality_policy(folder, list(HOLDOUT))


def test_the_validation_reports_each_criterion_with_its_counts(tmp_path):
    validation = _fit(tmp_path, tolerance=0.2)["validation"]

    for criterion in ("V0", "V1", "V2", "V4"):
        assert "passes" in validation[criterion]
    assert validation["V0"]["clips"] == 80
    assert validation["V2"]["compressed"] == validation["with_probe"]["compressed"]
    assert all(found["n"] == 80 for found in validation["V1"]["by_level"].values())
    assert set(validation["G3"]) == {"V0", "V1", "V2", "V4", "passes"}
    assert len(validation["bins"]) == 6


def test_the_probe_signal_passes_when_it_tracks_the_drift(tmp_path):
    validation = _fit(tmp_path, tolerance=0.2)["validation"]

    assert validation["V0"]["passes"]
    assert validation["V1"]["used_levels"]
    assert validation["V1"]["passes"]


def test_a_policy_sending_everything_to_dense_is_degenerate(tmp_path):
    gate = _fit(tmp_path, tolerance=1e-9)["validation"]["G3"]
    assert gate == {"V0": False, "V1": None, "V2": None, "V4": None, "passes": False}


def test_a_holdout_driver_with_nothing_compressed_is_reported_not_divided(tmp_path):
    validation = _fit(tmp_path, tolerance=0.2, outside=("h8",))["validation"]

    driver = validation["with_probe"]["coverage_by_driver"]["8"]
    assert driver == {"clips": 40, "compressed": 0, "within": 0, "coverage": None}
    assert validation["with_probe"]["reasons"]["out_of_range"] == 40


def test_the_ablation_never_needs_the_probe(tmp_path):
    assert _fit(tmp_path, no_probe=True)["policy"]["alpha"] == 1.0


def test_the_curve_covers_the_registered_tolerances(tmp_path):
    curve = _fit(tmp_path)["validation"]["curve"]

    for name in ("with_probe", "without_probe"):
        assert [point["tolerance"] for point in curve[name]] == [0.10, 0.15, 0.20, 0.30]
    assert set(curve["fixed"]) == {"16", "32", "48"}
    assert set(curve["fixed"]["16"]["within"]) == {"0.1", "0.15", "0.2", "0.3"}


def test_a_quality_aware_policy_is_read_back_and_checked(tmp_path):
    from vitprune.core.policy import QualityAwareBudgetPolicy
    from vitprune.pipeline.calibration import read_policy, write_policy

    path = write_policy(tmp_path / "qabp.json", _fit(tmp_path))
    assert isinstance(read_policy(path, prune_layer=2, frame_step=2, precision="off"),
                      QualityAwareBudgetPolicy)
    with pytest.raises(ValueError, match="precision"):
        read_policy(path, prune_layer=2, precision="fp16")
    with pytest.raises(ValueError, match="prune_layer"):
        read_policy(path, prune_layer=1)


def test_a_policy_of_unknown_type_is_refused(tmp_path):
    from vitprune.pipeline.calibration import read_policy, write_policy

    content = _fit(tmp_path)
    content["policy"]["policy_type"] = "something_else"
    path = write_policy(tmp_path / "other.json", content)
    with pytest.raises(ValueError, match="policy type"):
        read_policy(path, prune_layer=2)


def test_an_edited_quality_aware_policy_is_refused(tmp_path):
    from vitprune.pipeline.calibration import read_policy, write_policy

    content = _fit(tmp_path)
    content["policy"]["tolerance"] = 0.5
    path = write_policy(tmp_path / "edited.json", content)
    with pytest.raises(ValueError, match="version"):
        read_policy(path, prune_layer=2)


def test_calibrate_quality_writes_the_policy_and_prints_the_verdict(tmp_path, capsys):
    folder = _write_records(tmp_path / "records", FIT | HOLDOUT)
    arguments = ["calibrate-quality", "--records", str(folder), "--holdout", *HOLDOUT,
                 "--out", str(tmp_path / "qabp.json")]
    assert main(arguments) == 0
    assert "G3: " in capsys.readouterr().out
    assert main(arguments) == 2  # the policy written is never replaced
