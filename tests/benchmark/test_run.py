"""The run end to end, on a miniature backbone and a synthetic video.

Real model, weights and footage are replaced; what stays real is everything the run does
around them: sampling clips, running the four variants, and writing the features, the
rows, the figures' inputs and the record.
"""

import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

cv2 = pytest.importorskip("cv2")

from vitprune.core.testing import reference_backbone, reference_geometry  # noqa: E402
from vitprune.pipeline.benchmark import inference, settings  # noqa: E402
from vitprune.pipeline.benchmark.settings import VARIANTS, Benchmark  # noqa: E402
from vitprune.pipeline.provenance import MANIFEST_NAME, SourceFile  # noqa: E402
from vitprune.pipeline.region import write_region  # noqa: E402

FPS, SECONDS = 8, 8


@pytest.fixture
def geometry(monkeypatch):
    """A 4x4 grid, 4 temporal slices and 8-frame clips, in place of the Giant."""
    found = reference_geometry(depth=4)
    monkeypatch.setattr(settings, "GEOMETRY", found)
    monkeypatch.setattr(inference, "GEOMETRY", found)
    return found


@pytest.fixture
def model(monkeypatch, geometry):
    backbone = reference_backbone(geometry)
    backbone.head = nn.Linear(geometry.embed_dim, 3)
    monkeypatch.setattr(inference, "load_model", lambda bench: backbone)
    return backbone


def _video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(FPS * SECONDS):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("this OpenCV build cannot write mp4v")


@pytest.fixture
def bench(tmp_path, geometry, uniform_budget):
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos / "v.mp4")
    _video(videos / "unused.mp4")
    policy = uniform_budget(geometry, 16)
    (tmp_path / "policy.json").write_text(json.dumps({
        "policy": policy.to_dict(), "calibration": {"prune_layer": 2, "frame_step": 2},
    }), encoding="utf-8")
    prior = torch.zeros(geometry.tokens_per_slice)
    prior[4:] = 1.0
    write_region(tmp_path / "region.json", {
        "region": {"grid_size": geometry.grid_size, "prior": prior.tolist()},
        "calibration": {},
    })
    (tmp_path / "w.pth").write_bytes(b"weights")
    return Benchmark(
        videos=videos, out=tmp_path / "out",
        policy=tmp_path / "policy.json", region=tmp_path / "region.json",
        weights=tmp_path / "w.pth", model_source=tmp_path, evaluate=("v",), device="cpu",
        batch=2, prune_layer=2, fixed_rows=1, show=2,
    )


def _rows(bench):
    return [json.loads(line) for line in bench.clips.read_text("utf-8").splitlines()]


def test_the_run_writes_what_the_next_steps_read(bench, model, geometry):
    inference.run_benchmark(bench)

    clips = len(bench.clip_starts(FPS * SECONDS))
    rows = _rows(bench)
    assert len(rows) == clips * len(VARIANTS)
    assert {row["video"] for row in rows} == {"v"}  # the video not named is left alone
    for variant in VARIANTS:
        with np.load(bench.features / variant / "v.npz") as stored:
            assert stored["feats"].shape == (clips, geometry.embed_dim)
            assert stored["a_prob"].shape == (clips, 3)
        assert (bench.features / variant / MANIFEST_NAME).exists()
    record = json.loads(bench.record.read_text("utf-8"))
    video = {"fps": float(FPS), "frames": FPS * SECONDS, "clips": clips}
    assert record["schedules"] == {"v": video}
    assert "annotations" not in record and record["settings"]["show"] == 2
    assert record["settings"]["evaluate"] == ["v"]
    assert record["settings"]["fixed_mask"] == {"rows": 1, "weight": 0.15}
    assert all("segment" not in row for row in rows)


def test_the_three_pruned_variants_keep_the_same_tokens_but_not_the_dense(bench, model, geometry):
    inference.run_benchmark(bench)

    by_variant = {v: [r for r in _rows(bench) if r["variant"] == v] for v in VARIANTS}
    assert {r["tokens"] for r in by_variant["denso"]} == {geometry.num_tokens}
    pruned = [[r["tokens"] for r in by_variant[v]] for v in VARIANTS[1:]]
    assert pruned[0] == pruned[1] == pruned[2]
    assert max(pruned[0]) < geometry.num_tokens
    assert all(r["inside_region"] is None for r in by_variant["denso"])
    assert all(0.0 <= r["inside_region"] <= 1.0 for r in by_variant["vitprune"])
    assert all(r["cosine_to_dense"] == pytest.approx(1.0) for r in by_variant["denso"])


def test_the_inputs_of_the_figures_are_kept(bench, model):
    inference.run_benchmark(bench)

    shown = sorted(bench.selections.glob("v_*.npz"))
    assert len(shown) == 2  # the fixture's --show
    with np.load(shown[0]) as arrays:
        assert {"vitprune_kept", "vitprune_mascara_fixa_prior",
                "vitprune_mascara_inteligente_prior"} <= set(arrays.files)
    assert len(list((bench.frames / "v").glob("*.jpg"))) == 2


def test_the_figures_show_clips_spread_over_the_video(bench, model):
    bench.show = 3
    inference.run_benchmark(bench)

    starts = sorted(int(p.stem.rsplit("_", 1)[1]) for p in bench.selections.glob("v_*.npz"))
    assert starts == [0, 32, 48]  # 4 clips (0, 16, 32, 48); the first and the last included
    assert len(list((bench.frames / "v").glob("*.jpg"))) == 3


def test_show_larger_than_the_clips_shows_them_all(bench, model):
    bench.show = 99
    inference.run_benchmark(bench)

    assert len(list(bench.selections.glob("v_*.npz"))) == 4


def test_an_interrupted_run_resumes_without_repeating_a_video(bench, model, capsys):
    inference.run_benchmark(bench)
    first = _rows(bench)

    inference.run_benchmark(bench)

    assert _rows(bench) == first
    assert "v: already done" in capsys.readouterr().out


def test_another_configuration_is_not_resumed_next_to_a_finished_run(bench, model):
    inference.run_benchmark(bench)
    bench.batch = 1

    with pytest.raises(ValueError, match="another configuration"):
        inference.run_benchmark(bench)


def test_a_policy_calibrated_for_another_cut_is_refused_before_the_model_loads(
    bench, monkeypatch, geometry
):
    def never(bench):
        raise AssertionError("the model must not load")

    monkeypatch.setattr(inference, "load_model", never)
    bench.prune_layer = 1

    with pytest.raises(ValueError, match="prune_layer"):
        inference.run_benchmark(bench)


def _quality_file(path, drivers=("1",)):
    """A quality-aware policy file, calibrated on synthetic clips of ``drivers``."""
    from vitprune.core.compressibility import CompressibilityFeatures
    from vitprune.core.policy import calibrate_quality_aware

    generator, count = torch.Generator().manual_seed(0), 12
    features = CompressibilityFeatures(*(torch.rand(count, generator=generator) for _ in range(4)))
    drift = torch.rand(count, 3, generator=generator, dtype=torch.float64) * 0.2
    names = [f"c{i % 3}" for i in range(count)]
    policy = calibrate_quality_aware(
        features, drift, names, [drivers[i % len(drivers)] for i in range(count)],
        (16, 32, 48, 64), bins=2, min_support=1, min_videos=1, probe_cells=(2, 2), probe_top=2,
    )
    videos = [{"stem": f"c{i}", "driver": driver, "split": "fit",
               "source": {"sha256": str(i) * 64}} for i, driver in enumerate(drivers)]
    path.write_text(json.dumps({"policy": policy.to_dict(), "calibration": {
        "prune_layer": 2, "frame_step": 2, "precision": "off", "videos": videos}}), "utf-8")
    return path


def test_the_policy_study_runs_its_variants(bench, model, tmp_path):
    from vitprune.core.policy import REASONS

    bench.policies = (("qabp", _quality_file(tmp_path / "q.json")),)
    bench.fixed, bench.proportional = (32,), (32,)
    inference.run_benchmark(bench)

    variants = ("denso", "legacy_original", "qabp", "fixo_32", "fixo_32_pa")
    rows = _rows(bench)
    assert len(rows) == len(bench.clip_starts(FPS * SECONDS)) * len(variants)
    assert {r["variant"] for r in rows} == set(variants)
    assert all(r["tokens"] == 32 for r in rows if r["variant"].startswith("fixo_32"))
    assert all(r["inside_region"] is None for r in rows)  # no inspection pass in this study
    for row in rows:
        if row["variant"] == "qabp":
            assert row["reason"] in REASONS and isinstance(row["predicted_risk"], float)
        else:
            assert row["reason"] is None and row["predicted_risk"] is None
        if row["variant"] == "denso":
            assert row["rel_l2_to_dense"] == pytest.approx(0.0, abs=1e-6)
    record = json.loads(bench.record.read_text("utf-8"))
    assert "masked" not in record["settings"]  # so the runs made before it still resume
    assert record["settings"]["variants"] == list(variants)
    assert set(record["policies"]) == {"legacy_original", "qabp"}
    assert record["probe_timing"]["probe_ms_per_clip"] > 0


def test_the_variant_timed_first_changes_from_batch_to_batch(bench, model):
    bench.policy, bench.fixed = None, (32, 48)
    inference.run_benchmark(bench)

    rows = _rows(bench)
    firsts = [rows[i]["variant"] for i in range(0, len(rows), bench.batch * 3)]
    assert firsts[:2] == ["denso", "fixo_32"]


def test_a_quality_aware_policy_calibrated_on_an_evaluated_driver_is_refused(
    bench, monkeypatch, tmp_path
):
    def never(bench):
        raise AssertionError("the model must not load")

    monkeypatch.setattr(inference, "load_model", never)
    bench.policies = (("qabp", _quality_file(tmp_path / "q.json", drivers=("v",))),)

    with pytest.raises(ValueError, match="evaluated"):
        inference.run_benchmark(bench)


def test_the_legacy_overlap_with_the_evaluated_videos_is_recorded(bench, model, tmp_path):
    from dataclasses import asdict

    content = json.loads(bench.policy.read_text("utf-8"))
    content["calibration"]["videos"] = [{"source": asdict(SourceFile.of(bench.videos / "v.mp4"))}]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(content), encoding="utf-8")
    bench.policies = (("legacy_recalibrada", legacy),)
    inference.run_benchmark(bench)

    overlap = json.loads(bench.record.read_text("utf-8"))["calibration_overlap"]
    assert overlap["legacy_recalibrada"] == {"videos": ["v"], "drivers": ["v"]}
    assert overlap["legacy_original"] == {"videos": [], "drivers": []}


def test_the_proxy_study_joins_the_cut_signals_with_what_the_head_lost(bench, model, geometry):
    from vitprune.core import PrunedEncoder, PruningPlan
    from vitprune.pipeline.benchmark import proxy

    bench.policy, bench.fixed = None, (32,)
    inference.run_benchmark(bench)
    ama = bench.out / "ama"
    ama.mkdir()
    (ama / "labels.json").write_text(json.dumps({"database": {"v": {
        "subset": "validation", "annotations": [{"label_id": 1, "segment": [0.0, 4.0]},
                                                 {"label_id": 2, "segment": [4.0, 8.0]}]}}}))
    columns = "video-id,t-start,t-end,label,score\n"
    for variant, rows in {"denso": ["v,0.0,4.0,1,0.9", "v,4.0,8.0,2,0.9"],
                          "fixo_32": ["v,0.0,4.0,1,0.9"]}.items():
        (ama / variant).mkdir()
        (ama / variant / "predictions.csv").write_text(columns + "\n".join(rows) + "\n")
    encoder = PrunedEncoder(model, geometry, PruningPlan(prune_layer=bench.prune_layer))

    proxy.collect_cut_features(encoder, bench.out, device="cpu", precision="off", batch=2,
                               cells=(2, 2), top=2, progress=lambda _: None)
    found = proxy.analyse(bench.out, permutations=50)

    cut = [json.loads(line) for line in
           (bench.out / "proxy" / "cut_features.jsonl").read_text("utf-8").splitlines()]
    assert [row["start"] for row in cut] == [0, 16, 32, 48]
    variant = found["variants"]["fixo_32"]
    assert variant["labelled_clips"] == 4  # windows of 2 s inside the two 4 s actions
    assert set(variant["signals"]) == {"probe_p90", "probe_mean", "g", "motion", "redundancy",
                                       "drift"}
    assert variant["actions"] == {"annotated": 2, "found_dense": 2, "found": 1, "lost": 1}
    assert set(variant["probe_beyond_motion_redundancy"]) == {"delta_ce", "delta_p"}
    with (bench.out / "proxy" / "clips.csv").open(encoding="utf-8") as stream:
        clips = list(csv.DictReader(stream))
    assert [float(c["start_s"]) for c in clips] == [0.0, 2.0, 4.0, 6.0]  # frames over 8 fps
    with (bench.out / "proxy" / "actions.csv").open(encoding="utf-8") as stream:
        actions = list(csv.DictReader(stream))
    assert [(a["label"], a["clips"], a["found"]) for a in actions] == [
        ("1", "2", "True"), ("2", "2", "False")]


def test_a_masked_budget_keeps_its_tokens_inside_the_driver_region(bench, model, geometry):
    from vitprune.pipeline.region import read_region

    bench.policy, bench.masked = None, (32,)
    inference.run_benchmark(bench)

    assert {r["variant"] for r in _rows(bench)} == {"denso", "fixo_32_mascara"}
    assert json.loads(bench.record.read_text("utf-8"))["settings"]["masked"] == [32]
    region = read_region(bench.region, geometry)
    encoder = inference.encoders(bench, model, {}, region)["fixo_32_mascara"]
    kept = encoder.select(torch.randn(*geometry.input_shape(batch=2))).kept
    assert region[kept].all()  # 8 a slice, all among the 12 positions of the region


def test_a_video_an_interruption_cut_is_redone_without_duplicate_rows(bench, model):
    bench.policy, bench.fixed = None, (32,)
    inference.run_benchmark(bench)
    complete = sorted((r["variant"], r["start_frame"]) for r in _rows(bench))
    # What a crash leaves behind: a variant's features missing, the last row cut mid-line.
    (bench.features / "fixo_32" / "v.npz").unlink()
    with bench.clips.open("a", encoding="utf-8") as stream:
        stream.write('{"video": "v", "start_fr')

    inference.run_benchmark(bench)

    assert sorted((r["variant"], r["start_frame"]) for r in _rows(bench)) == complete
    assert (bench.features / "fixo_32" / "v.npz").exists()


def test_a_crash_while_writing_the_rows_leaves_the_video_to_redo(bench, model, monkeypatch):
    bench.policy, bench.fixed = None, (32,)
    write = inference.append_rows

    def power_cut(path, rows):
        raise OSError("power cut")

    monkeypatch.setattr(inference, "append_rows", power_cut)
    with pytest.raises(OSError):
        inference.run_benchmark(bench)
    monkeypatch.setattr(inference, "append_rows", write)
    inference.run_benchmark(bench)

    clips = len(bench.clip_starts(FPS * SECONDS))
    assert sorted(r["variant"] for r in _rows(bench)) == ["denso"] * clips + ["fixo_32"] * clips
