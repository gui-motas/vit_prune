"""The run end to end, on a miniature backbone and a synthetic video.

Real model, weights and footage are replaced; what stays real is everything the run does
around them: sampling clips, running the four variants, and writing the features, the
rows, the figures' inputs and the record.
"""

import json

import numpy as np
import pytest
import torch
from torch import nn

cv2 = pytest.importorskip("cv2")

from vitprune.core.testing import reference_backbone, reference_geometry  # noqa: E402
from vitprune.pipeline.benchmark import inference, settings  # noqa: E402
from vitprune.pipeline.benchmark.settings import VARIANTS, Benchmark  # noqa: E402
from vitprune.pipeline.provenance import MANIFEST_NAME  # noqa: E402
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
