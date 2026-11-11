import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK))


def load(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(rows):
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end", "label", "score"])


def test_actions_match_best_tiou_above_score_threshold():
    compare = load("arms_compare")
    actions = frame([["v", 0, 10, 1, 1], ["v", 20, 30, 2, 1],
                     ["v", 40, 50, 3, 1]])
    detections = {
        "a": frame([["v", 0, 6, 1, .31], ["v", 20, 30, 2, .29],
                    ["v", 40, 44, 3, .7], ["v", 40, 50, 3, .29], ["v", 0, 10, 9, .8]]),
        "b": frame([["v", 0, 10, 1, .3], ["v", 1, 9, 1, .84],
                    ["v", 40, 50, 9, .9]]),
    }
    rows = compare.action_table(actions, detections).set_index(["action", "arm"])
    assert len(rows) == 6
    assert bool(rows.loc[(0, "a"), "recovered"])
    assert rows.loc[(0, "a"), "score"] == .31
    assert rows.loc[(0, "a"), "tiou"] == pytest.approx(.6)
    assert not rows.loc[(1, "a"), "recovered"]
    assert pd.isna(rows.loc[(1, "a"), "score"])
    assert rows.loc[(1, "a"), "tiou"] == 0
    assert not rows.loc[(2, "a"), "recovered"]
    assert rows.loc[(2, "a"), "tiou"] == pytest.approx(.4)
    assert rows.loc[(0, "b"), "score"] == .3
    assert rows.loc[(0, "b"), "start_err"] == 0
    assert rows.loc[(0, "b"), "end_err"] == 0
    assert pd.isna(rows.loc[(1, "b"), "score"])
    assert rows.loc[(1, "b"), "best_any"] is None


def test_competing_detections_choose_best_tiou_not_highest_score():
    compare = load("arms_compare")
    actions = frame([["v", 0, 10, 1, 1]])
    found = frame([["v", 1, 9, 1, .8], ["v", 0, 10, 1, .4]])
    row = compare.action_table(actions, {"a": found}).iloc[0]
    assert row["recovered"]
    assert (row["score"], row["tiou"], row["start_err"], row["end_err"]) == (.4, 1, 0, 0)


def test_below_threshold_best_any_is_visible_without_changing_recovery():
    compare = load("arms_compare")
    actions = frame([["v", 0, 10, 1, 1]])
    found = frame([["v", 0, 10, 1, .29], ["v", 0, 4, 1, .8]])
    row = compare.action_table(actions, {"a": found}).iloc[0]
    assert not row["recovered"] and row["score"] == .8 and row["tiou"] == .4
    assert row["best_any"] == {"score": .29, "tiou": 1.0}


def test_compatibility_checks_provenance_and_ignores_intended_differences():
    compare = load("arms_compare")
    original = {"weights": {"sha256": "w"}, "region": {"sha256": "r"},
                "schedules": {"v": {"fps": 30, "frames": 100, "clips": 10}},
                "settings": {"stride": 16, "frame_step": 2, "batch": 4, "prune_layer": 2,
                             "evaluate": ["v"], "precision": "off", "variants": ["denso"]}}
    other = json.loads(json.dumps(original))
    other["settings"].update(attention="sdpa", precision="fp16", masked=[32],
                             progressive=[[2, 32], [3, 24]], variants=["other"])
    assert compare.compatibility({"a": original, "b": other})["compatible"]
    for path in (("weights", "sha256"), ("region", "sha256"), ("schedules",),
                 *(("settings", k) for k in ("stride", "frame_step", "batch", "prune_layer",
                                             "evaluate"))):
        changed = json.loads(json.dumps(other))
        if len(path) == 1:
            changed[path[0]]["v"]["clips"] = 11
        else:
            changed[path[0]][path[1]] = "different"
        result = compare.compatibility({"a": original, "b": changed})
        assert not result["compatible"]
        assert any(".".join(path) in mismatch for mismatch in result["mismatches"])
    del other["weights"]["sha256"]
    assert any("weights.sha256" in item for item in
               compare.compatibility({"a": original, "b": other})["mismatches"])
    compatible = compare.compatibility({"a": original, "b": original})
    assert compatible["runs"]["a"]["attention"] == "model"
    del other["schedules"]["v"]["fps"]
    assert any("schedules.v.fps" in item for item in
               compare.compatibility({"a": original, "b": other})["mismatches"])


def test_markdown_action_cell_includes_signed_boundary_errors():
    compare = load("arms_compare")
    result = {"arms": {"a": {"latency": {"p50": 1, "p95": 1, "mean": 1}, "map": .5,
                             "recovered": 1, "lost": 0, "false_positives": 0, "detections": []}},
              "actions": [{"video-id": "v", "label": 1, "t-start": 0, "t-end": 10,
                           "arms": {"a": {"recovered": True, "score": .84, "tiou": .91,
                                          "start_err": -.5, "end_err": 1.2, "fallback": False}}}]}
    assert "✓ 0.84 (0.91) -0.5/+1.2s" in compare.markdown(result)


def test_false_positives_require_score_class_video_and_tiou():
    compare = load("arms_compare")
    actions = frame([["v", 0, 10, 1, 1]])
    detections = {"a": frame([["v", 0, 10, 1, .3], ["v", 0, 10, 2, .8],
                               ["v", 0, 4, 1, .4], ["v", 0, 10, 2, .29],
                               ["other", 0, 10, 1, .5]])}
    false = compare.false_positive_table(actions, detections)
    assert len(false) == 3
    assert set(false["label"]) == {1, 2}
    assert set(false["video-id"]) == {"v", "other"}


def test_latency_discards_only_first_video_first_eight_per_variant():
    compare = load("arms_compare")
    rows = pd.DataFrame([{"video": v, "variant": arm, "seconds": seconds}
                         for v, seconds in [("first", 10), ("second", 2)]
                         for arm in ("a", "b") for _ in range(10)])
    timing = compare.latency(rows, 8)
    assert timing["a"]["clips"] == 12
    assert timing["a"]["mean"] == pytest.approx(40 / 12)
    assert timing["b"] == timing["a"]
    assert compare.latency(rows, 3)["a"]["clips"] == 17


def test_sample_batches_preserve_first_and_middle_schedule():
    check = load("c2_check")
    assert check.sample_starts(list(range(40))) == [list(range(8)), list(range(16, 24))]
    assert check.sample_starts(list(range(8))) == [list(range(8))]
    assert check.sample_starts(list(range(4))) == [list(range(4))]


def test_progressive_plan_and_mean_tokens():
    latency = load("arms_latency")
    from vitprune.pipeline.benchmark.settings import GEOMETRY

    plans = latency.make_plans(["denso", "fixo_1024_mascara", "prog:1536,10:1024,20:768"],
                               None, "model")
    assert plans["prog:1536,10:1024,20:768"].stages == ((10, 1024), (20, 768))
    assert latency.mean_tokens(plans["denso"], GEOMETRY) == 2048
    assert latency.mean_tokens(plans["fixo_1024_mascara"], GEOMETRY) == 1075.2
    assert latency.mean_tokens(plans["prog:1536,10:1024,20:768"], GEOMETRY) == 1049.6
    with pytest.raises(ValueError):
        latency.make_plans(["prog:1536,10:768,9:512"], None, "model")


def test_comparison_cli_reads_real_run_records_and_writes_both_outputs(tmp_path):
    import shutil

    run = tmp_path / "run"
    ama = run / "ama"
    (ama / "denso").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "settings": {"variants": ["denso"], "batch": 8},
        "schedules": {"v": {"fps": 30, "frames": 100, "clips": 10}},
    }))
    (ama / "labels.json").write_text(json.dumps({"database": {"v": {"annotations": [
        {"segment": [0, 10], "label_id": 1}, {"segment": [20, 30], "label_id": 2},
    ]}}}))
    frame([["v", 0, 6, 1, .31], ["v", 20, 30, 2, .29], ["v", 0, 10, 9, .8]]).to_csv(
        ama / "denso" / "predictions.csv", index=False)
    (ama / "denso" / "map.json").write_text(json.dumps({"average": .42}))
    (run / "clips.jsonl").write_text("\n".join(json.dumps({
        "video": "v", "variant": "denso", "seconds": 10 if i < 8 else 2,
    }) for i in range(10)))
    out = tmp_path / "comparison.json"
    subprocess.run([sys.executable, str(BENCHMARK / "arms_compare.py"), "--run", f"test={run}",
                    "--out", str(out)], check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    arm = result["arms"]["test/denso"]
    assert (arm["recovered"], arm["lost"], arm["false_positives"], arm["map"]) == (1, 1, 1, .42)
    assert arm["latency"]["p50"] == 2
    assert result["actions"][1]["arms"]["test/denso"]["score"] is None
    assert result["actions"][1]["arms"]["test/denso"]["best_any"] == {"score": .29, "tiou": 1.0}
    assert "✓ 0.31 (0.60) +0.0/-4.0s" in out.with_suffix(".md").read_text()
    assert "✗ (0.29 @1.00)" in out.with_suffix(".md").read_text()
    assert "WARNING" in out.with_suffix(".md").read_text()

    other = tmp_path / "other"
    shutil.copytree(run, other)
    for folder, batch, weight in ((run, 3, "a"), (other, 4, "b")):
        path = folder / "run.json"
        record = json.loads(path.read_text())
        record.update(weights={"sha256": weight}, region={"sha256": "region"})
        record["settings"].update(batch=batch, stride=16, frame_step=2, prune_layer=2,
                                  evaluate=["v"])
        path.write_text(json.dumps(record))
    output = subprocess.run([sys.executable, str(BENCHMARK / "arms_compare.py"),
                             "--run", f"a={run}", "--run", f"b={other}", "--out", str(out)],
                            check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert result["arms"]["a/denso"]["latency"]["clips"] == 7
    assert result["arms"]["b/denso"]["latency"]["clips"] == 6
    assert not result["compatibility"]["compatible"]
    for text in (output.stdout, out.with_suffix(".md").read_text()):
        assert "WARNING" in text and "weights.sha256" in text and "settings.batch" in text


@pytest.mark.parametrize("issue,gate", [("none", True), ("nan", False), ("drift", False)])
def test_c2_gate_and_profiler_with_cpu_model(tmp_path, monkeypatch, capsys, issue, gate):
    import torch
    import torch.profiler

    import vitprune.core

    check = load("c2_check")

    class Encoder:
        def __init__(self, model, geometry, plan):
            self.plan = plan

        def encode(self, x):
            if self.plan.attention == "sdpa":
                if issue == "nan":
                    return x * float("nan")
                if issue == "drift":
                    return x + torch.tensor([.1, -.1])
            return x

        def select(self, x):
            return SimpleNamespace(kept=torch.zeros((len(x), 2, 1), dtype=torch.long))

    class Profiler:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def key_averages(self):
            return [SimpleNamespace(key="FlashAttention",
                                    device_type=torch.autograd.DeviceType.CUDA),
                    SimpleNamespace(key="aten::softmax", device_type=torch.autograd.DeviceType.CPU),
                    SimpleNamespace(key="other", device_type=torch.autograd.DeviceType.CUDA)]

    model = SimpleNamespace(head=lambda x: x)
    monkeypatch.setattr(check, "setup", lambda args: (SimpleNamespace(device="cpu"), model, None))
    monkeypatch.setattr(check, "sample_batches", lambda bench: ([torch.ones((8, 2))], []))
    monkeypatch.setattr(vitprune.core, "PrunedEncoder", Encoder)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(sys, "argv", ["c2_check.py", "--videos", "v", "--region", "r",
                                     "--weights", "w", "--model-source", "s", "--out",
                                     str(tmp_path)])
    check.main()
    result = json.loads((tmp_path / "check.json").read_text())
    assert result["gate"] is gate
    assert result["kernels"]["fp16_sdpa"] == ["FlashAttention"]
    assert result["configurations"]["fixo_1024_mascara/fp32_model"]["kept_identical_fraction"] == 1
    output = capsys.readouterr().out
    assert ("GATE PASS" if gate else "GATE FAIL") in output
    assert "| Configuration |" in output


def test_sample_batches_refuses_undecodable_clip(monkeypatch):
    import torch

    from vitprune.pipeline import video
    from vitprune.pipeline.benchmark import inference

    check = load("c2_check")
    path = Path("v.mp4")
    bench = SimpleNamespace(evaluated=lambda: [path], clip_starts=lambda _: list(range(40)))
    monkeypatch.setattr(inference, "video_info", lambda _: {"frames": 40})
    monkeypatch.setattr(video, "iter_clips", lambda path, geometry, stride, frame_step, starts:
                        ((start, torch.tensor(start)) for start in starts if start != 22))
    with pytest.raises(ValueError, match=r"v.*incomplete.*7/8"):
        check.sample_batches(bench)


@pytest.mark.parametrize("count", [4, 8, 15])
def test_sample_batches_refuses_fewer_than_two_complete_scheduled_batches(monkeypatch, count):
    import torch

    from vitprune.pipeline import video
    from vitprune.pipeline.benchmark import inference

    check = load("c2_check")
    bench = SimpleNamespace(evaluated=lambda: [Path("v.mp4")],
                            clip_starts=lambda _: list(range(count)))
    monkeypatch.setattr(inference, "video_info", lambda _: {"frames": count})
    monkeypatch.setattr(video, "iter_clips", lambda path, geometry, stride, frame_step, starts:
                        ((start, torch.tensor(start)) for start in starts))
    with pytest.raises(ValueError, match=r"v.*two complete.*8"):
        check.sample_batches(bench)


def test_sample_batches_requires_two_full_batches_per_video(monkeypatch):
    import torch

    from vitprune.pipeline import video
    from vitprune.pipeline.benchmark import inference

    check = load("c2_check")
    videos = [Path(f"v{i}.mp4") for i in range(3)]
    bench = SimpleNamespace(evaluated=lambda: videos, clip_starts=lambda _: list(range(40)))
    monkeypatch.setattr(inference, "video_info", lambda _: {"frames": 40})
    monkeypatch.setattr(video, "iter_clips", lambda path, geometry, stride, frame_step, starts:
                        ((start, torch.tensor(start)) for start in starts))
    batches, manifest = check.sample_batches(bench)
    assert len(batches) == 6
    assert sum(len(batch) for batch in batches) == 48
    assert manifest == [{"video": path.stem, "starts": starts} for path in videos
                        for starts in (list(range(8)), list(range(16, 24)))]


def test_latency_rotates_arms_and_excludes_warmup_on_cpu(tmp_path, monkeypatch):
    import torch

    import vitprune.core

    latency = load("arms_latency")
    calls = []

    class Encoder:
        def __init__(self, model, geometry, plan):
            self.name = "fixed" if plan.prunes else "dense"

        def encode(self, x):
            calls.append(self.name)
            return x

    tick = iter(i * .001 for i in range(8))
    model = SimpleNamespace(head=lambda x: x)
    monkeypatch.setattr(latency, "setup", lambda args: (SimpleNamespace(device="cpu"), model, None))
    monkeypatch.setattr(latency, "sample_batches", lambda bench: ([torch.ones((8, 2))], []))
    monkeypatch.setattr(latency.time, "perf_counter", lambda: next(tick))
    monkeypatch.setattr(vitprune.core, "PrunedEncoder", Encoder)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(sys, "argv", ["arms_latency.py", "--videos", "v", "--region", "r",
                                     "--weights", "w", "--model-source", "s", "--out",
                                     str(tmp_path), "--arms", "denso", "fixo_1024_mascara",
                                     "--warmup", "1", "--passes", "2"])
    latency.main()
    assert calls == ["dense", "fixed", "fixed", "dense", "dense", "fixed"]
    result = json.loads((tmp_path / "latency.json").read_text())
    assert result["arms"]["denso"]["batches"] == 2
    assert result["arms"]["denso"]["p50_ms"] == pytest.approx(.125)
    assert result["arms"]["denso"]["ratio_vs_fixo_1024_mascara"]["mean_ms"] == pytest.approx(1)
