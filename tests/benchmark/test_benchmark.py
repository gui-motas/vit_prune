"""The benchmark's settings, its record and the commands that drive them.

The commands only translate arguments; what they run is replaced here, so these tests need
neither a GPU, weights nor a video.
"""

import json

import pytest
import torch

from vitprune.pipeline.benchmark.inference import inside_share, refuse_other_runs
from vitprune.pipeline.benchmark.settings import Benchmark, RunRecord
from vitprune.pipeline.cli import main
from vitprune.pipeline.provenance import SourceFile


def _benchmark(tmp_path, **changes) -> Benchmark:
    inputs = {"policy": "policy.json", "region": "region.json", "weights": "w.pth"}
    inputs = {name: tmp_path / file for name, file in inputs.items()} | changes
    return Benchmark(videos=tmp_path / "videos", out=tmp_path / "out", model_source=tmp_path,
                     **inputs)


def test_the_share_inside_the_region_ignores_the_padding():
    region = torch.tensor([1.0, 1.0, 0.0, 0.0])
    kept = torch.tensor([
        [[0, 2], [1, 3]],  # two of four kept tokens inside
        [[0, -1], [1, -1]],  # one per slice, padded to the widest clip; both inside
    ])
    assert inside_share(kept, region) == [0.5, 1.0]


@pytest.mark.parametrize("changes", [
    {"precision": "half"}, {"batch": 0}, {"stride": 0}, {"frame_step": 0},
    {"prune_layer": 41}, {"fixed_rows": 17}, {"fixed_weight": 1.5}, {"show": 0},
])
def test_a_run_with_invalid_settings_is_refused_before_anything_loads(tmp_path, changes):
    with pytest.raises(ValueError):
        _benchmark(tmp_path, **changes)


def test_the_evaluated_videos_are_the_ones_named(tmp_path):
    (tmp_path / "videos").mkdir()
    for name in ("a.MP4", "b.mp4", "c.mp4"):
        (tmp_path / "videos" / name).touch()

    assert [p.stem for p in _benchmark(tmp_path, evaluate=("c", "a")).evaluated()] == ["a", "c"]
    assert [p.stem for p in _benchmark(tmp_path).evaluated()] == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError, match="'z'"):
        _benchmark(tmp_path, evaluate=("a", "z")).evaluated()


def test_a_folder_without_videos_is_refused(tmp_path):
    (tmp_path / "videos").mkdir()
    with pytest.raises(FileNotFoundError, match="no video found"):
        _benchmark(tmp_path).evaluated()


def test_a_run_of_another_configuration_is_not_resumed(tmp_path):
    policy, region = tmp_path / "policy.json", tmp_path / "region.json"
    policy.write_text("{}", encoding="utf-8")
    region.write_text("{}", encoding="utf-8")
    record = tmp_path / "run.json"
    settings = {"stride": 16, "batch": 8}

    refuse_other_runs(record, settings, policy, region)  # nothing to resume yet
    record.write_text(json.dumps({
        "settings": settings,
        "policy": {"sha256": SourceFile.of(policy).sha256},
        "region": {"sha256": SourceFile.of(region).sha256},
    }), encoding="utf-8")
    refuse_other_runs(record, settings, policy, region)
    with pytest.raises(ValueError, match="another configuration"):
        refuse_other_runs(record, {**settings, "batch": 4}, policy, region)
    region.write_text('{"other": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="another configuration"):
        refuse_other_runs(record, settings, policy, region)


def test_a_run_of_an_older_version_is_refused(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"schedules": {"v": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="run again"):
        RunRecord.read(tmp_path)


def test_a_record_lists_the_evaluated_videos(tmp_path):
    video = {"fps": 8.0, "frames": 64, "clips": 4}
    (tmp_path / "run.json").write_text(json.dumps({"schedules": {"v": video}}), encoding="utf-8")
    record = RunRecord.read(tmp_path)
    assert record.stems == ["v"] and record.video("v")["frames"] == 64


def test_a_record_needs_a_finished_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="vitprune benchmark run"):
        RunRecord.read(tmp_path)


# ---------------------------------------------------------------------------
# the commands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", ["run", "ama", "report"])
def test_every_benchmark_step_has_help(step, capsys):
    with pytest.raises(SystemExit):
        main(["benchmark", step, "--help"])
    assert capsys.readouterr().out


def test_benchmark_without_a_step_shows_its_steps(capsys):
    assert main(["benchmark"]) == 1
    assert "report" in capsys.readouterr().out


def test_the_run_requires_its_inputs(capsys):
    with pytest.raises(SystemExit):
        main(["benchmark", "run", "--videos", "v"])
    err = capsys.readouterr().err
    assert "--region" in err and "--weights" in err


def test_the_mask_study_needs_its_policy(tmp_path, capsys):
    command = ["benchmark", "run", "--videos", str(tmp_path), "--region", "r.json",
               "--weights", "w.pth", "--model-source", "src"]
    assert main(command) == 2
    assert "--policy" in capsys.readouterr().err


def test_the_ama_requires_the_model(capsys):
    with pytest.raises(SystemExit):
        main(["benchmark", "ama"])
    assert "--ama-project" in capsys.readouterr().err


def _run_command(tmp_path, *extra) -> list[str]:
    return ["benchmark", "run", "--videos", str(tmp_path / "videos"),
            "--policy", "p.json", "--region", "r.json", "--weights", "w.pth",
            "--model-source", "src", *extra]


def test_every_setting_of_the_run_comes_from_its_flags(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from vitprune.pipeline.benchmark import inference

    seen = []
    monkeypatch.setattr(inference, "run_benchmark", seen.append)
    command = _run_command(
        tmp_path, "--evaluate", "a", "b", "--device", "cpu",
        "--precision", "bf16", "--batch", "2", "--stride", "8", "--frame-step", "3",
        "--prune-layer", "4", "--fixed-rows", "2", "--fixed-weight", "0.5", "--out", "o",
        "--learnable-positions", "--show", "3",
    )

    assert main(command) == 0

    bench = seen[0]
    assert (bench.evaluate, bench.device, bench.precision, bench.batch) == (
        ("a", "b"), "cpu", "bf16", 2)
    assert (bench.stride, bench.frame_step, bench.prune_layer) == (8, 3, 4)
    assert bench.fixed_mask == {"rows": 2, "weight": 0.5}
    assert bench.learnable_positions and bench.show == 3
    assert str(bench.out) == "o"
    assert (str(bench.policy), str(bench.region)) == ("p.json", "r.json")
    assert (str(bench.weights), str(bench.model_source)) == ("w.pth", "src")


def test_a_run_of_another_configuration_is_reported_as_an_error(tmp_path, monkeypatch, capsys):
    pytest.importorskip("cv2")
    from vitprune.pipeline.benchmark import inference

    def refuse(bench):
        raise ValueError("holds results of another configuration")

    monkeypatch.setattr(inference, "run_benchmark", refuse)
    assert main(_run_command(tmp_path)) == 2
    assert "another configuration" in capsys.readouterr().err


def test_policy_variant_names_are_checked(tmp_path):
    from vitprune.pipeline.benchmark.settings import VARIANTS

    study = _benchmark(tmp_path, policies=(("qabp", tmp_path / "q.json"),), fixed=(1024,))
    assert study.variants == ("denso", "legacy_original", "qabp", "fixo_1024")
    assert _benchmark(tmp_path).variants == VARIANTS  # the mask study, unchanged
    for name in ("denso", "legacy_original", "fixo_1", "prog_custom", "Upper", "a-b", ""):
        with pytest.raises(ValueError):
            _benchmark(tmp_path, policies=((name, tmp_path / "q.json"),))
    with pytest.raises(ValueError):
        _benchmark(tmp_path, policies=(("a", tmp_path / "x"), ("a", tmp_path / "y")))
    with pytest.raises(ValueError):
        _benchmark(tmp_path, fixed=(0,))


def test_the_policies_come_from_name_equals_path(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from pathlib import Path

    from vitprune.pipeline.benchmark import inference

    seen = []
    monkeypatch.setattr(inference, "run_benchmark", seen.append)
    command = _run_command(tmp_path, "--policies", "qabp=q.json", "legacy_recalibrada=l.json",
                           "--fixed", "1024", "1536")

    assert main(command) == 0
    assert seen[0].policies == (("qabp", Path("q.json")), ("legacy_recalibrada", Path("l.json")))
    assert seen[0].fixed == (1024, 1536)
    assert main(_run_command(tmp_path, "--policies", "qabp")) == 2  # NAME=PATH or nothing
    assert main(_run_command(tmp_path, "--policies", "prog_1536_1024_mascara=p.json")) == 2


def test_the_policy_study_adds_proportional_attention_and_needs_no_legacy_policy(tmp_path):
    study = _benchmark(tmp_path, policy=None, fixed=(1536, 1024), proportional=(1536,))
    assert study.variants == ("denso", "fixo_1536", "fixo_1024", "fixo_1536_pa")
    with pytest.raises(ValueError, match="--policy"):
        _benchmark(tmp_path, policy=None)  # the mask study
    with pytest.raises(ValueError):
        _benchmark(tmp_path, policy=None, fixed=(1536, 1536))
    with pytest.raises(ValueError):
        _benchmark(tmp_path, policy=None, proportional=(1001,))  # not a multiple of the slices


def test_proportional_attention_comes_from_its_flag(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from vitprune.pipeline.benchmark import inference

    seen = []
    monkeypatch.setattr(inference, "run_benchmark", seen.append)
    command = ["benchmark", "run", "--videos", str(tmp_path), "--region", "r.json",
               "--weights", "w.pth", "--model-source", "src", "--fixed", "1024",
               "--proportional", "1536", "1024"]

    assert main(command) == 0
    assert seen[0].policy is None and seen[0].proportional == (1536, 1024)
    assert seen[0].variants == ("denso", "fixo_1024", "fixo_1536_pa", "fixo_1024_pa")


def test_the_policy_study_adds_budgets_masked_by_the_driver_region(tmp_path):
    study = _benchmark(tmp_path, policy=None, masked=(1024, 768))
    assert study.policy_study
    assert study.variants == ("denso", "fixo_1024_mascara", "fixo_768_mascara")
    with pytest.raises(ValueError):
        _benchmark(tmp_path, policy=None, masked=(1024, 1024))
    with pytest.raises(ValueError):
        _benchmark(tmp_path, policy=None, masked=(1001,))  # not a multiple of the slices


def test_masked_budgets_come_from_their_flag(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from vitprune.pipeline.benchmark import inference

    seen = []
    monkeypatch.setattr(inference, "run_benchmark", seen.append)
    command = ["benchmark", "run", "--videos", str(tmp_path), "--region", "r.json",
               "--weights", "w.pth", "--model-source", "src", "--masked", "1024", "768"]

    assert main(command) == 0
    assert seen[0].policy is None and seen[0].masked == (1024, 768)
    assert seen[0].variants == ("denso", "fixo_1024_mascara", "fixo_768_mascara")


def test_progressive_options_come_from_the_command_line(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from vitprune.pipeline.benchmark import inference

    seen = []
    monkeypatch.setattr(inference, "run_benchmark", seen.append)
    command = ["benchmark", "run", "--videos", str(tmp_path), "--region", "r.json",
               "--weights", "w.pth", "--model-source", "src", "--attention", "sdpa",
               "--progressive", "1536", "10:1024", "20:768", "--allow-region-overlap"]
    assert main(command) == 0
    bench = seen[0]
    assert bench.progressive == ((2, 1536), (10, 1024), (20, 768))
    assert bench.variants == ("denso", "prog_1536_1024_768_mascara")
    assert bench.policy_study and bench.attention == "sdpa" and bench.allow_region_overlap


@pytest.mark.parametrize("progressive", [
    ("1001", "10:768"), ("1536",), ("1536", "bad"), ("1536", "10:xx"),
    ("1536", "2:1024"), ("1536", "40:1024"), ("1536", "10:512"),
])
def test_invalid_progressive_command_is_refused(tmp_path, progressive):
    command = _run_command(tmp_path, "--progressive", *progressive)
    assert main(command) == 2


@pytest.mark.parametrize("changes", [
    {"attention": "bad"},
    {"progressive": ((2, 1536),)},
    {"progressive": ((3, 1536), (10, 1024))},
    {"progressive": ((2, 1536), (10, 1024)),
     "policies": (("prog_1536_1024_mascara", "p.json"),)},
])
def test_invalid_progressive_settings_are_refused(tmp_path, changes):
    with pytest.raises(ValueError):
        _benchmark(tmp_path, **changes)
