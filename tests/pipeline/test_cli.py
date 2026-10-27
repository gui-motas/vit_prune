"""The command line only translates arguments and prints. These tests cover that
translation and the error messages, not the logic, which belongs to the modules
being called and is tested there.
"""

import json

import pytest

from vitprune.pipeline.cli import main


def test_the_cost_command_runs_without_a_gpu_or_checkpoint(capsys):
    """This has to work on day one, before any machine is available."""
    assert main(["cost"]) == 0
    out = capsys.readouterr().out
    assert "dense: 2048 tokens per clip" in out
    assert "adaptive budget: 256, 512, 768 tokens per clip" in out


def test_cost_reports_the_predicted_speedup(capsys):
    main(["cost", "--tokens", "256", "512"])
    assert "predicted speedup" in capsys.readouterr().out


def test_cost_states_that_it_is_an_estimate(capsys):
    """Without that note the number would be mistaken for a result."""
    main(["cost"])
    out = capsys.readouterr().out
    assert "estimates" in out
    assert "vitprune latency" in out


def test_cost_accepts_a_catalogued_accelerator(capsys):
    main(["cost", "--gpu", "a40"])
    assert "A40" in capsys.readouterr().out


def test_cost_accepts_a_hand_described_accelerator(capsys):
    main(["cost", "--peak-tflops", "500", "--bandwidth-gbs", "2000"])
    assert "500 TFLOP/s" in capsys.readouterr().out


def test_an_unknown_accelerator_produces_a_readable_error(capsys):
    assert main(["cost", "--gpu", "no-such-card"]) == 2
    assert "Known" in capsys.readouterr().err


def test_cost_writes_the_result_when_asked(tmp_path):
    target = tmp_path / "cost.json"
    main(["cost", "--tokens", "256", "768", "--save", str(target)])

    data = json.loads(target.read_text(encoding="utf-8"))

    assert [point["label"] for point in data["points"]] == ["dense", "adaptive budget"]
    assert data["points"][0]["predicted_speedup"] == 1.0
    assert data["points"][1]["tokens"] == [256, 768]


def test_cost_estimates_machine_time(capsys):
    main(["cost", "--hours", "7.2"])
    assert "GPU hours" in capsys.readouterr().out


def test_without_a_command_the_help_is_shown(capsys):
    assert main([]) == 1
    assert "cost" in capsys.readouterr().out


def test_the_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "vitprune" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command", ["check", "calibrate", "cost", "latency", "drift", "run", "extract", "evaluate"]
)
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit):
        main([command, "--help"])
    assert capsys.readouterr().out


def test_extract_requires_weights_and_a_model_source(capsys):
    with pytest.raises(SystemExit):
        main(["extract", "--videos", "x", "--out", "y"])
    assert "--weights" in capsys.readouterr().err


def test_run_requires_a_video(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--weights", "w", "--model-source", "s"])
    assert "--video" in capsys.readouterr().err


def test_evaluate_requires_the_downstream_model(capsys):
    with pytest.raises(SystemExit):
        main(["evaluate", "--features", "x", "--out", "y"])
    assert "--downstream-project" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--tokens", "4096"], ["--prune-layer", "41"]])
def test_cost_rejects_invalid_plans(arguments):
    assert main(["cost", *arguments]) == 2


def test_latency_requires_a_policy_and_videos(capsys):
    with pytest.raises(SystemExit):
        main(["latency", "--weights", "w", "--model-source", "s"])
    err = capsys.readouterr().err
    assert "--policy" in err and "--videos" in err


def test_drift_requires_a_policy(capsys):
    arguments = ["drift", "--weights", "w", "--model-source", "s", "--videos", "v"]
    assert main(arguments) == 2
    assert "--policy" in capsys.readouterr().err


def test_check_command_uses_the_original_reference(monkeypatch, capsys):
    import torch

    from vitprune.core import ClipGeometry
    from vitprune.core.testing import reference_backbone, reference_geometry
    from vitprune.pipeline import checkpoint

    geometry = reference_geometry()
    backbone = reference_backbone(geometry)
    monkeypatch.setattr(ClipGeometry, "videomae_giant", lambda: geometry)
    monkeypatch.setattr(checkpoint, "load_videomae", lambda *args, **kwargs: backbone)
    assert main(["check", "--weights", "w", "--model-source", "s", "--device", "cpu"]) == 0
    assert "matches reference" in capsys.readouterr().out

    backbone.forward_features = lambda clips: torch.zeros(1, geometry.embed_dim)
    assert main(["check", "--weights", "w", "--model-source", "s", "--device", "cpu"]) == 2
