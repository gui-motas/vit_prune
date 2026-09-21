"""The command line only translates arguments and prints. These tests cover that
translation and the error messages, not the logic, which belongs to the modules
being called and is tested there.
"""

import json

import pytest

from vitprune.pipeline.cli import main


def test_the_cost_command_runs_without_a_gpu_or_checkpoint(capsys):
    """This has to work on day one, before any machine is available."""
    assert main(["cost", "--retention", "1.0", "0.25"]) == 0
    out = capsys.readouterr().out
    assert "retention 100%" in out
    assert "retention 25%" in out


def test_cost_reports_the_predicted_speedup(capsys):
    main(["cost", "--retention", "0.25"])
    assert "predicted speedup" in capsys.readouterr().out


def test_cost_states_that_it_is_an_estimate(capsys):
    """Without that note the number would be mistaken for a result."""
    main(["cost", "--retention", "0.25"])
    out = capsys.readouterr().out
    assert "estimates" in out
    assert "vitprune latency" in out


def test_cost_accepts_a_catalogued_accelerator(capsys):
    main(["cost", "--gpu", "a40", "--retention", "0.25"])
    assert "A40" in capsys.readouterr().out


def test_cost_accepts_a_hand_described_accelerator(capsys):
    main(["cost", "--peak-tflops", "500", "--bandwidth-gbs", "2000", "--retention", "0.25"])
    assert "500 TFLOP/s" in capsys.readouterr().out


def test_an_unknown_accelerator_produces_a_readable_error(capsys):
    assert main(["cost", "--gpu", "no-such-card"]) == 2
    assert "Known" in capsys.readouterr().err


def test_cost_writes_the_result_when_asked(tmp_path):
    target = tmp_path / "cost.json"
    main(["cost", "--retention", "1.0", "0.25", "--save", str(target)])

    data = json.loads(target.read_text(encoding="utf-8"))

    assert len(data["points"]) == 2
    assert data["points"][0]["predicted_speedup"] == 1.0


def test_cost_estimates_machine_time(capsys):
    main(["cost", "--hours", "7.2", "--retention", "0.25"])
    assert "GPU hours" in capsys.readouterr().out


def test_without_a_command_the_help_is_shown(capsys):
    assert main([]) == 1
    assert "cost" in capsys.readouterr().out


def test_the_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "vitprune" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["cost", "latency", "drift", "run", "extract", "evaluate"])
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
