"""The defect these tests prevent: evaluating pruned embeddings while silently
reading the dense folder. It raises no error, produces a plausible number, and makes
the accuracy curve come out flat -- inverting the conclusion of the study.
"""

import pytest

pytest.importorskip("yaml")

import yaml

from vitprune.pipeline.evaluation import (
    EVALUATION_MANIFEST,
    PREPARED_CONFIG,
    DownstreamModel,
    EvaluationRequest,
    build_command,
    evaluate,
    prepare_config,
    resolve_stride,
)
from vitprune.pipeline.provenance import Manifest


@pytest.fixture
def downstream(tmp_path):
    """A stand-in downstream model with the file layout of a real one."""
    project = tmp_path / "localiser"
    project.mkdir()
    (project / "eval.py").write_text("", encoding="utf-8")

    config = project / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "feat_folder": "/old/path/dense",
                    "feat_stride": 16,
                    "json_file": "/old/labels.json",
                    "input_dim": 1408,
                },
                "model": {"n_head": 8},
            }
        ),
        encoding="utf-8",
    )

    weights = project / "weights.pth.tar"
    weights.write_bytes(b"x")
    return DownstreamModel(project_dir=project, config=config, weights=weights)


@pytest.fixture
def features(tmp_path):
    folder = tmp_path / "features_adaptive"
    folder.mkdir()
    Manifest(settings={"prune_layer": 2, "stride": 16}).write(folder)
    return folder


def _prepared(destination):
    return yaml.safe_load((destination / PREPARED_CONFIG).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The main defect
# ---------------------------------------------------------------------------
def test_the_embedding_folder_is_rewritten(downstream, features, tmp_path):
    """Without this the model would read the old folder and return the baseline
    result as though it were the pruned one."""
    destination = tmp_path / "out"
    prepare_config(EvaluationRequest(features, downstream, destination))

    assert _prepared(destination)["dataset"]["feat_folder"] == str(features)


def test_the_original_configuration_is_untouched(downstream, features, tmp_path):
    before = downstream.config.read_text(encoding="utf-8")
    prepare_config(EvaluationRequest(features, downstream, tmp_path / "out"))
    assert downstream.config.read_text(encoding="utf-8") == before


def test_the_copy_sits_with_the_results(downstream, features, tmp_path):
    destination = tmp_path / "out"
    path = prepare_config(EvaluationRequest(features, downstream, destination))
    assert path.parent == destination


def test_the_remaining_configuration_survives(downstream, features, tmp_path):
    destination = tmp_path / "out"
    prepare_config(EvaluationRequest(features, downstream, destination))
    prepared = _prepared(destination)
    assert prepared["dataset"]["input_dim"] == 1408
    assert prepared["model"]["n_head"] == 8


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------
def test_the_stride_comes_from_the_extraction_record(downstream, features, tmp_path):
    assert resolve_stride(EvaluationRequest(features, downstream, tmp_path / "out")) == 16


def test_a_divergent_stride_aborts(downstream, features, tmp_path):
    """A wrong stride shifts every interval in time and mimics an accuracy drop
    that has nothing to do with pruning."""
    request = EvaluationRequest(features, downstream, tmp_path / "out", stride=8)
    with pytest.raises(ValueError, match="disagrees"):
        resolve_stride(request)


def test_a_matching_stride_is_accepted(downstream, features, tmp_path):
    assert (
        resolve_stride(EvaluationRequest(features, downstream, tmp_path / "out", stride=16)) == 16
    )


def test_without_a_record_and_without_a_stride_it_aborts(downstream, tmp_path):
    empty = tmp_path / "no_record"
    empty.mkdir()
    with pytest.raises(ValueError, match="wrong scale"):
        resolve_stride(EvaluationRequest(empty, downstream, tmp_path / "out"))


def test_without_a_record_the_supplied_stride_is_used(downstream, tmp_path):
    empty = tmp_path / "no_record"
    empty.mkdir()
    assert resolve_stride(EvaluationRequest(empty, downstream, tmp_path / "out", stride=32)) == 32


def test_the_recorded_stride_reaches_the_configuration(downstream, tmp_path):
    folder = tmp_path / "features_s8"
    folder.mkdir()
    Manifest(settings={"prune_layer": 2, "stride": 8}).write(folder)

    destination = tmp_path / "out"
    prepare_config(EvaluationRequest(folder, downstream, destination))

    assert _prepared(destination)["dataset"]["feat_stride"] == 8


# ---------------------------------------------------------------------------
# Labels and command
# ---------------------------------------------------------------------------
def test_the_label_file_is_replaced_when_supplied(downstream, features, tmp_path):
    labels = tmp_path / "validation.json"
    labels.write_text("{}", encoding="utf-8")
    destination = tmp_path / "out"

    prepare_config(EvaluationRequest(features, downstream, destination, labels=labels))

    assert _prepared(destination)["dataset"]["json_file"] == str(labels)


def test_without_labels_the_original_stays(downstream, features, tmp_path):
    destination = tmp_path / "out"
    prepare_config(EvaluationRequest(features, downstream, destination))
    assert _prepared(destination)["dataset"]["json_file"] == "/old/labels.json"


def test_the_command_uses_the_prepared_configuration(downstream, features, tmp_path):
    destination = tmp_path / "out"
    request = EvaluationRequest(features, downstream, destination)
    prepared = prepare_config(request)

    command = build_command(request, prepared)

    assert str(prepared) in command
    assert str(downstream.weights) in command
    assert "--output_csv" in command


def test_extra_arguments_reach_the_command(downstream, features, tmp_path):
    request = EvaluationRequest(
        features, downstream, tmp_path / "out", extra_arguments=["--topk", "100"]
    )
    assert build_command(request, prepare_config(request))[-2:] == ["--topk", "100"]


# ---------------------------------------------------------------------------
# Missing paths
# ---------------------------------------------------------------------------
def test_a_missing_embedding_folder_is_rejected(downstream, tmp_path):
    request = EvaluationRequest(tmp_path / "nowhere", downstream, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="embedding folder"):
        prepare_config(request)


def test_a_project_without_an_entry_point_is_rejected(tmp_path, features):
    project = tmp_path / "empty"
    project.mkdir()
    config = project / "c.yaml"
    config.write_text("{}", encoding="utf-8")
    weights = project / "w.pth.tar"
    weights.write_bytes(b"x")

    request = EvaluationRequest(features, DownstreamModel(project, config, weights), tmp_path / "o")
    with pytest.raises(FileNotFoundError, match="eval.py"):
        prepare_config(request)


def test_missing_downstream_weights_are_rejected(downstream, features, tmp_path):
    downstream.weights.unlink()
    with pytest.raises(FileNotFoundError, match="downstream model weights"):
        prepare_config(EvaluationRequest(features, downstream, tmp_path / "out"))


def test_frame_step_mismatch_aborts(downstream, features, tmp_path):
    Manifest(settings={"stride": 16, "frame_step": 2}).write(features)
    with pytest.raises(ValueError, match="frame_step.*disagrees"):
        prepare_config(EvaluationRequest(features, downstream, tmp_path / "out", frame_step=1))


def test_frame_step_is_checked_when_declared_by_the_model(downstream, features, tmp_path):
    Manifest(settings={"stride": 16, "frame_step": 2}).write(features)
    config = yaml.safe_load(downstream.config.read_text(encoding="utf-8"))
    config["dataset"]["frame_step"] = 1
    downstream.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frame_step.*disagrees"):
        prepare_config(EvaluationRequest(features, downstream, tmp_path / "out"))


def test_frame_step_assertion_requires_a_record(downstream, features, tmp_path):
    with pytest.raises(ValueError, match="no value recorded"):
        prepare_config(EvaluationRequest(features, downstream, tmp_path / "out", frame_step=2))


def test_input_width_mismatch_aborts(downstream, features, tmp_path):
    Manifest(settings={"stride": 16, "geometry": {"embed_dim": 1024}}).write(features)
    with pytest.raises(ValueError, match="input_dim"):
        prepare_config(EvaluationRequest(features, downstream, tmp_path / "out"))


@pytest.mark.parametrize("code", [0, 1])
def test_evaluation_records_command_inputs_and_exit_status(
    downstream, features, tmp_path, monkeypatch, code
):
    import json
    from types import SimpleNamespace

    monkeypatch.setattr(
        "vitprune.pipeline.evaluation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=code),
    )
    out = tmp_path / "out"
    request = EvaluationRequest(features, downstream, out)
    if code:
        with pytest.raises(RuntimeError, match="exited"):
            evaluate(request)
    else:
        assert evaluate(request) == out
    record = json.loads((out / EVALUATION_MANIFEST).read_text(encoding="utf-8"))
    assert record["statistics"]["returncode"] == code
    assert record["settings"]["prepared_config"]["sha256"]
    assert record["settings"]["extraction_manifest"]["settings"]["prune_layer"] == 2
    assert record["weights"]["sha256"]
