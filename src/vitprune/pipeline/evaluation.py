"""Evaluating pruned embeddings with an already-trained downstream model.

No weight is altered here, neither in the backbone nor in the downstream model.
Pruned embeddings are handed to a model trained on dense ones, and the cost of that
substitution is measured.

Doing it this way has two advantages. It is honest about what is being tested --
whether pruning preserves the information the model needs. And it is immune to a
criticism that retraining invites: with no training, pruning cannot influence the
model that judges it.

## The mistake this module exists to prevent

The downstream model finds its embeddings by reading a field in its configuration
file. If that field is not updated, it reads the previous folder -- the dense one --
and returns the baseline result as though it were the pruned one.

That raises no error, emits no warning, and produces a perfectly plausible number.
The effect on a study is severe: the accuracy curve comes out flat, and the
conclusion comes out inverted, suggesting pruning costs nothing.

So preparing the configuration here is mandatory and explicit, rather than a side
effect of some other step. Three fields are always rewritten:

* where the embeddings are;
* the stride they were produced with;
* which annotation file to use.

The stride matters because it is how clip indices are converted into timestamps. A
wrong stride shifts every predicted interval and degrades accuracy for a reason
unrelated to pruning -- an expensive failure to diagnose. It is therefore checked
against the extraction manifest before anything else runs.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vitprune.pipeline.provenance import Manifest, SourceFile, read_manifest

#: Name given to the prepared copy of the configuration, inside the results folder.
PREPARED_CONFIG = "prepared-config.yaml"
EVALUATION_MANIFEST = "evaluation-manifest.json"
#: Everything the downstream model printed, with the figures it reports.
DOWNSTREAM_LOG = "downstream.log"


@dataclass
class DownstreamModel:
    """Where the parts of the downstream model live.

    That model is not part of this project. It is treated as something to run, not
    something to modify.
    """

    project_dir: Path
    config: Path
    weights: Path
    entry_point: str = "eval.py"

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        self.config = Path(self.config).resolve()
        self.weights = Path(self.weights).resolve()

    def check(self) -> None:
        if not (self.project_dir / self.entry_point).exists():
            raise FileNotFoundError(
                f"{self.entry_point} not found in {self.project_dir}. Point "
                f"--downstream-project at the folder holding the model."
            )
        if not self.config.exists():
            raise FileNotFoundError(f"configuration not found: {self.config}")
        if not self.weights.exists():
            raise FileNotFoundError(f"downstream model weights not found: {self.weights}")


@dataclass
class EvaluationRequest:
    """Everything that defines one evaluation."""

    features: Path
    model: DownstreamModel
    destination: Path
    labels: Path | None = None
    stride: int | None = None
    extra_arguments: list[str] = field(default_factory=list)
    frame_step: int | None = None

    def __post_init__(self) -> None:
        self.features = Path(self.features).resolve()
        self.destination = Path(self.destination).resolve()
        if self.labels is not None:
            self.labels = Path(self.labels).resolve()


def prepare_config(request: EvaluationRequest) -> Path:
    """Write a copy of the configuration pointing at the right embeddings.

    The original configuration is never modified. Each evaluation gets its own copy,
    written alongside the results, which leaves a record of exactly what was run.

    Returns:
        The path of the prepared copy.
    """
    import yaml

    request.model.check()
    if not request.features.exists():
        raise FileNotFoundError(f"embedding folder not found: {request.features}")

    stride = resolve_stride(request)
    settings: dict[str, Any] = yaml.safe_load(request.model.config.read_text(encoding="utf-8"))
    dataset = settings.setdefault("dataset", {})

    dataset["feat_folder"] = str(request.features)
    dataset["feat_stride"] = stride
    if request.labels is not None:
        dataset["json_file"] = str(request.labels)
    _validate_protocol(request, dataset)

    request.destination.mkdir(parents=True, exist_ok=True)
    prepared = request.destination / PREPARED_CONFIG
    prepared.write_text(
        yaml.safe_dump(settings, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    return prepared


def resolve_stride(request: EvaluationRequest) -> int:
    """Determine the stride, preferring what the extraction recorded.

    If a stride is supplied by hand and disagrees with the record, the evaluation
    stops. A divergent stride produces intervals shifted in time and an accuracy drop
    that looks like an effect of pruning but is not.
    """
    recorded = _recorded_stride(request.features)

    if request.stride is None:
        if recorded is None:
            raise ValueError(
                f"no stride recorded in {request.features} and none supplied. Without "
                f"it, clip indices are converted into timestamps at the wrong scale."
            )
        if recorded < 1:
            raise ValueError("recorded stride must be positive")
        return recorded

    if recorded is not None and recorded != request.stride:
        raise ValueError(
            f"the supplied stride ({request.stride}) disagrees with the one recorded "
            f"at extraction ({recorded}). Evaluating like this shifts every predicted "
            f"interval in time and degrades accuracy for a reason unrelated to pruning."
        )
    if request.stride < 1:
        raise ValueError("stride must be positive")
    return request.stride


def _validate_protocol(request: EvaluationRequest, dataset: dict) -> None:
    """Check declared temporal sampling and feature width without guessing a model's schema."""
    try:
        settings = read_manifest(request.features).get("settings", {})
    except FileNotFoundError:
        settings = {}
    recorded = settings.get("frame_step")
    for expected in (request.frame_step, dataset.get("frame_step")):
        if expected is None:
            continue
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            raise ValueError("frame_step must be a positive integer")
        if recorded is None:
            raise ValueError("cannot verify frame_step: no value recorded at extraction")
        if expected != recorded:
            raise ValueError(f"frame_step ({expected}) disagrees with extraction ({recorded})")
    width = settings.get("geometry", {}).get("embed_dim")
    if width is not None and "input_dim" in dataset and dataset["input_dim"] != width:
        raise ValueError("dataset.input_dim disagrees with the extraction embed_dim")


def build_command(request: EvaluationRequest, prepared_config: Path) -> list[str]:
    """Assemble the command line that runs the downstream model."""
    return [
        sys.executable,
        request.model.entry_point,
        str(prepared_config),
        str(request.model.weights),
        "--output_csv",
        str(request.destination / "predictions.csv"),
        *request.extra_arguments,
    ]


def evaluate(request: EvaluationRequest, progress=print) -> Path:
    """Prepare the configuration and run the downstream model.

    The model's output goes to ``progress`` as it runs and to ``downstream.log`` in the
    results folder, which holds the figures the model printed.

    Returns:
        The results folder.
    """
    if (request.destination / EVALUATION_MANIFEST).exists():
        raise FileExistsError(f"refusing to overwrite the evaluation in {request.destination}")
    prepared = prepare_config(request)
    command = build_command(request, prepared)
    try:
        extraction = read_manifest(request.features)
    except FileNotFoundError:
        extraction = None
    record = Manifest(
        settings={
            "stage": "evaluation",
            "features": str(request.features),
            "stride": resolve_stride(request),
            "asserted_frame_step": request.frame_step,
            "command": command,
            "source_config": asdict(SourceFile.of(request.model.config)),
            "prepared_config": asdict(SourceFile.of(prepared)),
            "labels": asdict(SourceFile.of(request.labels)) if request.labels else None,
            "entry_point": asdict(SourceFile.of(
                request.model.project_dir / request.model.entry_point
            )),
            "extraction_manifest": extraction,
        },
        weights=SourceFile.of(request.model.weights),
        statistics={"status": "started"},
    )
    record.write(request.destination, EVALUATION_MANIFEST)

    progress(f"[evaluate] embeddings: {request.features}")
    progress(f"[evaluate] prepared configuration: {prepared}")
    progress(f"[evaluate] running: {' '.join(command)}")

    try:
        with (request.destination / DOWNSTREAM_LOG).open("w", encoding="utf-8") as log, \
                subprocess.Popen(command, cwd=request.model.project_dir, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace") as child:
            for line in child.stdout:
                log.write(line)
                progress(line.rstrip("\n"))
    except OSError:
        record.statistics = {"status": "launch_failed"}
        record.write(request.destination, EVALUATION_MANIFEST)
        raise
    record.statistics = {
        "status": "finished" if child.returncode == 0 else "failed",
        "returncode": child.returncode,
    }
    record.write(request.destination, EVALUATION_MANIFEST)
    if child.returncode != 0:
        raise RuntimeError(
            f"the downstream model exited with code {child.returncode}. "
            f"Configuration used: {prepared}; its output is in {DOWNSTREAM_LOG}"
        )

    progress(f"[evaluate] finished -> {request.destination}")
    return request.destination


def _recorded_stride(features: Path) -> int | None:
    try:
        manifest = read_manifest(features)
    except FileNotFoundError:
        return None
    value = manifest.get("settings", {}).get("stride")
    return int(value) if value is not None else None
