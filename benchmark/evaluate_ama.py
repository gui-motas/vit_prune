"""Run the AMA temporal localizer of the AI City solution on each variant's features.

Needs the features that ``run_inference.py`` writes with ``"clips": "all"``. Run it with
the Python environment of the AMA project: its ``eval.py`` runs under the same
interpreter. Writes, under ``results``:

* ``ama/labels.json`` -- the annotations with the ``subset`` field the AMA loader needs;
* ``ama/<variant>/`` -- the prepared configuration, the predictions and the AMA log;
* ``ama.json`` -- mAP per tIoU and its average, per variant.

The AMA's configuration is copied, never edited: only the feature folder, the stride
and the annotation file change, as ``vitprune evaluate`` does.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from common import FRAME_STEP, RESULTS, SETTINGS, annotations

from vitprune.pipeline.evaluation import (
    DownstreamModel,
    EvaluationRequest,
    build_command,
    prepare_config,
)

VARIANTS = ("denso", "vitprune", "vitprune_mascara_antiga", "vitprune_regiao")
PER_TIOU = re.compile(r"tIoU = ([\d.]+): mAP = ([\d.]+|nan) %")
AVERAGE = re.compile(r"Average Map is :\s*([\d.]+|nan) %")


def labels() -> Path:
    database = annotations()
    for video in database["database"].values():
        video["subset"] = "validation"
    path = RESULTS / "ama" / "labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(database, indent=2), encoding="utf-8")
    return path


def main() -> None:
    model = DownstreamModel(
        project_dir=SETTINGS["ama_project"],
        config=SETTINGS["ama_config"],
        weights=SETTINGS["ama_weights"],
    )
    annotated = labels()
    found = {}
    for variant in VARIANTS:
        request = EvaluationRequest(
            features=RESULTS / "features" / variant,
            model=model,
            destination=RESULTS / "ama" / variant,
            labels=annotated,
            frame_step=FRAME_STEP,
        )
        command = build_command(request, prepare_config(request))
        print(f"[{variant}] {' '.join(command)}", flush=True)
        finished = subprocess.run(
            command, cwd=model.project_dir, capture_output=True, text=True, check=False
        )
        log = finished.stdout + finished.stderr
        (request.destination / "ama.log").write_text(log, encoding="utf-8")
        if finished.returncode != 0:
            log_path = request.destination / "ama.log"
            raise RuntimeError(f"the AMA failed on {variant}; see {log_path}")
        average = AVERAGE.search(log)
        found[variant] = {
            "map": {tiou: float(value) for tiou, value in PER_TIOU.findall(log)},
            "average": float(average.group(1)) if average else None,
            "predictions": str(request.destination / "predictions.csv"),
        }
        print(f"[{variant}] average mAP {found[variant]['average']}", flush=True)
    (RESULTS / "ama.json").write_text(json.dumps(found, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
