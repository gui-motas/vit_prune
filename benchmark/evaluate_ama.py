"""Run the AMA temporal localizer of the AI City solution on each variant's features.

Needs the features that ``run_inference.py`` writes. Run it with the Python environment
of the AMA project: ``vitprune evaluate`` runs its ``eval.py`` under the same
interpreter, with this folder on its path ahead of the installed packages, so that
``nms_1d_cpu.py`` stands in for the compiled extension on every machine. Writes, under
``results``:

* ``ama/labels.json`` -- the annotations of the evaluated videos, with the ``subset``
  field the AMA loader needs;
* ``ama/<variant>/`` -- the prepared configuration, the predictions, the AMA's output in
  ``downstream.log`` and the evaluation record;
* ``ama.json`` -- mAP per tIoU and its average, per variant.

The AMA's configuration is copied, never edited: only the feature folder, the stride
and the annotation file change. A variant already evaluated is refused; move
``results/ama`` out to run it again.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from common import FRAME_STEP, HERE, RESULTS, SETTINGS, VARIANTS, annotations

from vitprune.pipeline.evaluation import (
    DOWNSTREAM_LOG,
    DownstreamModel,
    EvaluationRequest,
    evaluate,
)

PER_TIOU = re.compile(r"tIoU = ([\d.]+): mAP = ([\d.]+|nan) %")
AVERAGE = re.compile(r"Average Map is :\s*([\d.]+|nan) %")


def labels() -> Path:
    """The annotations of the evaluated videos only. The AMA skips a video with no
    features, but its scorer would still count that video's actions as missed."""
    database = annotations()
    database["database"] = {
        stem: video for stem, video in database["database"].items()
        if stem in SETTINGS["evaluate"]
    }
    for video in database["database"].values():
        video["subset"] = "validation"
    path = RESULTS / "ama" / "labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(database, indent=2), encoding="utf-8")
    return path


def main() -> None:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(HERE), os.environ.get("PYTHONPATH")])
    )
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
        evaluate(request)
        log = (request.destination / DOWNSTREAM_LOG).read_text(encoding="utf-8")
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
