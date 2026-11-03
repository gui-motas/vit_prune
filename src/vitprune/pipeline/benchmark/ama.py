"""Run the AMA temporal localizer of the AI City solution on each variant's features.

Needs the features that ``vitprune benchmark run`` writes. The AMA is a stage of the
pipeline, not an evaluator: it turns the features into detections, and the report compares
each variant's detections with the dense ones. No annotation is needed and the AMA's
``eval.py`` is not used.

The model runs in ``ama_infer.py``, a script started as a subprocess from the AMA's folder
under ``python``: the interpreter of the AMA's environment, which need not be the one
running vitprune. The ``ama_compat`` folder goes first on its path, so ``nms_1d_cpu.py``
stands in for the compiled extension on every machine. Writes, under the results folder:

* ``ama/job.json`` -- what the script was given;
* ``ama/downstream.log`` -- everything the script and the AMA printed;
* ``ama/<variant>/predictions.csv`` -- the detections: ``video-id, t-start, t-end, label,
  score``, in seconds;
* ``ama/<variant>/ama-manifest.json`` -- the AMA's configuration and weights and the
  features it read.

A variant already run is refused: move the results' ``ama`` folder out to run it again.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from vitprune.pipeline.benchmark.settings import VARIANTS, RunRecord
from vitprune.pipeline.provenance import Manifest, SourceFile

HERE = Path(__file__).resolve().parent
#: Holds ``nms_1d_cpu.py``.
COMPAT = HERE / "ama_compat"
SCRIPT = HERE / "ama_infer.py"
PREDICTIONS = "predictions.csv"
MANIFEST = "ama-manifest.json"
LOG = "downstream.log"
#: What ``ama_infer.py`` prints before its JSON of the interpreter it runs under.
ENVIRONMENT = "ama-environment: "


def run_ama(out: str | Path, project: str | Path, config: str | Path, weights: str | Path,
            python: str | None = None, progress=print) -> Path:
    """Run the AMA on every variant's features and write its detections.

    Args:
        out: the results folder of ``vitprune benchmark run``.
        project: the AMA's folder, the one holding ``libs``.
        config, weights: the AMA's configuration and trained weights.
        python: the interpreter of the AMA's environment; the current one when ``None``.
        progress: where to send what the AMA prints.

    Returns:
        The folder the detections were written to.
    """
    out, project, config, weights = (Path(p).resolve() for p in (out, project, config, weights))
    record = RunRecord.read(out)
    for path, what in ((project / "libs", "the AMA's libs folder"),
                       (config, "the AMA's configuration"), (weights, "the AMA's weights")):
        if not path.exists():
            raise FileNotFoundError(f"{what} not found: {path}")
    absent = [out / "features" / variant / f"{stem}.npz" for variant in VARIANTS
              for stem in record.stems if not (out / "features" / variant / f"{stem}.npz").exists()]
    if absent:
        raise FileNotFoundError(f"features not found, 'vitprune benchmark run' first: {absent}")
    folder = out / "ama"
    done = [variant for variant in VARIANTS if (folder / variant / PREDICTIONS).exists()]
    if done:
        raise FileExistsError(
            f"refusing to overwrite the AMA's detections for {done} in {folder}; "
            "move that folder out to run again"
        )

    stride = record.settings["stride"]
    job = {
        "stride": stride,
        "variants": {variant: {"features": str(out / "features" / variant),
                               "destination": str(folder / variant)} for variant in VARIANTS},
        "videos": {stem: {"fps": record.video(stem)["fps"], "frames": record.video(stem)["frames"]}
                   for stem in record.stems},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")

    paths = [str(COMPAT), str(project), os.environ.get("PYTHONPATH")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))}
    command = [python or sys.executable, str(SCRIPT), str(config), str(weights),
               str(folder / "job.json")]
    log = folder / LOG
    environment = {}
    with log.open("w", encoding="utf-8") as stream, \
            subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace") as child:
        for line in child.stdout:
            if line.startswith(ENVIRONMENT):
                environment = json.loads(line[len(ENVIRONMENT):])
            stream.write(line)
            progress(line.rstrip("\n"))
    if child.returncode != 0:
        raise RuntimeError(
            f"the AMA exited with code {child.returncode}; its output is in {log}"
        )

    for variant in VARIANTS:
        with (folder / variant / PREDICTIONS).open(encoding="utf-8") as stream:
            detections = sum(1 for _ in stream) - 1  # the header
        features = {stem: asdict(SourceFile.of(out / "features" / variant / f"{stem}.npz"))
                    for stem in record.stems}
        Manifest(
            settings={"stage": "ama", "variant": variant, "stride": stride,
                      "features": str(out / "features" / variant), "feature_files": features,
                      "config": asdict(SourceFile.of(config))},
            weights=SourceFile.of(weights),
            # The AMA ran under its own interpreter, which is not the one writing this.
            environment=environment or {"executable": command[0]},
            statistics={"detections": detections},
        ).write(folder / variant, MANIFEST)
    return folder
