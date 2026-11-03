"""Run the AMA temporal localizer on feature files, with no ground truth.

Started by ``vitprune benchmark ama`` under the AMA's own interpreter, from the AMA's
folder. Self-contained on purpose: it imports the standard library, numpy, torch and the
AMA's own ``libs``, never vitprune, which the AMA's environment need not have.

The model is built, loaded and fed as the AMA's ``eval.py`` does, minus the dataset and the
evaluator, which exist to compare the detections with annotations: the detections are the
result here. Usage::

    python ama_infer.py CONFIG WEIGHTS JOB

``JOB`` is the JSON ``vitprune benchmark ama`` writes: the stride of the features, and per
variant the folder of its features and where to write its ``predictions.csv``; per video
the frame rate and the frame count; and, when given, the annotations of those videos, which
the AMA's own evaluator scores the detections against, writing each variant's ``map.json``.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

COLUMNS = ["video-id", "t-start", "t-end", "label", "score"]
#: The tIoUs the AMA's evaluator averages over, as in its own evaluation.
TIOUS = [0.1, 0.2, 0.3, 0.4, 0.5]
#: The line the parent looks for, to record which interpreter the AMA ran under.
ENVIRONMENT = "ama-environment: "


def load_model(config: str, weights: str):
    """The configuration, the model in eval mode and the device it runs on."""
    from libs.core import load_config
    from libs.modeling import make_meta_arch
    from libs.utils import fix_random_seed

    cfg = load_config(config)
    fix_random_seed(0, include_cuda=True)
    device = torch.device(cfg["devices"][0])
    model = make_meta_arch(cfg["model_name"], **cfg["model"]).to(device)
    # The checkpoint comes from a DataParallel model: its keys start with "module.".
    state = torch.load(weights, map_location="cpu")["state_dict_ema"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"the AMA weights do not match the model: missing {missing}, unexpected {unexpected}"
        )
    return cfg, model.eval(), device


def time_base(cfg: dict, video: dict) -> float:
    """The frame rate the AMA converts feature steps to seconds with: its configuration's,
    as its dataset does, or the video's when the configuration names none."""
    return cfg["dataset"].get("default_fps") or video["fps"]


def detect(model, cfg: dict, stride: int, stem: str, video: dict, feats: np.ndarray) -> list:
    """The detections of one video, as rows of ``COLUMNS``, times in seconds."""
    dataset = cfg["dataset"]
    fps = time_base(cfg, video)
    item = {
        "video_id": stem,
        "feats": torch.from_numpy(np.ascontiguousarray(feats.astype(np.float32).T)),  # [C, T]
        "segments": None,
        "labels": None,
        "fps": fps,
        "duration": video["frames"] / fps,
        "feat_stride": stride,
        "feat_num_frames": dataset["num_frames"],
        "segmentation_labels": None,
    }
    with torch.no_grad():
        found = model([item], is_training=False)[0]
    return [
        [stem, float(start), float(end), int(label), float(score)]
        for (start, end), label, score in zip(
            found["segments"].tolist(), found["labels"].tolist(), found["scores"].tolist(),
            strict=True,
        )
    ]


def score(labels: str, rows: list, destination: Path) -> None:
    """The AMA's mAP per tIoU against the annotations, and its average; zero without any
    detection, which the evaluator does not take."""
    if rows:
        import pandas as pd
        from libs.utils import ANETdetection

        evaluator = ANETdetection(labels, split="validation", tiou_thresholds=np.array(TIOUS))
        found, average, _ = evaluator.evaluate(pd.DataFrame(rows, columns=COLUMNS),
                                               str(destination / "evaluated.csv"), verbose=False)
        found, average = [float(value) for value in found], float(average)
    else:
        found, average = [0.0] * len(TIOUS), 0.0
    (destination / "map.json").write_text(
        json.dumps({"tiou": TIOUS, "map": found, "average": average}), encoding="utf-8")


def main(argv: list[str]) -> int:
    config, weights, job_path = argv
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    cfg, model, _ = load_model(config, weights)
    environment = {"executable": sys.executable, "python": platform.python_version(),
                   "torch": torch.__version__}
    print(ENVIRONMENT + json.dumps(environment), flush=True)
    for stem, video in job["videos"].items():
        fps = time_base(cfg, video)
        if abs(video["fps"] - fps) > 1e-3:
            print(f"warning: {stem} is {video['fps']:g} fps but the AMA reads time at "
                  f"{fps:g} fps; its detection times are frames over {fps:g}", flush=True)
    for name, variant in job["variants"].items():
        rows = []
        for stem, video in job["videos"].items():
            with np.load(Path(variant["features"]) / f"{stem}.npz") as data:
                rows += detect(model, cfg, job["stride"], stem, video, data["feats"])
        destination = Path(variant["destination"])
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"[{name}] {len(rows)} detections", flush=True)
        if job.get("labels"):
            score(job["labels"], rows, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
