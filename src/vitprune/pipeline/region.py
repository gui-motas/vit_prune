"""The driver region: where a person detector found the driver in calibration footage.

The region is the set of grid positions the driver occupied at least once, joined over
the middle frame of every calibration clip. A plan carrying it as its spatial prior
fills each slice's slots from inside the region first; tokens outside only take the
slots left over, and still merge into kept ones. The backbone input is untouched, so a
model trained on the whole frame still sees the whole frame.

The detector runs here, once per camera setup, never during extraction: any COCO model
Ultralytics loads, such as YOLO26s. The rule follows the AI City preprocessing the
downstream model came from: COCO class 0 (person), confidence above 0.5, the largest
box in the frame.

Joining rather than counting is deliberate. A frequency would rank down the places the
driver visits rarely, which is where the rare actions happen. A union has no parameter
and errs on the safe side: a spurious box can only widen the region.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vitprune.core.geometry import ClipGeometry
from vitprune.core.priors import box_prior
from vitprune.pipeline.provenance import SourceFile
from vitprune.pipeline.video import (
    _clip_starts,
    clip_span,
    find_videos,
    frame_count,
    read_frames,
    spread_starts,
)

#: COCO class of a person, the only one the AI City preprocessing keeps.
PERSON = 0
#: The AI City preprocessing keeps detections strictly above this confidence.
CONFIDENCE = 0.5

#: From an RGB frame ``[height, width, 3]`` to rows of ``x1, y1, x2, y2, confidence,
#: class`` in pixels.
Detector = Callable[[np.ndarray], np.ndarray]


def largest_person(
    detections: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """The largest confident person box, as fractions of the frame, or ``None``."""
    best, best_area = None, 0.0
    for x1, y1, x2, y2, confidence, label in np.asarray(detections, float).reshape(-1, 6):
        if int(label) != PERSON or not confidence > CONFIDENCE:
            continue
        box = np.clip([x1 / width, y1 / height, x2 / width, y2 / height], 0.0, 1.0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best, best_area = tuple(float(value) for value in box), area
    return best


def calibrate_region(
    videos: str | Path,
    detector: Detector,
    geometry: ClipGeometry,
    stride: int = 16,
    frame_step: int = 2,
    per_video: int | None = None,
    detector_record: dict[str, Any] | None = None,
    progress=print,
) -> dict[str, Any]:
    """Detect the driver in the middle frame of every clip and join the boxes.

    Args:
        videos: a folder or a single video file, not part of the evaluation.
        detector: see :data:`Detector`.
        geometry: the backbone geometry, which sets the grid.
        stride, frame_step: how clips are sampled; use the extraction values.
        per_video: clips taken from each video, spread evenly over it; every clip
            when ``None``.
        detector_record: identity of the detector, stored with the region.
        progress: where to send progress output.

    Returns:
        The content of a region file, ready for :func:`write_region`.

    Videos are read a frame at a time, so their length is not limited by memory.
    """
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")
    if stride < 1 or frame_step < 1:
        raise ValueError("stride and frame_step must be at least 1")
    if per_video is not None and per_video < 1:
        raise ValueError("clips per video must be positive")

    span = clip_span(geometry, frame_step)
    boxes, sources, total = [], [], 0
    for video in found:
        count = frame_count(video)
        starts = (spread_starts(count, per_video, geometry, stride, frame_step) if per_video
                  else _clip_starts(count, span, stride))
        middles = {min(start + span // 2, count - 1) for start in starts}
        seen = detected = 0
        for index, frame in enumerate(read_frames(video)):
            if index in middles:
                seen += 1
                box = largest_person(detector(frame), frame.shape[1], frame.shape[0])
                if box is not None:
                    boxes.append(box)
                    detected += 1
            if index >= max(middles):
                break
        total += seen
        sources.append(
            {"source": asdict(SourceFile.of(video)), "clips": seen, "detected": detected}
        )
        progress(f"[region] {video.name}: driver in {detected} of {seen} clips")

    if not boxes:
        raise ValueError("no person was detected in the calibration clips; there is no region")
    prior = box_prior(geometry.grid_size, boxes)
    return {
        "region": {"grid_size": geometry.grid_size, "prior": prior.tolist()},
        "calibration": {
            "clips": total,
            "clips_per_video": per_video,
            "detected": len(boxes),
            "area": float(prior.mean()),
            "videos": sources,
            "stride": stride,
            "frame_step": frame_step,
            "rule": {"coco_class": PERSON, "confidence_above": CONFIDENCE, "box": "largest"},
            "detector": detector_record,
        },
    }


def write_region(path: str | Path, content: dict[str, Any]) -> Path:
    """Write a region file, refusing to replace one: like a policy, it is frozen once used."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite the region at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return path


def read_region(path: str | Path, geometry: ClipGeometry) -> torch.Tensor:
    """Read a region file as a spatial prior, checked against the backbone's grid."""
    region = json.loads(Path(path).read_text(encoding="utf-8"))["region"]
    prior = torch.tensor(region["prior"], dtype=torch.float32)
    if region.get("grid_size") != geometry.grid_size or prior.numel() != geometry.tokens_per_slice:
        raise ValueError(
            f"the region was built for a grid of {region.get('grid_size')}, "
            f"the backbone has {geometry.grid_size}"
        )
    if not torch.all((prior == 0) | (prior == 1)) or not prior.any():
        raise ValueError("a region marks positions with 0 or 1 and holds at least one")
    return prior


def render_region(prior: torch.Tensor, grid_size: int) -> str:
    """The region as text, one grid row per line: ``#`` inside, ``.`` outside."""
    rows = prior.reshape(grid_size, grid_size).tolist()
    return "\n".join("".join("#" if cell else "." for cell in row) for row in rows)


def load_yolo(weights: str | Path, device: str = "cuda") -> tuple[Detector, dict[str, Any]]:
    """A COCO person detector through Ultralytics, such as ``yolo26s.pt``.

    Args:
        weights: the detector's checkpoint file.
        device: ``cuda``, ``cuda:N`` or ``cpu``.

    Returns:
        The detector and a record of its identity.
    """
    weights = Path(weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"detector weights not found: {weights}")
    try:
        import ultralytics
    except ImportError as error:
        raise RuntimeError(
            "the region command needs ultralytics: pip install 'vitprune[region]'"
        ) from error

    model = ultralytics.YOLO(str(weights))

    def detect(frame: np.ndarray) -> np.ndarray:
        # Ultralytics reads arrays the way OpenCV does, in BGR order.
        bgr = np.ascontiguousarray(frame[..., ::-1])
        return model.predict(bgr, device=device, verbose=False)[0].boxes.data.cpu().numpy()

    record = {
        "name": "ultralytics",
        "version": ultralytics.__version__,
        "weights": asdict(SourceFile.of(weights)),
    }
    return detect, record
