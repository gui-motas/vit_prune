"""Paths, settings, variants and ground truth shared by the benchmark scripts.

The benchmark runs every clip of the videos listed in ``evaluate`` through four
variants: the dense VideoMAE, and ViT-Prune with no mask, with the fixed mask and with
the smart mask, the driver region. The policy and the region come from
``vitprune calibrate`` and ``vitprune region`` run on other videos of the same camera,
so nothing is ever judged on the footage it was fitted to.

Paths and run settings default to the development machine. Another machine overrides
any of them in ``benchmark/settings.json``, for example::

    {
      "videos": "D:/data/benchmark/videos",
      "policy": "D:/data/benchmark/policy.json",
      "region": "D:/data/benchmark/region.json",
      "weights": "D:/models/videomae-v2_finetune_aicity.pth",
      "model_source": "D:/AIcity2024-track3/feature_extraction",
      "ama_project": "D:/AMA",
      "ama_config": "D:/AMA/configs/aicity_ego_vitl.yaml",
      "ama_weights": "D:/AMA/ckpt/aicity_ego_vitl_ckpt/best_performance.pth.tar",
      "device": "cuda",
      "batch": 8
    }

``precision`` is ``off`` (fp32, as the AI City feature extraction ran), ``fp16`` or
``bf16``. ``evaluate`` lists the videos under evaluation; none of them may be among
the calibration videos.
"""

from __future__ import annotations

import json
from pathlib import Path

from vitprune.core import ClipGeometry
from vitprune.pipeline.video import _clip_starts, clip_span

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = HERE / "cache"

DEFAULTS = {
    "videos": str(HERE / "videos"),
    "policy": str(RESULTS / "policy.json"),
    "region": str(RESULTS / "region.json"),
    "weights": r"C:\vitprune-temp\videomae-v2_finetune_aicity.pth",
    "model_source": r"C:\Users\GSILV257\OneDrive - azureford\Documents\Summit 2025"
                    r"\AIcity2024-track3\feature_extraction",
    "ama_project": r"C:\Python\AMA",
    "ama_config": r"C:\Python\AMA\configs\aicity_ego_vitl.yaml",
    "ama_weights": r"C:\Python\AMA\ckpt\aicity_ego_vitl_ckpt\best_performance.pth.tar",
    "device": "cpu",
    "precision": "off",
    "batch": 1,
    "evaluate": ["Dashboard_user_id_13522_NoAudio_5", "Dashboard_user_id_13522_NoAudio_7"],
}


def _settings() -> dict:
    path = HERE / "settings.json"
    given = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    settings = {**DEFAULTS, **given}
    if settings["precision"] not in ("off", "fp16", "bf16"):
        raise ValueError("precision must be 'off', 'fp16' or 'bf16'")
    return settings


SETTINGS = _settings()
VIDEOS = Path(SETTINGS["videos"])
WEIGHTS = Path(SETTINGS["weights"])
MODEL_SOURCE = Path(SETTINGS["model_source"])
DEVICE = SETTINGS["device"]

STRIDE = 16
FRAME_STEP = 2
GEOMETRY = ClipGeometry.videomae_giant()
SPAN = clip_span(GEOMETRY, FRAME_STEP)

#: The pruned variants share one policy, so they keep the same tokens per clip; the
#: masks only change which ones.
VARIANTS = ("denso", "vitprune", "vitprune_mascara_fixa", "vitprune_mascara_inteligente")


def videos() -> list[Path]:
    """The videos under evaluation, listed in ``evaluate``."""
    found = sorted(p for p in VIDEOS.iterdir()
                   if p.suffix.lower() == ".mp4" and p.stem in SETTINGS["evaluate"])
    missing = set(SETTINGS["evaluate"]) - {p.stem for p in found}
    if missing:
        raise FileNotFoundError(f"evaluation videos not found in {VIDEOS}: {sorted(missing)}")
    return found


def annotations() -> dict:
    return json.loads((VIDEOS / "class-intervals.json").read_text(encoding="utf-8"))


def segments(stem: str) -> list[dict]:
    """Annotated actions of a video: label id, label, start and end in seconds."""
    return [
        {"label_id": a["label_id"], "label": a["label"], "start": a["segment"][0],
         "end": a["segment"][1]}
        for a in annotations()["database"][stem]["annotations"]
    ]


def clip_starts(total_frames: int) -> list[int]:
    """First frame of every clip, as the extraction pipeline samples them."""
    return _clip_starts(total_frames, SPAN, STRIDE)


def segment_of(start: int, fps: float, annotated: list[dict]) -> dict | None:
    """The action segment holding the whole clip, or ``None`` across a boundary or gap."""
    begin, end = start / fps, (start + SPAN) / fps
    for segment in annotated:
        if segment["start"] <= begin and end <= segment["end"]:
            return segment
    return None
