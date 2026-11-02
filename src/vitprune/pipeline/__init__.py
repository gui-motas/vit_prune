"""Application glue: video decoding, checkpoint loading, extraction and evaluation.

Everything here is specific to running the method against real files and real
downstream models, rather than to the method itself. It needs the ``pipeline`` extra:

    pip install "vitprune[pipeline]"

:mod:`vitprune.core` does not depend on any of it.
"""

from vitprune.pipeline.calibration import calibrate_videos, read_policy, write_policy
from vitprune.pipeline.checkpoint import (
    LoadReport,
    disable_activation_checkpointing,
    install_weights,
    load_videomae,
    read_weights,
)
from vitprune.pipeline.evaluation import DownstreamModel, EvaluationRequest, evaluate
from vitprune.pipeline.extraction import (
    encode_clips,
    encode_clips_with_budget,
    extract_folder,
)
from vitprune.pipeline.provenance import Manifest, SourceFile, read_manifest
from vitprune.pipeline.video import (
    clip_span,
    find_videos,
    gather_clips,
    iter_clips,
    read_frames,
)

__all__ = [
    "DownstreamModel",
    "EvaluationRequest",
    "LoadReport",
    "Manifest",
    "SourceFile",
    "calibrate_videos",
    "clip_span",
    "disable_activation_checkpointing",
    "encode_clips",
    "encode_clips_with_budget",
    "evaluate",
    "extract_folder",
    "find_videos",
    "gather_clips",
    "install_weights",
    "iter_clips",
    "load_videomae",
    "read_frames",
    "read_manifest",
    "read_policy",
    "read_weights",
    "write_policy",
]
