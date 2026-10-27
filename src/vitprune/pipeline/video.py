"""Turning a video file into the clips a backbone consumes.

The backbone does not see a whole video at once. It sees clips of a fixed number of
frames, one at a time, and returns one embedding per clip.

The distance between the start of one clip and the next is the **stride**. With a
stride of 8 and clips of 16 frames, half of each clip also appears in its neighbour.
That is repeated work: the same content is processed twice.

Doubling the stride to 16 removes the overlap and halves the number of clips without
touching the model. It is the cheapest saving available, and worth measuring before
anything else. The trade is lower temporal resolution, which a downstream model may
or may not absorb -- another thing to measure rather than assume.

**Frame step** is a separate axis. The backbone always receives the same number of
frames, but they need not be consecutive. With a step of 2, a clip takes every other
frame across twice the time span, at the same compute cost.

That default is 2 because it is how the embeddings a downstream model was trained on
were produced. Using 1 would yield embeddings describing a different time span from
the ones that model learned to interpret, and the resulting accuracy drop would look
like an effect of pruning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vitprune.core.geometry import ClipGeometry

#: Normalisation the backbone was trained with. Changing it moves the input outside
#: the range the model knows.
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)

VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


def read_video(path: str | Path) -> np.ndarray:
    """Read every frame of a video file.

    Returns:
        ``[frames, height, width, 3]`` in RGB, values from 0 to 255.

    Uses ``decord`` when available, since it is considerably faster, and falls back
    to OpenCV otherwise.
    """
    path = str(path)
    frames = _read_with_decord(path)
    return frames if frames is not None else _read_with_opencv(path)


def find_videos(folder: str | Path) -> list[Path]:
    """List the video files in a folder, in a fixed order.

    The order is deterministic so that two runs produce the same sequence and remain
    comparable.
    """
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def to_clips(
    frames: np.ndarray,
    geometry: ClipGeometry | None = None,
    stride: int = 16,
    frame_step: int = 2,
) -> torch.Tensor:
    """Split a frame sequence into the clips the backbone expects.

    Args:
        frames: ``[frames, height, width, 3]``, values from 0 to 255.
        geometry: defines frames per clip and image size.
        stride: frames between the start of consecutive clips.
        frame_step: take every ``frame_step`` frames within a clip.

    Returns:
        ``[clips, 3, frames_per_clip, side, side]``, already normalised.

    Videos shorter than one clip are not discarded: the last frame is repeated to
    fill. Dropping them would lose the short examples, which are often the difficult
    ones.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [frames, height, width, 3], got {tuple(frames.shape)}")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if frame_step < 1:
        raise ValueError(f"frame_step must be at least 1, got {frame_step}")

    total = frames.shape[0]
    if total == 0:
        raise ValueError("the video has no frames")

    normalised = _normalise(frames, geometry.image_size)
    span = clip_span(geometry, frame_step)
    starts = _clip_starts(total, span, stride)

    clips = []
    for start in starts:
        picked = [
            min(start + offset * frame_step, total - 1)
            for offset in range(geometry.frames_per_clip)
        ]
        clips.append(normalised[picked].permute(1, 0, 2, 3))
    return torch.stack(clips)


def clip_span(geometry: ClipGeometry | None = None, frame_step: int = 2) -> int:
    """How many video frames one clip spans.

    With 16 frames and a step of 2, a clip spans 32 video frames even though only 16
    are processed.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    return geometry.frames_per_clip * frame_step


def count_clips(
    total_frames: int,
    geometry: ClipGeometry | None = None,
    stride: int = 16,
    frame_step: int = 2,
) -> int:
    """How many clips a video yields, without reading it.

    Useful for estimating cost before spending machine time.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    return len(_clip_starts(total_frames, clip_span(geometry, frame_step), stride))


def gather_clips(
    videos: str | Path,
    geometry: ClipGeometry,
    stride: int,
    frame_step: int,
    count: int,
) -> torch.Tensor:
    """The first ``count`` clips of a folder of videos, in a fixed order."""
    if count < 1:
        raise ValueError("the clip count must be positive")
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")

    gathered, total = [], 0
    for video in found:
        gathered.append(to_clips(read_video(video), geometry, stride, frame_step))
        total += gathered[-1].shape[0]
        if total >= count:
            break
    return torch.cat(gathered)[:count]


def _clip_starts(total: int, span: int, stride: int) -> list[int]:
    if total < span:
        return [0]
    return list(range(0, total - span + 1, stride))


def _normalise(frames: np.ndarray, side: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    tensor = F.interpolate(tensor, size=(side, side), mode="bilinear", align_corners=False)
    mean = torch.tensor(CHANNEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CHANNEL_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _read_with_decord(path: str) -> np.ndarray | None:
    try:
        import decord
    except ImportError:
        return None
    return decord.VideoReader(path, num_threads=2)[:].asnumpy()


def _read_with_opencv(path: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "reading video requires decord or opencv-python: pip install 'vitprune[pipeline]'"
        ) from error

    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise RuntimeError(f"could not read any frame from {path}")
    return np.stack(frames)
