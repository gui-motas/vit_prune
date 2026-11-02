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

from collections.abc import Iterable, Iterator
from itertools import islice
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


def read_frames(path: str | Path, start: int = 0) -> Iterator[np.ndarray]:
    """Yield the frames of a video file one at a time, in order, from frame ``start``.

    Each frame is ``[height, width, 3]`` in RGB, values from 0 to 255. Every machine
    decodes with OpenCV, so the same file gives the same frames everywhere.
    """
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    try:
        if start:
            # shortcut: OpenCV lands on the exact frame in constant-frame-rate files, as in
            # the AI City footage (checked frame by frame); decode from 0 for variable rate.
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def frame_count(path: str | Path) -> int:
    """The frame count the container declares, without decoding.

    Containers can be off by a few frames; use it for planning, not for clip starts.
    """
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def find_videos(folder: str | Path) -> list[Path]:
    """List the video files in a folder, in a fixed order.

    The order is deterministic so that two runs produce the same sequence and remain
    comparable.
    """
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def clip_span(geometry: ClipGeometry | None = None, frame_step: int = 2) -> int:
    """How many video frames one clip spans.

    With 16 frames and a step of 2, a clip spans 32 video frames even though only 16
    are processed.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    return geometry.frames_per_clip * frame_step


def iter_clips(
    path: str | Path,
    geometry: ClipGeometry | None = None,
    stride: int = 16,
    frame_step: int = 2,
    starts: Iterable[int] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(start, clip)`` in order, decoding each frame once.

    A clip starts every ``stride`` frames and takes ``frames_per_clip`` frames
    ``frame_step`` apart, normalised, as ``[3, frames_per_clip, side, side]``. Only
    the frames the next clips need are held. A video shorter than one clip still gives
    one, its last frame repeated to fill: dropping it would lose the short examples,
    which are often the difficult ones.

    ``starts`` keeps only those clips, and the reader then jumps over the frames none
    of them covers; a start that is not a clip of this video is never yielded.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    _check_sampling(stride, frame_step)
    span = clip_span(geometry, frame_step)
    offsets = [frame_step * k for k in range(geometry.frames_per_clip)]
    wanted = None if starts is None else {s for s in starts if s >= 0 and s % stride == 0}
    if wanted is not None and not wanted:
        return

    def kept(start: int) -> bool:
        return start % stride == 0 and (wanted is None or start in wanted)

    def needed(index: int) -> bool:
        first = max(0, index - offsets[-1])
        return any(kept(start) and (index - start) % frame_step == 0
                   for start in range(-(-first // stride) * stride, index + 1, stride))

    frames: dict[int, torch.Tensor] = {}
    count, last = 0, None
    for index, frame in _indexed_frames(path, wanted, span):
        count, last = index + 1, frame
        if needed(index):
            frames[index] = _normalise(frame[None], geometry.image_size)[0]
        start = index + 1 - span
        if start >= 0 and kept(start):
            yield start, torch.stack([frames[start + offset] for offset in offsets], dim=1)
        oldest = -(-max(0, index + 2 - span) // stride) * stride
        for done in [f for f in frames if f < oldest]:
            del frames[done]
    if count == 0:
        if wanted is None or 0 in wanted:
            raise ValueError("the video has no frames")
        return
    if count < span and kept(0):
        frames[count - 1] = _normalise(last[None], geometry.image_size)[0]
        yield 0, torch.stack([frames[min(offset, count - 1)] for offset in offsets], dim=1)


def _indexed_frames(path: str | Path, wanted: set[int] | None, span: int):
    """``(index, frame)`` for every frame, or for those the wanted clips cover, jumping
    over the gaps between them."""
    if wanted is None:
        yield from enumerate(read_frames(path))
        return
    runs: list[list[int]] = []
    for start in sorted(wanted):
        if runs and start <= runs[-1][1]:
            runs[-1][1] = start + span
        else:
            runs.append([start, start + span])
    for first, end in runs:
        yield from enumerate(islice(read_frames(path, first), end - first), start=first)


def spread_starts(
    total_frames: int,
    count: int,
    geometry: ClipGeometry | None = None,
    stride: int = 16,
    frame_step: int = 2,
) -> list[int]:
    """``count`` clip starts spread evenly over a video, or all of them if fewer."""
    geometry = geometry or ClipGeometry.videomae_giant()
    return _spread(_clip_starts(total_frames, clip_span(geometry, frame_step), stride), count)


def gather_clips(
    videos: str | Path,
    geometry: ClipGeometry,
    stride: int,
    frame_step: int,
    count: int,
) -> torch.Tensor:
    """``count`` clips spread evenly over a folder of videos, in a fixed order.

    The first clips of a folder are a few seconds of one video, not a sample of it.
    """
    found = find_videos(videos)
    if not found:
        raise FileNotFoundError(f"no video found in {videos}")
    span = clip_span(geometry, frame_step)
    every = [(video, start) for video in found
             for start in _clip_starts(frame_count(video), span, stride)]
    picked = _spread(every, count)

    gathered = []
    for video in found:
        starts = [start for source, start in picked if source == video]
        if starts:
            clips = iter_clips(video, geometry, stride, frame_step, starts)
            gathered += [clip for _, clip in clips]
    return torch.stack(gathered)


def _spread(items: list, count: int) -> list:
    """``count`` items spread evenly over a list, the first and the last included."""
    if count < 1:
        raise ValueError("the clip count must be positive")
    return sorted({items[round(i)] for i in np.linspace(0, len(items) - 1, count)})


def _check_sampling(stride: int, frame_step: int) -> None:
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if frame_step < 1:
        raise ValueError(f"frame_step must be at least 1, got {frame_step}")


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


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "reading video requires opencv-python: pip install 'vitprune[pipeline]'"
        ) from error
    return cv2
