"""Per-position multipliers that adjust the score by where a token sits in the frame.

Both criteria in :mod:`vitprune.core.scoring` are blind to meaning. In a
vehicle-mounted camera that has a concrete side effect: the windscreen scores very
high, being at once the region of greatest motion and of greatest uniqueness. The
method spends its token budget looking out of the window, which is exactly where the
information it is after is not.

A per-position multiplier fixes that by lowering the score of known bands of the
frame, without any training.

Read this before using it: a prior encodes knowledge about where the camera was
installed, not a property of the method. If the videos are already cropped around the
subject, the top band may contain their head rather than the window. Use
:mod:`vitprune.core.inspection` to check against a real frame first.

:func:`box_prior` takes the band from data instead of by hand: the positions a person
detector found the subject in, across calibration frames.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def top_rows_prior(grid_size: int, rows: int, weight: float = 0.15) -> torch.Tensor:
    """Lower the score of the first rows of the patch grid.

    Args:
        grid_size: patches per side of the grid.
        rows: how many rows from the top to lower.
        weight: multiplier applied to scores in those rows. Near zero all but
            removes the region; 1 changes nothing.

    Returns:
        A flat vector of ``grid_size * grid_size`` entries, in the same order the
        backbone emits tokens: row by row, left to right.

    >>> top_rows_prior(4, rows=1, weight=0.5).tolist()
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    if rows < 0 or rows > grid_size:
        raise ValueError(f"rows must be between 0 and {grid_size}, got {rows}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    mask = torch.ones(grid_size, grid_size)
    mask[:rows, :] = weight
    return mask.flatten()


def box_prior(grid_size: int, boxes: Iterable[tuple[float, float, float, float]]) -> torch.Tensor:
    """Weight 1 where any of the boxes reaches, 0 elsewhere.

    A position counts as inside when a box overlaps any part of its patch, so the
    union never leaves out a hand that only grazed a cell. Positions at zero still
    take the slots left over; see :func:`vitprune.core.scoring.score_tokens`.

    Args:
        grid_size: patches per side of the grid.
        boxes: ``(x1, y1, x2, y2)`` as fractions of the frame width and height. The
            whole frame is resized to a square before patching, so fractions map
            straight onto the grid whatever the original aspect ratio.

    >>> box_prior(4, [(0.0, 0.0, 0.5, 0.25)]).reshape(4, 4)[:2].tolist()
    [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    """
    mask = torch.zeros(grid_size, grid_size)
    for x1, y1, x2, y2 in boxes:
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"a box must be a non-empty fraction of the frame: {(x1, y1, x2, y2)}")
        rows = slice(math.floor(y1 * grid_size), math.ceil(y2 * grid_size))
        columns = slice(math.floor(x1 * grid_size), math.ceil(x2 * grid_size))
        mask[rows, columns] = 1.0
    return mask.flatten()
