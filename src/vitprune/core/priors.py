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
"""

from __future__ import annotations

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


def region_prior(
    grid_size: int,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    weight: float = 0.15,
) -> torch.Tensor:
    """Lower the score of bands on any edge of the frame.

    More general than :func:`top_rows_prior`, for cameras mounted differently. In the
    driver-facing videos this project was built around, the window turned out to sit
    on the left edge rather than the top.

    Args:
        grid_size: patches per side of the grid.
        top, bottom, left, right: how many rows or columns to lower on each edge.
        weight: multiplier applied inside those bands.
    """
    for name, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if value < 0 or value > grid_size:
            raise ValueError(f"{name} must be between 0 and {grid_size}, got {value}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    mask = torch.ones(grid_size, grid_size)
    if top:
        mask[:top, :] = weight
    if bottom:
        mask[grid_size - bottom :, :] = weight
    if left:
        mask[:, :left] = weight
    if right:
        mask[:, grid_size - right :] = weight
    return mask.flatten()


def uniform_prior(grid_size: int) -> torch.Tensor:
    """A neutral prior. It exists so that "no correction" can be stated explicitly."""
    return torch.ones(grid_size * grid_size)
