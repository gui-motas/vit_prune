"""How compressible a clip is, measured cheaply where the budget is decided.

The budget asks how many tokens a clip can lose before its embedding drifts. Motion and
redundancy describe the clip as a whole; the probe asks the question directly, on a small
scale: if a handful of tokens had to stand for all the others, how far would each token
sit from its nearest stand-in?

The stand-ins are **anchors**, chosen per temporal slice and deterministically:

* half at fixed places, the centres of the cells of a coarse grid over the frame, so that
  every region has one;
* half where the motion-aware score is highest, the tokens a merge would keep first.

Each token's error is ``1 - max cos`` to the anchors of its slice. A clip whose tokens all
resemble one of a few anchors should merge with little loss; one whose tokens resemble
none of them should not. Whether that holds -- whether the error predicts the drift of the
final embedding -- is measured, not assumed: the probe is a cheap proxy, not the drift.

The cost is one cosine per token and anchor: ``slices x tokens x anchors x width``
multiply-adds, about 46 million for the Giant with 16 anchors, against hundreds of
billions for the blocks. No ``tokens x tokens`` matrix is ever formed.
"""

from __future__ import annotations

from math import isqrt
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vitprune.core.statistics import ClipStatistics, clip_statistics

#: The fixed anchors: one at the centre of each cell of a grid of these rows and columns.
PROBE_CELLS = (2, 4)
#: The anchors taken where the score is highest, besides the fixed ones.
PROBE_TOP = 8


class CompressibilityFeatures(NamedTuple):
    """What the budget sees of each clip, measured at the cut before any merge.

    Attributes:
        motion, redundancy: ``[batch]``, from :func:`vitprune.core.statistics.clip_statistics`.
        probe_mean, probe_p90: ``[batch]``, the mean and the 90th percentile of the error of
            every token of the clip against its nearest anchor. NaN when not measured.
    """

    motion: torch.Tensor
    redundancy: torch.Tensor
    probe_mean: torch.Tensor
    probe_p90: torch.Tensor


def fixed_anchors(grid_size: int, cells: tuple[int, int] = PROBE_CELLS) -> torch.Tensor:
    """Raster positions of the centre of each cell, row by row."""
    rows, columns = cells
    if not (1 <= rows <= grid_size and 1 <= columns <= grid_size):
        raise ValueError(f"a {grid_size}x{grid_size} grid cannot hold {rows}x{columns} cells")
    height, width = grid_size // rows, grid_size // columns
    return torch.tensor([
        (row * height + height // 2) * grid_size + column * width + width // 2
        for row in range(rows) for column in range(columns)
    ])


@torch.no_grad()
def probe_errors(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    cells: tuple[int, int] = PROBE_CELLS,
    top: int = PROBE_TOP,
) -> torch.Tensor:
    """Each token's error against the nearest anchor of its slice.

    Args:
        tokens: ``[batch, slices, tokens, width]``, the tokens of a square grid per slice.
        scores: ``[batch, slices, tokens]``, the motion-aware score before any prior.
        cells: the grid of fixed anchors.
        top: how many anchors to take by score, besides the fixed ones.

    Returns:
        ``[batch, slices, tokens]`` in ``[0, 2]``; 1 for a zero token.
    """
    if tokens.dim() != 4 or scores.shape != tokens.shape[:3]:
        raise ValueError("expected tokens [batch, slices, tokens, width] and their scores")
    count = tokens.shape[2]
    side = isqrt(count)
    if side * side != count:
        raise ValueError(f"the {count} tokens of a slice do not form a square grid")
    fixed = fixed_anchors(side, cells).to(tokens.device)
    if not 0 <= top <= count - fixed.numel():
        raise ValueError(f"cannot take {top} anchors by score besides {fixed.numel()} fixed ones")

    batch, slices = tokens.shape[:2]
    best = scores.float().index_fill(-1, fixed, float("-inf")).topk(top, dim=-1).indices
    anchors_at = torch.cat([fixed.expand(batch, slices, -1), best], dim=-1)
    unit = F.normalize(tokens.float(), dim=-1)
    anchors = torch.gather(unit, 2, anchors_at.unsqueeze(-1).expand(-1, -1, -1, unit.shape[-1]))
    return 1.0 - torch.einsum("bspd,bsqd->bspq", unit, anchors).amax(dim=-1)


@torch.no_grad()
def compressibility_features(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    statistics: ClipStatistics | None = None,
    cells: tuple[int, int] = PROBE_CELLS,
    top: int = PROBE_TOP,
    probe: bool = True,
) -> CompressibilityFeatures:
    """The features of each clip, from the grid at the cut.

    Args:
        tokens, scores, cells, top: as for :func:`probe_errors`.
        statistics: the clips' statistics, when already measured.
        probe: ``False`` measures only the statistics and leaves the probe fields NaN.
    """
    if statistics is None:
        statistics = clip_statistics(tokens)
    if probe:
        errors = probe_errors(tokens, scores, cells, top).flatten(1)
        mean, p90 = errors.mean(dim=1), torch.quantile(errors, 0.9, dim=1)
    else:
        mean = p90 = torch.full((tokens.shape[0],), float("nan"), device=tokens.device)
    return CompressibilityFeatures(statistics.motion, statistics.redundancy, mean, p90)
