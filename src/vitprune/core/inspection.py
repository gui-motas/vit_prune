"""Exposes which visual tokens the method kept, and with what score.

The embedding coming out of the backbone says nothing about the path taken to
produce it. This module makes the decision visible: which grid positions survived,
what score each received, and what size each kept token ended up carrying.

That serves two purposes. For debugging: if the method is spending retention budget
on an irrelevant region, this is where it shows. For explaining: a figure showing
that the hand was kept and the ceiling pruned says more than a paragraph.

Inspection is cheap. Only the blocks up to the pruning layer have to run, against all
of them for a full extraction.

## What the visualisation is not

Pruning does not happen on pixels. It happens on embeddings, after the backbone has
run two blocks of self-attention, in which information mixes across positions.

So the token at position (3, 7) no longer represents only that patch of the frame: it
has already absorbed context from the whole clip. Drawing it back onto the original
image is a useful approximation, not a faithful picture.

Put differently: the visualisation shows where the method chose to look, not what the
backbone sees after pruning. There is no "video after pruning" -- there is a selection
of positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TokenSelection:
    """What the method decided in one pruning step.

    Attributes:
        scores: score of every token, ``[batch, temporal_slices, tokens]``.
        kept: kept positions, ``[batch, temporal_slices, kept]``, in grid order:
            position ``n`` sits at row ``n // grid_size``, column ``n % grid_size``.
            Under an adaptive budget clips keep different counts; the shorter ones are
            padded with ``-1``.
        sizes: how many original tokens each kept token stands for,
            ``[batch, temporal_slices, kept]``, zero where ``kept`` is padding.
        grid_size: patches per side of the grid.
    """

    scores: torch.Tensor
    kept: torch.Tensor
    sizes: torch.Tensor
    grid_size: int

    @property
    def clips(self) -> int:
        return self.scores.shape[0]

    @property
    def temporal_slices(self) -> int:
        return self.scores.shape[1]

    @property
    def kept_per_slice(self) -> int:
        """The widest count kept per slice in this batch; see :meth:`kept_in`."""
        return self.kept.shape[2]

    @property
    def tokens_per_slice(self) -> int:
        return self.scores.shape[2]

    def kept_in(self, clip: int) -> int:
        """Tokens one clip kept in each of its slices."""
        return int((self.kept[clip, 0] >= 0).sum())

    def keep_mask(self, clip: int, slice_index: int) -> np.ndarray:
        """Boolean mask of the positions that survived.

        Returns:
            ``[grid_size, grid_size]``, laid out as the frame is.
        """
        flat = np.zeros(self.tokens_per_slice, dtype=bool)
        flat[self._positions(clip, slice_index)] = True
        return flat.reshape(self.grid_size, self.grid_size)

    def score_grid(self, clip: int, slice_index: int) -> np.ndarray:
        """Score of every position, laid out as the frame is."""
        return self.scores[clip, slice_index].cpu().numpy().reshape(self.grid_size, self.grid_size)

    def size_grid(self, clip: int, slice_index: int) -> np.ndarray:
        """Size of every kept token, laid out as the frame is.

        Pruned positions read zero. A large size marks a token that absorbed a lot of
        neighbours, which usually means a large uniform region.
        """
        flat = np.zeros(self.tokens_per_slice, dtype=np.float32)
        positions = self._positions(clip, slice_index)
        flat[positions] = self.sizes[clip, slice_index, : positions.size].cpu().numpy()
        return flat.reshape(self.grid_size, self.grid_size)

    def rows_share(self, clip: int, slice_index: int, rows: int) -> float:
        """Fraction of the retention budget spent on the first rows of the frame.

        Compare it against the share of screen area those rows occupy. A budget share
        well above the area share means the method is drawn to the top of the frame,
        which in a vehicle camera usually means the windscreen.
        """
        mask = self.keep_mask(clip, slice_index)
        return float(mask[:rows].sum()) / max(int(mask.sum()), 1)

    def describe(self) -> str:
        counts = sorted({self.kept_in(clip) for clip in range(self.clips)})
        keeping = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
        return (
            f"{self.clips} clips x {self.temporal_slices} temporal slices | "
            f"keeping {keeping} of {self.tokens_per_slice} per slice"
        )

    def _positions(self, clip: int, slice_index: int) -> np.ndarray:
        positions = self.kept[clip, slice_index].cpu().numpy()
        return positions[positions >= 0]
