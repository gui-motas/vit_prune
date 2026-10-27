"""Clip-level statistics that inform how many tokens a clip keeps.

The score in :mod:`vitprune.core.scoring` ranks tokens within a clip: it decides
*which* tokens survive. The statistics here describe a clip as a whole, so that a
budget can decide *how many* survive:

* **Redundancy** -- the mean cosine similarity of the clip's tokens to the clip mean
  and to their slice mean. A static, uniform scene scores high.
* **Motion** -- the mean L2 distance between the same position in consecutive
  temporal slices. A still clip scores zero.

Both are measured on the grid the score sees, at the pruning layer and before any
merge: merging breaks the position-to-position correspondence motion relies on.

Neither is normalised per slice. The score needs that normalisation to compare tokens
within a clip; applied here, it would erase exactly the differences between clips
that a budget has to see. Putting the values on a common scale is left to whatever
consumes them, against a set of reference clips.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class ClipStatistics(NamedTuple):
    """One value per clip for each statistic.

    Attributes:
        redundancy: ``[batch]``, in ``[-1, 1]``. Higher means more repetitive content.
        motion: ``[batch]``, non-negative. Zero for a still clip, and always zero for a
            single-slice geometry, which has no consecutive slices to compare.
    """

    redundancy: torch.Tensor
    motion: torch.Tensor


@torch.no_grad()
def clip_statistics(tokens: torch.Tensor) -> ClipStatistics:
    """Measure the redundancy and motion of each clip.

    Args:
        tokens: ``[batch, temporal_slices, tokens_per_slice, embed_dim]``, before any
            merge.

    Returns:
        A :class:`ClipStatistics` in single precision. Each clip's values depend only on
        that clip, never on the rest of the batch.
    """
    if tokens.dim() != 4:
        raise ValueError(f"expected [batch, slices, tokens, embed_dim], got {tuple(tokens.shape)}")

    values = tokens.float()
    batch, slices, _, dim = values.shape

    clip_mean = values.reshape(batch, -1, dim).mean(dim=1).view(batch, 1, 1, dim)
    slice_mean = values.mean(dim=2, keepdim=True)
    against_clip = F.cosine_similarity(values, clip_mean, dim=-1)
    against_slice = F.cosine_similarity(values, slice_mean, dim=-1)
    redundancy = ((against_clip + against_slice) / 2).mean(dim=(1, 2))

    if slices > 1:
        motion = torch.norm(values[:, 1:] - values[:, :-1], p=2, dim=-1).mean(dim=(1, 2))
    else:
        motion = values.new_zeros(batch)
    return ClipStatistics(redundancy, motion)
