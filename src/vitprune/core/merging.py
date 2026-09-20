"""Merging pruned tokens into kept ones, with size tracking.

The backbone produces a clip embedding by mean pooling over all its tokens. That sets
a trap: if pruning simply dropped most of them, the mean would change meaning. Only
the salient regions would remain, and the embedding would describe a crop rather than
the scene.

The way out is to merge rather than drop. Each pruned token is assigned to the most
similar kept token, which then carries a ``size`` equal to the number of tokens it
now stands for.

That makes an exact identity hold: size-weighted pooling over the surviving tokens
equals mean pooling over all the original ones. The clip is four times smaller and
the embedding is unchanged.

The identity holds *at the moment of the merge*. The remaining blocks keep
transforming the surviving tokens, and drift against the dense path accumulates along
the way. Measuring that drift is what :mod:`vitprune.bench.drift` is for; it cannot
be derived on paper.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class MergedTokens(NamedTuple):
    """The result of a merge.

    Attributes:
        tokens: embeddings of the kept tokens, shaped
            ``[batch, temporal_slices, kept, embed_dim]``.
        sizes: how many original tokens each kept token stands for, shaped
            ``[batch, temporal_slices, kept, 1]``. The sum is invariant.
    """

    tokens: torch.Tensor
    sizes: torch.Tensor


def initial_sizes(tokens: torch.Tensor) -> torch.Tensor:
    """Starting sizes: every token stands for itself and nothing more."""
    batch, slices, count, _ = tokens.shape
    return torch.ones(batch, slices, count, 1, device=tokens.device, dtype=tokens.dtype)


@torch.no_grad()
def merge_tokens(
    tokens: torch.Tensor,
    sizes: torch.Tensor,
    scores: torch.Tensor,
    keep: int,
) -> MergedTokens:
    """Keep the ``keep`` highest scoring tokens per slice and merge the rest into them.

    Args:
        tokens: ``[batch, slices, tokens, embed_dim]``.
        sizes: ``[batch, slices, tokens, 1]``, as produced by :func:`initial_sizes`.
        scores: ``[batch, slices, tokens]``, as produced by
            :func:`vitprune.core.scoring.score_tokens`.
        keep: how many tokens to keep per temporal slice.

    Returns:
        A :class:`MergedTokens`.

    Kept tokens come out in grid order, not score order. That matters because the
    positional embedding has already been added; reordering by score would scramble
    it.
    """
    _check_shapes(tokens, sizes, scores)
    total = tokens.shape[2]
    if not 1 <= keep <= total:
        raise ValueError(f"keep must be between 1 and {total}, got {keep}")

    ranking = scores.argsort(dim=-1, descending=True)
    kept_at = ranking[..., :keep].sort(dim=-1).values
    pruned_at = ranking[..., keep:]

    kept_tokens = _take(tokens, kept_at)
    kept_sizes = _take(sizes, kept_at)

    if pruned_at.shape[-1] == 0:
        return MergedTokens(kept_tokens, kept_sizes)

    pruned_tokens = _take(tokens, pruned_at)
    pruned_sizes = _take(sizes, pruned_at)
    destination = _nearest_kept_token(pruned_tokens, kept_tokens)

    dim = tokens.shape[-1]
    weighted_sum = (kept_tokens * kept_sizes).clone()
    size_sum = kept_sizes.clone()
    weighted_sum.scatter_add_(
        2,
        destination.unsqueeze(-1).expand(-1, -1, -1, dim),
        pruned_tokens * pruned_sizes,
    )
    size_sum.scatter_add_(2, destination.unsqueeze(-1), pruned_sizes)

    return MergedTokens(weighted_sum / size_sum.clamp(min=1e-6), size_sum)


def size_weighted_mean(tokens: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """Size-weighted mean pooling, producing the clip embedding.

    This is the function that makes pruning honest. Using plain mean pooling here
    would undo everything :func:`merge_tokens` set up.

    Args:
        tokens: ``[batch, tokens, embed_dim]``, already flattened over slices.
        sizes: ``[batch, tokens, 1]``, in the same order.

    Returns:
        ``[batch, embed_dim]``.
    """
    if tokens.dim() != 3 or sizes.dim() != 3:
        raise ValueError(
            "expected [batch, tokens, embed_dim] and [batch, tokens, 1], got "
            f"{tuple(tokens.shape)} and {tuple(sizes.shape)}"
        )
    if tokens.shape[:2] != sizes.shape[:2]:
        raise ValueError(
            f"tokens and sizes disagree on the count: "
            f"{tuple(tokens.shape[:2])} against {tuple(sizes.shape[:2])}"
        )
    return (tokens * sizes).sum(dim=1) / sizes.sum(dim=1).clamp(min=1e-6)


def _take(source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Select tokens at the given positions, preserving the other dimensions."""
    expanded = positions.unsqueeze(-1).expand(*positions.shape, source.shape[-1])
    return torch.gather(source, 2, expanded)


def _nearest_kept_token(pruned: torch.Tensor, kept: torch.Tensor) -> torch.Tensor:
    """For each pruned token, the kept token with the highest cosine similarity.

    Similarity ignores magnitude. Two ceiling tokens point the same way even if one
    is better lit, and that equivalence is what the merge exploits.
    """
    a = F.normalize(pruned.float(), dim=-1)
    b = F.normalize(kept.float(), dim=-1)
    return torch.einsum("btid,btjd->btij", a, b).argmax(dim=-1)


def _check_shapes(tokens: torch.Tensor, sizes: torch.Tensor, scores: torch.Tensor) -> None:
    if tokens.dim() != 4:
        raise ValueError(
            f"tokens must be [batch, slices, tokens, embed_dim], got {tuple(tokens.shape)}"
        )
    if sizes.shape != (*tokens.shape[:3], 1):
        raise ValueError(f"sizes must be {(*tokens.shape[:3], 1)}, got {tuple(sizes.shape)}")
    if scores.shape != tokens.shape[:3]:
        raise ValueError(f"scores must be {tuple(tokens.shape[:3])}, got {tuple(scores.shape)}")
