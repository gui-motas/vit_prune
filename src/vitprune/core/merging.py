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

from numbers import Integral
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

    Kept tokens come out in grid order, not score order, for predictable inspection.
    Absolute positions already travel with each token; a consistent permutation is
    not intrinsically invalid for permutation-equivariant attention and mean pooling.
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


@torch.no_grad()
def bipartite_merge(
    tokens: torch.Tensor, sizes: torch.Tensor, metric: torch.Tensor, keep: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge the strongest alternating-partition matches, preserving token mass."""
    if tokens.ndim != 3:
        raise ValueError("tokens must be [batch, tokens, embed_dim]")
    batch, count, dim = tokens.shape
    if sizes.shape != (batch, count, 1):
        raise ValueError("sizes must be [batch, tokens, 1]")
    if metric.ndim != 3 or metric.shape[:2] != (batch, count):
        raise ValueError("metric must be [batch, tokens, channels]")
    if (isinstance(keep, bool) or not isinstance(keep, Integral)
            or not max(1, count // 2) <= keep <= count):
        raise ValueError(f"keep must be between {max(1, count // 2)} and {count}")
    if keep == count:
        return tokens, sizes

    metric = F.normalize(metric.float(), dim=-1)
    similarity = metric[:, ::2] @ metric[:, 1::2].transpose(1, 2)
    best, destination = similarity.max(dim=-1)
    order = best.argsort(dim=-1, descending=True).unsqueeze(-1)
    unmerged, source = order[:, count - keep:], order[:, :count - keep]
    destination = destination.unsqueeze(-1).gather(1, source)

    weighted = tokens.float() * sizes.float()
    a, b = weighted[:, ::2], weighted[:, 1::2]
    sa, sb = sizes.float()[:, ::2], sizes.float()[:, 1::2]
    b = b.scatter_add(1, destination.expand(-1, -1, dim),
                      a.gather(1, source.expand(-1, -1, dim)))
    sb = sb.scatter_add(1, destination, sa.gather(1, source))
    weighted = torch.cat((a.gather(1, unmerged.expand(-1, -1, dim)), b), dim=1)
    mass = torch.cat((sa.gather(1, unmerged), sb), dim=1)
    return (weighted / mass).to(tokens.dtype), mass.to(sizes.dtype)


@torch.no_grad()
def merge_error(
    tokens: torch.Tensor, scores: torch.Tensor, keep: int, merged: MergedTokens
) -> torch.Tensor:
    """How badly each original token is represented after a merge.

    ``1 - cos(token, representative)``: a kept token is represented by its own merged
    value, a pruned one by the merged value of the kept token it went into. This is the
    loss per token. The difference between the pooled means before and after the merge
    is zero by construction and measures nothing.

    Args:
        tokens, scores, keep: what :func:`merge_tokens` was given.
        merged: what it returned.

    Returns:
        ``[batch, slices, tokens]``.
    """
    kept_at = scores.argsort(dim=-1, descending=True)[..., :keep].sort(dim=-1).values
    destination = _nearest_kept_token(tokens, _take(tokens, kept_at))
    # A kept token stands for itself, even when another kept token points the same way.
    own = torch.arange(keep, device=tokens.device).expand_as(kept_at)
    destination = destination.scatter(2, kept_at, own)
    return 1.0 - F.cosine_similarity(tokens, _take(merged.tokens, destination), dim=-1)


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
