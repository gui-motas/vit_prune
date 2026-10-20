"""The score that decides which visual tokens survive pruning.

Every token gets a score combining two independent criteria:

1. **Uniqueness** -- how far the token sits from the rest of the scene. A patch of
   ceiling resembles hundreds of others and adds nothing. A hand holding an object
   resembles nothing around it.

2. **Motion** -- how much the content at that position changed between consecutive
   temporal slices. Movement is what carries evidence of an action.

The two are combined through a weighted geometric mean rather than a sum. The
difference matters: with a sum, a token scoring high on one criterion and zero on the
other would still pass. With a product it has to do reasonably well on both, which
rules out what is merely unusual but static, and what merely flickers without being
relevant.

``motion_weight`` sets the balance: 0 uses uniqueness alone, 1 uses motion alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPSILON = 1e-6


@torch.no_grad()
def score_tokens(
    tokens: torch.Tensor,
    motion_weight: float = 0.5,
    spatial_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score every visual token in a clip.

    Args:
        tokens: token embeddings shaped
            ``[batch, temporal_slices, tokens_per_slice, embed_dim]``.
        motion_weight: between 0 and 1, the weight of motion against uniqueness.
        spatial_prior: optional per-position multiplier shaped
            ``[tokens_per_slice]``, used to discourage regions of the frame that are
            salient but uninformative. See :mod:`vitprune.core.priors`.

    Returns:
        Scores shaped ``[batch, temporal_slices, tokens_per_slice]``. Higher means
        more worth keeping.

    The motion term assumes the token at position ``n`` of slice ``t`` corresponds to
    the same location as position ``n`` of slice ``t-1``. That correspondence holds
    only *before* any merge, since merging reorders tokens. Calling this after
    pruning produces numbers without meaning.
    """
    if not 0.0 <= motion_weight <= 1.0:
        raise ValueError(f"motion_weight must be in [0, 1], got {motion_weight}")
    if tokens.dim() != 4:
        raise ValueError(f"expected [batch, slices, tokens, embed_dim], got {tuple(tokens.shape)}")

    values = tokens.float()
    score = _combine(
        _normalise_per_slice(_uniqueness(values)),
        _normalise_per_slice(_motion(values)),
        motion_weight,
    )

    if spatial_prior is not None:
        score = _apply_prior(score, spatial_prior)
    return score


def _uniqueness(values: torch.Tensor) -> torch.Tensor:
    """How far each token departs from its context.

    Measured at two levels and summed: against the mean of the whole clip and
    against the mean of its temporal slice. This favours deviation at both levels,
    but the sum is not a logical requirement that both terms be large.
    """
    batch, _, _, dim = values.shape

    clip_mean = values.reshape(batch, -1, dim).mean(dim=1)
    against_clip = -F.cosine_similarity(values, clip_mean.view(batch, 1, 1, dim), dim=-1)

    slice_mean = values.mean(dim=2, keepdim=True)
    against_slice = -F.cosine_similarity(values, slice_mean, dim=-1)

    return against_clip + against_slice


def _motion(values: torch.Tensor) -> torch.Tensor:
    """L2 norm of the embedding difference between consecutive slices.

    The first slice gets zero: there is no earlier slice to compare it against.
    """
    batch, slices, tokens, _ = values.shape
    out = torch.zeros(batch, slices, tokens, device=values.device, dtype=values.dtype)
    if slices > 1:
        out[:, 1:] = torch.norm(values[:, 1:] - values[:, :-1], p=2, dim=-1)
    return out


def _normalise_per_slice(raw: torch.Tensor) -> torch.Tensor:
    """Min-max normalisation within each temporal slice.

    The two criteria arrive in incomparable units: uniqueness is a cosine between -1
    and 1, motion is an unbounded distance. Without rescaling, whichever has the
    larger range would dominate.

    Normalising per slice rather than per clip keeps one busy slice from flattening
    the scores of all the others.
    """
    low = raw.amin(dim=-1, keepdim=True)
    high = raw.amax(dim=-1, keepdim=True)
    return (raw - low) / (high - low + EPSILON)


def _combine(uniqueness: torch.Tensor, motion: torch.Tensor, weight: float) -> torch.Tensor:
    """Weighted geometric mean of the two criteria.

    Computed in the log domain rather than by multiplying directly. Same result, but
    it avoids underflow when the values are small.
    """
    return torch.exp(
        (1.0 - weight) * torch.log(uniqueness + EPSILON) + weight * torch.log(motion + EPSILON)
    )


def _apply_prior(score: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    tokens = score.shape[-1]
    if prior.ndim != 1 or not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("spatial prior must be one-dimensional, finite and non-negative")
    if prior.numel() != tokens:
        raise ValueError(
            f"the spatial prior has {prior.numel()} positions but a slice holds {tokens} tokens"
        )
    return score * prior.view(1, 1, -1).to(score.device, score.dtype)
