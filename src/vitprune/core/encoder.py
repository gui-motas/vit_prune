"""The full forward pass, from clip to embedding, with pruning in the middle.

This is the module the others exist to serve. It takes a clip and returns the
embedding that summarises it, having pruned most of the visual tokens along the way.

Control runs the other way round from what one might expect. The encoder does not
hook into the backbone; it drives it. ``backbone.forward()`` is never called. The
encoder runs the blocks one at a time, because that is the only way to intervene
between two of them without modifying the model.

The design decision that matters most here is **which layer to prune at**. Pruning
happens after the second block, and the reasoning has two opposing sides:

* Pruning earlier does not work. A token fresh out of the patch embedding has not
  been through any attention layer and cannot tell whether it is rare or common. It
  needs a few blocks to acquire context.
* Pruning later wastes. Cutting at block thirty means having paid full cost for
  thirty blocks to save on ten.

Two blocks give enough context, and thirty-eight remain to save on. A progressive
schedule, pruning a little at each block, was considered and dropped: with forty
blocks the early ones cost little, so the advantage disappears in exchange for
considerably more complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from vitprune.core.attention import run_block
from vitprune.core.geometry import ClipGeometry
from vitprune.core.inspection import TokenSelection
from vitprune.core.merging import initial_sizes, merge_tokens, size_weighted_mean
from vitprune.core.scoring import score_tokens


@dataclass(frozen=True)
class PruningPlan:
    """How much to prune, where, and by what criterion.

    Attributes:
        retention: retention ratio, the fraction of tokens kept per temporal slice.
        prune_layer: which block to prune after, counting from 1. Zero disables
            pruning entirely.
        motion_weight: weight of motion against uniqueness in the score.
        spatial_prior: per-position multiplier, or ``None``.
        size_biased_attention: enables the proportional attention described in
            :mod:`vitprune.core.attention`. Experimental.
    """

    retention: float = 0.25
    prune_layer: int = 2
    motion_weight: float = 0.5
    spatial_prior: torch.Tensor | None = None
    size_biased_attention: bool = False

    @classmethod
    def dense(cls) -> PruningPlan:
        """A plan that prunes nothing.

        The baseline for every comparison, and the sanity check: with this plan the
        result has to match the backbone's own forward pass exactly.
        """
        return cls(retention=1.0, prune_layer=0)

    @property
    def prunes(self) -> bool:
        return self.prune_layer > 0 and self.retention < 1.0

    def with_retention(self, retention: float) -> PruningPlan:
        """A copy at a different retention ratio, for sweeping the curve."""
        if retention >= 1.0:
            return PruningPlan.dense()
        return replace(self, retention=retention)

    def describe(self) -> str:
        if not self.prunes:
            return "no pruning (dense baseline)"
        extra = ", size-biased attention" if self.size_biased_attention else ""
        return (
            f"retention {self.retention:.0%} after block {self.prune_layer}, "
            f"motion_weight {self.motion_weight}{extra}"
        )


class PrunedEncoder(nn.Module):
    """Wraps a video backbone and prunes its tokens mid-forward.

    The backbone is frozen: no parameter is trained or altered. This class decides
    which tokens are worth carrying forward; it does not learn to decide.

    See :mod:`vitprune.core.protocols` for the attributes the backbone must expose,
    and :func:`vitprune.core.compat.check_backbone` for verifying a given model
    before relying on it.
    """

    def __init__(
        self,
        backbone: nn.Module,
        geometry: ClipGeometry | None = None,
        plan: PruningPlan | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.geometry = geometry or ClipGeometry.videomae_giant()
        self.plan = plan or PruningPlan()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """Produce the embedding for each clip.

        Args:
            clips: ``[batch, channels, frames, height, width]``.

        Returns:
            ``[batch, embed_dim]``.
        """
        tokens = self._tokenise(clips)
        self._check_grid(tokens.shape[1])

        per_slice = self.geometry.tokens_per_slice
        sizes = None

        for index, block in enumerate(self.backbone.blocks):
            tokens = self._run(block, tokens, sizes)

            if self._should_prune(index):
                tokens, sizes, per_slice = self._prune_and_merge(tokens, per_slice)

        return self._pool(tokens, sizes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.encode(clips)

    @torch.no_grad()
    def select(self, clips: torch.Tensor) -> TokenSelection:
        """Expose the pruning decision without finishing the forward pass.

        Runs only up to the pruning layer and returns the scores, the kept positions
        and the resulting sizes. With the default cut that is two blocks out of
        forty, so inspection costs a fraction of a full extraction.

        The decision is taken along the same path as :meth:`encode`, with the same
        score and the same criterion. It is not an approximate reconstruction.
        """
        if not self.plan.prunes:
            raise ValueError(
                "a dense plan prunes nothing, so there is no selection to inspect. "
                "Use a retention ratio below 1."
            )

        tokens = self._tokenise(clips)
        self._check_grid(tokens.shape[1])

        for index, block in enumerate(self.backbone.blocks):
            tokens = self._run(block, tokens, None)
            if self._should_prune(index):
                break

        grid = self._as_grid(tokens, self.geometry.tokens_per_slice)
        scores = score_tokens(grid, self.plan.motion_weight, self.plan.spatial_prior)
        keep = self.geometry.kept_per_slice(self.plan.retention)

        ranking = scores.argsort(dim=-1, descending=True)
        merged = merge_tokens(grid, initial_sizes(grid), scores, keep)

        return TokenSelection(
            scores=scores,
            kept=ranking[..., :keep].sort(dim=-1).values,
            sizes=merged.sizes.squeeze(-1),
            grid_size=self.geometry.grid_size,
        )

    # -- internals ----------------------------------------------------------
    def _tokenise(self, clips: torch.Tensor) -> torch.Tensor:
        """Apply the patch embedding and add the positional embedding.

        VideoMAE uses a fixed sinusoidal positional embedding rather than a learned
        one. That is why pruning tokens later invalidates no parameter: position has
        already been folded in before any cut.
        """
        out = self.backbone.patch_embed(clips)
        if out.dim() == 5:
            out = out.flatten(2).transpose(1, 2)

        positions = getattr(self.backbone, "pos_embed", None)
        if positions is not None:
            out = out + positions.type_as(out).to(out.device).detach()

        dropout = getattr(self.backbone, "pos_drop", None)
        if dropout is not None:
            out = dropout(out)
        return out

    def _run(self, block, tokens, sizes):
        if not (self.plan.size_biased_attention and sizes is not None):
            return block(tokens)
        log_sizes = torch.log(sizes.reshape(tokens.shape[0], -1).clamp(min=1e-6))
        return run_block(block, tokens, log_sizes)

    def _should_prune(self, index: int) -> bool:
        return self.plan.prunes and (index + 1) == self.plan.prune_layer

    def _prune_and_merge(self, tokens: torch.Tensor, per_slice: int):
        grid = self._as_grid(tokens, per_slice)
        scores = score_tokens(grid, self.plan.motion_weight, self.plan.spatial_prior)
        keep = self.geometry.kept_per_slice(self.plan.retention)
        merged = merge_tokens(grid, initial_sizes(grid), scores, keep)

        batch, slices, _, dim = merged.tokens.shape
        return merged.tokens.reshape(batch, slices * keep, dim), merged.sizes, keep

    def _as_grid(self, tokens: torch.Tensor, per_slice: int) -> torch.Tensor:
        batch, _, dim = tokens.shape
        return tokens.view(batch, self.geometry.temporal_slices, per_slice, dim)

    def _pool(self, tokens: torch.Tensor, sizes: torch.Tensor | None):
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            tokens = norm(tokens)

        if sizes is None:
            embedding = tokens.mean(dim=1)
        else:
            flat = sizes.reshape(tokens.shape[0], tokens.shape[1], 1)
            embedding = size_weighted_mean(tokens, flat)

        final_norm = getattr(self.backbone, "fc_norm", None)
        if final_norm is not None:
            embedding = final_norm(embedding)
        return embedding

    def _check_grid(self, total: int) -> None:
        if total != self.geometry.num_tokens:
            raise ValueError(
                f"the backbone produced {total} tokens but the configured geometry "
                f"expects {self.geometry.num_tokens} "
                f"({self.geometry.temporal_slices} temporal slices x "
                f"{self.geometry.tokens_per_slice}). Check patch_size and tubelet_size."
            )
