"""The full forward pass, from clip to embedding, with pruning in the middle.

This is the module the others exist to serve. It takes a clip and returns the
embedding that summarises it, having pruned most of the visual tokens along the way.

Control runs the other way round from what one might expect. The encoder does not
hook into the backbone; it drives it. ``backbone.forward()`` is never called. The
encoder runs the blocks one at a time, because that is the only way to intervene
between two of them without modifying the model.

The design decision that matters most here is **which layer to prune at**. Pruning
happens after the second block, and the reasoning has two opposing sides:

* Earlier tokens have less attention context, which may weaken selection quality.
* Pruning later wastes. Cutting at block thirty means having paid full cost for
  thirty blocks to save on ten.

Two blocks are the default hypothesis, not a universally validated optimum. Optional
later stages allow progressive merging for comparison with the single-cut baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from vitprune.core.attention import run_block
from vitprune.core.compressibility import (
    PROBE_CELLS,
    PROBE_TOP,
    CompressibilityFeatures,
    compressibility_features,
)
from vitprune.core.geometry import ClipGeometry, validate_prune_layer
from vitprune.core.inspection import TokenSelection
from vitprune.core.merging import (
    bipartite_merge,
    initial_sizes,
    merge_error,
    merge_tokens,
    size_weighted_mean,
)
from vitprune.core.policy import (
    BudgetDecision,
    BudgetPolicy,
    FixedBudget,
    QualityAwareBudgetPolicy,
)
from vitprune.core.protocols import missing_attributes
from vitprune.core.scoring import apply_prior, score_tokens
from vitprune.core.statistics import ClipStatistics, clip_statistics


@dataclass(frozen=True)
class PruningPlan:
    """Where to prune, by what criterion, and how many tokens each clip keeps.

    Attributes:
        prune_layer: which block to prune after, counting from 1. Zero disables
            pruning entirely.
        motion_weight: weight of motion against uniqueness in the score that decides
            which tokens survive, in [0, 1). Motion alone is refused: the first slice has no
            earlier one, so its tokens would all tie and be kept by grid position.
        spatial_prior: per-position multiplier, or ``None``.
        size_biased_attention: enables the proportional attention described in
            :mod:`vitprune.core.attention`. Experimental.
        attention: use the backbone's own attention or the explicit SDPA path.
        stages: further (block, tokens per clip) merges after the initial cut.
        separate_region: penalize merging across ``spatial_prior > 0``; requires a prior.
        discard_outside: discard positions outside the prior at the cut; requires a budget
            and prior, and cannot be combined with redundant ``separate_region``. Scores and
            adaptive budget decisions still see the full grid before positions are discarded.
        budget: what decides how many tokens each clip keeps: an adaptive policy, a
            quality-aware one, or a fixed budget. Without one, nothing is pruned; the cut
            still marks where clip statistics are measured, which is what calibration
            needs. A quality-aware policy only holds for the selection it was calibrated
            with, so it refuses another motion_weight, a prior or size-biased attention.
    """

    prune_layer: int = 2
    motion_weight: float = 0.5
    spatial_prior: torch.Tensor | None = None
    size_biased_attention: bool = False
    budget: BudgetPolicy | QualityAwareBudgetPolicy | FixedBudget | None = None
    attention: str = "model"
    stages: tuple[tuple[int, int], ...] = ()
    separate_region: bool = False
    discard_outside: bool = False

    def __post_init__(self) -> None:
        validate_prune_layer(self.prune_layer)
        if (
            isinstance(self.motion_weight, bool)
            or not isinstance(self.motion_weight, Real)
            or not isfinite(self.motion_weight)
            or not 0.0 <= self.motion_weight < 1.0
        ):
            raise ValueError("motion_weight must be finite and in [0, 1): motion alone leaves "
                             "the first slice without a score")
        if not isinstance(self.size_biased_attention, bool):
            raise ValueError("size_biased_attention must be a bool")
        if not isinstance(self.separate_region, bool):
            raise ValueError("separate_region must be a bool")
        if self.separate_region and self.spatial_prior is None:
            raise ValueError("separate_region requires spatial_prior")
        if not isinstance(self.discard_outside, bool):
            raise ValueError("discard_outside must be a bool")
        if self.discard_outside and (self.spatial_prior is None or self.budget is None):
            raise ValueError("discard_outside requires spatial_prior and a budget")
        if self.discard_outside and self.separate_region:
            raise ValueError("discard_outside cannot be combined with separate_region")
        if self.attention not in ("model", "sdpa"):
            raise ValueError("attention must be 'model' or 'sdpa'")
        if self.spatial_prior is not None:
            prior = self.spatial_prior
            if not isinstance(prior, torch.Tensor) or prior.ndim != 1:
                raise ValueError("spatial_prior must be a one-dimensional tensor")
            if not torch.isfinite(prior).all() or (prior < 0).any():
                raise ValueError("spatial_prior must contain finite, non-negative values")
        if self.budget is not None:
            if not isinstance(self.budget, (BudgetPolicy, QualityAwareBudgetPolicy, FixedBudget)):
                raise ValueError(
                    "budget must be a BudgetPolicy, a QualityAwareBudgetPolicy or a FixedBudget"
                )
            budget = self.budget
            if isinstance(budget, QualityAwareBudgetPolicy) and (
                self.motion_weight != budget.motion_weight
                or self.spatial_prior is not None
                or self.size_biased_attention
            ):
                raise ValueError(
                    f"the quality-aware policy was calibrated with motion_weight "
                    f"{budget.motion_weight}, no spatial prior and no size-biased attention; "
                    "its risk table does not hold for another selection"
                )
            if self.prune_layer == 0:
                raise ValueError("a budget needs a pruning layer; prune_layer is 0")
        if not isinstance(self.stages, (tuple, list)):
            raise ValueError("stages must be a sequence of (block, tokens) pairs")
        if self.stages:
            if not isinstance(self.budget, FixedBudget):
                raise ValueError("stages require a FixedBudget")
            previous_block, previous_tokens = self.prune_layer, self.budget.tokens
            for stage in self.stages:
                if not isinstance(stage, (tuple, list)) or len(stage) != 2:
                    raise ValueError("each stage must be a (block, tokens) pair")
                block, tokens = stage
                if any(isinstance(v, bool) or not isinstance(v, Integral) for v in stage):
                    raise ValueError("stage blocks and tokens must be integers")
                if block <= previous_block:
                    raise ValueError("stage blocks must increase after the pruning layer")
                if not max(1, previous_tokens // 2) <= tokens < previous_tokens:
                    raise ValueError("stage tokens must decrease by at most half")
                previous_block, previous_tokens = block, tokens
        object.__setattr__(self, "stages", tuple(tuple(stage) for stage in self.stages))

    def validate_for(self, geometry: ClipGeometry) -> None:
        """Validate the cut, prior and budget against a concrete backbone geometry."""
        validate_prune_layer(self.prune_layer, geometry.depth)
        if self.spatial_prior is not None:
            if self.spatial_prior.numel() != geometry.tokens_per_slice:
                raise ValueError("spatial_prior positions must match geometry.tokens_per_slice")
        if self.budget is not None:
            self.budget.validate_for(geometry)
        if self.discard_outside and max(self.budget.levels) > (
            int((self.spatial_prior > 0).sum()) * geometry.temporal_slices
        ):
            raise ValueError("discard_outside budget exceeds positions inside spatial_prior")
        if self.stages and self.stages[-1][0] >= geometry.depth:
            raise ValueError("stage blocks must precede the last block")

    @classmethod
    def dense(cls, attention: str = "model") -> PruningPlan:
        """A plan that prunes nothing.

        The baseline for every comparison, and the sanity check: with this plan the
        result has to match the backbone's own forward pass exactly.
        """
        return cls(prune_layer=0, attention=attention)

    @property
    def prunes(self) -> bool:
        return self.budget is not None

    def describe(self) -> str:
        attention = ", SDPA attention" if self.attention == "sdpa" else ""
        if not self.prunes:
            return "no pruning (dense baseline)" + attention
        extra = attention + (", size-biased attention" if self.size_biased_attention else "")
        if self.spatial_prior is not None:
            extra += ", spatial prior"
        if self.separate_region:
            extra += ", separate region merging"
        if self.discard_outside:
            extra += ", outside region discarded"
        if self.stages:
            extra += ", stages " + " ".join(f"{block}:{tokens}" for block, tokens in self.stages)
        levels = "/".join(str(level) for level in self.budget.levels)
        kind = (
            "fixed" if isinstance(self.budget, FixedBudget)
            else "quality-aware" if isinstance(self.budget, QualityAwareBudgetPolicy)
            else "adaptive"
        )
        return (
            f"{kind} budget of {levels} tokens after block {self.prune_layer}, "
            f"motion_weight {self.motion_weight}, policy {self.budget.version}{extra}"
        )


class ClipBudget(NamedTuple):
    """What the adaptive budget decided for each clip, in the order received.

    Attributes:
        tokens: ``[batch]``, tokens each clip kept after the cut.
        regimes: ``[batch]``, the regime each clip fell in, from 0 upwards.
        statistics: the redundancy and motion the decision was taken from.
        decision: what a quality-aware policy decided and why; ``None`` for the others.
    """

    tokens: torch.Tensor
    regimes: torch.Tensor
    statistics: ClipStatistics
    decision: BudgetDecision | None = None


class LevelEncoding(NamedTuple):
    """The same clips at several fixed budgets, measured from one cut.

    Attributes:
        features: what a budget sees of each clip at the cut, before any merge.
        embeddings: ``{level: [batch, embed_dim]}``. The full level is the dense path.
        stages: ``{level: {block: [batch, embed_dim]}}``, the size-weighted mean of the
            tokens after that block, before any normalisation.
        merge_error: ``{level: [batch, slices, tokens]}``, as
            :func:`vitprune.core.merging.merge_error`; zero for the full level.
    """

    features: CompressibilityFeatures
    embeddings: dict[int, torch.Tensor]
    stages: dict[int, dict[int, torch.Tensor]]
    merge_error: dict[int, torch.Tensor]


class PrunedEncoder(nn.Module):
    """Wraps a video backbone and prunes its tokens mid-forward.

    Shares the supplied backbone: construction freezes its parameters and puts it
    in eval mode, without changing weight values. Inference rejects training mode.
    Do not share this object with a concurrent training job.

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
        self._validate_configuration()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """Produce the embedding for each clip.

        Args:
            clips: ``[batch, channels, frames, height, width]``.

        Returns:
            ``[batch, embed_dim]``.
        """
        return self._encode(clips)[0]

    @torch.no_grad()
    def encode_with_budget(self, clips: torch.Tensor) -> tuple[torch.Tensor, ClipBudget]:
        """Produce the embeddings and the budget decided for each clip.

        Only an adaptive plan has a per-clip budget to report.
        """
        if self.plan.budget is None:
            raise ValueError("only an adaptive plan decides a budget per clip; set plan.budget")
        return self._encode(clips)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.encode(clips)

    @torch.no_grad()
    def select(self, clips: torch.Tensor) -> TokenSelection:
        """Expose the pruning decision without finishing the forward pass.

        Runs only up to the pruning layer and returns the scores, the kept positions
        and the resulting sizes. With the default cut that is two blocks out of
        forty, so inspection costs a fraction of a full extraction.

        The decision is taken along the same path as :meth:`encode`, with the same
        score and the same criterion. It is not an approximate reconstruction. Clips
        keeping fewer tokens than the widest one are padded with position ``-1`` and
        size zero.
        """
        if self.plan.discard_outside:
            raise ValueError("select does not support discard_outside")
        if self.plan.stages:
            raise ValueError("select does not support progressive stages")
        grid = self._grid_at_cut(clips, needs_budget=True)
        base = score_tokens(grid, self.plan.motion_weight)
        keeps, _ = self._keeps(grid, base)
        scores = self._with_prior(base)

        batch, slices, _ = scores.shape
        widest = int(keeps.max())
        kept = torch.full((batch, slices, widest), -1, dtype=torch.long, device=scores.device)
        sizes = grid.new_zeros((batch, slices, widest))
        for keep, members in self._groups(keeps):
            part, part_scores = grid[members], scores[members]
            ranking = part_scores.argsort(dim=-1, descending=True)
            merged = merge_tokens(part, initial_sizes(part), part_scores, keep,
                                  self._region_groups(part.device))
            kept[members, :, :keep] = ranking[..., :keep].sort(dim=-1).values
            sizes[members, :, :keep] = merged.sizes.squeeze(-1)

        return TokenSelection(
            scores=scores, kept=kept, sizes=sizes, grid_size=self.geometry.grid_size
        )

    @torch.no_grad()
    def statistics(self, clips: torch.Tensor) -> ClipStatistics:
        """Measure each clip's redundancy and motion at the pruning layer.

        Like :meth:`select`, only the blocks up to the cut run, and the statistics are
        taken from the same grid the score sees, before any merge.
        """
        return clip_statistics(self._grid_at_cut(clips))

    @torch.no_grad()
    def features(
        self,
        clips: torch.Tensor,
        cells: tuple[int, int] = PROBE_CELLS,
        top: int = PROBE_TOP,
    ) -> CompressibilityFeatures:
        """What a budget sees of each clip at the pruning layer: the statistics and the
        probe on the scores before any prior. Only the blocks up to the cut run."""
        grid = self._grid_at_cut(clips)
        base = score_tokens(grid, self.plan.motion_weight)
        return compressibility_features(grid, base, cells=cells, top=top)

    @torch.no_grad()
    def encode_levels(
        self,
        clips: torch.Tensor,
        levels: Sequence[int],
        depths: Sequence[int] = (),
        cells: tuple[int, int] = PROBE_CELLS,
        top: int = PROBE_TOP,
    ) -> LevelEncoding:
        """Every level from one cut: for calibration and diagnosis, never for inference.

        The blocks up to the cut run once; the remaining blocks then run once per level,
        which inference never does. The point is to see what each budget would have done
        to the same clip, with the plan's own selection. The full level, the dense path,
        is always measured: it is the reference.

        Args:
            clips: ``[batch, channels, frames, height, width]``.
            levels: tokens kept per clip, each a multiple of the temporal slices.
            depths: blocks after which to record the mean of the tokens, from the pruning
                layer (right after the merge) to the last.
            cells, top: the probe's anchors, as in
                :func:`vitprune.core.compressibility.probe_errors`.
        """
        if self.plan.discard_outside:
            raise ValueError("encode_levels does not support discard_outside")
        if self.plan.stages:
            raise ValueError("encode_levels does not support progressive stages")
        levels = sorted({*levels, self.geometry.num_tokens})
        for level in levels:
            FixedBudget(level).validate_for(self.geometry)
        cut = self.plan.prune_layer
        if any(not cut <= depth <= self.geometry.depth for depth in depths):
            raise ValueError(f"depths must lie between block {cut} and {self.geometry.depth}")

        grid = self._grid_at_cut(clips)
        base = score_tokens(grid, self.plan.motion_weight)
        scores = self._with_prior(base)
        features = compressibility_features(grid, base, cells=cells, top=top)

        count, slices, per_slice, dim = grid.shape
        embeddings, stages, errors = {}, {}, {}
        for level in levels:
            keep = level // slices
            if keep == per_slice:
                group, sizes = grid.reshape(count, slices * keep, dim), None
                errors[level] = grid.new_zeros(grid.shape[:3])
            else:
                groups = self._region_groups(grid.device)
                merged = merge_tokens(grid, initial_sizes(grid), scores, keep, groups)
                group, sizes = merged.tokens.reshape(count, slices * keep, dim), merged.sizes
                errors[level] = merge_error(grid, scores, keep, merged, groups)
            stages[level] = {}
            if cut in depths:
                stages[level][cut] = self._mean(group, sizes)
            for index, block in enumerate(list(self.backbone.blocks)[cut:], start=cut + 1):
                group = self._run(block, group, sizes)
                if index in depths:
                    stages[level][index] = self._mean(group, sizes)
            embeddings[level] = self._pool(group, sizes)
        return LevelEncoding(features, embeddings, stages, errors)

    # -- internals ----------------------------------------------------------
    def _encode(self, clips: torch.Tensor) -> tuple[torch.Tensor, ClipBudget | None]:
        """The forward pass.

        After the cut, clips are grouped by how many tokens they keep, and each group
        runs the remaining blocks as its own batch: uniform within a group, so nothing
        is padded. Embeddings come back in the order the clips arrived. A group keeping
        every token takes the dense path itself, with no merge.
        """
        self._validate_configuration()
        self._check_eval()
        tokens = self._tokenise(clips)
        self._check_grid(tokens.shape[1])

        blocks = list(self.backbone.blocks)
        cut = self.plan.prune_layer if self.plan.prunes else len(blocks)
        for block in blocks[:cut]:
            tokens = self._run(block, tokens, None)
        if not self.plan.prunes:
            return self._pool(tokens, None), None

        grid = self._as_grid(tokens, self.geometry.tokens_per_slice)
        base = score_tokens(grid, self.plan.motion_weight)
        keeps, budget = self._keeps(grid, base)
        scores = self._with_prior(base)
        if self.plan.discard_outside:
            inside = self.plan.spatial_prior.to(grid.device) > 0
            grid, scores = grid[:, :, inside], scores[:, :, inside]

        embeddings = None
        for keep, members in self._groups(keeps):
            part = grid[members]
            count, slices, _, dim = part.shape
            region = self._region_groups(part.device)
            groups = None
            if region is not None:
                kept_at = scores[members].argsort(dim=-1, descending=True)[..., :keep]
                groups = region[kept_at.sort(dim=-1).values].reshape(count, slices * keep)
            if keep == part.shape[2]:
                group = part.reshape(count, slices * keep, dim)
                sizes = initial_sizes(part) if self.plan.discard_outside else None
                if sizes is not None and (self.plan.stages or self.plan.discard_outside):
                    sizes = sizes.float()
            else:
                sizes = initial_sizes(part)
                if self.plan.stages or self.plan.discard_outside:
                    sizes = sizes.float()
                merged = merge_tokens(part, sizes, scores[members], keep, region)
                group = merged.tokens.to(part.dtype).reshape(count, slices * keep, dim)
                sizes = merged.sizes
            stages = dict(self.plan.stages)
            for index, block in enumerate(blocks[cut:], start=cut + 1):
                group = self._run(block, group, sizes)
                if index in stages:
                    if sizes is None:
                        sizes = torch.ones((*group.shape[:2], 1), device=group.device)
                    else:
                        sizes = sizes.reshape(count, -1, 1)
                    group, sizes, groups = bipartite_merge(
                        group, sizes, self._stage_keys(blocks[index], group), stages[index], groups
                    )
            pooled = self._pool(group, sizes)
            if embeddings is None:
                embeddings = pooled.new_empty((clips.shape[0], pooled.shape[1]))
            embeddings[members] = pooled
        return embeddings, budget

    @staticmethod
    def _stage_keys(block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, dim = tokens.shape
        attention = block.attn
        bias = None if getattr(attention, "q_bias", None) is not None else getattr(
            attention.qkv, "bias", None
        )
        if bias is not None:
            bias = bias[dim:2 * dim]
        keys = F.linear(block.norm1(tokens), attention.qkv.weight[dim:2 * dim], bias)
        return keys.reshape(batch, count, attention.num_heads, -1).mean(dim=2)

    def _keeps(self, grid: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, ClipBudget]:
        """Tokens kept per slice by each clip, and the record of the decision.

        ``base`` is the motion-aware score before any prior: what a quality-aware policy's
        probe reads, while the selection that follows reads the score with the prior.
        """
        budget = self.plan.budget
        statistics = clip_statistics(grid)
        decision = None
        if isinstance(budget, QualityAwareBudgetPolicy):
            features = compressibility_features(grid, base, statistics, budget.probe_cells,
                                                budget.probe_top, probe=budget.uses_probe)
            decision = budget.choose_budget(features)
            tokens, regimes = decision.tokens, decision.regimes
        elif isinstance(budget, FixedBudget):
            tokens = torch.full((grid.shape[0],), budget.tokens, device=grid.device)
            regimes = torch.zeros_like(tokens)
        else:
            tokens = budget.tokens(statistics)
            regimes = budget.regimes(statistics)
        record = ClipBudget(tokens, regimes, statistics, decision)
        return tokens // self.geometry.temporal_slices, record

    def _with_prior(self, base: torch.Tensor) -> torch.Tensor:
        prior = self.plan.spatial_prior
        return base if prior is None else apply_prior(base, prior)

    def _region_groups(self, device: torch.device) -> torch.Tensor | None:
        return (self.plan.spatial_prior.to(device) > 0) if self.plan.separate_region else None

    @staticmethod
    def _groups(keeps: torch.Tensor):
        """Each distinct count, with the positions of the clips keeping it."""
        for keep in keeps.unique().tolist():
            yield keep, (keeps == keep).nonzero(as_tuple=True)[0]

    def _grid_at_cut(self, clips: torch.Tensor, needs_budget: bool = False) -> torch.Tensor:
        """Run the blocks up to the pruning layer and return the token grid there."""
        self._validate_configuration()
        self._check_eval()
        if self.plan.prune_layer == 0:
            raise ValueError(
                "a dense plan prunes nothing, so there is no pruning layer to inspect. "
                "Set prune_layer."
            )
        if needs_budget and self.plan.budget is None:
            raise ValueError(
                "a plan without a budget prunes nothing, so there is no selection to "
                "inspect. Set plan.budget."
            )

        tokens = self._tokenise(clips)
        self._check_grid(tokens.shape[1])

        for block in list(self.backbone.blocks)[: self.plan.prune_layer]:
            tokens = self._run(block, tokens, None)
        return self._as_grid(tokens, self.geometry.tokens_per_slice)

    def _validate_configuration(self) -> None:
        missing = missing_attributes(self.backbone)
        if missing:
            raise ValueError(f"backbone is missing required attributes: {', '.join(missing)}")
        if not callable(self.backbone.patch_embed):
            raise ValueError("backbone.patch_embed must be callable")
        blocks = self.backbone.blocks
        if not hasattr(blocks, "__len__") or not hasattr(blocks, "__iter__"):
            raise ValueError("backbone.blocks must be a sized sequence of callable blocks")
        if len(blocks) != self.geometry.depth:
            raise ValueError(
                f"backbone depth ({len(blocks)}) differs from geometry.depth "
                f"({self.geometry.depth})"
            )
        self.plan.validate_for(self.geometry)
        for block in blocks:
            if not callable(block):
                raise ValueError("each backbone block must be callable")
        stage_blocks = {stage for stage, _ in self.plan.stages}
        for index, block in enumerate(blocks):
            if self.plan.attention == "sdpa":
                reason = "SDPA attention"
            elif (self.plan.prunes and self.plan.size_biased_attention
                  and index >= self.plan.prune_layer):
                reason = "size-biased attention"
            elif index in stage_blocks:
                reason = "progressive stages"
            else:
                continue
            for name in ("norm1", "norm2", "mlp", "attn"):
                if not callable(getattr(block, name, None)):
                    raise ValueError(f"{reason} requires block.{name}")
            attention = block.attn
            for name in ("qkv", "proj"):
                if not callable(getattr(attention, name, None)):
                    raise ValueError(f"{reason} requires attention.{name}")
            if (
                not isinstance(getattr(attention, "num_heads", None), int)
                or attention.num_heads <= 0
            ):
                raise ValueError(f"{reason} requires attention.num_heads")
            if self.geometry.embed_dim % attention.num_heads:
                raise ValueError("attention.num_heads must divide geometry.embed_dim")
            if index in stage_blocks and not hasattr(attention.qkv, "weight"):
                raise ValueError("progressive stages require attention.qkv.weight")

    def _check_eval(self) -> None:
        if any(module.training for module in self.modules()):
            raise ValueError("PrunedEncoder is inference-only; call encoder.eval() before encoding")

    def _tokenise(self, clips: torch.Tensor) -> torch.Tensor:
        """Apply the patch embedding and add the positional embedding.

        VideoMAE uses a fixed sinusoidal positional embedding rather than a learned
        one. That is why pruning tokens later invalidates no parameter: position has
        already been folded in before any cut.
        """
        if clips.ndim != 5 or tuple(clips.shape[1:]) != self.geometry.input_shape()[1:]:
            raise ValueError(f"clips must have shape [batch, {self.geometry.input_shape()[1:]}]")
        if clips.shape[0] < 1 or not clips.is_floating_point():
            raise ValueError("clips must be a non-empty floating-point batch")
        out = self.backbone.patch_embed(clips)
        if out.dim() == 5:
            out = out.flatten(2).transpose(1, 2)
        if out.ndim != 3 or out.shape[0] != clips.shape[0]:
            raise ValueError("patch_embed must return [B,L,D] or [B,D,S,H,W]")
        if out.shape[-1] != self.geometry.embed_dim:
            raise ValueError("patch_embed width differs from geometry.embed_dim")

        positions = getattr(self.backbone, "pos_embed", None)
        if positions is not None:
            out = out + positions.type_as(out).to(out.device).detach()

        dropout = getattr(self.backbone, "pos_drop", None)
        if dropout is not None:
            out = dropout(out)
        return out

    def _run(self, block, tokens, sizes):
        biased = self.plan.size_biased_attention and sizes is not None
        if self.plan.attention != "sdpa" and not biased:
            return block(tokens)
        log_sizes = None
        if biased:
            log_sizes = torch.log(sizes.reshape(tokens.shape[0], -1).clamp(min=1e-6))
        return run_block(block, tokens, log_sizes)

    def _as_grid(self, tokens: torch.Tensor, per_slice: int) -> torch.Tensor:
        batch, _, dim = tokens.shape
        return tokens.reshape(batch, self.geometry.temporal_slices, per_slice, dim)

    @staticmethod
    def _mean(tokens: torch.Tensor, sizes: torch.Tensor | None) -> torch.Tensor:
        """The size-weighted mean of the tokens, with no normalisation."""
        if sizes is None:
            return tokens.mean(dim=1)
        return size_weighted_mean(tokens, sizes.reshape(tokens.shape[0], tokens.shape[1], 1))

    def _pool(self, tokens: torch.Tensor, sizes: torch.Tensor | None):
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            tokens = norm(tokens)

        if sizes is None:
            embedding = tokens.mean(dim=1)
        else:
            flat = sizes.reshape(tokens.shape[0], tokens.shape[1], 1)
            embedding = size_weighted_mean(tokens, flat)
            if self.plan.stages or self.plan.discard_outside:
                embedding = embedding.to(tokens.dtype)

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
