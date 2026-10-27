"""A readable report of one run over a video.

What the encoder produces is a matrix of embeddings. Written to disk it is a binary
file that tells an onlooker nothing: not whether pruning happened, not what it cost,
not whether the result is plausible.

This module turns that matrix into a report answering three questions:

* **What came out?** How many clips, what width, how the embeddings vary along the
  video.
* **What did it cost?** Time per clip, separating video decoding from the backbone,
  and how much of that matches the analytic prediction.
* **What was pruned?** How many tokens remained, what size each carries, and which
  region of the frame the token budget went to.

The third question is why this module exists. Without it, a run with pruning and a
run without produce indistinguishable reports, and there is no way to notice that a
budget was ignored because of a misconfiguration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingSummary:
    """How the embeddings behave along the video.

    This does not judge quality -- that needs a downstream model. It detects
    degenerate output: repeated embeddings warrant inspection; identical norms alone
    do not, since LayerNorm can produce those for healthy, different embeddings.
    """

    clips: int
    embed_dim: int
    norm_mean: float
    norm_min: float
    norm_max: float
    consecutive_similarity_mean: float
    all_finite: bool
    identical_embeddings: bool = False

    @classmethod
    def of(cls, embeddings: np.ndarray) -> EmbeddingSummary:
        norms = np.linalg.norm(embeddings, axis=1)
        return cls(
            clips=int(embeddings.shape[0]),
            embed_dim=int(embeddings.shape[1]),
            norm_mean=round(float(norms.mean()), 4),
            norm_min=round(float(norms.min()), 4),
            norm_max=round(float(norms.max()), 4),
            consecutive_similarity_mean=round(_consecutive_similarity(embeddings), 4),
            all_finite=bool(np.isfinite(embeddings).all()),
            identical_embeddings=bool(
                embeddings.shape[0] > 1 and np.allclose(embeddings, embeddings[:1])
            ),
        )

    def describe(self) -> list[str]:
        lines = [
            f"  matrix            {self.clips} clips x {self.embed_dim}",
            f"  norm              mean {self.norm_mean:.2f}, "
            f"from {self.norm_min:.2f} to {self.norm_max:.2f}",
            f"  adjacent clips    mean similarity {self.consecutive_similarity_mean:.3f}",
        ]
        if not self.all_finite:
            lines.append("  WARNING: the embeddings contain non-finite values")
        if self.identical_embeddings:
            lines.append("  WARNING: all embeddings identical; inspect input and backbone")
        return lines


@dataclass
class PruningSummary:
    """What pruning actually did.

    Present so that pruning is verifiable. Without these numbers, a run where the
    budget was silently ignored would look exactly like a correct one. Under an
    adaptive budget, token counts are means over the inspected clips.
    """

    tokens_before: int
    tokens_after: float
    kept_per_slice: float
    tokens_per_slice: int
    mean_merge_size: float
    largest_merge_size: float
    top_rows_budget_share: float
    top_rows_screen_share: float

    @property
    def token_reduction(self) -> float:
        return self.tokens_before / max(self.tokens_after, 1)

    def describe(self) -> list[str]:
        lines = [
            f"  tokens            {self.tokens_before} -> {self.tokens_after:g} "
            f"({self.token_reduction:.2f}x fewer)",
            f"  per slice         keeping {self.kept_per_slice:g} of {self.tokens_per_slice}",
            f"  merge             mean {self.mean_merge_size:.1f} tokens per kept, "
            f"largest {self.largest_merge_size:.0f}",
            f"  top rows          {self.top_rows_budget_share:.1%} of budget for "
            f"{self.top_rows_screen_share:.0%} of the frame",
        ]
        if self.top_rows_budget_share > self.top_rows_screen_share * 1.3:
            lines.append("  -> the method is favouring the top of the frame")
        elif self.top_rows_budget_share < self.top_rows_screen_share * 0.7:
            lines.append("  -> the method is avoiding the top of the frame")
        return lines


@dataclass
class TimingSummary:
    """Where the time went, and how much of the prediction was realised."""

    read_seconds: float
    encode_seconds: float
    clips: int
    predicted_speedup: float
    measured_speedup: float | None = None

    @property
    def seconds_per_clip(self) -> float:
        return self.encode_seconds / max(self.clips, 1)

    @property
    def realisation(self) -> float | None:
        if self.measured_speedup is None:
            return None
        return self.measured_speedup / self.predicted_speedup

    def describe(self) -> list[str]:
        lines = [
            f"  video decoding    {self.read_seconds:.1f}s",
            f"  backbone          {self.encode_seconds:.1f}s "
            f"({self.seconds_per_clip:.2f}s per clip)",
            f"  predicted speedup {self.predicted_speedup:.2f}x",
        ]
        if self.measured_speedup is not None:
            lines.append(f"  measured speedup  {self.measured_speedup:.2f}x")
            lines.append(f"  realisation       {self.realisation:.0%} of predicted")
        else:
            lines.append("  measured speedup  not measured (use --compare for the dense baseline)")
        return lines


@dataclass
class RunReport:
    """The complete report of one run."""

    video: str
    plan: str
    geometry: str
    embeddings: EmbeddingSummary
    timing: TimingSummary
    pruning: PruningSummary | None = None
    output_files: list[str] = field(default_factory=list)

    def render(self) -> str:
        rule = "=" * 74
        parts = [
            rule,
            "vitprune",
            rule,
            f"  video             {Path(self.video).name}",
            f"  grid              {self.geometry}",
            f"  plan              {self.plan}",
            "",
            "Embeddings produced",
            *self.embeddings.describe(),
            "",
            "Time",
            *self.timing.describe(),
        ]
        if self.pruning is not None:
            parts += ["", "Pruning", *self.pruning.describe()]
        if self.output_files:
            parts += ["", "Files"]
            parts += [f"  {name}" for name in self.output_files]
        parts += [
            "",
            "The embeddings above are what vitprune produces. Which actions occur and",
            "over what intervals is the job of a temporal model, which is not part of",
            "this package -- see 'vitprune evaluate'.",
        ]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timing"]["seconds_per_clip"] = self.timing.seconds_per_clip
        data["timing"]["realisation"] = self.timing.realisation
        if self.pruning is not None:
            data["pruning"]["token_reduction"] = self.pruning.token_reduction
        return data


def summarise_pruning(selection, geometry, top_rows: int = 4) -> PruningSummary:
    """Summarise a pruning decision from a :class:`~vitprune.core.inspection.TokenSelection`."""
    merge_sizes, budget_shares = [], []
    for clip in range(selection.clips):
        for index in range(selection.temporal_slices):
            sizes = selection.size_grid(clip, index)
            merge_sizes.append(sizes[sizes > 0])
            budget_shares.append(selection.rows_share(clip, index, top_rows))

    stacked = np.concatenate(merge_sizes)
    kept = float(np.mean([selection.kept_in(clip) for clip in range(selection.clips)]))
    return PruningSummary(
        tokens_before=geometry.num_tokens,
        tokens_after=selection.temporal_slices * kept,
        kept_per_slice=kept,
        tokens_per_slice=selection.tokens_per_slice,
        mean_merge_size=round(float(stacked.mean()), 3),
        largest_merge_size=round(float(stacked.max()), 1),
        top_rows_budget_share=round(float(np.mean(budget_shares)), 4),
        top_rows_screen_share=round(top_rows / selection.grid_size, 4),
    )


def _consecutive_similarity(embeddings: np.ndarray) -> float:
    """Mean similarity between adjacent clips.

    Video is continuous, so neighbouring clips should resemble each other. A very low
    value suggests something is wrong in clip assembly or in the forward pass.
    """
    if embeddings.shape[0] < 2:
        return 1.0
    a, b = embeddings[:-1], embeddings[1:]
    numerator = (a * b).sum(axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(numerator / np.clip(denominator, 1e-9, None)))
