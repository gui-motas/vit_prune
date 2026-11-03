"""How many tokens each clip keeps: the adaptive budget.

The score in :mod:`vitprune.core.scoring` decides *which* tokens survive. This
module decides *how many*, clip by clip, from the statistics measured in
:mod:`vitprune.core.statistics`. Three steps:

1. **Normalise.** Redundancy and motion have no natural scale: it depends on the
   checkpoint and on the layer. Each value is turned into its rank in ``[0, 1]``
   against a reference distribution measured on calibration clips. The transform is
   monotonic, so the order between clips is preserved.
2. **Combine.** ``g = β·motion + (1 − β)·(1 − redundancy)``, on the ranks. More
   motion and less redundancy both ask for more tokens. ``β`` is
   ``motion_balance``; it decides how many tokens a clip gets, while the plan's
   ``motion_weight`` still decides which ones.
3. **Quantise.** Ascending thresholds cut ``g`` into regimes, each with a fixed
   token count. A few regimes rather than a continuous count keep a batch to a few
   uniform sub-batches, so nothing has to be padded.

A policy is data: frozen, serialisable, and identified by a version hash, so a
result can name exactly the policy that produced it.

:class:`QualityAwareBudgetPolicy` decides differently: it asks how much fidelity each budget
is likely to cost the clip, from a risk table calibrated on the drift of the final embedding
against the dense path, and takes the smallest budget whose risk is within a tolerance.
:class:`FixedBudget` gives every clip the same count, the baseline both are judged against.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any, NamedTuple

import torch

from vitprune.core.compressibility import CompressibilityFeatures
from vitprune.core.geometry import ClipGeometry
from vitprune.core.statistics import ClipStatistics

#: Reference values kept per statistic: enough to rank against, few enough to read.
REFERENCE_POINTS = 101


@dataclass(frozen=True)
class BudgetPolicy:
    """Maps a clip's statistics to the number of tokens it keeps.

    Attributes:
        redundancy_reference: ascending reference values of redundancy, from
            calibration clips. A clip's rank among them is its normalised redundancy.
        motion_reference: the same for motion.
        thresholds: ascending cut points on ``g``, in ``[0, 1]``; one fewer than the
            levels.
        levels: tokens kept per clip in each regime, strictly ascending. Each must be
            a multiple of the temporal slices, so every slice keeps the same count.
        motion_balance: ``β``, the weight of motion against redundancy in ``g``.
    """

    redundancy_reference: tuple[float, ...]
    motion_reference: tuple[float, ...]
    thresholds: tuple[float, ...]
    levels: tuple[int, ...]
    motion_balance: float = 0.5

    def __post_init__(self) -> None:
        for name in ("redundancy_reference", "motion_reference", "thresholds", "levels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        for name in ("redundancy_reference", "motion_reference"):
            values = getattr(self, name)
            if len(values) < 2 or not all(_is_real(v) and isfinite(v) for v in values):
                raise ValueError(f"{name} must hold at least two finite numbers")
            if any(b < a for a, b in pairwise(values)):
                raise ValueError(f"{name} must be in ascending order")

        levels = self.levels
        if len(levels) < 2:
            raise ValueError("a budget needs at least two levels")
        if not all(isinstance(v, Integral) and not isinstance(v, bool) for v in levels):
            raise ValueError("levels must be integers")
        if levels[0] < 1 or any(b <= a for a, b in pairwise(levels)):
            raise ValueError(f"levels must be positive and strictly ascending, got {levels}")

        thresholds = self.thresholds
        if len(thresholds) != len(levels) - 1:
            raise ValueError("there must be exactly one threshold fewer than levels")
        if not all(_is_real(t) and isfinite(t) and 0.0 <= t <= 1.0 for t in thresholds):
            raise ValueError("thresholds must be finite and in [0, 1]")
        if any(b <= a for a, b in pairwise(thresholds)):
            raise ValueError("thresholds must be strictly ascending")

        balance = self.motion_balance
        if not _is_real(balance) or not isfinite(balance) or not 0.0 <= balance <= 1.0:
            raise ValueError("motion_balance must be finite and in [0, 1]")

    def validate_for(self, geometry: ClipGeometry) -> None:
        """Check that every level fits the token grid of a concrete backbone."""
        _check_levels_fit(self.levels, geometry)

    def score(self, statistics: ClipStatistics) -> torch.Tensor:
        """``g`` for each clip, in ``[0, 1]``. Higher asks for more tokens."""
        redundancy, motion = statistics
        if not (torch.isfinite(redundancy).all() and torch.isfinite(motion).all()):
            raise ValueError("clip statistics must be finite")
        r = _rank(redundancy, self.redundancy_reference)
        m = _rank(motion, self.motion_reference)
        return self.motion_balance * m + (1.0 - self.motion_balance) * (1.0 - r)

    def regimes(self, statistics: ClipStatistics) -> torch.Tensor:
        """Index of each clip's regime, from 0 (fewest tokens) upwards."""
        g = self.score(statistics)
        edges = torch.tensor(self.thresholds, dtype=g.dtype, device=g.device)
        return torch.searchsorted(edges, g.contiguous(), right=True)

    def tokens(self, statistics: ClipStatistics) -> torch.Tensor:
        """Tokens each clip keeps, as integers."""
        regimes = self.regimes(statistics)
        return torch.tensor(self.levels, device=regimes.device)[regimes]

    @property
    def version(self) -> str:
        """A short hash of the policy's content, for naming it in results."""
        canonical = json.dumps(self._fields(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {**self._fields(), "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetPolicy:
        """Rebuild a policy, refusing one whose content no longer matches its version."""
        data = dict(data)
        recorded = data.pop("version", None)
        policy = cls(**data)
        if recorded is not None and recorded != policy.version:
            raise ValueError(
                f"policy content does not match its recorded version ({recorded}); "
                "it was edited after it was produced"
            )
        return policy

    def _fields(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


def calibrate_policy(
    statistics: ClipStatistics,
    levels: Sequence[int],
    motion_balance: float = 0.5,
    reference_points: int = REFERENCE_POINTS,
) -> BudgetPolicy:
    """Fit a policy to the statistics of calibration clips.

    The reference values are evenly spaced quantiles of the calibration statistics,
    and the thresholds split the calibration clips into regimes of equal size. The
    mean token count on those clips is therefore the mean of the levels: levels
    symmetric around a target, such as 256, 512 and 768, hit that target.

    Calibration clips must not come from the evaluation set, and the policy should be
    frozen before evaluating; otherwise it is tuned on the data it is judged on.

    Raises:
        ValueError: when there are fewer clips than levels, or when they vary too
            little to separate the regimes.
    """
    redundancy = statistics.redundancy.detach().double().cpu().flatten()
    motion = statistics.motion.detach().double().cpu().flatten()
    levels = tuple(levels)
    if redundancy.numel() != motion.numel():
        raise ValueError("redundancy and motion must describe the same clips")
    if redundancy.numel() < len(levels):
        raise ValueError(
            f"calibration needs at least one clip per level, got {redundancy.numel()}"
        )
    if not (torch.isfinite(redundancy).all() and torch.isfinite(motion).all()):
        raise ValueError("calibration statistics must be finite")
    if isinstance(reference_points, bool) or not isinstance(reference_points, int):
        raise ValueError("reference_points must be an integer")
    if reference_points < 2:
        raise ValueError("reference_points must be at least 2")

    quantiles = torch.linspace(0.0, 1.0, reference_points, dtype=torch.float64)
    split = torch.arange(1, len(levels), dtype=torch.float64) / len(levels)
    draft = BudgetPolicy(
        redundancy_reference=torch.quantile(redundancy, quantiles).tolist(),
        motion_reference=torch.quantile(motion, quantiles).tolist(),
        thresholds=split.tolist(),
        levels=levels,
        motion_balance=motion_balance,
    )
    g = draft.score(ClipStatistics(redundancy.float(), motion.float()))
    thresholds = torch.quantile(g.double(), split).tolist()
    if any(b <= a for a, b in pairwise(thresholds)):
        raise ValueError("the calibration clips vary too little to separate the regimes")
    return replace(draft, thresholds=thresholds)


def _check_levels_fit(levels: Sequence[int], geometry: ClipGeometry) -> None:
    slices = geometry.temporal_slices
    for level in levels:
        if level % slices or level > geometry.num_tokens:
            raise ValueError(
                f"budget level {level} must be a multiple of the {slices} temporal "
                f"slices and at most {geometry.num_tokens} tokens"
            )


@dataclass(frozen=True)
class FixedBudget:
    """Every clip keeps the same number of tokens: the baseline a policy is judged against.

    With every token of the clip, the clip takes the dense path itself, with no merge.
    """

    tokens: int

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, Integral):
            raise ValueError("a fixed budget keeps a whole number of tokens")
        if self.tokens < 1:
            raise ValueError(f"a fixed budget keeps at least one token, got {self.tokens}")

    @property
    def levels(self) -> tuple[int, ...]:
        return (self.tokens,)

    def validate_for(self, geometry: ClipGeometry) -> None:
        _check_levels_fit(self.levels, geometry)

    @property
    def version(self) -> str:
        return f"fixed-{self.tokens}"

    def to_dict(self) -> dict[str, Any]:
        return {"policy_type": "fixed", "tokens": self.tokens, "version": self.version}


#: Why a clip got its budget. The last three send it to the dense path as a fallback.
REASONS = ("ok", "no_level_within_tolerance", "non_finite", "out_of_range", "low_support")
QUALITY_AWARE = "quality_aware_v1"


class BudgetDecision(NamedTuple):
    """What the quality-aware policy decided for each clip, and why.

    Attributes:
        tokens: ``[batch]``, tokens kept, one of the levels.
        regimes: ``[batch]``, the index of that level.
        predicted_risk: ``[batch]``, the calibrated risk of the level taken; 0 for the dense
            path, which does not drift.
        support: ``[batch]``, calibration clips in the clip's bin; 0 when no bin was read.
        used_fallback: ``[batch]``, the dense path taken because the decision could not be
            trusted: a feature not finite, outside the calibrated range, or a thin bin.
        index: ``[batch]``, the clip's ``h``; NaN when not finite.
        reasons: one of :data:`REASONS` per clip.
        features: what the decision was taken from.
    """

    tokens: torch.Tensor
    regimes: torch.Tensor
    predicted_risk: torch.Tensor
    support: torch.Tensor
    used_fallback: torch.Tensor
    index: torch.Tensor
    reasons: tuple[str, ...]
    features: CompressibilityFeatures


@dataclass(frozen=True)
class QualityAwareBudgetPolicy:
    """The smallest budget whose calibrated risk of drift is within a tolerance.

    The clip's index ``h = alpha * rank(g) + (1 - alpha) * rank(c)`` mixes the legacy score
    ``g`` (motion against redundancy, as :class:`BudgetPolicy` computes it) with the probe's
    90th percentile error ``c``, both ranked against calibration clips. ``h`` falls in one of
    a few bins of equal size on the calibration clips; ``risk[bin][level]`` is a high
    quantile of the drift ``1 - cos(E_dense, E_level)`` observed there. The clip takes the
    smallest level whose risk is at most ``tolerance``, or the dense path, the last level.

    The table is an empirical estimate, subject to sampling error and to clips unlike the
    calibration ones: its quantile is not a guarantee about future clips. So the dense path
    is also taken, and marked as a fallback, for a feature not finite, a feature outside
    the calibrated range, or a bin with fewer than ``min_support`` clips or ``min_videos``
    videos. With ``alpha = 1`` the probe is not used, nor measured.

    The table holds only for the selection it was calibrated with: ``motion_weight``, no
    spatial prior and no size-biased attention. :class:`vitprune.core.encoder.PruningPlan`
    refuses any other.
    """

    levels: tuple[int, ...]
    motion_reference: tuple[float, ...]
    redundancy_reference: tuple[float, ...]
    score_reference: tuple[float, ...]
    probe_reference: tuple[float, ...]
    bin_edges: tuple[float, ...]
    risk: tuple[tuple[float, ...], ...]
    support: tuple[int, ...]
    support_videos: tuple[int, ...]
    support_drivers: tuple[int, ...]
    feature_range: tuple[tuple[float, float], ...]
    tolerance: float
    alpha: float
    motion_balance: float = 0.5
    risk_quantile: float = 0.9
    min_support: int = 30
    min_videos: int = 3
    motion_weight: float = 0.5
    probe_cells: tuple[int, int] = (2, 4)
    probe_top: int = 8
    policy_type: str = QUALITY_AWARE

    def __post_init__(self) -> None:
        for name in ("levels", "motion_reference", "redundancy_reference", "score_reference",
                     "probe_reference", "bin_edges", "support", "support_videos",
                     "support_drivers", "probe_cells"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "risk", tuple(tuple(row) for row in self.risk))
        object.__setattr__(self, "feature_range", tuple(tuple(r) for r in self.feature_range))

        if self.policy_type != QUALITY_AWARE:
            raise ValueError(f"unknown policy type {self.policy_type!r}")
        levels = self.levels
        if len(levels) < 2 or not all(_is_count(v) and v > 0 for v in levels):
            raise ValueError("levels must be at least two positive integers")
        if any(b <= a for a, b in pairwise(levels)):
            raise ValueError(f"levels must be strictly ascending, got {levels}")
        for name in ("alpha", "motion_balance", "motion_weight"):
            if not _in_unit(getattr(self, name)):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not (_is_real(self.tolerance) and isfinite(self.tolerance) and self.tolerance > 0):
            raise ValueError("tolerance must be finite and positive")
        if not (_in_unit(self.risk_quantile) and self.risk_quantile > 0):
            raise ValueError("risk_quantile must be in (0, 1]")
        references = ["motion_reference", "redundancy_reference", "score_reference"]
        if self.uses_probe:
            references.append("probe_reference")
        elif self.probe_reference:
            raise ValueError("a policy without the probe holds no probe reference")
        for name in references:
            values = getattr(self, name)
            if len(values) < 2 or not all(_is_real(v) and isfinite(v) for v in values):
                raise ValueError(f"{name} must hold at least two finite numbers")
            if any(b < a for a, b in pairwise(values)):
                raise ValueError(f"{name} must be in ascending order")
        edges = self.bin_edges
        if not all(_in_unit(edge) for edge in edges) or any(b <= a for a, b in pairwise(edges)):
            raise ValueError("bin_edges must be strictly ascending and in [0, 1]")
        bins = len(edges) + 1
        if len(self.risk) != bins or any(len(row) != len(levels) - 1 for row in self.risk):
            raise ValueError("risk must hold one row per bin and one value per level below "
                             "the last")
        if not all(_is_real(v) and v >= 0 for row in self.risk for v in row):
            raise ValueError("risks must be non-negative numbers")
        for name in ("support", "support_videos", "support_drivers"):
            counts = getattr(self, name)
            if len(counts) != bins or not all(_is_count(c) and c >= 0 for c in counts):
                raise ValueError(f"{name} must hold one non-negative count per bin")
        if len(self.feature_range) != (3 if self.uses_probe else 2) or not all(
            len(r) == 2 and all(_is_real(v) and isfinite(v) for v in r) and r[0] <= r[1]
            for r in self.feature_range
        ):
            raise ValueError("feature_range must hold a finite (low, high) per feature used")
        for name in ("min_support", "min_videos"):
            if not _is_count(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.probe_cells) != 2 or not all(_is_count(c) and c > 0 for c in self.probe_cells):
            raise ValueError("probe_cells must be two positive integers")
        if not _is_count(self.probe_top) or self.probe_top < 0:
            raise ValueError("probe_top must be a non-negative integer")

    @property
    def uses_probe(self) -> bool:
        return self.alpha < 1.0

    def validate_for(self, geometry: ClipGeometry) -> None:
        """Every level fits the grid, and the last is the dense path the fallback takes."""
        _check_levels_fit(self.levels, geometry)
        if self.levels[-1] != geometry.num_tokens:
            raise ValueError(f"the last level must be the dense path, {geometry.num_tokens} "
                             f"tokens, got {self.levels[-1]}")

    def index(self, features: CompressibilityFeatures) -> torch.Tensor:
        """``h`` for each clip, in ``[0, 1]``, in double precision."""
        return _quality_index(
            features.motion.double(), features.redundancy.double(),
            features.probe_p90.double(), self.motion_reference, self.redundancy_reference,
            self.score_reference, self.probe_reference, self.motion_balance, self.alpha,
        )

    def choose_budget(self, features: CompressibilityFeatures) -> BudgetDecision:
        """The level of each clip, independently of the rest of the batch."""
        used = [features.motion, features.redundancy]
        if self.uses_probe:
            used.append(features.probe_p90)
        values = torch.stack([value.double() for value in used], dim=1)
        device = values.device
        finite = torch.isfinite(values).all(dim=1)
        bounds = torch.tensor(self.feature_range, dtype=torch.float64, device=device)
        inside = finite & ((values >= bounds[:, 0]) & (values <= bounds[:, 1])).all(dim=1)

        safe = CompressibilityFeatures(*(torch.nan_to_num(v.double(), 0.0, 0.0, 0.0)
                                         for v in features))
        index = self.index(safe)
        edges = torch.tensor(self.bin_edges, dtype=torch.float64, device=device)
        bins = torch.searchsorted(edges, index.contiguous(), right=True)
        support = torch.tensor(self.support, device=device)[bins]
        videos = torch.tensor(self.support_videos, device=device)[bins]
        supported = (support >= self.min_support) & (videos >= self.min_videos)
        risk = torch.tensor(self.risk, dtype=torch.float64, device=device)[bins]
        admissible = risk <= self.tolerance
        found = admissible.any(dim=1)
        dense = len(self.levels) - 1
        trusted = inside & supported
        regimes = torch.where(trusted & found, admissible.int().argmax(dim=1),
                              torch.full_like(bins, dense))
        predicted = risk.gather(1, regimes.clamp(max=dense - 1).unsqueeze(1)).squeeze(1)
        reasons = tuple(
            "non_finite" if not f else "out_of_range" if not i else "low_support" if not s
            else "ok" if k else "no_level_within_tolerance"
            for f, i, s, k in zip(finite.tolist(), inside.tolist(), supported.tolist(),
                                  found.tolist(), strict=True)
        )
        return BudgetDecision(
            tokens=torch.tensor(self.levels, device=device)[regimes],
            regimes=regimes,
            predicted_risk=torch.where(regimes < dense, predicted, torch.zeros_like(predicted)),
            support=torch.where(inside, support, torch.zeros_like(support)),
            used_fallback=~trusted,
            index=torch.where(finite, index, torch.full_like(index, float("nan"))),
            reasons=reasons,
            features=features,
        )

    @property
    def version(self) -> str:
        """A short hash of the policy's content, for naming it in results."""
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {**json.loads(json.dumps(asdict(self))), "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityAwareBudgetPolicy:
        """Rebuild a policy, refusing another type or content edited after it was written."""
        data = dict(data)
        recorded = data.pop("version", None)
        if data.get("policy_type") != QUALITY_AWARE:
            raise ValueError(f"unknown policy type {data.get('policy_type')!r}")
        try:
            policy = cls(**data)
        except TypeError as error:
            raise ValueError(f"not a quality-aware policy: {error}") from error
        if recorded is not None and recorded != policy.version:
            raise ValueError(
                f"policy content does not match its recorded version ({recorded}); "
                "it was edited after it was produced"
            )
        return policy


def calibrate_quality_aware(
    features: CompressibilityFeatures,
    drift: torch.Tensor,
    videos: Sequence[str],
    drivers: Sequence[str],
    levels: Sequence[int],
    tolerance: float = 0.10,
    alpha: float = 0.5,
    bins: int = 6,
    risk_quantile: float = 0.9,
    min_support: int = 30,
    min_videos: int = 3,
    motion_balance: float = 0.5,
    motion_weight: float = 0.5,
    probe_cells: tuple[int, int] = (2, 4),
    probe_top: int = 8,
    reference_points: int = REFERENCE_POINTS,
) -> QualityAwareBudgetPolicy:
    """Fit the risk table on calibration clips.

    Args:
        features: of each calibration clip, at the cut.
        drift: ``[clips, levels - 1]``, ``1 - cos(E_dense, E_level)`` for each level below the
            last.
        videos, drivers: the video and the driver of each clip, to count the support of
            each bin.
        levels: tokens per level, ascending, the last being the dense path.
        tolerance, alpha, bins, risk_quantile, min_support, min_videos, motion_balance,
            motion_weight, probe_cells, probe_top: see :class:`QualityAwareBudgetPolicy`.
        reference_points: quantiles kept per reference distribution.

    The clips must come from calibration videos only, split from any validation by driver.
    """
    levels = tuple(levels)
    uses_probe = alpha < 1.0
    motion = features.motion.detach().double().cpu().flatten()
    redundancy = features.redundancy.detach().double().cpu().flatten()
    probe = features.probe_p90.detach().double().cpu().flatten()
    drift = torch.as_tensor(drift, dtype=torch.float64).cpu()
    count = motion.numel()
    if drift.shape != (count, len(levels) - 1):
        raise ValueError(f"drift must be [{count}, {len(levels) - 1}], got {tuple(drift.shape)}")
    if len(videos) != count or len(drivers) != count:
        raise ValueError("every clip needs its video and its driver")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= count:
        raise ValueError("bins must be a positive integer, at most the number of clips")
    measured = [motion, redundancy, drift] + ([probe] if uses_probe else [])
    if not all(torch.isfinite(value).all() for value in measured):
        raise ValueError("calibration features and drift must be finite")

    quantiles = torch.linspace(0.0, 1.0, reference_points, dtype=torch.float64)
    motion_reference = tuple(torch.quantile(motion, quantiles).tolist())
    redundancy_reference = tuple(torch.quantile(redundancy, quantiles).tolist())
    score = motion_balance * _rank(motion, motion_reference) + (1.0 - motion_balance) * (
        1.0 - _rank(redundancy, redundancy_reference)
    )
    score_reference = tuple(torch.quantile(score, quantiles).tolist())
    probe_reference = tuple(torch.quantile(probe, quantiles).tolist()) if uses_probe else ()
    index = _quality_index(motion, redundancy, probe, motion_reference, redundancy_reference,
                           score_reference, probe_reference, motion_balance, alpha)
    edges = torch.quantile(index, torch.arange(1, bins, dtype=torch.float64) / bins)
    if any(b <= a for a, b in pairwise(edges.tolist())):
        raise ValueError("the calibration clips vary too little to separate the bins")

    where = torch.searchsorted(edges, index.contiguous(), right=True)
    risk, support, support_videos, support_drivers = [], [], [], []
    for b in range(bins):
        members = (where == b).nonzero(as_tuple=True)[0]
        support.append(int(members.numel()))
        support_videos.append(len({videos[i] for i in members.tolist()}))
        support_drivers.append(len({drivers[i] for i in members.tolist()}))
        if members.numel():
            row = torch.quantile(drift[members], risk_quantile, dim=0, interpolation="higher")
            risk.append(tuple(row.tolist()))
        else:
            risk.append((float("inf"),) * (len(levels) - 1))
    feature_range = [(float(motion.min()), float(motion.max())),
                     (float(redundancy.min()), float(redundancy.max()))]
    if uses_probe:
        feature_range.append((float(probe.min()), float(probe.max())))

    return QualityAwareBudgetPolicy(
        levels=levels, motion_reference=motion_reference,
        redundancy_reference=redundancy_reference, score_reference=score_reference,
        probe_reference=probe_reference, bin_edges=tuple(edges.tolist()), risk=tuple(risk),
        support=tuple(support), support_videos=tuple(support_videos),
        support_drivers=tuple(support_drivers), feature_range=tuple(feature_range),
        tolerance=tolerance, alpha=alpha, motion_balance=motion_balance,
        risk_quantile=risk_quantile, min_support=min_support, min_videos=min_videos,
        motion_weight=motion_weight, probe_cells=tuple(probe_cells), probe_top=probe_top,
    )


def _quality_index(motion, redundancy, probe, motion_reference, redundancy_reference,
                   score_reference, probe_reference, motion_balance, alpha) -> torch.Tensor:
    score = motion_balance * _rank(motion, motion_reference) + (1.0 - motion_balance) * (
        1.0 - _rank(redundancy, redundancy_reference)
    )
    index = _rank(score, score_reference)
    if alpha < 1.0:
        index = alpha * index + (1.0 - alpha) * _rank(probe, probe_reference)
    return index


def _rank(values: torch.Tensor, reference: tuple[float, ...]) -> torch.Tensor:
    """Position of each value in the reference distribution, in ``[0, 1]``.

    The reference values are evenly spaced quantiles, and the position is interpolated
    linearly between them, which approximates the cumulative distribution of the
    calibration clips. Monotonic; values outside the reference clamp to 0 or 1, and a
    value tied with a run of reference values sits in the middle of that run.
    """
    knots = torch.tensor(reference, dtype=values.dtype, device=values.device)
    last = len(reference) - 1
    values = values.contiguous()
    left = torch.searchsorted(knots, values, right=False)
    right = torch.searchsorted(knots, values, right=True)

    lower = knots[(left - 1).clamp(min=0)]
    upper = knots[left.clamp(max=last)]
    gap = upper - lower
    fraction = ((values - lower) / torch.where(gap > 0, gap, torch.ones_like(gap))).clamp(0, 1)
    between = (left - 1).to(values.dtype) + fraction
    tied = (left + right - 1).to(values.dtype) / 2
    position = torch.where(right > left, tied, between).clamp(0, last)
    return position / last


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_count(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _in_unit(value: object) -> bool:
    return _is_real(value) and isfinite(value) and 0.0 <= value <= 1.0
