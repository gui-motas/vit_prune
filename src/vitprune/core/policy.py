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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any

import torch

from vitprune.core.geometry import ClipGeometry
from vitprune.core.statistics import ClipStatistics


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
            if not values or not all(_is_real(v) and isfinite(v) for v in values):
                raise ValueError(f"{name} must be a non-empty sequence of finite numbers")
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
        slices = geometry.temporal_slices
        for level in self.levels:
            if level % slices or level > geometry.num_tokens:
                raise ValueError(
                    f"budget level {level} must be a multiple of the {slices} temporal "
                    f"slices and at most {geometry.num_tokens} tokens"
                )

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


def _rank(values: torch.Tensor, reference: tuple[float, ...]) -> torch.Tensor:
    """Mid-rank of each value among the reference values, in ``[0, 1]``.

    Monotonic, and ties are handled evenly: a value equal to every reference value
    sits at 0.5.
    """
    knots = torch.tensor(reference, dtype=values.dtype, device=values.device)
    values = values.contiguous()
    below = torch.searchsorted(knots, values, right=False)
    upto = torch.searchsorted(knots, values, right=True)
    return (below + upto).to(values.dtype) / (2 * len(reference))


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
