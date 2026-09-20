"""Checks whether a backbone can be driven by the encoder, before anything else.

An incompatible backbone rarely fails loudly. It usually runs, produces embeddings of
the right shape, and only shows up later as degraded accuracy with no obvious cause.
By then the degradation is indistinguishable from the pruning itself, which is the
one thing the experiment is trying to measure.

So the check runs first, and it runs in two stages:

**Structure.** Does the backbone expose the attributes the encoder reads?

**Behaviour.** With pruning disabled, does the encoder reproduce the backbone's own
forward pass? This is the stage that matters. It catches a wrong token grid, a
positional embedding applied in a different place, and, most importantly, a model
that summarises through a class token rather than by averaging tokens.

That last case deserves a note. The merge guarantees that size-weighted pooling over
the kept tokens equals mean pooling over all of them. A class token is not a mean: it
is a token that reads the others through attention. Pruning changes what it sees, and
no bookkeeping recovers that. The method still runs, but the property that justifies
it is gone -- and the behaviour check is where that surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vitprune.core.geometry import ClipGeometry
from vitprune.core.protocols import missing_attributes, present_optional

#: Tolerance for the dense comparison. Generous enough to absorb reduction-order
#: differences in floating point, tight enough that a real mismatch fails.
DEFAULT_TOLERANCE = 1e-5


@dataclass(frozen=True)
class CompatibilityReport:
    """What the check found."""

    structure_ok: bool
    missing: tuple[str, ...]
    optional_present: tuple[str, ...]
    tokens_produced: int | None
    tokens_expected: int
    dense_matches_reference: bool | None
    max_difference: float | None

    @property
    def ok(self) -> bool:
        grid_ok = self.tokens_produced == self.tokens_expected
        behaviour_ok = self.dense_matches_reference is not False
        return self.structure_ok and grid_ok and behaviour_ok

    def describe(self) -> str:
        lines = []
        if self.missing:
            lines.append(f"missing required attributes: {', '.join(self.missing)}")
        else:
            lines.append("structure: all required attributes present")
            lines.append(f"optional in use: {', '.join(self.optional_present) or 'none'}")

        if self.tokens_produced is not None:
            verdict = "matches" if self.tokens_produced == self.tokens_expected else "MISMATCH"
            lines.append(
                f"token grid: backbone produced {self.tokens_produced}, "
                f"geometry expects {self.tokens_expected} ({verdict})"
            )

        if self.dense_matches_reference is None:
            lines.append("behaviour: not checked, no reference forward pass supplied")
        elif self.dense_matches_reference:
            lines.append(
                f"behaviour: dense path matches reference (max diff {self.max_difference:.2e})"
            )
        else:
            lines.append(f"behaviour: DIFFERS from reference (max diff {self.max_difference:.2e})")
        return "\n".join(lines)


def check_backbone(
    backbone: torch.nn.Module,
    geometry: ClipGeometry | None = None,
    reference: Callable[[torch.Tensor], torch.Tensor] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    device: str = "cpu",
    raise_on_failure: bool = True,
) -> CompatibilityReport:
    """Verify that a backbone can be driven by :class:`~vitprune.core.encoder.PrunedEncoder`.

    Args:
        backbone: the model to check.
        geometry: the token grid it is expected to produce.
        reference: the backbone's own forward pass, for example
            ``backbone.forward_features``. Supplying it is what enables the
            behaviour check, which is the part worth having.
        tolerance: maximum absolute difference accepted in that comparison.
        device: where to run the probe.
        raise_on_failure: raise instead of returning a failed report.

    Returns:
        A :class:`CompatibilityReport`.

    Raises:
        ValueError: when the check fails and ``raise_on_failure`` is set.

    The probe runs a single clip of random values. That is enough: the check is about
    wiring, not accuracy, and wiring does not depend on the content.
    """
    geometry = geometry or ClipGeometry.videomae_giant()
    missing = missing_attributes(backbone)

    report = _run_checks(backbone, geometry, reference, tolerance, device, missing)

    if raise_on_failure and not report.ok:
        raise ValueError(
            "the backbone is not compatible with the encoder.\n"
            + report.describe()
            + "\n\nSee docs/integration.md for what the encoder requires."
        )
    return report


def _run_checks(backbone, geometry, reference, tolerance, device, missing):
    if missing:
        return CompatibilityReport(
            structure_ok=False,
            missing=missing,
            optional_present=present_optional(backbone),
            tokens_produced=None,
            tokens_expected=geometry.num_tokens,
            dense_matches_reference=None,
            max_difference=None,
        )

    probe = torch.randn(*geometry.input_shape(batch=1), device=device, dtype=_dtype_of(backbone))
    tokens_produced = _count_tokens(backbone, probe)

    matches, difference = None, None
    if reference is not None and tokens_produced == geometry.num_tokens:
        matches, difference = _compare_dense(backbone, geometry, probe, reference, tolerance)

    return CompatibilityReport(
        structure_ok=True,
        missing=(),
        optional_present=present_optional(backbone),
        tokens_produced=tokens_produced,
        tokens_expected=geometry.num_tokens,
        dense_matches_reference=matches,
        max_difference=difference,
    )


@torch.no_grad()
def _count_tokens(backbone, probe: torch.Tensor) -> int:
    out = backbone.patch_embed(probe)
    if out.dim() == 5:
        out = out.flatten(2).transpose(1, 2)
    return int(out.shape[1])


def _dtype_of(backbone: torch.nn.Module) -> torch.dtype:
    """The dtype the backbone's own parameters use.

    The probe has to match it. Feeding single-precision input to a
    double-precision model fails with a message about bias types, which points
    nowhere near the actual cause.
    """
    for parameter in backbone.parameters():
        return parameter.dtype
    return torch.float32


@torch.no_grad()
def _compare_dense(backbone, geometry, probe, reference, tolerance):
    from vitprune.core.encoder import PrunedEncoder, PruningPlan

    expected = reference(probe)
    if isinstance(expected, tuple):
        expected = expected[0]

    actual = PrunedEncoder(backbone, geometry, PruningPlan.dense()).encode(probe)
    difference = float((expected - actual).abs().max())
    return difference <= tolerance, difference
