"""The contract a backbone has to satisfy.

The attributes below are its single definition; the encoder and the compatibility
check read them from here instead of restating them.

The encoder does not call ``backbone.forward()``. It runs the blocks one at a time,
because pruning has to happen between two of them. That is why the contract lists
parts rather than a single entry point, and why a model that only exposes ``forward``
cannot be used without refactoring it first.
"""

from __future__ import annotations

#: Attributes the encoder reads. The first two are required; the rest are used when
#: present, which is what allows a backbone without a positional embedding or
#: without a final normalisation to work unchanged.
REQUIRED_ATTRIBUTES = ("patch_embed", "blocks")
OPTIONAL_ATTRIBUTES = ("pos_embed", "pos_drop", "norm", "fc_norm")


def missing_attributes(backbone: object) -> tuple[str, ...]:
    """Required attributes the backbone does not have.

    Returns an empty tuple when the contract is satisfied.
    """
    return tuple(name for name in REQUIRED_ATTRIBUTES if not hasattr(backbone, name))


def present_optional(backbone: object) -> tuple[str, ...]:
    """Optional attributes the backbone does have.

    Useful in reports: it records which parts of the forward pass were reproduced.
    """
    return tuple(name for name in OPTIONAL_ATTRIBUTES if getattr(backbone, name, None) is not None)
