"""The contract a backbone has to satisfy.

Written as a Protocol rather than prose so that a type checker can verify it, and so
that the requirement has a single definition instead of being restated in several
docstrings.

The encoder does not call ``backbone.forward()``. It runs the blocks one at a time,
because pruning has to happen between two of them. That is why the contract lists
parts rather than a single entry point, and why a model that only exposes ``forward``
cannot be used without refactoring it first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import nn

#: Attributes the encoder reads. The first two are required; the rest are used when
#: present, which is what allows a backbone without a positional embedding or
#: without a final normalisation to work unchanged.
REQUIRED_ATTRIBUTES = ("patch_embed", "blocks")
OPTIONAL_ATTRIBUTES = ("pos_embed", "pos_drop", "norm", "fc_norm")


@runtime_checkable
class VideoBackbone(Protocol):
    """A video transformer whose blocks can be driven one at a time.

    ``patch_embed`` may return either ``[batch, embed_dim, slices, height, width]``,
    as a 3D convolution does, or ``[batch, tokens, embed_dim]`` already flattened.
    Both are accepted.

    ``blocks`` must be iterable and indexable, and each element callable as
    ``block(x) -> x`` with ``x`` shaped ``[batch, tokens, embed_dim]``.
    """

    blocks: Sequence[nn.Module]

    def patch_embed(self, clips: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class AttentionModule(Protocol):
    """The attention submodule inside a block.

    Only needed when proportional attention is enabled. VideoMAE keeps separate bias
    parameters for query and value and none for key, which is why the split cannot be
    assumed to come from a single ``qkv`` bias.
    """

    num_heads: int
    qkv: nn.Module
    proj: nn.Module


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
