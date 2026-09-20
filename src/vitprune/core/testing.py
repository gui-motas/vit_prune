"""A reference backbone with the same contract as a real one, at a small scale.

A real video transformer has a billion parameters and needs a checkpoint of several
gigabytes. Neither fits in a test suite nor on a development machine without a GPU.

This module provides a small stand-in that honours exactly the same contract: the
same attribute names, the same order of operations, the same input and output shapes.
Everything the encoder needs to know about a real backbone is reproduced here in
miniature.

It serves two purposes. In tests, it allows verifying behaviour without depending on
large files. When integrating, it gives a known-good backbone to compare against: if
the encoder works here and not with your model, the difference is in the model.

What is reproduced is the structure, not the competence. This backbone recognises
nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vitprune.core.geometry import ClipGeometry


class ReferenceAttention(nn.Module):
    """Attention laid out as VideoMAE lays it out.

    The quirk reproduced here is bias handling: query and value have their own bias
    parameters and the key has none. That is why callers cannot simply invoke
    ``qkv(x)``.
    """

    def __init__(self, width: int, heads: int, with_split_bias: bool = True):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        self.num_heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(width, width * 3, bias=not with_split_bias)
        self.proj = nn.Linear(width, width)
        if with_split_bias:
            self.q_bias = nn.Parameter(torch.zeros(width))
            self.v_bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention written the explicit way, materialising the logit matrix.

        Deliberately the slow version: it exists to act as an oracle in tests, and it
        is how the real VideoMAE is written.
        """
        batch, count, width = x.shape
        q, k, v = self._qkv(x)
        logits = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (logits @ v).transpose(1, 2).reshape(batch, count, width)
        return self.proj(out)

    def _qkv(self, x: torch.Tensor):
        batch, count, _ = x.shape
        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias)
            merged = F.linear(x, self.qkv.weight, torch.cat((self.q_bias, k_bias, self.v_bias)))
        else:
            merged = self.qkv(x)
        merged = merged.reshape(batch, count, 3, self.num_heads, self.head_dim)
        return merged.permute(2, 0, 3, 1, 4).unbind(0)


class ReferenceBlock(nn.Module):
    """One transformer block: norm, attend, norm, project.

    The two residual connections are what allow dozens of these to be stacked
    without the signal degrading.
    """

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = ReferenceAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ReferencePatchEmbed(nn.Module):
    """Patch embedding in the same layout as a real one.

    The convolution sits in a field called ``proj`` rather than directly on this
    module. That looks like a detail, but it is the name that appears in checkpoints,
    and the name shape validation looks for.
    """

    def __init__(self, geometry: ClipGeometry):
        super().__init__()
        step = (geometry.tubelet_size, geometry.patch_size, geometry.patch_size)
        self.proj = nn.Conv3d(3, geometry.embed_dim, kernel_size=step, stride=step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ReferenceBackbone(nn.Module):
    """A complete video backbone in miniature.

    Satisfies the contract in :mod:`vitprune.core.protocols`: patch embedding, fixed
    positional embedding, a stack of blocks, and a final normalisation.
    """

    def __init__(self, geometry: ClipGeometry, depth: int | None = None):
        super().__init__()
        self.geometry = geometry
        self.patch_embed = ReferencePatchEmbed(geometry)
        self.register_buffer(
            "pos_embed",
            sinusoid_positions(geometry.num_tokens, geometry.embed_dim),
            persistent=False,
        )
        blocks = geometry.depth if depth is None else depth
        self.blocks = nn.ModuleList(
            ReferenceBlock(geometry.embed_dim, geometry.heads) for _ in range(blocks)
        )
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(geometry.embed_dim)

    def forward_features(self, clips: torch.Tensor) -> torch.Tensor:
        """The backbone's own forward pass, as a real one would expose it.

        Used as the reference in :func:`vitprune.core.compat.check_backbone`.
        """
        return reference_encode(self, clips)


def sinusoid_positions(count: int, width: int) -> torch.Tensor:
    """A fixed positional table, in the same scheme as the real model.

    Fixed rather than learned, which is why pruning tokens invalidates no parameter.
    """
    position = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    pair = torch.arange(width, dtype=torch.float32).div(2, rounding_mode="floor") * 2
    angle = position / torch.pow(10000.0, pair / width)
    table = torch.zeros(count, width)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.unsqueeze(0)


def reference_geometry(depth: int = 4) -> ClipGeometry:
    """A geometry small enough for a test, with the same structure as a real one.

    Produces 4 temporal slices of 16 tokens each, 64 in total, against 2048 for the
    Giant. The proportions and divisibility rules are the same.
    """
    return ClipGeometry(
        image_size=32,
        patch_size=8,
        tubelet_size=2,
        frames_per_clip=8,
        embed_dim=32,
        depth=depth,
        heads=4,
        mlp_ratio=2.0,
    )


def reference_backbone(geometry: ClipGeometry | None = None, seed: int = 0) -> ReferenceBackbone:
    """A ready-to-use reference backbone with reproducible weights."""
    torch.manual_seed(seed)
    return ReferenceBackbone(geometry or reference_geometry()).eval()


@torch.no_grad()
def reference_encode(backbone: nn.Module, clips: torch.Tensor) -> torch.Tensor:
    """The dense path, with no pruning at all.

    The oracle: with a dense plan the encoder has to reproduce this exactly. If it
    does not, there is a defect in the code, and any difference measured afterwards
    would be ambiguous.
    """
    out = backbone.patch_embed(clips)
    if out.dim() == 5:
        out = out.flatten(2).transpose(1, 2)
    if getattr(backbone, "pos_embed", None) is not None:
        out = out + backbone.pos_embed.type_as(out)
    for block in backbone.blocks:
        out = block(out)
    if getattr(backbone, "norm", None) is not None:
        out = backbone.norm(out)
    out = out.mean(dim=1)
    if getattr(backbone, "fc_norm", None) is not None:
        out = backbone.fc_norm(out)
    return out
