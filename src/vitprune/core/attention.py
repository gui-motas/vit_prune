"""Proportional attention: giving a merged token the influence of its size.

After a merge, a kept token may carry the size of twelve pruned ones. In the blocks
that follow, when other tokens attend to it, it would be consistent for it to have
twelve times the influence -- it is standing in for twelve.

The direct way to do that is to add ``log(size)`` to the attention logits. The problem
is that efficient attention kernels exist precisely so the logit matrix is never
materialised: they compute the result in tiles. Adding anything to that matrix forces
it into memory, and the speed advantage disappears -- which would be an odd trade in a
project about speed.

The alternative embeds the size in the compared vectors through a rank-1 augmentation:
a constant 1 appended to the query and ``log(size)`` appended to the key. Their inner
product contributes exactly the desired term with nothing materialised. Dimensions are
padded to a multiple of 8, which is what the fast backend requires.

This path is experimental and off by default. The pooling identity does not depend on
it; that comes from size-weighted pooling at the end. Enabling it is a hypothesis to
test, not a requirement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ALIGNMENT = 8


def run_block(block: torch.nn.Module, x: torch.Tensor, log_sizes: torch.Tensor | None):
    """Run one transformer block, optionally with the size bias.

    Args:
        block: a block in VideoMAE layout: norm, attention, norm, MLP, with residual
            connections around each half.
        x: ``[batch, tokens, embed_dim]``.
        log_sizes: ``[batch, tokens]`` holding the log of each token's size, or
            ``None`` for ordinary attention.

    Returns:
        ``[batch, tokens, embed_dim]``.

    With ``log_sizes`` set to ``None`` the result is identical to the block's own
    forward pass. The tests assert that, which makes it a safety net rather than a
    claim.
    """
    layer_scale_1 = getattr(block, "gamma_1", None)
    layer_scale_2 = getattr(block, "gamma_2", None)
    drop_path = getattr(block, "drop_path", None)
    if not callable(drop_path):
        drop_path = _identity

    attended = attend(block.attn, block.norm1(x), log_sizes)
    x = x + drop_path(attended if layer_scale_1 is None else layer_scale_1 * attended)

    projected = block.mlp(block.norm2(x))
    x = x + drop_path(projected if layer_scale_2 is None else layer_scale_2 * projected)
    return x


def attend(attention: torch.nn.Module, x: torch.Tensor, log_sizes: torch.Tensor | None):
    """Self-attention for one block, with an optional size bias.

    Args:
        attention: a module exposing ``num_heads``, ``qkv``, ``proj`` and optionally
            ``q_bias`` / ``v_bias``.
        x: ``[batch, tokens, embed_dim]``, already normalised.
        log_sizes: ``[batch, tokens]`` or ``None``.
    """
    batch, count, dim = x.shape
    heads = attention.num_heads
    head_dim = dim // heads
    scale = getattr(attention, "scale", head_dim**-0.5)

    query, key, value = _split_qkv(attention, x, heads, head_dim)

    if log_sizes is None:
        out = F.scaled_dot_product_attention(query, key, value, scale=scale)
    else:
        out = _proportional_attention(query, key, value, log_sizes, scale, heads, head_dim)

    out = out.transpose(1, 2).reshape(batch, count, dim)
    return attention.proj(out)


def _proportional_attention(query, key, value, log_sizes, scale, heads, head_dim):
    batch, count = log_sizes.shape
    bias = log_sizes.view(batch, 1, count, 1).expand(batch, heads, count, 1).to(query.dtype)

    query = torch.cat([query * scale, torch.ones_like(query[..., :1])], dim=-1)
    key = torch.cat([key, bias], dim=-1)

    padding = (-query.shape[-1]) % _ALIGNMENT
    if padding:
        filler = torch.zeros(*query.shape[:-1], padding, dtype=query.dtype, device=query.device)
        query = torch.cat([query, filler], dim=-1)
        key = torch.cat([key, filler], dim=-1)

    missing = query.shape[-1] - head_dim
    value = torch.cat(
        [value, torch.zeros(*value.shape[:-1], missing, dtype=value.dtype, device=value.device)],
        dim=-1,
    )

    # The scale is already folded into the query; applying it again would distort
    # the bias, which must stay additive in the logit.
    out = F.scaled_dot_product_attention(query, key, value, scale=1.0)
    return out[..., :head_dim]


def _split_qkv(attention: torch.nn.Module, x: torch.Tensor, heads: int, head_dim: int):
    """Split query, key and value.

    VideoMAE keeps query and value biases in separate parameters and uses none for
    the key. Handling that here keeps the detail from leaking into the rest of the
    code.
    """
    batch, count, _ = x.shape
    q_bias = getattr(attention, "q_bias", None)
    if q_bias is not None:
        v_bias = attention.v_bias
        k_bias = torch.zeros_like(v_bias, requires_grad=False)
        combined = F.linear(x, attention.qkv.weight, torch.cat((q_bias, k_bias, v_bias)))
    else:
        combined = attention.qkv(x)

    combined = combined.reshape(batch, count, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    return combined[0], combined[1], combined[2]


def _identity(value):
    return value
