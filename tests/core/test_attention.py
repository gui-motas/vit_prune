"""The anchor here is equivalence: with no sizes, the new path has to reproduce the
original block exactly. Without that guarantee, any difference measured later would
be ambiguous between an effect of the method and a defect in the code.
"""

import pytest
import torch

from vitprune.core.attention import attend, run_block
from vitprune.core.testing import ReferenceAttention, ReferenceBlock


@pytest.fixture
def block():
    torch.manual_seed(0)
    return ReferenceBlock(width=16, heads=4).double().eval()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 12, 16, dtype=torch.float64)


def test_without_sizes_attention_reproduces_the_original(block, x):
    expected = block.attn(block.norm1(x))
    actual = attend(block.attn, block.norm1(x), log_sizes=None)
    assert torch.allclose(expected, actual, atol=1e-10)


def test_without_sizes_the_whole_block_reproduces_the_original(block, x):
    assert torch.allclose(block(x), run_block(block, x, log_sizes=None), atol=1e-10)


def test_sizes_of_one_change_nothing(block, x):
    """A size of 1 means the token stands only for itself; its log is zero and the
    bias vanishes by construction."""
    zeros = torch.zeros(x.shape[:2], dtype=torch.float64)
    assert torch.allclose(run_block(block, x, None), run_block(block, x, zeros), atol=1e-10)


def test_a_larger_size_increases_a_token_influence(block, x):
    sizes = torch.zeros(x.shape[:2], dtype=torch.float64)
    neutral = attend(block.attn, x, log_sizes=sizes)

    sizes[:, 3] = torch.log(torch.tensor(50.0, dtype=torch.float64))
    boosted = attend(block.attn, x, log_sizes=sizes)

    assert not torch.allclose(neutral, boosted, atol=1e-6)


def test_a_dominant_size_collapses_every_output_onto_it(block, x):
    """With an extreme size on one token, all queries attend almost solely to it and
    the outputs become nearly identical."""
    sizes = torch.full(x.shape[:2], -50.0, dtype=torch.float64)
    sizes[:, 5] = 50.0

    out = attend(block.attn, x, log_sizes=sizes)

    assert out.std(dim=1).max() < 1e-6


def test_it_works_without_the_split_biases():
    """Not every backbone separates query and value biases the way VideoMAE does."""
    torch.manual_seed(2)
    attn = ReferenceAttention(width=8, heads=2, with_split_bias=False).double().eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64)

    assert torch.allclose(attn(x), attend(attn, x, None), atol=1e-10)


@pytest.mark.parametrize("width,heads", [(16, 4), (24, 3), (8, 2), (176, 2)])
def test_padding_to_a_multiple_of_eight_does_not_alter_the_result(width, heads):
    """The padding exists only to keep the fast backend eligible; it is zeros and
    must not contribute."""
    torch.manual_seed(3)
    attn = ReferenceAttention(width, heads).double().eval()
    x = torch.randn(1, 10, width, dtype=torch.float64)
    zeros = torch.zeros(1, 10, dtype=torch.float64)

    assert torch.allclose(attend(attn, x, None), attend(attn, x, zeros), atol=1e-10)
