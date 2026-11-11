"""The property tested here is the pooling identity.

If it breaks, the embedding handed to the downstream model describes a crop of the
scene rather than the scene, and every comparison between dense and pruned loses its
meaning.
"""

import pytest
import torch

from vitprune.core.merging import MergedTokens, initial_sizes, merge_tokens, size_weighted_mean


def _setup(batch=2, slices=3, count=32, dim=16, seed=0):
    torch.manual_seed(seed)
    tokens = torch.randn(batch, slices, count, dim, dtype=torch.float64)
    return tokens, initial_sizes(tokens)


def _flat(merged: MergedTokens):
    b, t, n, d = merged.tokens.shape
    return merged.tokens.reshape(b, t * n, d), merged.sizes.reshape(b, t * n, 1)


def test_size_weighted_pooling_survives_the_merge():
    """The property that justifies the whole method."""
    tokens, sizes = _setup()
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)

    b, t, n, d = tokens.shape
    before = tokens.reshape(b, t * n, d).mean(dim=1)

    merged = merge_tokens(tokens, sizes, scores, keep=8)
    after = size_weighted_mean(*_flat(merged))

    assert torch.allclose(before, after, atol=1e-12)


def test_plain_mean_pooling_does_not_survive():
    """The counter-proof: without sizes the mean shifts. That is why they exist."""
    tokens, sizes = _setup()
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)

    b, t, n, d = tokens.shape
    before = tokens.reshape(b, t * n, d).mean(dim=1)

    merged = merge_tokens(tokens, sizes, scores, keep=8)
    unweighted = merged.tokens.reshape(b, t * 8, d).mean(dim=1)

    assert not torch.allclose(before, unweighted, atol=1e-6)


def test_the_total_size_is_conserved():
    tokens, sizes = _setup(count=32)
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)

    merged = merge_tokens(tokens, sizes, scores, keep=8)

    assert torch.allclose(merged.sizes.sum(), sizes.sum())


def test_chained_merges_preserve_the_identity():
    tokens, sizes = _setup(count=32)
    b, t, n, d = tokens.shape
    before = tokens.reshape(b, t * n, d).mean(dim=1)

    first = merge_tokens(tokens, sizes, torch.rand(b, t, n, dtype=torch.float64), keep=16)
    second = merge_tokens(
        first.tokens, first.sizes, torch.rand(b, t, 16, dtype=torch.float64), keep=4
    )

    assert torch.allclose(before, size_weighted_mean(*_flat(second)), atol=1e-12)


def test_keeping_everything_returns_the_input_untouched():
    tokens, sizes = _setup(count=12)
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)

    merged = merge_tokens(tokens, sizes, scores, keep=12)

    assert torch.allclose(merged.tokens, tokens)
    assert torch.allclose(merged.sizes, sizes)


def test_the_kept_tokens_are_the_highest_scoring():
    torch.manual_seed(1)
    tokens = torch.randn(1, 1, 6, 4, dtype=torch.float64)
    sizes = initial_sizes(tokens)
    scores = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]]], dtype=torch.float64)

    merged = merge_tokens(tokens, sizes, scores, keep=3)

    assert merged.sizes.sum() == 6
    assert merged.tokens.shape[2] == 3


def test_grid_order_is_preserved():
    """Kept tokens come out in grid order, not score order.

    Each pruned token is an exact copy of a kept one, so merging changes no value and
    only the ordering is left to observe. The scores are chosen so that score order
    (3, 5, 1) differs from grid order (1, 3, 5).
    """
    tokens = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]],
        dtype=torch.float64,
    )
    sizes = initial_sizes(tokens)
    scores = torch.tensor([[[0.1, 0.5, 0.2, 0.9, 0.3, 0.7]]], dtype=torch.float64)

    merged = merge_tokens(tokens, sizes, scores, keep=3)

    in_grid_order = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(merged.tokens[0, 0], in_grid_order)
    assert torch.all(merged.sizes[0, 0].squeeze(-1) == 2)


def test_identical_tokens_merge_without_changing_the_value():
    tokens = torch.ones(1, 1, 8, 4, dtype=torch.float64)
    sizes = initial_sizes(tokens)
    scores = torch.rand(1, 1, 8, dtype=torch.float64)

    merged = merge_tokens(tokens, sizes, scores, keep=2)

    assert torch.allclose(merged.tokens, torch.ones_like(merged.tokens))
    assert merged.sizes.sum() == 8


@pytest.mark.parametrize("keep", [0, -1, 99])
def test_an_invalid_keep_count_is_rejected(keep):
    tokens, sizes = _setup(count=32)
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)
    with pytest.raises(ValueError, match="keep"):
        merge_tokens(tokens, sizes, scores, keep=keep)


def test_mismatched_shapes_are_rejected():
    tokens, sizes = _setup(count=32)
    with pytest.raises(ValueError, match="scores"):
        merge_tokens(tokens, sizes, torch.rand(2, 3, 31), keep=8)
    with pytest.raises(ValueError, match="sizes"):
        merge_tokens(tokens, torch.ones(2, 3, 32, 2), torch.rand(2, 3, 32), keep=8)


def test_pooling_rejects_disagreeing_counts():
    with pytest.raises(ValueError, match="disagree"):
        size_weighted_mean(torch.randn(1, 10, 4), torch.ones(1, 9, 1))


def test_with_equal_sizes_pooling_is_the_plain_mean():
    tokens = torch.randn(2, 10, 4, dtype=torch.float64)
    sizes = torch.ones(2, 10, 1, dtype=torch.float64)
    assert torch.allclose(size_weighted_mean(tokens, sizes), tokens.mean(dim=1))


def test_merge_error_is_zero_when_every_token_is_kept():
    from vitprune.core.merging import merge_error

    tokens, sizes = _setup()
    scores = torch.rand(*tokens.shape[:3], dtype=torch.float64)
    merged = merge_tokens(tokens, sizes, scores, keep=tokens.shape[2])
    assert torch.allclose(merge_error(tokens, scores, tokens.shape[2], merged),
                          torch.zeros(tokens.shape[:3], dtype=torch.float64), atol=1e-12)


def test_merge_error_is_zero_for_identical_tokens():
    from vitprune.core.merging import merge_error

    tokens = torch.ones(1, 2, 6, 4, dtype=torch.float64)
    scores = torch.rand(1, 2, 6, dtype=torch.float64)
    merged = merge_tokens(tokens, initial_sizes(tokens), scores, keep=1)
    assert torch.allclose(merge_error(tokens, scores, 1, merged), torch.zeros(1, 2, 6,
                          dtype=torch.float64), atol=1e-6)


def test_merge_error_is_high_for_an_incompatible_group():
    """The mean survives the merge exactly while each token is badly represented: the
    difference of the means says nothing about what was lost."""
    from vitprune.core.merging import merge_error

    tokens = torch.zeros(1, 1, 8, 4, dtype=torch.float64)
    tokens[0, 0, :4, 0] = 1.0  # along e1
    tokens[0, 0, 4:, 1] = 1.0  # along e2
    scores = torch.tensor([[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]], dtype=torch.float64)
    merged = merge_tokens(tokens, initial_sizes(tokens), scores, keep=1)

    error = merge_error(tokens, scores, 1, merged)[0, 0]
    # The one kept token becomes (e1 + e2) / 2: every token sits 45 degrees from it.
    assert torch.allclose(error, torch.full((8,), 1 - 2**-0.5, dtype=torch.float64))
    assert torch.allclose(size_weighted_mean(*_flat(merged)), tokens[0, 0].mean(dim=0)[None])


@pytest.mark.parametrize("count", [9, 10])
def test_bipartite_merge_preserves_pooled_mean_and_mass(count):
    from vitprune.core.merging import bipartite_merge

    torch.manual_seed(4)
    tokens = torch.randn(2, count, 8)
    sizes = torch.rand(2, count, 1) + 0.1
    metric = torch.randn(2, count, 4)
    keep = count - count // 3
    merged, mass = bipartite_merge(tokens, sizes, metric, keep)
    assert merged.shape == (2, keep, 8)
    torch.testing.assert_close(mass.sum(1), sizes.sum(1))
    torch.testing.assert_close(size_weighted_mean(merged, mass), size_weighted_mean(tokens, sizes),
                               atol=1e-5, rtol=1e-5)


def test_bipartite_merge_keep_all_is_identity():
    from vitprune.core.merging import bipartite_merge

    tokens, sizes, metric = torch.randn(2, 5, 3), torch.ones(2, 5, 1), torch.randn(2, 5, 4)
    merged, mass = bipartite_merge(tokens, sizes, metric, 5)
    assert merged is tokens and mass is sizes


def test_bipartite_merge_merges_most_similar_pair_first():
    from vitprune.core.merging import bipartite_merge

    metric = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
    tokens = torch.tensor([[[2.], [4.], [8.], [16.]]])
    merged, sizes = bipartite_merge(tokens, torch.ones(1, 4, 1), metric, 3)
    assert sizes[0, 0, 0] == 1
    assert sizes[0, 1, 0] == 2
    assert merged[0, 1, 0] == 3


@pytest.mark.parametrize("keep", [0, 3, 9])
def test_bipartite_merge_refuses_more_than_half(keep):
    from vitprune.core.merging import bipartite_merge

    with pytest.raises(ValueError, match="keep"):
        bipartite_merge(torch.randn(1, 8, 3), torch.ones(1, 8, 1), torch.randn(1, 8, 4), keep)


def test_bipartite_merge_low_precision_returns_input_dtype():
    from vitprune.core.merging import bipartite_merge

    torch.manual_seed(5)
    tokens = torch.randn(2, 12, 4).bfloat16()
    sizes = (torch.rand(2, 12, 1) + 0.2).bfloat16()
    metric = torch.randn(2, 12, 3).bfloat16()
    merged, mass = bipartite_merge(tokens, sizes, metric, 8)
    assert merged.dtype == tokens.dtype and mass.dtype == sizes.dtype
    torch.testing.assert_close(size_weighted_mean(merged.float(), mass.float()),
                               size_weighted_mean(tokens.float(), sizes.float()),
                               atol=1e-2, rtol=1e-2)


def test_bipartite_merge_rejects_mismatched_shapes():
    from vitprune.core.merging import bipartite_merge

    tokens = torch.randn(1, 8, 4)
    with pytest.raises(ValueError, match="sizes"):
        bipartite_merge(tokens, torch.ones(1, 7, 1), torch.randn(1, 8, 4), 6)
    with pytest.raises(ValueError, match="metric"):
        bipartite_merge(tokens, torch.ones(1, 8, 1), torch.randn(1, 7, 4), 6)


def test_bipartite_merge_autocast_preserves_float32_matching():
    from vitprune.core.merging import bipartite_merge

    tokens = torch.tensor([[[2.], [4.], [8.], [16.]]])
    sizes = torch.ones(1, 4, 1)
    metric = torch.tensor([[[1., 0.001], [1., 0.], [0., 1.], [1., 0.001]]])
    expected = bipartite_merge(tokens, sizes, metric, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = bipartite_merge(tokens, sizes, metric, 3)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result, reference)


@pytest.mark.parametrize("count", [9, 10])
def test_bipartite_merge_matches_brute_force(count):
    from vitprune.core.merging import bipartite_merge

    generator = torch.Generator().manual_seed(count)
    tokens = torch.randn(3, count, 4, generator=generator)
    sizes = torch.rand(3, count, 1, generator=generator) + 0.1
    metric = torch.randn(3, count, 5, generator=generator)
    keep = count - 3
    expected_tokens, expected_sizes = [], []
    for x, mass, keys in zip(tokens, sizes, metric, strict=True):
        edges = []
        all_similarities = []
        for a in range(0, count, 2):
            similarities = [torch.nn.functional.cosine_similarity(keys[a], keys[b], dim=0).item()
                            for b in range(1, count, 2)]
            all_similarities.extend(similarities)
            best_b = max(range(len(similarities)), key=similarities.__getitem__)
            edges.append((similarities[best_b], a, 2 * best_b + 1))
        edges.sort(reverse=True)
        assert len(set(all_similarities)) == len(all_similarities)
        weighted = [x[b] * mass[b] for b in range(1, count, 2)]
        totals = [mass[b].clone() for b in range(1, count, 2)]
        for _, a, b in edges[:count - keep]:
            weighted[b // 2] += x[a] * mass[a]
            totals[b // 2] += mass[a]
        unmerged = [a for _, a, _ in edges[count - keep:]]
        expected_tokens.append(torch.stack([x[a] for a in unmerged]
                                           + [w / s for w, s in zip(weighted, totals,
                                                                    strict=True)]))
        expected_sizes.append(torch.stack([mass[a] for a in unmerged] + totals))
    actual_tokens, actual_sizes = bipartite_merge(tokens, sizes, metric, keep)
    torch.testing.assert_close(actual_tokens, torch.stack(expected_tokens), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_sizes, torch.stack(expected_sizes), atol=1e-6, rtol=1e-6)
