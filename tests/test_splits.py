import pytest

from cofi_gene.splits import (normalize_undirected_edges, split_edges,
                              split_perturbation_targets, validate_negative_edges)


def test_edge_dedup_direction_and_self_loop():
    assert normalize_undirected_edges([(2, 1), (1, 2), (1, 1), (3, 0)], 4) == ((0, 3), (1, 2))


def test_edge_split_is_deterministic_disjoint_and_complete():
    edges = [(i, (i + 1) % 20) for i in range(20)]
    a = split_edges(edges, 20, seed=17)
    b = split_edges(edges[::-1], 20, seed=17)
    assert a == b
    train, val, test = map(set, (a.train, a.val, a.test))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(normalize_undirected_edges(edges, 20))
    assert min(map(len, (train, val, test))) >= 1


def test_held_out_positive_must_not_be_sampled_as_negative():
    with pytest.raises(ValueError, match="observed positive"):
        validate_negative_edges([(2, 1)], [(1, 2), (2, 3)], 5)
    with pytest.raises(ValueError, match="self loops"):
        validate_negative_edges([(1, 1)], [(2, 3)], 5)


def test_unseen_perturbation_split_never_leaks_across_contexts():
    # Multiple cell lines with identical target must remain in one group.
    targets = [f"G{i}" for i in range(30)]
    records = [(target, context) for target in targets for context in ("K562", "RPE1")]
    split = split_perturbation_targets(target for target, _ in records)
    partitions = [set(split.train), set(split.val), set(split.test)]
    assert set.union(*partitions) == set(targets)
    assert all(partitions[i].isdisjoint(partitions[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(sum(target in part for part in partitions) == 1 for target, _ in records)


def test_tiny_splits_raise_instead_of_silently_empty_validation():
    with pytest.raises(ValueError, match="not enough"):
        split_perturbation_targets(["A", "B"])
    with pytest.raises(ValueError, match="not enough"):
        split_edges([(0, 1), (1, 2)], 3)
