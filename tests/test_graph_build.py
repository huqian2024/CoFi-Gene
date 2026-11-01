"""Graph construction helpers: dedup, weight policy, statistics, overlap."""

import numpy as np
import pytest

from cofi_gene.graph_build import (build_channel_graph, canonicalize_pairs, graph_stats,
                                   index_array_to_pairs, normalize_edge_weights,
                                   overlap_stats, pairs_to_index_array,
                                   undirected_pair_keys)


def test_canonicalize_pairs_keeps_max_score_and_drops_self_loops():
    merged = canonicalize_pairs([(1, 2, 100), (2, 1, 250), (3, 3, 900), (0, 4, 50)])
    assert merged == {(1, 2): 250.0, (0, 4): 50.0}


def test_build_channel_graph_counts_drops_and_keeps_indices_in_range():
    protein_to_index = {"p0": 0, "p1": 1, "p2": 2}
    edges = [("p0", "p1", 700), ("p1", "p0", 500), ("p0", "p2", 100),
             ("p0", "unknown", 900), ("p1", "p1", 800), ("p1", "p2", 10)]
    graph = build_channel_graph(edges, protein_to_index, 3, min_score=50)
    assert graph.stats["n_rows_below_threshold"] == 1
    assert graph.stats["n_edges_unknown_protein"] == 1
    assert graph.stats["n_self_loops_dropped"] == 1
    assert graph.stats["n_duplicate_pairs_merged"] == 1
    assert graph.num_edges == 2
    assert index_array_to_pairs(graph.edge_index) == ((0, 1), (0, 2))
    assert graph.weights.tolist() == [700.0, 100.0]


def test_weight_modes():
    weights = [100.0, 300.0, 500.0]
    assert normalize_edge_weights(weights, "none").tolist() == weights
    scaled = normalize_edge_weights(weights, "minmax")
    assert scaled.min() == pytest.approx(0.0) and scaled.max() == pytest.approx(1.0)
    assert normalize_edge_weights([7.0], "minmax").tolist() == [1.0]
    with pytest.raises(ValueError, match="unknown weight mode"):
        normalize_edge_weights(weights, "zscore")
    with pytest.raises(ValueError, match="weight_mode"):
        build_channel_graph([], {}, 2, weight_mode="zscore")


def test_graph_stats_describe_isolated_nodes_and_density():
    edge_index = np.array([[0, 1], [1, 2]])
    stats = graph_stats(edge_index, 4, weights=[1.0, 2.0])
    assert stats["num_edges"] == 2
    assert stats["num_isolated_nodes"] == 1
    assert stats["max_degree"] == 2
    assert stats["weight_mean"] == pytest.approx(1.5)
    empty = graph_stats(np.zeros((2, 0)), 5)
    assert empty["num_edges"] == 0 and empty["num_isolated_nodes"] == 5
    assert empty["density"] == 0.0


def test_overlap_stats_uses_canonical_pairs():
    a = np.array([[0, 1, 2], [1, 2, 3]])
    b = np.array([[1, 9], [2, 8]])          # (1,2) shared, (9,8) unique
    stats = overlap_stats(a, b)
    assert stats["n_edges_a"] == 3 and stats["n_edges_b"] == 2
    assert stats["n_shared_edges"] == 1
    assert stats["jaccard"] == pytest.approx(1 / 4)
    assert stats["frac_a_in_b"] == pytest.approx(1 / 3)


def test_pairs_to_index_array_validates_ranges_and_canonicalises():
    array = pairs_to_index_array([(3, 1), (1, 3), (0, 0)], 4)
    assert array.shape == (2, 1)
    assert array[:, 0].tolist() == [1, 3]
    assert pairs_to_index_array([], 4).shape == (2, 0)
    with pytest.raises(ValueError, match="outside"):
        pairs_to_index_array([(0, 9)], 4)
    assert undirected_pair_keys(array) == frozenset({(1, 3)})
    with pytest.raises(ValueError, match="shape"):
        index_array_to_pairs(np.zeros(3))
