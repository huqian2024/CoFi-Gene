"""M5.1 negative sampling: cross-split exclusion and the three strategies.

Permanent guard for finding B-01: before M5.1 each split was sampled independently
against ``known_positives`` only, so a pair could be a training negative *and* a
test negative (server audit: 12 pairs in channel A, 354 in channel B).
"""

from __future__ import annotations

import numpy as np
import pytest

from cofi_gene.graph_build import index_array_to_pairs
from cofi_gene.negative_sampling import (FALLBACK_POLICIES, NEGATIVE_STRATEGIES,
                                         degrees_from_edges, negative_pair_count,
                                         sample_split_negatives)

COUNTS = {"train": 40, "val": 12, "test": 12}
STRATEGIES = NEGATIVE_STRATEGIES


def _ring(num_nodes: int) -> tuple[set, np.ndarray]:
    """Ring graph: its canonical positive set and the ``[2, E]`` training array."""
    pairs = {((i), (i + 1) % num_nodes) for i in range(num_nodes)}
    canonical = {(min(u, v), max(u, v)) for u, v in pairs}
    return canonical, np.asarray(sorted(canonical), dtype=np.int64).T.copy()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_splits_are_pairwise_disjoint_and_avoid_positives(strategy):
    positives, edges = _ring(60)
    reference = edges if strategy != "uniform_random" else None
    partition = sample_split_negatives(60, COUNTS, positives, seed=11, strategy=strategy,
                                       reference_edges=reference)
    sampled = partition.negatives
    assert {key: len(value) for key, value in sampled.items()} == COUNTS
    for split in ("train", "val", "test"):
        assert set(sampled[split]).isdisjoint(positives), split
        assert len(set(sampled[split])) == len(sampled[split]), "duplicate within split"
        for u, v in sampled[split]:
            assert u != v and u < v, "self-loop or non-canonical pair"
    assert partition.stats["cross_split_overlap"] == {"train_val": 0, "train_test": 0,
                                                      "val_test": 0}
    every_pair = set(sampled["train"]) | set(sampled["val"]) | set(sampled["test"])
    assert len(every_pair) == sum(COUNTS.values())


def test_sampling_is_deterministic_for_the_same_seed_and_strategy():
    positives, edges = _ring(40)
    for strategy in STRATEGIES:
        reference = edges if strategy != "uniform_random" else None
        first = sample_split_negatives(40, COUNTS, positives, seed=5, strategy=strategy,
                                       reference_edges=reference).negatives
        second = sample_split_negatives(40, COUNTS, positives, seed=5, strategy=strategy,
                                        reference_edges=reference).negatives
        assert first == second, strategy
        other = sample_split_negatives(40, COUNTS, positives, seed=6, strategy=strategy,
                                       reference_edges=reference).negatives
        assert other != first, strategy


def test_train_negatives_do_not_depend_on_the_evaluation_sizes():
    """The evaluation size must not silently reshuffle what the model trains on."""
    positives, _ = _ring(50)
    small = sample_split_negatives(50, {"train": 10, "val": 2, "test": 2}, positives,
                                   seed=3)
    large = sample_split_negatives(50, {"train": 10, "val": 9, "test": 9}, positives,
                                   seed=3)
    assert small.negatives["train"] == large.negatives["train"]


def test_impossible_request_raises_with_an_actionable_message():
    """A dense graph cannot supply disjoint negatives; fail loudly, not silently."""
    positives = {(u, v) for u in range(6) for v in range(u + 1, 6) if (u + v) % 3}
    with pytest.raises(ValueError, match="lower negatives_ratio"):
        sample_split_negatives(6, {"train": 20}, positives, seed=1)
    with pytest.raises(ValueError, match="fallback"):
        sample_split_negatives(6, {"train": 20}, positives, seed=1, fallback="error")


def test_degree_matched_reports_the_match_rate_and_hits_the_target_degrees():
    positives, edges = _ring(80)
    partition = sample_split_negatives(80, {"train": 60}, positives, seed=9,
                                       strategy="degree_matched", reference_edges=edges)
    stats = partition.stats["splits"]["train"]
    assert stats["sampled"] == 60 and stats["matched_draws"] > 0
    # Ring graph: every node has degree 2, so matching must succeed exactly.
    assert stats["fallback_rate"] == 0.0
    assert stats["mean_endpoint_degree"] == pytest.approx(
        stats["mean_positive_endpoint_degree"])


def test_two_hop_prefers_locally_close_pairs():
    positives, edges = _ring(40)
    partition = sample_split_negatives(40, {"train": 30}, positives, seed=4,
                                       strategy="two_hop", reference_edges=edges,
                                       fallback="uniform")
    neighbours = {i: {(i - 1) % 40, (i + 1) % 40} for i in range(40)}
    close = sum(1 for u, v in partition.negatives["train"]
                if v in {w for x in neighbours[u] for w in neighbours[x]})
    assert close >= 0.5 * len(partition.negatives["train"])


def test_fallback_is_recorded_and_bounded():
    """A tiny candidate pool degrades to uniform, and the rate is reported."""
    positives, edges = _ring(10)
    partition = sample_split_negatives(10, {"train": 20, "val": 5}, positives, seed=2,
                                       strategy="two_hop", reference_edges=edges,
                                       fallback="uniform")
    stats = partition.stats["splits"]
    assert stats["train"]["sampled"] == 20
    assert (stats["train"]["fallback_draws"] + stats["train"]["matched_draws"]
            == stats["train"]["sampled"])
    assert partition.stats["fallback"] == "uniform"
    assert set(partition.negatives["train"]).isdisjoint(positives)


def test_bad_arguments_are_rejected():
    positives, _ = _ring(20)
    with pytest.raises(ValueError, match="strategy must be"):
        sample_split_negatives(20, {"train": 5}, positives, strategy="magic")
    with pytest.raises(ValueError, match="fallback must be"):
        sample_split_negatives(20, {"train": 5}, positives, fallback="magic")
    with pytest.raises(ValueError, match="unknown split"):
        sample_split_negatives(20, {"training": 5}, positives)
    with pytest.raises(ValueError, match="non-negative int"):
        sample_split_negatives(20, {"train": -1}, positives)
    with pytest.raises(ValueError, match="needs the training"):
        sample_split_negatives(20, {"train": 5}, positives, strategy="two_hop")
    with pytest.raises(ValueError, match="at least two nodes"):
        sample_split_negatives(1, {"train": 1}, positives)
    assert FALLBACK_POLICIES == ("uniform", "relaxed_degree", "error")


def test_degrees_come_from_the_given_training_edges_only():
    """Held-out edges must never inflate the degree used for hard negatives."""
    train = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    held_out = np.asarray([[2], [5]], dtype=np.int64)
    train_degree = degrees_from_edges(train, 6)
    full_degree = degrees_from_edges(np.concatenate([train, held_out], axis=1), 6)
    assert train_degree.tolist() == [1, 2, 1, 0, 0, 0]
    assert full_degree[5] == 1 and train_degree[5] == 0


def test_negative_pair_count_matches_the_configured_ratio():
    assert negative_pair_count(10, 1.0) == 10
    assert negative_pair_count(10, 5.0) == 50
    assert negative_pair_count(0, 2.0) == 0
    with pytest.raises(ValueError):
        negative_pair_count(-1, 1.0)
    with pytest.raises(ValueError):
        negative_pair_count(1, -1.0)


def test_pipeline_build_records_a_mutually_exclusive_partition(processed_dataset):
    """The real build path (not just the helper) must produce disjoint splits."""
    for channel in processed_dataset.channel_names:
        part = processed_dataset.channel(channel)
        recorded = part.stats["negative_sampling"]
        assert recorded["mutually_exclusive"] is True
        assert recorded["strategy"] == "uniform_random"
        assert recorded["cross_split_overlap"] == {"train_val": 0, "train_test": 0,
                                                   "val_test": 0}
        train = set(index_array_to_pairs(part.neg_train))
        val = set(index_array_to_pairs(part.neg_val))
        test = set(index_array_to_pairs(part.neg_test))
        assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
        assert (train | val | test).isdisjoint(part.known_positives())

