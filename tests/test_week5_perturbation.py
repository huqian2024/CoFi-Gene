"""Week 5 (M5) protocol perturbations: missing edges, cross-view mask, derangement.

Numpy-only tests: they must run in the fast CI job without torch, because they
guard the experiment protocol (fixed evaluation set, leakage, determinism) rather
than any model maths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cofi_gene.config import load_config, validate_config
from cofi_gene.dataset import ChannelSplit, CoFiDataset
from cofi_gene.graph_build import index_array_to_pairs
from cofi_gene.perturbation import (apply_data_perturbations, channel_seed,
                                    cross_view_permutation,
                                    mask_cross_view_positives,
                                    permutation_fingerprint, thin_channel)

REPO = Path(__file__).resolve().parents[1]


def _edges(pairs, weights=None):
    """``[(u, v), ...]`` -> canonical ``[2, E]`` array plus a weight array."""
    inside = (np.asarray(pairs, dtype=np.int64).T.copy() if pairs
              else np.zeros((2, 0), dtype=np.int64))
    values = (np.asarray(weights, dtype=np.float32) if weights is not None
              else np.ones(inside.shape[1], dtype=np.float32))
    return inside, values


def _tiny_dataset() -> CoFiDataset:
    """Six genes where A's held-out pair (0,1) is a training edge of B."""
    a_train, a_train_w = _edges([(2, 3)], [0.4])
    a_val, a_val_w = _edges([(0, 1)], [0.8])
    a_test, a_test_w = _edges([(1, 2)], [0.6])
    b_train, b_train_w = _edges([(0, 1), (3, 4)], [0.5, 0.9])
    b_val, b_val_w = _edges([(1, 3)], [0.7])
    b_test, b_test_w = _edges([(2, 4)], [0.3])
    channel_a = ChannelSplit(
        train=a_train, train_weights=a_train_w, val=a_val, val_weights=a_val_w,
        test=a_test, test_weights=a_test_w,
        neg_train=_edges([(0, 5)])[0], neg_val=_edges([(4, 5)])[0],
        neg_test=_edges([(0, 4)])[0],
        encode_edges=a_train.copy(), encode_weights=a_train_w.copy())
    channel_b = ChannelSplit(
        train=b_train, train_weights=b_train_w, val=b_val, val_weights=b_val_w,
        test=b_test, test_weights=b_test_w,
        neg_train=_edges([(0, 2)])[0], neg_val=_edges([(2, 5)])[0],
        neg_test=_edges([(4, 5)])[0],
        encode_edges=b_train.copy(), encode_weights=b_train_w.copy())
    dataset = CoFiDataset(gene_ids=tuple(f"G{i}" for i in range(6)),
                          X=np.zeros((6, 3), dtype=np.float32),
                          channels={"A": channel_a, "B": channel_b})
    dataset.validate()
    return dataset


# ------------------------------------------------------------------ seeds
def test_channel_seed_is_stable_and_channel_specific():
    assert channel_seed(42, 0) == channel_seed(42, 0)
    assert channel_seed(42, 0) != channel_seed(42, 1)
    assert channel_seed(43, 0) != channel_seed(42, 0)
    assert 0 <= channel_seed(0, 0) < 2 ** 32


# --------------------------------------------------------------- thinning
def test_thinning_only_shrinks_encoder_edges_and_keeps_weights(processed_dataset):
    part = processed_dataset.channel("A")
    before_pairs = index_array_to_pairs(part.encode_edges)
    before_weights = {(u, v): float(w) for (u, v), w in
                      zip(before_pairs, np.asarray(part.encode_weights, dtype=np.float64))}
    thinned, stats = thin_channel(part, 0.3, seed=7,
                                  num_nodes=processed_dataset.num_nodes)

    kept = index_array_to_pairs(thinned.encode_edges)
    assert set(kept) <= set(before_pairs)
    assert stats["num_encode_edges_before"] == len(set(before_pairs))
    assert stats["num_encode_edges_after"] == len(kept)
    assert stats["num_encode_edges_removed"] == int(round(len(before_pairs) * 0.3))
    # Surviving edges keep their own weights (no shifting after a drop).
    for (u, v), weight in zip(kept, np.asarray(thinned.encode_weights, dtype=np.float64)):
        assert float(weight) == before_weights[(u, v)]
    # Nothing else in the channel moves.
    for attribute in ("train", "val", "test", "train_weights", "val_weights",
                      "test_weights", "neg_train", "neg_val", "neg_test"):
        assert np.array_equal(getattr(part, attribute), getattr(thinned, attribute)), attribute


def test_thinning_is_deterministic_and_nested_across_ratios(processed_dataset):
    part = processed_dataset.channel("B")
    num_nodes = processed_dataset.num_nodes
    first, _ = thin_channel(part, 0.1, seed=11, num_nodes=num_nodes)
    second, _ = thin_channel(part, 0.1, seed=11, num_nodes=num_nodes)
    assert np.array_equal(first.encode_edges, second.encode_edges)
    assert np.array_equal(first.encode_weights, second.encode_weights)

    kept = {ratio: set(index_array_to_pairs(
                thin_channel(part, ratio, seed=11, num_nodes=num_nodes)[0].encode_edges))
            for ratio in (0.1, 0.3, 0.5)}
    assert kept[0.5] <= kept[0.3] <= kept[0.1]


def test_zero_fraction_is_a_noop_and_bad_fractions_are_rejected(processed_dataset):
    part = processed_dataset.channel("A")
    unchanged, stats = thin_channel(part, 0.0, seed=3,
                                    num_nodes=processed_dataset.num_nodes)
    assert (index_array_to_pairs(unchanged.encode_edges)
            == index_array_to_pairs(part.encode_edges))
    assert np.allclose(unchanged.encode_weights, part.encode_weights)
    assert stats["num_encode_edges_removed"] == 0
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            thin_channel(part, bad, seed=3, num_nodes=processed_dataset.num_nodes)


def test_perturbed_dataset_still_passes_every_leakage_guard(processed_dataset):
    for ratio in (0.1, 0.3, 0.5):
        config = {"data": {"edge_drop_fraction": ratio}, "runtime": {"seed": 42}}
        perturbed, stats = apply_data_perturbations(processed_dataset, config)
        perturbed.validate()   # raises on any leak / overlap / bad negative
        assert stats["edge_drop_fraction"] == ratio
        for name in processed_dataset.channel_names:
            part = perturbed.channel(name)
            held_out = part.pairs("val") | part.pairs("test")
            assert held_out.isdisjoint(index_array_to_pairs(part.encode_edges))
            assert stats["channels"][name]["num_encode_edges_removed"] > 0



# ------------------------------------------------------ cross-view masking
def test_cross_view_masking_hides_the_pair_in_both_views():
    dataset = _tiny_dataset()
    masked, stats = mask_cross_view_positives(dataset)
    masked.validate()

    # A's held-out pair (0,1) was a training/encoding edge of B -> gone from both.
    assert (0, 1) not in index_array_to_pairs(masked.channel("B").encode_edges)
    assert (0, 1) not in index_array_to_pairs(masked.channel("B").train)
    assert stats["channels"]["B"]["num_encode_edges_removed"] == 1
    assert stats["channels"]["B"]["num_train_positives_removed"] == 1
    # B's remaining training edge keeps its own weight.
    pairs = index_array_to_pairs(masked.channel("B").encode_edges)
    assert pairs == ((3, 4),)
    assert float(np.asarray(masked.channel("B").encode_weights)[0]) == pytest.approx(0.9)
    # A had nothing to hide, and evaluation/negatives are untouched everywhere.
    assert stats["channels"]["A"]["num_encode_edges_removed"] == 0
    for name in ("A", "B"):
        for attribute in ("val", "test", "val_weights", "test_weights",
                          "neg_train", "neg_val", "neg_test"):
            assert np.array_equal(getattr(dataset.channel(name), attribute),
                                  getattr(masked.channel(name), attribute))
    # The original dataset object is not mutated.
    assert (0, 1) in index_array_to_pairs(dataset.channel("B").encode_edges)


def test_cross_view_masking_needs_two_views():
    single = CoFiDataset(gene_ids=("G0", "G1"), X=np.zeros((2, 2), dtype=np.float32),
                         channels={"A": _tiny_dataset().channel("A")})
    with pytest.raises(ValueError, match="two channels"):
        mask_cross_view_positives(single)


# ------------------------------------------------------- gene derangement
def test_cross_view_permutation_is_a_stable_derangement():
    for seed in (0, 1, 42, 2026):
        perm = cross_view_permutation(11, seed)
        assert sorted(perm.tolist()) == list(range(11))
        assert not np.any(perm == np.arange(11))
        assert np.array_equal(perm, cross_view_permutation(11, seed))
    assert not np.array_equal(cross_view_permutation(50, 1),
                              cross_view_permutation(50, 2))
    # Two genes can only be swapped.
    assert cross_view_permutation(2, 5).tolist() == [1, 0]
    with pytest.raises(ValueError, match="at least two genes"):
        cross_view_permutation(1, 5)


def test_permutation_fingerprint_tracks_the_mapping():
    first = cross_view_permutation(17, 42)
    assert permutation_fingerprint(first) == permutation_fingerprint(first.copy())
    assert permutation_fingerprint(first) != permutation_fingerprint(
        cross_view_permutation(17, 43))
    assert len(permutation_fingerprint(first)) == 16


# --------------------------------------------------------- config wiring
def test_week5_config_keys_are_optional_and_validated():
    base = load_config(REPO / "configs" / "b3.yaml")
    assert "edge_drop_fraction" not in base["data"]
    assert "mask_other_view" not in base["data"]
    assert "misalign_cross_view" not in base["data"]

    with_fraction = load_config(REPO / "configs" / "b3.yaml",
                                {"data": {"edge_drop_fraction": 0.3}})
    assert with_fraction["data"]["edge_drop_fraction"] == 0.3

    with pytest.raises(ValueError, match="edge_drop_fraction"):
        load_config(REPO / "configs" / "b3.yaml", {"data": {"edge_drop_fraction": 1.0}})
    with pytest.raises(ValueError, match="mask_other_view"):
        load_config(REPO / "configs" / "b3.yaml", {"data": {"mask_other_view": "yes"}})


def test_misalign_only_accepts_cross_view_models():
    for name in ("b3", "b4", "b5", "b6", "b7"):
        config = load_config(REPO / "configs" / f"{name}.yaml",
                             {"data": {"misalign_cross_view": True}})
        assert config["data"]["misalign_cross_view"] is True
    for name in ("b0", "b1", "b2"):
        with pytest.raises(ValueError, match="misalign_cross_view"):
            load_config(REPO / "configs" / f"{name}.yaml",
                        {"data": {"misalign_cross_view": True}})


def test_unknown_keys_are_still_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        load_config(REPO / "configs" / "b3.yaml", {"data": {"edge_drop_ratio": 0.3}})

