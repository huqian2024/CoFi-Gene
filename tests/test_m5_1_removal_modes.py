"""M5.1 edge-removal modes: adjacency dropout vs supervision-aware removal."""

from __future__ import annotations

import numpy as np
import pytest

from cofi_gene.graph_build import index_array_to_pairs
from cofi_gene.perturbation import (apply_data_perturbations, apply_edge_removal,
                                    channel_seed, remove_training_edges, thin_channel)


def test_adjacency_mode_only_shrinks_the_encoder_graph(processed_dataset):
    part = processed_dataset.channel("A")
    thinned, stats = remove_training_edges(part, 0.3, seed=7,
                                          num_nodes=processed_dataset.num_nodes,
                                          supervision=False)
    assert stats["mode"] == "adjacency"
    assert stats["supervision_changed"] is False
    assert set(index_array_to_pairs(thinned.encode_edges)) < set(
        index_array_to_pairs(part.encode_edges))
    assert np.array_equal(thinned.train, part.train)          # supervision kept
    assert np.array_equal(thinned.neg_train, part.neg_train)


def test_supervision_mode_removes_the_same_pairs_from_the_training_positives(
        processed_dataset):
    part = processed_dataset.channel("B")
    thinned, stats = remove_training_edges(part, 0.3, seed=7,
                                          num_nodes=processed_dataset.num_nodes,
                                          supervision=True)
    assert stats["mode"] == "supervision_aware"
    assert stats["supervision_changed"] is True
    assert stats["num_train_positives_removed"] == stats["num_encode_edges_removed"]
    assert set(index_array_to_pairs(thinned.train)) == set(
        index_array_to_pairs(thinned.encode_edges))
    # Held-out positives and the fixed negatives are untouched.
    assert np.array_equal(thinned.val, part.val)
    assert np.array_equal(thinned.test, part.test)
    assert np.array_equal(thinned.neg_test, part.neg_test)


def test_supervision_removal_is_nested_across_ratios(processed_dataset):
    part = processed_dataset.channel("B")
    kept = {}
    for ratio in (0.1, 0.3, 0.5):
        thinned, _ = remove_training_edges(part, ratio, seed=11,
                                          num_nodes=processed_dataset.num_nodes,
                                          supervision=True)
        kept[ratio] = set(index_array_to_pairs(thinned.encode_edges))
    assert kept[0.5] <= kept[0.3] <= kept[0.1]


def test_zero_fraction_is_an_exact_no_op_in_both_modes(processed_dataset):
    for mode in ("adjacency", "supervision_aware"):
        same, stats = apply_edge_removal(processed_dataset, mode, 0.0, seed=5)
        assert stats["changed"] is False
        assert same is processed_dataset
        for channel in processed_dataset.channel_names:
            original, after = processed_dataset.channel(channel), same.channel(channel)
            for attribute in ("train", "encode_edges", "val", "test", "neg_test"):
                assert np.array_equal(getattr(original, attribute),
                                      getattr(after, attribute))


def test_channel_selection_damages_only_the_requested_view(processed_dataset):
    damaged, stats = apply_edge_removal(processed_dataset, "supervision_aware", 0.3,
                                        seed=3, channels=["A"])
    assert stats["channels_selected"] == ["A"]
    assert stats["channels"]["A"]["num_encode_edges_removed"] > 0
    assert stats["channels"]["B"]["untouched"] is True
    assert np.array_equal(damaged.channel("B").encode_edges,
                          processed_dataset.channel("B").encode_edges)
    assert np.array_equal(damaged.channel("B").train,
                          processed_dataset.channel("B").train)
    assert damaged.channel("A").pairs("val") == processed_dataset.channel("A").pairs("val")


def test_damaged_graphs_still_pass_every_leakage_guard(processed_dataset):
    for mode in ("adjacency", "supervision_aware"):
        damaged, _ = apply_edge_removal(processed_dataset, mode, 0.5, seed=9)
        damaged.validate()
        for channel in damaged.channel_names:
            part = damaged.channel(channel)
            held_out = part.pairs("val") | part.pairs("test")
            assert held_out.isdisjoint(index_array_to_pairs(part.encode_edges))
            # Removed positives never reappear as negatives.
            assert part.pairs("train").isdisjoint(part.negative_pairs("train"))


def test_two_models_see_the_identical_removed_subset(processed_dataset):
    first, stats_a = apply_edge_removal(processed_dataset, "supervision_aware", 0.5,
                                        seed=42)
    second, stats_b = apply_edge_removal(processed_dataset, "supervision_aware", 0.5,
                                        seed=42)
    assert stats_a == stats_b
    for channel in processed_dataset.channel_names:
        assert np.array_equal(first.channel(channel).encode_edges,
                              second.channel(channel).encode_edges)
        assert np.array_equal(first.channel(channel).train,
                              second.channel(channel).train)


def test_config_wiring_records_protocol_mode_and_channels(processed_dataset):
    config = {"data": {"visibility_protocol": "dual_view_strict_holdout",
                       "edge_removal_mode": "supervision_aware",
                       "edge_drop_fraction": 0.5,
                       "drop_channels": ["A"]},
              "runtime": {"seed": 42}}
    damaged, stats = apply_data_perturbations(processed_dataset, config)
    assert stats["visibility_protocol"] == "dual_view_strict_holdout"
    assert stats["edge_removal_mode"] == "supervision_aware"
    assert stats["drop_channels"] == ["A"]
    assert stats["mask_other_view"] is True          # legacy alias, derived
    assert stats["changed"] is True
    assert stats["channels"]["A"]["cross_view_mask"] is not None
    assert stats["channels"]["A"]["edge_drop"]["mode"] == "supervision_aware"


def test_legacy_mask_flag_maps_to_the_strict_protocol(processed_dataset):
    legacy, stats = apply_data_perturbations(processed_dataset,
                                             {"data": {"mask_other_view": True},
                                              "runtime": {"seed": 1}})
    modern, modern_stats = apply_data_perturbations(
        processed_dataset, {"data": {"visibility_protocol": "dual_view_strict_holdout"},
                            "runtime": {"seed": 1}})
    assert stats["visibility_protocol"] == "dual_view_strict_holdout"
    for channel in processed_dataset.channel_names:
        assert np.array_equal(legacy.channel(channel).encode_edges,
                              modern.channel(channel).encode_edges)
    assert stats["edge_drop_fraction"] == modern_stats["edge_drop_fraction"] == 0.0


def test_conflicting_mask_flag_and_protocol_is_rejected(processed_dataset):
    with pytest.raises(ValueError, match="conflicts"):
        apply_data_perturbations(processed_dataset,
                                 {"data": {"mask_other_view": True,
                                           "visibility_protocol": "cross_view_transfer"},
                                  "runtime": {"seed": 1}})


def test_bad_selection_and_modes_are_rejected(processed_dataset):
    with pytest.raises(ValueError, match="unknown channel"):
        apply_edge_removal(processed_dataset, "adjacency", 0.1, seed=1, channels=["Z"])
    with pytest.raises(ValueError, match="at least one channel"):
        apply_edge_removal(processed_dataset, "adjacency", 0.1, seed=1, channels=[])
    with pytest.raises(ValueError, match="mode must be one of"):
        apply_edge_removal(processed_dataset, "magic", 0.1, seed=1)
    with pytest.raises(ValueError, match="fraction"):
        apply_edge_removal(processed_dataset, "adjacency", 1.0, seed=1)


def test_channel_seed_is_stable_and_channel_specific():
    assert channel_seed(42, 0) == channel_seed(42, 0)
    assert channel_seed(42, 0) != channel_seed(42, 1)
    assert channel_seed(43, 0) != channel_seed(42, 0)


def test_thin_channel_stays_a_supervision_preserving_helper(processed_dataset):
    """The historic M5 helper must keep the training positives untouched."""
    part = processed_dataset.channel("A")
    thinned, stats = thin_channel(part, 0.3, seed=2,
                                  num_nodes=processed_dataset.num_nodes)
    assert stats["num_encode_edges_removed"] > 0
    assert np.array_equal(thinned.train, part.train)

