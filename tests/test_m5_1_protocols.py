"""M5.1 visibility protocols: cross-view transfer vs dual-view strict holdout."""

from __future__ import annotations

import numpy as np
import pytest

from cofi_gene import config as config_module
from cofi_gene import perturbation as perturbation_module
from cofi_gene import protocols
from cofi_gene.dataset import ChannelSplit, CoFiDataset
from cofi_gene.graph_build import undirected_pair_keys
from cofi_gene.perturbation import mask_cross_view_positives


def _edges(*pairs) -> np.ndarray:
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([list(pair) for pair in pairs], dtype=np.int64).T.copy()


def _tiny() -> CoFiDataset:
    """Six genes; A's held-out pair (0,1) is a training edge of B."""
    channel_a = ChannelSplit(
        train=_edges((2, 3)), train_weights=np.ones(1, dtype=np.float32),
        val=_edges((0, 1)), val_weights=np.ones(1, dtype=np.float32),
        test=_edges((1, 2)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 5)), neg_val=_edges((4, 5)), neg_test=_edges((0, 4)),
        encode_edges=_edges((2, 3)), encode_weights=np.ones(1, dtype=np.float32))
    channel_b = ChannelSplit(
        train=_edges((0, 1), (3, 4)), train_weights=np.ones(2, dtype=np.float32),
        val=_edges((1, 3)), val_weights=np.ones(1, dtype=np.float32),
        test=_edges((2, 4)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 2)), neg_val=_edges((2, 5)), neg_test=_edges((0, 3)),
        encode_edges=_edges((0, 1), (3, 4)), encode_weights=np.ones(2, dtype=np.float32))
    dataset = CoFiDataset(gene_ids=tuple(f"G{i}" for i in range(6)),
                          X=np.zeros((6, 3), dtype=np.float32),
                          channels={"A": channel_a, "B": channel_b})
    dataset.validate()
    return dataset


def test_protocol_names_are_shared_with_config_and_perturbation():
    assert config_module.VISIBILITY_PROTOCOLS == protocols.VISIBILITY_PROTOCOLS
    assert perturbation_module.EDGE_REMOVAL_MODES == config_module.EDGE_REMOVAL_MODES
    assert set(protocols.PROTOCOL_DESCRIPTIONS) == set(protocols.VISIBILITY_PROTOCOLS)


def test_visibility_report_counts_what_the_other_view_can_see():
    report = protocols.visibility_report(_tiny())
    assert report["n_channels"] == 2
    # A's held-out (0,1) is a training edge of B -> visible there.
    assert report["channels"]["A"]["visible_in_other_encode"] == 1
    assert report["channels"]["A"]["visible_in_other_train"] == 1
    assert report["channels"]["A"]["visibility_ratio_encode"] == pytest.approx(0.5)
    # B's held-out (1,3)/(2,4) are not in A at all.
    assert report["channels"]["B"]["visible_in_other_encode"] == 0
    # Strict holdout would cost B one training positive (0,1) and A nothing.
    assert report["channels"]["A"]["train_removed_if_strict"] == 0
    assert report["channels"]["B"]["train_removed_if_strict"] == 1


def test_transfer_protocol_is_an_identity_transform_that_reports_numbers():
    dataset = _tiny()
    same, stats = protocols.apply_visibility_protocol(dataset, "cross_view_transfer")
    assert stats["changed"] is False
    assert stats["protocol"] == "cross_view_transfer"
    assert stats["visibility_before"] == stats["visibility_after"]
    for channel in dataset.channel_names:
        assert np.array_equal(same.channel(channel).encode_edges,
                              dataset.channel(channel).encode_edges)
        assert np.array_equal(same.channel(channel).train,
                              dataset.channel(channel).train)


def test_strict_holdout_removes_the_union_from_both_graphs_and_supervision():
    dataset = _tiny()
    masked, stats = protocols.apply_visibility_protocol(dataset, "dual_view_strict_holdout")
    masked.validate()
    assert stats["changed"] is True
    banned = protocols.holdout_union(dataset)
    assert stats["holdout_union_size"] == len(banned)
    for channel in masked.channel_names:
        part = masked.channel(channel)
        assert undirected_pair_keys(part.encode_edges).isdisjoint(banned)
        assert part.pairs("train").isdisjoint(banned)
    # Nothing else moves: val/test positives and the fixed negatives are untouched.
    for channel in dataset.channel_names:
        original, after = dataset.channel(channel), masked.channel(channel)
        for attribute in ("val", "test", "neg_train", "neg_val", "neg_test"):
            assert np.array_equal(getattr(original, attribute),
                                  getattr(after, attribute)), attribute
    residual = {name: entry["visible_in_other_encode"]
                for name, entry in stats["visibility_after"]["channels"].items()}
    assert residual == {"A": 0, "B": 0}
    # B loses training edge (0,1) because it is A's held-out pair.
    assert stats["channels"]["B"]["num_train_positives_removed"] == 1
    assert stats["channels"]["A"]["num_train_positives_removed"] == 0


def test_mask_cross_view_positives_uses_the_same_implementation():
    dataset = _tiny()
    via_wrapper, _ = mask_cross_view_positives(dataset)
    via_protocol, _ = protocols.apply_visibility_protocol(
        dataset, "dual_view_strict_holdout")
    for channel in dataset.channel_names:
        for attribute in ("train", "encode_edges", "val", "test", "neg_test"):
            assert np.array_equal(getattr(via_wrapper.channel(channel), attribute),
                                  getattr(via_protocol.channel(channel), attribute)), \
                f"{channel}.{attribute}"


def test_single_channel_dataset_is_left_alone_by_the_strict_protocol():
    single = CoFiDataset(gene_ids=("G0", "G1", "G2"), X=np.zeros((3, 2), dtype=np.float32),
                         channels={"A": _tiny().channel("A")})
    same, stats = protocols.apply_visibility_protocol(single, "dual_view_strict_holdout")
    assert stats["changed"] is False
    assert "single-channel" in stats["note"]
    assert np.array_equal(same.channel("A").train, single.channel("A").train)


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError, match="protocol must be one of"):
        protocols.apply_visibility_protocol(_tiny(), "mask_everything")


def test_strict_holdout_on_the_real_build_keeps_supervision(processed_dataset):
    masked, stats = protocols.apply_visibility_protocol(
        processed_dataset, "dual_view_strict_holdout")
    masked.validate()
    for channel in masked.channel_names:
        assert len(masked.channel(channel).pairs("train")) > 0
        assert stats["visibility_after"]["channels"][channel][
            "visible_in_other_encode"] == 0

