"""Leakage guards: the dataset container must refuse to represent a leaky split.

These are adversarial tests - each one deliberately builds a broken dataset and
asserts that validation raises, so the guard cannot silently disappear later.

Node set: G0..G3. Base positives are train=(G0,G1),(G2,G3), val=(G0,G3),
test=(G1,G2); negatives are the remaining non-edges (G0,G2) and (G1,G3).
"""

import numpy as np
import pytest

from cofi_gene.dataset import ChannelSplit, CoFiDataset


def _edges(*pairs) -> np.ndarray:
    """Build a ``[2, E]`` index array from ``(u, v)`` pairs."""
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([list(pair) for pair in pairs], dtype=np.int64).T.copy()


def _base_channel() -> ChannelSplit:
    return ChannelSplit(
        train=_edges((0, 1), (2, 3)),
        val=_edges((0, 3)),
        test=_edges((1, 2)),
        train_weights=np.ones(2, dtype=np.float32),
        val_weights=np.ones(1, dtype=np.float32),
        test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 2), (1, 3)),
        neg_val=_edges((0, 2)),
        neg_test=_edges((1, 3)),
        encode_edges=_edges((0, 1), (2, 3)),
        encode_weights=np.ones(2, dtype=np.float32))


def _dataset(channel: ChannelSplit, target_splits=None) -> CoFiDataset:
    return CoFiDataset(gene_ids=("G0", "G1", "G2", "G3"),
                       X=np.zeros((4, 3), dtype=np.float32),
                       channels={"A": channel},
                       target_splits=target_splits or {"train": ("G0",),
                                                       "val": ("G1",), "test": ("G2",)})


def test_clean_dataset_validates_and_reports_counts():
    report = _dataset(_base_channel()).validate()
    assert report["num_nodes"] == 4
    assert report["channels"]["A"]["num_train"] == 2
    assert report["channels"]["A"]["num_val"] == 1
    assert report["channels"]["A"]["num_neg_val"] == 1
    assert report["num_targets"] == 3


def test_held_out_positive_inside_encoder_graph_is_rejected():
    channel = _base_channel()
    channel.encode_edges = _edges((0, 1), (2, 3), (1, 2))   # (1,2) is a test edge
    with pytest.raises(ValueError, match="held-out positive"):
        _dataset(channel).validate()


def test_overlapping_positive_splits_are_rejected():
    channel = _base_channel()
    channel.val = _edges((0, 1))                            # duplicates a train edge
    with pytest.raises(ValueError, match="splits overlap"):
        _dataset(channel).validate()


def test_negative_that_is_a_known_positive_is_rejected():
    channel = _base_channel()
    channel.neg_test = _edges((0, 3))                       # a val positive
    with pytest.raises(ValueError, match="known positives"):
        _dataset(channel).validate()
    channel = _base_channel()
    channel.neg_train = _edges((2, 3))                      # a train positive
    with pytest.raises(ValueError, match="known positives"):
        _dataset(channel).validate()


def test_duplicate_negatives_are_rejected():
    channel = _base_channel()
    channel.neg_train = _edges((0, 2), (2, 0))
    with pytest.raises(ValueError, match="duplicate negatives"):
        _dataset(channel).validate()


def test_out_of_range_edge_index_is_rejected():
    channel = _base_channel()
    channel.train = _edges((0, 9))
    channel.encode_edges = _edges((0, 9))
    with pytest.raises(ValueError, match="out of range"):
        _dataset(channel).validate()


def test_downstream_target_leaking_across_splits_is_rejected():
    channel = _base_channel()
    leaky_targets = {"train": ("G0", "G1"), "val": ("G1",), "test": ("G2",)}
    with pytest.raises(ValueError, match="target splits overlap"):
        _dataset(channel, leaky_targets).validate()


def test_constructor_rejects_bad_node_bookkeeping():
    with pytest.raises(ValueError, match="duplicates"):
        CoFiDataset(gene_ids=("G0", "G0"), X=np.zeros((2, 2), dtype=np.float32),
                    channels={"A": _base_channel()})
    with pytest.raises(ValueError, match="X must be"):
        CoFiDataset(gene_ids=("G0", "G1"), X=np.zeros((3, 2), dtype=np.float32),
                    channels={"A": _base_channel()})
    with pytest.raises(ValueError, match="at least one channel"):
        CoFiDataset(gene_ids=("G0",), X=np.zeros((1, 2), dtype=np.float32), channels={})


def test_processed_pipeline_output_is_leakage_free(processed_dataset):
    report = processed_dataset.validate()
    for channel in ("A", "B"):
        stats = report["channels"][channel]
        assert stats["num_train"] > 0 and stats["num_val"] > 0 and stats["num_test"] > 0
        assert stats["num_edges_encode"] <= stats["num_train"]
