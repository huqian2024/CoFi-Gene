"""Leakage guards: the dataset container must refuse to represent a leaky split.

These are adversarial tests - each one deliberately builds a broken dataset and
asserts that validation raises, so the guard cannot silently disappear later.

Node set: G0..G5. Base positives are train=(G0,G1),(G2,G3), val=(G0,G3),
test=(G1,G2). Negatives are drawn from the remaining non-edges and, since M5.1,
must be pairwise disjoint across splits: train=(G0,G2),(G1,G4), val=(G3,G4),
test=(G0,G5).
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
        neg_train=_edges((0, 2), (1, 4)),
        neg_val=_edges((3, 4)),
        neg_test=_edges((0, 5)),
        encode_edges=_edges((0, 1), (2, 3)),
        encode_weights=np.ones(2, dtype=np.float32))


def _dataset(channel: ChannelSplit, target_splits=None) -> CoFiDataset:
    return CoFiDataset(gene_ids=("G0", "G1", "G2", "G3", "G4", "G5"),
                       X=np.zeros((6, 3), dtype=np.float32),
                       channels={"A": channel},
                       target_splits=target_splits or {"train": ("G0",),
                                                       "val": ("G1",), "test": ("G2",)})


def test_clean_dataset_validates_and_reports_counts():
    report = _dataset(_base_channel()).validate()
    assert report["num_nodes"] == 6
    assert report["channels"]["A"]["num_train"] == 2
    assert report["channels"]["A"]["num_val"] == 1
    assert report["channels"]["A"]["num_neg_val"] == 1
    assert report["num_targets"] == 3
    assert report["channels"]["A"]["negative_split_overlap"] == {
        "train&val": 0, "train&test": 0, "val&test": 0}


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
        # M5.1: negatives are mutually exclusive across the three splits.
        assert stats["negative_split_overlap"] == {"train&val": 0, "train&test": 0,
                                                   "val&test": 0}


# ------------------------------------------------------- M5.1 hardening (B-01/B-02)
def test_negative_splits_overlapping_are_rejected():
    """A pair reused by train and test negatives is the historic M5.1 bug."""
    channel = _base_channel()
    channel.neg_test = _edges((0, 2))                      # also a train negative
    with pytest.raises(ValueError, match="negative splits overlap"):
        _dataset(channel).validate()


def test_self_loop_in_a_positive_split_is_rejected():
    channel = _base_channel()
    channel.train = _edges((0, 1), (2, 2))
    with pytest.raises(ValueError, match="self-loop"):
        _dataset(channel).validate()


def test_reversed_duplicate_positive_is_rejected():
    channel = _base_channel()
    channel.train = _edges((0, 1), (1, 0))
    channel.encode_edges = _edges((0, 1))
    with pytest.raises(ValueError, match="duplicate"):
        _dataset(channel).validate()


def test_encoder_edge_outside_training_positives_is_rejected():
    """The encoder may shrink (masking) but may never invent supervision."""
    channel = _base_channel()
    channel.encode_edges = _edges((0, 1), (2, 3), (4, 5))
    channel.encode_weights = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="not training positives"):
        _dataset(channel).validate()


def test_weight_length_mismatch_is_rejected():
    channel = _base_channel()
    channel.encode_weights = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="encode_weights"):
        _dataset(channel).validate()


def test_inspect_reports_every_check_and_never_raises():
    """The audit path must be able to *describe* a broken dataset, not just die."""
    broken = _base_channel()
    broken.neg_test = _edges((0, 2))                       # cross-split overlap
    checks, report, failures = _dataset(broken).inspect()
    assert failures and all(check["status"] == "FAIL" for check in failures)
    assert any(check["id"].startswith("C09") for check in failures)
    assert report["num_nodes"] == 6
    assert {check["status"] for check in checks} <= {"PASS", "FAIL"}
    # A clean dataset reports no failure at all.
    _, _, clean_failures = _dataset(_base_channel()).inspect()
    assert clean_failures == []

