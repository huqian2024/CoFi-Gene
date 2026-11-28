"""M5.1 non-neural baselines: formulas, protocol visibility and N/A handling."""

from __future__ import annotations

import numpy as np
import pytest

from cofi_gene.baselines import (BASELINE_NAMES, TopologyBaseline, build_baseline,
                                 evaluate_baselines)
from cofi_gene.dataset import ChannelSplit, CoFiDataset

TRAIN = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)   # chain 0-1-2-3


def _pairs(*pairs) -> np.ndarray:
    return np.asarray([list(pair) for pair in pairs], dtype=np.int64).T.copy()


def test_topology_formulas_match_the_textbook_definitions():
    pairs = _pairs((0, 2), (0, 3))
    pa = TopologyBaseline("preferential_attachment", TRAIN, 4).score(pairs)
    assert pa.tolist() == [2.0, 1.0]                  # deg(0)*deg(2)=1*2, 1*1
    cn = TopologyBaseline("common_neighbors", TRAIN, 4).score(pairs)
    assert cn.tolist() == [1.0, 0.0]                  # {1} vs {}
    aa = TopologyBaseline("adamic_adar", TRAIN, 4).score(pairs)
    assert aa[0] == pytest.approx(1.0 / np.log(2)) and aa[1] == 0.0
    ra = TopologyBaseline("resource_allocation", TRAIN, 4).score(pairs)
    assert ra[0] == pytest.approx(0.5) and ra[1] == 0.0


def test_feature_cosine_uses_the_shared_features():
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                          dtype=np.float32)
    baseline = TopologyBaseline("feature_cosine", TRAIN, 4, features=features)
    scores = baseline.score(_pairs((0, 1), (0, 2), (0, 3)))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(1.0)


def test_baselines_never_see_held_out_edges():
    """A test positive that is absent from the training graph scores 0 / no evidence."""
    test_edge = _pairs((0, 3))
    for name in ("common_neighbors", "adamic_adar", "resource_allocation"):
        assert TopologyBaseline(name, TRAIN, 4).score(test_edge).tolist() == [0.0]
    pa = TopologyBaseline("preferential_attachment", TRAIN, 4).score(test_edge)[0]
    assert pa == pytest.approx(1.0)          # only degrees, which come from TRAIN


def test_empty_candidate_set_returns_an_empty_score_vector():
    empty = np.zeros((2, 0), dtype=np.int64)
    for name in BASELINE_NAMES:
        features = np.zeros((4, 2), dtype=np.float32)
        assert TopologyBaseline(name, TRAIN, 4, features=features).score(empty).size == 0


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        TopologyBaseline("magic", TRAIN, 4)
    with pytest.raises(ValueError, match="needs the shared feature matrix"):
        TopologyBaseline("feature_cosine", TRAIN, 4)
    with pytest.raises(ValueError, match="features must be"):
        TopologyBaseline("feature_cosine", TRAIN, 4, features=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="num_nodes"):
        TopologyBaseline("common_neighbors", TRAIN, 0)


def _dataset(neg_test) -> CoFiDataset:
    channel = ChannelSplit(
        train=_pairs((0, 1), (2, 3)), train_weights=np.ones(2, dtype=np.float32),
        val=_pairs((0, 3)), val_weights=np.ones(1, dtype=np.float32),
        test=_pairs((1, 2)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_pairs((0, 2)), neg_val=_pairs((1, 4)),
        neg_test=neg_test,
        encode_edges=_pairs((0, 1), (2, 3)),
        encode_weights=np.ones(2, dtype=np.float32))
    return CoFiDataset(gene_ids=tuple(f"G{i}" for i in range(6)),
                       X=np.arange(12, dtype=np.float32).reshape(6, 2),
                       channels={"A": channel})


def test_evaluate_baselines_reports_the_shared_candidate_set():
    dataset = _dataset(_pairs((0, 4), (1, 5)))
    results = evaluate_baselines(dataset, split="test")
    for name in BASELINE_NAMES:
        entry = results["A"][name]
        assert entry["n_positives"] == 1 and entry["n_negatives"] == 2
        assert entry["positive_rate"] == pytest.approx(1 / 3)
        assert entry["prevalence_baseline"] == pytest.approx(1 / 3)
        assert 0.0 <= entry["average_precision"] <= 1.0
        assert 0.0 <= entry["roc_auc"] <= 1.0
        assert entry["graph"] == "encode_edges"


def test_evaluate_baselines_marks_undefined_splits_as_na():
    dataset = _dataset(np.zeros((2, 0), dtype=np.int64))
    results = evaluate_baselines(dataset, split="test")
    entry = results["A"]["common_neighbors"]
    assert entry["status"] == "N/A"
    assert "undefined" in entry["reason"]
    assert "average_precision" not in entry


def test_build_baseline_rejects_unknown_graphs(processed_dataset):
    with pytest.raises(ValueError, match="graph must be"):
        build_baseline("common_neighbors", processed_dataset, "A", graph="full")
    baseline = build_baseline("feature_cosine", processed_dataset, "A",
                              graph="encode_edges")
    assert baseline.graph == "encode_edges"
    assert baseline.degree.size == processed_dataset.num_nodes
