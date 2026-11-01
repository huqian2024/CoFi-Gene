"""Ranking metrics: hand-computed cases, tie handling, and baseline behaviour."""

import numpy as np
import pytest

from cofi_gene.metrics import (average_precision, binary_summary, prevalence_baseline,
                              roc_auc)


def test_metrics_are_one_for_perfect_ranking_and_zero_for_inverted():
    labels = np.array([1, 1, 0, 0])
    assert roc_auc([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(1.0)
    assert roc_auc([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(0.0)
    assert average_precision([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(1.0)


def test_average_precision_matches_hand_computation():
    # positives at ranks 1 and 3 -> (1/1 + 2/3) / 2
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 1, 0]
    assert average_precision(scores, labels) == pytest.approx((1.0 + 2 / 3) / 2)


def test_ties_do_not_inflate_average_precision():
    # every score identical: AP must equal the positive prevalence
    scores = [0.5] * 10
    labels = [1] * 3 + [0] * 7
    assert average_precision(scores, labels) == pytest.approx(0.3)
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_handles_tied_scores_symmetrically():
    labels = [1, 0, 1, 0]
    scores = [0.4, 0.4, 0.6, 0.2]
    # pairs: (0.4,0.2) pos>neg; (0.4,0.4) tie = 0.5; (0.6,0.2),(0.6,0.4) win
    assert roc_auc(scores, labels) == pytest.approx((1 + 0.5 + 1 + 1) / 4)


def test_degenerate_inputs_raise_or_report_clearly():
    with pytest.raises(ValueError, match="positive"):
        average_precision([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_auc([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="binary"):
        average_precision([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        average_precision([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError, match="match"):
        average_precision([0.1], [0, 1])


def test_binary_summary_reports_prevalence_and_missing_auc_gracefully():
    summary = binary_summary([0.9, 0.1], [1, 0])
    assert summary["positive_rate"] == pytest.approx(0.5)
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["roc_auc"] == pytest.approx(1.0)
    only_positive = binary_summary([0.9, 0.1], [1, 1])
    assert "roc_auc" not in only_positive
    assert prevalence_baseline([0, 1, 1, 0]) == pytest.approx(0.5)
