"""Ranking metrics implemented in numpy (no scikit-learn dependency).

The two upstream/downstream headline metrics are AUPRC (average precision) and
AUROC. Both are computed with explicit tie handling so that repeated scores
cannot silently inflate or deflate the reported numbers, and both are pure
functions that can be unit-tested without torch or a GPU.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_precision", "roc_auc", "binary_summary", "prevalence_baseline"]


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must match, got {scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise ValueError("empty score/label arrays")
    if not np.all(np.isin(labels, (0, 1, False, True))):
        raise ValueError("labels must be binary (0/1)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores, labels.astype(bool)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties receiving the average of their positions."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def average_precision(scores, labels) -> float:
    """Average precision = area under the precision-recall curve.

    Implemented as the recall-increment sum ``Σ (R_k - R_{k-1}) P_k`` over the
    score thresholds, where every tied score forms a single threshold. With that
    convention a constant score vector returns exactly the positive prevalence,
    which is the honest baseline rather than an inflated number.
    """
    scores, labels = _as_arrays(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError("average_precision needs at least one positive label")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, labels.size + 1)
    precision = true_positive / predicted_positive
    recall = true_positive / n_pos
    # Keep only the last index of each distinct-score block (one threshold per block).
    boundary = np.r_[np.diff(sorted_scores) != 0, True]
    precision_at = precision[boundary]
    recall_at = recall[boundary]
    previous_recall = np.r_[0.0, recall_at[:-1]]
    return float(np.sum((recall_at - previous_recall) * precision_at))



def roc_auc(scores, labels) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic."""
    scores, labels = _as_arrays(scores, labels)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both positive and negative labels")
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prevalence_baseline(labels) -> float:
    """Positive rate: the AUPRC a constant/sklearn-random predictor reaches."""
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        raise ValueError("empty label array")
    return float(np.mean(labels.astype(bool)))


def binary_summary(scores, labels) -> dict[str, float]:
    """AUPRC / AUROC / prevalence for one (scores, labels) block."""
    scores, labels = _as_arrays(scores, labels)
    summary = {
        "n": int(labels.size),
        "positive_rate": float(labels.mean()),
    }
    if labels.any():
        summary["average_precision"] = average_precision(scores, labels)
    if labels.any() and not labels.all():
        summary["roc_auc"] = roc_auc(scores, labels)
    return summary
