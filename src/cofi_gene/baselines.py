"""Non-neural baselines for the upstream link-prediction task (M5.1 Phase 6).

The question these baselines answer: are AP values around 0.95-0.97 explained by
simple graph topology or by the sampling protocol rather than by a learned gene
representation? They are dependency-free (numpy only), need no training, and score
the *same* positives and negatives as B0-B7, so a baseline row can sit next to a
model row in one table.

Protocol rules (identical to the models, enforced here):

* scores come from the graph the encoder is allowed to see (``encode_edges`` by
  default, i.e. the training graph after the visibility protocol and any edge
  removal). Never from the full graph, never from val/test positives.
* the candidate set is the dataset's fixed ``<split>`` positives and
  ``neg_<split>`` negatives, so every model and baseline is ranked on one
  candidate set with one positive rate.
* nothing is materialised as an ``N x N`` matrix: topology scores walk adjacency
  sets and the feature cosine is computed in pair batches.

Scores are ranking scores, not probabilities: they are only compared through
AUPRC/AUROC.
"""

from __future__ import annotations

import numpy as np

from .dataset import CoFiDataset
from .graph_build import index_array_to_pairs
from .metrics import binary_summary, prevalence_baseline

__all__ = [
    "BASELINE_NAMES",
    "TopologyBaseline",
    "build_baseline",
    "score_pairs",
    "evaluate_baselines",
]

#: Stable baseline ids used in configs, reports and the training guide.
BASELINE_NAMES = (
    "preferential_attachment",
    "common_neighbors",
    "adamic_adar",
    "resource_allocation",
    "feature_cosine",
)


def _adjacency_sets(edge_index, num_nodes: int) -> list[set[int]]:
    neighbours: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for u, v in index_array_to_pairs(edge_index):
        if u == v:
            continue
        neighbours[u].add(v)
        neighbours[v].add(u)
    return neighbours


class TopologyBaseline:
    """One closed-form pair scorer over a fixed training graph.

    ``name`` selects the formula; ``features`` is optional and only required by
    ``feature_cosine``. ``graph`` is recorded so a report can state which graph
    the baseline saw.
    """

    def __init__(self, name: str, edge_index, num_nodes: int, features=None,
                 graph: str = "encode_edges") -> None:
        if name not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline {name!r}; available: {BASELINE_NAMES}")
        if int(num_nodes) <= 0:
            raise ValueError("num_nodes must be positive")
        self.name = name
        self.num_nodes = int(num_nodes)
        self.graph = graph
        self.neighbours = _adjacency_sets(edge_index, num_nodes)
        self.degree = np.asarray([len(n) for n in self.neighbours], dtype=np.int64)
        if name == "feature_cosine":
            if features is None:
                raise ValueError("feature_cosine needs the shared feature matrix")
            array = np.asarray(features, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != self.num_nodes:
                raise ValueError(f"features must be [N, F] with N={self.num_nodes}, "
                                 f"got {array.shape}")
            norm = np.linalg.norm(array, axis=1, keepdims=True)
            self._unit = array / np.clip(norm, 1e-12, None)

    # ------------------------------------------------------------- formulas
    def _common(self, u: int, v: int) -> set[int]:
        first, second = self.neighbours[u], self.neighbours[v]
        return first & second if len(first) <= len(second) else second & first

    def _score_one(self, u: int, v: int) -> float:
        if self.name == "preferential_attachment":
            return float(self.degree[u] * self.degree[v])
        if self.name == "common_neighbors":
            return float(len(self._common(u, v)))
        common = self._common(u, v)
        if self.name == "adamic_adar":
            total = 0.0
            for w in common:
                degree = int(self.degree[w])
                if degree > 1:
                    total += 1.0 / np.log(degree)
            return float(total)
        if self.name == "resource_allocation":
            return float(sum(1.0 / int(self.degree[w]) for w in common
                             if self.degree[w] > 0))
        # feature_cosine
        return float(np.dot(self._unit[u], self._unit[v]))

    def score(self, edge_index, chunk_size: int = 200_000) -> np.ndarray:
        """Ranking score per pair, in the order of ``edge_index``."""
        pairs = index_array_to_pairs(edge_index)
        if not pairs:
            return np.zeros((0,), dtype=np.float64)
        if self.name == "feature_cosine":
            return self._score_cosine(pairs, chunk_size)
        return np.asarray([self._score_one(u, v) for u, v in pairs], dtype=np.float64)

    def _score_cosine(self, pairs, chunk_size: int) -> np.ndarray:
        sources = np.asarray([u for u, _ in pairs], dtype=np.int64)
        targets = np.asarray([v for _, v in pairs], dtype=np.int64)
        scores = np.empty(sources.size, dtype=np.float64)
        for start in range(0, sources.size, chunk_size):
            stop = min(start + chunk_size, sources.size)
            left = self._unit[sources[start:stop]]
            right = self._unit[targets[start:stop]]
            scores[start:stop] = np.einsum("ij,ij->i", left, right)
        return scores


def build_baseline(name: str, dataset: CoFiDataset, channel: str,
                   graph: str = "encode_edges") -> TopologyBaseline:
    """Build one baseline on the protocol-visible graph of ``channel``."""
    part = dataset.channel(channel)
    if graph not in ("encode_edges", "train"):
        raise ValueError("graph must be 'encode_edges' (default) or 'train'")
    edges = part.encode_edges if graph == "encode_edges" else part.train
    return TopologyBaseline(name, edges, dataset.num_nodes, features=dataset.X,
                            graph=graph)


def score_pairs(baseline: TopologyBaseline, edge_index) -> np.ndarray:
    """Convenience wrapper used by tests and scripts."""
    return baseline.score(edge_index)


def evaluate_baselines(dataset: CoFiDataset, split: str = "test",
                       names=BASELINE_NAMES, channels=None,
                       graph: str = "encode_edges") -> dict:
    """AUPRC/AUROC of every baseline on the dataset's fixed candidate set.

    Returns ``{channel: {baseline: metrics}}`` with the same keys the model
    trainer writes (``average_precision``, ``roc_auc``, ``n``, ``positive_rate``)
    plus ``prevalence_baseline`` and the graph the scores came from. A split
    without positives or negatives yields ``status="N/A"`` instead of a fake 0/1.
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test' (train is not an evaluation set)")
    channels = tuple(dataset.channel_names if channels is None else channels)
    results: dict = {}
    for channel in channels:
        part = dataset.channel(channel)
        positives = np.asarray(getattr(part, split), dtype=np.int64)
        negatives = np.asarray(getattr(part, f"neg_{split}"), dtype=np.int64)
        results[channel] = {}
        if positives.size == 0 or negatives.size == 0:
            for name in names:
                results[channel][name] = {
                    "status": "N/A",
                    "reason": "split has no positives or no negatives; AP/AUROC "
                              "would be undefined",
                    "n_positives": int(positives.shape[1]) if positives.size else 0,
                    "n_negatives": int(negatives.shape[1]) if negatives.size else 0,
                }
            continue
        for name in names:
            baseline = build_baseline(name, dataset, channel, graph=graph)
            positive_scores = baseline.score(positives)
            negative_scores = baseline.score(negatives)
            scores = np.concatenate([positive_scores, negative_scores])
            labels = np.concatenate([np.ones(positive_scores.size),
                                     np.zeros(negative_scores.size)])
            summary = binary_summary(scores, labels)
            summary["n_positives"] = int(positive_scores.size)
            summary["n_negatives"] = int(negative_scores.size)
            summary["prevalence_baseline"] = prevalence_baseline(labels)
            summary["graph"] = graph
            summary["baseline"] = name
            results[channel][name] = summary
    return results

