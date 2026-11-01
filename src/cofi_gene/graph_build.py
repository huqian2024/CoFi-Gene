"""Turn STRING channel rows into index-based graphs and describe them.

All functions here are numpy-only: graph construction must be testable on a CPU
laptop before any GPU work happens. Node ids are positional indices into the
canonical ``gene_ids`` list, so nothing depends on dict ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ChannelGraph",
    "canonicalize_pairs",
    "build_channel_graph",
    "pairs_to_index_array",
    "index_array_to_pairs",
    "graph_stats",
    "undirected_pair_keys",
    "overlap_stats",
    "normalize_edge_weights",
    "WEIGHT_MODES",
]

WEIGHT_MODES = ("none", "minmax")


@dataclass
class ChannelGraph:
    """One evidence channel as ``edge_index`` [2, E] plus raw channel weights."""

    edge_index: np.ndarray
    weights: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def canonicalize_pairs(pairs_and_scores) -> dict[tuple[int, int], float]:
    """Undirected, deduplicated ``(min,max) -> max score`` mapping.

    Keeping the maximum score for duplicate rows is deterministic and mirrors
    "this pair has at least this evidence"; the merge policy is recorded as a
    pending decision (D-010) because mean-vs-max slightly changes edge weights.
    """
    merged: dict[tuple[int, int], float] = {}
    for u, v, score in pairs_and_scores:
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        previous = merged.get(key)
        value = float(score)
        if previous is None or value > previous:
            merged[key] = value
    return merged


def build_channel_graph(edges, protein_to_index: dict[str, int], num_nodes: int,
                        min_score: int = 0, weight_mode: str = "none") -> ChannelGraph:
    """Build one graph from ``(protein1, protein2, score)`` triples.

    Edges referencing a protein outside ``protein_to_index`` are dropped and
    counted; the caller decides whether the resulting coverage is acceptable.
    """
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
    mapped = []
    n_unknown_protein = 0
    n_below_threshold = 0
    n_self_loop = 0
    for p1, p2, score in edges:
        if score < min_score:
            n_below_threshold += 1
            continue
        u = protein_to_index.get(p1)
        v = protein_to_index.get(p2)
        if u is None or v is None:
            n_unknown_protein += 1
            continue
        if u == v:
            n_self_loop += 1
            continue
        mapped.append((u, v, score))
    merged = canonicalize_pairs(mapped)
    keys = sorted(merged)
    if keys:
        edge_index = np.asarray(keys, dtype=np.int64).T.copy()
        weights = np.asarray([merged[k] for k in keys], dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float32)
    weights = normalize_edge_weights(weights, weight_mode)
    stats = {
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.shape[1]),
        "n_rows_mapped": len(mapped),
        "n_rows_below_threshold": n_below_threshold,
        "n_edges_unknown_protein": n_unknown_protein,
        "n_self_loops_dropped": n_self_loop,
        "n_duplicate_pairs_merged": len(mapped) - len(merged),
        "min_score": int(min_score),
        "weight_mode": weight_mode,
    }
    return ChannelGraph(edge_index=edge_index, weights=weights, stats=stats)


def pairs_to_index_array(pairs, num_nodes: int) -> np.ndarray:
    """Sorted tuple-of-pairs -> ``[2, E]`` int64 array (deterministic order)."""
    cleaned = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                      for u, v in pairs if int(u) != int(v)})
    for u, v in cleaned:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"pair outside [0,{num_nodes}): {(u, v)}")
    if not cleaned:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(cleaned, dtype=np.int64).T.copy()


def index_array_to_pairs(edge_index) -> tuple[tuple[int, int], ...]:
    """``[2, E]`` array -> tuple of canonical pairs (for split algebra)."""
    array = np.asarray(edge_index)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return tuple((int(min(u, v)), int(max(u, v))) for u, v in zip(array[0], array[1]))


def undirected_pair_keys(edge_index) -> frozenset:
    """Set of canonical pairs; the unit of comparison for overlap/leakage."""
    return frozenset(index_array_to_pairs(edge_index))


def normalize_edge_weights(weights, mode: str = "none") -> np.ndarray:
    """Scale raw STRING scores (0-1000) for the GCN.

    ``none`` keeps raw magnitudes (a documented pending choice, D-011);
    ``minmax`` maps the observed range onto [0, 1].
    """
    array = np.asarray(weights, dtype=np.float32)
    if mode == "none":
        return array
    if mode != "minmax":
        raise ValueError(f"unknown weight mode: {mode}")
    if array.size == 0:
        return array
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.ones_like(array)
    return ((array - low) / (high - low)).astype(np.float32)


def graph_stats(edge_index, num_nodes: int, weights=None) -> dict:
    """Descriptive statistics used in the manifest and in STATUS reports."""
    pairs = index_array_to_pairs(edge_index)
    if not pairs:
        return {"num_nodes": int(num_nodes), "num_edges": 0,
                "num_isolated_nodes": int(num_nodes), "density": 0.0,
                "mean_degree": 0.0, "max_degree": 0}
    u = np.asarray([p[0] for p in pairs], dtype=np.int64)
    v = np.asarray([p[1] for p in pairs], dtype=np.int64)
    degree = np.bincount(np.concatenate([u, v]), minlength=num_nodes)
    possible = num_nodes * (num_nodes - 1) / 2.0
    stats = {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(pairs)),
        "num_isolated_nodes": int((degree == 0).sum()),
        "density": float(len(pairs) / possible) if possible else 0.0,
        "mean_degree": float(degree.mean()),
        "max_degree": int(degree.max()),
    }
    if weights is not None and np.asarray(weights).size:
        array = np.asarray(weights, dtype=np.float64)
        stats.update({"weight_min": float(array.min()), "weight_max": float(array.max()),
                      "weight_mean": float(array.mean())})
    return stats


def overlap_stats(edge_index_a, edge_index_b) -> dict:
    """Edge overlap between the two evidence views.

    Reported because STRING's two channels are *not* independent sources; the
    write-up has to state how many evaluation pairs are visible in the other
    view (see the dual-view protocol in ``docs/DATA_CONTRACT.md``).
    """
    set_a = undirected_pair_keys(edge_index_a)
    set_b = undirected_pair_keys(edge_index_b)
    shared = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "n_edges_a": len(set_a),
        "n_edges_b": len(set_b),
        "n_shared_edges": shared,
        "jaccard": float(shared / union) if union else 0.0,
        "frac_a_in_b": float(shared / len(set_a)) if set_a else 0.0,
        "frac_b_in_a": float(shared / len(set_b)) if set_b else 0.0,
    }

