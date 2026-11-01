"""Deterministic, leakage-aware split primitives.

These pure-Python helpers are intentionally independent of Torch/PyG so their
invariants can be tested on CPU before full graph training is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

Edge = tuple[int, int]


@dataclass(frozen=True)
class ThreeWaySplit:
    train: tuple
    val: tuple
    test: tuple


def _validate_ratios(train_fraction: float, val_fraction: float) -> None:
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be in (0,1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("test fraction must be > 0")


def normalize_undirected_edges(edges: Iterable[Edge], num_nodes: int) -> tuple[Edge, ...]:
    """Canonicalize (u, v) into (min, max), dedupe, omit self-loops.

    This helper intentionally cannot inspect graph evidence weights; upstream
    data code must define a deterministic policy for merging duplicate scores.
    """
    if type(num_nodes) is not int or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")
    result: set[Edge] = set()
    for u, v in edges:
        if type(u) is not int or type(v) is not int:
            raise TypeError("edge endpoints must be integer node indices")
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge index outside [0,{num_nodes}): {(u, v)}")
        if u != v:
            result.add((min(u, v), max(u, v)))
    return tuple(sorted(result))


def split_edges(
    edges: Iterable[Edge], num_nodes: int,
    train_fraction: float = 0.8, val_fraction: float = 0.1, seed: int = 42,
) -> ThreeWaySplit:
    """Split distinct undirected positive edges once for a target graph.

    Use ONLY `train` to construct that graph's encoding/feedback adjacency.
    In the dual-view masked experiment, remove overlapping held-out edges from
    the other graph separately; this function is not a full cross-view protocol.
    """
    _validate_ratios(train_fraction, val_fraction)
    items = list(normalize_undirected_edges(edges, num_nodes))
    Random(seed).shuffle(items)
    n = len(items)
    n_train, n_val = int(n * train_fraction), int(n * val_fraction)
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("not enough unique edges for nonempty train/val/test")
    return ThreeWaySplit(tuple(items[:n_train]),
                         tuple(items[n_train:n_train+n_val]),
                         tuple(items[n_train+n_val:]))


def split_perturbation_targets(
    targets: Iterable[str], train_fraction: float = 0.7,
    val_fraction: float = 0.15, seed: int = 42,
) -> ThreeWaySplit:
    """Split unique perturbation TARGET IDs, not response rows.

    The caller filters all cell-line/response records by these target groups.
    Repeated targets across contexts must remain within the same partition.
    """
    _validate_ratios(train_fraction, val_fraction)
    target_set = set(targets)
    if any(not isinstance(x, str) or not x.strip() for x in target_set):
        raise ValueError("targets must be non-empty gene ID strings")
    items = sorted(target_set)
    Random(seed).shuffle(items)
    n = len(items)
    n_train, n_val = int(n * train_fraction), int(n * val_fraction)
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("not enough unique targets for nonempty train/val/test")
    return ThreeWaySplit(tuple(items[:n_train]),
                         tuple(items[n_train:n_train+n_val]),
                         tuple(items[n_train+n_val:]))


def validate_negative_edges(negatives: Iterable[Edge], all_positive_edges: Iterable[Edge],
                            num_nodes: int) -> None:
    """Reject negatives that are known positives in ANY split of target graph.

    Call with the *full* positive-edge universe before holding out val/test.
    """
    positives = set(normalize_undirected_edges(all_positive_edges, num_nodes))
    seen: set[Edge] = set()
    for u, v in negatives:
        if type(u) is not int or type(v) is not int or not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("invalid negative edge endpoint")
        if u == v:
            raise ValueError("negative samples cannot be self loops")
        e = (min(u, v), max(u, v))
        if e in positives:
            raise ValueError(f"negative is an observed positive edge: {e}")
        if e in seen:
            raise ValueError(f"duplicate negative edge: {e}")
        seen.add(e)
