"""Deterministic negative sampling for link prediction.

Rules that must never regress (they are asserted in ``tests/test_leakage.py``):

1. A negative pair is never a known positive in *any* split of that graph,
   including the held-out validation/test edges.
2. Negatives are unique and never self-loops.
3. The sample is a pure function of ``(count, seed, forbidden_pairs)`` so two
   models see identical negatives given the same seed.
"""

from __future__ import annotations

from random import Random

__all__ = ["sample_negative_pairs", "negative_pair_count"]

Pair = tuple[int, int]


def negative_pair_count(num_positives: int, ratio: float) -> int:
    """Number of negatives to draw for a given positive count and ratio."""
    if num_positives < 0:
        raise ValueError("num_positives must be >= 0")
    if ratio < 0:
        raise ValueError("ratio must be >= 0")
    return int(round(num_positives * ratio))


def sample_negative_pairs(num_nodes: int, count: int, forbidden_pairs,
                          seed: int = 42, max_attempts_factor: int = 50) -> tuple[Pair, ...]:
    """Draw ``count`` unique non-edges uniformly at random.

    ``forbidden_pairs`` must be the *complete* known-positive universe of the
    target graph (train + val + test). Pass a ``frozenset`` of canonical
    ``(min, max)`` pairs; a plain set is accepted and converted.
    """
    if num_nodes < 2:
        raise ValueError("need at least two nodes to sample negatives")
    if count < 0:
        raise ValueError("count must be >= 0")
    forbidden = forbidden_pairs if isinstance(forbidden_pairs, (set, frozenset)) \
        else frozenset(forbidden_pairs)
    rng = Random(seed)
    sampled: set[Pair] = set()
    attempts = 0
    max_attempts = max(1000, count * max_attempts_factor)
    while len(sampled) < count:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"could not sample {count} negatives without collisions after "
                f"{attempts} attempts; the graph is probably too dense")
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in forbidden or pair in sampled:
            continue
        sampled.add(pair)
    return tuple(sorted(sampled))
