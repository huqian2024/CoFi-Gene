"""Deterministic negative sampling for link prediction (M5.1 rewrite).

Invariants asserted in ``tests/test_m5_1_negative_splits.py`` and
``tests/test_leakage.py``:

1. A negative pair is never a known positive in *any* split of that graph,
   including the held-out validation/test edges.
2. Negatives are unique, never self-loops, and stored as canonical
   ``(min, max)`` pairs.
3. **The three splits draw from one shared pool**: train/val/test negatives are
   pairwise disjoint. Before M5.1 each split was sampled independently with its
   own seed and only ``known_positives`` as the forbidden set, so a pair could be
   a training negative *and* a test negative (server audit: 12 pairs in channel
   A, 354 in channel B; see ``docs/M5_1_EVALUATION_AUDIT.md`` finding B-01).
4. The sample is a pure function of ``(count, seed, forbidden_pairs, strategy,
   reference_edges)``, so every model sees byte-identical negatives.

Exclusion is done by *sequential accumulation* rather than "sample then split":
the three splits have different sizes (1:1 against their own positive counts)
and resampling would silently move the fixed evaluation set. ``train`` is drawn
first, then ``val`` avoiding ``train``, then ``test`` avoiding both, so each
split keeps exactly the configured size and the exclusion is auditable.

Strategies (M5.1 Phase 5, ``docs/M5_1_EVALUATION_AUDIT.md`` E-02):

``uniform_random``
    Historic protocol, kept under an explicit name so old and new rows can be
    labelled and compared.
``degree_matched``
    Draw the endpoint *degree pair* of a training positive (degrees computed on
    the protocol-visible training graph only), then sample nodes from the
    matching degree buckets. Removes the "tell positives from negatives by node
    degree alone" shortcut.
``two_hop``
    Sample ``u - w - v`` with ``w`` a neighbour of ``u`` and ``v`` a neighbour of
    ``w``: two-hop non-edges that are locally close in the training graph.
    Sparse (walks adjacency lists, never an ``N x N`` matrix).

Every strategy bounds its attempts per pair and declares a ``fallback`` policy
(``"uniform"``, ``"relaxed_degree"`` or ``"error"``); the number of fallbacks is
reported, never silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

__all__ = [
    "NEGATIVE_STRATEGIES",
    "FALLBACK_POLICIES",
    "SPLIT_ORDER",
    "NegativePartition",
    "negative_pair_count",
    "sample_negative_pairs",
    "sample_split_negatives",
    "degrees_from_edges",
]

Pair = tuple[int, int]

#: Sampling strategies accepted by :func:`sample_split_negatives`.
NEGATIVE_STRATEGIES = ("uniform_random", "degree_matched", "two_hop")
#: What to do when a hard strategy cannot fill a slot within its attempt budget.
FALLBACK_POLICIES = ("uniform", "relaxed_degree", "error")
#: Fixed draw order: ``train`` first, so training negatives never depend on the
#: requested evaluation sizes.
SPLIT_ORDER = ("train", "val", "test")


def negative_pair_count(num_positives: int, ratio: float) -> int:
    """Number of negatives to draw for a given positive count and ratio."""
    if num_positives < 0:
        raise ValueError("num_positives must be >= 0")
    if ratio < 0:
        raise ValueError("ratio must be >= 0")
    return int(round(num_positives * ratio))


def _as_forbidden(forbidden_pairs) -> frozenset:
    if isinstance(forbidden_pairs, (set, frozenset)):
        return frozenset(forbidden_pairs)
    return frozenset(forbidden_pairs)


def sample_negative_pairs(num_nodes: int, count: int, forbidden_pairs,
                          seed: int = 42, max_attempts_factor: int = 50) -> tuple[Pair, ...]:
    """Draw ``count`` unique non-edges uniformly at random (single split).

    ``forbidden_pairs`` must be the *complete* known-positive universe of the
    target graph (train + val + test). This is the historic helper; use
    :func:`sample_split_negatives` for a full build, which adds cross-split
    mutual exclusion.
    """
    if num_nodes < 2:
        raise ValueError("need at least two nodes to sample negatives")
    if count < 0:
        raise ValueError("count must be >= 0")
    forbidden = _as_forbidden(forbidden_pairs)
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


def degrees_from_edges(edge_index, num_nodes: int) -> np.ndarray:
    """Undirected degree per node on the given (training) edge array.

    The caller must pass the *protocol-visible* training graph. Using a graph
    that still contains validation/test positives would leak the structure the
    experiment claims to hide and would inflate the degree of exactly the nodes
    involved in the evaluation edges.
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if int(num_nodes) <= 0:
        raise ValueError("num_nodes must be positive")
    if edges.size == 0:
        return np.zeros(int(num_nodes), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError(f"edge_index outside [0, {int(num_nodes)})")
    return np.bincount(edges.reshape(-1), minlength=int(num_nodes)).astype(np.int64)


# --------------------------------------------------------------------------- #
# degree buckets / adjacency lists
# --------------------------------------------------------------------------- #
def _degree_buckets(degrees: np.ndarray, num_buckets: int):
    """Quantile buckets of the degree distribution.

    Returns ``(bucket_of_node, buckets)`` where ``buckets[b]`` is the node list
    of bucket ``b``. Ties share a bucket, so a bucket never mixes nodes whose
    degrees differ a lot.
    """
    degrees = np.asarray(degrees, dtype=np.int64)
    num_nodes = int(degrees.size)
    if num_nodes == 0:
        return np.zeros(0, dtype=np.int64), []
    num_buckets = max(1, min(int(num_buckets), num_nodes))
    quantiles = np.quantile(degrees.astype(np.float64),
                            np.linspace(0.0, 1.0, num_buckets + 1))
    bucket_of_node = np.searchsorted(quantiles[1:-1], degrees, side="right")
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    for node, bucket in enumerate(bucket_of_node.tolist()):
        buckets[int(bucket)].append(int(node))
    return bucket_of_node.astype(np.int64), buckets


def _adjacency_lists(edge_index, num_nodes: int) -> list[list[int]]:
    """Neighbour lists of the (training) graph, used by the two-hop strategy."""
    edges = np.asarray(edge_index, dtype=np.int64)
    neighbours: list[list[int]] = [[] for _ in range(int(num_nodes))]
    if edges.size == 0:
        return neighbours
    for u, v in zip(edges[0].tolist(), edges[1].tolist()):
        if u == v:
            continue
        neighbours[int(u)].append(int(v))
        neighbours[int(v)].append(int(u))
    return neighbours


# --------------------------------------------------------------------------- #
# split-wise, mutually exclusive sampler
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NegativePartition:
    """Negatives per split plus the full sampling record for the manifest."""

    negatives: dict[str, tuple[Pair, ...]]
    stats: dict

    def pairs(self, split: str) -> frozenset:
        return frozenset(self.negatives[split])


class _Drawer:
    """Draws valid non-edges for one split against a growing forbidden set."""

    def __init__(self, num_nodes: int, forbidden: frozenset, rng: Random, strategy: str,
                 fallback: str, adjacency: list[list[int]] | None,
                 degree_pairs: list[tuple[int, int]] | None,
                 buckets: list[list[int]] | None, split_name: str = "?") -> None:
        self.num_nodes = int(num_nodes)
        self.forbidden = forbidden
        self.split_name = split_name
        self.used: set[Pair] = set()
        self.rng = rng
        self.strategy = strategy
        self.fallback = fallback
        self.adjacency = adjacency
        self.degree_pairs = degree_pairs or []
        self.buckets = buckets or []
        self.attempts = 0
        self.matched_draws = 0
        self.fallback_draws = 0
        self.uniform_draws = 0
        self._degraded = False
        self._uniform_only = strategy == "uniform_random" or not self._hard_available()

    def _hard_available(self) -> bool:
        if self.strategy == "degree_matched":
            return bool(self.degree_pairs) and any(self.buckets)
        if self.strategy == "two_hop":
            return bool(self.adjacency) and any(self.adjacency)
        return False

    # ---------------------------------------------------------------- draws
    def _candidate_uniform(self) -> Pair:
        u = self.rng.randrange(self.num_nodes)
        v = self.rng.randrange(self.num_nodes)
        return (u, v) if u < v else (v, u)

    def _candidate_degree_matched(self) -> Pair | None:
        """One degree-matched draw, or ``None`` if no candidate was produced."""
        bu, bv = self.degree_pairs[self.rng.randrange(len(self.degree_pairs))]
        pool_u = self.buckets[bu] if bu < len(self.buckets) else []
        pool_v = self.buckets[bv] if bv < len(self.buckets) else []
        if not pool_u or not pool_v:
            return None
        u = pool_u[self.rng.randrange(len(pool_u))]
        v = pool_v[self.rng.randrange(len(pool_v))]
        return (u, v) if u < v else (v, u)

    def _candidate_two_hop(self) -> Pair | None:
        u = self.rng.randrange(self.num_nodes)
        first = self.adjacency[u]
        if not first:
            return None
        w = first[self.rng.randrange(len(first))]
        second = self.adjacency[w]
        if not second:
            return None
        v = second[self.rng.randrange(len(second))]
        return (u, v) if u < v else (v, u)

    def draw(self, count: int, per_pair_attempts: int) -> tuple[Pair, ...]:
        """Fill ``count`` slots; raises only when no fallback can help."""
        accepted: list[Pair] = []
        for _ in range(count):
            try:
                pair, used_fallback = self._try_one(per_pair_attempts)
            except ValueError as exc:
                available = (self.num_nodes * (self.num_nodes - 1) // 2
                             - len(self.forbidden) - len(self.used))
                raise ValueError(
                    f"split {self.split_name!r}: {exc} "
                    f"[requested={count}, accepted={len(accepted)}, "
                    f"num_nodes={self.num_nodes}, "
                    f"forbidden_known_positives={len(self.forbidden)}, "
                    f"remaining_non_edges≈{available}; "
                    "lower negatives_ratio or build the graph with more nodes]") from exc
            if self.strategy == "uniform_random":
                self.uniform_draws += 1
            elif used_fallback:
                self.fallback_draws += 1
            else:
                self.matched_draws += 1
            accepted.append(pair)
        return tuple(sorted(accepted))

    def _try_one(self, per_pair_attempts: int) -> tuple[Pair, bool]:
        """Return ``(pair, used_fallback)``; fallback is declared in the stats."""
        for _ in range(per_pair_attempts):
            self.attempts += 1
            if not self._uniform_only:
                pair = (self._candidate_degree_matched()
                        if self.strategy == "degree_matched"
                        else self._candidate_two_hop())
                if pair is not None and self._accept(pair):
                    return pair, False
                continue
            pair = self._candidate_uniform()
            if self._accept(pair):
                return pair, self._degraded
        # Per-pair budget exhausted: apply the declared fallback.
        if self.fallback == "error":
            raise ValueError(
                f"could not draw a '{self.strategy}' negative within "
                f"{per_pair_attempts} attempts; the candidate pool is exhausted "
                "(pass fallback='uniform' to degrade instead of failing)")
        self._uniform_only = True
        self._degraded = True
        for _ in range(per_pair_attempts):
            self.attempts += 1
            pair = self._candidate_uniform()
            if self._accept(pair):
                return pair, True
        raise ValueError(
            "could not draw any valid negative; the graph is probably too dense "
            "for the requested ratio")

    def _accept(self, pair: Pair) -> bool:
        u, v = pair
        if u == v or pair in self.forbidden or pair in self.used:
            return False
        self.used.add(pair)
        return True


def _split_seed(seed: int, index: int) -> int:
    """Per-split RNG seed; depends on ``(seed, split index)`` only."""
    state = np.random.SeedSequence([int(seed), int(index), 0x4E47]).generate_state(1)
    return int(state[0])


def sample_split_negatives(num_nodes: int, counts: dict[str, int], forbidden_pairs,
                           seed: int = 42, strategy: str = "uniform_random",
                           reference_edges=None, degree_buckets: int = 10,
                           max_attempts_factor: int = 50,
                           fallback: str = "uniform") -> NegativePartition:
    """Sample ``train``/``val``/``test`` negatives that are mutually disjoint.

    Parameters
    ----------
    num_nodes:
        Graph size; negatives are drawn from ``[0, num_nodes)``.
    counts:
        ``{"train": n_train, "val": n_val, "test": n_test}``; any subset of the
        three keys is allowed. Splits are drawn in :data:`SPLIT_ORDER`, so the
        training negatives never depend on the evaluation sizes.
    forbidden_pairs:
        Complete known-positive universe of the target channel.
    reference_edges:
        The *training* positive edges used by ``degree_matched`` (endpoint degree
        pairs) and ``two_hop`` (walks). Never pass held-out edges.
    fallback:
        ``"uniform"`` / ``"relaxed_degree"`` degrade to uniform and are counted;
        ``"error"`` raises instead of degrading.
    """
    if strategy not in NEGATIVE_STRATEGIES:
        raise ValueError(f"strategy must be one of {NEGATIVE_STRATEGIES}, got {strategy!r}")
    if fallback not in FALLBACK_POLICIES:
        raise ValueError(f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}")
    if num_nodes < 2:
        raise ValueError("need at least two nodes to sample negatives")
    if not isinstance(counts, dict):
        raise TypeError("counts must be a mapping split -> int")
    unknown = set(counts) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown split(s) in counts: {sorted(unknown)}")
    for split, value in counts.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"counts[{split!r}] must be a non-negative int, got {value!r}")
    forbidden = _as_forbidden(forbidden_pairs)

    adjacency = degree_pairs = buckets = None
    reference_count = 0
    degrees = None
    if strategy != "uniform_random":
        if reference_edges is None:
            raise ValueError(f"strategy {strategy!r} needs the training `reference_edges` "
                             "(degree/neighbourhood must come from the visible graph)")
        edges = np.asarray(reference_edges, dtype=np.int64)
        reference_count = 0 if edges.size == 0 else int(edges.shape[1])
        degrees = degrees_from_edges(edges, num_nodes)
        if strategy == "degree_matched":
            bucket_of_node, buckets = _degree_buckets(degrees, degree_buckets)
            degree_pairs = [(int(bucket_of_node[u]), int(bucket_of_node[v]))
                            for u, v in zip(edges[0].tolist(), edges[1].tolist())] \
                if edges.size else []
        else:
            adjacency = _adjacency_lists(edges, num_nodes)

    negatives: dict[str, tuple[Pair, ...]] = {}
    per_split: dict[str, dict] = {}
    used_so_far: set[Pair] = set()
    total_requested = 0
    for index, split in enumerate(SPLIT_ORDER):
        count = int(counts.get(split, 0))
        total_requested += count
        rng = Random(_split_seed(seed, index))
        excluded_before = len(used_so_far)
        drawer = _Drawer(num_nodes, forbidden | frozenset(used_so_far), rng, strategy,
                         fallback, adjacency, degree_pairs, buckets, split_name=split)
        sampled = drawer.draw(count, per_pair_attempts=max(20, min(200, max_attempts_factor)))
        negatives[split] = sampled
        used_so_far |= set(sampled)
        entry = {
            "requested": count,
            "sampled": len(sampled),
            "attempts": int(drawer.attempts),
            "matched_draws": int(drawer.matched_draws),
            "fallback_draws": int(drawer.fallback_draws),
            "uniform_draws": int(drawer.uniform_draws),
            "fallback_rate": (float(drawer.fallback_draws) / len(sampled)) if sampled else 0.0,
            "num_forbidden_known_positives": len(forbidden),
            "num_excluded_previous_splits": excluded_before,
        }
        if degrees is not None and sampled:
            endpoints = np.asarray(sampled, dtype=np.int64)
            entry["mean_endpoint_degree"] = float(
                np.concatenate([degrees[endpoints[:, 0]], degrees[endpoints[:, 1]]]).mean())
            if reference_count:
                positives = np.asarray(reference_edges, dtype=np.int64)
                entry["mean_positive_endpoint_degree"] = float(
                    np.concatenate([degrees[positives[0]], degrees[positives[1]]]).mean())
        per_split[split] = entry

    overlap = {
        "train_val": len(set(negatives.get("train", ())) & set(negatives.get("val", ()))),
        "train_test": len(set(negatives.get("train", ())) & set(negatives.get("test", ()))),
        "val_test": len(set(negatives.get("val", ())) & set(negatives.get("test", ()))),
    }
    if any(overlap.values()):  # pragma: no cover - the accumulator prevents this
        raise RuntimeError(f"negative splits overlap after sampling: {overlap}")

    stats = {
        "strategy": strategy,
        "fallback": fallback,
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "num_reference_edges": reference_count,
        "reference_graph": "training positives only (held-out edges never enter the reference graph)",
        "degree_buckets": int(degree_buckets) if strategy == "degree_matched" else None,
        "requested_total": total_requested,
        "sampled_total": sum(len(value) for value in negatives.values()),
        "splits": per_split,
        "cross_split_overlap": overlap,
        "mutually_exclusive": True,
    }
    return NegativePartition(negatives=negatives, stats=stats)



