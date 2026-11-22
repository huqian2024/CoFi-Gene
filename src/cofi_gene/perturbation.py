"""Week 5 (M5) experiment perturbations: missing edges, cross-view masking, gene shuffle.

These helpers implement *experiment protocols*, not new models: nothing here
changes the B0-B7 definitions in ``AGENTS.md``. Everything is numpy-only, so the
protocol can be unit-tested and audited without torch or a GPU.

Three perturbations exist (``AGENTS.md`` §6, experiments 2 and 3):

``thin_channel``
    Drop a fixed fraction (10/30/50%) of one channel's **training/encoding**
    edges. Held-out val/test positives and the fixed negative sample are never
    touched, so the evaluation set is identical at every ratio and only the
    information the encoder may see shrinks. The removed subsets are nested
    (the 10% draw is contained in the 30% draw) because the shuffle seed depends
    only on ``(seed, channel)``, never on the fraction.

``mask_cross_view_positives``
    "Both views masked" (D-016): when gene pair ``p`` is a held-out positive of
    one channel, ``p`` must not stay visible in the *other* channel either --
    neither in its encoding graph nor as one of its training positives. The
    default protocol (``mask_other_view: false``) deliberately leaves the other
    view visible, so both口径 stay reproducible and comparable.

``cross_view_permutation``
    One fixed derangement of the shared gene order. Applying it to a single
    view's rows right before fusion breaks the cross-view gene correspondence
    while leaving both encoders, the supervision and the evaluation edges
    untouched (experiment 3: does B5 really use the correct A/B pairing?).

Determinism: every function is a pure function of ``(seed, fraction, channel)``,
so all models of one comparison see byte-identical perturbed graphs.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from .dataset import ChannelSplit, CoFiDataset
from .graph_build import index_array_to_pairs, undirected_pair_keys
from .splits import simulate_missing_edges

__all__ = [
    "channel_seed",
    "thin_channel",
    "mask_cross_view_positives",
    "cross_view_permutation",
    "permutation_fingerprint",
    "apply_data_perturbations",
]


def channel_seed(seed: int, index: int) -> int:
    """Per-channel drop seed, independent of the training RNG stream.

    ``SeedSequence`` is documented as stable across numpy versions, so the same
    ``(run seed, channel index)`` yields the same missing-edge draw on every
    machine. Callers must pass the channel's position in the canonical
    ``sorted(dataset.channels)`` order, not its insertion order.
    """
    state = np.random.SeedSequence([int(seed), int(index), 0xC0F1]).generate_state(1)
    return int(state[0])


def _pair_weights(edge_index, weights) -> dict[tuple[int, int], float]:
    """Canonical ``(min,max) -> weight`` view of one edge array."""
    pairs = index_array_to_pairs(edge_index)
    array = np.zeros((0,), dtype=np.float32) if weights is None else np.asarray(
        weights, dtype=np.float32).reshape(-1)
    if array.size == 0:
        if pairs:
            raise ValueError(f"{len(pairs)} edges come without weights")
        return {}
    if array.size != len(pairs):
        raise ValueError(f"{array.size} weights for {len(pairs)} edges")
    mapping: dict[tuple[int, int], float] = {}
    for pair, value in zip(pairs, array.tolist()):
        previous = mapping.get(pair)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting weights for edge {pair}")
        mapping[pair] = float(value)
    return mapping


def _subset_arrays(edge_index, weights, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``pairs`` (already canonical) and their weights, in sorted order."""
    pairs = tuple(pairs)
    mapping = _pair_weights(edge_index, weights)
    if not pairs:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.float32))
    missing = [pair for pair in pairs if pair not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} requested edges are not in the graph, "
                         f"e.g. {missing[:3]}")
    edges = np.asarray([[u for u, _ in pairs], [v for _, v in pairs]], dtype=np.int64)
    values = np.asarray([mapping[pair] for pair in pairs], dtype=np.float32)
    return edges, values


def thin_channel(channel: ChannelSplit, fraction: float, seed: int,
                 num_nodes: int) -> tuple[ChannelSplit, dict]:
    """Drop ``fraction`` of one channel's encoder edges; everything else is intact.

    Only ``encode_edges``/``encode_weights`` change. ``train``/``val``/``test``
    positives and all negatives are returned untouched, which is what keeps the
    evaluation set fixed across the 10/30/50% settings (``AGENTS.md`` §6).
    """
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    pairs = index_array_to_pairs(channel.encode_edges)
    kept, removed = simulate_missing_edges(pairs, fraction, num_nodes, seed=int(seed))
    edges, weights = _subset_arrays(channel.encode_edges, channel.encode_weights, kept)
    stats = {
        "fraction": fraction,
        "seed": int(seed),
        "num_encode_edges_before": len(set(pairs)),
        "num_encode_edges_after": len(kept),
        "num_encode_edges_removed": len(removed),
    }
    return replace(channel, encode_edges=edges, encode_weights=weights), stats


def mask_cross_view_positives(dataset: CoFiDataset) -> tuple[CoFiDataset, dict]:
    """Remove held-out positives of one view from the other view entirely.

    For every channel ``c`` the banned set is the union of the other channels'
    held-out (val + test) positives. Those pairs are removed from ``c``'s
    encoding graph **and** from ``c``'s training positives, so a pair hidden for
    evaluation is not re-taught to the shared representation through the other
    view's supervision. Negatives, val/test splits and the node set never change.
    """
    channels = dataset.channel_names
    if len(channels) < 2:
        raise ValueError("cross-view masking needs at least two channels")
    held_out = {name: (dataset.channel(name).pairs("val")
                       | dataset.channel(name).pairs("test"))
                for name in channels}
    stats: dict = {"channels": {}}
    updated: dict[str, ChannelSplit] = {}
    for name in channels:
        part = dataset.channel(name)
        banned = frozenset().union(*(held_out[other] for other in channels
                                     if other != name))
        kept_encode = tuple(pair for pair in index_array_to_pairs(part.encode_edges)
                            if pair not in banned)
        kept_train = tuple(pair for pair in index_array_to_pairs(part.train)
                           if pair not in banned)
        encode_edges, encode_weights = _subset_arrays(part.encode_edges,
                                                      part.encode_weights, kept_encode)
        train, train_weights = _subset_arrays(part.train, part.train_weights, kept_train)
        updated[name] = replace(part, train=train, train_weights=train_weights,
                                encode_edges=encode_edges, encode_weights=encode_weights)
        encode_before = len(set(index_array_to_pairs(part.encode_edges)))
        train_before = len(set(index_array_to_pairs(part.train)))
        stats["channels"][name] = {
            "num_encode_edges_before": encode_before,
            "num_encode_edges_after": len(kept_encode),
            "num_encode_edges_removed": encode_before - len(kept_encode),
            "num_train_positives_before": train_before,
            "num_train_positives_after": len(kept_train),
            "num_train_positives_removed": train_before - len(kept_train),
        }
    return replace(dataset, channels=updated), stats



def cross_view_permutation(num_nodes: int, seed: int) -> np.ndarray:
    """Fixed derangement of ``0..N-1`` derived only from ``(num_nodes, seed)``.

    A derangement (no ``perm[i] == i``) is required: the identity mapping would
    be a no-op and could not serve as a misalignment control. The construction
    samples a random permutation and repairs its fixed points by rotating their
    values, so the result stays a bijection without biasing the rest of the map.
    """
    num_nodes = int(num_nodes)
    if num_nodes < 2:
        raise ValueError("a derangement needs at least two genes")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5EED]))
    permutation = rng.permutation(num_nodes).astype(np.int64)
    fixed = np.flatnonzero(permutation == np.arange(num_nodes))
    if fixed.size == 1:
        index = int(fixed[0])
        neighbour = (index + 1) % num_nodes
        permutation[index], permutation[neighbour] = (permutation[neighbour],
                                                      permutation[index])
    elif fixed.size > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    if np.any(permutation == np.arange(num_nodes)):
        raise RuntimeError("failed to build a fixed-point-free permutation")
    return permutation


def permutation_fingerprint(permutation) -> str:
    """Short sha256 of a permutation, recorded in ``metrics.json`` for auditing."""
    array = np.ascontiguousarray(np.asarray(permutation, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def apply_data_perturbations(dataset: CoFiDataset, config: dict) -> tuple[CoFiDataset, dict]:
    """Apply ``data.edge_drop_fraction`` / ``data.mask_other_view`` from a config.

    Order is fixed: cross-view masking first, then edge thinning on the masked
    graph. The returned ``stats`` block is what the trainer writes into
    ``metrics.json``; the on-disk ``dataset.npz`` is never modified, so the
    recorded dataset hash keeps pointing at the unperturbed build.
    """
    data_cfg = dict(config.get("data") or {})
    runtime_cfg = dict(config.get("runtime") or {})
    fraction = float(data_cfg.get("edge_drop_fraction") or 0.0)
    mask = bool(data_cfg.get("mask_other_view", False))
    seed = int(runtime_cfg.get("seed", 0))
    if not 0.0 <= fraction < 1.0:
        raise ValueError("data.edge_drop_fraction must be in [0, 1)")

    channels = dataset.channel_names
    before = {name: len(undirected_pair_keys(dataset.channel(name).encode_edges))
              for name in channels}
    train_before = {name: len(undirected_pair_keys(dataset.channel(name).train))
                    for name in channels}

    current = dataset
    mask_stats: dict = {}
    if mask:
        current, mask_stats = mask_cross_view_positives(current)

    drop_stats: dict = {}
    if fraction > 0.0:
        thinned: dict[str, ChannelSplit] = {}
        for index, name in enumerate(channels):
            part, stats = thin_channel(current.channel(name), fraction,
                                       channel_seed(seed, index), current.num_nodes)
            thinned[name] = part
            drop_stats[name] = stats
        current = replace(current, channels=thinned)

    after = {name: len(undirected_pair_keys(current.channel(name).encode_edges))
             for name in channels}
    train_after = {name: len(undirected_pair_keys(current.channel(name).train))
                   for name in channels}
    stats = {
        "edge_drop_fraction": fraction,
        "mask_other_view": mask,
        "seed": seed,
        "channels": {
            name: {
                "num_encode_edges_before": before[name],
                "num_encode_edges_after": after[name],
                "num_encode_edges_removed": before[name] - after[name],
                "num_train_positives_before": train_before[name],
                "num_train_positives_after": train_after[name],
                "cross_view_mask": (mask_stats.get("channels", {}).get(name)
                                    if mask else None),
                "edge_drop": drop_stats.get(name),
            }
            for name in channels
        },
    }
    return current, stats

