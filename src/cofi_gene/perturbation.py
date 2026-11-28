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
from .graph_build import (index_array_to_pairs, subset_edge_arrays, undirected_pair_keys)
from .protocols import apply_visibility_protocol
from .splits import simulate_missing_edges

__all__ = [
    "channel_seed",
    "thin_channel",
    "remove_training_edges",
    "apply_edge_removal",
    "mask_cross_view_positives",
    "cross_view_permutation",
    "permutation_fingerprint",
    "apply_data_perturbations",
    "EDGE_REMOVAL_MODES",
]

#: ``adjacency`` drops only encoder edges (M5 Mode 1); ``supervision_aware`` also
#: drops the matching training positives (M5.1 Mode 2).
EDGE_REMOVAL_MODES = ("adjacency", "supervision_aware")


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


def remove_training_edges(channel: ChannelSplit, fraction: float, seed: int,
                          num_nodes: int, supervision: bool = True
                          ) -> tuple[ChannelSplit, dict]:
    """Remove encoder edges and, optionally, the matching training positives.

    ``supervision=True`` implements M5.1 Mode 2 (*supervision-aware* removal): the
    experiment claims "the training relation is unknown", so the removed pairs
    must also disappear from the training positives that supervise the edge head.
    ``supervision=False`` is the historic M5 Mode 1 (adjacency dropout), which
    keeps the supervision and only shrinks what the encoder can read.

    Selection is the same nested draw as :func:`thin_channel` (seed depends on
    ``(seed, channel)`` only), so 10% ⊂ 30% ⊂ 50% and every model of one
    comparison sees the identical removed subset. val/test positives, the
    negative sample and the node set are never touched; a removed pair cannot
    reappear as a negative because it was a training positive.
    """
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    encode_pairs = index_array_to_pairs(channel.encode_edges)
    kept, removed = simulate_missing_edges(encode_pairs, fraction, num_nodes, seed=int(seed))
    encode_edges, encode_weights = subset_edge_arrays(channel.encode_edges,
                                                      channel.encode_weights, kept)
    stats = {
        "mode": "supervision_aware" if supervision else "adjacency",
        "fraction": fraction,
        "seed": int(seed),
        "num_encode_edges_before": len(set(encode_pairs)),
        "num_encode_edges_after": len(set(kept)),
        "num_encode_edges_removed": len(removed),
    }
    updated = replace(channel, encode_edges=encode_edges, encode_weights=encode_weights)
    if not supervision or not removed:
        stats.update({"supervision_changed": False,
                      "num_train_positives_before": len(channel.pairs("train")),
                      "num_train_positives_after": len(channel.pairs("train")),
                      "num_train_positives_removed": 0})
        return updated, stats

    banned = set(removed)
    train_pairs = index_array_to_pairs(channel.train)
    keep_train = tuple(pair for pair in train_pairs if pair not in banned)
    train, train_weights = subset_edge_arrays(channel.train, channel.train_weights, keep_train)
    stats.update({
        "supervision_changed": len(keep_train) != len(train_pairs),
        "num_train_positives_before": len(set(train_pairs)),
        "num_train_positives_after": len(set(keep_train)),
        "num_train_positives_removed": len(set(train_pairs)) - len(set(keep_train)),
    })
    return replace(updated, train=train, train_weights=train_weights), stats


def apply_edge_removal(dataset: CoFiDataset, mode: str, fraction: float, seed: int,
                       channels=None) -> tuple[CoFiDataset, dict]:
    """Apply one edge-removal mode to the selected channels of ``dataset``.

    ``channels=None`` means "every channel"; a subset (e.g. ``["A"]``) makes a
    single-source damage experiment distinguishable from "both views damaged",
    and the returned ``stats`` records exactly which channels were touched.
    """
    if mode not in EDGE_REMOVAL_MODES:
        raise ValueError(f"mode must be one of {EDGE_REMOVAL_MODES}, got {mode!r}")
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    available = dataset.channel_names
    selected = tuple(available if channels is None else channels)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; available: {available}")
    if not selected:
        raise ValueError("at least one channel must be selected")
    stats: dict = {"mode": mode, "fraction": fraction, "seed": int(seed),
                   "channels_selected": list(selected), "channels": {}}
    if fraction == 0.0:
        for name in available:
            part = dataset.channel(name)
            stats["channels"][name] = _unchanged_channel_stats(part, mode, seed)
        stats["changed"] = False
        return dataset, stats

    supervision = mode == "supervision_aware"
    updated: dict[str, ChannelSplit] = {}
    for index, name in enumerate(available):
        part = dataset.channel(name)
        if name in selected:
            part, channel_stats = remove_training_edges(part, fraction,
                                                        channel_seed(seed, index),
                                                        dataset.num_nodes,
                                                        supervision=supervision)
        else:
            channel_stats = _unchanged_channel_stats(part, mode, seed)
            channel_stats["untouched"] = True
        updated[name] = part
        stats["channels"][name] = channel_stats
    result = replace(dataset, channels=updated)
    result.validate()
    stats["changed"] = True
    return result, stats


def _unchanged_channel_stats(channel: ChannelSplit, mode: str, seed: int) -> dict:
    """Per-channel block for a channel that was not perturbed (audit transparency)."""
    encode_count = len(undirected_pair_keys(channel.encode_edges))
    train_count = len(channel.pairs("train"))
    return {
        "mode": mode, "fraction": 0.0, "seed": int(seed),
        "num_encode_edges_before": encode_count,
        "num_encode_edges_after": encode_count,
        "num_encode_edges_removed": 0,
        "num_train_positives_before": train_count,
        "num_train_positives_after": train_count,
        "num_train_positives_removed": 0,
        "supervision_changed": False,
    }


def mask_cross_view_positives(dataset: CoFiDataset) -> tuple[CoFiDataset, dict]:
    """Hide each view's held-out pairs in the other view (M5 wrapper).

    Kept as the historic entry point used by the Week 5 report. It now delegates
    to :func:`cofi_gene.protocols.apply_visibility_protocol` with
    ``dual_view_strict_holdout``, so the pipeline flag, the trainer perturbation
    and the experiment protocols share one implementation (M5.1 finding B-04).
    """
    if len(dataset.channel_names) < 2:
        raise ValueError("cross-view masking needs at least two channels")
    masked, stats = apply_visibility_protocol(dataset, "dual_view_strict_holdout")
    return masked, {
        "protocol": stats["protocol"],
        "channels": stats.get("channels", {}),
        "visibility_before": stats.get("visibility_before"),
        "visibility_after": stats.get("visibility_after"),
    }



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
    """Apply the visibility protocol and the configured edge-removal mode.

    Order is fixed and recorded: **visibility protocol first**, then edge removal
    on the resulting graph. Both steps are pure functions of
    ``(dataset, protocol/mode, fraction, seed)``, so all models of one comparison
    see byte-identical graphs.

    Config keys (all optional; absent keys reproduce the historic behaviour):

    ``data.visibility_protocol``   ``cross_view_transfer`` (default) | ``dual_view_strict_holdout``
    ``data.mask_other_view``       legacy alias for the strict protocol
    ``data.edge_removal_mode``     ``adjacency`` (default) | ``supervision_aware``
    ``data.edge_drop_fraction``    fraction of encoder edges to remove
    ``data.drop_channels``         subset of channels to damage (default: all)
    """
    data_cfg = dict(config.get("data") or {})
    runtime_cfg = dict(config.get("runtime") or {})
    seed = int(runtime_cfg.get("seed", 0))
    legacy_mask = data_cfg.get("mask_other_view")
    protocol = data_cfg.get("visibility_protocol")
    if protocol is None:
        protocol = ("dual_view_strict_holdout" if legacy_mask
                    else "cross_view_transfer")
    elif legacy_mask is not None and bool(legacy_mask) != (
            protocol == "dual_view_strict_holdout"):
        raise ValueError("data.mask_other_view conflicts with data.visibility_protocol; "
                         "set only visibility_protocol for new runs")
    mode = data_cfg.get("edge_removal_mode") or "adjacency"
    fraction = float(data_cfg.get("edge_drop_fraction") or 0.0)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("data.edge_drop_fraction must be in [0, 1)")
    drop_channels = data_cfg.get("drop_channels")

    current, protocol_stats = apply_visibility_protocol(dataset, protocol)
    current, removal_stats = apply_edge_removal(current, mode, fraction, seed,
                                                drop_channels)

    channels = dataset.channel_names
    before = {name: len(undirected_pair_keys(dataset.channel(name).encode_edges))
              for name in channels}
    after = {name: len(undirected_pair_keys(current.channel(name).encode_edges))
             for name in channels}
    train_before = {name: len(dataset.channel(name).pairs("train")) for name in channels}
    train_after = {name: len(current.channel(name).pairs("train")) for name in channels}
    changed = bool(protocol_stats["changed"] or removal_stats["changed"])
    stats = {
        "edge_drop_fraction": fraction,
        # Legacy field kept so the Week 5 report/readers keep working; it is now
        # derived from the protocol instead of being an independent flag.
        "mask_other_view": protocol == "dual_view_strict_holdout",
        "visibility_protocol": protocol,
        "edge_removal_mode": mode,
        "drop_channels": list(removal_stats["channels_selected"]),
        "seed": seed,
        "changed": changed,
        "visibility": protocol_stats,
        "edge_removal": removal_stats,
        "channels": {
            name: {
                "num_encode_edges_before": before[name],
                "num_encode_edges_after": after[name],
                "num_encode_edges_removed": before[name] - after[name],
                "num_train_positives_before": train_before[name],
                "num_train_positives_after": train_after[name],
                "num_train_positives_removed": train_before[name] - train_after[name],
                "cross_view_mask": protocol_stats.get("channels", {}).get(name),
                "edge_drop": (removal_stats["channels"].get(name)
                              if removal_stats["changed"] else None),
            }
            for name in channels
        },
    }
    return current, stats

