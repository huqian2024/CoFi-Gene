"""Cross-view visibility protocols (M5.1 Phase 3, findings B-03/B-04).

Two scientifically different questions need two different protocols. Before M5.1
the repository had "no masking" (a pipeline flag) and "mask the other view" (a
trainer perturbation) implemented in two places, with nothing proving they
agreed. This module is the single implementation; ``cofi_gene.perturbation`` and
``cofi_gene.pipeline`` both delegate here.

``cross_view_transfer`` (Protocol A, historic default)
    Each channel hides its own val/test positives from its encoder and its
    supervision, but the *other* channel may still contain that gene pair. The
    claim is "use evidence that already exists in another source to fill this
    source", i.e. cross-evidence transfer -- not recovery of a truly unknown
    relation. The number of evaluation positives visible in the other view must
    therefore be reported, never assumed to be zero.

``dual_view_strict_holdout`` (Protocol B)
    For every channel, the union of *all* channels' val/test positives is removed
    from **both** encoding graphs and from **both** training-positive sets. What
    remains has no direct supervision in either view, so a good score cannot come
    from copying the answer out of the sibling graph. Removed positives never
    become negatives: they are known positives and the dataset build already
    excludes every known positive from the negative pools.

Both protocols keep nodes, features, val/test positives and the negative sample
untouched, and both re-run ``CoFiDataset.validate()`` afterwards. The returned
``stats`` block is written into ``metrics.json``/``audit.json`` so a reader can
tell which protocol produced a number.
"""

from __future__ import annotations

from dataclasses import replace

from .dataset import ChannelSplit, CoFiDataset
from .graph_build import index_array_to_pairs, subset_edge_arrays, undirected_pair_keys

__all__ = [
    "VISIBILITY_PROTOCOLS",
    "PROTOCOL_DESCRIPTIONS",
    "visibility_report",
    "holdout_union",
    "apply_visibility_protocol",
]

#: Protocol A (historic) and Protocol B (strict), in reporting order.
VISIBILITY_PROTOCOLS = ("cross_view_transfer", "dual_view_strict_holdout")

PROTOCOL_DESCRIPTIONS = {
    "cross_view_transfer": (
        "each channel hides its own val/test positives from its encoder and "
        "supervision; the other view may still contain the pair (cross-evidence "
        "transfer, not unknown-relation recovery)"),
    "dual_view_strict_holdout": (
        "the union of all channels' val/test positives is removed from every "
        "encoding graph and every training-positive set (no direct supervision "
        "in either view)"),
}


def holdout_union(dataset: CoFiDataset) -> frozenset:
    """Union of every channel's held-out (val + test) positive pairs."""
    pairs: frozenset = frozenset()
    for name in dataset.channel_names:
        channel = dataset.channel(name)
        pairs = pairs | channel.pairs("val") | channel.pairs("test")
    return pairs


def visibility_report(dataset: CoFiDataset) -> dict:
    """How much of each channel's evaluation set is visible in the other view.

    Reports, per channel and for the union of the other channels:

    ``visible_in_other_encode``
        held-out positives present in the other channel's *encoder* graph, i.e.
        information the GCN can read.
    ``visible_in_other_train``
        held-out positives present in the other channel's *training positives*,
        i.e. information the shared representation is explicitly taught.
    ``train_removed_if_strict``
        how many of this channel's own training positives disappear under
        ``dual_view_strict_holdout`` (the cost of the stricter protocol).
    """
    names = dataset.channel_names
    banned = holdout_union(dataset)
    report: dict = {"channels": {}, "holdout_union_size": len(banned),
                    "n_channels": len(names)}
    for name in names:
        channel = dataset.channel(name)
        val, test = channel.pairs("val"), channel.pairs("test")
        held_out = val | test
        others = [other for other in names if other != name]
        other_encode: frozenset = frozenset()
        other_train: frozenset = frozenset()
        for other in others:
            part = dataset.channel(other)
            other_encode = other_encode | undirected_pair_keys(part.encode_edges)
            other_train = other_train | part.pairs("train")
        visible_encode = held_out & other_encode
        visible_train = held_out & other_train
        test_visible = test & other_encode
        report["channels"][name] = {
            "num_val": len(val),
            "num_test": len(test),
            "num_train": len(channel.pairs("train")),
            "visible_in_other_encode": len(visible_encode),
            "visible_in_other_train": len(visible_train),
            "visible_test_in_other_encode": len(test_visible),
            "visible_val_in_other_encode": len(val & other_encode),
            "visibility_ratio_encode": (len(visible_encode) / len(held_out))
            if held_out else 0.0,
            "visibility_ratio_test_in_other_encode": (len(test_visible) / len(test))
            if test else 0.0,
            "train_removed_if_strict": len(channel.pairs("train") & banned),
            "other_channels": list(others),
        }
    return report


def _remove_pairs_from_channel(channel: ChannelSplit,
                               banned: frozenset) -> tuple[ChannelSplit, dict]:
    """Drop ``banned`` from one channel's encoder graph and training positives."""
    encode_pairs = index_array_to_pairs(channel.encode_edges)
    train_pairs = index_array_to_pairs(channel.train)
    keep_encode = tuple(pair for pair in encode_pairs if pair not in banned)
    keep_train = tuple(pair for pair in train_pairs if pair not in banned)
    encode_edges, encode_weights = subset_edge_arrays(channel.encode_edges,
                                                      channel.encode_weights, keep_encode)
    train, train_weights = subset_edge_arrays(channel.train, channel.train_weights, keep_train)
    stats = {
        "num_encode_edges_before": len(set(encode_pairs)),
        "num_encode_edges_after": len(set(keep_encode)),
        "num_encode_edges_removed": len(set(encode_pairs)) - len(set(keep_encode)),
        "num_train_positives_before": len(set(train_pairs)),
        "num_train_positives_after": len(set(keep_train)),
        "num_train_positives_removed": len(set(train_pairs)) - len(set(keep_train)),
    }
    return replace(channel, train=train, train_weights=train_weights,
                   encode_edges=encode_edges, encode_weights=encode_weights), stats


def apply_visibility_protocol(dataset: CoFiDataset,
                              protocol: str = "cross_view_transfer"
                              ) -> tuple[CoFiDataset, dict]:
    """Return ``(dataset, stats)`` for one visibility protocol.

    ``cross_view_transfer`` is the identity transform: it only *measures* the
    cross-view visibility that the protocol deliberately allows. The strict
    protocol removes the held-out union from both graphs and both supervision
    sets, then re-validates the result.
    """
    if protocol not in VISIBILITY_PROTOCOLS:
        raise ValueError(f"protocol must be one of {VISIBILITY_PROTOCOLS}, got {protocol!r}")
    before = visibility_report(dataset)
    stats: dict = {"protocol": protocol,
                   "protocol_description": PROTOCOL_DESCRIPTIONS[protocol],
                   "visibility_before": before,
                   "channels": {}}
    if protocol == "cross_view_transfer":
        stats["changed"] = False
        stats["visibility_after"] = before
        return dataset, stats

    banned = holdout_union(dataset)
    if len(dataset.channel_names) < 2:
        stats["changed"] = False
        stats["note"] = "single-channel dataset: strict holdout equals per-channel hiding"
        stats["visibility_after"] = before
        return dataset, stats
    updated: dict[str, ChannelSplit] = {}
    for name in dataset.channel_names:
        part, channel_stats = _remove_pairs_from_channel(dataset.channel(name), banned)
        updated[name] = part
        stats["channels"][name] = channel_stats
    masked = replace(dataset, channels=updated)
    masked.validate()
    after = visibility_report(masked)
    stats["changed"] = True
    stats["holdout_union_size"] = len(banned)
    stats["visibility_after"] = after
    residual = {name: entry["visible_in_other_encode"]
                for name, entry in after["channels"].items()}
    stats["residual_visible_in_other_encode"] = residual
    if any(residual.values()):  # pragma: no cover - validate() would already fail
        raise RuntimeError(f"strict holdout left held-out pairs visible: {residual}")
    return masked, stats

