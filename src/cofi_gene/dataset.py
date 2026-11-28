"""The processed dataset container shared by every B0-B7 experiment.

Design notes:

- Node order is a single canonical ``gene_ids`` tuple; every array is indexed by
  position, so no stage may re-sort genes independently.
- Each channel stores the *held-out* positives (val/test) separately from the
  encoding graph. The encoder only ever sees ``encode_edges`` (the training
  subset by default), which is what makes the leakage tests meaningful.
- Negatives already exclude every known positive of that channel, including the
  held-out ones.
- The container is torch-free so data tests run on a CPU with no deep-learning
  stack installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .graph_build import undirected_pair_keys
from .splits import THREE_WAY

__all__ = ["ChannelSplit", "CoFiDataset", "DATASET_VERSION"]

DATASET_VERSION = 1
_ARRAY_SUFFIXES = ("train", "val", "test", "train_w", "val_w", "test_w",
                   "neg_train", "neg_val", "neg_test", "encode", "encode_w")

#: npz suffix -> :class:`ChannelSplit` attribute name.
_ATTR = {
    "train": "train", "val": "val", "test": "test",
    "train_w": "train_weights", "val_w": "val_weights", "test_w": "test_weights",
    "neg_train": "neg_train", "neg_val": "neg_val", "neg_test": "neg_test",
    "encode": "encode_edges", "encode_w": "encode_weights",
}


def _empty_edges() -> np.ndarray:
    return np.zeros((2, 0), dtype=np.int64)


def _empty_weights() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


def edge_pairs_checked(label: str, edges, num_nodes: int) -> frozenset:
    """Canonical pair set of one edge array, rejecting malformed content.

    Raises on: wrong shape, non-integer dtype, out-of-range endpoint, self-loop
    and duplicates (a reversed duplicate ``(u, v)``/``(v, u)`` collapses into one
    canonical pair and is therefore caught here too). ``label`` is used verbatim
    in the error message so the failing array is unambiguous.
    """
    array = np.asarray(edges)
    if array.size == 0:
        return frozenset()
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"{label}: edge_index must be [2, E], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label}: edge_index must be an integer array, got {array.dtype}")
    if int(array.min()) < 0 or int(array.max()) >= int(num_nodes):
        raise ValueError(f"{label}: edge index out of range [0, {int(num_nodes)})")
    if np.any(array[0] == array[1]):
        row = int(np.flatnonzero(array[0] == array[1])[0])
        raise ValueError(f"{label}: self-loop at edge {row} "
                         f"({int(array[0, row])}, {int(array[1, row])})")
    pairs = undirected_pair_keys(array)
    if len(pairs) != int(array.shape[1]):
        raise ValueError(f"{label}: {int(array.shape[1]) - len(pairs)} duplicate/reversed "
                         "edge(s); every undirected pair must appear once")
    return pairs


def _check(check_id: str, title: str, status: str, detail: str) -> dict:
    """One audit row: stable id, human title, PASS/FAIL and the measured detail."""
    return {"id": check_id, "title": title, "status": status, "detail": detail}


@dataclass
class ChannelSplit:
    """One evidence channel: held-out positives, negatives, and encoder graph."""

    train: np.ndarray = field(default_factory=_empty_edges)
    val: np.ndarray = field(default_factory=_empty_edges)
    test: np.ndarray = field(default_factory=_empty_edges)
    train_weights: np.ndarray = field(default_factory=_empty_weights)
    val_weights: np.ndarray = field(default_factory=_empty_weights)
    test_weights: np.ndarray = field(default_factory=_empty_weights)
    neg_train: np.ndarray = field(default_factory=_empty_edges)
    neg_val: np.ndarray = field(default_factory=_empty_edges)
    neg_test: np.ndarray = field(default_factory=_empty_edges)
    encode_edges: np.ndarray = field(default_factory=_empty_edges)
    encode_weights: np.ndarray = field(default_factory=_empty_weights)
    stats: dict = field(default_factory=dict)

    @property
    def num_positives(self) -> int:
        return int(self.train.shape[1] + self.val.shape[1] + self.test.shape[1])

    def pairs(self, split: str) -> frozenset:
        """Canonical pair set for one positive split (``train``/``val``/``test``)."""
        if split not in THREE_WAY:
            raise ValueError(f"split must be one of {THREE_WAY}")
        return undirected_pair_keys(getattr(self, split))

    def negative_pairs(self, split: str) -> frozenset:
        """Canonical pair set for the negatives of one split."""
        if split not in THREE_WAY:
            raise ValueError(f"split must be one of {THREE_WAY}")
        return undirected_pair_keys(getattr(self, f"neg_{split}"))

    def known_positives(self) -> frozenset:
        """Complete positive universe of this channel (train + val + test)."""
        return self.pairs("train") | self.pairs("val") | self.pairs("test")



@dataclass
class CoFiDataset:
    """Canonical node set + per-channel graphs + splits + downstream targets."""

    gene_ids: tuple[str, ...]
    X: np.ndarray
    channels: dict[str, ChannelSplit]
    target_splits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gene_ids:
            raise ValueError("gene_ids must not be empty")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("gene_ids contain duplicates")
        self.X = np.ascontiguousarray(np.asarray(self.X, dtype=np.float32))
        if self.X.ndim != 2 or self.X.shape[0] != len(self.gene_ids):
            raise ValueError(f"X must be [N, F] with N={len(self.gene_ids)}, got {self.X.shape}")
        if not self.channels:
            raise ValueError("at least one channel is required")

    @property
    def num_nodes(self) -> int:
        return len(self.gene_ids)

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.channels))

    def channel(self, name: str) -> ChannelSplit:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise KeyError(f"unknown channel {name!r}; available: {self.channel_names}") from exc

    def validate(self) -> dict:
        """Assert every leakage invariant and return a small report.

        Raises ``ValueError`` with the *detail* of the first failing check, so a
        broken build fails loudly instead of silently training on held-out edges.
        The checks live in :meth:`inspect`, which never raises and is what
        ``cofi_gene.audit`` turns into a PASS/FAIL JSON report.
        """
        _, _, failures = self.inspect()
        if failures:
            raise ValueError(failures[0]["detail"])
        return self.report()

    def inspect(self) -> tuple[list[dict], dict, list[dict]]:
        """Run all checks; return ``(checks, report, failures)`` without raising."""
        checks = self._structure_checks() + self._supervision_checks()
        failures = [check for check in checks if check["status"] == "FAIL"]
        return checks, self.report(), failures

    def audit_checks(self) -> list[dict]:
        """Public accessor used by ``cofi_gene.audit`` and the CLI."""
        return self.inspect()[0]

    def _structure_checks(self) -> list[dict]:
        checks: list[dict] = []
        num_nodes = self.num_nodes
        finite = bool(np.all(np.isfinite(self.X)))
        checks.append(_check(
            "C01-node-features", "node set and shared feature matrix",
            "PASS" if finite else "FAIL",
            f"{num_nodes} genes, {self.num_features} features, all finite" if finite
            else "X contains non-finite values"))
        for name, channel in self.channels.items():
            pairs: dict[str, frozenset] = {}
            for attribute in ("train", "val", "test", "encode_edges"):
                label = f"channel {name} {attribute}"
                try:
                    pairs[attribute] = edge_pairs_checked(label, getattr(channel, attribute),
                                                          num_nodes)
                    checks.append(_check(f"C02[{name}].{attribute}", f"{label} well-formed",
                                         "PASS", f"{len(pairs[attribute])} undirected edges"))
                except ValueError as exc:
                    pairs[attribute] = frozenset()
                    checks.append(_check(f"C02[{name}].{attribute}", f"{label} well-formed",
                                         "FAIL", str(exc)))
            train, val, test = pairs["train"], pairs["val"], pairs["test"]
            encode = pairs["encode_edges"]
            overlaps = {"train&val": len(train & val), "train&test": len(train & test),
                        "val&test": len(val & test)}
            bad = any(overlaps.values())
            checks.append(_check(
                f"C03[{name}].positive-disjoint", f"channel {name} positive splits disjoint",
                "PASS" if not bad else "FAIL",
                "train/val/test positives are disjoint" if not bad
                else f"positive splits overlap: {overlaps}"))
            leaked = encode & (val | test)
            checks.append(_check(
                f"C04[{name}].heldout-absent", f"channel {name} encoder excludes held-out",
                "PASS" if not leaked else "FAIL",
                f"{len(encode)} encoder edges, none of the {len(val) + len(test)} held-out "
                "positives present" if not leaked else
                f"{len(leaked)} held-out positive edge(s) are present in the encoder graph, "
                f"e.g. {sorted(leaked)[:3]}"))
            extra = encode - train
            checks.append(_check(
                f"C05[{name}].encode-subset", f"channel {name} encoder ⊆ training positives",
                "PASS" if not extra else "FAIL",
                "encoder graph is a subset of the training positives" if not extra else
                f"{len(extra)} encoder edge(s) are not training positives, "
                f"e.g. {sorted(extra)[:3]}"))
            for attribute, weight_name in (("train", "train_weights"),
                                           ("val", "val_weights"),
                                           ("test", "test_weights"),
                                           ("encode_edges", "encode_weights")):
                weights = np.asarray(getattr(channel, weight_name))
                expected = len(pairs[attribute])
                ok = weights.size == expected
                checks.append(_check(
                    f"C06[{name}].{weight_name}", f"channel {name} {weight_name} length",
                    "PASS" if ok else "FAIL",
                    f"{expected} weights" if ok
                    else f"{weight_name} has {weights.size} values for {expected} edges"))
        return checks

    def _supervision_checks(self) -> list[dict]:
        """Negatives vs positives, negative cross-split disjointness, target groups."""
        checks: list[dict] = []
        num_nodes = self.num_nodes
        for name, channel in self.channels.items():
            known = channel.known_positives()
            negative_pairs: dict[str, frozenset] = {}
            for split in THREE_WAY:
                label = f"channel {name} neg_{split}"
                try:
                    pairs = edge_pairs_checked(label, getattr(channel, f"neg_{split}"),
                                               num_nodes)
                    negative_pairs[split] = pairs
                    checks.append(_check(f"C07[{name}].neg_{split}", f"{label} well-formed",
                                         "PASS", f"{len(pairs)} sampled non-edges"))
                except ValueError as exc:
                    negative_pairs[split] = frozenset()
                    message = str(exc)
                    if "duplicate" in message:
                        message = (f"channel {name} neg_{split}: duplicate negatives "
                                   f"({message.split(': ', 1)[-1]})")
                    checks.append(_check(f"C07[{name}].neg_{split}", f"{label} well-formed",
                                         "FAIL", message))
                conflicts = negative_pairs[split] & known
                checks.append(_check(
                    f"C08[{name}].neg_{split}-positive", f"{label} vs known positives",
                    "PASS" if not conflicts else "FAIL",
                    "no sampled negative is a known positive" if not conflicts else
                    f"{len(conflicts)} negative(s) are known positives, "
                    f"e.g. {sorted(conflicts)[:3]}"))
            cross = {"train&val": len(negative_pairs["train"] & negative_pairs["val"]),
                     "train&test": len(negative_pairs["train"] & negative_pairs["test"]),
                     "val&test": len(negative_pairs["val"] & negative_pairs["test"])}
            any_cross = any(cross.values())
            checks.append(_check(
                f"C09[{name}].neg-disjoint", f"channel {name} negative splits disjoint",
                "PASS" if not any_cross else "FAIL",
                "train/val/test negatives are pairwise disjoint (M5.1 fix B-01)"
                if not any_cross else f"negative splits overlap: {cross}"))
        partitions = [set(self.target_splits.get(s, ())) for s in THREE_WAY]
        overlap = any(partitions[i] & partitions[j]
                      for i in range(len(partitions)) for j in range(i + 1, len(partitions)))
        checks.append(_check(
            "C10-target-splits", "downstream target split grouping",
            "PASS" if not overlap else "FAIL",
            f"{sum(len(part) for part in partitions)} targets in disjoint groups"
            if not overlap else "downstream target splits overlap (ungrouped leakage)"))
        return checks

    def report(self) -> dict:
        """Counts per channel/split (JSON-ready), independent of the checks."""
        report: dict = {"num_nodes": self.num_nodes, "num_features": self.num_features,
                        "channels": {}}
        for name, channel in self.channels.items():
            encode = undirected_pair_keys(channel.encode_edges)
            train = channel.pairs("train")
            val = channel.pairs("val")
            test = channel.pairs("test")
            neg_train = channel.negative_pairs("train")
            neg_val = channel.negative_pairs("val")
            neg_test = channel.negative_pairs("test")
            report["channels"][name] = {
                "num_edges_encode": len(encode),
                "num_train": len(train), "num_val": len(val), "num_test": len(test),
                "num_neg_train": int(channel.neg_train.shape[1]),
                "num_neg_val": int(channel.neg_val.shape[1]),
                "num_neg_test": int(channel.neg_test.shape[1]),
                "negative_split_overlap": {
                    "train&val": len(neg_train & neg_val),
                    "train&test": len(neg_train & neg_test),
                    "val&test": len(neg_val & neg_test),
                },
            }
        report["num_targets"] = sum(len(set(self.target_splits.get(s, ())))
                                   for s in THREE_WAY)
        return report


    def summary(self) -> dict:
        return self.validate()

    # ------------------------------------------------------------------ io
    def save(self, directory: str | Path) -> Path:
        """Persist arrays to ``dataset.npz`` and metadata to ``dataset.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"X": self.X}
        for name, channel in self.channels.items():
            for suffix in _ARRAY_SUFFIXES:
                arrays[f"{name}_{suffix}"] = np.asarray(getattr(channel, _ATTR[suffix]))
        np.savez_compressed(directory / "dataset.npz", **arrays)
        payload = {
            "dataset_version": DATASET_VERSION,
            "gene_ids": list(self.gene_ids),
            "channels": {name: channel.stats for name, channel in self.channels.items()},
            "target_splits": {k: list(v) for k, v in self.target_splits.items()},
            "meta": self.meta,
        }
        (directory / "dataset.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CoFiDataset":
        directory = Path(directory)
        payload = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        if payload.get("dataset_version") != DATASET_VERSION:
            raise ValueError(f"unsupported dataset version: {payload.get('dataset_version')}")
        channels: dict[str, ChannelSplit] = {}
        with np.load(directory / "dataset.npz", allow_pickle=False) as archive:
            X = archive["X"]
            for name, stats in payload["channels"].items():
                values = {suffix: archive[f"{name}_{suffix}"] for suffix in _ARRAY_SUFFIXES
                          if f"{name}_{suffix}" in archive}
                kwargs = {_ATTR[suffix]: values.get(suffix) for suffix in _ARRAY_SUFFIXES}
                missing = [attr for attr, value in kwargs.items() if value is None]
                if missing:
                    raise ValueError(f"channel {name} is missing arrays for: {missing}")
                channels[name] = ChannelSplit(stats=stats or {}, **kwargs)
        return cls(gene_ids=tuple(payload["gene_ids"]), X=X, channels=channels,
                   target_splits={k: tuple(v) for k, v in payload.get("target_splits", {}).items()},
                   meta=payload.get("meta", {}))


