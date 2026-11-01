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
        """Assert the leakage invariants and return a small report.

        Raises ``ValueError`` with a specific message on the first violation, so a
        broken build fails loudly instead of silently training on held-out edges.
        """
        num_nodes = self.num_nodes
        report: dict = {"num_nodes": num_nodes, "num_features": self.num_features,
                        "channels": {}}
        for name, channel in self.channels.items():
            for split in THREE_WAY:
                edges = getattr(channel, split)
                if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
                    raise ValueError(f"channel {name} {split}: edge index out of range")
            train, val, test = channel.pairs("train"), channel.pairs("val"), channel.pairs("test")
            if (train & val) or (train & test) or (val & test):
                raise ValueError(f"channel {name}: positive splits overlap")
            encode = undirected_pair_keys(channel.encode_edges)
            leaked = encode & (val | test)
            if leaked:
                raise ValueError(
                    f"channel {name}: {len(leaked)} held-out positive edge(s) are present "
                    f"in the encoder graph, e.g. {sorted(leaked)[:3]}")
            known = train | val | test
            for split in THREE_WAY:
                negatives = getattr(channel, f"neg_{split}")
                negative_pairs = channel.negative_pairs(split)
                if len(negative_pairs) != negatives.shape[1]:
                    raise ValueError(f"channel {name} neg_{split}: duplicate negatives")
                bad = negative_pairs & known
                if bad:
                    raise ValueError(
                        f"channel {name} neg_{split}: {len(bad)} negative(s) are known "
                        f"positives, e.g. {sorted(bad)[:3]}")
            report["channels"][name] = {
                "num_edges_encode": len(encode),
                "num_train": len(train), "num_val": len(val), "num_test": len(test),
                "num_neg_train": channel.neg_train.shape[1],
                "num_neg_val": channel.neg_val.shape[1],
                "num_neg_test": channel.neg_test.shape[1],
            }
        partitions = [set(self.target_splits.get(s, ())) for s in THREE_WAY]
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                if partitions[i] & partitions[j]:
                    raise ValueError("downstream target splits overlap (ungrouped leakage)")
        report["num_targets"] = sum(len(part) for part in partitions)
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


