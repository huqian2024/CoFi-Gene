"""M5.1 dataset audit: a machine-readable PASS/FAIL gate before training.

``CoFiDataset.validate()`` answers "is this dataset legal?" by raising on the
first violation. This module answers the questions a *report* needs:

* which checks passed, which failed, and with how many conflicting pairs;
* how much of each channel's evaluation set is visible in the other view (the
  number that decides whether a "masked" claim is honest);
* which protocol/edge-removal mode the run is about to apply, and what strict
  holdout costs in training supervision;
* the dataset content hash, so a result can be traced to the exact arrays.

The report is JSON-serialisable and is written to ``audit.json`` next to the run
and (in compact form) into ``metrics.json``. Any FAIL blocks training unless the
caller explicitly downgrades the audit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .dataset import CoFiDataset
from .manifest import git_commit, sha256_file, utc_timestamp

__all__ = [
    "AUDIT_VERSION",
    "AuditFailure",
    "audit_dataset",
    "require_pass",
    "summarise",
    "render_markdown",
    "write_report",
]

AUDIT_VERSION = 1


class AuditFailure(RuntimeError):
    """Raised by :func:`require_pass`; carries the full report for the caller."""

    def __init__(self, report: dict) -> None:
        failed = [check for check in report["checks"] if check["status"] == "FAIL"]
        details = "; ".join(f"{check['id']}: {check['detail']}" for check in failed[:3])
        super().__init__(f"dataset audit FAILED ({len(failed)} check(s)): {details}")
        self.report = report


def _check(check_id: str, title: str, status: str, detail: str) -> dict:
    return {"id": check_id, "title": title, "status": status, "detail": detail}


def _protocol_checks(dataset: CoFiDataset, protocol: str, mode: str) -> list[dict]:
    """Feasibility/recording checks for the protocol the run is about to apply."""
    checks = [_check("A01-protocol", "visibility protocol and edge-removal mode",
                     "PASS", f"protocol={protocol}, edge_removal_mode={mode}")]
    if protocol != "dual_view_strict_holdout":
        checks.append(_check(
            "A02-strict-holdout", "strict holdout feasibility", "SKIP",
            "protocol keeps the other view visible (Protocol A); nothing to remove"))
        return checks

    from .protocols import apply_visibility_protocol

    try:
        masked, stats = apply_visibility_protocol(dataset, protocol)
    except (ValueError, RuntimeError) as exc:
        return [_check("A02-strict-holdout", "strict holdout feasibility", "FAIL",
                       f"strict holdout could not be applied: {type(exc).__name__}: {exc}")]
    empty = [name for name in masked.channel_names
             if len(masked.channel(name).pairs("train")) == 0]
    residual = {name: entry["visible_in_other_encode"]
                for name, entry in stats["visibility_after"]["channels"].items()}
    removed = ", ".join(
        f"{name}={stats['channels'].get(name, {}).get('num_train_positives_removed', 0)}"
        for name in masked.channel_names)
    detail = (f"union={stats.get('holdout_union_size', 0)} pairs; supervision removed: "
              f"{removed}; residual visible in other encode={residual}")
    if empty:
        detail += f"; channel(s) {empty} would lose every training positive"
    checks.append(_check("A02-strict-holdout", "strict holdout feasibility",
                         "PASS" if not empty and not any(residual.values()) else "FAIL",
                         detail))
    return checks


def _recording_checks(dataset: CoFiDataset) -> list[dict]:
    """Was the negative protocol recorded, and can the feature source be traced?"""
    checks: list[dict] = []
    recorded = dataset.meta.get("negative_sampling")
    if recorded:
        strategies = {name: entry.get("strategy") for name, entry in recorded.items()}
        exclusive = all(bool(entry.get("mutually_exclusive")) for entry in recorded.values())
        checks.append(_check(
            "A04-negative-protocol", "negative sampling protocol recorded",
            "PASS" if exclusive else "FAIL",
            f"strategies={strategies}, mutually_exclusive={exclusive}"))
    else:
        checks.append(_check(
            "A04-negative-protocol", "negative sampling protocol recorded", "WARN",
            "dataset.meta has no 'negative_sampling' block: pre-M5.1 build "
            "(e.g. real_v1), so cross-split exclusion was not enforced"))
    features = dataset.meta.get("feature_reducer")
    provenance = dataset.meta.get("provenance")
    if features and provenance:
        checks.append(_check(
            "A05-feature-provenance", "feature provenance recorded", "PASS",
            f"reducer={features}, provenance={provenance}; the reducer must stay fitted on "
            "public gene priors only (DE labels never used)"))
    else:
        checks.append(_check(
            "A05-feature-provenance", "feature provenance recorded", "WARN",
            "meta lacks feature_reducer/provenance: feature generation cannot be "
            "audited from this file"))
    return checks


def _hash_check(dataset_dir: str | Path | None) -> list[dict]:
    if dataset_dir is None:
        return [_check("A06-dataset-hash", "dataset content hash", "SKIP",
                       "no dataset directory given")]
    npz = Path(dataset_dir) / "dataset.npz"
    if not npz.exists():
        return [_check("A06-dataset-hash", "dataset content hash", "FAIL",
                       f"{npz} does not exist")]
    return [_check("A06-dataset-hash", "dataset content hash", "PASS",
                   f"sha256={sha256_file(npz)}")]


def _roundtrip_check(dataset: CoFiDataset) -> list[dict]:
    """Save/load must not change the split (guards against serialisation drift)."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        dataset.save(tmp)
        restored = CoFiDataset.load(tmp)
    differences = []
    for name in dataset.channel_names:
        original, copy = dataset.channel(name), restored.channel(name)
        for attribute in ("train", "val", "test", "encode_edges",
                          "neg_train", "neg_val", "neg_test"):
            if not np.array_equal(getattr(original, attribute), getattr(copy, attribute)):
                differences.append(f"{name}.{attribute}")
    if dataset.gene_ids != restored.gene_ids or not np.array_equal(dataset.X, restored.X):
        differences.append("gene_ids/X")
    return [_check("A07-roundtrip", "save/load round-trip stability",
                   "PASS" if not differences else "FAIL",
                   "arrays identical after save+load" if not differences
                   else f"changed arrays: {differences[:5]}")]


def audit_dataset(dataset: CoFiDataset, *, dataset_dir: str | Path | None = None,
                  config: dict | None = None, protocol: str | None = None,
                  edge_removal_mode: str | None = None,
                  include_roundtrip: bool = True) -> dict:
    """Run every dataset/protocol check and return a JSON-ready report."""
    from .protocols import VISIBILITY_PROTOCOLS, visibility_report

    config = config or {}
    data_cfg = dict(config.get("data") or {})
    protocol = protocol or data_cfg.get("visibility_protocol") or (
        "dual_view_strict_holdout" if data_cfg.get("mask_other_view")
        else "cross_view_transfer")
    mode = edge_removal_mode or data_cfg.get("edge_removal_mode") or "adjacency"
    if protocol not in VISIBILITY_PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {VISIBILITY_PROTOCOLS}")

    checks = list(dataset.audit_checks())
    checks += _protocol_checks(dataset, protocol, mode)
    checks += _recording_checks(dataset)
    checks += _hash_check(dataset_dir)
    if include_roundtrip:
        checks += _roundtrip_check(dataset)
    failures = [check for check in checks if check["status"] == "FAIL"]
    warnings = [check for check in checks if check["status"] == "WARN"]
    report = {
        "audit_version": AUDIT_VERSION,
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "status": "FAIL" if failures else "PASS",
        "counts": {"checks": len(checks),
                   "passed": sum(1 for check in checks if check["status"] == "PASS"),
                   "failed": len(failures), "warnings": len(warnings)},
        "protocol": protocol,
        "edge_removal_mode": mode,
        "dataset": {
            "processed_dir": str(dataset_dir) if dataset_dir is not None else None,
            "num_nodes": dataset.num_nodes,
            "num_features": dataset.num_features,
            "channels": list(dataset.channel_names),
            "meta": dataset.meta,
        },
        "checks": checks,
        "visibility": visibility_report(dataset),
        "failed_ids": [check["id"] for check in failures],
        "warning_ids": [check["id"] for check in warnings],
    }
    return report


def require_pass(report: dict) -> dict:
    """Raise :class:`AuditFailure` unless the report status is PASS."""
    if report.get("status") != "PASS":
        raise AuditFailure(report)
    return report


def summarise(report: dict) -> dict:
    """Compact block for ``metrics.json`` (the full report goes to ``audit.json``)."""
    return {
        "status": report["status"],
        "protocol": report["protocol"],
        "edge_removal_mode": report["edge_removal_mode"],
        "counts": report["counts"],
        "failed_ids": report["failed_ids"],
        "warning_ids": report["warning_ids"],
        "holdout_union_size": report["visibility"]["holdout_union_size"],
        "visibility": {
            name: {"visible_in_other_encode": entry["visible_in_other_encode"],
                   "num_val": entry["num_val"], "num_test": entry["num_test"],
                   "visibility_ratio_encode": entry["visibility_ratio_encode"]}
            for name, entry in report["visibility"]["channels"].items()
        },
    }


def write_report(report: dict, path: str | Path) -> Path:
    """Write the full report as sorted JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path


def render_markdown(report: dict) -> str:
    """Human-readable mirror used by ``scripts/audit_dataset.py``."""
    dataset = report["dataset"]
    lines = [
        "# M5.1 dataset audit",
        "",
        f"- created_at: {report['created_at']}",
        f"- git_commit: {report['git_commit']}",
        f"- dataset: {dataset['processed_dir']}",
        f"- nodes/features: {dataset['num_nodes']}/{dataset['num_features']}",
        f"- protocol: {report['protocol']} (edge removal: {report['edge_removal_mode']})",
        f"- status: **{report['status']}** ({report['counts']['passed']} passed, "
        f"{report['counts']['failed']} failed, {report['counts']['warnings']} warning)",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        detail = str(check["detail"]).replace("|", "/")
        lines.append(f"| {check['id']} | {check['status']} | {detail} |")
    lines += ["", "## Cross-view visibility", "",
              "| channel | val | test | visible in other encode | ratio |",
              "| --- | --- | --- | --- | --- |"]
    for name, entry in report["visibility"]["channels"].items():
        lines.append(f"| {name} | {entry['num_val']} | {entry['num_test']} | "
                     f"{entry['visible_in_other_encode']} | "
                     f"{entry['visibility_ratio_encode']:.4f} |")
    return "\n".join(lines) + "\n"


