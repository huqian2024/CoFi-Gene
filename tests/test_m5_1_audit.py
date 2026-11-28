"""M5.1 audit gate: structured PASS/FAIL, visibility numbers, hash and round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cofi_gene.audit import (AUDIT_VERSION, AuditFailure, audit_dataset, render_markdown,
                            require_pass, summarise, write_report)
from cofi_gene.dataset import ChannelSplit, CoFiDataset


def _edges(*pairs) -> np.ndarray:
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray([list(pair) for pair in pairs], dtype=np.int64).T.copy()


def _dataset(neg_val=None) -> CoFiDataset:
    """Six genes, one channel, valid unless ``neg_val`` is overridden."""
    channel = ChannelSplit(
        train=_edges((0, 1), (2, 3)), train_weights=np.ones(2, dtype=np.float32),
        val=_edges((0, 3)), val_weights=np.ones(1, dtype=np.float32),
        test=_edges((1, 2)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 2), (1, 4)),
        neg_val=_edges((3, 4)) if neg_val is None else _edges(*neg_val),
        neg_test=_edges((0, 5)),
        encode_edges=_edges((0, 1), (2, 3)),
        encode_weights=np.ones(2, dtype=np.float32))
    return CoFiDataset(gene_ids=tuple(f"G{i}" for i in range(6)),
                       X=np.zeros((6, 3), dtype=np.float32),
                       channels={"A": channel},
                       target_splits={"train": ("G0",), "val": ("G1",), "test": ("G2",)},
                       meta={"feature_reducer": "pca_svd_sign_fixed",
                             "provenance": "unit-test",
                             "negative_sampling": {"A": {"strategy": "uniform_random",
                                                         "mutually_exclusive": True}}})


def test_clean_dataset_passes_every_check_including_roundtrip():
    report = audit_dataset(_dataset())
    assert report["audit_version"] == AUDIT_VERSION
    assert report["status"] == "PASS"
    ids = {check["id"] for check in report["checks"]}
    assert {"C01-node-features", "A01-protocol", "A04-negative-protocol",
            "A05-feature-provenance", "A07-roundtrip"} <= ids
    assert any(check_id.startswith("C09") for check_id in ids)
    assert report["counts"]["failed"] == 0 and report["failed_ids"] == []
    require_pass(report)                      # must not raise
    assert json.dumps(report)                 # JSON-serialisable


def test_cross_split_negative_overlap_fails_the_audit():
    report = audit_dataset(_dataset(neg_val=((0, 2),)))
    assert report["status"] == "FAIL"
    assert any(check_id.startswith("C09") for check_id in report["failed_ids"])
    with pytest.raises(AuditFailure) as excinfo:
        require_pass(report)
    assert excinfo.value.report["status"] == "FAIL"
    assert "negative splits overlap" in str(excinfo.value)


def test_missing_dataset_npz_fails_the_hash_check(tmp_path):
    report = audit_dataset(_dataset(), dataset_dir=tmp_path)
    assert "A06-dataset-hash" in report["failed_ids"]
    assert report["status"] == "FAIL"


def test_roundtrip_check_can_be_skipped():
    report = audit_dataset(_dataset(), include_roundtrip=False)
    assert "A07-roundtrip" not in {check["id"] for check in report["checks"]}
    assert report["status"] == "PASS"


def test_strict_holdout_infeasible_build_fails_before_training():
    """If strict masking would delete every training positive, the audit blocks it."""
    channel_a = ChannelSplit(
        train=_edges((0, 1)), train_weights=np.ones(1, dtype=np.float32),
        val=_edges((2, 3)), val_weights=np.ones(1, dtype=np.float32),
        test=_edges((4, 5)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 2)), neg_val=_edges((1, 3)), neg_test=_edges((0, 4)),
        encode_edges=_edges((0, 1)), encode_weights=np.ones(1, dtype=np.float32))
    channel_b = ChannelSplit(
        train=_edges((1, 2)), train_weights=np.ones(1, dtype=np.float32),
        val=_edges((0, 1)), val_weights=np.ones(1, dtype=np.float32),
        test=_edges((3, 4)), test_weights=np.ones(1, dtype=np.float32),
        neg_train=_edges((0, 5)), neg_val=_edges((1, 5)), neg_test=_edges((2, 5)),
        encode_edges=_edges((1, 2)), encode_weights=np.ones(1, dtype=np.float32))
    dataset = CoFiDataset(gene_ids=tuple(f"G{i}" for i in range(6)),
                          X=np.zeros((6, 3), dtype=np.float32),
                          channels={"A": channel_a, "B": channel_b})
    dataset.validate()
    assert audit_dataset(dataset, protocol="cross_view_transfer")["status"] == "PASS"
    strict = audit_dataset(dataset, protocol="dual_view_strict_holdout")
    assert "A02-strict-holdout" in strict["failed_ids"]


def test_unknown_protocol_is_rejected_before_any_check_runs():
    with pytest.raises(ValueError, match="unknown protocol"):
        audit_dataset(_dataset(), protocol="mask_all")


def test_report_is_written_and_rendered(tmp_path):
    report = audit_dataset(_dataset())
    json_path = write_report(report, tmp_path / "audit.json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    markdown = render_markdown(payload)
    assert "M5.1 dataset audit" in markdown
    assert "| C09[A].neg-disjoint | PASS |" in markdown
    assert "## Cross-view visibility" in markdown


def test_summarise_keeps_only_the_reusable_numbers():
    block = summarise(audit_dataset(_dataset()))
    assert set(block) == {"status", "protocol", "edge_removal_mode", "counts",
                          "failed_ids", "warning_ids", "holdout_union_size",
                          "visibility"}
    assert block["status"] == "PASS"
    assert block["protocol"] == "cross_view_transfer"
    assert "A" in block["visibility"]


def test_processed_build_audits_clean(processed_dataset):
    report = audit_dataset(processed_dataset, protocol="dual_view_strict_holdout")
    assert report["status"] == "PASS", report["failed_ids"]
    for channel in processed_dataset.channel_names:
        assert report["visibility"]["channels"][channel]["num_test"] > 0

