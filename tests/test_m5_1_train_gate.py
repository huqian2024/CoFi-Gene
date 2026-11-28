"""M5.1 training gate: the audit can stop a run, and protocol/checkpoint are traceable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cofi_gene.audit import AuditFailure  # noqa: E402
from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.evaluate import evaluate_checkpoint  # noqa: E402
from cofi_gene.train import train_link_prediction  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _config(model: str, out_dir, data: dict | None = None, epochs: int = 2):
    overrides = {"optim": {"epochs": epochs, "eval_every": 1, "early_stop_patience": 1},
                 "runtime": {"device": "cpu", "out_dir": str(out_dir), "seed": 7}}
    if data:
        overrides["data"] = data
    return load_config(REPO / "configs" / f"{model}.yaml", overrides)


def test_strict_audit_stops_training_when_the_dataset_cannot_be_verified(
        processed_dataset, tmp_path):
    config = _config("b1", tmp_path)
    config["data"]["processed_dir"] = str(tmp_path / "missing_dataset")
    with pytest.raises(AuditFailure, match="dataset audit FAILED"):
        train_link_prediction(processed_dataset, config, audit_mode="strict")
    # ... while an explicit downgrade trains and records the failure.
    record = train_link_prediction(processed_dataset, config, audit_mode="warn",
                                   evaluate_test=False)
    assert record["audit"]["status"] == "FAIL"
    assert "A06-dataset-hash" in record["audit"]["failed_ids"]


def test_protocol_block_and_audit_files_are_written(processed_dataset, tmp_path):
    config = _config("b1", tmp_path, data={"edge_removal_mode": "supervision_aware",
                                           "edge_drop_fraction": 0.5})
    record = train_link_prediction(processed_dataset, config)
    protocol = record["protocol"]
    assert protocol["visibility_protocol"] == "cross_view_transfer"
    assert protocol["edge_removal_mode"] == "supervision_aware"
    assert protocol["edge_drop_fraction"] == 0.5
    assert protocol["negative_strategy"] == {"A": "uniform_random",
                                             "B": "uniform_random"}
    removed = record["perturbation"]["channels"]["A"]["num_train_positives_removed"]
    assert removed > 0
    run_dir = Path(tmp_path) / record["run_name"]
    assert (run_dir / "audit.json").exists()
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["protocol"] == protocol
    assert saved["audit"]["status"] == "PASS"


def test_audit_off_skips_the_report(processed_dataset, tmp_path):
    record = train_link_prediction(processed_dataset, _config("b1", tmp_path),
                                   audit_mode="off", evaluate_test=False)
    assert record["audit"] is None
    assert not (Path(tmp_path) / record["run_name"] / "audit.json").exists()


def test_strict_protocol_run_records_zero_residual_visibility(processed_dataset, tmp_path):
    record = train_link_prediction(
        processed_dataset,
        _config("b4", tmp_path, data={"visibility_protocol": "dual_view_strict_holdout"}),
        evaluate_test=False)
    assert record["protocol"]["visibility_protocol"] == "dual_view_strict_holdout"
    visibility = record["perturbation"]["visibility"]["visibility_after"]["channels"]
    assert all(entry["visible_in_other_encode"] == 0 for entry in visibility.values())
    assert record["audit"]["status"] == "PASS"


def test_evaluate_checkpoint_reproduces_the_training_test_metrics(
        processed_dataset, tmp_path):
    record = train_link_prediction(processed_dataset, _config("b1", tmp_path))
    checkpoint = Path(tmp_path) / record["run_name"] / "model.pt"
    evaluated = evaluate_checkpoint(processed_dataset, checkpoint, split="test",
                                    device="cpu")
    for channel in record["test"]:
        assert evaluated["metrics"][channel]["average_precision"] == pytest.approx(
            record["test"][channel]["average_precision"], abs=1e-9)
        assert evaluated["metrics"][channel]["n_positives"] == \
            record["test"][channel]["n_positives"]


def test_checkpoint_can_be_re_scored_under_a_harder_protocol(processed_dataset, tmp_path):
    record = train_link_prediction(processed_dataset, _config("b6", tmp_path,
                                                              epochs=1))
    checkpoint = Path(tmp_path) / record["run_name"] / "model.pt"
    evaluated = evaluate_checkpoint(processed_dataset, checkpoint, split="test",
                                    device="cpu",
                                    protocol="dual_view_strict_holdout",
                                    edge_removal_mode="supervision_aware",
                                    edge_drop_fraction=0.5)
    assert evaluated["protocol"] == "dual_view_strict_holdout"
    assert evaluated["edge_removal_mode"] == "supervision_aware"
    assert evaluated["changed"] is True
    assert evaluated["metrics"]["A"]["n_positives"] > 0


def test_unknown_audit_mode_is_rejected(processed_dataset, tmp_path):
    with pytest.raises(ValueError, match="audit_mode"):
        train_link_prediction(processed_dataset, _config("b1", tmp_path),
                              audit_mode="sometimes")
