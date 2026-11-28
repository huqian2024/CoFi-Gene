"""M5.1 stage runner: plan expansion, completion detection, resume and CLI smoke."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_experiments.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_script()


def test_plan_expands_models_ratios_and_seeds_with_self_describing_names():
    cells = runner.plan(1)
    assert len(cells) == len(runner.ALL_MODELS)
    assert {cell["model"] for cell in cells} == set(runner.ALL_MODELS)
    assert cells[0]["run_name"] == "s1_b0_xfer_drop00_adj_seed42"
    assert cells[0]["protocol"] == "cross_view_transfer"
    assert cells[0]["mode"] == "adjacency"

    stage4 = runner.plan(4)
    assert len(stage4) == 6 * 2
    assert {cell["ratio"] for cell in stage4} == {0.0, 0.5}
    assert all(cell["mode"] == "supervision_aware" for cell in stage4)

    stage5 = runner.plan(5, seeds=[43, 44])
    assert {cell["seed"] for cell in stage5} == {43, 44}
    assert len(stage5) == 8 * 2


def test_plan_rejects_out_of_scope_requests():
    with pytest.raises(ValueError, match="stage must be one of"):
        runner.plan(9)
    with pytest.raises(ValueError, match="unknown model"):
        runner.plan(1, models=["b9"])
    with pytest.raises(ValueError, match="only runs"):
        runner.plan(2, models=["b0"])
    with pytest.raises(ValueError, match="only runs ratios"):
        runner.plan(4, ratios=[0.3])
    with pytest.raises(ValueError, match="ratio must be in"):
        runner.plan(1, ratios=[1.5])


def test_stage0_plan_is_a_single_audit_cell():
    cells = runner.plan(0)
    assert len(cells) == 1 and cells[0]["kind"] == "audit"
    assert cells[0]["run_name"] == "s0_audit"


def test_completion_requires_a_parsable_complete_metrics_file(tmp_path):
    run_dir = tmp_path / "cell"
    assert runner.is_complete(run_dir) is False
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{not json", encoding="utf-8")
    assert runner.is_complete(run_dir) is False
    (run_dir / "metrics.json").write_text(json.dumps({"status": "running"}),
                                          encoding="utf-8")
    assert runner.is_complete(run_dir) is False
    (run_dir / "metrics.json").write_text(json.dumps({"status": "complete"}),
                                          encoding="utf-8")
    assert runner.is_complete(run_dir) is True


def test_summarise_reports_mean_and_sample_std():
    rows = [
        {"model": "b1", "status": "complete", "channels": {"A": 0.8}, "auroc": {}},
        {"model": "b1", "status": "complete", "channels": {"A": 0.9}, "auroc": {}},
        {"model": "b2", "status": "missing", "channels": {"A": 0.1}, "auroc": {}},
    ]
    summary = runner.summarise(rows)
    assert summary["b1"]["A"]["n"] == 2
    assert summary["b1"]["A"]["mean_ap"] == pytest.approx(0.85)
    assert summary["b1"]["A"]["std_ap"] == pytest.approx(0.0707, abs=1e-3)
    assert "b2" not in summary                       # incomplete cells never count
    assert "M5.1 stage 5" in runner.render_markdown(summary, 5, rows)


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "stage1"
    code = runner.main(["--stage", "1", "--dataset", str(tmp_path / "unused"),
                        "--out-dir", str(out), "--only-models", "b3", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [cell["run_name"] for cell in payload["cells"]] == [
        "s1_b3_xfer_drop00_adj_seed42"]
    assert not out.exists()


def test_stage0_audits_a_real_build_and_writes_the_report(processed_dir, tmp_path):
    out = tmp_path / "stage0"
    code = runner.main(["--stage", "0", "--dataset", str(processed_dir),
                        "--out-dir", str(out)])
    assert code == 0
    report = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert (out / "audit.md").exists()


def test_stage1_trains_one_cell_then_resume_skips_it_and_stage3_reuses_it(
        processed_dir, tmp_path):
    pytest.importorskip("torch")
    stage1 = tmp_path / "stage1"
    args = ["--stage", "1", "--dataset", str(processed_dir),
            "--only-models", "b1", "--epochs", "1", "--out-dir", str(stage1)]
    assert runner.main(args) == 0
    run_dir = stage1 / "s1_b1_xfer_drop00_adj_seed42"
    record = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["protocol"]["visibility_protocol"] == "cross_view_transfer"
    assert record["audit"]["status"] == "PASS"
    assert (run_dir / "audit.json").exists()
    assert (run_dir / "model.pt").exists()

    # A second run with --resume must not retrain the cell.
    assert runner.main(args + ["--resume"]) == 0

    # Stage 3 scores the frozen checkpoint without training.
    stage3 = tmp_path / "stage3"
    assert runner.main(["--stage", "3", "--dataset", str(processed_dir),
                        "--only-models", "b1", "--frozen-from", str(stage1),
                        "--out-dir", str(stage3)]) == 0
    evaluated = json.loads((stage3 / "s3_b1_xfer_drop00_adj_seed42"
                            / "eval.json").read_text(encoding="utf-8"))
    assert evaluated["model"] == "b1"
    assert evaluated["metrics"]["A"]["n_positives"] > 0
    assert 0.0 <= evaluated["metrics"]["A"]["average_precision"] <= 1.0
    assert evaluated["visibility"]["protocol"] == "cross_view_transfer"


def test_stage3_without_a_checkpoint_fails_but_keeps_the_summary(processed_dir, tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "stage3"
    code = runner.main(["--stage", "3", "--dataset", str(processed_dir),
                        "--only-models", "b2", "--frozen-from", str(tmp_path / "nope"),
                        "--out-dir", str(out)])
    assert code == 1
    summary = json.loads((out / "stage3_summary.json").read_text(encoding="utf-8"))
    assert summary["summary"] == {}
    assert any(row["status"] == "missing" for row in summary["rows"])

