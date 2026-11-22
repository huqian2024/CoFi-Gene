"""Week 5 (M5) sweep driver contract: plan, summary tables, dry-run and one cell."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_robustness.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_robustness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


robustness = _load_script()


def _fake_record(model: str, ratio: float, ap: float) -> dict:
    return {
        "model": model,
        "edge_drop_fraction": ratio,
        "run_name": f"{model}_drop{int(round(ratio * 100)):02d}",
        "best_epoch": 3,
        "best_val_mean_ap": ap - 0.01,
        "test": {"A": {"average_precision": ap, "roc_auc": ap - 0.05}},
        "perturbation": {"edge_drop_fraction": ratio, "mask_other_view": False},
    }


# ------------------------------------------------------------------- plan
def test_plan_runs_builds_one_cell_per_model_and_ratio():
    plan = robustness.plan_runs(["b1", "b5"], [0.0, 0.1], seed=42)
    assert [(item["model"], item["edge_drop_fraction"]) for item in plan] == [
        ("b1", 0.0), ("b1", 0.1), ("b5", 0.0), ("b5", 0.1)]
    assert [item["run_name"] for item in plan] == [
        "b1_drop00_seed42", "b1_drop10_seed42", "b5_drop00_seed42", "b5_drop10_seed42"]
    masked = robustness.plan_runs(["b5"], [0.5], seed=7, mask_other_view=True,
                                  misalign_cross_view=True)
    assert masked[0]["run_name"] == "b5_drop50_maskother_misalign_seed7"


def test_plan_runs_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown model"):
        robustness.plan_runs(["b9"], [0.0], seed=42)
    with pytest.raises(ValueError, match="at least one model"):
        robustness.plan_runs([], [0.0], seed=42)
    with pytest.raises(ValueError, match="ratio must be in"):
        robustness.plan_runs(["b1"], [1.0], seed=42)
    with pytest.raises(ValueError, match="cross-view model"):
        robustness.plan_runs(["b1", "b2"], [0.0], seed=42, misalign_cross_view=True)


# ---------------------------------------------------------------- summary
def test_summarize_builds_tables_and_markdown():
    results = [_fake_record("b1", 0.0, 0.91), _fake_record("b1", 0.1, 0.81)]
    settings = {"seed": 42, "ratios": [0.0, 0.1], "models": ["b1"],
                "mask_other_view": False, "misalign_cross_view": False,
                "device": "cpu", "epochs_override": None}
    summary = robustness.summarize(results, settings, {"processed_dir": "toy",
                                                       "sha256": "abc"})
    table = summary["table"]["average_precision"]["A"]["b1"]
    assert table == {"0.00": 0.91, "0.10": 0.81}
    assert summary["table"]["roc_auc"]["A"]["b1"]["0.10"] == pytest.approx(0.76)
    markdown = robustness.render_markdown(summary)
    assert "## AUPRC: channel A" in markdown
    assert "| b1 | 0.9100 | 0.8100 |" in markdown
    assert "sha256 abc" in markdown


def test_render_markdown_writes_an_em_dash_for_missing_cells():
    results = [_fake_record("b1", 0.0, 0.9)]
    settings = {"seed": 1, "ratios": [0.0, 0.3], "models": ["b1"],
                "mask_other_view": False, "misalign_cross_view": False,
                "device": "cpu", "epochs_override": None}
    summary = robustness.summarize(results, settings, {"processed_dir": "toy",
                                                       "sha256": None})
    assert "| b1 | 0.9000 | — |" in robustness.render_markdown(summary)


# -------------------------------------------------------------------- CLI
def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "sweep"
    code = robustness.main(["--dataset", str(tmp_path / "unused"),
                            "--models", "b3", "--ratios", "0", "0.5",
                            "--out-dir", str(out), "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["edge_drop_fraction"] for item in payload["plan"]] == [0.0, 0.5]
    assert not out.exists()


def test_sweep_writes_summary_and_run_directories(processed_dir, tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "sweep"
    code = robustness.main(["--dataset", str(processed_dir), "--models", "b1",
                            "--ratios", "0", "0.5", "--epochs", "1",
                            "--out-dir", str(out), "--summary-name", "week5"])
    assert code == 0
    summary = json.loads((out / "week5.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["num_nodes"] > 0
    assert summary["dataset"]["sha256"]
    assert summary["settings"]["ratios"] == [0.0, 0.5]
    table = summary["table"]["average_precision"]["A"]["b1"]
    assert set(table) == {"0.00", "0.50"}
    assert len(summary["runs"]) == 2
    for entry in summary["runs"]:
        assert (out / entry["run_name"] / "metrics.json").exists()
        if entry["edge_drop_fraction"] > 0.0:
            # A zero-fraction cell is an unperturbed run, so it records no block.
            assert (entry["perturbation"]["edge_drop_fraction"]
                    == entry["edge_drop_fraction"])
        else:
            assert entry["perturbation"] is None
    markdown = (out / "week5.md").read_text(encoding="utf-8")
    assert "## AUPRC: channel A" in markdown
