"""Week 7 CLI contracts: aggregate_seeds, make_figures and export_feedback_case."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate_cli = _load("aggregate_seeds_cli", "aggregate_seeds.py")
figures_cli = _load("make_figures_cli", "make_figures.py")
case_cli = _load("export_feedback_case_cli", "export_feedback_case.py")


def _run(root: Path, model: str, seed: int, ap: float, best_epoch: int = 10,
         cap: int = 200) -> None:
    run_dir = root / f"{model}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_dir.name,
        "model": {"name": model},
        "config": {"runtime": {"seed": seed}, "optim": {"epochs": cap}},
        "dataset": {"dataset_sha256": "sha"},
        "protocol": {"visibility_protocol": "cross_view_transfer",
                     "edge_removal_mode": "adjacency", "edge_drop_fraction": 0.0},
        "best_epoch": best_epoch,
        "test": {"A": {"average_precision": ap, "roc_auc": ap - 0.05}},
        "status": "complete",
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_cli_writes_json_and_markdown(tmp_path):
    for seed, ap in ((42, 0.90), (43, 0.92), (44, 0.94)):
        _run(tmp_path / "runs", "b1", seed, ap)
    out = tmp_path / "aggregate"
    assert aggregate_cli.main(["--runs", str(tmp_path / "runs"),
                               "--out-dir", str(out)]) == 0
    summary = json.loads((out / "seed_aggregate.json").read_text(encoding="utf-8"))
    assert summary["models"]["b1"]["A"]["n"] == 3
    markdown = (out / "seed_aggregate.md").read_text(encoding="utf-8")
    assert "mean ± std" in markdown and "0.9200" in markdown


def test_aggregate_cli_reports_a_missing_directory(tmp_path, capsys):
    code = aggregate_cli.main(["--runs", str(tmp_path / "nope"),
                               "--out-dir", str(tmp_path / "out")])
    assert code == 2
    assert "no metrics.json" in capsys.readouterr().err


def _aggregate_payload_with_warning(tmp_path: Path) -> Path:
    payload = {
        "metric": "average_precision",
        "models": {"b0": {"A": {"mean": 0.8, "std": 0.01, "n": 3}},
                   "b1": {"A": {"mean": 0.9, "std": 0.02, "n": 3}}},
        "warnings": ["the table mixes several dataset hashes"],
    }
    path = tmp_path / "agg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_make_figures_draws_upstream_bars(tmp_path, capsys):
    payload = {
        "metric": "average_precision",
        "models": {"b0": {"A": {"mean": 0.80, "std": 0.01},
                          "B": {"mean": 0.70, "std": 0.02}},
                   "b1": {"A": {"mean": 0.90, "std": 0.02},
                          "B": {"mean": 0.75, "std": 0.03}}},
    }
    src = tmp_path / "agg.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "fig1.svg"
    assert figures_cli.main(["--kind", "upstream", "--input", str(src),
                             "--out", str(out)]) == 0
    svg = out.read_text(encoding="utf-8")
    assert "channel A" in svg and "channel B" in svg and "b0" in svg
    captured = capsys.readouterr().out.splitlines()
    printed = "\n".join(line for line in captured if not line.startswith("[figures]"))
    assert json.loads(printed)["format"] == "svg"


def test_make_figures_refuses_a_flagged_summary_unless_forced(tmp_path, capsys):
    src = _aggregate_payload_with_warning(tmp_path)
    out = tmp_path / "fig.svg"
    assert figures_cli.main(["--kind", "upstream", "--input", str(src),
                             "--out", str(out)]) == 1
    assert not out.exists()
    assert figures_cli.main(["--kind", "upstream", "--input", str(src),
                             "--out", str(out), "--allow-warnings"]) == 0
    assert out.exists()
    assert "warning" in capsys.readouterr().out.lower()
def test_make_figures_downstream_flat_table(tmp_path):
    payload = {"metric": "macro_target_ap",
               "models": {"b0": {"mean": 0.29, "std": 0.001, "n": 3},
                          "b7": {"mean": 0.28, "std": 0.002, "n": 3}}}
    src = tmp_path / "down.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "down.svg"
    assert figures_cli.main(["--kind", "downstream", "--input", str(src),
                             "--out", str(out)]) == 0
    svg = out.read_text(encoding="utf-8")
    assert "macro_target_ap" in svg and "b7" in svg


def test_make_figures_robustness_from_stage_rows(tmp_path):
    payload = {"rows": [
        {"model": "b1", "ratio": 0.0, "channels": {"A": 0.96}, "status": "complete"},
        {"model": "b1", "ratio": 0.5, "channels": {"A": 0.94}, "status": "complete"},
        {"model": "b5", "ratio": 0.0, "channels": {"A": 0.95}, "status": "complete"},
        {"model": "b5", "ratio": 0.5, "channels": {"A": 0.93}, "status": "complete"},
    ]}
    src = tmp_path / "rob.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "rob.svg"
    assert figures_cli.main(["--kind", "robustness", "--input", str(src),
                             "--channel", "A", "--out", str(out)]) == 0
    svg = out.read_text(encoding="utf-8")
    assert "robustness" in svg.lower() and "missing-edge ratio" in svg


def test_make_figures_robustness_requires_two_shared_ratios(tmp_path):
    payload = {"rows": [
        {"model": "b1", "ratio": 0.0, "channels": {"A": 0.96}, "status": "complete"},
        {"model": "b5", "ratio": 0.5, "channels": {"A": 0.95}, "status": "complete"},
    ]}
    src = tmp_path / "rob.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than two ratios"):
        figures_cli.robustness_series(payload, channel="A")


def test_make_figures_draws_the_feedback_case(tmp_path):
    case = {"channel": "A",
            "summary": {"n_up": 1, "n_down": 1},
            "edges": [
                {"u_id": "G1", "v_id": "G2", "weight_before": 0.4,
                 "weight_after": 0.5, "delta": 0.1},
                {"u_id": "G2", "v_id": "G3", "weight_before": 0.4,
                 "weight_after": 0.3, "delta": -0.1}]}
    src = tmp_path / "case.json"
    src.write_text(json.dumps({"case": case}), encoding="utf-8")
    out = tmp_path / "case.svg"
    assert figures_cli.main(["--kind", "feedback", "--input", str(src),
                             "--out", str(out)]) == 0
    svg = out.read_text(encoding="utf-8")
    assert "G1" in svg and "up-weighted" in svg


def test_export_feedback_case_writes_a_case_json(processed_dataset, processed_dir,
                                                 tmp_path):
    pytest.importorskip("torch")
    from cofi_gene.config import load_config
    from cofi_gene.train import train_link_prediction

    config = load_config(REPO / "configs" / "b7.yaml", {
        "optim": {"epochs": 1, "eval_every": 1, "early_stop_patience": 1},
        "runtime": {"device": "cpu", "out_dir": str(tmp_path / "runs"), "seed": 7}})
    record = train_link_prediction(processed_dataset, config, evaluate_test=False)
    checkpoint = tmp_path / "runs" / record["run_name"] / "model.pt"
    out = tmp_path / "case.json"
    code = case_cli.main(["--checkpoint", str(checkpoint),
                          "--dataset", str(processed_dir), "--channel", "A",
                          "--top-k", "4", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model"] == "b7"
    assert payload["case"]["channel"] == "A"
    assert len(payload["case"]["edges"]) == 4
    assert payload["protocol"]["visibility_protocol"] == "cross_view_transfer"
    assert payload["dataset"]["sha256"]
    assert payload["case"]["summary"]["num_selected_edges"] == 4



def test_make_figures_allows_a_budget_only_warning(tmp_path):
    payload = {
        "metric": "average_precision",
        "models": {"b1": {"A": {"mean": 0.9, "std": 0.01}}},
        "warnings": ["6/6 run(s) selected their best epoch at the epoch cap"],
        "integrity": {"blocking": False},
    }
    src = tmp_path / "agg.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "fig.svg"
    assert figures_cli.main(["--kind", "upstream", "--input", str(src),
                             "--out", str(out)]) == 0
    assert out.exists()


def test_upstream_bars_allow_a_single_graph_model(tmp_path):
    payload = {"metric": "average_precision",
               "models": {"b0": {"A": {"mean": 0.8, "std": 0.01},
                                  "B": {"mean": 0.7, "std": 0.01}},
                          "b1": {"A": {"mean": 0.9, "std": 0.02}}}}
    categories, groups, _ = figures_cli.upstream_bars(payload)
    assert categories == ["b0", "b1"]
    by_label = {group["label"]: group for group in groups}
    assert by_label["channel B"]["means"] == [0.7, None]
    src = tmp_path / "agg.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "fig.svg"
    assert figures_cli.main(["--kind", "upstream", "--input", str(src),
                             "--out", str(out)]) == 0


def test_week7_modules_import_on_a_torch_less_job():
    """CI's numpy-only job collects these tests, so nothing may import torch at
    module scope (regression guard: export_feedback_case once did)."""
    code = textwrap.dedent(
        """
        import builtins, importlib.util
        real = builtins.__import__
        def fake(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch blocked for this check")
            return real(name, *args, **kwargs)
        builtins.__import__ = fake
        import cofi_gene.aggregate, cofi_gene.feedback_report, cofi_gene.figures  # noqa
        for path in ("scripts/export_feedback_case.py", "scripts/make_figures.py",
                     "scripts/aggregate_seeds.py"):
            spec = importlib.util.spec_from_file_location("m", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        print("ok")
        """)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
