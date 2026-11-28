"""M5.1 CLI contracts: ``scripts/audit_dataset.py`` and ``scripts/run_baselines.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_cli = _load("audit_dataset_cli", "audit_dataset.py")
baselines_cli = _load("run_baselines_cli", "run_baselines.py")


def test_audit_cli_writes_json_and_markdown_and_exits_zero(processed_dir, tmp_path):
    out = tmp_path / "audit.json"
    code = audit_cli.main(["--dataset", str(processed_dir), "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert out.with_suffix(".md").exists()


def test_audit_cli_fails_loudly_on_a_missing_dataset(tmp_path, capsys):
    code = audit_cli.main(["--dataset", str(tmp_path / "nope")])
    assert code == 2
    assert "cannot load" in capsys.readouterr().err


def test_audit_cli_json_mode_prints_the_full_report(processed_dir, capsys):
    assert audit_cli.main(["--dataset", str(processed_dir), "--json",
                           "--no-roundtrip"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert "checks" in payload and "visibility" in payload


def test_baselines_cli_dry_run_prints_settings_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "b"
    code = baselines_cli.main(["--dataset", str(tmp_path / "unused"),
                               "--out-dir", str(out), "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["visibility_protocol"] == "cross_view_transfer"
    assert not out.exists()


def test_baselines_cli_writes_json_and_markdown(processed_dir, tmp_path):
    out = tmp_path / "baselines"
    code = baselines_cli.main(["--dataset", str(processed_dir), "--out-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    for channel in ("A", "B"):
        for name in baselines_cli.BASELINE_NAMES:
            entry = summary["results"][channel][name]
            assert 0.0 <= entry["average_precision"] <= 1.0
            assert entry["graph"] == "encode_edges"
            assert entry["n_positives"] > 0
    markdown = (out / "baselines.md").read_text(encoding="utf-8")
    assert "| common_neighbors |" in markdown
    assert "positive rate" in markdown


def test_baselines_cli_honours_the_protocol_and_edge_removal(processed_dir, tmp_path):
    out = tmp_path / "strict"
    code = baselines_cli.main([
        "--dataset", str(processed_dir), "--out-dir", str(out),
        "--protocol", "dual_view_strict_holdout",
        "--edge-removal-mode", "supervision_aware",
        "--edge-drop-fraction", "0.5", "--split", "val"])
    assert code == 0
    summary = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    assert summary["settings"]["visibility_protocol"] == "dual_view_strict_holdout"
    assert summary["settings"]["edge_removal_mode"] == "supervision_aware"
    assert summary["perturbation"]["changed"] is True
    assert summary["results"]["A"]["common_neighbors"]["n_positives"] > 0


def test_baselines_cli_rejects_an_unknown_baseline(processed_dir, tmp_path):
    with pytest.raises(SystemExit):
        baselines_cli.main(["--dataset", str(processed_dir),
                            "--baselines", "magic", "--out-dir", str(tmp_path)])


def test_data_template_parses_as_a_pipeline_config():
    """``configs/data.yaml`` must stay a valid PipelineConfig mapping after M5.1."""
    import yaml

    from cofi_gene.negative_sampling import NEGATIVE_STRATEGIES
    from cofi_gene.pipeline import PipelineConfig

    values = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    config = PipelineConfig(**values)
    assert config.negatives_strategy in NEGATIVE_STRATEGIES
    assert config.negatives_strategy == "uniform_random"      # historic default


def test_build_dataset_cli_accepts_the_new_negative_flags(synthetic_raw, tmp_path):
    """The degree-matched dataset must be buildable from the documented CLI."""
    import json as _json

    build_cli = _load("build_dataset_cli", "build_dataset.py")
    out = tmp_path / "deg"
    assert build_cli.main([
        "--string-links", str(synthetic_raw["string_links"]),
        "--string-info", str(synthetic_raw["string_info"]),
        "--gene-text", str(synthetic_raw["gene_text"]),
        "--gene-order", str(synthetic_raw["gene_order"]),
        "--n-components", "8", "--out", str(out),
        "--negatives-strategy", "degree_matched",
        "--negatives-fallback", "uniform",
        "--negative-degree-buckets", "5"]) == 0
    meta = _json.loads((out / "dataset.json").read_text(encoding="utf-8"))["meta"]
    recorded = meta["negative_sampling"]["A"]
    assert recorded["strategy"] == "degree_matched"
    assert recorded["degree_buckets"] == 5
    assert recorded["mutually_exclusive"] is True
    assert meta["split"]["negatives_strategy"] == "degree_matched"

