"""CLI contract tests: the documented commands must actually run.

The roadmap promises two reproducible entry points, so they are exercised as
subprocesses (exactly how a teammate would run them), not just imported.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("pyarrow")


def _run(args, expect_success=True):
    completed = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                               cwd=REPO, timeout=600)
    if expect_success and completed.returncode != 0:
        pytest.fail(f"command failed: {args}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed


def test_make_synthetic_then_build_dataset_via_cli(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _run(["scripts/make_synthetic_dataset.py", "--raw", str(raw),
          "--n-genes", "50", "--n-raw-features", "32", "--n-components", "8"])
    for expected in ("gene_text.npy", "gene_text.ids.txt", "synthetic_truth.json"):
        assert (raw / expected).exists(), expected

    _run(["scripts/build_dataset.py", "--dataset-name", "cli_smoke",
          "--string-links", str(next(raw.glob("string_*.links.detailed.*"))),
          "--string-info", str(next(raw.glob("string_*.protein.info.*"))),
          "--gene-text", str(raw / "gene_text.npy"),
          "--gene-order", str(raw / "gene_text.ids.txt"),
          "--n-components", "8", "--out", str(processed)])

    assert (processed / "dataset.npz").exists()
    assert (processed / "dataset.json").exists()
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "cli_smoke"
    assert manifest["processed"]["num_nodes"] == 50
    assert manifest["split_decisions"]["edge_split_seed"] == 42
    assert manifest["inputs"]["gene_text"]["sha256"]
    assert manifest["report"]["overlap"]["n_shared_edges"] > 0


def test_build_dataset_dry_run_writes_nothing(tmp_path):
    raw = tmp_path / "raw"
    _run(["scripts/make_synthetic_dataset.py", "--raw", str(raw),
          "--n-genes", "40", "--n-raw-features", "24"])
    out = tmp_path / "should_not_exist"
    completed = _run(["scripts/build_dataset.py", "--dry-run",
                      "--string-links", str(next(raw.glob("string_*.links.detailed.*"))),
                      "--string-info", str(next(raw.glob("string_*.protein.info.*"))),
                      "--gene-text", str(raw / "gene_text.npy"),
                      "--gene-order", str(raw / "gene_text.ids.txt"),
                      "--n-components", "6", "--out", str(out)])
    assert not out.exists()
    assert "genes" in json.loads(completed.stdout)


def test_build_dataset_rejects_unknown_config_key(tmp_path):
    raw = tmp_path / "raw"
    _run(["scripts/make_synthetic_dataset.py", "--raw", str(raw), "--n-genes", "30",
          "--n-raw-features", "16"])
    config = tmp_path / "bad.yaml"
    config.write_text("n_components: 4\nnot_a_real_key: 1\n", encoding="utf-8")
    completed = _run(["scripts/build_dataset.py", "--config", str(config), "--dry-run"],
                     expect_success=False)
    assert completed.returncode != 0
    assert "unknown config key" in (completed.stdout + completed.stderr)
