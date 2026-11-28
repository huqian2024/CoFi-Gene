"""Week 7 aggregation: mean/std, paired deltas, provenance and budget saturation."""

from __future__ import annotations

import json

import pytest

from cofi_gene.aggregate import aggregate, load_records, render_markdown, write_summary


def _run(root, model, seed, ap, auroc=None, best_epoch=5, cap=200,
         status="complete", protocol="cross_view_transfer", mode="adjacency",
         ratio=0.0, sha="abc123", run_name=None):
    run_dir = root / (run_name or f"{model}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_dir.name,
        "model": {"name": model},
        "config": {"runtime": {"seed": seed}, "optim": {"epochs": cap}},
        "dataset": {"dataset_sha256": sha},
        "protocol": {"visibility_protocol": protocol, "edge_removal_mode": mode,
                     "edge_drop_fraction": ratio},
        "best_epoch": best_epoch,
        "test": {"A": {"average_precision": ap,
                       "roc_auc": auroc if auroc is not None else ap - 0.05}},
    }
    if status is not None:
        payload["status"] = status
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_mean_and_sample_std_across_seeds(tmp_path):
    for seed, ap in ((42, 0.90), (43, 0.92), (44, 0.94)):
        _run(tmp_path, "b1", seed, ap)
    records, skipped = load_records(tmp_path)
    assert skipped == [] and len(records) == 3
    summary = aggregate(records)
    entry = summary["models"]["b1"]["A"]
    assert entry["n"] == 3
    assert entry["mean"] == pytest.approx(0.92)
    assert entry["std"] == pytest.approx(0.02)          # sample std of 0.90/0.92/0.94
    assert entry["values"] == {"42": 0.90, "43": 0.92, "44": 0.94}
    assert summary["runs"]["seeds"] == [42, 43, 44]


def test_paired_deltas_use_the_same_seed_and_flag_sign_stability(tmp_path):
    for seed in (42, 43, 44):
        _run(tmp_path, "b0", seed, 0.80)
        _run(tmp_path, "b1", seed, 0.90 + 0.001 * (seed - 42))
    summary = aggregate(load_records(tmp_path)[0])
    delta = summary["comparisons"]["b1-b0"]["A"]
    assert delta["n"] == 3
    assert delta["mean"] == pytest.approx(0.101)
    assert delta["std"] == pytest.approx(0.001)
    assert delta["consistent_sign"] is True
    assert delta["per_seed"] == {"42": pytest.approx(0.1),
                                "43": pytest.approx(0.101),
                                "44": pytest.approx(0.102)}


def test_sign_flip_is_reported_as_unstable(tmp_path):
    for seed, first, second in ((42, 0.90, 0.80), (43, 0.80, 0.90), (44, 0.85, 0.85)):
        _run(tmp_path, "b4", seed, first)
        _run(tmp_path, "b5", seed, second)
    summary = aggregate(load_records(tmp_path)[0])
    delta = summary["comparisons"]["b5-b4"]["A"]
    assert delta["consistent_sign"] is False
    assert "sign flips" in render_markdown(summary)


def test_budget_saturation_is_counted_and_warned(tmp_path):
    _run(tmp_path, "b3", 42, 0.9, best_epoch=200, cap=200)
    _run(tmp_path, "b3", 43, 0.9, best_epoch=150, cap=200)
    summary = aggregate(load_records(tmp_path)[0])
    assert summary["budget"]["n_saturated"] == 1
    assert summary["budget"]["saturated"][0]["best_epoch"] == 200
    assert any("epoch cap" in warning for warning in summary["warnings"])


def test_incomplete_and_legacy_runs_are_handled_explicitly(tmp_path):
    _run(tmp_path, "b1", 42, 0.9, status="complete")
    _run(tmp_path, "b1", 43, 0.9, status=None)          # legacy file, no status
    _run(tmp_path, "b1", 44, 0.9, status="running")     # incomplete
    records = load_records(tmp_path)[0]
    strict = aggregate(records)
    assert strict["models"]["b1"]["A"]["n"] == 2         # complete + legacy
    assert any("without status=complete" in text for text in strict["warnings"])
    loose = aggregate(records, require_complete=False)
    assert loose["models"]["b1"]["A"]["n"] == 3


def test_mixed_provenance_and_too_few_seeds_are_warned(tmp_path):
    _run(tmp_path, "b1", 42, 0.9, sha="aaa")
    _run(tmp_path, "b1", 43, 0.9, sha="bbb", protocol="dual_view_strict_holdout")
    summary = aggregate(load_records(tmp_path)[0])
    joined = " ".join(summary["warnings"])
    assert "dataset hashes" in joined
    assert "visibility protocols" in joined
    assert "seed(s) (< 3)" in joined
    assert len(summary["provenance"]["datasets"]) == 2


def test_duplicate_model_seed_is_ignored_after_the_first(tmp_path):
    _run(tmp_path, "b2", 42, 0.90, run_name="b2_seed42_a")
    _run(tmp_path, "b2", 42, 0.95, run_name="b2_seed42_b")  # same model+seed
    summary = aggregate(load_records(tmp_path)[0])
    assert summary["models"]["b2"]["A"]["n"] == 1
    assert any("duplicate" in text for text in summary["warnings"])


def test_missing_or_broken_files_are_skipped_not_counted(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    _, skipped = load_records(empty)
    assert skipped and "no metrics.json" in skipped[0]["reason"]
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{oops", encoding="utf-8")
    records, skipped = load_records(broken)
    assert records == [] and "JSONDecodeError" in skipped[0]["reason"]


def test_summary_is_json_serialisable_and_written_both_ways(tmp_path):
    _run(tmp_path, "b1", 42, 0.9)
    summary = aggregate(load_records(tmp_path)[0])
    paths = write_summary(summary, tmp_path / "out")
    assert json.loads(open(paths["json"], encoding="utf-8").read())["metric"] == \
        "average_precision"
    markdown = open(paths["markdown"], encoding="utf-8").read()
    assert "mean ± std" in markdown and "saturated runs" in markdown


def test_metric_can_switch_to_auc(tmp_path):
    _run(tmp_path, "b1", 42, 0.90, auroc=0.70)
    _run(tmp_path, "b1", 43, 0.92, auroc=0.80)
    summary = aggregate(load_records(tmp_path)[0], metric="roc_auc")
    assert summary["metric"] == "roc_auc"
    assert summary["models"]["b1"]["A"]["mean"] == pytest.approx(0.75)


def test_integrity_flag_separates_budget_caveats_from_mixed_tables(tmp_path):
    """A binding budget is a caveat; a mixed dataset/protocol table is a blocker."""
    _run(tmp_path / "a", "b1", 42, 0.9, best_epoch=200, cap=200)
    budget_only = aggregate(load_records(tmp_path / "a")[0])
    assert budget_only["integrity"]["blocking"] is False
    assert budget_only["warnings"]

    _run(tmp_path / "b", "b1", 42, 0.9, sha="aaa")
    _run(tmp_path / "b", "b1", 43, 0.9, sha="bbb")
    mixed = aggregate(load_records(tmp_path / "b")[0])
    assert mixed["integrity"]["mixed_datasets"] is True
    assert mixed["integrity"]["blocking"] is True
