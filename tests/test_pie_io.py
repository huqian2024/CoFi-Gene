"""PIE DE table access and label rules (needs the optional pyarrow extra)."""

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from cofi_gene.pie_io import (DE_COLUMNS, de_labels, inspect_de_parquet, label_summary,  # noqa: E402
                              load_de_table, unique_targets)


@pytest.fixture()
def de_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({
        "target": ["TP53", "TP53", "TP53", "EGFR", "EGFR"],
        "feature": ["MDM2", "CDKN1A", "GAPDH", "STAT3", "ACTB"],
        "ref_mean": [4.0, 3.0, 9.0, 5.0, 8.0],
        "log2_fold_change": [2.0, 1.5, 0.01, -1.2, 0.02],
        "p_adj": [0.001, 0.02, 0.9, 0.03, 0.5],
    })
    path = tmp_path / "K562.parquet"
    pq.write_table(table, path)
    return path


def test_load_de_table_attaches_context_and_dtypes(de_parquet):
    table = load_de_table(de_parquet, context="K562")
    assert set(DE_COLUMNS).issubset(table)
    assert table["target"].tolist()[:2] == ["TP53", "TP53"]
    assert table["p_adj"].dtype == np.float32
    assert set(table["context"]) == {"K562"}
    assert unique_targets(table) == ("EGFR", "TP53")
    limited = load_de_table(de_parquet, max_rows=2)
    assert limited["p_adj"].size == 2


def test_label_rule_p_adj_only_and_with_effect_size_floor(de_parquet):
    table = load_de_table(de_parquet)
    labels = de_labels(table)
    assert labels.tolist() == [True, True, False, True, False]
    strict = de_labels(table, p_adj_threshold=0.05, min_abs_log2fc=1.0)
    assert strict.tolist() == [True, True, False, True, False]
    very_strict = de_labels(table, p_adj_threshold=0.05, min_abs_log2fc=1.8)
    assert very_strict.tolist() == [True, False, False, False, False]
    summary = label_summary(labels)
    assert summary["n_positive"] == 3 and summary["positive_rate"] == pytest.approx(0.6)


def test_invalid_label_arguments(de_parquet):
    table = load_de_table(de_parquet)
    with pytest.raises(ValueError, match="p_adj_threshold"):
        de_labels(table, p_adj_threshold=0.0)
    with pytest.raises(ValueError, match="min_abs_log2fc"):
        de_labels(table, min_abs_log2fc=-1.0)
    with pytest.raises(ValueError, match="max_rows"):
        load_de_table(de_parquet, max_rows=0)


def test_inspect_reports_schema_and_sample_prevalence(de_parquet):
    report = inspect_de_parquet(de_parquet, max_rows=10)
    assert report["row_count_metadata"] == 5
    assert report["sampled_rows"] == 5
    assert report["sample_targets"] == 2
    assert report["sample_positive_fraction_padj_lt_005"] == pytest.approx(0.6)
    assert "not final" in report["caution"]


def test_missing_columns_raise(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "broken.parquet"
    pq.write_table(pa.table({"target": ["A"], "feature": ["B"]}), path)
    with pytest.raises(ValueError, match="missing PIE DE columns"):
        load_de_table(path)
    with pytest.raises(ValueError, match="missing PIE DE columns"):
        inspect_de_parquet(path)
