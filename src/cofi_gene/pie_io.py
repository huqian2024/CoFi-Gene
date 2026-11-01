"""PIE differential-expression table access and label construction.

Labels are built by exactly one rule for every model (B0-B7). The rule is a
pending team decision (D-012): the default here is ``p_adj < 0.05`` with an
optional ``|log2_fold_change|`` floor that is *off* unless the caller sets it.

Nothing in this module may look at a downstream test gene's response labels
during upstream training; callers receive plain arrays and are responsible for
grouping by perturbation target (``cofi_gene.splits.split_perturbation_targets``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "DE_COLUMNS",
    "DEFAULT_P_ADJ_THRESHOLD",
    "load_de_table",
    "inspect_de_parquet",
    "de_labels",
    "label_summary",
    "unique_targets",
]

DE_COLUMNS = ("target", "feature", "ref_mean", "log2_fold_change", "p_adj")
DEFAULT_P_ADJ_THRESHOLD = 0.05


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401 - availability probe
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "PIE parquet support needs the optional data extra: pip install -e '.[data]'"
        ) from exc


def load_de_table(path: str | Path, context: str | None = None,
                  max_rows: int | None = None) -> dict:
    """Load one ``de/<cell_line>.parquet`` into column arrays.

    ``context`` (usually the cell line) is attached as a constant column so the
    downstream model can receive a cell-context feature without a second join.
    """
    import pyarrow.parquet as pq

    _require_pyarrow()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    present = set(parquet.schema_arrow.names)
    missing = set(DE_COLUMNS) - present
    if missing:
        raise ValueError(f"{path} is missing PIE DE columns: {sorted(missing)}")
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        table = parquet.read_row_group(0).slice(0, max_rows) if parquet.num_row_groups else None
        if table is None:
            raise ValueError("parquet file has no row groups")
        data = table.to_pydict()
    else:
        data = parquet.read(columns=list(DE_COLUMNS)).to_pydict()
    out = {
        "target": np.asarray(data["target"], dtype=object),
        "feature": np.asarray(data["feature"], dtype=object),
        "ref_mean": np.asarray(data["ref_mean"], dtype=np.float32),
        "log2_fold_change": np.asarray(data["log2_fold_change"], dtype=np.float32),
        "p_adj": np.asarray(data["p_adj"], dtype=np.float32),
    }
    if context is not None:
        out["context"] = np.asarray([context] * out["p_adj"].size, dtype=object)
    return out


def inspect_de_parquet(path: str | Path, max_rows: int = 20000) -> dict:
    """Cheap schema + prevalence probe; explicitly not a global audit."""
    import pyarrow.parquet as pq

    _require_pyarrow()
    path = Path(path)
    parquet = pq.ParquetFile(path)
    present = set(parquet.schema_arrow.names)
    missing = set(DE_COLUMNS) - present
    if missing:
        raise ValueError(f"{path} is missing PIE DE columns: {sorted(missing)}")
    columns = sorted(DE_COLUMNS)
    sampled = 0
    nulls = {column: 0 for column in columns}
    targets: set[str] = set()
    features: set[str] = set()
    positives = 0
    for batch in parquet.iter_batches(batch_size=min(10000, max_rows), columns=columns):
        if sampled >= max_rows:
            break
        data = batch.to_pydict()
        length = min(len(data[columns[0]]), max_rows - sampled)
        for column in columns:
            nulls[column] += sum(value is None for value in data[column][:length])
        targets.update(str(v) for v in data["target"][:length] if v is not None)
        features.update(str(v) for v in data["feature"][:length] if v is not None)
        positives += sum(v is not None and v < DEFAULT_P_ADJ_THRESHOLD
                         for v in data["p_adj"][:length])
        sampled += length
    return {
        "file": str(path),
        "row_count_metadata": parquet.metadata.num_rows,
        "columns": parquet.schema_arrow.names,
        "sampled_rows": sampled,
        "sample_targets": len(targets),
        "sample_features": len(features),
        "missing_in_sample": nulls,
        "sample_positive_fraction_padj_lt_005": (positives / sampled) if sampled else None,
        "caution": ("Samples may be ordered by perturbation; not a global prevalence "
                    "estimate. Label rule is not final (D-012)."),
    }


def de_labels(table: dict, p_adj_threshold: float = DEFAULT_P_ADJ_THRESHOLD,
              min_abs_log2fc: float | None = None) -> np.ndarray:
    """Boolean DE label per row, using one rule for all models."""
    if not 0 < p_adj_threshold <= 1:
        raise ValueError("p_adj_threshold must be in (0, 1]")
    p_adj = np.asarray(table["p_adj"], dtype=np.float64)
    labels = p_adj < p_adj_threshold
    if min_abs_log2fc is not None:
        if min_abs_log2fc < 0:
            raise ValueError("min_abs_log2fc must be >= 0")
        labels &= np.abs(np.asarray(table["log2_fold_change"], dtype=np.float64)) >= min_abs_log2fc
    labels &= np.isfinite(p_adj)
    return labels


def label_summary(labels) -> dict:
    labels = np.asarray(labels, dtype=bool)
    return {
        "n_rows": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else float("nan"),
    }


def unique_targets(table: dict) -> tuple[str, ...]:
    """Sorted unique perturbation targets (the downstream grouping key)."""
    return tuple(sorted({str(t) for t in table["target"] if t is not None}))
