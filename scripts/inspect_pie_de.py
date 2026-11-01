#!/usr/bin/env python3
"""Inspect a small sample of published PIE DE parquet (NOT a full dataset audit)."""
import argparse
import json
from pathlib import Path

REQUIRED = {"target", "feature", "ref_mean", "log2_fold_change", "p_adj"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path, help="a local PIE de/<cell_line>.parquet file")
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args()
    if args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install the optional data dependencies: pip install -e '.[data]'") from exc
    parquet = pq.ParquetFile(args.parquet)
    present = set(parquet.schema_arrow.names)
    if missing := (REQUIRED - present):
        raise SystemExit(f"Missing PIE DE columns: {sorted(missing)}")
    required = sorted(REQUIRED)
    sampled = 0
    positives = 0
    nulls = {column: 0 for column in required}
    targets, features = set(), set()
    for batch in parquet.iter_batches(batch_size=min(10000, args.max_rows), columns=required):
        if sampled >= args.max_rows:
            break
        data = batch.to_pydict()
        length = min(len(next(iter(data.values()))), args.max_rows - sampled)
        for col in required:
            nulls[col] += sum(x is None for x in data[col][:length])
        targets.update(str(x) for x in data["target"][:length] if x is not None)
        features.update(str(x) for x in data["feature"][:length] if x is not None)
        positives += sum((x is not None and x < 0.05) for x in data["p_adj"][:length])
        sampled += length
    print(json.dumps({"file": str(args.parquet), "row_count_metadata": parquet.metadata.num_rows,
                      "columns": parquet.schema_arrow.names, "sampled_rows": sampled,
                      "sample_targets": len(targets), "sample_features": len(features),
                      "missing_in_sample": nulls,
                      "sample_positive_fraction_padj_lt_005": positives / sampled if sampled else None,
                      "caution": "Samples may be ordered by perturbation; not a global prevalence estimate. Label rule is not final."},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
