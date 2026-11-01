#!/usr/bin/env python3
"""Build a processed CoFi-Gene dataset from raw STRING + gene_text + PIE files.

Example (synthetic inputs, CPU only):

    python scripts/make_synthetic_dataset.py --raw data/raw/synthetic
    python scripts/build_dataset.py --config configs/data.synthetic.yaml

The script writes ``dataset.npz`` / ``dataset.json`` plus a ``manifest.json`` that
records input hashes, the full build report and the exact split decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.manifest import (git_commit, runtime_versions, sha256_array,  # noqa: E402
                                sha256_file, utc_timestamp, write_manifest)
from cofi_gene.pipeline import PipelineConfig, build_dataset, split_decisions  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML with PipelineConfig fields (CLI wins)")
    parser.add_argument("--dataset-name", help="name recorded in the dataset metadata")
    parser.add_argument("--string-links", help="STRING *.protein.links.detailed.*.txt[.gz]")
    parser.add_argument("--string-info", help="STRING *.protein.info.*.txt")
    parser.add_argument("--gene-text", help="gene_text matrix (.npy/.npz)")
    parser.add_argument("--gene-order", help="one gene id per line, aligned with gene_text")
    parser.add_argument("--pie-de", action="append", default=None, metavar="CONTEXT=PATH",
                        help="repeatable, e.g. --pie-de K562=data/raw/K562.parquet")
    parser.add_argument("--out", help="output directory for the processed dataset")
    parser.add_argument("--min-score-a", type=int)
    parser.add_argument("--min-score-b", type=int)
    parser.add_argument("--weight-mode", choices=["none", "minmax"])
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--p-adj-threshold", type=float)
    parser.add_argument("--min-abs-log2fc", type=float)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--negatives-ratio", type=float)
    parser.add_argument("--negative-seed", type=int)
    parser.add_argument("--mask-other-view", action="store_true", default=None)
    parser.add_argument("--max-link-rows", type=int)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--dry-run", action="store_true", help="build and report, do not write")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    values: dict = {}
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"{args.config} must contain a YAML mapping")
        values.update(loaded)
    if args.pie_de:
        tables = {}
        for item in args.pie_de:
            context, _, path = item.partition("=")
            if not path:
                raise SystemExit(f"--pie-de expects CONTEXT=PATH, got {item!r}")
            tables[context.strip()] = path.strip()
        values["de_tables"] = tables
    mapping = {
        "dataset_name": "dataset_name", "string_links": "string_links",
        "string_info": "string_info", "gene_text": "gene_text",
        "gene_order": "gene_order", "min_score_a": "min_score_a",
        "min_score_b": "min_score_b", "weight_mode": "weight_mode",
        "n_components": "n_components", "p_adj_threshold": "p_adj_threshold",
        "min_abs_log2fc": "min_abs_log2fc", "train_fraction": "train_fraction",
        "val_fraction": "val_fraction", "split_seed": "split_seed",
        "negatives_ratio": "negatives_ratio", "negative_seed": "negative_seed",
        "mask_other_view": "mask_other_view", "max_link_rows": "max_link_rows",
        "max_genes": "max_genes",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            values[field_name] = value
    allowed = {field.name for field in fields(PipelineConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise SystemExit(f"unknown config key(s): {sorted(unknown)}")
    return PipelineConfig(**values)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)
    dataset, report = build_dataset(config)

    print(json.dumps({"genes": dataset.num_nodes, "features": dataset.num_features,
                      "channels": report.channels, "overlap": report.overlap,
                      "splits": report.splits,
                      "downstream": report.downstream.get("n_targets_per_split"),
                      "warnings": report.warnings}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --dry-run is set")

    out_dir = Path(args.out)
    dataset.save(out_dir)
    inputs = {}
    for label, path in (("string_links", config.string_links),
                        ("string_info", config.string_info),
                        ("gene_text", config.gene_text),
                        ("gene_order", config.gene_order)):
        if path and Path(path).exists():
            inputs[label] = {"path": str(path), "sha256": sha256_file(path),
                             "bytes": Path(path).stat().st_size}
    for context, path in (config.de_tables or {}).items():
        if Path(path).exists():
            inputs[f"pie_de_{context}"] = {"path": str(path), "sha256": sha256_file(path),
                                           "bytes": Path(path).stat().st_size}

    manifest = {
        "dataset_name": config.dataset_name,
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "versions": runtime_versions(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "config": {field.name: getattr(config, field.name) for field in fields(PipelineConfig)},
        "split_decisions": split_decisions(config),
        "inputs": inputs,
        "processed": {
            "dir": str(out_dir),
            "dataset_npz_sha256": sha256_file(out_dir / "dataset.npz"),
            "X_sha256": sha256_array(dataset.X),
            "num_nodes": dataset.num_nodes,
            "num_features": dataset.num_features,
        },
        "report": report.as_dict(),
    }
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"[build_dataset] wrote {out_dir} (manifest.json + dataset.npz + dataset.json)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

