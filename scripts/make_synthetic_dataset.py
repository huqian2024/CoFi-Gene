#!/usr/bin/env python3
"""Write a synthetic raw-data tree that mimics STRING + gene_text + PIE layout.

    python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1
    python scripts/make_synthetic_dataset.py --raw data/raw/synthetic_v1 \\
        --processed data/processed/synthetic_v1

Nothing here is biological data: it exists so the whole pipeline, the leakage
checks and B0-B3 training can be exercised on a CPU laptop and in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.manifest import utc_timestamp, write_manifest  # noqa: E402
from cofi_gene.pipeline import PipelineConfig, build_dataset  # noqa: E402
from cofi_gene.synthetic import SyntheticSpec, write_synthetic_raw  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="directory for the fake raw files")
    parser.add_argument("--processed", help="also build a processed dataset here")
    parser.add_argument("--n-genes", type=int, default=60)
    parser.add_argument("--n-raw-features", type=int, default=64)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--n-downstream-targets", type=int, default=24)
    parser.add_argument("--n-contexts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = SyntheticSpec(n_genes=args.n_genes, n_raw_features=args.n_raw_features,
                         n_downstream_targets=args.n_downstream_targets,
                         n_contexts=args.n_contexts, seed=args.seed)
    paths = write_synthetic_raw(args.raw, spec)
    print(json.dumps({"raw": str(paths["root"]),
                      "pie_written": paths["truth"]["pie_written"],
                      "links": paths["truth"]["links"]}, indent=2, sort_keys=True))
    if not paths["truth"]["pie_written"]:
        print("[make_synthetic] pyarrow missing: downstream tables were skipped "
              "(pip install -e '.[data]')", file=sys.stderr)

    if args.processed:
        config = PipelineConfig(
            dataset_name="synthetic_v1",
            string_links=str(paths["string_links"]),
            string_info=str(paths["string_info"]),
            gene_text=str(paths["gene_text"]),
            gene_order=str(paths["gene_order"]),
            de_tables={context: str(path) for context, path in paths["de_tables"].items()},
            n_components=args.n_components,
            split_seed=42,
        )
        dataset, report = build_dataset(config)
        out_dir = Path(args.processed)
        dataset.save(out_dir)
        write_manifest(out_dir / "manifest.json", {
            "dataset_name": config.dataset_name,
            "created_at": utc_timestamp(),
            "synthetic": True,
            "spec": paths["truth"]["spec"],
            "raw_root": str(paths["root"]),
            "config": {k: v for k, v in vars(config).items()},
            "report": report.as_dict(),
            "processed": {"dir": str(out_dir), "num_nodes": dataset.num_nodes,
                          "num_features": dataset.num_features},
        })
        print(f"[make_synthetic] wrote processed dataset to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
