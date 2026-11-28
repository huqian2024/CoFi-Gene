#!/usr/bin/env python3
"""Week 7: aggregate multi-seed runs into mean±std, paired deltas and a budget report.

    python scripts/aggregate_seeds.py --runs runs/week7/stage5
    python scripts/aggregate_seeds.py --runs runs/m5_1/stage5 runs/week7/stage5 \\
        --metric roc_auc --out-dir runs/week7/aggregate --name seed_aggregate

Reads only ``metrics.json`` files that ``cofi_gene.train`` already wrote. Reports,
per model and channel, the mean and **sample** standard deviation across seeds;
then the planned paired comparisons (same seed first, then mean/std of the
differences); then how many runs selected their best epoch at the epoch cap
because a binding budget cannot support an ablation claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.aggregate import aggregate, load_records, write_summary  # noqa: E402

METRICS = ("average_precision", "roc_auc")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run directories (searched recursively) or metrics.json files")
    parser.add_argument("--out-dir", default="runs/week7/aggregate")
    parser.add_argument("--name", default="seed_aggregate")
    parser.add_argument("--metric", choices=METRICS, default="average_precision")
    parser.add_argument("--reference", default="b0", help="reference model for the deltas")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--include-incomplete", action="store_true",
                        help="also count runs without status=complete (not recommended)")
    parser.add_argument("--json", action="store_true", help="print the full JSON summary")
    args = parser.parse_args(argv)

    records, skipped = load_records(args.runs, split=args.split)
    if not records:
        print(f"[aggregate] no metrics.json found under {args.runs}; skipped={skipped}",
              file=sys.stderr)
        return 2
    summary = aggregate(records, metric=args.metric, reference=args.reference,
                        require_complete=not args.include_incomplete,
                        min_seeds=args.min_seeds)
    summary["skipped"] = skipped
    summary["sources"] = [str(path) for path in args.runs]
    paths = write_summary(summary, args.out_dir, args.name)

    for model in sorted(summary["models"]):
        cells = []
        for channel, entry in sorted(summary["models"][model].items()):
            cells.append(f"{channel}: {entry['mean']:.4f} ± {entry['std']:.4f} "
                         f"(n={entry['n']})")
        print(f"[aggregate] {model}: " + " | ".join(cells))
    budget = summary["budget"]
    print(f"[aggregate] budget: {budget['n_saturated']}/{budget['n_runs']} run(s) at the "
          f"epoch cap {budget['epoch_caps']}")
    for warning in summary["warnings"]:
        print(f"[aggregate] warning: {warning}")
    if skipped:
        print(f"[aggregate] skipped {len(skipped)} file(s): {skipped[:3]}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"[aggregate] wrote {paths['json']} and {paths['markdown']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
