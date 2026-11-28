#!/usr/bin/env python3
"""M5.1 dataset audit CLI: PASS/FAIL gate with cross-view visibility numbers.

    python scripts/audit_dataset.py --dataset data/processed/real_v1_1
    python scripts/audit_dataset.py --dataset data/processed/real_v1_1 \\
        --protocol dual_view_strict_holdout --out runs/m5_1/audit_strict.json
    python scripts/audit_dataset.py --dataset ... --json      # machine-readable stdout

Exit code 0 means every check passed, 1 means at least one FAIL (so a shell chain
can stop the server pipeline before training), 2 means the dataset could not be
loaded at all. The dataset directory is never written to unless ``--out`` says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.audit import audit_dataset, render_markdown, summarise  # noqa: E402
from cofi_gene.audit import write_report as write_audit_report  # noqa: E402
from cofi_gene.dataset import CoFiDataset  # noqa: E402
from cofi_gene.perturbation import EDGE_REMOVAL_MODES  # noqa: E402
from cofi_gene.protocols import VISIBILITY_PROTOCOLS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="processed dataset directory (contains dataset.npz/json)")
    parser.add_argument("--protocol", choices=VISIBILITY_PROTOCOLS,
                        default="cross_view_transfer",
                        help="visibility protocol the run will use (default: transfer)")
    parser.add_argument("--edge-removal-mode", choices=EDGE_REMOVAL_MODES,
                        default="adjacency", help="edge-removal mode of the planned runs")
    parser.add_argument("--out", default=None,
                        help="write the full report here (.json) and a .md mirror next to it")
    parser.add_argument("--json", action="store_true",
                        help="print the full JSON report instead of the markdown table")
    parser.add_argument("--no-roundtrip", action="store_true",
                        help="skip the save/load stability check (faster on big data)")
    args = parser.parse_args(argv)

    try:
        dataset = CoFiDataset.load(args.dataset)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[audit] cannot load {args.dataset}: {exc}", file=sys.stderr)
        return 2
    report = audit_dataset(dataset, dataset_dir=args.dataset, protocol=args.protocol,
                           edge_removal_mode=args.edge_removal_mode,
                           include_roundtrip=not args.no_roundtrip)
    if args.out:
        out = Path(args.out)
        write_audit_report(report, out)
        out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
        print(f"[audit] wrote {out} and {out.with_suffix('.md')}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        block = summarise(report)
        print(f"[audit] status={report['status']} checks={block['counts']['checks']} "
              f"failed={block['failed_ids']} warnings={block['warning_ids']}")
        for name, entry in report["visibility"]["channels"].items():
            print(f"[audit] channel {name}: val={entry['num_val']} test={entry['num_test']} "
                  f"visible_in_other_encode={entry['visible_in_other_encode']} "
                  f"(ratio {entry['visibility_ratio_encode']:.4f})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
