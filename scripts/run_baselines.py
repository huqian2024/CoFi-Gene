#!/usr/bin/env python3
"""M5.1 non-neural baselines on one processed dataset (no torch, no training).

    python scripts/run_baselines.py --dataset data/processed/real_v1_1
    python scripts/run_baselines.py --dataset data/processed/real_v1_1 \\
        --protocol dual_view_strict_holdout --split test
    python scripts/run_baselines.py --dataset ... --edge-removal-mode supervision_aware \\
        --edge-drop-fraction 0.5 --out-dir runs/m5_1/stage2_baselines

The baselines see exactly the graph a model would see (visibility protocol +
optional edge removal applied first) and are scored on the dataset's fixed
positives/negatives, so the resulting AP/AUROC rows are directly comparable to
B0-B7 rows from the same stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.baselines import BASELINE_NAMES, evaluate_baselines  # noqa: E402
from cofi_gene.dataset import CoFiDataset  # noqa: E402
from cofi_gene.manifest import git_commit, sha256_file, utc_timestamp  # noqa: E402
from cofi_gene.perturbation import EDGE_REMOVAL_MODES, apply_data_perturbations  # noqa: E402
from cofi_gene.protocols import VISIBILITY_PROTOCOLS  # noqa: E402


def _format(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def render_markdown(summary: dict) -> str:
    settings = summary["settings"]
    lines = [
        "# M5.1 non-neural baselines",
        "",
        f"- created_at: {summary['created_at']}",
        f"- git_commit: {summary['git_commit']}",
        f"- dataset: {summary['dataset']['processed_dir']} "
        f"(sha256 {summary['dataset']['sha256']})",
        f"- split: {settings['split']} (graph: {settings['graph']})",
        f"- protocol: {settings['visibility_protocol']}",
        f"- edge removal: {settings['edge_removal_mode']} "
        f"fraction={settings['edge_drop_fraction']} channels={settings['drop_channels']}",
        f"- positive rate: {summary['positive_rate']} "
        "(AP of a random ranker equals this value)",
        "",
    ]
    for channel, results in sorted(summary["results"].items()):
        lines += [f"## Channel {channel}", "",
                  "| baseline | AUPRC | AUROC | n_pos | n_neg | status |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name in settings["baselines"]:
            entry = results.get(name, {})
            lines.append(
                f"| {name} | {_format(entry.get('average_precision'))} | "
                f"{_format(entry.get('roc_auc'))} | {entry.get('n_positives', '—')} | "
                f"{entry.get('n_negatives', '—')} | {entry.get('status', 'ok')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="processed dataset directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_NAMES,
                        default=list(BASELINE_NAMES))
    parser.add_argument("--channels", nargs="+", default=None,
                        help="channels to evaluate (default: every channel in the dataset)")
    parser.add_argument("--graph", choices=("encode_edges", "train"), default="encode_edges",
                        help="training graph the baselines may read (default: encode_edges)")
    parser.add_argument("--protocol", choices=VISIBILITY_PROTOCOLS,
                        default="cross_view_transfer")
    parser.add_argument("--edge-removal-mode", choices=EDGE_REMOVAL_MODES,
                        default="adjacency")
    parser.add_argument("--edge-drop-fraction", type=float, default=0.0)
    parser.add_argument("--drop-channels", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="perturbation seed (must match the model runs)")
    parser.add_argument("--out-dir", default="runs/m5_1_baselines")
    parser.add_argument("--summary-name", default="baselines")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned settings and exit")
    args = parser.parse_args(argv)

    settings = {
        "split": args.split,
        "baselines": list(args.baselines),
        "channels": list(args.channels) if args.channels else None,
        "graph": args.graph,
        "visibility_protocol": args.protocol,
        "edge_removal_mode": args.edge_removal_mode,
        "edge_drop_fraction": float(args.edge_drop_fraction),
        "drop_channels": list(args.drop_channels) if args.drop_channels else None,
        "seed": int(args.seed),
    }
    if args.dry_run:
        print(json.dumps({"settings": settings, "dataset": args.dataset}, indent=2,
                         sort_keys=True))
        return 0

    dataset = CoFiDataset.load(args.dataset)
    config = {"data": {"visibility_protocol": args.protocol,
                       "edge_removal_mode": args.edge_removal_mode,
                       "edge_drop_fraction": args.edge_drop_fraction},
              "runtime": {"seed": args.seed}}
    if args.drop_channels:
        config["data"]["drop_channels"] = list(args.drop_channels)
    perturbed, perturbation = apply_data_perturbations(dataset, config)
    results = evaluate_baselines(perturbed, split=args.split, names=args.baselines,
                                 channels=args.channels, graph=args.graph)
    positive_rates = {}
    for channel, per_baseline in results.items():
        rates = {entry.get("positive_rate") for entry in per_baseline.values()
                 if "positive_rate" in entry}
        positive_rates[channel] = rates.pop() if len(rates) == 1 else None
    npz = Path(args.dataset) / "dataset.npz"
    summary = {
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "dataset": {"processed_dir": str(args.dataset),
                    "sha256": sha256_file(npz) if npz.exists() else None,
                    "num_nodes": dataset.num_nodes,
                    "num_features": dataset.num_features},
        "settings": settings,
        "perturbation": perturbation,
        "positive_rate": positive_rates,
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.summary_name}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out_dir / f"{args.summary_name}.md").write_text(render_markdown(summary),
                                                     encoding="utf-8")
    for channel, per_baseline in sorted(results.items()):
        for name in settings["baselines"]:
            entry = per_baseline.get(name, {})
            print(f"[baselines] {channel} {name}: "
                  f"AP={_format(entry.get('average_precision'))} "
                  f"AUROC={_format(entry.get('roc_auc'))} "
                  f"n_pos={entry.get('n_positives', '—')} "
                  f"n_neg={entry.get('n_negatives', '—')} {entry.get('status', '')}")
    print(f"[baselines] wrote {out_dir / (args.summary_name + '.json')} and .md")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

