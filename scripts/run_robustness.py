#!/usr/bin/env python3
"""Week 5 (M5) robustness sweep: B0-B7 under 10/30/50% missing training edges.

    python scripts/run_robustness.py --dataset data/processed/real_v1 --device cuda
    python scripts/run_robustness.py --dataset data/processed/real_v1 \\
        --models b1 b4 b5 b7 --ratios 0 0.1 0.3 0.5
    python scripts/run_robustness.py --dataset data/processed/real_v1 \\
        --misalign-cross-view --ratios 0      # experiment 3 control
    python scripts/run_robustness.py --dataset ... --dry-run

Every ``(model, ratio)`` cell is a full training run on the *same* processed
dataset: only each channel's encoding (training) graph is thinned by ``ratio``,
and with ``--mask-other-view`` each view's held-out pairs are also hidden in the
other view. Validation/test positives and the fixed negative sample never
change, so all ratios are scored on one evaluation set (``AGENTS.md`` §6).

The script writes ``<summary>.json`` and ``<summary>.md`` next to the run
directories, so the robustness curve can be plotted without re-reading every
``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.config import CROSS_VIEW_MODELS, load_config  # noqa: E402
from cofi_gene.dataset import CoFiDataset  # noqa: E402
from cofi_gene.manifest import git_commit, sha256_file, utc_timestamp  # noqa: E402

MODELS = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")
DEFAULT_RATIOS = (0.0, 0.1, 0.3, 0.5)


def run_name_for(model: str, ratio: float, seed: int, mask_other_view: bool = False,
                 misalign_cross_view: bool = False) -> str:
    """Deterministic run name, so re-running the sweep overwrites instead of piling up."""
    percent = int(round(float(ratio) * 100))
    suffix = ("_maskother" if mask_other_view else "") + (
        "_misalign" if misalign_cross_view else "")
    return f"{model}_drop{percent:02d}{suffix}_seed{int(seed)}"


def plan_runs(models, ratios, seed: int, mask_other_view: bool = False,
              misalign_cross_view: bool = False) -> list[dict]:
    """Validate the sweep and return one entry per ``(model, ratio)`` pair."""
    models = tuple(models)
    if not models:
        raise ValueError("at least one model is required")
    unknown = [model for model in models if model not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; available: {MODELS}")
    if misalign_cross_view:
        unsupported = [model for model in models if model not in CROSS_VIEW_MODELS]
        if unsupported:
            raise ValueError(
                f"--misalign-cross-view needs a cross-view model {CROSS_VIEW_MODELS}; "
                f"got {unsupported}")
    plan = []
    for model in models:
        for ratio in ratios:
            ratio = float(ratio)
            if not 0.0 <= ratio < 1.0:
                raise ValueError(f"ratio must be in [0, 1), got {ratio}")
            plan.append({
                "model": model,
                "edge_drop_fraction": ratio,
                "run_name": run_name_for(model, ratio, seed, mask_other_view,
                                         misalign_cross_view),
            })
    return plan


def _metric_table(results: list[dict], metric: str) -> dict:
    """``{channel: {model: {ratio: value}}}`` for one metric."""
    table: dict = {}
    for entry in results:
        for channel, metrics in (entry.get("test") or {}).items():
            per_model = table.setdefault(channel, {}).setdefault(entry["model"], {})
            per_model[f"{float(entry['edge_drop_fraction']):.2f}"] = metrics.get(metric)
    return table



def summarize(results: list[dict], settings: dict, dataset_info: dict) -> dict:
    """Collect run records into one JSON-ready summary (tables + raw runs)."""
    return {
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "dataset": dataset_info,
        "settings": settings,
        "table": {
            "average_precision": _metric_table(results, "average_precision"),
            "roc_auc": _metric_table(results, "roc_auc"),
        },
        "runs": results,
    }


def _format(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def render_markdown(summary: dict) -> str:
    """Human-readable mirror of :func:`summarize` (one table per metric/channel)."""
    settings = summary["settings"]
    dataset = summary["dataset"]
    lines = [
        "# Week 5 robustness sweep (M5)",
        "",
        f"- created_at: {summary['created_at']}",
        f"- git_commit: {summary['git_commit']}",
        f"- dataset: {dataset.get('processed_dir')} (sha256 {dataset.get('sha256')})",
        f"- seed: {settings['seed']}",
        f"- ratios: {settings['ratios']}",
        f"- mask_other_view: {settings['mask_other_view']}",
        f"- misalign_cross_view: {settings['misalign_cross_view']}",
        "",
        "Evaluation positives and the fixed negative sample are identical in every",
        "cell; only each channel's encoding (training) graph was thinned. B0 ignores",
        "the graph, so its row is a flat feature baseline; B1 is scored on A only and",
        "B2 on B only.",
        "",
    ]
    for metric, title in (("average_precision", "AUPRC"), ("roc_auc", "AUROC")):
        for channel, per_model in sorted(summary["table"][metric].items()):
            # Show every requested ratio, not just the ones already finished, so a
            # partially written summary still lines up with the final table.
            observed = {ratio for values in per_model.values() for ratio in values}
            requested = {f"{float(ratio):.2f}" for ratio in settings.get("ratios") or ()}
            ratios = sorted(observed | requested, key=float)
            lines.append(f"## {title}: channel {channel}")
            lines.append("")
            lines.append("| model | "
                         + " | ".join(f"drop {ratio}" for ratio in ratios) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in ratios) + " |")
            for model in sorted(per_model):
                cells = " | ".join(_format(per_model[model].get(ratio)) for ratio in ratios)
                lines.append(f"| {model} | {cells} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def _write_summary(summary: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render_markdown(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="processed dataset directory (contains dataset.npz/json)")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help=f"models to sweep (default: {' '.join(MODELS)})")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS),
                        help="missing-edge fractions (default: 0 0.1 0.3 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu / cuda")
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--out-dir", default="runs/m5_robustness")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override optim.epochs (smoke runs only)")
    parser.add_argument("--mask-other-view", action="store_true",
                        help="hide each view's held-out pairs in the other view too")
    parser.add_argument("--misalign-cross-view", action="store_true",
                        help="shuffle the A/B gene correspondence inside fusion")
    parser.add_argument("--summary-name", default="summary",
                        help="basename of the summary json/md files")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan and exit (no dataset, no torch)")
    args = parser.parse_args(argv)

    plan = plan_runs(args.models, args.ratios, args.seed, args.mask_other_view,
                     args.misalign_cross_view)
    out_dir = Path(args.out_dir)
    if args.dry_run:
        print(json.dumps({"plan": plan, "out_dir": str(out_dir),
                          "config_dir": args.config_dir}, indent=2, sort_keys=True))
        return 0

    from cofi_gene.train import train_link_prediction  # torch is imported lazily

    dataset = CoFiDataset.load(args.dataset)
    dataset.validate()
    npz = Path(args.dataset) / "dataset.npz"
    dataset_info = {"processed_dir": str(args.dataset),
                    "sha256": sha256_file(npz) if npz.exists() else None,
                    "num_nodes": dataset.num_nodes,
                    "num_features": dataset.num_features}
    settings = {"seed": int(args.seed), "ratios": [float(r) for r in args.ratios],
                "models": list(args.models),
                "mask_other_view": bool(args.mask_other_view),
                "misalign_cross_view": bool(args.misalign_cross_view),
                "device": args.device, "epochs_override": args.epochs}

    results: list[dict] = []
    for item in plan:
        overrides: dict = {
            "data": {"edge_drop_fraction": item["edge_drop_fraction"],
                     "mask_other_view": bool(args.mask_other_view),
                     "misalign_cross_view": bool(args.misalign_cross_view)},
            "runtime": {"seed": int(args.seed), "out_dir": str(out_dir),
                        "run_name": item["run_name"], "device": args.device},
        }
        if args.epochs is not None:
            overrides["optim"] = {"epochs": int(args.epochs)}
        config = load_config(Path(args.config_dir) / f"{item['model']}.yaml", overrides)
        record = train_link_prediction(dataset, config, device=args.device,
                                       out_dir=out_dir)
        entry = {
            "model": item["model"],
            "edge_drop_fraction": item["edge_drop_fraction"],
            "run_name": record["run_name"],
            "run_dir": str(out_dir / record["run_name"]),
            "best_epoch": record["best_epoch"],
            "best_val_mean_ap": record["best_val_mean_ap"],
            "test": record["test"],
            "perturbation": record.get("perturbation"),
        }
        results.append(entry)
        test_ap = {channel: metrics.get("average_precision")
                   for channel, metrics in (record["test"] or {}).items()}
        print(f"[run_robustness] {item['run_name']}: "
              f"best_epoch={record['best_epoch']} "
              f"val_mean_ap={record['best_val_mean_ap']:.4f} "
              f"test_ap={json.dumps(test_ap, sort_keys=True)}", flush=True)
        # Write after every cell, so a long sweep keeps partial, readable results.
        _write_summary(summarize(results, settings, dataset_info), out_dir,
                       args.summary_name)

    summary = summarize(results, settings, dataset_info)
    _write_summary(summary, out_dir, args.summary_name)
    print(f"[run_robustness] wrote {out_dir / (args.summary_name + '.json')} and .md")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

