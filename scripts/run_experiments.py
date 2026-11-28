#!/usr/bin/env python3
"""M5.1 staged experiment matrix (Stage 0-5): dry-run, resume, skip finished cells.

Expands one declarative stage into concrete cells, runs one training (or one
frozen evaluation) per cell, skips cells that already finished, can re-run only
failed cells, and writes a stage summary next to the runs. The server owner must
never guess which protocol produced a number, so every run name encodes it.

  0  environment + dataset audit only (no training); non-zero exit on FAIL
  1  B0-B7, Protocol A (cross_view_transfer), uniform negatives, 0% drop
  2  B1/B2/B4/B5, Protocol B (dual_view_strict_holdout), 0% drop
  3  frozen stage-1 checkpoints evaluated on a hard-negative dataset
  4  B1/B2/B4/B5/B6/B7, supervision-aware removal, 0% and 50%
  5  B0-B7, locked protocol, seeds 42/43/44 (model seed; split fixed)

    python scripts/run_experiments.py --stage 0 --dataset data/processed/real_v1_1
    python scripts/run_experiments.py --stage 1 --dataset ... --device cuda --resume
    python scripts/run_experiments.py --stage 3 --dataset data/processed/real_v1_1_deg \\
        --frozen-from runs/m5_1/stage1
    python scripts/run_experiments.py --stage 4 --dataset ... --only-models b1 b5

``--resume`` skips a cell only when ``metrics.json`` is present, parses, and
records ``status == "complete"``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.perturbation import EDGE_REMOVAL_MODES  # noqa: E402
from cofi_gene.protocols import VISIBILITY_PROTOCOLS  # noqa: E402

ALL_MODELS = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")

#: Declarative description of every stage. ``kind`` is "audit"/"train"/"evaluate".
STAGES: dict[int, dict] = {
    0: {"title": "environment and dataset audit", "kind": "audit", "models": ()},
    1: {"title": "post-fix baseline results (Protocol A, uniform negatives)",
        "kind": "train", "models": ALL_MODELS,
        "protocol": "cross_view_transfer", "mode": "adjacency",
        "ratios": (0.0,), "seeds": (42,)},
    2: {"title": "strict dual-view holdout diagnosis (Protocol B)",
        "kind": "train", "models": ("b1", "b2", "b4", "b5"),
        "protocol": "dual_view_strict_holdout", "mode": "adjacency",
        "ratios": (0.0,), "seeds": (42,)},
    3: {"title": "hard-negative diagnosis on frozen stage-1 checkpoints",
        "kind": "evaluate", "models": ALL_MODELS,
        "protocol": "cross_view_transfer", "mode": "adjacency",
        "ratios": (0.0,), "seeds": (42,), "frozen_from_stage": 1},
    4: {"title": "supervision-aware edge removal (0% and 50%)",
        "kind": "train", "models": ("b1", "b2", "b4", "b5", "b6", "b7"),
        "protocol": "cross_view_transfer", "mode": "supervision_aware",
        "ratios": (0.0, 0.5), "seeds": (42,)},
    5: {"title": "three model seeds under the locked protocol",
        "kind": "train", "models": ALL_MODELS,
        "protocol": "cross_view_transfer", "mode": "adjacency",
        "ratios": (0.0,), "seeds": (42, 43, 44)},
}

_PROTOCOL_ABBR = {"cross_view_transfer": "xfer", "dual_view_strict_holdout": "strict"}
_MODE_ABBR = {"adjacency": "adj", "supervision_aware": "sup"}


def run_name_for(stage: int, model: str, ratio: float, seed: int, protocol: str,
                 mode: str) -> str:
    """Deterministic, self-describing run name (protocol and mode are visible)."""
    percent = int(round(float(ratio) * 100))
    return (f"s{stage}_{model}_{_PROTOCOL_ABBR[protocol]}_drop{percent:02d}"
            f"_{_MODE_ABBR[mode]}_seed{int(seed)}")


def plan(stage: int, models=None, ratios=None, seeds=None) -> list[dict]:
    """Expand one stage into cells, applying the optional filters."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    spec = STAGES[stage]
    if spec["kind"] == "audit":
        return [{"stage": stage, "kind": "audit", "model": None, "ratio": None,
                 "seed": None, "protocol": None, "mode": None,
                 "run_name": f"s{stage}_audit"}]
    selected_models = tuple(models or spec["models"])
    unknown = [model for model in selected_models if model not in ALL_MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; available: {ALL_MODELS}")
    not_in_stage = [model for model in selected_models if model not in spec["models"]]
    if not_in_stage:
        raise ValueError(f"stage {stage} only runs {spec['models']}; got {not_in_stage}")
    selected_ratios = tuple(float(value) for value in (ratios or spec["ratios"]))
    for ratio in selected_ratios:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1), got {ratio}")
        if ratio not in spec["ratios"]:
            raise ValueError(f"stage {stage} only runs ratios {spec['ratios']}, "
                             f"got {ratio}")
    selected_seeds = tuple(int(value) for value in (seeds or spec["seeds"]))
    cells = []
    for model in selected_models:
        for ratio in selected_ratios:
            for seed in selected_seeds:
                cells.append({
                    "stage": stage, "kind": spec["kind"], "model": model,
                    "ratio": ratio, "seed": seed, "protocol": spec["protocol"],
                    "mode": spec["mode"],
                    "run_name": run_name_for(stage, model, ratio, seed,
                                             spec["protocol"], spec["mode"]),
                })
    return cells


def is_complete(run_dir: Path) -> bool:
    """A cell finished only if metrics.json parses and says ``status=complete``."""
    metrics = Path(run_dir) / "metrics.json"
    if not metrics.exists():
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("status") == "complete"


def collect_metrics(out_dir: Path, cells: list[dict]) -> list[dict]:
    """Read each finished cell's evaluation metrics into one flat table."""
    rows = []
    for cell in cells:
        run_dir = Path(out_dir) / cell["run_name"]
        record_path = run_dir / "metrics.json"
        row = {key: cell.get(key) for key in ("model", "ratio", "seed", "run_name")}
        if cell["kind"] == "evaluate":
            eval_path = run_dir / "eval.json"
            payload = (json.loads(eval_path.read_text(encoding="utf-8"))
                       if eval_path.exists() else None)
            metrics = (payload or {}).get("metrics") or {}
            row["status"] = "complete" if payload else "missing"
        else:
            payload = (json.loads(record_path.read_text(encoding="utf-8"))
                       if record_path.exists() else None)
            metrics = (payload or {}).get("test") or {}
            row["status"] = "complete" if (payload or {}).get("status") == "complete" \
                else "missing"
        row["channels"] = {channel: entry.get("average_precision")
                           for channel, entry in metrics.items()}
        row["auroc"] = {channel: entry.get("roc_auc")
                        for channel, entry in metrics.items()}
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Group by model and report mean/std of the per-channel AP across seeds."""
    table: dict = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        for channel, value in row["channels"].items():
            if value is None:
                continue
            values = table.setdefault(row["model"], {}).setdefault(channel, [])
            values.append(float(value))
    summary = {}
    for model, per_channel in table.items():
        summary[model] = {}
        for channel, values in per_channel.items():
            summary[model][channel] = {
                "n": len(values),
                "mean_ap": statistics.fmean(values),
                "std_ap": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def render_markdown(summary: dict, stage: int, rows: list[dict]) -> str:
    spec = STAGES[stage]
    lines = [f"# M5.1 stage {stage}: {spec['title']}", "",
             f"- protocol: {spec.get('protocol')} | edge removal: {spec.get('mode')}",
             f"- cells: {len(rows)} "
             f"({sum(1 for row in rows if row['status'] == 'complete')} complete)", ""]
    channels = sorted({channel for per_model in summary.values()
                       for channel in per_model})
    if channels:
        lines.append("| model | " + " | ".join(f"{channel} mean AP" for channel in channels)
                     + " |")
        lines.append("| --- | " + " | ".join("---" for _ in channels) + " |")
        for model in sorted(summary):
            cells = []
            for channel in channels:
                entry = summary[model].get(channel)
                cells.append("—" if entry is None else
                             f"{entry['mean_ap']:.4f} ± {entry['std_ap']:.4f} "
                             f"(n={entry['n']})")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    missing = [row.get("run_name") for row in rows if row["status"] != "complete"]
    missing = [name for name in missing if name]
    if missing:
        lines += ["", f"Missing/incomplete cells ({len(missing)}): " + ", ".join(missing)]
    return "\n".join(lines) + "\n"


def _write_summary(out_dir: Path, stage: int, rows: list[dict], settings: dict) -> dict:
    payload = {"stage": stage, "title": STAGES[stage]["title"], "settings": settings,
               "rows": rows, "summary": summarise(rows)}
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"stage{stage}_summary"
    (out_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out_dir / f"{name}.md").write_text(render_markdown(payload["summary"], stage, rows),
                                        encoding="utf-8")
    return payload


def _run_audit(dataset_dir: str, protocol: str, edge_removal_mode: str,
               out_dir: Path) -> int:
    """Stage 0: environment/dataset audit only, non-zero exit on FAIL."""
    from cofi_gene.audit import audit_dataset, render_markdown
    from cofi_gene.audit import write_report as write_audit_report
    from cofi_gene.dataset import CoFiDataset

    dataset = CoFiDataset.load(dataset_dir)
    report = audit_dataset(dataset, dataset_dir=dataset_dir, protocol=protocol,
                           edge_removal_mode=edge_removal_mode)
    json_path = out_dir / "audit.json"
    write_audit_report(report, json_path)
    (out_dir / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"[stage0] status={report['status']} failed={report['failed_ids']} "
          f"warnings={report['warning_ids']} -> {json_path}")
    return 0 if report["status"] == "PASS" else 1


def _train_cell(cell: dict, args, dataset, out_dir: Path) -> None:
    from cofi_gene.config import load_config
    from cofi_gene.train import train_link_prediction

    overrides = {
        "data": {"visibility_protocol": cell["protocol"],
                 "edge_removal_mode": cell["mode"],
                 "edge_drop_fraction": cell["ratio"]},
        "runtime": {"seed": cell["seed"], "out_dir": str(out_dir),
                    "run_name": cell["run_name"], "device": args.device},
    }
    if args.epochs is not None:
        overrides["optim"] = {"epochs": int(args.epochs)}
    config = load_config(Path(args.config_dir) / f"{cell['model']}.yaml", overrides)
    train_link_prediction(dataset, config, device=args.device, out_dir=out_dir,
                          audit_mode=args.audit)


def _evaluate_cell(cell: dict, args, dataset, out_dir: Path, frozen_root: Path) -> None:
    """Stage 3: score a frozen stage-1 checkpoint, no training."""
    from cofi_gene.evaluate import evaluate_checkpoint

    frozen_stage = STAGES[cell["stage"]]["frozen_from_stage"]
    frozen_name = run_name_for(frozen_stage, cell["model"], cell["ratio"], cell["seed"],
                               cell["protocol"], cell["mode"])
    checkpoint = frozen_root / frozen_name / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"missing frozen checkpoint {checkpoint}; run stage {frozen_stage} first "
            "or pass --frozen-from")
    result = evaluate_checkpoint(dataset, checkpoint, split=args.split,
                                 device=args.device, protocol=cell["protocol"],
                                 edge_removal_mode=cell["mode"],
                                 edge_drop_fraction=cell["ratio"])
    run_dir = out_dir / cell["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    parser.add_argument("--dataset", required=True, help="processed dataset directory")
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--out-dir", default=None, help="default: runs/m5_1/stage<stage>")
    parser.add_argument("--device", default="cpu", help="e.g. cpu / cuda")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override optim.epochs (smoke runs only)")
    parser.add_argument("--only-models", nargs="+", default=None)
    parser.add_argument("--only-ratios", nargs="+", type=float, default=None)
    parser.add_argument("--only-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--frozen-from", default=None,
                        help="stage output dir holding frozen checkpoints "
                             "(default runs/m5_1/stage1)")
    parser.add_argument("--split", choices=("val", "test"), default="test",
                        help="split scored by stage 3 (default test)")
    parser.add_argument("--audit", choices=("strict", "warn", "off"), default="strict")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells whose metrics.json is complete")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop the stage at the first failing cell")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the expanded plan and exit")
    args = parser.parse_args(argv)

    cells = plan(args.stage, args.only_models, args.only_ratios, args.only_seeds)
    out_dir = Path(args.out_dir or (ROOT / "runs" / "m5_1" / f"stage{args.stage}"))
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "out_dir": str(out_dir),
                          "cells": cells}, indent=2, sort_keys=True))
        return 0

    spec = STAGES[args.stage]
    if spec["kind"] == "audit":
        return _run_audit(args.dataset, spec.get("protocol") or "cross_view_transfer",
                          spec.get("mode") or "adjacency", out_dir)

    from cofi_gene.dataset import CoFiDataset

    dataset = CoFiDataset.load(args.dataset)
    frozen_root = Path(args.frozen_from or (ROOT / "runs" / "m5_1" / "stage1"))
    settings = {"dataset": str(args.dataset), "device": args.device,
                "epochs_override": args.epochs, "audit": args.audit,
                "protocol": spec["protocol"], "edge_removal_mode": spec["mode"],
                "ratios": sorted({cell["ratio"] for cell in cells}),
                "seeds": sorted({cell["seed"] for cell in cells}),
                "split": args.split}
    failures: list[dict] = []
    for index, cell in enumerate(cells, 1):
        run_dir = out_dir / cell["run_name"]
        if args.resume and is_complete(run_dir):
            print(f"[stage{args.stage}] skip complete: {cell['run_name']}", flush=True)
            continue
        print(f"[stage{args.stage}] {index}/{len(cells)} {cell['run_name']}", flush=True)
        try:
            if cell["kind"] == "evaluate":
                _evaluate_cell(cell, args, dataset, out_dir, frozen_root)
            else:
                _train_cell(cell, args, dataset, out_dir)
        except Exception as exc:  # keep going so one bad cell does not lose the stage
            failures.append({"cell": cell["run_name"],
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"[stage{args.stage}] FAILED {cell['run_name']}: {exc}", flush=True)
            if args.fail_fast:
                break
        _write_summary(out_dir, args.stage, collect_metrics(out_dir, cells), settings)
    rows = collect_metrics(out_dir, cells)
    _write_summary(out_dir, args.stage, rows, settings)
    complete = sum(1 for row in rows if row["status"] == "complete")
    print(f"[stage{args.stage}] {complete}/{len(rows)} complete; summary -> "
          f"{out_dir / f'stage{args.stage}_summary.json'}")
    if failures:
        print(f"[stage{args.stage}] failures: {json.dumps(failures)}")
        return 1
    return 0 if complete == len(rows) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())




