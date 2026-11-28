"""Week 7 (M7) seed aggregation: mean/std, paired deltas and budget saturation.

Week 7 of ``AGENTS.md`` asks for "fix the hyper-parameters, run three seeds, report
mean/std plus three figures". This module answers the numeric half:

* load every ``metrics.json`` of a set of run directories (or one stage output);
* group by model and channel, report mean and **sample** standard deviation;
* compute the *paired* comparison the write-up needs (same seed first, then mean and
  std of the differences -- never "std of one group plus std of the other");
* flag runs whose best epoch sits on the epoch cap, because a budget that is still
  binding cannot support an ablation;
* record the provenance (dataset hash, protocol, edge-removal mode, ratio, epoch
  cap) and warn when a table would mix them.

Only files already produced by ``cofi_gene.train`` are read; nothing is trained,
nothing is invented, and a missing/incomplete run is reported instead of silently
counted.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import git_commit, utc_timestamp

__all__ = [
    "DEFAULT_COMPARISONS",
    "RunRecord",
    "load_records",
    "aggregate",
    "render_markdown",
    "write_summary",
]

#: Planned comparisons of ``AGENTS.md`` §6 (same seed, then difference).
DEFAULT_COMPARISONS = (
    ("b1-b0", "b1", "b0"),
    ("b2-b0", "b2", "b0"),
    ("b3-b0", "b3", "b0"),
    ("b4-b0", "b4", "b0"),
    ("b5-b0", "b5", "b0"),
    ("b6-b0", "b6", "b0"),
    ("b7-b0", "b7", "b0"),
    ("b5-b4", "b5", "b4"),
    ("b6-b5", "b6", "b5"),
    ("b7-b4", "b7", "b4"),
    ("b6-b7", "b6", "b7"),
)


@dataclass
class RunRecord:
    """One ``metrics.json`` reduced to the fields a Week-7 table needs."""

    run_name: str
    model: str
    seed: int
    metrics: dict = field(default_factory=dict)      # channel -> {ap, auroc}
    best_epoch: int | None = None
    epoch_cap: int | None = None
    protocol: str | None = None
    edge_removal_mode: str | None = None
    edge_drop_fraction: float | None = None
    dataset_sha256: str | None = None
    status: str = "legacy"
    split: str = "test"

    @property
    def saturates_budget(self) -> bool:
        """True when the best epoch is the last allowed epoch (budget still binding)."""
        return (self.best_epoch is not None and self.epoch_cap is not None
                and int(self.best_epoch) >= int(self.epoch_cap))


def _record_from_metrics(path: Path, split: str) -> RunRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics_block = payload.get(split) or {}
    metrics = {}
    for channel, entry in metrics_block.items():
        if not isinstance(entry, dict):
            continue
        metrics[channel] = {
            "average_precision": entry.get("average_precision"),
            "roc_auc": entry.get("roc_auc"),
        }
    protocol = payload.get("protocol") or {}
    dataset = payload.get("dataset") or {}
    optim = (payload.get("config") or {}).get("optim") or {}
    model_cfg = payload.get("model") or {}
    return RunRecord(
        run_name=payload.get("run_name") or path.parent.name,
        model=str(model_cfg.get("name") or path.parent.name.split("_")[0]).lower(),
        seed=int((payload.get("config") or {}).get("runtime", {}).get("seed", -1)),
        metrics=metrics,
        best_epoch=payload.get("best_epoch"),
        epoch_cap=int(optim["epochs"]) if "epochs" in optim else None,
        protocol=protocol.get("visibility_protocol"),
        edge_removal_mode=protocol.get("edge_removal_mode"),
        edge_drop_fraction=protocol.get("edge_drop_fraction"),
        dataset_sha256=dataset.get("dataset_sha256"),
        status=str(payload.get("status") or "legacy"),
        split=split,
    )


def load_records(paths, split: str = "test") -> tuple[list[RunRecord], list[dict]]:
    """Load every ``metrics.json`` under ``paths``; return ``(records, skipped)``.

    ``paths`` may be directories (searched recursively) or explicit JSON files.
    A directory without any ``metrics.json`` yields a ``skipped`` entry rather
    than an empty, silently successful table.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[RunRecord] = []
    skipped: list[dict] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        found = sorted(path.rglob("metrics.json")) if path.is_dir() else [path]
        if not found:
            skipped.append({"path": str(path), "reason": "no metrics.json found"})
            continue
        for metrics_path in found:
            if metrics_path in seen:
                continue
            seen.add(metrics_path)
            try:
                records.append(_record_from_metrics(metrics_path, split))
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                skipped.append({"path": str(metrics_path),
                                "reason": f"{type(exc).__name__}: {exc}"})
    records.sort(key=lambda record: (record.model, record.seed, record.run_name))
    return records, skipped


def _stats(values: list[float]) -> dict:
    """Mean, **sample** std (n-1), min/max and n for a non-empty value list."""
    if not values:
        raise ValueError("cannot summarise an empty value list")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(records: list[RunRecord], metric: str = "average_precision",
              reference: str = "b0", comparisons=None,
              require_complete: bool = True,
              min_seeds: int = 3) -> dict:
    """Aggregate per-model means, paired comparisons and the budget report.

    ``require_complete`` drops runs that never reached ``status == "complete"``
    (pre-M5.1 ``metrics.json`` counts as ``legacy``), because a half-written run
    would silently corrupt a mean. ``min_seeds`` only produces a warning: with two
    seeds a sample std exists but is weak evidence, and the caller decides.
    """
    kept = [record for record in records if not require_complete
            or record.status in ("complete", "legacy")]
    warnings: list[str] = []
    kept_ids = {id(record) for record in kept}
    dropped = [record.run_name for record in records if id(record) not in kept_ids]
    if dropped:
        warnings.append(f"{len(dropped)} run(s) without status=complete were dropped: "
                        + ", ".join(dropped[:5]))

    channels = sorted({channel for record in kept for channel in record.metrics})
    models_block: dict = {}
    duplicate_seeds: list[str] = []
    for model in sorted({record.model for record in kept}):
        per_channel: dict = {}
        for channel in channels:
            values: dict[int, float] = {}
            for record in kept:
                if record.model != model:
                    continue
                value = record.metrics.get(channel, {}).get(metric)
                if value is None:
                    continue
                if record.seed in values:
                    duplicate_seeds.append(f"{model}/seed{record.seed}/{channel}")
                    continue
                values[record.seed] = float(value)
            if not values:
                continue
            entry = _stats([values[seed] for seed in sorted(values)])
            entry["values"] = {str(seed): values[seed] for seed in sorted(values)}
            per_channel[channel] = entry
        if per_channel:
            models_block[model] = per_channel
    if duplicate_seeds:
        warnings.append("duplicate (model, seed) runs ignored after the first: "
                        + ", ".join(sorted(set(duplicate_seeds))[:5]))

    for model, per_channel in models_block.items():
        seeds = {seed for entry in per_channel.values() for seed in entry["values"]}
        if len(seeds) < min_seeds:
            warnings.append(f"model {model} has {len(seeds)} seed(s) (< {min_seeds}); "
                            "the standard deviation is not yet a stability estimate")

    plans = comparisons or DEFAULT_COMPARISONS
    return _finish_aggregate(kept, models_block, channels, metric, reference, plans,
                             warnings, duplicate_seeds)


def _finish_aggregate(kept: list[RunRecord], models_block: dict, channels: list[str],
                      metric: str, reference: str, plans, warnings: list[str],
                      duplicate_seeds=None) -> dict:
    """Comparisons, budget saturation and provenance for :func:`aggregate`."""
    duplicate_seeds = duplicate_seeds or []
    comparisons_block: dict = {}
    for name, left, right in plans:
        if left not in models_block or right not in models_block:
            continue
        per_channel: dict = {}
        for channel in channels:
            first = models_block[left].get(channel, {}).get("values", {})
            second = models_block[right].get(channel, {}).get("values", {})
            shared = sorted(set(first) & set(second), key=int)
            if not shared:
                continue
            deltas = {seed: first[seed] - second[seed] for seed in shared}
            entry = _stats([deltas[seed] for seed in shared])
            entry["per_seed"] = deltas
            entry["consistent_sign"] = all(value > 0 for value in deltas.values()) \
                or all(value < 0 for value in deltas.values())
            per_channel[channel] = entry
        if per_channel:
            comparisons_block[name] = per_channel

    caps = sorted({record.epoch_cap for record in kept if record.epoch_cap is not None})
    saturated = [{"run_name": record.run_name, "model": record.model,
                  "seed": record.seed, "best_epoch": record.best_epoch,
                  "epoch_cap": record.epoch_cap}
                 for record in kept if record.saturates_budget]
    if saturated:
        warnings.append(f"{len(saturated)}/{len(kept)} run(s) selected their best epoch at "
                        "the epoch cap; the budget is still binding (D-018)")

    provenance = {
        "datasets": sorted({record.dataset_sha256 for record in kept
                            if record.dataset_sha256}),
        "protocols": sorted({record.protocol for record in kept if record.protocol}),
        "edge_removal_modes": sorted({record.edge_removal_mode for record in kept
                                      if record.edge_removal_mode}),
        "edge_drop_fractions": sorted({record.edge_drop_fraction for record in kept
                                       if record.edge_drop_fraction is not None}),
        "epoch_caps": caps,
        "seeds": sorted({record.seed for record in kept}),
        "statuses": sorted({record.status for record in kept}),
    }
    if len(provenance["datasets"]) > 1:
        warnings.append("the table mixes several dataset hashes; split the table by data "
                        "version before reporting it")
    if len(provenance["protocols"]) > 1 or len(provenance["edge_removal_modes"]) > 1:
        warnings.append("the table mixes visibility protocols / edge-removal modes")
    integrity = {
        "mixed_datasets": len(provenance["datasets"]) > 1,
        "mixed_protocols": (len(provenance["protocols"]) > 1
                            or len(provenance["edge_removal_modes"]) > 1),
        "duplicate_seeds": bool(duplicate_seeds),
        #: Blocking means "do not draw a table from this summary"; a binding budget
        #: or a small seed count is a caveat to print, not a reason to hide the figure.
        "blocking": bool(duplicate_seeds) or len(provenance["datasets"]) > 1
        or len(provenance["protocols"]) > 1 or len(provenance["edge_removal_modes"]) > 1,
    }
    return {
        "created_at": utc_timestamp(),
        "git_commit": git_commit(),
        "metric": metric,
        "reference": reference,
        "split": sorted({record.split for record in kept}),
        "runs": {"n": len(kept), "seeds": provenance["seeds"],
                 "models": sorted(models_block)},
        "provenance": provenance,
        "integrity": integrity,
        "models": models_block,
        "comparisons": comparisons_block,
        "budget": {"epoch_caps": caps, "n_runs": len(kept),
                   "n_saturated": len(saturated), "saturated": saturated},
        "warnings": warnings,
    }


def _format(value, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_markdown(summary: dict) -> str:
    """Human-readable mirror: provenance, mean±std, paired deltas, budget."""
    provenance = summary["provenance"]
    lines = [
        f"# Week 7 seed aggregate ({summary['metric']})",
        "",
        f"- created_at: {summary['created_at']}",
        f"- git_commit: {summary['git_commit']}",
        f"- runs: {summary['runs']['n']} | models: {', '.join(summary['runs']['models'])}",
        f"- seeds: {provenance['seeds']} | epoch caps: {provenance['epoch_caps']}",
        f"- protocol(s): {provenance['protocols']} | "
        f"edge removal: {provenance['edge_removal_modes']} | "
        f"ratios: {provenance['edge_drop_fractions']}",
        f"- dataset hash(es): {len(provenance['datasets'])} distinct",
        "",
        "Means are over seeds; `±` is the **sample** standard deviation (n-1).",
        "",
    ]
    channels = sorted({channel for per_channel in summary["models"].values()
                       for channel in per_channel})
    for channel in channels:
        lines += [f"## Channel {channel}", "",
                  "| model | n | mean ± std | min | max | per-seed |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for model in sorted(summary["models"]):
            entry = summary["models"][model].get(channel)
            if entry is None:
                continue
            per_seed = ", ".join(f"{seed}:{value:.4f}"
                                 for seed, value in entry["values"].items())
            lines.append(f"| {model} | {entry['n']} | "
                         f"{entry['mean']:.4f} ± {entry['std']:.4f} | "
                         f"{_format(entry['min'])} | {_format(entry['max'])} | "
                         f"{per_seed} |")
        lines.append("")
    if summary["comparisons"]:
        lines += ["## Paired comparisons (same seed first, then mean ± std)", ""]
        for name in sorted(summary["comparisons"]):
            for channel in sorted(summary["comparisons"][name]):
                entry = summary["comparisons"][name][channel]
                flag = "stable" if entry["consistent_sign"] else "sign flips"
                lines.append(f"- `{name}` channel {channel}: "
                             f"{entry['mean']:+.4f} ± {entry['std']:.4f} "
                             f"(n={entry['n']}, {flag})")
        lines.append("")
    budget = summary["budget"]
    lines += [f"## Budget (cap {budget['epoch_caps']})", "",
              f"- saturated runs: {budget['n_saturated']}/{budget['n_runs']}", ""]
    if budget["saturated"]:
        lines += ["| run | model | seed | best epoch | cap |",
                  "| --- | --- | --- | --- | --- |"]
        for row in budget["saturated"][:20]:
            lines.append(f"| {row['run_name']} | {row['model']} | {row['seed']} | "
                         f"{row['best_epoch']} | {row['epoch_cap']} |")
        lines.append("")
    if summary["warnings"]:
        lines += ["## Warnings", ""] + [f"- {text}" for text in summary["warnings"]] + [""]
    return "\n".join(lines) + "\n"


def write_summary(summary: dict, out_dir, name: str = "seed_aggregate") -> dict:
    """Write ``<name>.json`` and ``<name>.md``; return the paths actually written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}

