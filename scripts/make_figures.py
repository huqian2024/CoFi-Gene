#!/usr/bin/env python3
"""Week 7: draw the required figures from run files (no numbers typed by hand).

    python scripts/make_figures.py --kind upstream \
        --input runs/week7/aggregate/seed_aggregate.json \
        --out figures/week7/fig1_upstream_ap.svg
    python scripts/make_figures.py --kind downstream \
        --input runs/week7/downstream_table.json \
        --out figures/week7/fig1_downstream_de.svg
    python scripts/make_figures.py --kind robustness \
        --input runs/m5_1/stage4/stage4_summary.json --channel A \
        --out figures/week7/fig2_robustness_A.svg
    python scripts/make_figures.py --kind feedback \
        --input runs/week7/case_b7_A.json --out figures/week7/fig3_feedback_A.svg

Inputs are the JSON files produced by ``scripts/aggregate_seeds.py``,
``scripts/export_feedback_case.py``, ``scripts/run_robustness.py`` or
``scripts/run_experiments.py``. The script refuses to draw a table that mixes
protocols or dataset versions unless ``--allow-warnings`` is passed explicitly.
SVG output is dependency-free; convert to PNG/PDF with Inkscape, ``rsvg-convert``
or PowerPoint's "Save as Picture".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without installation
    sys.path.insert(0, str(ROOT / "src"))

from cofi_gene.figures import bar_chart, graph_case, line_chart, load_json  # noqa: E402

KINDS = ("upstream", "downstream", "robustness", "feedback")


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upstream_bars(payload: dict, channel: str | None = None):
    """Aggregate summary -> (categories, groups, ylabel) for a bar chart."""
    models = payload.get("models") or {}
    if not models:
        raise ValueError("upstream input has no 'models' block; run aggregate_seeds.py first")
    channels = sorted({name for entry in models.values() for name in entry})
    if channel is not None:
        if channel not in channels:
            raise ValueError(f"channel {channel!r} not in the summary; available: {channels}")
        channels = [channel]
    categories = sorted(models)
    groups = []
    for name in channels:
        # A single-graph model (B1/B2) legitimately has no bar for the other channel;
        # ``None`` keeps that hole visible instead of silently dropping the channel.
        means = [_as_float(models[model].get(name, {}).get("mean")) for model in categories]
        stds = [_as_float(models[model].get(name, {}).get("std")) or 0.0
                for model in categories]
        groups.append({"label": f"channel {name}", "means": means, "stds": stds})
    return categories, groups, f"{payload.get('metric', 'average_precision')} (mean ± std)"


def downstream_bars(payload: dict, channel: str | None = None):
    """Downstream table -> (categories, groups, ylabel).

    Accepted shapes:
      ``{"metric": "...", "models": {"b0": {"mean": .., "std": .., "n": 3}, ...}}``
      ``{"metric": "...", "models": {"b0": {"A": {"mean": .., "std": ..}}, ...}}``
    """
    models = payload.get("models") or {}
    if not models:
        raise ValueError("downstream input has no 'models' block")
    categories = sorted(models)
    metric = payload.get("metric", "macro_target_ap")
    flat = all(isinstance(models[model].get("mean"), (int, float)) for model in categories)
    if flat:
        means = [float(models[model]["mean"]) for model in categories]
        stds = [float(models[model].get("std") or 0.0) for model in categories]
        return categories, [{"label": str(metric), "means": means, "stds": stds}], metric
    channels = sorted({name for entry in models.values() for name in entry
                       if isinstance(entry.get(name), dict)})
    if channel is not None:
        if channel not in channels:
            raise ValueError(f"channel/metric {channel!r} not in the table; "
                             f"available: {channels}")
        channels = [channel]
    groups = []
    for name in channels:
        means = [float(models[model][name]["mean"]) for model in categories]
        stds = [float(models[model][name].get("std") or 0.0) for model in categories]
        groups.append({"label": name, "means": means, "stds": stds})
    return categories, groups, metric
def robustness_series(payload: dict, metric: str = "average_precision",
                      channel: str = "A"):
    """Robustness summary -> (x values, series for line_chart, dropped ratios).

    Format A: ``run_robustness.py`` summary (``table[metric][channel][model][ratio]``).
    Format B: ``run_experiments.py`` stage summary (``rows`` with model/ratio/channels).
    Only ratios present for **every** series are kept, so no line invents a point.
    """
    per_model: dict[str, dict[float, float]] = {}
    if "table" in payload:
        table = payload["table"].get(metric, {}).get(channel, {})
        if not table:
            raise ValueError(f"robustness summary has no table[{metric}][{channel}]")
        for model, values in table.items():
            per_model[model] = {float(ratio): float(value)
                                for ratio, value in values.items()
                                if value is not None}
    elif "rows" in payload:
        for row in payload["rows"]:
            if row.get("status") not in (None, "complete"):
                continue
            value = _as_float((row.get("channels") or {}).get(channel))
            if value is None or row.get("ratio") is None:
                continue
            per_model.setdefault(str(row["model"]), {})[float(row["ratio"])] = value
    else:
        raise ValueError("robustness input needs a 'table' (run_robustness) or "
                         "'rows' (run_experiments) block")
    if not per_model:
        raise ValueError(f"no usable values for channel {channel!r}")
    shared = sorted(set.intersection(*(set(values) for values in per_model.values())))
    dropped = sorted(set.union(*(set(values) for values in per_model.values()))
                     - set(shared))
    if len(shared) < 2:
        raise ValueError("fewer than two ratios are shared by every model; cannot draw "
                         "a curve without inventing points")
    series = [{"label": model, "ys": [per_model[model][ratio] for ratio in shared]}
              for model in sorted(per_model)]
    return shared, series, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--input", required=True, help="JSON file described above")
    parser.add_argument("--out", required=True, help="output .svg path")
    parser.add_argument("--channel", default=None,
                        help="channel for upstream/robustness (default: every channel / A)")
    parser.add_argument("--metric", default="average_precision",
                        help="metric key for robustness tables")
    parser.add_argument("--title", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--max-edges", type=int, default=16,
                        help="feedback case: edges drawn")
    parser.add_argument("--allow-warnings", action="store_true",
                        help="draw even if the summary reports mixed datasets/protocols")
    args = parser.parse_args(argv)

    payload = load_json(args.input)
    warnings = payload.get("warnings") or []
    integrity = payload.get("integrity")
    blocking = bool(integrity.get("blocking")) if isinstance(integrity, dict) \
        else bool(warnings)
    if warnings:
        print("[figures] summary warnings:")
        for text in warnings:
            print(f"[figures]   - {text}")
    if blocking and not args.allow_warnings:
        print("[figures] refusing to draw a figure from a summary that mixes datasets/"
              "protocols or repeats a seed; split the table or pass --allow-warnings",
              file=sys.stderr)
        return 1

    if args.kind == "upstream":
        categories, groups, ylabel = upstream_bars(payload, args.channel)
        result = bar_chart(categories, groups, args.out,
                           title=args.title or "B0-B7 upstream link prediction",
                           ylabel=ylabel, note=args.note)
    elif args.kind == "downstream":
        categories, groups, ylabel = downstream_bars(payload, args.channel)
        result = bar_chart(categories, groups, args.out,
                           title=args.title or "B0-B7 downstream DE prediction",
                           ylabel=ylabel, note=args.note)
    elif args.kind == "robustness":
        channel = args.channel or "A"
        xs, series, dropped = robustness_series(payload, args.metric, channel)
        if dropped:
            print(f"[figures] dropped ratios not shared by every model: {dropped}")
        result = line_chart(xs, series, args.out,
                            title=args.title or f"Robustness (channel {channel})",
                            xlabel="missing-edge ratio", ylabel=args.metric,
                            note=args.note, x_fmt="{:.2f}")
    else:
        case = payload.get("case", payload)
        result = graph_case(case, args.out,
                            title=args.title or "Graph feedback case",
                            note=args.note, max_edges=args.max_edges)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[figures] wrote {result['path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

