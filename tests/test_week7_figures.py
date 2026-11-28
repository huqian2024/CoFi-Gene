"""Week 7 figures: SVG bar/line/case rendering and input validation."""

from __future__ import annotations

import re

import pytest

from cofi_gene.figures import bar_chart, graph_case, line_chart


def _read(path) -> str:
    return path.read_text(encoding="utf-8")


def test_bar_chart_writes_a_grouped_chart_with_error_bars(tmp_path):
    out = tmp_path / "fig1.svg"
    result = bar_chart(["b0", "b1"], [{"label": "channel A",
                                      "means": [0.80, 0.92],
                                      "stds": [0.010, 0.020]}],
                       out, title="upstream AP", ylabel="average_precision")
    svg = _read(out)
    assert result["format"] == "svg" and result["path"] == str(out)
    assert svg.startswith("<?xml") and "<svg" in svg and svg.rstrip().endswith("</svg>")
    for label in ("b0", "b1", "channel A", "upstream AP", "average_precision"):
        assert label in svg, label
    assert svg.count("<rect") >= 3            # background + two bars
    assert svg.count("<line") >= 12           # axes, gridlines and error bars


def test_bar_chart_without_error_bars_still_draws(tmp_path):
    out = tmp_path / "single.svg"
    bar_chart(["b3"], [{"label": "A", "means": [0.9], "stds": [0.0]}], out)
    svg = _read(out)
    assert "b3" in svg and "<svg" in svg


def test_bar_chart_rejects_mismatched_or_missing_input(tmp_path):
    with pytest.raises(ValueError, match="one mean"):
        bar_chart(["b0", "b1"], [{"label": "A", "means": [0.9], "stds": [0.0]}],
                  tmp_path / "x.svg")
    with pytest.raises(ValueError, match="at least one category"):
        bar_chart([], [{"label": "A", "means": [], "stds": []}], tmp_path / "x.svg")
    with pytest.raises(ValueError, match="at least one group"):
        bar_chart(["b0"], [], tmp_path / "x.svg")
    with pytest.raises(ValueError, match="only SVG"):
        bar_chart(["b0"], [{"label": "A", "means": [0.9], "stds": [0.0]}],
                  tmp_path / "x.png")


def test_line_chart_draws_one_polyline_per_series(tmp_path):
    out = tmp_path / "fig2.svg"
    result = line_chart([0.0, 0.1, 0.3, 0.5],
                        [{"label": "b1", "ys": [0.96, 0.96, 0.95, 0.95]},
                         {"label": "b5", "ys": [0.95, 0.95, 0.94, 0.94]}],
                        out, title="robustness", xlabel="missing-edge ratio",
                        ylabel="average_precision")
    svg = _read(out)
    assert result["series"] == ["b1", "b5"]
    assert result["x_values"] == [0.0, 0.1, 0.3, 0.5]
    assert svg.count("<polyline") == 2
    for label in ("b1", "b5", "missing-edge ratio", "0.50"):
        assert label in svg, label
    assert len(re.findall(r"<circle", svg)) >= 8


def test_line_chart_rejects_bad_series(tmp_path):
    with pytest.raises(ValueError, match="one y per x"):
        line_chart([0.0, 0.5], [{"label": "b1", "ys": [0.9]}], tmp_path / "x.svg")
    with pytest.raises(ValueError, match="at least one series"):
        line_chart([0.0, 0.5], [], tmp_path / "x.svg")
    with pytest.raises(ValueError, match="at least one x"):
        line_chart([], [{"label": "b1", "ys": []}], tmp_path / "x.svg")


def _case() -> dict:
    return {
        "channel": "A",
        "alpha": 0.1,
        "margin": 0.5,
        "summary": {"n_up": 2, "n_down": 1},
        "edges": [
            {"u_id": "TP53", "v_id": "MDM2", "weight_before": 0.40,
             "weight_after": 0.52, "delta": 0.12},
            {"u_id": "TP53", "v_id": "BAX", "weight_before": 0.30,
             "weight_after": 0.24, "delta": -0.06},
            {"u_id": "MDM2", "v_id": "BAX", "weight_before": 0.20,
             "weight_after": 0.20, "delta": 0.0},
        ],
    }


def test_graph_case_draws_nodes_edges_and_a_change_panel(tmp_path):
    out = tmp_path / "fig3.svg"
    result = graph_case(_case(), out, title="feedback case B7")
    svg = _read(out)
    assert result["channel"] == "A" and result["edges_drawn"] == 3
    for label in ("TP53", "MDM2", "BAX", "up-weighted", "down-weighted",
                  "top", "feedback case B7"):
        assert label in svg, label
    # edge colours: red for up, blue for down, grey for unchanged
    assert "#C0392B" in svg and "#2471A3" in svg and "#95A5A6" in svg
    assert svg.count("<circle") >= 3


def test_graph_case_limits_nodes_and_edges(tmp_path):
    case = _case()
    result = graph_case(case, tmp_path / "small.svg", max_edges=1, panel_rows=1)
    assert result["edges_drawn"] == 1
    assert len(result["nodes"]) == 2


def test_graph_case_rejects_empty_or_incomplete_edges(tmp_path):
    with pytest.raises(ValueError, match="at least one edge"):
        graph_case({"edges": []}, tmp_path / "x.svg")
    broken = _case()
    broken["edges"][0].pop("delta")
    with pytest.raises(ValueError, match="lacks 'delta'"):
        graph_case(broken, tmp_path / "x.svg")


def test_bar_chart_skips_a_missing_value_instead_of_failing(tmp_path):
    """A single-graph model has no bar for the channel it does not use."""
    out = tmp_path / "holes.svg"
    bar_chart(["b0", "b1"], [{"label": "A", "means": [0.8, 0.9], "stds": [0.01, 0.02]},
                              {"label": "B", "means": [0.7, None], "stds": [0.01, 0.0]}],
              out, title="holes")
    assert "holes" in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="nothing to draw"):
        bar_chart(["b0"], [{"label": "A", "means": [None], "stds": [0.0]}],
                  tmp_path / "empty.svg")
