"""Streamlit dashboard for tokbench runs.

Relationship to the HTML report
-------------------------------
``tokbench.render`` produces the *shareable* artefact: one self-contained file,
no runtime, no interactivity.  This panel is the opposite tool for the same
data — a local exploration surface with filters, what-if sliders and plotly
hover.  They share exactly one thing: :class:`~tokbench.report_schema.
ReportPayload`.  Neither imports the other, so the HTML report keeps working
if streamlit is not installed and the panel keeps working if the SVG renderer
is being refactored.

Design decisions
----------------
* **Loading prefers ``report.json``.**  A finished run already has a validated
  payload on disk; re-deriving it is slower and could disagree with the file
  someone else is reading.  Only when it is missing do we call
  ``tokbench.analysis.report.build_report`` — imported lazily, so this module
  still imports cleanly while the analysis layer is mid-build.
* **Caching is keyed on ``(path, mtime, size)``, not on the path alone.**
  Re-running a report in place must invalidate the cache, and ``st.cache_data``
  hashes arguments, so the mtime has to be part of the key.  ``load_payload``
  takes the *key* explicitly to make that contract visible.
* **Figure builders are pure functions of the payload.**  They take a payload
  and return a plotly figure; no Streamlit calls inside.  That is what makes
  them testable without a server (see ``tests/test_render.py``), and it keeps
  the tab bodies to layout only.
* **The what-if slider re-prices, it does not re-run.**  Cache hit rate is a
  cost-model input, so we can show its effect by interpolating the payload's
  own sensitivity curve — the same numbers the HTML report draws.

Run it with::

    streamlit run tokbench/panel/app.py --server.headless true
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ``streamlit run app.py`` executes this file as a bare script (``__main__``), with
# no package context — so ``from ..models import ...`` raises
# "attempted relative import with no known parent package".  Streamlit also does
# not put the project root on ``sys.path``.  Bootstrap both here, before any
# package-relative import, so the panel works identically whether it is launched
# by ``streamlit run``, by ``tokbench panel``, or imported by the test suite.
if __package__ in (None, ""):  # launched as a script
    _here = Path(__file__).resolve()
    _project_root = _here.parents[2]  # .../<project>/tokbench/panel/app.py
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from tokbench.pricing import DEFAULT_PRICE_BOOK, PriceBook, load_price_book
    from tokbench.report_schema import (
        CacheSensitivity,
        ParetoFrontier,
        ReportPayload,
        RetryAnalysis,
    )
    from tokbench.store import RunStore, discover_runs, resolve_run_dir
else:  # imported as a module (tests, notebooks, `python -m`)
    from ..pricing import DEFAULT_PRICE_BOOK, PriceBook, load_price_book
    from ..report_schema import (
        CacheSensitivity,
        ParetoFrontier,
        ReportPayload,
        RetryAnalysis,
    )
    from ..store import RunStore, discover_runs, resolve_run_dir

__all__ = [
    "main",
    "load_payload",
    "main_payload",
    "run_choices",
    "build_pareto_figure",
    "build_series_figure",
    "build_stacked_figure",
    "build_retry_figure",
    "build_cache_figure",
    "build_delta_figure",
    "leaderboard_frame",
    "retry_frame",
    "stacked_frame",
    "series_frame",
    "deltas_frame",
    "raw_frames",
    "cache_what_if",
]

# --------------------------------------------------------------------------- #
# Palette (kept in sync with tokbench.render.html's semantic colours)
# --------------------------------------------------------------------------- #
ACCENT = "#22d3ee"  # cyan  — primary / cheap / frontier
ACCENT_DIM = "#0e7490"
AMBER = "#fbbf24"  # amber — cost / caution / knee
MAGENTA = "#f472b6"  # magenta — retry waste
GOOD = "#34d399"
INK_2 = "#93a1b0"
INK_3 = "#5b6875"
GRID = "#1b2430"
PANEL = "#0b0f14"

#: Plotly layout shared by every figure so the dashboard looks like one product.
_BASE_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": 'ui-monospace, "SFMono-Regular", Consolas, monospace',
        "size": 12,
        "color": INK_2,
    },
    "margin": {"l": 64, "r": 24, "t": 48, "b": 56},
    "hoverlabel": {
        "bgcolor": "#121821",
        "bordercolor": GRID,
        "font": {"color": "#e6edf3", "size": 12},
    },
    "xaxis": {
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "linecolor": GRID,
        "tickcolor": GRID,
        "automargin": True,
    },
    "yaxis": {
        "gridcolor": GRID,
        "zerolinecolor": GRID,
        "linecolor": GRID,
        "tickcolor": GRID,
        "automargin": True,
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "x": 0,
        "bgcolor": "rgba(0,0,0,0)",
    },
    "colorway": [ACCENT, AMBER, MAGENTA, GOOD, ACCENT_DIM],
}


def _layout(**overrides: Any) -> dict[str, Any]:
    """Merge axis-specific overrides into the shared layout without mutating it."""
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _BASE_LAYOUT.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout


# --------------------------------------------------------------------------- #
# Formatting (mirrors the HTML renderer so both surfaces read the same)
# --------------------------------------------------------------------------- #
def _usd(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"${value:,.{digits}f}" if abs(value) >= 0.01 else f"${value:.6f}"


def _tok(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _stat_key(path: Path) -> tuple[str, int, int]:
    """Cache key for a file: ``(path, mtime_ns, size)``.

    ``st.cache_data`` hashes the *arguments*, so passing this tuple is what
    makes an in-place ``report.json`` rewrite invalidate the cached payload.
    """
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _run_store(run_dir: str | Path) -> RunStore:
    """Resolve a path to a :class:`RunStore` (accepts a run dir or a parent dir).

    ``tokbench.store.resolve_run_dir`` treats a directory as a run only when it
    contains ``manifest.json`` or ``calls.jsonl``.  A report-only directory (a
    hand-copied ``report.json``, or a run whose manifest was pruned) is still
    perfectly viewable, so that case is short-circuited here instead of being
    rejected as "no runs found".
    """
    path = Path(run_dir)
    if (path / "report.json").exists():
        return RunStore(path)
    return resolve_run_dir(path)


def load_payload(run_dir: str | Path) -> ReportPayload:
    """Load the payload for a run directory, preferring its cached ``report.json``.

    Order of preference:

    1. ``<run_dir>/report.json`` — already computed and validated; this is what
       a finished run leaves behind, and reading it guarantees the panel shows
       exactly what the CLI wrote.
    2. ``tokbench.analysis.report.build_report`` — computed on the fly for a run
       that has not been reported yet.  Imported lazily on purpose: the analysis
       layer is a separate moving part, and a missing/broken import here must
       produce a clear error rather than breaking panel startup.

    Args:
        run_dir: A run directory, or a parent directory containing runs (the
            newest is used).

    Returns:
        The validated :class:`ReportPayload`.

    Raises:
        FileNotFoundError: No run and no report could be found.
        RuntimeError: The run exists but neither ``report.json`` nor the
            analysis layer could produce a payload.
    """
    store = _run_store(run_dir)
    report_path = store.report_json_path
    if report_path.exists():
        return ReportPayload.model_validate_json(report_path.read_text(encoding="utf-8"))

    # Lazy import: keeps this module importable while analysis is mid-build.
    try:
        from ..analysis.report import build_report
    except Exception as exc:  # pragma: no cover - depends on parallel work
        raise RuntimeError(
            f"{store.root} has no report.json and the analysis layer is "
            f"unavailable ({exc}). Run `tokbench report --run {store.root}` first."
        ) from exc

    payload = build_report(store, price_book=_default_price_book())
    if not isinstance(payload, ReportPayload):
        raise RuntimeError(
            f"build_report returned {type(payload).__name__}, expected ReportPayload"
        )
    return payload


def _default_price_book() -> PriceBook:
    """Price book from ``TOKBENCH_PRICES`` when set, else the bundled default."""
    override = os.environ.get("TOKBENCH_PRICES")
    if override:
        try:
            return load_price_book(override)
        except Exception:
            return DEFAULT_PRICE_BOOK
    return DEFAULT_PRICE_BOOK


@st.cache_data(show_spinner="loading run…", max_entries=8)
def _cached_payload(key: tuple[str, int, int], run_dir: str) -> ReportPayload:
    """Cached wrapper around :func:`load_payload`.

    The cache key is the ``(path, mtime, size)`` tuple, so editing
    ``report.json`` in place is picked up on the next interaction.
    """
    return load_payload(run_dir)


def main_payload(run_dir: str | Path) -> ReportPayload:
    """Public, cached payload accessor used by the UI and by tests.

    Resolves the run, computes the stat key, and defers to the cached loader.
    Exposed as a named function (rather than inlined in ``main``) so the caching
    contract is testable without a running Streamlit server.
    """
    store = _run_store(run_dir)
    report_path = store.report_json_path
    key = _stat_key(report_path if report_path.exists() else store.root)
    return _cached_payload(key, str(store.root))


def run_choices(base_dir: str | Path) -> list[tuple[str, str]]:
    """Discover runs under *base_dir* as ``(label, path)`` pairs, newest first."""
    choices: list[tuple[str, str]] = []
    for store in discover_runs(base_dir):
        manifest = store.read_manifest()
        if manifest is not None:
            label = (
                f"{manifest.run_id} · {manifest.experiment_name} · "
                f"{_tok(manifest.total_usage.total_tokens)} tok · "
                f"{_usd(manifest.total_cost_usd, 2)}"
            )
        else:
            label = store.run_id
        choices.append((label, str(store.root)))
    return choices


# --------------------------------------------------------------------------- #
# Frame builders (pandas)
# --------------------------------------------------------------------------- #
def leaderboard_frame(payload: ReportPayload) -> pd.DataFrame:
    """Leaderboard as a display-ready frame with stable, human column names."""
    rows = [
        {
            "rank": r.rank,
            "model": r.model_id,
            "strategy": r.strategy_id,
            "provider": r.provider,
            "cells": r.cells,
            "success": r.success_rate,
            "tok/task": r.billable_tokens_per_task,
            "quality": r.mean_quality,
            "$/task": r.cost_per_task_usd,
            "total $": r.cost_usd,
            "cache": r.cache_hit_rate,
            "latency ms": r.mean_latency_ms,
            "score": r.efficiency_score,
        }
        for r in sorted(payload.leaderboard, key=lambda x: x.rank or 10**6)
    ]
    return pd.DataFrame(rows)


def retry_frame(payload: ReportPayload) -> pd.DataFrame:
    """Retry rows sorted by wasted dollars, with a derived cost column."""
    rows = [
        {
            "model": r.model_id,
            "strategy": r.strategy_id,
            "cells": r.cells,
            "retried": r.retried_cells,
            "retry rate": r.retry_rate,
            "extra attempts": r.extra_attempts,
            "retry tokens": r.retry_usage.total_tokens,
            "retry $": r.retry_cost_usd,
            "total $": r.total_cost_usd,
            "share": r.retry_cost_share,
            "top error": r.dominant_error_kind or "—",
        }
        for r in sorted(payload.retry.rows, key=lambda x: -x.retry_cost_usd)
    ]
    return pd.DataFrame(rows)


def stacked_frame(payload: ReportPayload) -> pd.DataFrame:
    """Token-type decomposition, sorted by billable tokens per task."""
    rows = [
        {
            "model": r.model_id,
            "strategy": r.strategy_id,
            "uncached input": r.uncached_input_tokens,
            "cache read": r.cache_read_tokens,
            "cache write": r.cache_write_tokens,
            "output": r.output_tokens,
            "reasoning": r.reasoning_tokens,
            "cost $": r.cost_usd,
            "cells": r.cells,
            "pair": f"{r.model_id} · {r.strategy_id}",
            "billable/task": ((r.uncached_input_tokens + r.output_tokens) / max(r.cells, 1)),
        }
        for r in payload.stacked
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("billable/task", ascending=False).reset_index(drop=True)
    return frame


def series_frame(payload: ReportPayload) -> pd.DataFrame:
    """Cumulative series as a frame ordered by cell index."""
    rows = [
        {
            "index": p.index,
            "time": p.t,
            "model": p.model_id,
            "strategy": p.strategy_id,
            "tokens": p.tokens,
            "billable": p.billable_tokens,
            "cost $": p.cost_usd,
            "cum tokens": p.cumulative_tokens,
            "cum billable": p.cumulative_billable_tokens,
            "cum cost $": p.cumulative_cost_usd,
            "cum cache": p.cumulative_cache_hit_rate,
        }
        for p in sorted(payload.series, key=lambda x: x.index)
    ]
    return pd.DataFrame(rows)


def deltas_frame(payload: ReportPayload) -> pd.DataFrame:
    """Strategy-ablation deltas with percentage columns kept as fractions."""
    rows = [
        {
            "strategy": d.strategy_id,
            "model": d.model_id or "all",
            "baseline": d.baseline_strategy,
            "tok/task": d.strategy_billable_per_task,
            "Δ tokens": d.token_delta_pct,
            "$/task": d.strategy_cost_per_task_usd,
            "Δ cost": d.cost_delta_pct,
            "quality": d.quality,
            "Δ quality": d.quality_delta,
            "verdict": d.verdict or "—",
        }
        for d in payload.strategy_deltas
    ]
    return pd.DataFrame(rows)


def raw_frames(payload: ReportPayload) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw-tab frames: ``(calls, results)`` flattened from ``payload.raw``.

    The payload's ``raw`` dict is analysis-defined, so this reads it
    defensively: anything list-of-dicts becomes a frame, anything else becomes a
    single-row frame.  Both frames are given the same columns so the two tabs
    line up and the tests can assert on a stable shape.
    """
    raw = payload.raw or {}
    calls = raw.get("calls")
    results = raw.get("results") or raw.get("cells")

    def _to_frame(value: Any) -> pd.DataFrame:
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return pd.json_normalize(value)
        if isinstance(value, dict):
            return pd.json_normalize([value])
        return pd.DataFrame(columns=["field", "value"])

    calls_frame = _to_frame(calls)
    results_frame = _to_frame(results)
    if calls_frame.empty and results_frame.empty:
        # Fall back to the scalar diagnostics the analysis layer always emits,
        # so the Raw tab is never a blank pane.
        #
        # Every value is stringified here on purpose: mixing ints and JSON
        # strings in one column makes the frame ``object``-dtype, and
        # ``st.dataframe`` serialises through pyarrow, which then raises
        # ``ArrowTypeError: Expected bytes, got a 'int' object``.  A diagnostics
        # pane has no use for typed numerics, so uniform text is both safer and
        # perfectly readable.
        scalars = {
            k: v for k, v in raw.items() if not isinstance(v, (list, dict)) or isinstance(v, dict)
        }
        rows = [
            {
                "field": str(k),
                "value": (json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)),
            }
            for k, v in scalars.items()
        ]
        frame = (
            pd.DataFrame(rows, columns=["field", "value"], dtype="string")
            if rows
            else pd.DataFrame(columns=["field", "value"], dtype="string")
        )
        return frame, frame
    # Align columns so callers can rely on identical shapes.
    columns = sorted(set(calls_frame.columns) | set(results_frame.columns))
    return (
        calls_frame.reindex(columns=columns),
        results_frame.reindex(columns=columns),
    )


# --------------------------------------------------------------------------- #
# Figure builders (pure: payload in, plotly figure out)
# --------------------------------------------------------------------------- #
def build_pareto_figure(payload: ReportPayload) -> go.Figure:
    """Cost-per-task vs quality scatter with the frontier drawn as a line.

    Mirrors the HTML hero chart: frontier in cyan and connected, dominated
    points dimmed grey, the knee marked with an amber diamond, and the baseline
    ringed.  Hover carries the full precision the axis cannot.
    """
    pareto: ParetoFrontier = payload.pareto
    fig = go.Figure()
    points = list(pareto.points)
    if not points:
        fig.update_layout(**_layout(title="No Pareto data"))
        return fig

    frontier = [p for p in points if p.on_frontier]
    dominated = [p for p in points if not p.on_frontier]
    frontier.sort(key=lambda p: (p.cost_per_task_usd, p.quality))

    if dominated:
        fig.add_trace(
            go.Scatter(
                x=[p.cost_per_task_usd for p in dominated],
                y=[p.quality for p in dominated],
                mode="markers",
                name="dominated",
                marker={"color": INK_3, "size": 9, "opacity": 0.6},
                text=[p.label for p in dominated],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "cost/task $%{x:.6f}<br>quality %{y:.3f}<extra>dominated</extra>"
                ),
            )
        )

    if frontier:
        fig.add_trace(
            go.Scatter(
                x=[p.cost_per_task_usd for p in frontier],
                y=[p.quality for p in frontier],
                mode="lines+markers+text",
                name="pareto frontier",
                line={"color": ACCENT, "width": 2.4},
                marker={"color": ACCENT, "size": 12, "line": {"color": PANEL, "width": 1.5}},
                text=[p.label for p in frontier],
                textposition="top center",
                textfont={"size": 10, "color": INK_2},
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "cost/task $%{x:.6f}<br>quality %{y:.3f}<extra>frontier</extra>"
                ),
            )
        )

        # Knee: furthest frontier point from the cheapest→best chord.
        if len(frontier) >= 3:
            ax, ay = frontier[0].cost_per_task_usd, frontier[0].quality
            bx, by = frontier[-1].cost_per_task_usd, frontier[-1].quality
            norm = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            if norm > 0:
                best_i, best_d = 1, -1.0
                for i in range(1, len(frontier) - 1):
                    px, py = frontier[i].cost_per_task_usd, frontier[i].quality
                    dist = abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / norm
                    if dist > best_d:
                        best_i, best_d = i, dist
                knee = frontier[best_i]
                fig.add_trace(
                    go.Scatter(
                        x=[knee.cost_per_task_usd],
                        y=[knee.quality],
                        mode="markers",
                        name="knee",
                        marker={
                            "color": AMBER,
                            "size": 15,
                            "symbol": "diamond",
                            "line": {"color": PANEL, "width": 1.5},
                        },
                        hovertemplate=(
                            f"<b>knee: {knee.label}</b><br>"
                            "cost/task $%{x:.6f}<br>quality %{y:.3f}<extra></extra>"
                        ),
                    )
                )

    baseline = next(
        (p for p in points if pareto.baseline_label and p.label == pareto.baseline_label),
        None,
    )
    if baseline is not None:
        fig.add_trace(
            go.Scatter(
                x=[baseline.cost_per_task_usd],
                y=[baseline.quality],
                mode="markers",
                name="baseline",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 17,
                    "line": {"color": INK_2, "width": 1.6},
                },
                hovertemplate=f"<b>baseline: {baseline.label}</b><extra></extra>",
            )
        )

    fig.update_layout(
        **_layout(
            title="Pareto frontier — cost per task vs mean quality",
            xaxis={"title": "cost per task (USD)", "tickformat": "$.4f"},
            yaxis={"title": "mean quality (0–1)"},
            hovermode="closest",
        )
    )
    return fig


def build_series_figure(payload: ReportPayload) -> go.Figure:
    """Cumulative tokens (area) and cumulative cost (line, right axis)."""
    fig = go.Figure()
    frame = series_frame(payload)
    if frame.empty:
        fig.update_layout(**_layout(title="No series data"))
        return fig

    fig.add_trace(
        go.Scatter(
            x=frame["index"],
            y=frame["cum tokens"],
            name="cumulative tokens",
            mode="lines",
            line={"color": ACCENT, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.16)",
            hovertemplate="cell %{x}<br>%{y:,.0f} tokens<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["index"],
            y=frame["cum cost $"],
            name="cumulative cost",
            mode="lines",
            line={"color": AMBER, "width": 1.9},
            yaxis="y2",
            hovertemplate="cell %{x}<br>$%{y:,.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            title="Cumulative tokens and cost",
            xaxis={"title": "cell index"},
            yaxis={"title": "cumulative tokens"},
            yaxis2={
                "title": "cumulative cost (USD)",
                "overlaying": "y",
                "side": "right",
                "gridcolor": "rgba(0,0,0,0)",
                "automargin": True,
            },
        )
    )
    return fig


def build_stacked_figure(payload: ReportPayload) -> go.Figure:
    """Token-type decomposition as stacked bars, ranked by billable tokens/task."""
    fig = go.Figure()
    frame = stacked_frame(payload)
    if frame.empty:
        fig.update_layout(**_layout(title="No token decomposition"))
        return fig

    shown = frame.head(14)
    series = [
        ("uncached input", ACCENT),
        ("cache read", ACCENT_DIM),
        ("cache write", "rgba(34,211,238,0.45)"),
        ("output", AMBER),
        ("reasoning", MAGENTA),
    ]
    for column, colour in series:
        if column not in shown or shown[column].sum() <= 0:
            continue
        fig.add_trace(
            go.Bar(
                x=shown["pair"],
                y=shown[column],
                name=column,
                marker_color=colour,
                hovertemplate="%{x}<br>" + column + ": %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout(
            title="Token-type decomposition",
            barmode="stack",
            xaxis={"title": "", "tickangle": -40},
            yaxis={"title": "tokens"},
        )
    )
    return fig


def build_retry_figure(payload: ReportPayload) -> go.Figure:
    """Retry spend per pair as horizontal bars, worst first."""
    fig = go.Figure()
    rows = sorted(payload.retry.rows, key=lambda r: -r.retry_cost_usd)[:14]
    if not rows:
        fig.update_layout(**_layout(title="No retries recorded"))
        return fig

    labels = [f"{r.model_id} · {r.strategy_id}" for r in rows]
    first_attempt = [max(r.total_cost_usd - r.retry_cost_usd, 0.0) for r in rows]
    wasted = [r.retry_cost_usd for r in rows]

    fig.add_trace(
        go.Bar(
            y=labels,
            x=first_attempt,
            name="first-attempt spend",
            orientation="h",
            marker_color=ACCENT_DIM,
            hovertemplate="%{y}<br>$%{x:,.4f}<extra>first attempt</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=wasted,
            name="retry spend (waste)",
            orientation="h",
            marker_color=MAGENTA,
            hovertemplate="%{y}<br>$%{x:,.4f}<extra>retry waste</extra>",
        )
    )
    fig.update_layout(
        **_layout(
            title=(
                f"Retry spend by pair — {_pct(payload.retry.overall_retry_cost_share)} "
                f"of total budget"
            ),
            barmode="stack",
            xaxis={"title": "cost (USD)", "tickformat": "$.4f"},
            yaxis={"title": "", "autorange": "reversed"},
        )
    )
    return fig


def build_cache_figure(
    payload: ReportPayload, *, what_if_hit_rate: float | None = None
) -> go.Figure:
    """Projected cost vs cache hit rate, with observed and optional what-if marks."""
    fig = go.Figure()
    cache: CacheSensitivity = payload.cache
    curve = sorted(cache.curve, key=lambda p: p.cache_hit_rate)
    if not curve:
        fig.update_layout(**_layout(title="No cache sensitivity data"))
        return fig

    hs = [p.cache_hit_rate for p in curve]
    costs = [p.projected_cost_usd for p in curve]
    fig.add_trace(
        go.Scatter(
            x=hs,
            y=costs,
            mode="lines",
            name="projected cost",
            line={"color": ACCENT, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.07)",
            hovertemplate="hit rate %{x:.1%}<br>$%{y:,.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[cache.observed_hit_rate],
            y=[cache.observed_cost_usd],
            mode="markers",
            name="observed",
            marker={"color": AMBER, "size": 14, "line": {"color": PANEL, "width": 1.5}},
            hovertemplate=("observed hit rate %{x:.1%}<br>$%{y:,.4f}<extra></extra>"),
        )
    )
    if what_if_hit_rate is not None:
        projected = _interp_cost(curve, what_if_hit_rate)
        fig.add_trace(
            go.Scatter(
                x=[what_if_hit_rate],
                y=[projected],
                mode="markers+text",
                name="what-if",
                marker={"color": MAGENTA, "size": 13, "symbol": "diamond"},
                text=[f"{what_if_hit_rate:.0%} → {_usd(projected, 2)}"],
                textposition="top center",
                textfont={"size": 11, "color": MAGENTA},
                hovertemplate="what-if %{x:.1%}<br>$%{y:,.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout(
            title="Cache sensitivity — projected cost vs hit rate",
            xaxis={"title": "cache hit rate", "tickformat": ".0%"},
            yaxis={"title": "projected total cost (USD)", "tickformat": "$.2f"},
        )
    )
    return fig


def _interp_cost(curve: Sequence[Any], h: float) -> float:
    """Linear interpolation of projected cost at hit rate *h*.

    The sensitivity curve is linear by construction (METHODOLOGY §7), so
    interpolating between sampled points reproduces the model's own answer.
    """
    pts = sorted(((p.cache_hit_rate, p.projected_cost_usd) for p in curve), key=lambda t: t[0])
    if not pts:
        return 0.0
    if h <= pts[0][0]:
        return pts[0][1]
    if h >= pts[-1][0]:
        return pts[-1][1]
    for (h0, c0), (h1, c1) in zip(pts, pts[1:], strict=False):
        if h0 <= h <= h1:
            if h1 == h0:
                return c0
            return c0 + (h - h0) / (h1 - h0) * (c1 - c0)
    return pts[-1][1]


def cache_what_if(payload: ReportPayload, hit_rate: float) -> dict[str, float]:
    """Re-price the run at a hypothetical cache hit rate.

    Returns the projected total and per-task cost plus the delta against the
    observed point, so the UI can show "you would save $X" without duplicating
    the interpolation logic.
    """
    cache = payload.cache
    projected = _interp_cost(cache.curve, hit_rate)
    observed = cache.observed_cost_usd or _interp_cost(cache.curve, cache.observed_hit_rate)
    tasks = payload.summary.completed_cells or 1
    return {
        "hit_rate": hit_rate,
        "projected_cost_usd": projected,
        "projected_cost_per_task_usd": projected / tasks,
        "delta_vs_observed_usd": projected - observed,
        "savings_vs_zero_usd": cache.zero_cache_cost_usd - projected,
    }


def build_delta_figure(payload: ReportPayload) -> go.Figure:
    """Cost delta per strategy versus the baseline, one bar group per strategy."""
    fig = go.Figure()
    frame = deltas_frame(payload)
    if frame.empty:
        fig.update_layout(**_layout(title="No strategy deltas (single-strategy run?)"))
        return fig

    grouped = (
        frame.groupby("strategy", as_index=False)
        .agg({"Δ cost": "mean", "Δ tokens": "mean", "Δ quality": "mean"})
        .sort_values("Δ cost")
    )
    fig.add_trace(
        go.Bar(
            x=grouped["strategy"],
            y=grouped["Δ cost"],
            name="mean Δ cost",
            marker_color=[AMBER if v > 0 else GOOD for v in grouped["Δ cost"]],
            hovertemplate="%{x}<br>Δ cost %{y:+.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["strategy"],
            y=grouped["Δ tokens"],
            name="mean Δ tokens",
            mode="markers",
            marker={"color": MAGENTA, "size": 12, "symbol": "diamond"},
            hovertemplate="%{x}<br>Δ tokens %{y:+.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            title="Strategy ablation vs baseline",
            xaxis={"title": ""},
            yaxis={"title": "delta vs baseline", "tickformat": ".0%"},
        )
    )
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _kpi_row(payload: ReportPayload) -> None:
    """Six headline metrics, mirroring the HTML report's KPI cards."""
    s = payload.summary
    cols = st.columns(6)
    cols[0].metric("total spend", _usd(s.total_cost_usd, 2))
    cols[1].metric(
        "retry waste",
        _pct(payload.retry.overall_retry_cost_share),
        help="Share of total spend burned on attempts after the first.",
    )
    cols[2].metric("billable tokens", _tok(s.total_billable_tokens))
    cols[3].metric("cache hit rate", _pct(s.cache_hit_rate))
    best = max((p.quality for p in payload.pareto.points), default=0.0)
    cols[4].metric("best quality", f"{best:.3f}")
    cols[5].metric("cells", f"{s.completed_cells:,}/{s.planned_cells:,}")


def _sidebar(base_dir: str) -> tuple[ReportPayload | None, dict[str, Any]]:
    """Render the sidebar and return ``(payload, options)``.

    Filters are applied here and passed down as plain data, so every tab body
    stays a pure function of ``(payload, options)``.
    """
    st.sidebar.title("tokbench")
    st.sidebar.caption("LLM token & cost benchmark dashboard")

    runs_dir = st.sidebar.text_input("runs directory", value=base_dir, key="runs_dir")
    choices = run_choices(runs_dir)
    if not choices:
        st.sidebar.error(f"no runs found under `{runs_dir}`")
        return None, {}

    labels = [label for label, _ in choices]
    picked = st.sidebar.selectbox("run", labels, index=0, key="run_pick")
    run_dir = dict(choices)[picked]

    try:
        payload = main_payload(run_dir)
    except Exception as exc:
        st.sidebar.error(f"could not load run: {exc}")
        return None, {}

    st.sidebar.divider()
    st.sidebar.subheader("filters")

    all_models = sorted({r.model_id for r in payload.leaderboard})
    all_strategies = sorted({r.strategy_id for r in payload.leaderboard})
    models = st.sidebar.multiselect("models", all_models, default=all_models)
    strategies = st.sidebar.multiselect("strategies", all_strategies, default=all_strategies)
    baseline = st.sidebar.selectbox(
        "baseline strategy",
        all_strategies,
        index=all_strategies.index("baseline") if "baseline" in all_strategies else 0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("cache what-if")
    what_if = st.sidebar.slider(
        "hypothetical cache hit rate",
        min_value=0.0,
        max_value=1.0,
        value=float(min(max(payload.cache.observed_hit_rate, 0.0), 1.0)),
        step=0.01,
        format="%.0f%%",
        help="Re-prices the run along its own cache sensitivity curve.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("price book")
    price_note = st.sidebar.text_input(
        "override YAML path", value=os.environ.get("TOKBENCH_PRICES", ""), key="prices"
    )
    if price_note:
        st.sidebar.caption("Re-pricing a historical run needs a full analysis pass.")

    options: dict[str, Any] = {
        "run_dir": run_dir,
        "models": models,
        "strategies": strategies,
        "baseline": baseline,
        "what_if": what_if,
        "prices": price_note,
    }
    return payload, options


def _filter_rows(
    rows: Iterable[Any], models: Sequence[str], strategies: Sequence[str]
) -> list[Any]:
    """Keep rows whose model and strategy survive the sidebar filters.

    An empty multiselect means "no filter" rather than "nothing" — that is the
    behaviour users expect when they clear a filter box, and it avoids a blank
    dashboard from one stray click.
    """
    out = []
    for row in rows:
        if models and row.model_id not in models:
            continue
        if strategies and row.strategy_id not in strategies:
            continue
        out.append(row)
    return out


def _filter_payload(payload: ReportPayload, options: dict[str, Any]) -> ReportPayload:
    """Return a copy of *payload* restricted to the sidebar's model/strategy filters.

    Rebuilt as a new payload (rather than filtering inside each tab) so every
    tab — and the KPI row — agrees on exactly the same subset.
    """
    models = options.get("models") or []
    strategies = options.get("strategies") or []
    if not models and not strategies:
        return payload

    def keep(model_id: str, strategy_id: str) -> bool:
        return (not models or model_id in models) and (not strategies or strategy_id in strategies)

    leaderboard = [r for r in payload.leaderboard if keep(r.model_id, r.strategy_id)]
    points = [p for p in payload.pareto.points if keep(p.model_id, p.strategy_id)]
    frontier = [p for p in points if p.on_frontier]
    frontier.sort(key=lambda p: (p.cost_per_task_usd, p.quality))

    return payload.model_copy(
        update={
            "leaderboard": leaderboard,
            "pareto": payload.pareto.model_copy(
                update={
                    "points": points,
                    "frontier_labels": [p.label for p in frontier],
                }
            ),
            "retry": payload.retry.model_copy(
                update={"rows": [r for r in payload.retry.rows if keep(r.model_id, r.strategy_id)]}
            ),
            "stacked": [r for r in payload.stacked if keep(r.model_id, r.strategy_id)],
            "series": [p for p in payload.series if keep(p.model_id, p.strategy_id)],
            "strategy_deltas": [
                d
                for d in payload.strategy_deltas
                if d.model_id is None or keep(d.model_id, d.strategy_id)
            ],
            "categories": [c for c in payload.categories if keep(c.model_id, c.strategy_id)],
        }
    )


def _overview_tab(payload: ReportPayload) -> None:
    """KPIs, the run's own notes, and the full leaderboard."""
    _kpi_row(payload)
    meta = payload.meta
    st.caption(
        f"run `{meta.run_id}` · experiment `{meta.experiment}` · "
        f"price book `{meta.price_book}` · generated {meta.generated_at:%Y-%m-%d %H:%M} UTC"
    )
    if meta.notes:
        with st.expander("run notes", expanded=False):
            for note in meta.notes:
                st.markdown(f"- {note}")

    st.subheader("token efficiency leaderboard")
    frame = leaderboard_frame(payload)
    if frame.empty:
        st.info("No leaderboard rows for this selection.")
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "tok/task": st.column_config.NumberColumn(format="%.0f"),
            "quality": st.column_config.NumberColumn(format="%.3f"),
            "$/task": st.column_config.NumberColumn(format="$%.6f"),
            "total $": st.column_config.NumberColumn(format="$%.4f"),
            "cache": st.column_config.NumberColumn(format="%.1%%"),
            "score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )


def _pareto_tab(payload: ReportPayload) -> None:
    """The hero chart, plus the frontier's own labels and explanation."""
    pareto = payload.pareto
    if pareto.explanation:
        st.caption(pareto.explanation)
    st.plotly_chart(build_pareto_figure(payload), width="stretch")
    cols = st.columns(4)
    cols[0].metric("frontier points", len(pareto.frontier_labels))
    cols[1].metric("best value", pareto.best_value_label or "—")
    cols[2].metric("best quality", pareto.best_quality_label or "—")
    cols[3].metric("cheapest", pareto.cheapest_label or "—")
    if pareto.frontier_labels:
        st.markdown(
            "**frontier (cheap → expensive):** "
            + " → ".join(f"`{x}`" for x in pareto.frontier_labels)
        )


def _tokens_tab(payload: ReportPayload) -> None:
    """Cumulative time series and the token-type decomposition."""
    st.plotly_chart(build_series_figure(payload), width="stretch")
    st.plotly_chart(build_stacked_figure(payload), width="stretch")
    frame = stacked_frame(payload)
    if not frame.empty:
        with st.expander("token decomposition table"):
            st.dataframe(frame.drop(columns=["pair"]), width="stretch", hide_index=True)


def _retry_tab(payload: ReportPayload) -> None:
    """Retry waste: headline number, per-pair bars, full table, error mix."""
    retry: RetryAnalysis = payload.retry
    cols = st.columns(3)
    cols[0].metric("retry cost share", _pct(retry.overall_retry_cost_share))
    cols[1].metric("retry spend", _usd(retry.total_retry_cost_usd, 4))
    cols[2].metric("retry tokens", _tok(retry.total_retry_tokens))
    if retry.explanation:
        st.caption(retry.explanation)

    st.plotly_chart(build_retry_figure(payload), width="stretch")

    if retry.error_breakdown:
        st.markdown("**failure kinds**")
        err = pd.DataFrame(
            sorted(retry.error_breakdown.items(), key=lambda kv: -kv[1]),
            columns=["error kind", "count"],
        )
        st.dataframe(err, width="stretch", hide_index=True)

    frame = retry_frame(payload)
    if not frame.empty:
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "retry rate": st.column_config.NumberColumn(format="%.1%%"),
                "share": st.column_config.NumberColumn(format="%.1%%"),
                "retry $": st.column_config.NumberColumn(format="$%.4f"),
                "total $": st.column_config.NumberColumn(format="$%.4f"),
            },
        )


def _cache_tab(payload: ReportPayload, options: dict[str, Any]) -> None:
    """Sensitivity curve plus the what-if slider's re-priced numbers."""
    cache = payload.cache
    what_if = float(options.get("what_if", cache.observed_hit_rate))
    if cache.explanation:
        st.caption(cache.explanation)

    result = cache_what_if(payload, what_if)
    cols = st.columns(4)
    cols[0].metric("observed hit rate", _pct(cache.observed_hit_rate))
    cols[1].metric("observed cost", _usd(cache.observed_cost_usd, 2))
    cols[2].metric(
        f"what-if @ {what_if:.0%}",
        _usd(result["projected_cost_usd"], 2),
        delta=f"{result['delta_vs_observed_usd']:+,.4f}",
        delta_color="inverse",
    )
    cols[3].metric("savings vs no cache", _usd(result["savings_vs_zero_usd"], 2))

    st.plotly_chart(build_cache_figure(payload, what_if_hit_rate=what_if), width="stretch")
    cols = st.columns(3)
    cols[0].metric("no cache", _usd(cache.zero_cache_cost_usd, 2))
    cols[1].metric("full cache", _usd(cache.full_cache_cost_usd, 2))
    cols[2].metric("per +10pp", _usd(cache.marginal_usd_per_10pp, 4))


def _strategy_tab(payload: ReportPayload) -> None:
    """Ablation deltas versus the baseline, as a chart and a table."""
    frame = deltas_frame(payload)
    if frame.empty:
        st.info("This run has no strategy deltas (it may use a single strategy).")
        return
    st.plotly_chart(build_delta_figure(payload), width="stretch")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Δ tokens": st.column_config.NumberColumn(format="%+.1%%"),
            "Δ cost": st.column_config.NumberColumn(format="%+.1%%"),
            "Δ quality": st.column_config.NumberColumn(format="%+.3f"),
            "$/task": st.column_config.NumberColumn(format="$%.6f"),
        },
    )


def _raw_tab(payload: ReportPayload) -> None:
    """Raw call/result records with CSV and JSON download buttons."""
    calls, results = raw_frames(payload)
    tab_calls, tab_results, tab_payload = st.tabs(["calls", "results", "payload"])

    with tab_calls:
        st.dataframe(calls, width="stretch", hide_index=True)
        st.download_button(
            "download calls CSV",
            data=calls.to_csv(index=False).encode("utf-8"),
            file_name="calls.csv",
            mime="text/csv",
            key="dl_calls_csv",
        )
    with tab_results:
        st.dataframe(results, width="stretch", hide_index=True)
        st.download_button(
            "download results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="results.csv",
            mime="text/csv",
            key="dl_results_csv",
        )
    with tab_payload:
        blob = payload.model_dump_json(indent=2, exclude_computed_fields=True)
        st.download_button(
            "download payload JSON",
            data=blob.encode("utf-8"),
            file_name="report.json",
            mime="application/json",
            key="dl_payload_json",
        )
        st.code(
            blob[:20000] + ("\n… truncated in preview" if len(blob) > 20000 else ""),
            language="json",
        )


def main() -> None:
    """Entry point: render the sidebar, then the seven tabs.

    Called at import time when streamlit executes this file as a script, which
    is why the module-level guard at the bottom exists: ``streamlit run`` does
    not use ``if __name__ == "__main__"`` the way a normal script does, but
    plain ``python -c "import tokbench.panel"`` must not start a UI.
    """
    st.set_page_config(
        page_title="tokbench",
        page_icon="▚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    base_dir = os.environ.get("TOKBENCH_RUNS_DIR", "reports")

    payload, options = _sidebar(base_dir)
    if payload is None:
        st.title("tokbench")
        st.warning(
            "No run loaded. Point the sidebar at a directory containing "
            "tokbench run directories (each with a `manifest.json`)."
        )
        return

    filtered = _filter_payload(payload, options)
    st.title(filtered.meta.experiment)
    st.caption(
        f"run `{filtered.meta.run_id}` · {len(filtered.leaderboard)} pairs shown · "
        f"source `{options.get('run_dir', '')}`"
    )

    tabs = st.tabs(
        [
            "Overview",
            "Pareto",
            "Tokens",
            "Retry",
            "Cache",
            "Strategy ablation",
            "Raw",
        ]
    )
    with tabs[0]:
        _overview_tab(filtered)
    with tabs[1]:
        _pareto_tab(filtered)
    with tabs[2]:
        _tokens_tab(filtered)
    with tabs[3]:
        _retry_tab(filtered)
    with tabs[4]:
        _cache_tab(filtered, options)
    with tabs[5]:
        _strategy_tab(filtered)
    with tabs[6]:
        _raw_tab(filtered)


# ``streamlit run`` executes this module as ``__main__``; importing it (tests,
# ``tokbench panel``'s import check) must not launch the UI.
if __name__ == "__main__":
    main()
