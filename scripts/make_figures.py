"""Render the README's headline figures from a finished run's report.json.

The HTML report already contains hand-authored inline SVG (dependency-free, opens
offline).  This script exists for a different job: producing **static PNGs** for
the README, where a rasterised image is what renders on GitHub.

Design notes
------------
- It reads ``report.json`` (the analysed payload), never the raw JSONL, so the
  figures cannot disagree with the numbers quoted in the README.
- matplotlib is used purely as a rasteriser: the layout here is intentionally
  matched to the HTML report's dark terminal aesthetic so the two artifacts look
  like one product.
- ``Agg`` backend: headless, no display, no browser sandbox involved.

Usage
-----
    python scripts/make_figures.py reports/<run-id>
    python scripts/make_figures.py reports/<run-id> --out docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- #
# Theme — mirrors tokbench/tokbench/render/html.py's CSS custom properties.
# --------------------------------------------------------------------------- #
BG = "#0b0f14"
PANEL = "#111820"
GRID = "#1e2a36"
TEXT = "#e6edf3"
MUTED = "#8b9bab"
ACCENT = "#22d3ee"  # cyan  — frontier
DOMINATED = "#3b4a5a"  # dimmed points
AMBER = "#fbbf24"  # knee
MAGENTA = "#e879f9"  # retry
GREEN = "#34d399"

MONO = ["DejaVu Sans Mono", "Consolas", "monospace"]


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def _fmt_usd(v: float) -> str:
    if v == 0:
        return "$0"
    if abs(v) < 0.001:
        return f"${v:.5f}"
    if abs(v) < 1:
        return f"${v:.4f}"
    return f"${v:,.2f}"


def _usd(v: float) -> str:
    """Matplotlib-safe USD string.

    matplotlib parses ``$...$`` as mathtext, so a raw dollar sign silently
    swallows the text between two of them ("$0.07 of $1.96" renders as
    "0.07of1.96" in italics).  Every literal ``$`` must be escaped.
    """
    return _fmt_usd(v).replace("$", r"\$")


def _place_labels(
    ax: plt.Axes,
    points: list[tuple[float, float, str]],
    *,
    min_gap_px: float = 15.0,
) -> None:
    """Draw labels beside markers, resolving overlaps in *display* space.

    Why not just offset every label by a constant?  Because a Pareto frontier is
    exactly the kind of data that clusters: several efficient options can sit
    within a fraction of a percent of each other on both axes.  A fixed offset
    turns that cluster into unreadable overstrike.

    This does a greedy one-pass push in pixel space: labels are sorted by height,
    and any label closer than ``min_gap_px`` to the previous one is pushed down.
    A leader line is drawn whenever a label ends up displaced, so the reader can
    always tell which marker it belongs to.  Labels are clamped to the axes so
    nothing is clipped at the top or bottom.
    """
    if not points:
        return

    fig = ax.figure
    fig.canvas.draw()  # needed for a valid transData
    trans = ax.transData

    # Measure label heights in pixels (one line per row; 2 rows for "model / strat").
    heights: list[float] = []
    for _, _, label in points:
        text = ax.text(0, 0, label, fontsize=7.6, fontfamily=MONO, alpha=0)
        bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
        heights.append(bb.height)
        text.remove()

    order = sorted(range(len(points)), key=lambda i: points[i][1], reverse=True)
    ax_top = ax.get_window_extent().y1
    ax_bot = ax.get_window_extent().y0

    placed_y: dict[int, float] = {}
    prev_y: float | None = None
    for idx in order:
        x, y, _ = points[idx]
        px, py = trans.transform((x, y))
        target = py
        if prev_y is not None and prev_y - target < heights[idx] + min_gap_px:
            target = prev_y - heights[idx] - min_gap_px
        target = max(min(target, ax_top - heights[idx] / 2), ax_bot + heights[idx] / 2)
        placed_y[idx] = target
        prev_y = target

    for idx, (x, y, label) in enumerate(points):
        _, py = trans.transform((x, y))
        ty = placed_y[idx]
        # ``xy`` must be in DATA coordinates (the marker position); only the
        # offset is in points.  The push-apart maths above is in pixels, so
        # convert that delta back to points here.
        dy_points = (ty - py) * 72.0 / fig.dpi
        ax.annotate(
            label.replace(" / ", "\n"),
            xy=(x, y),
            xytext=(11.0, dy_points),
            textcoords="offset points",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=7.6,
            fontfamily=MONO,
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.24", fc=BG, ec="none", alpha=0.80),
            arrowprops=(
                dict(arrowstyle="-", color=MUTED, linewidth=0.7, alpha=0.55)
                if abs(dy_points) > 2.5
                else None
            ),
        )


# --------------------------------------------------------------------------- #
# Figure 1 — Pareto frontier (the hero chart)
# --------------------------------------------------------------------------- #
def figure_pareto(payload: dict, out: Path) -> Path:
    pareto = payload["pareto"]
    points = pareto["points"]
    if not points:
        raise SystemExit("report has no pareto points; run a real experiment first")

    fig, ax = plt.subplots(figsize=(12.4, 6.6), facecolor=BG)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.86, bottom=0.13)

    # Plot cost in milli-USD. Sub-cent costs on a raw USD axis produce tick labels
    # like "$0.00050", which is unreadable; "0.50 mUSD" is the same number with a
    # sane unit. The axis label says so explicitly.
    xs = [p["cost_per_task_usd"] * 1000.0 for p in points]
    ys = [p["quality"] for p in points]
    x_max, y_max = max(xs), max(ys)
    ax.set_xlim(-0.06 * x_max, x_max * 1.30)
    ax.set_ylim(0, y_max * 1.12)

    frontier = sorted((p for p in points if p["on_frontier"]), key=lambda p: p["cost_per_task_usd"])
    dominated = [p for p in points if not p["on_frontier"]]

    # Shade the efficient region: everything above/right of the frontier is
    # reachable only by paying more for the same quality.
    if len(frontier) >= 2:
        fx = [p["cost_per_task_usd"] * 1000.0 for p in frontier]
        fy = [p["quality"] for p in frontier]
        ax.fill_between(fx, fy, [0] * len(fx), color=ACCENT, alpha=0.07, zorder=1, linewidth=0)
        ax.plot(fx, fy, color=ACCENT, linewidth=2.2, zorder=3, solid_capstyle="round")

    # Dominated points: grey, smaller, behind the frontier.
    if dominated:
        ax.scatter(
            [p["cost_per_task_usd"] * 1000.0 for p in dominated],
            [p["quality"] for p in dominated],
            s=58,
            color=DOMINATED,
            edgecolors=GRID,
            linewidths=1.0,
            zorder=2,
            label="dominated",
        )

    # Frontier points.
    ax.scatter(
        [p["cost_per_task_usd"] * 1000.0 for p in frontier],
        [p["quality"] for p in frontier],
        s=110,
        color=ACCENT,
        edgecolors=BG,
        linewidths=1.6,
        zorder=5,
        label="pareto frontier",
    )

    # Knee annotation (the best-value point).  The callout is anchored in the
    # empty mid-band rather than next to the marker: the efficient front is
    # extremely dense near the origin, and any callout placed there collides with
    # the point labels.
    knee_label = pareto.get("best_value_label")
    for p in points:
        if p["label"] == knee_label:
            kx = p["cost_per_task_usd"] * 1000.0
            ax.scatter(
                [kx],
                [p["quality"]],
                s=300,
                facecolors="none",
                edgecolors=AMBER,
                linewidths=2.0,
                zorder=6,
            )
            ax.annotate(
                f"best value  ·  {p['label'].replace(' / ', ' / ')}\n"
                f"{_usd(p['cost_per_task_usd'])}/task  ·  quality {p['quality']:.3f}",
                xy=(kx, p["quality"]),
                xytext=(x_max * 0.30, y_max * 0.24),
                textcoords="data",
                color=AMBER,
                fontsize=9,
                fontfamily=MONO,
                fontweight="bold",
                zorder=9,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.42", fc=BG, ec=AMBER, lw=0.9, alpha=0.95),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=AMBER,
                    linewidth=1.1,
                    alpha=0.85,
                    connectionstyle="arc3,rad=0.18",
                ),
            )
            break

    # Labels, de-collided in pixel space.  Frontier points are labelled; dominated
    # ones would turn the chart into a wall of text.
    _place_labels(
        ax,
        [(p["cost_per_task_usd"] * 1000.0, p["quality"], p["label"]) for p in frontier],
    )

    _style_axes(
        ax,
        "Pareto frontier — cost per task vs mean quality",
        "cost per task (milli-USD; 1.00 = $0.001)",
        "mean graded quality",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    legend = ax.legend(
        loc="lower right",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_fontfamily(MONO)

    # Methodology footer — the chart should be self-explanatory out of context.
    fig.text(
        0.075,
        0.945,
        f"{payload['meta']['experiment']}  ·  {payload['summary']['completed_cells']:,} cells"
        f"  ·  {_usd(payload['summary']['total_cost_usd'])} total"
        f"  ·  {len(frontier)}/{len(points)} points non-dominated",
        color=MUTED,
        fontsize=9.5,
        fontfamily=MONO,
        va="bottom",
    )
    fig.text(
        0.075,
        0.032,
        "A point is dominated when another option is both cheaper and higher quality. "
        "Only frontier points are rational choices;\nwhich one depends on your quality floor.",
        color=MUTED,
        fontsize=8,
        fontfamily=MONO,
        va="bottom",
        linespacing=1.5,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 2 — retry cost share
# --------------------------------------------------------------------------- #
def figure_retry(payload: dict, out: Path) -> Path:
    rows = [r for r in payload["retry"]["rows"] if r["retried_cells"] > 0]
    overall = payload["retry"]["overall_retry_cost_share"]
    if not rows:
        raise SystemExit("report has no retried cells; nothing to plot")

    rows.sort(key=lambda r: r["retry_cost_share"], reverse=True)
    labels = [f"{r['model_id']} / {r['strategy_id']}" for r in rows]
    shares = [r["retry_cost_share"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(12.4, 7.6), facecolor=BG)
    fig.subplots_adjust(left=0.285, right=0.955, top=0.815, bottom=0.135)

    ypos = list(range(len(rows)))[::-1]  # worst at the top
    # A 100%-wide stacked bar is the wrong encoding here: every value lives in the
    # 1-8% band, so the informative part collapses to a sliver.  Plot the retry
    # share itself on an absolute axis and draw the 3.75% run-wide average as a
    # reference line — the reader compares bars to the average, not to 100%.
    bars = ax.barh(
        ypos,
        shares,
        color=MAGENTA,
        edgecolor="none",
        height=0.62,
        alpha=0.92,
        label="retry waste",
    )

    worst = max(shares)
    for y, share, row in zip(ypos, shares, rows, strict=False):
        ax.text(
            share + worst * 0.022,
            y,
            f"{share:.1f}%   {row['retried_cells']}/{row['cells']} cells   "
            f"{row['dominant_error_kind'] or '-'}",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=8.2,
            fontfamily=MONO,
        )
    # Highlight the worst offender.
    bars[0].set_color("#f472b6")
    bars[0].set_edgecolor(AMBER)
    bars[0].set_linewidth(1.4)

    ax.axvline(overall * 100, color=ACCENT, linestyle="--", linewidth=1.4, alpha=0.85, zorder=4)
    ax.text(
        overall * 100,
        len(rows) - 0.35,
        f"  run average {overall * 100:.2f}%",
        color=ACCENT,
        fontsize=8.6,
        fontfamily=MONO,
        va="bottom",
        ha="left",
        fontweight="bold",
    )

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.4, color=TEXT, fontfamily=MONO)
    ax.set_xlim(0, worst * 1.55)
    ax.set_ylim(-0.75, len(rows) - 0.05)
    ax.set_xlabel("retry spend as a share of that pair's total spend (%)", color=MUTED, fontsize=10)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Retry cost share — {overall * 100:.2f}% of total spend bought nothing",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        pad=14,
        loc="left",
    )

    legend = ax.legend(
        loc="lower right",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_fontfamily(MONO)

    fig.text(
        0.285,
        0.935,
        f"pure waste: {_usd(payload['retry']['total_retry_cost_usd'])} of "
        f"{_usd(payload['retry']['total_cost_usd'])}   ·   "
        f"{payload['retry']['total_retry_tokens']:,} tokens re-sent   ·   "
        f"worst: {payload['retry'].get('worst_offender') or '-'}",
        color=MAGENTA,
        fontsize=9.5,
        fontfamily=MONO,
        va="bottom",
        fontweight="bold",
    )
    fig.text(
        0.285,
        0.028,
        "Every attempt is billed, including the ones that failed. A pair with a high retry share and a dominant\n"
        "'rate_limit' error is a concurrency problem, not a model problem — fix the client, not the prompt.",
        color=MUTED,
        fontsize=8,
        fontfamily=MONO,
        va="bottom",
        linespacing=1.5,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 3 — cache sensitivity
# --------------------------------------------------------------------------- #
def figure_cache(payload: dict, out: Path) -> Path:
    cache = payload["cache"]
    curve = cache["curve"]
    if not curve:
        raise SystemExit("report has no cache curve")

    fig, ax = plt.subplots(figsize=(12.4, 6.2), facecolor=BG)
    fig.subplots_adjust(left=0.085, right=0.97, top=0.85, bottom=0.15)

    xs = [p["cache_hit_rate"] * 100 for p in curve]
    ys = [p["projected_cost_usd"] for p in curve]
    ax.plot(xs, ys, color=ACCENT, linewidth=2.4, zorder=3)
    ax.fill_between(xs, ys, min(ys) * 0.985, color=ACCENT, alpha=0.09, linewidth=0)

    obs_x = cache["observed_hit_rate"] * 100
    obs_y = cache["observed_cost_usd"]
    ax.axvline(obs_x, color=AMBER, linestyle="--", linewidth=1.4, alpha=0.9, zorder=2)
    ax.scatter([obs_x], [obs_y], s=130, color=AMBER, edgecolors=BG, linewidths=1.6, zorder=5)
    ax.annotate(
        f"observed {obs_x:.1f}%\n{_usd(obs_y)}",
        xy=(obs_x, obs_y),
        xytext=(12, 18),
        textcoords="offset points",
        color=AMBER,
        fontsize=9,
        fontfamily=MONO,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=AMBER, linewidth=1.0, alpha=0.8),
    )

    _style_axes(
        ax,
        "Cache sensitivity — projected bill vs prompt-cache hit rate",
        "prompt cache hit rate (%)",
        "projected total cost (USD)",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _usd(v)))
    ax.set_xlim(0, 100)

    fig.text(
        0.085,
        0.945,
        f"each +10pp is worth {_usd(abs(cache['marginal_usd_per_10pp']))}  ·  "
        f"already saved {_usd(cache['savings_at_observed_usd'])} "
        f"({cache['savings_at_observed_pct']:.2f}%) vs no cache",
        color=GREEN,
        fontsize=9.5,
        fontfamily=MONO,
        va="bottom",
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.035,
        "Total input and output tokens are held fixed while input is redistributed between full-price and\n"
        "cache-read. Cost is affine in hit rate, so the slope is a budget line item, not an estimate.",
        color=MUTED,
        fontsize=8,
        fontfamily=MONO,
        va="bottom",
        linespacing=1.5,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory (or a report.json path)")
    parser.add_argument("--out", default="docs", help="output directory for PNGs")
    args = parser.parse_args(argv)

    run = Path(args.run)
    report_path = run / "report.json" if run.is_dir() else run
    if not report_path.exists():
        print(f"error: {report_path} not found (run `tokbench report` first)", file=sys.stderr)
        return 1

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    out = Path(args.out)

    made = [
        figure_pareto(payload, out / "pareto-frontier.png"),
        figure_retry(payload, out / "retry-cost-share.png"),
        figure_cache(payload, out / "cache-sensitivity.png"),
    ]
    for path in made:
        print(f"wrote {path}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
