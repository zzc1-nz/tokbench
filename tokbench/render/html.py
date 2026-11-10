"""Self-contained HTML report renderer.

Design contract
---------------
``render_html`` turns a :class:`~tokbench.report_schema.ReportPayload` into a
**single file** that opens from ``file://`` with zero network activity: no CDN,
no webfont, no JS bundle, no ``<img src>``, not even the SVG XML namespace URI.
Every chart is hand-authored inline SVG, so the document is deterministic,
diff-able, screenshot-able and printable.

Why hand-rolled SVG instead of plotly/matplotlib here?
    The HTML report is the artefact people *share* (attach to a PR, drop in a
    ticket).  A 40 kB dependency-free file that renders identically on any
    machine beats a 3 MB plotly bundle that needs a JS runtime.  The Streamlit
    panel (``tokbench.panel``) keeps plotly for interactivity — different job,
    different tool.

Visual language
    Terminal/instrument-panel: near-black ground, hairline grid, monospace for
    every number, one restrained neon accent family (cyan = primary / cheap,
    amber = caution / cost, magenta = waste).  Colour is *semantic*, never
    decorative: magenta only ever means "money that bought nothing".

Escaping policy
    Model ids, strategy ids, error kinds and free-text explanations are all
    attacker-controlled (they come from configs and provider responses).  Every
    dynamic string passes through :func:`esc`, which uses ``html.escape(...,
    quote=True)`` so it is safe in text nodes, attributes and ``<title>``
    tooltips alike.  Numbers never reach the output via ``str()``: they go
    through the ``fmt_*`` helpers, which coerce non-finite values to ``"—"`` so
    a ``nan``/``inf`` can never leak into the document.

Structure
    masthead → KPI cards → hero Pareto → retry burn → cache sensitivity →
    cumulative series → token-type stacked bars → leaderboard → strategy
    ablation → categories → methodology footer.

Public API
    :func:`render_html`, :func:`write_html`.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..report_schema import (
    CacheSensitivity,
    EfficiencyRow,
    ParetoFrontier,
    ReportPayload,
    RetryAnalysis,
    SeriesPoint,
    StackedBarRow,
)

__all__ = ["render_html", "write_html"]

#: Rendered in the footer; keep in sync with ``tokbench.__version__``.
GENERATOR = "tokbench"

#: Sentinel shown wherever a value is absent.  Deliberately not ``None``/``nan``:
#: a report full of Python literals reads like a bug, an em dash reads like data.
MISSING = "\u2014"

#: Cap on rows drawn in the bar/stacked charts.  Beyond this the chart becomes
#: unreadable and the tables below carry the full data anyway.
MAX_CHART_ROWS = 12

#: Character budget for Pareto point labels.  Real runs produce labels such as
#: ``deepseek-reasoner / fewshot-compress`` (36 chars), so the budget is set to
#: fit those whole: at the 5.95px mono advance width that is ~214px, comfortably
#: inside the ~890px plot, and the column layout absorbs the extra width by
#: pushing long labels to their own side.
LABEL_BUDGET = 38


# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Palette:
    """One resolved colour scheme.

    Kept as a dataclass (rather than raw dicts) so a typo in a token name is an
    ``AttributeError`` at import time instead of a silently unstyled chart.
    """

    bg: str
    bg_soft: str
    panel: str
    panel_2: str
    line: str
    line_soft: str
    ink: str
    ink_2: str
    ink_3: str
    grid: str
    axis: str
    accent: str  # cyan   — primary, "good", cheap
    accent_dim: str
    accent_2: str  # amber  — cost / caution / knee
    accent_3: str  # magenta — retry waste
    good: str
    font_sans: str
    font_mono: str


_DARK = Palette(
    bg="#07090c",
    bg_soft="#0c1620",
    panel="#0b0f14",
    panel_2="#121821",
    line="#1b2430",
    line_soft="#151d27",
    ink="#e6edf3",
    ink_2="#93a1b0",
    ink_3="#5b6875",
    grid="#19222d",
    axis="#2a3644",
    accent="#22d3ee",
    accent_dim="#0e7490",
    accent_2="#fbbf24",
    accent_3="#f472b6",
    good="#34d399",
    font_sans=(
        'ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    ),
    font_mono=(
        'ui-monospace, "SFMono-Regular", "JetBrains Mono", "Cascadia Mono", '
        'Consolas, "Liberation Mono", monospace'
    ),
)

_LIGHT = Palette(
    bg="#ffffff",
    bg_soft="#f4f7fa",
    panel="#fbfcfd",
    panel_2="#f1f5f9",
    line="#dbe3ec",
    line_soft="#e8eef4",
    ink="#0f172a",
    ink_2="#475569",
    ink_3="#7b8a9c",
    grid="#e6ecf3",
    axis="#c3cedb",
    accent="#0e7490",
    accent_dim="#67c6d8",
    accent_2="#b45309",
    accent_3="#be185d",
    good="#047857",
    font_sans=_DARK.font_sans,
    font_mono=_DARK.font_mono,
)

_THEMES: dict[str, Palette] = {"dark": _DARK, "light": _LIGHT}


def _palette(theme: str) -> Palette:
    """Resolve a theme name to a :class:`Palette`."""
    try:
        return _THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; expected one of {sorted(_THEMES)}") from None


# --------------------------------------------------------------------------- #
# Escaping + number formatting
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    """HTML-escape any dynamic value for use in text nodes *and* attributes.

    ``quote=True`` is not optional here: labels land inside ``title="..."`` and
    ``aria-label="..."``, where an unescaped quote would break out of the
    attribute.  ``None`` becomes an em dash rather than the string ``"None"``.
    """
    if value is None:
        return MISSING
    return html.escape(str(value), quote=True)


def _finite(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float, mapping ``None``/``nan``/``inf`` to *default*.

    This is the single choke point that keeps ``nan`` out of the document: every
    formatter below funnels through it.
    """
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def fmt_int(value: Any) -> str:
    """``1234567`` → ``"1,234,567"``; absent/non-finite → em dash."""
    if value is None:
        return MISSING
    try:
        f = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(f):
        return MISSING
    return f"{int(round(f)):,}"


def fmt_tokens(value: Any) -> str:
    """Compact token count: ``1_200_000`` → ``"1.2M"``, ``45_300`` → ``"45.3k"``.

    One decimal is intentional: at a glance you need magnitude, not precision.
    Exact counts live in the tables.
    """
    if value is None:
        return MISSING
    try:
        n = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(n):
        return MISSING
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}k"
    return f"{sign}{int(round(n))}"


def fmt_usd(value: Any) -> str:
    """USD with precision that scales down with magnitude.

    ``$12.34`` · ``$0.0123`` · ``$0.000420``.  Sub-cent values are the normal
    case for per-task cost, so they must not round to ``$0.00``.
    """
    if value is None:
        return MISSING
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(v):
        return MISSING
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000:
        return f"{sign}${a:,.2f}"
    if a >= 1:
        return f"{sign}${a:.2f}"
    if a >= 0.01:
        return f"{sign}${a:.4f}"
    if a == 0:
        return "$0.00"
    return f"{sign}${a:.6f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Fraction → percent string.  ``0.184`` → ``"18.4%"``.

    The schema stores shares as fractions in ``[0, 1]``; the report always shows
    them as percentages, so the conversion lives here and nowhere else.
    """
    if value is None:
        return MISSING
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(v):
        return MISSING
    return f"{v * 100:.{digits}f}%"


def fmt_pp(value: Any, digits: int = 1) -> str:
    """Fraction delta → signed percentage-points string (``"+2.3pp"``)."""
    if value is None:
        return MISSING
    v = _finite(value)
    return f"{v * 100:+.{digits}f}pp"


def fmt_quality(value: Any) -> str:
    """Mean quality is a ``[0, 1]`` score; three decimals distinguishes rows."""
    if value is None:
        return MISSING
    v = _finite(value)
    return f"{v:.3f}"


def fmt_ratio(value: Any, digits: int = 2) -> str:
    """Plain fixed-point number for scores/ratios that are not money or tokens."""
    if value is None:
        return MISSING
    return f"{_finite(value):.{digits}f}"


def fmt_duration(value: Any) -> str:
    """Seconds → ``"820ms"`` / ``"12.4s"`` / ``"3m 07s"`` / ``"1h 04m"``."""
    if value is None:
        return MISSING
    s = _finite(value)
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 90:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60):02d}m"


def fmt_dt(value: Any) -> str:
    """ISO-ish UTC timestamp; the report never shows a bare ``datetime`` repr."""
    if value is None:
        return MISSING
    if not isinstance(value, datetime):
        return MISSING
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_dt_short(value: Any) -> str:
    """``"2026-11-10 13:38"`` — used on dense chart axes."""
    if not isinstance(value, datetime):
        return MISSING
    dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%m-%d %H:%M")


def ellipsis(text: Any, limit: int) -> str:
    """Truncate for chart labels, preserving the head of the string."""
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[: max(limit - 1, 1)] + "\u2026"


def _attr(**kwargs: Any) -> str:
    """Build an attribute string, dropping empty values and escaping the rest.

    ``cls`` is translated to the real ``class`` attribute: it is spelled ``cls``
    in the call sites because ``class`` is a Python keyword, and getting this
    mapping wrong silently emits ``cls="..."`` (styled by nothing) instead of a
    hard error — so it is handled once, here, for every primitive.
    """
    parts = []
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        name = "class" if key == "cls" else key.replace("_", "-")
        parts.append(f'{name}="{esc(value)}"')
    return (" " + " ".join(parts)) if parts else ""


# --------------------------------------------------------------------------- #
# Scales and tick generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LinearScale:
    """Map a data domain onto a pixel range (linear, no clamping).

    ``px_lo``/``px_hi`` are *pixel* positions for ``lo``/``hi``; for a y axis the
    caller passes ``(bottom, top)`` so the mapping flips naturally.
    """

    lo: float
    hi: float
    px_lo: float
    px_hi: float

    def __call__(self, value: float) -> float:
        span = self.hi - self.lo
        if span <= 0 or not math.isfinite(span):
            return (self.px_lo + self.px_hi) / 2.0
        t = (value - self.lo) / span
        return self.px_lo + t * (self.px_hi - self.px_lo)

    def clamp(self, value: float) -> float:
        return min(max(value, self.px_lo), self.px_hi)


def nice_ticks(lo: float, hi: float, target: int = 5) -> tuple[list[float], float]:
    """Human-friendly ticks: steps of 1/2/2.5/5 × 10ⁿ.

    Returns ``(ticks, step)``.  Used for every quantitative axis so gridlines
    land on round numbers instead of ``0.0375``.
    """
    lo, hi = _finite(lo), _finite(hi)
    if hi <= lo:
        hi = lo + (abs(lo) if lo else 1.0)
    raw = (hi - lo) / max(target, 1)
    if raw <= 0 or not math.isfinite(raw):
        return [lo, hi], hi - lo
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = magnitude
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mult * magnitude:
            step = mult * magnitude
            break
    start = math.ceil(lo / step - 1e-9) * step
    ticks: list[float] = []
    value = start
    guard = 0
    while value <= hi + step * 1e-6 and guard < 64:
        ticks.append(round(value, 12))
        value += step
        guard += 1
    if not ticks:
        ticks = [lo, hi]
    return ticks, step


def _pad_domain(
    values: Iterable[float],
    *,
    lo: float | None = None,
    hi: float | None = None,
    pad_frac: float = 0.06,
) -> tuple[float, float]:
    """Domain with headroom so markers are never clipped by the plot edge."""
    vals = [_finite(v) for v in values]
    if not vals:
        return (lo if lo is not None else 0.0, hi if hi is not None else 1.0)
    d_lo = min(vals) if lo is None else lo
    d_hi = max(vals) if hi is None else hi
    if d_hi <= d_lo:
        pad = abs(d_hi) * 0.1 or 1.0
        return d_lo - pad, d_hi + pad
    span = d_hi - d_lo
    return d_lo - span * pad_frac, d_hi + span * pad_frac


# --------------------------------------------------------------------------- #
# SVG primitives
# --------------------------------------------------------------------------- #
def _svg_open(
    width: float,
    height: float,
    *,
    label: str,
    desc: str,
    extra_class: str = "",
) -> str:
    """Open an inline ``<svg>`` with accessible name + description.

    No ``xmlns`` on purpose: this SVG only ever lives inside the HTML document,
    so the attribute would add a ``http://`` string to a file that promises zero
    external references.  Omitting it keeps that promise literally greppable.
    """
    cls = "chart-svg" + (f" {extra_class}" if extra_class else "")
    return (
        f'<svg class="{esc(cls)}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="{esc(label)}">'
        f"<title>{esc(label)}</title><desc>{esc(desc)}</desc>"
    )


def _rect(x: float, y: float, w: float, h: float, *, cls: str = "", rx: float = 0.0) -> str:
    rx_attr = f' rx="{rx:.1f}"' if rx else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" '
        f'height="{max(h, 0):.2f}"{rx_attr}{_attr(cls=cls)}/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    cls: str = "",
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{esc(dash)}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
        f"{_attr(cls=cls)}{dash_attr}/>"
    )


def _text(
    x: float,
    y: float,
    content: Any,
    *,
    cls: str = "",
    anchor: str | None = None,
    baseline: str | None = None,
) -> str:
    """SVG text node.  Content is escaped here — callers pass raw strings."""
    extra = ""
    if anchor:
        extra += f' text-anchor="{esc(anchor)}"'
    if baseline:
        extra += f' dominant-baseline="{esc(baseline)}"'
    return f'<text x="{x:.2f}" y="{y:.2f}"{_attr(cls=cls)}{extra}>{esc(content)}</text>'


def _circle(cx: float, cy: float, r: float, *, cls: str = "", tip: str | None = None) -> str:
    body = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"{_attr(cls=cls)}/>'
    if tip:
        return f"<g>{body}<title>{esc(tip)}</title></g>"
    return body


def _diamond(cx: float, cy: float, r: float, *, cls: str = "") -> str:
    return (
        f'<polygon class="{esc(cls)}" points="'
        f"{cx:.2f},{cy - r:.2f} {cx + r:.2f},{cy:.2f} "
        f'{cx:.2f},{cy + r:.2f} {cx - r:.2f},{cy:.2f}"/>'
    )


def _polyline(points: Sequence[tuple[float, float]], *, cls: str = "") -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline class="{esc(cls)}" points="{coords}"/>'


def _area_path(points: Sequence[tuple[float, float]], base_y: float) -> str:
    """Closed path under a polyline, used for the cumulative-token area fill."""
    if len(points) < 2:
        return ""
    first_x, _ = points[0]
    last_x, _ = points[-1]
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<path class="area-fill" d="M {first_x:.2f},{base_y:.2f} L {coords} '
        f'L {last_x:.2f},{base_y:.2f} Z"/>'
    )


def _legend(
    items: Sequence[tuple[str, str]], right_x: float, y: float, *, gap: float = 16.0
) -> str:
    """Right-aligned horizontal legend, drawn inside the SVG's top margin.

    Kept in the margin (not over the data) so it can never obscure a point —
    a small thing that separates "chart" from "publication figure".
    """
    # Right-align: lay items out backwards from ``right_x``.
    widths = [len(label) * 6.05 + 24.0 for _, label in items]
    total = sum(widths) + gap * (len(items) - 1 if items else 0)
    x = right_x - total
    out = ['<g class="legend" aria-hidden="true">']
    for (kind, label), width in zip(items, widths, strict=False):
        out.append(_legend_marker(kind, x + 8.0, y))
        out.append(_text(x + 21.0, y + 3.5, label, cls="legend-label"))
        x += width + gap
    out.append("</g>")
    return "".join(out)


def _legend_marker(kind: str, x: float, y: float) -> str:
    """Swatch for one legend entry; ``kind`` names the visual grammar."""
    if kind == "front":
        return (
            f'<line class="lg-line-accent" x1="{x - 2:.2f}" y1="{y:.2f}" '
            f'x2="{x + 14:.2f}" y2="{y:.2f}"/>'
            f'<circle class="p-front" cx="{x + 6:.2f}" cy="{y:.2f}" r="2.8"/>'
        )
    if kind == "dominated":
        return f'<circle class="p-dom" cx="{x + 6:.2f}" cy="{y:.2f}" r="2.6"/>'
    if kind == "knee":
        return _diamond(x + 6.0, y, 4.0, cls="knee-dot")
    if kind == "baseline":
        return f'<circle class="p-baseline" cx="{x + 6:.2f}" cy="{y:.2f}" r="3.4"/>'
    if kind == "line-accent":
        return (
            f'<line class="lg-line-accent" x1="{x - 2:.2f}" y1="{y:.2f}" '
            f'x2="{x + 14:.2f}" y2="{y:.2f}"/>'
        )
    if kind == "line-amber":
        return (
            f'<line class="lg-line-amber" x1="{x - 2:.2f}" y1="{y:.2f}" '
            f'x2="{x + 14:.2f}" y2="{y:.2f}"/>'
        )
    if kind == "area":
        return _rect(x - 1.0, y - 4.0, 15.0, 8.0, cls="lg-area", rx=1.0)
    if kind == "observed":
        return (
            f'<circle class="p-observed" cx="{x + 6:.2f}" cy="{y:.2f}" r="3.2"/>'
            f'<line class="obs-line" x1="{x + 6:.2f}" y1="{y - 6:.2f}" '
            f'x2="{x + 6:.2f}" y2="{y + 6:.2f}"/>'
        )
    if kind == "waste":
        return _rect(x - 1.0, y - 4.0, 15.0, 8.0, cls="lg-waste", rx=1.0)
    if kind == "first":
        return _rect(x - 1.0, y - 4.0, 15.0, 8.0, cls="lg-first", rx=1.0)
    if kind.startswith("swatch-"):
        # ``swatch-<segment>`` reuses the chart's own fill class, so a legend
        # swatch can never drift out of sync with the series it describes.
        return _rect(x - 1.0, y - 4.5, 15.0, 9.0, cls=kind, rx=1.5)
    return ""


def _axis_frame(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_scale: LinearScale,
    y_scale: LinearScale,
    fmt_x: Any,
    fmt_y: Any,
    x_label: str,
    y_label: str,
    minor_y: Sequence[float] = (),
) -> str:
    """Gridlines + ticks + axis titles for a standard cartesian chart.

    Shared by every cartesian chart so the report has exactly one axis
    convention: hairline grid, mono ticks, uppercase micro-caps axis titles.
    """
    out: list[str] = ['<g class="axes" aria-hidden="true">']
    for tick in minor_y:
        ty = y_scale(tick)
        if y0 - 0.5 <= ty <= y1 + 0.5:
            out.append(_line(x0, ty, x1, ty, cls="grid-minor"))
    for tick in y_ticks:
        ty = y_scale(tick)
        if ty < y0 - 0.5 or ty > y1 + 0.5:
            continue
        out.append(_line(x0, ty, x1, ty, cls="grid"))
        out.append(_text(x0 - 11.0, ty + 3.5, fmt_y(tick), cls="tick tick--y"))
    for tick in x_ticks:
        tx = x_scale(tick)
        if tx < x0 - 0.5 or tx > x1 + 0.5:
            continue
        out.append(_line(tx, y0, tx, y1, cls="grid grid--v"))
        out.append(_text(tx, y1 + 19.0, fmt_x(tick), cls="tick tick--x"))
    # Baseline axes are drawn last so they sit above the gridlines.
    out.append(_line(x0, y1, x1, y1, cls="axis"))
    out.append(_line(x0, y0, x0, y1, cls="axis"))
    out.append(_text((x0 + x1) / 2.0, y1 + 46.0, x_label, cls="axl axl--x", anchor="middle"))
    out.append(
        f'<text class="axl axl--y" text-anchor="middle" '
        f'transform="translate({x0 - 58.0:.2f},{(y0 + y1) / 2.0:.2f}) rotate(-90)">'
        f"{esc(y_label)}</text>"
    )
    out.append("</g>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Static CSS
# --------------------------------------------------------------------------- #
_CSS = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:var(--font-sans);font-size:14px;line-height:1.55;
  background-image:radial-gradient(1100px 520px at 8% -12%,var(--bg-soft),transparent 62%);
  background-repeat:no-repeat;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
.wrap{max-width:1180px;margin:0 auto;padding:36px 26px 72px}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-.01em}
a{color:var(--accent)}
.mono{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.eyebrow{
  font-family:var(--font-mono);font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 10px;
}
/* ---------- masthead ---------- */
.masthead{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:30px}
.masthead-top{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px}
.brand{
  font-family:var(--font-mono);font-size:12px;letter-spacing:.22em;
  text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:9px;
}
.brand::before{
  content:"";width:7px;height:7px;border-radius:2px;background:var(--accent);
  box-shadow:0 0 12px var(--accent);
}
.masthead h1{font-size:30px;line-height:1.18;margin:14px 0 6px}
.runline{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:12px;color:var(--ink-2);font-size:12.5px}
.runline .k{color:var(--ink-3);font-family:var(--font-mono);font-size:10.5px;
  letter-spacing:.13em;text-transform:uppercase;margin-right:7px}
.runline .v{font-family:var(--font-mono);color:var(--ink)}
.chip{
  display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:999px;
  border:1px solid var(--line);background:var(--panel-2);
  font-family:var(--font-mono);font-size:11px;color:var(--ink-2);white-space:nowrap;
}
.chip--accent{border-color:var(--accent-dim);color:var(--accent)}
.chip--warn{border-color:var(--accent-2);color:var(--accent-2)}
.chip--waste{border-color:var(--accent-3);color:var(--accent-3)}
.notes{margin:16px 0 0;padding:0;list-style:none;color:var(--ink-2);font-size:12.5px}
.notes li{padding-left:16px;position:relative;margin-top:3px}
.notes li::before{content:"›";position:absolute;left:0;color:var(--accent-dim);font-family:var(--font-mono)}
/* ---------- layout ---------- */
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(178px,1fr));margin-bottom:34px}
.two{grid-template-columns:1fr 1fr}
@media (max-width:900px){.two{grid-template-columns:1fr}}
.card{
  background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:24px 26px;position:relative;
}
.card + .card{margin-top:16px}
section{margin-top:34px}
.card-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px;margin-bottom:6px}
.card-head h2{font-size:17px}
.card-head .tag{
  font-family:var(--font-mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink-3);border:1px solid var(--line);border-radius:4px;padding:2px 6px;
}
.card-head .hero-tag{color:var(--accent);border-color:var(--accent-dim)}
.lede{color:var(--ink-2);font-size:13px;margin:8px 0 18px;max-width:82ch}
.lede b{color:var(--ink);font-weight:600}
.lede code{font-family:var(--font-mono);font-size:12px;color:var(--accent);background:var(--panel-2);
  padding:1px 5px;border-radius:4px}
/* ---------- KPI cards ---------- */
.kpi{
  background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:15px 16px 14px;display:flex;flex-direction:column;gap:5px;min-height:96px;
}
.kpi .k{
  font-family:var(--font-mono);font-size:9.8px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--ink-3);
}
.kpi .v{font-family:var(--font-mono);font-size:26px;line-height:1.1;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.kpi .s{font-family:var(--font-mono);font-size:11px;color:var(--ink-3)}
.kpi--accent .v{color:var(--accent)}
.kpi--warn .v{color:var(--accent-2)}
.kpi--waste .v{color:var(--accent-3);text-shadow:0 0 22px color-mix(in srgb,var(--accent-3) 34%,transparent)}
.kpi--good .v{color:var(--good)}
/* ---------- charts ---------- */
.figure{margin:0}
.figure figcaption{
  margin-top:14px;color:var(--ink-3);font-size:11.5px;font-family:var(--font-mono);
  letter-spacing:.02em;line-height:1.6;
}
.figure figcaption b{color:var(--ink-2);font-weight:600}
.chart-svg{display:block;width:100%;height:auto;overflow:visible}
/* SVG design system (classes only, so @media print can re-theme) */
.grid{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.grid--v{opacity:.55}
.grid-minor{stroke:var(--grid);stroke-width:1;opacity:.5;shape-rendering:crispEdges}
.axis{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.tick{fill:var(--ink-3);font-family:var(--font-mono);font-size:10.5px;
  font-variant-numeric:tabular-nums}
.tick--x{text-anchor:middle}
.tick--y{text-anchor:end}
.axl{fill:var(--ink-3);font-family:var(--font-mono);font-size:10px;
  letter-spacing:.14em;text-transform:uppercase}
.legend-label{fill:var(--ink-2);font-family:var(--font-mono);font-size:10px;letter-spacing:.04em}
.lg-line-accent{stroke:var(--accent);stroke-width:2;stroke-linecap:round}
.lg-line-amber{stroke:var(--accent-2);stroke-width:2;stroke-linecap:round}
.lg-area{fill:var(--accent);opacity:.3}
.lg-waste{fill:var(--accent-3);opacity:.85}
.lg-first{fill:var(--accent-dim);opacity:.85}
/* legend swatches reuse the series fill classes verbatim */
.swatch-uncached{fill:var(--accent)}
.swatch-cache-read{fill:var(--accent-dim)}
.swatch-cache-write{fill:var(--accent);opacity:.42}
.swatch-output{fill:var(--accent-2)}
.swatch-reasoning{fill:var(--accent-3);opacity:.85}
/* pareto */
.front-area{fill:var(--accent);opacity:.055}
.fline{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.fline-glow{fill:none;stroke:var(--accent);stroke-width:9;opacity:.10;stroke-linejoin:round;
  stroke-linecap:round}
.p-front{fill:var(--accent);stroke:var(--panel);stroke-width:1.5}
.p-dom{fill:var(--ink-3);opacity:.55}
.p-dom-hit{fill:transparent}
.p-baseline{fill:none;stroke:var(--ink-2);stroke-width:1.6;stroke-dasharray:2.5 2}
.p-knee{fill:var(--accent-2);stroke:var(--panel);stroke-width:1.4}
.knee-dot{fill:var(--accent-2)}
.knee-line{stroke:var(--accent-2);stroke-width:1;stroke-dasharray:3 3;fill:none;opacity:.8}
.knee-box{fill:var(--panel-2);stroke:var(--accent-2);stroke-width:1;opacity:.96}
.knee-label{fill:var(--accent-2);font-family:var(--font-mono);font-size:9.5px;
  letter-spacing:.15em;text-transform:uppercase}
.knee-text{fill:var(--ink);font-family:var(--font-mono);font-size:10.5px}
.plabel{fill:var(--ink-2);font-family:var(--font-mono);font-size:10px}
.plabel--front{fill:var(--ink)}
.leader{fill:none;stroke:var(--ink-3);stroke-width:1;opacity:.75}
.note{fill:var(--ink-3);font-family:var(--font-mono);font-size:10px;letter-spacing:.03em}
.hit{fill:transparent}
/* retry bars */
.track{fill:var(--panel-2)}
.seg-first{fill:var(--accent-dim);opacity:.9}
.seg-waste{fill:var(--accent-3);opacity:.92}
.rowlabel{fill:var(--ink-2);font-family:var(--font-mono);font-size:11px}
.rowlabel--hi{fill:var(--ink)}
.rowval{fill:var(--ink-3);font-family:var(--font-mono);font-size:10.5px;
  font-variant-numeric:tabular-nums}
.rowval--waste{fill:var(--accent-3)}
.bar-head{fill:var(--ink-3);font-family:var(--font-mono);font-size:9.5px;
  letter-spacing:.14em;text-transform:uppercase}
.maxline{stroke:var(--axis);stroke-width:1;stroke-dasharray:2 4;opacity:.7}
/* cache / series */
.curve{fill:none;stroke:var(--accent);stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
.savings{fill:var(--accent);opacity:.07}
.p-observed{fill:var(--accent-2);stroke:var(--panel);stroke-width:1.6}
.obs-line{stroke:var(--accent-2);stroke-width:1;stroke-dasharray:4 3;opacity:.85}
.obs-label{fill:var(--accent-2);font-family:var(--font-mono);font-size:10px}
.area-fill{fill:url(#areaGrad)}
.series-line{fill:none;stroke:var(--accent-2);stroke-width:1.9;stroke-linejoin:round;
  stroke-linecap:round}
.endpoint{fill:var(--ink);stroke:var(--panel);stroke-width:1.5}
.endpoint--amber{fill:var(--accent-2)}
/* stacked bars */
.seg-uncached{fill:var(--accent)}
.seg-cache-read{fill:var(--accent-dim)}
.seg-cache-write{fill:var(--accent);opacity:.42}
.seg-output{fill:var(--accent-2)}
.seg-reasoning{fill:var(--accent-3);opacity:.85}
/* ---------- retry hero ---------- */
.retry-hero{display:grid;grid-template-columns:minmax(240px,320px) 1fr;gap:26px;align-items:center}
@media (max-width:820px){.retry-hero{grid-template-columns:1fr}}
.waste-figure{display:flex;flex-direction:column;gap:4px}
.waste-num{
  font-family:var(--font-mono);font-size:clamp(52px,7.4vw,86px);line-height:.98;
  letter-spacing:-.045em;color:var(--accent-3);font-variant-numeric:tabular-nums;
  text-shadow:0 0 44px color-mix(in srgb,var(--accent-3) 30%,transparent);
}
.waste-cap{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3)}
.waste-sub{font-family:var(--font-mono);font-size:12px;color:var(--ink-2);margin-top:2px}
.meter{height:9px;border-radius:999px;background:var(--panel-2);overflow:hidden;
  border:1px solid var(--line);display:flex}
.meter-fill{background:linear-gradient(90deg,var(--accent-dim),var(--accent-3));height:100%}
.meter-scale{display:flex;justify-content:space-between;font-family:var(--font-mono);
  font-size:9.5px;color:var(--ink-3);margin-top:6px;letter-spacing:.08em}
.retry-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.stat{border-left:2px solid var(--line);padding-left:12px}
.stat .k{font-family:var(--font-mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3)}
.stat .v{font-family:var(--font-mono);font-size:16px;color:var(--ink)}
.stat .v.waste{color:var(--accent-3)}
.stat .v.warn{color:var(--accent-2)}
.errchips{display:flex;flex-wrap:wrap;gap:7px;margin-top:16px}
/* ---------- tables ---------- */
.table-wrap{overflow-x:auto;margin:0 -4px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{
  font-family:var(--font-mono);font-size:9.6px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;font-weight:500;padding:0 10px 9px;
  border-bottom:1px solid var(--line);white-space:nowrap;
}
tbody td{padding:8px 10px;border-bottom:1px solid var(--line-soft);vertical-align:middle}
tbody tr:hover td{background:var(--panel-2)}
tbody tr:last-child td{border-bottom:none}
.num{font-family:var(--font-mono);font-variant-numeric:tabular-nums;text-align:right;
  white-space:nowrap}
.model{font-family:var(--font-mono);font-size:12px;color:var(--ink)}
.strategy{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-2)}
.rank{
  display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
  border-radius:6px;font-family:var(--font-mono);font-size:11px;border:1px solid var(--line);
  color:var(--ink-2);background:var(--panel-2);
}
.rank--1{color:#0b0f14;background:linear-gradient(180deg,#fde68a,#f59e0b);border-color:#f59e0b;font-weight:700}
.rank--2{color:#0b0f14;background:linear-gradient(180deg,#e8eef5,#aab6c4);border-color:#aab6c4;font-weight:700}
.rank--3{color:#0b0f14;background:linear-gradient(180deg,#f2c9a0,#c2834b);border-color:#c2834b;font-weight:700}
.badge{
  display:inline-block;padding:2px 8px;border-radius:999px;font-family:var(--font-mono);
  font-size:10px;letter-spacing:.07em;text-transform:uppercase;border:1px solid var(--line);
  color:var(--ink-2);white-space:nowrap;
}
.badge--good{color:var(--good);border-color:color-mix(in srgb,var(--good) 45%,transparent);
  background:color-mix(in srgb,var(--good) 12%,transparent)}
.badge--bad{color:var(--accent-3);border-color:color-mix(in srgb,var(--accent-3) 45%,transparent);
  background:color-mix(in srgb,var(--accent-3) 12%,transparent)}
.badge--warn{color:var(--accent-2);border-color:color-mix(in srgb,var(--accent-2) 45%,transparent);
  background:color-mix(in srgb,var(--accent-2) 12%,transparent)}
.spark{height:5px;border-radius:999px;background:var(--panel-2);min-width:64px;overflow:hidden}
.spark span{display:block;height:100%;background:var(--accent);opacity:.85}
.delta{font-family:var(--font-mono);font-variant-numeric:tabular-nums;text-align:right;
  white-space:nowrap}
.delta--up{color:var(--accent-2)}
.delta--down{color:var(--good)}
.delta--flat{color:var(--ink-3)}
/* ---------- misc ---------- */
.empty{
  border:1px dashed var(--line);border-radius:8px;padding:22px;text-align:center;
  color:var(--ink-3);font-family:var(--font-mono);font-size:11.5px;
}
footer{margin-top:46px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--ink-3);font-size:11.5px}
footer h3{font-family:var(--font-mono);font-size:10px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-2);margin:0 0 10px}
footer ul{margin:0;padding-left:16px}
footer li{margin:3px 0}
footer .colophon{margin-top:18px;font-family:var(--font-mono);font-size:10.5px;
  letter-spacing:.05em;display:flex;flex-wrap:wrap;gap:6px 18px}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0,0,0,0);white-space:nowrap;border:0}
/* ---------- print ---------- */
@media print{
  :root{
    --bg:#fff;--bg-soft:#fff;--panel:#fff;--panel-2:#f4f6f9;--line:#c9d3de;
    --line-soft:#e4eaf1;--ink:#0f172a;--ink-2:#3d4b5c;--ink-3:#6b7a8c;
    --grid:#e7ecf2;--axis:#b8c4d2;--accent:#0e7490;--accent-dim:#67c6d8;
    --accent-2:#b45309;--accent-3:#be185d;--good:#047857;
  }
  body{background-image:none;font-size:10.5pt}
  .wrap{max-width:none;padding:0}
  .card{border-color:#c9d3de;break-inside:avoid;page-break-inside:avoid;box-shadow:none}
  section{break-inside:avoid}
  .table-wrap{overflow:visible}
  tbody tr:hover td{background:transparent}
  .waste-num{text-shadow:none}
  .kpi--waste .v{text-shadow:none}
  footer{break-inside:avoid}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


# --------------------------------------------------------------------------- #
# Label placement (used by the Pareto hero so labels never overlap)
# --------------------------------------------------------------------------- #
class _LabelPlacer:
    """Greedy non-overlapping label placement with leader lines.

    A scatter plot with per-point labels looks amateur the moment two labels
    touch — and real tokbench runs make that inevitable: the flagship run spans
    41x in cost with 9 of 15 pairs inside the cheapest 5% of a linear axis.

    Strategy: each label tries a ladder of candidate slots (right, left, above,
    below, then progressively further out).  A slot is rejected if its box hits
    an already-placed label, a data marker, or the plot bounds.  When every
    near slot is taken the label is pushed to an outer column and connected to
    its marker by a hairline leader, which keeps dense clusters readable
    instead of dropping labels or letting them overlap.

    Placement is deterministic (same payload -> same layout), so reports stay
    diff-able.
    """

    #: Advance width of the 10px mono label font, in px.
    CHAR_W = 5.95
    #: Label box height in px (font-size 10 + leading).
    LINE_H = 12.0
    #: Horizontal run of a leader line, in px.
    LEADER_RUN = 16.0

    def __init__(self, bounds: tuple[float, float, float, float]) -> None:
        self.x0, self.y0, self.x1, self.y1 = bounds
        self._boxes: list[tuple[float, float, float, float]] = []
        #: Labels that could not sit adjacent to their marker.
        self._leaders: list[str] = []

    def reserve(self, cx: float, cy: float, r: float) -> None:
        """Mark a marker's footprint so labels keep clear of it."""
        self._boxes.append((cx - r, cy - r, cx + r, cy + r))

    def reserve_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Reserve an arbitrary rectangle (used for the knee callout box)."""
        self._boxes.append((x0, y0, x1, y1))

    def _collides(self, box: tuple[float, float, float, float]) -> bool:
        bx0, by0, bx1, by1 = box
        for ox0, oy0, ox1, oy1 in self._boxes:
            if bx0 < ox1 and bx1 > ox0 and by0 < oy1 and by1 > oy0:
                return True
        return False

    def _box_for(
        self, lx: float, ly: float, anchor: str, width: float
    ) -> tuple[float, float, float, float]:
        """Bounding box of a label drawn at (lx, ly) with the given anchor."""
        if anchor == "start":
            bx0, bx1 = lx, lx + width
        elif anchor == "end":
            bx0, bx1 = lx - width, lx
        else:
            bx0, bx1 = lx - width / 2.0, lx + width / 2.0
        return (bx0, ly - self.LINE_H + 2.0, bx1, ly + 3.0)

    def place(self, cx: float, cy: float, text: str) -> tuple[float, float, str] | None:
        """Return ``(x, y, anchor)`` for the first free slot, else ``None``.

        The ladder widens in stages: tight adjacent slots first (they need no
        leader), then offset slots one line up/down, then the outer columns.
        """
        width = max(len(text), 1) * self.CHAR_W
        # Stage 1: adjacent, no leader needed.
        # Stage 2: one line of vertical stagger.
        # Stage 3: outer columns (leader lines drawn).
        candidates: list[tuple[float, float, str]] = [
            (cx + 9.0, cy + 3.6, "start"),
            (cx - 9.0, cy + 3.6, "end"),
            (cx + 9.0, cy - 8.0, "start"),
            (cx - 9.0, cy - 8.0, "end"),
            (cx + 9.0, cy + 15.0, "start"),
            (cx - 9.0, cy + 15.0, "end"),
            (cx, cy - 13.0, "middle"),
            (cx, cy + 20.0, "middle"),
            (cx + 9.0, cy - 20.0, "start"),
            (cx - 9.0, cy - 20.0, "end"),
        ]
        for lx, ly, anchor in candidates:
            box = self._box_for(lx, ly, anchor, width)
            if box[0] < self.x0 or box[2] > self.x1 or box[1] < self.y0 or box[3] > self.y1:
                continue
            if self._collides(box):
                continue
            self._boxes.append(box)
            return lx, ly, anchor
        return None

    def place_outer(self, cx: float, cy: float, text: str) -> tuple[float, float, str] | None:
        """Place a label in an outer column, recording a leader to the marker.

        Used as the last resort for points packed so tightly that no adjacent
        slot is free.  The label is stacked at the nearest free vertical slot on
        either side of the plot, and a leader line ties it back to its marker so
        the association is never ambiguous.
        """
        width = max(len(text), 1) * self.CHAR_W
        pad = self.LEADER_RUN + 8.0
        for anchor, edge in (("start", self.x0 + pad), ("end", self.x1 - pad)):
            # Search outward from the marker's own y, then alternate.
            for delta in (0.0, -14.0, 14.0, -28.0, 28.0, -42.0, 42.0, -56.0, 56.0, -70.0, 70.0):
                ly = cy + delta
                box = self._box_for(edge, ly, anchor, width)
                if box[1] < self.y0 or box[3] > self.y1:
                    continue
                if box[0] < self.x0 or box[2] > self.x1:
                    continue
                if self._collides(box):
                    continue
                self._boxes.append(box)
                self._leaders.append(self._leader(cx, cy, edge, ly, anchor))
                return edge, ly, anchor
        return None

    def _leader(self, cx: float, cy: float, lx: float, ly: float, anchor: str) -> str:
        """Hairline from a marker to an outer-column label."""
        # Stop short of the text itself so the line doesn't strike through it.
        end_x = lx - 3.0 if anchor == "start" else lx + 3.0
        return (
            f'<polyline class="leader" points="'
            f'{cx:.2f},{cy:.2f} {end_x:.2f},{ly - 3.5:.2f} {end_x:.2f},{ly - 3.5:.2f}"/>'
        )

    def leader_markup(self) -> str:
        """All leader lines, as one group (drawn beneath the labels)."""
        if not self._leaders:
            return ""
        return '<g class="leaders">' + "".join(self._leaders) + "</g>"


def _usd_axis_fmt(step: float) -> Any:
    """Pick a USD tick precision that matches the axis step size.

    ``step=0.002`` → ``$0.002``;  ``step=2.5`` → ``$2.50``.  Without this the
    axis either repeats identical labels (too few decimals) or reads like a
    float repr (too many).
    """
    if step <= 0 or not math.isfinite(step):
        return fmt_usd
    if step >= 1:
        return lambda v: f"${v:,.2f}"
    digits = max(2, min(8, int(-math.floor(math.log10(step)))))
    return lambda v: f"${v:.{digits}f}"


# --------------------------------------------------------------------------- #
# Chart 1 — Pareto frontier (the hero)
# --------------------------------------------------------------------------- #
def _knee_index(frontier: Sequence[Any]) -> int | None:
    """Index of the frontier "elbow": furthest perpendicular from the chord.

    Implements the knee heuristic documented in ``docs/METHODOLOGY.md`` §5.
    Computed here rather than read from the payload because
    :class:`~tokbench.report_schema.ParetoFrontier` has no ``knee_label`` field
    (see the schema-gap note in the report footer).  With fewer than three
    frontier points there is no interior elbow, so ``None`` is returned.
    """
    pts = list(frontier)
    if len(pts) < 3:
        return None
    ax, ay = _finite(pts[0].cost_per_task_usd), _finite(pts[0].quality)
    bx, by = _finite(pts[-1].cost_per_task_usd), _finite(pts[-1].quality)
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm <= 0:
        return None
    best_i, best_d = None, -1.0
    for i in range(1, len(pts) - 1):
        px = _finite(pts[i].cost_per_task_usd)
        py = _finite(pts[i].quality)
        # |cross| / |chord| = perpendicular distance, normalised to the chord
        # length so the metric is scale-free.
        dist = abs(dx * (py - ay) - dy * (px - ax)) / norm
        if dist > best_d:
            best_i, best_d = i, dist
    return best_i


def _layout_pareto_labels(
    entries: Sequence[tuple[float, float, str, bool]],
    bounds: tuple[float, float, float, float],
    *,
    obstacle: tuple[float, float, float, float] | None = None,
) -> tuple[list[tuple[float, float, str, str]], list[str]]:
    """Assign every Pareto label a non-overlapping slot inside the plot area.

    This is the classic **label-column** layout, and it is what makes a dense
    scatter readable:

    1. Each label is pushed to the side of its marker that points *into* the
       plot (left-half markers label right, right-half markers label left), so
       labels stay on canvas and the two columns don't fight.
    2. Within each column, labels are sorted by y and then **relaxed**: walking
       down the column, each label is pushed down until it clears its
       predecessor by at least one line.  That single pass removes every
       vertical collision in a dense cluster.
    3. If the column overflows the plot bottom, the whole column shifts up; if
       it still cannot fit, the inter-label gap is compressed to whatever the
       available height allows (rather than dropping labels).
    4. A label that had to move far from its marker gets a hairline **leader**
       so the association is never ambiguous.

    Args:
        entries: ``(cx, cy, text, on_frontier)`` in marker pixel coordinates.
        bounds: Plot rectangle ``(x0, y0, x1, y1)``.
        obstacle: Optional rect (the knee callout) that labels must avoid.

    Returns:
        ``(placements, leaders)`` where each placement is
        ``(x, y, anchor, text)`` and each leader is ready-to-emit SVG.
    """
    x0, y0, x1, y1 = bounds
    char_w = 5.95
    line_h = 12.0
    baseline = 3.6  # visual centre offset for a 10px label

    mid_x = (x0 + x1) / 2.0
    columns: dict[str, list[tuple[float, float, str, bool]]] = {
        "start": [],  # label to the RIGHT of its marker
        "end": [],  # label to the LEFT of its marker
    }
    for cx, cy, text, on_front in entries:
        anchor = "start" if cx <= mid_x else "end"
        columns[anchor].append((cx, cy, text, on_front))

    available = (y1 - y0) - line_h
    placements: list[tuple[float, float, str, str]] = []
    leaders: list[str] = []

    for anchor, items in columns.items():
        if not items:
            continue
        items.sort(key=lambda e: (e[1], e[0]))
        count = len(items)
        # Pack labels as tightly as the font allows: 12px line pitch is the
        # minimum for 10px text, and using the minimum keeps labels close to
        # their own markers instead of drifting down the column.
        gap = line_h + 1.0
        if count > 1:
            gap = min(gap, max(available / (count - 1), line_h * 0.9))

        # Seed each label at its marker's own y (clamped), then relax downward.
        ys = [max(y0 + line_h, min(cy + baseline, y1 - 2.0)) for _, cy, _, _ in items]
        for i in range(1, count):
            if ys[i] - ys[i - 1] < gap:
                ys[i] = ys[i - 1] + gap
        # Overflow at the bottom: lift the whole column and re-relax.
        overflow = ys[-1] - (y1 - 2.0)
        if overflow > 0:
            ys = [y - overflow for y in ys]
        for i in range(count):
            ys[i] = max(ys[i], y0 + line_h)
        for i in range(1, count):
            if ys[i] - ys[i - 1] < gap:
                ys[i] = ys[i - 1] + gap

        for (cx, cy, text, _on_front), ly in zip(items, ys, strict=False):
            lx = cx + 9.0 if anchor == "start" else cx - 9.0
            width = len(text) * char_w
            # Keep the text box inside the plot horizontally.
            if anchor == "start":
                lx = min(lx, x1 - width - 2.0)
                lx = max(lx, x0 + 2.0)
            else:
                lx = max(lx, x0 + width + 2.0)
                lx = min(lx, x1 - 2.0)
            # Step around the knee callout if this slot lands on it.
            if obstacle is not None:
                ox0, oy0, ox1, oy1 = obstacle
                box_x0 = lx if anchor == "start" else lx - width
                box_x1 = box_x0 + width
                if box_x0 < ox1 and box_x1 > ox0 and ly > oy0 - line_h and ly < oy1 + 4:
                    ly = oy1 + line_h + 3.0
                    if ly > y1 - 2.0:
                        ly = oy0 - 5.0
            placements.append((lx, ly, anchor, text))
            # Leader only when the label had to travel to reach its slot.
            if abs(ly - (cy + baseline)) > line_h + 2.0:
                end_x = lx - 3.0 if anchor == "start" else lx + 3.0
                leaders.append(
                    f'<polyline class="leader" points="'
                    f'{cx:.2f},{cy:.2f} {end_x:.2f},{ly - 3.5:.2f}"/>'
                )

    return placements, leaders


def _pareto_chart(payload: ReportPayload, pal: Palette) -> str:
    """Hero chart: cost-per-task vs mean quality, frontier emphasised.

    Visual grammar:
      * **frontier** — cyan line + filled markers; the only saturated element,
        so the eye lands there first;
      * **dominated** — grey and small: context, not conclusions;
      * **knee** — amber diamond + callout, the pragmatic recommendation;
      * **baseline** — dashed hollow ring, the thing everything is compared to.

    Three deliberate choices worth defending:

    *Log cost axis when the span is wide.*  Real runs span 1-2 decades of
    cost per task (the flagship run: $0.00008 -> $0.0033, 41x).  On a linear
    axis nine of fifteen pairs collapse into the leftmost 5% of the plot and
    their labels are unreadable no matter how cleverly they are placed.  A
    log10 axis gives every order of magnitude equal room, which is also the
    economically honest reading — cost differences are multiplicative.  Narrow
    spans stay linear, because log axes are harder to read and there is no
    reason to pay that cost when the data does not need it.

    *The quality axis is not zero-based.*  A realistic spread is 0.41->0.91;
    starting at 0 would compress every difference into the top third and hide
    exactly the trade-off the chart exists to show.  The truncation is disclosed
    in the caption.

    *The frontier is not filled.*  A filled "efficient region" was tried and
    dropped: with a truncated axis it renders as a stray background panel with
    a hard right edge.  Emphasis comes from the line and its glow instead.
    """
    pareto: ParetoFrontier = payload.pareto
    points = list(pareto.points)
    if not points:
        return _empty_figure("pareto-empty", "No Pareto data in this payload.")

    W, H = 1120.0, 560.0
    x0, x1 = 96.0, 986.0
    y0, y1 = 78.0, 462.0

    costs = [_finite(p.cost_per_task_usd) for p in points]
    quals = [_finite(p.quality) for p in points]
    positive = [c for c in costs if c > 0]
    if not positive:
        positive = [1e-6]

    # ---- choose the cost scale -------------------------------------------
    # A decade or more of span is the cue for log; below that, linear reads
    # more naturally and avoids implying ratios that aren't there.
    span_ratio = (max(positive) / min(positive)) if min(positive) > 0 else 1.0
    use_log = span_ratio >= 10.0

    if use_log:
        lo_v = min(positive) / 1.35
        hi_v = max(positive) * 1.35
        xs = LogScale(lo_v, hi_v, x0, x1)
        x_ticks = _log_ticks(lo_v, hi_v)
        fmt_x = _usd_log_fmt(x_ticks)
    else:
        d_lo, d_hi = _pad_domain(costs, pad_frac=0.07)
        d_lo = max(d_lo, 0.0)
        x_ticks, x_step = nice_ticks(d_lo, d_hi, target=6)
        d_lo, d_hi = min(d_lo, x_ticks[0]), max(d_hi, x_ticks[-1])
        xs = LinearScale(d_lo, d_hi, x0, x1)
        fmt_x = _usd_axis_fmt(x_step)

    q_lo, q_hi = _pad_domain(quals, pad_frac=0.05)
    q_lo, q_hi = max(q_lo, 0.0), min(q_hi, 1.0)
    y_ticks, _ = nice_ticks(q_lo, q_hi, target=6)
    q_lo, q_hi = min(q_lo, y_ticks[0]), max(q_hi, y_ticks[-1])
    ys = LinearScale(q_lo, q_hi, y1, y0)

    # ``on_frontier`` normally comes from the analysis layer; derive it when it
    # was never set so an older payload still yields a correct hero chart.
    frontier = [p for p in points if p.on_frontier] or _derive_frontier(points)
    frontier.sort(key=lambda p: (_finite(p.cost_per_task_usd), _finite(p.quality)))
    frontier_ids = {id(p) for p in frontier}

    # The analysis layer publishes the knee explicitly (``ParetoFrontier.knee_label``)
    # so every consumer agrees on where it is.  ``_knee_index`` remains only as a
    # fallback for payloads produced before that field existed.
    knee: Any | None = None
    if getattr(pareto, "knee_label", None):
        knee = next((p for p in frontier if p.label == pareto.knee_label), None)
    if knee is None:
        knee_i = _knee_index(frontier)
        knee = frontier[knee_i] if knee_i is not None else None
    baseline = next(
        (p for p in points if pareto.baseline_label and p.label == pareto.baseline_label),
        None,
    )
    dominated = [p for p in points if id(p) not in frontier_ids]

    scale_note = (
        "Cost is plotted on a log10 axis because the pairs span more than a "
        "decade; equal horizontal distance means an equal ratio, not an equal "
        "dollar amount."
        if use_log
        else "Cost is plotted on a linear axis."
    )
    out: list[str] = [
        _svg_open(
            W,
            H,
            label=(
                "Pareto frontier: cost per task versus mean quality across "
                f"{len(points)} model/strategy pairs"
            ),
            desc=(
                "Scatter plot. X axis is cost per task in USD"
                + (" on a logarithmic scale" if use_log else "")
                + "; Y axis is mean quality from 0 to 1, truncated to the "
                f"observed range. {len(frontier)} points are on the Pareto "
                "frontier and are connected by an ascending line; the rest are "
                "dominated and dimmed. The knee point is annotated."
            ),
        ),
        _axis_frame(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_scale=xs,
            y_scale=ys,
            fmt_x=fmt_x,
            fmt_y=lambda v: f"{v:.2f}",
            x_label=("cost per task (usd, log scale)" if use_log else "cost per task (usd)"),
            y_label="mean quality (0-1)",
        ),
    ]

    # ---- dominated points (drawn first: they sit behind the frontier) ------
    placer = _LabelPlacer((x0, y0, x1, y1))
    for p in dominated:
        placer.reserve(xs(_finite(p.cost_per_task_usd)), ys(_finite(p.quality)), 5.5)
    for p in frontier:
        placer.reserve(xs(_finite(p.cost_per_task_usd)), ys(_finite(p.quality)), 8.5)

    out.append('<g class="dominated">')
    for p in dominated:
        cx, cy = xs(_finite(p.cost_per_task_usd)), ys(_finite(p.quality))
        tip = (
            f"{p.label} \u2014 dominated\n"
            f"cost/task {fmt_usd(p.cost_per_task_usd)} \u00b7 "
            f"quality {fmt_quality(p.quality)}"
        )
        if p.dominated_by:
            tip += "\ndominated by: " + ", ".join(str(d) for d in p.dominated_by[:3])
        out.append(_circle(cx, cy, 3.4, cls="p-dom", tip=tip))
    out.append("</g>")

    # ---- frontier line + markers -------------------------------------------
    if len(frontier) >= 2:
        fpts = [(xs(_finite(p.cost_per_task_usd)), ys(_finite(p.quality))) for p in frontier]
        out.append(_polyline(fpts, cls="fline-glow"))
        out.append(_polyline(fpts, cls="fline"))

    out.append('<g class="frontier">')
    for p in frontier:
        cx, cy = xs(_finite(p.cost_per_task_usd)), ys(_finite(p.quality))
        is_knee = knee is not None and p is knee
        if baseline is not None and p is baseline:
            out.append(_circle(cx, cy, 6.4, cls="p-baseline"))
        if is_knee:
            out.append(
                f"<g><title>{esc(_pareto_tip(p, knee=True))}</title>"
                + _diamond(cx, cy, 6.6, cls="p-knee")
                + "</g>"
            )
        else:
            out.append(_circle(cx, cy, 4.6, cls="p-front", tip=_pareto_tip(p)))
    out.append("</g>")

    # ---- knee callout geometry (computed before labels so it can be dodged) --
    knee_geom: tuple[float, float, float, float] | None = None
    if knee is not None:
        knee_geom = _knee_box_geometry(
            kx=xs(_finite(knee.cost_per_task_usd)),
            ky=ys(_finite(knee.quality)),
            label=knee.label,
            cost=knee.cost_per_task_usd,
            quality=knee.quality,
            bounds=(x0, y0, x1, y1),
        )
        kbx, kby, kbw, kbh = knee_geom
        placer.reserve_box(kbx - 4.0, kby - 4.0, kbx + kbw + 4.0, kby + kbh + 4.0)

    # ---- labels ------------------------------------------------------------
    # A single column-relaxation pass (see _layout_pareto_labels) instead of
    # per-label slot probing: it guarantees no two labels overlap, which is the
    # failure mode that shows up as soon as several pairs share a cost band.
    entries: list[tuple[float, float, str, bool]] = [
        (
            xs(_finite(p.cost_per_task_usd)),
            ys(_finite(p.quality)),
            ellipsis(p.label, LABEL_BUDGET),
            id(p) in frontier_ids,
        )
        for p in frontier + dominated
    ]
    placements, leaders = _layout_pareto_labels(
        entries,
        (x0, y0, x1, y1),
        obstacle=(
            (
                knee_geom[0] - 4.0,
                knee_geom[1] - 4.0,
                knee_geom[0] + knee_geom[2] + 4.0,
                knee_geom[1] + knee_geom[3] + 4.0,
            )
            if knee_geom is not None
            else None
        ),
    )
    front_labels = {ellipsis(p.label, LABEL_BUDGET) for p in frontier}
    if leaders:
        out.append('<g class="leaders">' + "".join(leaders) + "</g>")
    labels: list[str] = ['<g class="labels">']
    for lx, ly, anchor, text in placements:
        cls = "plabel plabel--front" if text in front_labels else "plabel"
        labels.append(_text(lx, ly, text, cls=cls, anchor=anchor))
    labels.append("</g>")
    out.append("".join(labels))

    # ---- knee callout ------------------------------------------------------
    if knee is not None and knee_geom is not None:
        kbx, kby, kbw, kbh = knee_geom
        kx = xs(_finite(knee.cost_per_task_usd))
        ky = ys(_finite(knee.quality))
        anchor_x = kbx if kx > kbx + kbw / 2.0 else kbx + kbw
        detail = (
            f"{ellipsis(knee.label, 24)} \u00b7 {fmt_usd(knee.cost_per_task_usd)}/task"
            f" \u00b7 q {fmt_quality(knee.quality)}"
        )
        out.append(
            '<g class="knee">'
            + _line(kx, ky, anchor_x, kby + kbh / 2.0, cls="knee-line")
            + _rect(kbx, kby, kbw, kbh, cls="knee-box", rx=6.0)
            + _text(kbx + 12.0, kby + 18.0, "knee point", cls="knee-label")
            + _text(kbx + 12.0, kby + 34.0, detail, cls="knee-text")
            + "</g>"
        )

    # ---- legend + scale note ----------------------------------------------
    legend_items = [("front", "pareto frontier"), ("dominated", "dominated")]
    if knee is not None:
        legend_items.append(("knee", "knee"))
    if baseline is not None:
        legend_items.append(("baseline", "baseline"))
    out.append(_legend(legend_items, x1, 34.0))
    out.append(_text(x0, y1 + 68.0, scale_note.split(";")[0] + ".", cls="note"))

    out.append("</svg>")
    return "".join(out)


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Decade ticks (1-2-5 style) covering ``[lo, hi]``.

    Returns the values that should be labelled; for each decade it emits the
    1x and 5x marks, which is enough orientation without turning a log axis into
    a ladder of near-identical numbers.
    """
    if lo <= 0 or hi <= lo:
        return [lo, hi]
    start = math.floor(math.log10(lo))
    end = math.ceil(math.log10(hi))
    ticks: list[float] = []
    for decade in range(start, end + 1):
        for mult in (1.0, 2.0, 5.0):
            value = mult * (10.0**decade)
            if lo * 0.999 <= value <= hi * 1.001:
                ticks.append(value)
    return ticks or [lo, hi]


def _usd_log_fmt(ticks: Sequence[float]) -> Any:
    """Label formatter for log ticks: adapts precision to the tick magnitude."""

    def fmt(value: float) -> str:
        if value <= 0:
            return fmt_usd(value)
        if value >= 1:
            return f"${value:,.2f}"
        if value >= 0.01:
            return f"${value:.3f}"
        if value >= 0.0001:
            return f"${value:.5f}"
        return f"${value:.7f}"

    return fmt


class LogScale:
    """Map a positive domain onto a pixel range on a base-10 log axis."""

    def __init__(self, lo: float, hi: float, px_lo: float, px_hi: float) -> None:
        self.lo = lo if lo > 0 else 1e-12
        self.hi = hi if hi > self.lo else self.lo * 10
        self.px_lo = px_lo
        self.px_hi = px_hi

    def __call__(self, value: float) -> float:
        v = value if value > 0 else self.lo
        lo_l, hi_l = math.log10(self.lo), math.log10(self.hi)
        span = hi_l - lo_l
        if span <= 0:
            return (self.px_lo + self.px_hi) / 2.0
        t = (math.log10(v) - lo_l) / span
        return self.px_lo + t * (self.px_hi - self.px_lo)


def _knee_box_geometry(
    *,
    kx: float,
    ky: float,
    label: str,
    cost: float,
    quality: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Pick the knee callout's rectangle: ``(x, y, w, h)``, fully inside bounds.

    Split out from the drawing so the caller can *reserve* the box with the
    label placer before any label is positioned — otherwise point labels end up
    hidden underneath the callout.

    The box is sized to its text rather than to a fixed width, and the four
    quadrants around the knee are tried in preference order because a knee
    typically sits in the middle-left of the plot where a naive up-right offset
    can overflow the canvas.
    """
    x0, y0, x1, y1 = bounds
    detail = f"{ellipsis(label, 24)} \u00b7 {fmt_usd(cost)}/task \u00b7 q {fmt_quality(quality)}"
    # 5.95 px/char is the advance width of the 10.5px mono label font; +26 px
    # covers the horizontal padding.
    box_w = min(max(len(detail) * 5.95 + 26.0, 200.0), (x1 - x0) - 8.0)
    box_h = 46.0
    gap = 20.0
    candidates = [
        (kx + gap, ky - box_h - gap),  # up-right (preferred)
        (kx - gap - box_w, ky - box_h - gap),  # up-left
        (kx + gap, ky + gap),  # down-right
        (kx - gap - box_w, ky + gap),  # down-left
    ]
    for cx, cy in candidates:
        if x0 + 4 <= cx <= x1 - box_w - 4 and y0 + 4 <= cy <= y1 - box_h - 4:
            return cx, cy, box_w, box_h
    # Nothing fits cleanly: clamp the preferred slot into the plot area.
    bx = min(max(kx + gap, x0 + 4), max(x1 - box_w - 4, x0 + 4))
    by = min(max(ky - box_h - gap, y0 + 4), max(y1 - box_h - 4, y0 + 4))
    return bx, by, box_w, box_h


def _pareto_tip(point: Any, *, knee: bool = False) -> str:
    """Hover text for a Pareto point — the full precision the labels omit."""
    lines = [str(point.label)]
    if knee:
        lines[0] += " (knee)"
    lines.append(f"cost/task  {fmt_usd(point.cost_per_task_usd)}")
    lines.append(f"total cost {fmt_usd(point.cost_usd)}")
    lines.append(f"quality    {fmt_quality(point.quality)}")
    lines.append(f"tok/task   {fmt_tokens(point.billable_tokens_per_task)}")
    if point.value_score:
        lines.append(f"value      {fmt_ratio(point.value_score)}")
    return "\n".join(lines)


def _derive_frontier(points: Sequence[Any]) -> list[Any]:
    """Recompute the non-dominated set when the payload did not flag it.

    Same definition as ``docs/METHODOLOGY.md`` §5: A dominates B iff A is no
    more expensive *and* no worse in quality, with at least one strict
    improvement.  Keeping this fallback in the renderer means a payload produced
    by an older analysis build still yields a correct hero chart.
    """
    out: list[Any] = []
    for a in points:
        ac, aq = _finite(a.cost_per_task_usd), _finite(a.quality)
        dominated = False
        for b in points:
            if b is a:
                continue
            bc, bq = _finite(b.cost_per_task_usd), _finite(b.quality)
            if bc <= ac and bq >= aq and (bc < ac or bq > aq):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


# --------------------------------------------------------------------------- #
# Chart 2 — retry cost share ("what did we burn?")
# --------------------------------------------------------------------------- #
def _retry_panel(retry: RetryAnalysis, pal: Palette) -> str:
    """Headline retry-waste panel: giant % number + per-pair stacked bars.

    The stacked bar splits *first-attempt* spend (cool, expected) from *retry*
    spend (magenta, pure waste) on a shared absolute USD scale, so the bars are
    comparable across rows and the magenta block is visually self-explanatory.
    """
    rows = list(retry.rows)
    share = _finite(retry.overall_retry_cost_share)
    waste = _finite(retry.total_retry_cost_usd)
    total = _finite(retry.total_cost_usd)

    parts: list[str] = ['<div class="retry-hero">']
    # -- the unmissable number ------------------------------------------------
    parts.append('<div class="waste-figure">')
    parts.append('<span class="waste-cap">retry cost share</span>')
    parts.append(f'<span class="waste-num">{esc(fmt_pct(share, 1))}</span>')
    parts.append(
        f'<span class="waste-sub">{esc(fmt_usd(waste))} of {esc(fmt_usd(total))} total spend</span>'
    )
    pct = max(0.0, min(share * 100.0, 100.0))
    parts.append(
        '<div class="meter" role="img" '
        f'aria-label="retry cost share {esc(fmt_pct(share, 1))}">'
        f'<div class="meter-fill" style="width:{pct:.2f}%"></div></div>'
    )
    parts.append('<div class="meter-scale"><span>0%</span><span>50%</span><span>100%</span></div>')
    parts.append("</div>")

    # -- supporting stats -----------------------------------------------------
    parts.append('<div class="retry-stats">')
    stats: list[tuple[str, str, str]] = [
        ("retry tokens", fmt_tokens(retry.total_retry_tokens), ""),
        ("token share", fmt_pct(retry.retry_token_share, 1), "waste"),
        ("wasted budget", fmt_pct(retry.wasted_pct_of_budget, 1), "waste"),
        ("retry rate", _aggregate_retry_rate(rows), "warn"),
        ("worst offender", ellipsis(retry.worst_offender or MISSING, 20), "waste"),
    ]
    for label, value, tone in stats:
        cls = f"v {tone}".strip()
        parts.append(
            f'<div class="stat"><div class="k">{esc(label)}</div>'
            f'<div class="{esc(cls)}">{esc(value)}</div></div>'
        )
    parts.append("</div>")
    parts.append("</div>")  # /retry-hero

    # -- error breakdown chips ------------------------------------------------
    if retry.error_breakdown:
        items = sorted(retry.error_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
        chips = "".join(
            f'<span class="chip chip--waste">{esc(kind)} · {esc(fmt_int(count))}</span>'
            for kind, count in items[:8]
        )
        parts.append(f'<div class="errchips">{chips}</div>')

    # -- per-pair stacked bars ------------------------------------------------
    if rows:
        parts.append(_retry_bars(rows, pal))
    return "".join(parts)


def _aggregate_retry_rate(rows: Sequence[Any]) -> str:
    """Run-level retry rate = retried cells / cells, recomputed from rows.

    The payload exposes per-row rates but no single overall rate, so we
    aggregate the raw counts (rates cannot be averaged — that would weight a
    1-cell pair the same as a 90-cell pair).
    """
    cells = sum(int(_finite(getattr(r, "cells", 0))) for r in rows)
    retried = sum(int(_finite(getattr(r, "retried_cells", 0))) for r in rows)
    if cells <= 0:
        return MISSING
    return fmt_pct(retried / cells, 1)


def _retry_bars(rows: Sequence[Any], pal: Palette) -> str:
    """Horizontal stacked bars: first-attempt spend vs retry spend per pair.

    Layout is a three-band header so nothing collides: the legend gets its own
    line at the very top, the axis scale sits below it, and the per-row value
    columns are right-aligned inside a reserved gutter.
    """
    ranked = sorted(
        rows,
        key=lambda r: (-_finite(r.retry_cost_share), -_finite(r.retry_cost_usd)),
    )
    shown = ranked[:MAX_CHART_ROWS]
    row_h = 34.0
    label_w = 236.0
    W = 1120.0
    # Reserve a right gutter wide enough for "$12.34" plus the share column.
    share_x = W - 128.0
    cost_x = W - 78.0
    bar_x1 = W - 148.0
    top = 74.0
    H = top + row_h * len(shown) + 18.0
    bar_x0 = label_w

    max_cost = max((_finite(r.total_cost_usd) for r in shown), default=0.0) or 1.0
    bar_scale = LinearScale(0.0, max_cost, bar_x0, bar_x1)
    ticks, step = nice_ticks(0.0, max_cost, target=4)

    out = [
        _svg_open(
            W,
            H,
            label="Retry cost share per model and strategy",
            desc=(
                "Horizontal stacked bars. The cool segment is first-attempt "
                "spend, the magenta segment is spend burned on retries. Rows are "
                "sorted by retry cost share, highest first."
            ),
        )
    ]
    fmt_tick = _usd_axis_fmt(step)

    # ---- band 1: legend on its own line ------------------------------------
    out.append(
        _legend(
            [("first", "first-attempt spend"), ("waste", "retry spend (waste)")],
            W - 20.0,
            22.0,
        )
    )
    # ---- band 2: axis title + scale ticks ----------------------------------
    # The axis title sits to the LEFT of the plot band (right-aligned into the
    # row-label gutter) so it can never collide with the $0.00 tick.
    out.append(_text(bar_x0 - 14.0, 50.0, "cumulative cost (usd)", cls="bar-head", anchor="end"))
    for tick in ticks:
        tx = bar_scale(tick)
        if tx > bar_x1 + 0.5:
            continue
        out.append(_text(tx, 50.0, fmt_tick(tick), cls="tick tick--x"))
    out.append(_text(share_x, 50.0, "retry share", cls="bar-head", anchor="start"))
    out.append(_text(cost_x, 50.0, "total cost", cls="bar-head", anchor="start"))

    # ---- band 3: rows ------------------------------------------------------
    for i, row in enumerate(shown):
        y = top + i * row_h
        total = _finite(row.total_cost_usd)
        retry_cost = _finite(row.retry_cost_usd)
        first = max(total - retry_cost, 0.0)
        label = f"{row.model_id} \u00b7 {row.strategy_id}"
        hi = _finite(row.retry_cost_share) >= 0.15
        out.append(
            _text(
                bar_x0 - 14.0,
                y + 18.0,
                ellipsis(label, 34),
                cls="rowlabel rowlabel--hi" if hi else "rowlabel",
                anchor="end",
            )
        )
        tip = (
            f"{label}\nretry cost {fmt_usd(retry_cost)} of {fmt_usd(total)}\n"
            f"retry rate {fmt_pct(row.retry_rate, 1)} \u00b7 "
            f"{int(_finite(row.retried_cells))}/{int(_finite(row.cells))} cells retried\n"
            f"extra attempts {fmt_int(row.extra_attempts)}"
        )
        out.append(f"<g><title>{esc(tip)}</title>")
        out.append(_rect(bar_x0, y + 7.0, bar_x1 - bar_x0, 15.0, cls="track", rx=2.0))
        w_first = bar_scale(first) - bar_x0
        w_retry = bar_scale(retry_cost) - bar_x0
        out.append(_rect(bar_x0, y + 7.0, w_first, 15.0, cls="seg-first"))
        out.append(_rect(bar_x0 + w_first, y + 7.0, w_retry, 15.0, cls="seg-waste"))
        out.append("</g>")
        share_cls = "rowval rowval--waste" if hi else "rowval"
        out.append(
            _text(
                share_x,
                y + 18.0,
                fmt_pct(row.retry_cost_share, 1),
                cls=share_cls,
                anchor="start",
            )
        )
        out.append(_text(cost_x, y + 18.0, fmt_usd(total), cls="rowval", anchor="start"))

    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Chart 3 — cache sensitivity
# --------------------------------------------------------------------------- #
def _cache_chart(cache: CacheSensitivity, pal: Palette) -> str:
    """Cost vs hypothetical cache hit rate, with the observed point pinned.

    The curve is analytically a straight line (METHODOLOGY §7), but it is drawn
    as a path through the payload's sampled points rather than as a formula —
    the report should visualise *what the analysis computed*, never a
    re-derivation that could silently disagree with it.
    """
    curve = list(cache.curve)
    if len(curve) < 2:
        return _empty_figure("cache-empty", "No cache sensitivity curve in this payload.")

    W, H = 1120.0, 430.0
    x0, x1 = 92.0, 1000.0
    y0, y1 = 68.0, 344.0

    hs = [_finite(p.cache_hit_rate) for p in curve]
    costs = [_finite(p.projected_cost_usd) for p in curve]
    d_lo, d_hi = min(0.0, min(hs)), max(1.0, max(hs))
    c_lo, c_hi = _pad_domain(costs, lo=0.0, pad_frac=0.10)

    x_ticks, x_step = nice_ticks(d_lo, d_hi, target=6)
    y_ticks, y_step = nice_ticks(c_lo, c_hi, target=5)
    c_lo, c_hi = min(c_lo, y_ticks[0]), max(c_hi, y_ticks[-1])

    xs = LinearScale(d_lo, d_hi, x0, x1)
    ys = LinearScale(c_lo, c_hi, y1, y0)

    obs_h = _finite(cache.observed_hit_rate)
    obs_cost = _finite(cache.observed_cost_usd) or _interpolate(curve, obs_h)
    ox, oy = xs(obs_h), ys(obs_cost)

    pts = [(xs(_finite(p.cache_hit_rate)), ys(_finite(p.projected_cost_usd))) for p in curve]

    out = [
        _svg_open(
            W,
            H,
            label="Cache sensitivity: projected cost versus cache hit rate",
            desc=(
                "Line chart of projected total cost in USD against hypothetical "
                "cache hit rate from 0 to 1. The observed hit rate is marked with "
                "a vertical dashed line and a labelled point."
            ),
        ),
        # shade the savings achieved so far: between zero-cache cost and the curve
        f'<path class="savings" d="M {x0:.2f},{ys(_finite(cache.zero_cache_cost_usd)):.2f} '
        + " L ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        + f' L {x1:.2f},{ys(_finite(cache.zero_cache_cost_usd)):.2f} Z"/>',
        _axis_frame(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_scale=xs,
            y_scale=ys,
            fmt_x=lambda v: fmt_pct(v, 0),
            fmt_y=_usd_axis_fmt(y_step),
            x_label="cache hit rate",
            y_label="projected total cost (usd)",
        ),
        _polyline(pts, cls="curve"),
    ]

    # observed marker
    out.append(_line(ox, y0, ox, y1, cls="obs-line"))
    # NB: the arrow is built as a separate variable rather than inlined as a
    # ``\u2192`` escape inside the nested f-string — a backslash in an f-string
    # expression is only legal on Python 3.12+, and this package supports 3.10.
    _observed_title = f"observed hit rate {fmt_pct(obs_h, 1)} \u2192 {fmt_usd(obs_cost)} projected"
    out.append(
        f"<g><title>{esc(_observed_title)}</title>"
        + _circle(ox, oy, 5.0, cls="p-observed")
        + "</g>"
    )
    # Label the observed point *above* the curve and keep it inside the canvas;
    # placing it to the right would run it along the descending line.
    label = f"observed {fmt_pct(obs_h, 1)} \u2192 {fmt_usd(obs_cost)}"
    label_w = len(label) * 5.95
    lx = min(max(ox + 12.0, x0 + 4.0), x1 - label_w - 4.0)
    out.append(_text(lx, max(oy - 14.0, y0 + 14.0), label, cls="obs-label"))

    # Endpoint annotations: pinned to opposite corners, away from the curve.
    # "no cache" is the h=0 intercept, so it goes under the left axis start;
    # "full cache" is the h=1 intercept, placed below-left of that endpoint.
    out.append(
        _text(
            x0 + 10.0,
            y1 - 10.0,
            f"no cache {fmt_usd(cache.zero_cache_cost_usd)}",
            cls="obs-label",
            anchor="start",
        )
    )
    out.append(
        _text(
            x1 - 10.0,
            min(ys(_finite(cache.full_cache_cost_usd)) + 20.0, y1 - 8.0),
            f"full cache {fmt_usd(cache.full_cache_cost_usd)}",
            cls="obs-label",
            anchor="end",
        )
    )
    out.append(
        _legend(
            [("line-accent", "projected cost"), ("observed", "observed hit rate")],
            x1,
            30.0,
        )
    )
    out.append("</svg>")
    return "".join(out)


def _interpolate(curve: Sequence[Any], h: float) -> float:
    """Linear interpolation of projected cost at hit rate *h*.

    Only used when ``observed_cost_usd`` is missing from the payload, so the
    observed marker still lands on the drawn curve instead of at y=0.
    """
    pts = sorted(
        ((_finite(p.cache_hit_rate), _finite(p.projected_cost_usd)) for p in curve),
        key=lambda t: t[0],
    )
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
            t = (h - h0) / (h1 - h0)
            return c0 + t * (c1 - c0)
    return pts[-1][1]


def _empty_figure(anchor: str, message: str) -> str:
    """Consistent empty state — a dashed box, never a broken half-chart."""
    return f'<div class="empty" id="{esc(anchor)}">{esc(message)}</div>'


# --------------------------------------------------------------------------- #
# Chart 4 — cumulative token / cost time series
# --------------------------------------------------------------------------- #
def _series_chart(series: Sequence[SeriesPoint], pal: Palette) -> str:
    """Cumulative tokens (area) plus cumulative cost (line, right axis).

    Two measures with different units on one canvas is normally a sin; here it
    is the point — the gap between the two curves *is* the story (tokens that
    were cache-discounted, or spend that grew without matching token volume).
    Both are cumulative, so the ordering of ``index`` is all that matters.
    """
    pts = sorted(series, key=lambda p: (_finite(p.index), _finite(p.cumulative_tokens)))
    if len(pts) < 2:
        return _empty_figure("series-empty", "Not enough series points to plot a trend.")

    W, H = 1120.0, 400.0
    x0, x1 = 92.0, 1006.0
    y0, y1 = 66.0, 320.0

    idx = [_finite(p.index) for p in pts]
    cum_tok = [_finite(p.cumulative_tokens) for p in pts]
    cum_cost = [_finite(p.cumulative_cost_usd) for p in pts]

    i_lo, i_hi = min(idx), max(idx)
    t_lo, t_hi = 0.0, max(cum_tok) or 1.0
    c_lo, c_hi = 0.0, max(cum_cost) or 1.0

    xs = LinearScale(i_lo, i_hi, x0, x1)
    # ``nice_ticks`` returns ticks *within* [lo, hi]; if the data max sits above
    # the last tick the area chart would float past the top gridline. Bump the
    # upper bound to the next multiple of the step so the peak is inside.
    x_ticks, _ = nice_ticks(i_lo, i_hi, target=7)
    y_ticks, y_step = nice_ticks(t_lo, t_hi, target=5)
    if y_step > 0:
        t_hi = max(t_hi, math.ceil(t_hi / y_step - 1e-9) * y_step)
    ys = LinearScale(t_lo, t_hi, y1, y0)
    y2_ticks, y2_step = nice_ticks(c_lo, c_hi * 1.08, target=5)
    if y2_step > 0:
        c_hi = max(c_hi, math.ceil(c_hi * 1.08 / y2_step - 1e-9) * y2_step)
    ys2 = LinearScale(c_lo, c_hi, y1, y0)

    tok_pts = [(xs(_finite(p.index)), ys(_finite(p.cumulative_tokens))) for p in pts]
    cost_pts = [(xs(_finite(p.index)), ys2(_finite(p.cumulative_cost_usd))) for p in pts]

    out = [
        _svg_open(
            W,
            H,
            label="Cumulative tokens and cost over the run",
            desc=(
                "Area chart of cumulative tokens and a line of cumulative cost in "
                "USD, both plotted against the sequence of completed cells."
            ),
        ),
        "<defs>"
        '<linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{esc(pal.accent)}" stop-opacity="0.34"/>'
        f'<stop offset="100%" stop-color="{esc(pal.accent)}" stop-opacity="0.01"/>'
        "</linearGradient></defs>",
        _area_path(tok_pts, y1),
        _axis_frame(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_scale=xs,
            y_scale=ys,
            fmt_x=lambda v: f"{int(round(v)):,}",
            fmt_y=fmt_tokens,
            x_label="cell index",
            y_label="cumulative tokens",
        ),
        _polyline(tok_pts, cls="curve"),
        _polyline(cost_pts, cls="series-line"),
    ]

    # Right-hand axis for cost: ticks only, no gridlines (they would fight the
    # token gridlines and imply a shared scale).
    for tick in y2_ticks:
        ty = ys2(tick)
        if ty < y0 - 0.5 or ty > y1 + 0.5:
            continue
        out.append(_line(x1, ty, x1 + 5.0, ty, cls="axis"))
        out.append(
            _text(
                x1 + 10.0,
                ty + 3.5,
                _usd_axis_fmt(y2_step)(tick),
                cls="tick tick--y",
                anchor="start",
            )
        )
    out.append(
        f'<text class="axl" text-anchor="middle" '
        f'transform="translate({x1 + 62.0:.2f},{(y0 + y1) / 2.0:.2f}) rotate(90)">'
        "cumulative cost (usd)</text>"
    )

    # endpoints: the reader's eye needs a terminal value, not a gridline read.
    # Labels go to the LEFT of the final marker (inside the plot) because the
    # right gutter is already occupied by the cost axis ticks.
    last = pts[-1]
    ex, ey = tok_pts[-1]
    out.append(_circle(ex, ey, 4.2, cls="endpoint"))
    out.append(
        _text(
            ex - 10.0,
            ey - 10.0,
            fmt_tokens(last.cumulative_tokens),
            cls="plabel plabel--front",
            anchor="end",
        )
    )
    cx, cy = cost_pts[-1]
    out.append(_circle(cx, cy, 4.2, cls="endpoint--amber"))
    out.append(
        _text(
            cx - 10.0,
            cy + 17.0,
            fmt_usd(last.cumulative_cost_usd),
            cls="obs-label",
            anchor="end",
        )
    )
    out.append(
        _legend(
            [("area", "cumulative tokens"), ("line-amber", "cumulative cost")],
            x1,
            28.0,
        )
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Chart 5 — token-type decomposition (stacked bars)
# --------------------------------------------------------------------------- #
_STACK_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("uncached_input_tokens", "seg-uncached"),
    ("cache_read_tokens", "seg-cache-read"),
    ("cache_write_tokens", "seg-cache-write"),
    ("output_tokens", "seg-output"),
    ("reasoning_tokens", "seg-reasoning"),
)

_STACK_LABELS: dict[str, str] = {
    "uncached_input_tokens": "uncached input",
    "cache_read_tokens": "cache read",
    "cache_write_tokens": "cache write",
    "output_tokens": "output",
    "reasoning_tokens": "reasoning",
}


def _stacked_chart(rows: Sequence[StackedBarRow], pal: Palette) -> str:
    """Vertical stacked bars: token-type mix per model × strategy.

    Sorted by billable tokens per task so the bars form a ranking rather than
    an arbitrary order — the chart doubles as the token-efficiency leaderboard.
    """
    usable = [r for r in rows if _row_total(r) > 0]
    if not usable:
        return _empty_figure("stacked-empty", "No token decomposition in this payload.")

    ranked = sorted(usable, key=lambda r: -_billable_per_cell(r))[:MAX_CHART_ROWS]

    # The category labels are rotated -42°, so the required bottom margin grows
    # with the longest label: reserve enough that a long "model · strategy" pair
    # never runs off the canvas.
    longest = max((len(f"{r.model_id} \u00b7 {r.strategy_id}") for r in ranked), default=10)
    label_len = min(longest, 26) * 5.95
    bottom_pad = max(78.0, label_len * 0.67 + 26.0)

    W, H = 1120.0, 356.0 + bottom_pad
    x0, x1 = 92.0, 1010.0
    y0, y1 = 72.0, 330.0

    totals = [_row_total(r) for r in ranked]
    t_hi = max(totals) or 1.0
    y_ticks, _ = nice_ticks(0.0, t_hi, target=5)
    t_hi = max(t_hi, y_ticks[-1])
    ys = LinearScale(0.0, t_hi, y1, y0)

    slot = (x1 - x0) / max(len(ranked), 1)
    bar_w = min(slot * 0.56, 46.0)

    out = [
        _svg_open(
            W,
            H,
            label="Token-type decomposition per model and strategy",
            desc=(
                "Vertical stacked bars showing uncached input, cache read, cache "
                "write, output and reasoning tokens for each model/strategy pair, "
                "sorted by billable tokens per task."
            ),
        ),
        _axis_frame(
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            x_ticks=[],
            y_ticks=y_ticks,
            x_scale=LinearScale(0.0, 1.0, x0, x1),
            y_scale=ys,
            fmt_x=lambda v: "",
            fmt_y=fmt_tokens,
            x_label="",
            y_label="tokens",
        ),
    ]

    for i, row in enumerate(ranked):
        cx = x0 + slot * (i + 0.5)
        bx = cx - bar_w / 2.0
        cursor = y1
        label = f"{row.model_id} · {row.strategy_id}"
        tip_lines = [label, f"cost {fmt_usd(row.cost_usd)}"]
        for field, cls in _STACK_SEGMENTS:
            value = int(_finite(getattr(row, field, 0)))
            if value <= 0:
                continue
            h = y1 - ys(value)
            out.append(_rect(bx, cursor - h, bar_w, h, cls=cls))
            cursor -= h
            tip_lines.append(f"{_STACK_LABELS[field]:<15}{fmt_tokens(value)}")
        # invisible full-height hit area: one tooltip for the whole bar
        out.append(
            f'<rect class="hit" x="{bx:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" '
            f'height="{y1 - y0:.2f}"><title>{esc(chr(10).join(tip_lines))}</title></rect>'
        )
        # Rotated category label.  SVG rotates around the text's *anchor* point,
        # so the translate target must sit below the axis (y1 + 16) and be the
        # right-hand end of the string; rotating -42° then makes the label climb
        # up-and-left, clear of both the axis and the neighbouring bar.
        out.append(
            f'<text class="tick tick--x" text-anchor="end" '
            f'transform="translate({cx + bar_w / 2.0:.2f},{y1 + 16.0:.2f}) rotate(-42)">'
            f"{esc(ellipsis(label, 26))}</text>"
        )

    out.append(
        _legend(
            [
                ("swatch-uncached", _STACK_LABELS["uncached_input_tokens"]),
                ("swatch-cache-read", _STACK_LABELS["cache_read_tokens"]),
                ("swatch-cache-write", _STACK_LABELS["cache_write_tokens"]),
                ("swatch-output", _STACK_LABELS["output_tokens"]),
                ("swatch-reasoning", _STACK_LABELS["reasoning_tokens"]),
            ],
            x1,
            32.0,
        )
    )
    out.append("</svg>")
    return "".join(out)


def _row_total(row: Any) -> float:
    """Total tokens represented by a stacked row."""
    return float(
        int(_finite(getattr(row, "uncached_input_tokens", 0)))
        + int(_finite(getattr(row, "cache_read_tokens", 0)))
        + int(_finite(getattr(row, "cache_write_tokens", 0)))
        + int(_finite(getattr(row, "output_tokens", 0)))
        + int(_finite(getattr(row, "reasoning_tokens", 0)))
    )


def _billable_per_cell(row: Any) -> float:
    """Billable tokens per cell, derived when the row has no explicit field."""
    cells = int(_finite(getattr(row, "cells", 0))) or 1
    billable = int(_finite(getattr(row, "uncached_input_tokens", 0))) + int(
        _finite(getattr(row, "output_tokens", 0))
    )
    return billable / cells


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _rank_badge(rank: int) -> str:
    """Medal for the top three, monospace number for the rest."""
    cls = f"rank rank--{rank}" if 1 <= rank <= 3 else "rank"
    return f'<span class="{esc(cls)}">{esc(rank)}</span>'


def _spark(value: float, maximum: float) -> str:
    """Tiny inline bar for a ``[0, 1]``-ish score, width proportional to *maximum*."""
    pct = 0.0 if maximum <= 0 else max(0.0, min(_finite(value) / maximum, 1.0)) * 100.0
    return (
        f'<span class="spark" role="img" aria-label="{esc(fmt_ratio(value))}">'
        f'<span style="width:{pct:.1f}%"></span></span>'
    )


def _leaderboard_table(rows: Sequence[EfficiencyRow]) -> str:
    """Token-efficiency leaderboard: rank, pair, tokens, quality, cost, score.

    Every numeric column is right-aligned and tabular, because the reader's job
    here is *comparison down a column*, not reading prose across a row.
    """
    if not rows:
        return _empty_figure("leaderboard-empty", "No leaderboard rows in this payload.")
    ranked = sorted(rows, key=lambda r: r.rank if r.rank else 10**6)
    best_score = max((_finite(r.efficiency_score) for r in ranked), default=0.0)
    out = [
        '<div class="table-wrap"><table id="leaderboard-table">',
        '<caption class="sr-only">Token efficiency leaderboard</caption>',
        "<thead><tr>"
        '<th scope="col">#</th>'
        '<th scope="col">model</th>'
        '<th scope="col">strategy</th>'
        '<th scope="col" class="num">tok/task</th>'
        '<th scope="col" class="num">quality</th>'
        '<th scope="col" class="num">$/task</th>'
        '<th scope="col" class="num">cache</th>'
        '<th scope="col" class="num">latency</th>'
        '<th scope="col" class="num">efficiency</th>'
        "</tr></thead><tbody>",
    ]
    for row in ranked:
        out.append(
            "<tr>"
            f"<td>{_rank_badge(int(row.rank) if row.rank else 0)}</td>"
            f'<td class="model">{esc(row.model_id)}</td>'
            f'<td class="strategy">{esc(row.strategy_id)}</td>'
            f'<td class="num">{esc(fmt_tokens(row.billable_tokens_per_task))}</td>'
            f'<td class="num">{esc(fmt_quality(row.mean_quality))}</td>'
            f'<td class="num">{esc(fmt_usd(row.cost_per_task_usd))}</td>'
            f'<td class="num">{esc(fmt_pct(row.cache_hit_rate, 0))}</td>'
            f'<td class="num">{esc(fmt_duration(row.mean_latency_ms / 1000.0))}</td>'
            f'<td class="num">{_spark(row.efficiency_score, best_score)}</td>'
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _verdict_badge(verdict: str) -> str:
    """Map a free-text verdict to a colour-coded badge.

    Matching is on lowercase substrings because ``verdict`` is an open string
    field in the schema (the analysis layer owns its wording), so the renderer
    degrades to a neutral badge for wording it does not recognise.
    """
    text = (verdict or "").strip()
    low = text.lower()
    if not text:
        return f'<span class="badge">{esc(MISSING)}</span>'
    if any(k in low for k in ("worth", "win", "good", "yes", "gain")):
        return f'<span class="badge badge--good">{esc(text)}</span>'
    if any(k in low for k in ("waste", "bad", "regress", "no", "avoid", "loss")):
        return f'<span class="badge badge--bad">{esc(text)}</span>'
    return f'<span class="badge badge--warn">{esc(text)}</span>'


def _delta_cell(value: float, *, digits: int = 1) -> str:
    """Signed percentage delta, coloured by direction (up = amber = costlier)."""
    v = _finite(value)
    if abs(v) < 1e-9:
        return f'<td class="delta delta--flat">{esc("0.0%")}</td>'
    cls = "delta--up" if v > 0 else "delta--down"
    return f'<td class="delta {esc(cls)}">{esc(fmt_pct(v, digits))}</td>'


def _strategy_delta_table(rows: Sequence[Any]) -> str:
    """Ablation table: what each strategy costs relative to the baseline."""
    if not rows:
        return _empty_figure(
            "deltas-empty", "No strategy deltas in this payload (single-strategy run?)."
        )
    out = [
        '<div class="table-wrap"><table id="delta-table">',
        '<caption class="sr-only">Strategy ablation versus baseline</caption>',
        "<thead><tr>"
        '<th scope="col">strategy</th>'
        '<th scope="col">model</th>'
        '<th scope="col" class="num">tok/task</th>'
        '<th scope="col" class="num">Δ tokens</th>'
        '<th scope="col" class="num">$/task</th>'
        '<th scope="col" class="num">Δ cost</th>'
        '<th scope="col" class="num">quality</th>'
        '<th scope="col" class="num">Δ quality</th>'
        '<th scope="col">verdict</th>'
        "</tr></thead><tbody>",
    ]
    for row in rows:
        dq = _finite(row.quality_delta)
        dq_cls = "delta--down" if dq < -1e-9 else ("delta--up" if dq > 1e-9 else "delta--flat")
        model = row.model_id if row.model_id else f"all (vs {row.baseline_strategy})"
        out.append(
            "<tr>"
            f'<td class="strategy">{esc(row.strategy_id)}</td>'
            f'<td class="model">{esc(model)}</td>'
            f'<td class="num">{esc(fmt_tokens(row.strategy_billable_per_task))}</td>'
            + _delta_cell(row.token_delta_pct)
            + f'<td class="num">{esc(fmt_usd(row.strategy_cost_per_task_usd))}</td>'
            + _delta_cell(row.cost_delta_pct)
            + f'<td class="num">{esc(fmt_quality(row.quality))}</td>'
            f'<td class="delta {esc(dq_cls)}">{esc(f"{dq:+.3f}")}</td>'
            f"<td>{_verdict_badge(row.verdict)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _retry_table(rows: Sequence[Any]) -> str:
    """Full retry detail table (the chart shows the top rows only)."""
    if not rows:
        return _empty_figure("retry-table-empty", "No retry rows in this payload.")
    ranked = sorted(rows, key=lambda r: -_finite(r.retry_cost_usd))
    out = [
        '<div class="table-wrap"><table id="retry-table">',
        '<caption class="sr-only">Retry cost by model and strategy</caption>',
        "<thead><tr>"
        '<th scope="col">model</th>'
        '<th scope="col">strategy</th>'
        '<th scope="col" class="num">cells</th>'
        '<th scope="col" class="num">retried</th>'
        '<th scope="col" class="num">retry rate</th>'
        '<th scope="col" class="num">extra attempts</th>'
        '<th scope="col" class="num">retry tokens</th>'
        '<th scope="col" class="num">retry cost</th>'
        '<th scope="col" class="num">share</th>'
        '<th scope="col">top error</th>'
        "</tr></thead><tbody>",
    ]
    for row in ranked:
        out.append(
            "<tr>"
            f'<td class="model">{esc(row.model_id)}</td>'
            f'<td class="strategy">{esc(row.strategy_id)}</td>'
            f'<td class="num">{esc(fmt_int(row.cells))}</td>'
            f'<td class="num">{esc(fmt_int(row.retried_cells))}</td>'
            f'<td class="num">{esc(fmt_pct(row.retry_rate, 1))}</td>'
            f'<td class="num">{esc(fmt_int(row.extra_attempts))}</td>'
            f'<td class="num">{esc(fmt_tokens(row.retry_usage.total_tokens))}</td>'
            f'<td class="num">{esc(fmt_usd(row.retry_cost_usd))}</td>'
            f'<td class="num">{esc(fmt_pct(row.retry_cost_share, 1))}</td>'
            f'<td class="strategy">{esc(row.dominant_error_kind or MISSING)}</td>'
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def _category_table(rows: Sequence[Any]) -> str:
    """Per-category breakdown — where the workload is genuinely expensive."""
    if not rows:
        return _empty_figure("categories-empty", "No category breakdown in this payload.")
    out = [
        '<div class="table-wrap"><table id="category-table">',
        '<caption class="sr-only">Per-category token and cost breakdown</caption>',
        "<thead><tr>"
        '<th scope="col">category</th>'
        '<th scope="col">model</th>'
        '<th scope="col">strategy</th>'
        '<th scope="col" class="num">cells</th>'
        '<th scope="col" class="num">success</th>'
        '<th scope="col" class="num">tok/task</th>'
        '<th scope="col" class="num">$/task</th>'
        '<th scope="col" class="num">quality</th>'
        "</tr></thead><tbody>",
    ]
    for row in rows:
        out.append(
            "<tr>"
            f'<td class="model">{esc(row.category)}</td>'
            f'<td class="strategy">{esc(row.model_id)}</td>'
            f'<td class="strategy">{esc(row.strategy_id)}</td>'
            f'<td class="num">{esc(fmt_int(row.cells))}</td>'
            f'<td class="num">{esc(fmt_pct(row.success_rate, 1))}</td>'
            f'<td class="num">{esc(fmt_tokens(row.billable_tokens_per_task))}</td>'
            f'<td class="num">{esc(fmt_usd(row.cost_per_task_usd))}</td>'
            f'<td class="num">{esc(fmt_quality(row.mean_quality))}</td>'
            "</tr>"
        )
    out.append("</tbody></table></div>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Page furniture: KPI cards, masthead, footer
# --------------------------------------------------------------------------- #
def _kpi_card(label: str, value: str, sub: str = "", tone: str = "", kpi_id: str = "") -> str:
    """One KPI tile.  *tone* is ``accent|warn|waste|good`` — semantic only."""
    cls = "kpi" + (f" kpi--{tone}" if tone else "")
    id_attr = f' id="{esc(kpi_id)}"' if kpi_id else ""
    sub_html = f'<span class="s">{esc(sub)}</span>' if sub else ""
    return (
        f'<div class="{esc(cls)}"{id_attr}>'
        f'<span class="k">{esc(label)}</span>'
        f'<span class="v">{esc(value)}</span>'
        f"{sub_html}</div>"
    )


def _kpi_row(payload: ReportPayload) -> str:
    """The six numbers that answer "what did this run cost and buy?"."""
    s = payload.summary
    retry_share = _finite(payload.retry.overall_retry_cost_share)
    cards = [
        _kpi_card(
            "total spend",
            fmt_usd(s.total_cost_usd),
            f"{fmt_usd(s.cost_per_1k_billable_tokens)} / 1k billable",
            "accent",
            "kpi-spend",
        ),
        _kpi_card(
            "retry waste",
            fmt_pct(retry_share, 1),
            f"{fmt_usd(payload.retry.total_retry_cost_usd)} burned",
            "waste",
            "kpi-retry",
        ),
        _kpi_card(
            "billable tokens",
            fmt_tokens(s.total_billable_tokens),
            f"{fmt_tokens(s.total_usage.total_tokens)} total on the wire",
            "",
            "kpi-tokens",
        ),
        _kpi_card(
            "cache hit rate",
            fmt_pct(s.cache_hit_rate, 1),
            f"saves {fmt_usd(payload.cache.savings_at_observed_usd)} at observed",
            "good",
            "kpi-cache",
        ),
        _kpi_card(
            "quality",
            fmt_quality(_best_quality(payload)),
            f"best of {len(payload.pareto.points)} pairs",
            "warn",
            "kpi-quality",
        ),
        _kpi_card(
            "cells",
            f"{fmt_int(s.completed_cells)}/{fmt_int(s.planned_cells)}",
            f"{fmt_int(s.total_calls)} calls · {fmt_pct(s.success_rate, 1)} ok",
            "",
            "kpi-cells",
        ),
    ]
    return f'<div class="grid kpis">{"".join(cards)}</div>'


def _best_quality(payload: ReportPayload) -> float:
    """Highest mean quality in the payload (0.0 when there is nothing to show)."""
    candidates = [_finite(p.quality) for p in payload.pareto.points]
    candidates += [_finite(r.mean_quality) for r in payload.leaderboard]
    return max(candidates) if candidates else 0.0


def _masthead(payload: ReportPayload) -> str:
    """Header: identity of the run + the provenance needed to trust it."""
    meta, s = payload.meta, payload.summary
    notes_html = ""
    if meta.notes:
        notes = "".join(f"<li>{esc(n)}</li>" for n in meta.notes[:6])
        notes_html = f'<ul class="notes">{notes}</ul>'
    price_chip = f'<span class="chip chip--accent">price book {esc(meta.price_book)}</span>'
    if meta.price_book_date:
        price_chip += f'<span class="chip">{esc(meta.price_book_date)}</span>'
    mode = payload.manifest.adapter_mode if payload.manifest else ""
    mode_chip = f'<span class="chip chip--warn">mode {esc(mode)}</span>' if mode else ""
    return (
        '<header class="masthead">'
        '<div class="masthead-top">'
        '<span class="brand">tokbench</span>'
        f'<span class="chip">{esc(f"schema {meta.schema_version}")}</span>'
        f"{price_chip}{mode_chip}"
        "</div>"
        f"<h1>{esc(meta.experiment)}</h1>"
        '<div class="runline">'
        f'<span><span class="k">run</span><span class="v">{esc(meta.run_id)}</span></span>'
        f'<span><span class="k">generated</span><span class="v">{esc(fmt_dt(meta.generated_at))}</span></span>'
        f'<span><span class="k">models</span><span class="v">{esc(fmt_int(s.models))}</span></span>'
        f'<span><span class="k">strategies</span><span class="v">{esc(fmt_int(s.strategies))}</span></span>'
        f'<span><span class="k">tasks</span><span class="v">{esc(fmt_int(s.tasks))}</span></span>'
        f'<span><span class="k">repeats</span><span class="v">{esc(fmt_int(s.repeats))}</span></span>'
        f'<span><span class="k">wall clock</span><span class="v">{esc(fmt_duration(s.wall_clock_s))}</span></span>'
        f'<span><span class="k">currency</span><span class="v">{esc(meta.currency)}</span></span>'
        "</div>"
        f"{notes_html}"
        "</header>"
    )


def _footer(payload: ReportPayload) -> str:
    """Methodology + provenance.  The reader must be able to audit the numbers."""
    meta = payload.meta
    lines = [
        "Cost is derived only from token counts and the price book "
        "(<code>pricing.py</code>); adapters never compute money.",
        "Billable tokens = uncached input + output. Cache reads are billed at the "
        "cache-read rate, so cache efficiency is part of the ranking.",
        "The Pareto frontier marks points that no other pair beats on both cost "
        "and quality. The knee is the frontier point furthest from the "
        "cheapest-to-best chord — the pragmatic pick.",
        "Retry overhead is every token spent after the first attempt of a cell, "
        "i.e. spend that a successful first attempt would have avoided.",
        "Cache sensitivity holds observed input/output volume fixed and sweeps "
        "the hit rate; the resulting cost curve is linear by construction.",
        "Quality grading is heuristic and comparable only within a run.",
    ]
    items = "".join(f"<li>{line}</li>" for line in lines)
    manifest_bits = []
    if payload.manifest is not None:
        m = payload.manifest
        manifest_bits = [
            f"engine {m.engine_version}",
            f"seed {m.seed}" if m.seed is not None else "",
            f"config {m.config_hash[:12]}" if m.config_hash else "",
            f"mode {m.adapter_mode}",
        ]
    colophon = [
        f"{GENERATOR} {meta.generator_version}",
        f"schema {meta.schema_version}",
        f"price book {meta.price_book}",
        f"generated {fmt_dt(meta.generated_at)}",
        *[b for b in manifest_bits if b],
    ]
    colophon_html = "".join(f"<span>{esc(b)}</span>" for b in colophon)
    return (
        "<footer>"
        "<h3>methodology</h3>"
        f"<ul>{items}</ul>"
        f'<div class="colophon">{colophon_html}'
        "<span>no external requests</span></div>"
        "</footer>"
    )


def _section(
    section_id: str,
    title: str,
    *,
    tag: str = "",
    hero: bool = False,
    lede: str = "",
    body: str,
) -> str:
    """One titled card.  *section_id* is the stable anchor tests assert on."""
    tag_cls = "tag hero-tag" if hero else "tag"
    tag_html = f'<span class="{esc(tag_cls)}">{esc(tag)}</span>' if tag else ""
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    return (
        f'<section id="{esc(section_id)}">'
        '<div class="card">'
        f'<div class="card-head"><h2>{esc(title)}</h2>{tag_html}</div>'
        f"{lede_html}{body}"
        "</div></section>"
    )


# --------------------------------------------------------------------------- #
# Prose that explains the charts (kept close to the charts, not in a template)
# --------------------------------------------------------------------------- #
def _pareto_lede(payload: ReportPayload) -> str:
    """One-sentence takeaway above the hero chart.

    A chart without a claim forces the reader to do the analysis themselves.
    This states what the frontier means in *this* run, using its own numbers.
    """
    pareto = payload.pareto
    if not pareto.frontier_labels:
        return ""
    chain = " → ".join(esc(x) for x in pareto.frontier_labels[:5])
    bits = [f"Frontier: <code>{chain}</code>."]
    if pareto.frontier_quality_gain_per_usd:
        bits.append(
            "Each extra dollar buys "
            f"<b>{esc(fmt_ratio(pareto.frontier_quality_gain_per_usd, 3))}</b> quality "
            "along the frontier."
        )
    if pareto.best_value_label:
        bits.append(f"Best value: <b>{esc(pareto.best_value_label)}</b>.")
    return " ".join(bits)


def _retry_lede(payload: ReportPayload) -> str:
    """Frames the retry number as an avoidable line item, not trivia."""
    retry = payload.retry
    if not retry.total_cost_usd:
        return ""
    worst = retry.worst_offender
    parts = [
        f"Retries consumed <b>{esc(fmt_usd(retry.total_retry_cost_usd))}</b> of "
        f"<b>{esc(fmt_usd(retry.total_cost_usd))}</b> total spend "
        f"(<b>{esc(fmt_pct(retry.overall_retry_cost_share, 1))}</b>)."
    ]
    if worst:
        parts.append(f"Worst offender: <code>{esc(worst)}</code>.")
    if retry.retry_token_share:
        parts.append(
            f"The same failures also burned {esc(fmt_pct(retry.retry_token_share, 1))} "
            "of all tokens."
        )
    return " ".join(parts)


def _cache_lede(payload: ReportPayload) -> str:
    """Quantifies the cache lever in the currency the reader cares about."""
    cache = payload.cache
    if not cache.curve:
        return ""
    parts = [
        f"At the observed <b>{esc(fmt_pct(cache.observed_hit_rate, 1))}</b> hit rate the "
        f"run saved <b>{esc(fmt_usd(cache.savings_at_observed_usd))}</b> "
        f"(<b>{esc(fmt_pct(cache.savings_at_observed_pct, 1))}</b>) versus no cache."
    ]
    if cache.marginal_usd_per_10pp:
        parts.append(
            f"Every <b>+10pp</b> of hit rate is worth about "
            f"<b>{esc(fmt_usd(cache.marginal_usd_per_10pp))}</b>."
        )
    return " ".join(parts)


def _series_lede(payload: ReportPayload) -> str:
    """Explains the token/cost divergence the two-curve chart exposes."""
    series = payload.series
    if not series:
        return ""
    last = max(series, key=lambda p: _finite(p.index))
    return (
        f"Across <b>{esc(fmt_int(len(series)))}</b> completed cells the run pushed "
        f"<b>{esc(fmt_tokens(last.cumulative_tokens))}</b> tokens for "
        f"<b>{esc(fmt_usd(last.cumulative_cost_usd))}</b>. The gap between the two "
        "curves is the cache discount doing its job."
    )


def _stacked_lede(payload: ReportPayload) -> str:
    """Points at the composition question: what is the spend actually made of?

    The cache share is the *one canonical* hit rate from the payload
    (``summary.cache_hit_rate``), never re-derived from the stacked rows: the
    hit-rate denominator is ``input - cache_write``, so dividing raw cache-read
    tokens by all tokens here would print a second, disagreeing percentage in
    the same report.  The per-segment token mix is left to the chart.
    """
    rows = payload.stacked
    if not rows:
        return ""
    return (
        f"<b>{esc(fmt_pct(payload.summary.cache_hit_rate, 1))}</b> of cache-eligible "
        "prompt tokens were served from cache. Sorted by billable tokens per task, "
        "so the cheapest pair sits left."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_html(
    payload: ReportPayload,
    *,
    run_dir: str | Path | None = None,
    theme: str = "dark",
) -> str:
    """Render *payload* as one self-contained HTML document.

    Args:
        payload: The report contract from :mod:`tokbench.report_schema`.
        run_dir: Optional run directory.  Only used for display in the footer;
            nothing is read from disk, which keeps the renderer pure (same
            payload in → same bytes out, regardless of machine state).
        theme: ``"dark"`` (default) or ``"light"``.

    Returns:
        A complete ``<!doctype html>`` document as a string.  It references no
        external resource: every style is inline, every chart is inline SVG and
        there is no JavaScript at all (interactivity is ``<title>`` tooltips and
        CSS ``:hover``, which need neither).

    Raises:
        ValueError: If *theme* is not a known theme name.
    """
    pal = _palette(theme)
    meta = payload.meta

    title = f"tokbench · {meta.experiment} · {meta.run_id}"
    if run_dir is not None:
        # Path is display-only metadata; escaped like any other dynamic string.
        title = f"{title} · {Path(run_dir).name}"

    sections: list[str] = []
    sections.append(
        _section(
            "section-pareto",
            "Pareto frontier",
            tag="hero",
            hero=True,
            lede=_pareto_lede(payload),
            body=(
                '<figure class="figure">'
                + _pareto_chart(payload, pal)
                + "<figcaption>Cost per task (USD) against mean quality. "
                "Cyan points are non-dominated — nothing is both cheaper and better. "
                "Dimmed points are dominated. The amber diamond is the knee: "
                "beyond it, extra spend buys proportionally less quality.</figcaption>"
                "</figure>"
            ),
        )
    )
    sections.append(
        _section(
            "section-retry",
            "Retry burn",
            tag="waste",
            hero=True,
            lede=_retry_lede(payload),
            body=(
                _retry_panel(payload.retry, pal) + '<div class="card-head" style="margin-top:26px">'
                '<h2 style="font-size:14px">Retry detail</h2></div>'
                + _retry_table(payload.retry.rows)
            ),
        )
    )
    sections.append(
        _section(
            "section-cache",
            "Cache sensitivity",
            tag="what-if",
            lede=_cache_lede(payload),
            body=(
                '<figure class="figure">'
                + _cache_chart(payload.cache, pal)
                + "<figcaption>Projected total cost as the cache hit rate is swept "
                "from 0 to 100%, holding observed token volume fixed. The dashed "
                "line marks where this run actually landed.</figcaption>"
                "</figure>"
            ),
        )
    )
    sections.append(
        _section(
            "section-series",
            "Cumulative tokens and cost",
            tag="timeline",
            lede=_series_lede(payload),
            body=(
                '<figure class="figure">'
                + _series_chart(payload.series, pal)
                + "<figcaption>Both series are cumulative over completed cells, so "
                "the slope is the instantaneous rate. A flattening cost curve with a "
                "still-rising token curve means cache reads are absorbing the growth."
                "</figcaption></figure>"
            ),
        )
    )
    sections.append(
        _section(
            "section-stacked",
            "Token-type decomposition",
            tag="composition",
            lede=_stacked_lede(payload),
            body=(
                '<figure class="figure">'
                + _stacked_chart(payload.stacked, pal)
                + "<figcaption>Where every token went. Cache read is billed at a "
                "fraction of the input rate; output and reasoning tokens are billed "
                "at the output rate.</figcaption></figure>"
            ),
        )
    )
    sections.append(
        _section(
            "section-leaderboard",
            "Token efficiency leaderboard",
            tag="ranking",
            body=_leaderboard_table(payload.leaderboard),
        )
    )
    sections.append(
        _section(
            "section-strategy",
            "Strategy ablation",
            tag="deltas",
            body=_strategy_delta_table(payload.strategy_deltas),
        )
    )
    if payload.categories:
        sections.append(
            _section(
                "section-categories",
                "Category breakdown",
                tag="drill-down",
                body=_category_table(payload.categories),
            )
        )

    body = "".join(sections)
    return (
        "<!doctype html>\n"
        f'<html lang="en" data-theme="{esc(theme)}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="dark light">\n'
        f"<title>{esc(title)}</title>\n"
        f"<style>{_theme_vars(pal)}{_CSS}</style>\n"
        "</head>\n<body>\n"
        '<main class="wrap">\n'
        f"{_masthead(payload)}\n"
        f"{_kpi_row(payload)}\n"
        f"{body}\n"
        f"{_footer(payload)}\n"
        "</main>\n</body>\n</html>\n"
    )


def _theme_vars(pal: Palette) -> str:
    """Emit the palette as CSS custom properties on ``:root``.

    The stylesheet itself only ever refers to ``var(--token)``, so switching
    theme is a single emitted block — no duplicated rules, no per-theme classes.
    """
    tokens = {
        "--bg": pal.bg,
        "--bg-soft": pal.bg_soft,
        "--panel": pal.panel,
        "--panel-2": pal.panel_2,
        "--line": pal.line,
        "--line-soft": pal.line_soft,
        "--ink": pal.ink,
        "--ink-2": pal.ink_2,
        "--ink-3": pal.ink_3,
        "--grid": pal.grid,
        "--axis": pal.axis,
        "--accent": pal.accent,
        "--accent-dim": pal.accent_dim,
        "--accent-2": pal.accent_2,
        "--accent-3": pal.accent_3,
        "--good": pal.good,
        "--font-sans": pal.font_sans,
        "--font-mono": pal.font_mono,
    }
    decls = ";".join(f"{k}:{v}" for k, v in tokens.items())
    return f":root{{{decls}}}"


def write_html(
    payload: ReportPayload,
    path: str | Path,
    *,
    run_dir: str | Path | None = None,
    theme: str = "dark",
) -> Path:
    """Render *payload* and write it to *path* (parent dirs created as needed).

    Returns the resolved :class:`~pathlib.Path` that was written, so callers can
    print or open it without re-deriving the location.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_html(payload, run_dir=run_dir, theme=theme)
    target.write_text(text, encoding="utf-8")
    return target.resolve()
