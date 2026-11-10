"""Cost/quality Pareto frontier with a knee-point recommendation.

The frontier answers a different question from the leaderboard.  The leaderboard
asks *"who spends the fewest tokens per quality point?"*; the frontier asks
*"which options are not strictly dominated on the cost/quality plane, and which
of those is the sensible place to sit?"*.

Definitions
-----------
Each candidate is a ``(model, strategy)`` group, reduced to a single point by
:func:`tokbench.analysis.aggregate.group_stats`::

    x = cost_per_task_usd   (mean USD spent on one task)
    y = mean_quality        (mean grader score in [0, 1])

**Strict dominance** — point ``a`` dominates point ``b`` iff::

    a.cost <= b.cost  and  a.quality >= b.quality
    and (a.cost < b.cost or a.quality > b.quality)

i.e. at least as cheap *and* at least as good, with at least one strict
improvement.  A point is **on the frontier** iff nothing dominates it.  Equality
on both axes is *not* dominance: two identical points both stay on the frontier
(neither is worse), which is the correct reading — the tie is reported by
``value_score`` and the label ordering instead of by silently dropping a model.
The frontier is sorted by cost ascending, so it reads left-to-right as
"cheapest → best".

**value_score** — quality bought per dollar, normalised so the best point is 1::

    v_i   = quality_i / max(cost_per_task_i, 1e-9)
    value_i = v_i / max_j(v_j)

The ``1e-9`` floor lets a free (0 USD) model compete instead of dividing by zero;
free-and-good is legitimately the best value, so it normalises to 1.0.

**Knee point** — the recommendation.  The frontier's two extremes (cheapest point
and highest-quality point) define a straight line; the knee is the frontier point
with the greatest *perpendicular distance* from that line.  Intuition: the line is
"what a linear cost/quality trade-off would look like", and the point farthest
from it is where you get the most quality for the least extra money before the
price of quality accelerates.  Both axes are min-max normalised to ``[0, 1]``
before measuring distance — without that, cost (typically 1e-4 .. 1e-2 USD) would
be three orders of magnitude smaller than quality (0 .. 1) and the distance would
degenerate to a vertical measure that always picks the best-quality point.  If no
point has non-zero distance (a 1- or 2-point frontier, or a perfectly collinear
one) there is no meaningful knee and the explanation says so.

**frontier_quality_gain_per_usd** — the frontier's average exchange rate:
``(max_quality - min_quality) / (max_cost - min_cost)`` across frontier points.
It is the marginal price of quality at the frontier's scale, and the number to
quote when someone asks "how much more does better quality cost?".

Costs are re-derived from the :class:`~tokbench.pricing.PriceBook` via
:func:`tokbench.analysis.aggregate.cost_of`, so a report can be restated against
a different price book without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TaskResult
from ..pricing import PriceBook
from ..report_schema import ParetoFrontier, ParetoPoint
from .aggregate import (
    GroupStats,
    group_stats,
    label_of,
    resolve_baseline_strategy,
    safe_div,
)

#: Cost floor (USD) when turning quality into a quality-per-dollar ratio.
COST_EPSILON = 1e-9

#: Distance below which a frontier point is considered collinear with the extremes.
_KNEE_EPSILON = 1e-12


def dominates(a: GroupStats, b: GroupStats) -> bool:
    """True iff ``a`` is at least as cheap and at least as good, with one strict.

    See the module docstring for the exact relation.  Both axes are per-task
    values, so groups with different cell counts remain comparable.
    """
    a_cost, a_quality = a.cost_per_task_usd, a.quality
    b_cost, b_quality = b.cost_per_task_usd, b.quality
    no_worse = a_cost <= b_cost and a_quality >= b_quality
    strictly_better = a_cost < b_cost or a_quality > b_quality
    return no_worse and strictly_better


@dataclass(frozen=True, slots=True)
class _Knee:
    """Knee-point measurement for one frontier."""

    label: str | None
    distance: float
    quality: float
    cost_per_task_usd: float


def _knee_point(frontier: list[GroupStats]) -> _Knee:
    """Farthest frontier point from the line joining the cost/quality extremes.

    Axes are min-max normalised first (see the module docstring).  Returns a
    :class:`_Knee` with ``label=None`` when the frontier has fewer than three
    points or every point is collinear with the extremes.
    """
    if len(frontier) < 3:
        return _Knee(label=None, distance=0.0, quality=0.0, cost_per_task_usd=0.0)

    costs = [p.cost_per_task_usd for p in frontier]
    qualities = [p.quality for p in frontier]
    min_cost, max_cost = min(costs), max(costs)
    min_q, max_q = min(qualities), max(qualities)
    cost_span = max_cost - min_cost
    quality_span = max_q - min_q
    if cost_span <= 0.0 or quality_span <= 0.0:
        return _Knee(label=None, distance=0.0, quality=0.0, cost_per_task_usd=0.0)

    # Extremes: cheapest point (left) and highest-quality point (right).
    left = min(frontier, key=lambda p: (p.cost_per_task_usd, -p.quality, p.label))
    right = min(frontier, key=lambda p: (-p.quality, p.cost_per_task_usd, p.label))
    x1 = (left.cost_per_task_usd - min_cost) / cost_span
    y1 = (left.quality - min_q) / quality_span
    x2 = (right.cost_per_task_usd - min_cost) / cost_span
    y2 = (right.quality - min_q) / quality_span

    dx, dy = x2 - x1, y2 - y1
    line_len = (dx * dx + dy * dy) ** 0.5
    if line_len <= 0.0:  # degenerate: the two extremes coincide
        return _Knee(label=None, distance=0.0, quality=0.0, cost_per_task_usd=0.0)

    best = _Knee(label=None, distance=0.0, quality=0.0, cost_per_task_usd=0.0)
    for point in frontier:
        px = (point.cost_per_task_usd - min_cost) / cost_span
        py = (point.quality - min_q) / quality_span
        # |cross product| / |line| = perpendicular distance from the line.
        distance = abs(dx * (py - y1) - dy * (px - x1)) / line_len
        if distance > best.distance + _KNEE_EPSILON or (
            abs(distance - best.distance) <= _KNEE_EPSILON
            and best.label is not None
            and (point.quality, -point.cost_per_task_usd, point.label)
            > (best.quality, -best.cost_per_task_usd, best.label)
        ):
            best = _Knee(
                label=point.label,
                distance=distance,
                quality=point.quality,
                cost_per_task_usd=point.cost_per_task_usd,
            )
    if best.distance <= _KNEE_EPSILON:
        return _Knee(label=None, distance=0.0, quality=0.0, cost_per_task_usd=0.0)
    return best


def _pick_baseline_label(points: list[ParetoPoint], baseline_strategy: str | None) -> str | None:
    """Label of the baseline-strategy point, or ``None``.

    A strategy can be run on several models, so we take the strongest
    representative: highest quality, then cheaper, then label.  ``None`` when the
    strategy is absent from the run.
    """
    if baseline_strategy is None:
        return None
    candidates = [p for p in points if p.strategy_id == baseline_strategy]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (-p.quality, p.cost_per_task_usd, p.label)).label


def _explain(
    frontier: list[GroupStats],
    knee: _Knee,
    gain_per_usd: float,
    cheapest: str | None,
    best_value: str | None,
    best_quality: str | None,
    baseline_label: str | None,
) -> str:
    """One-paragraph, human-readable reading of the frontier."""
    if not frontier:
        return (
            "No cells to place on the cost/quality plane: the run contains no "
            "completed results. Re-run the experiment or point the report at a "
            "run directory that finished at least one cell."
        )

    parts: list[str] = []
    if len(frontier) == 1:
        only = frontier[0]
        parts.append(
            f"Only one candidate ({only.label}) is available, so it is trivially "
            f"the whole frontier at ${only.cost_per_task_usd:.6f}/task and "
            f"quality {only.quality:.3f}; run more models/strategies to get a "
            "real cost/quality trade-off."
        )
    else:
        parts.append(
            f"{len(frontier)} of the candidates are non-dominated: "
            + " -> ".join(p.label for p in frontier)
            + "."
        )
        parts.append(
            f"Cheapest is {cheapest}; the best value is {best_value} "
            f"(highest quality per dollar); the best quality is {best_quality}."
        )
        if knee.label is not None:
            parts.append(
                f"Knee point (farthest from the line joining the cost/quality "
                f"extremes) is {knee.label} at "
                f"${knee.cost_per_task_usd:.6f}/task and quality "
                f"{knee.quality:.3f} — the last point where extra spend still "
                "buys a proportionate amount of quality."
            )
        else:
            parts.append(
                "No interior knee point: the frontier is a single segment "
                "(or perfectly collinear), so the cost/quality trade-off is "
                "linear and there is no 'elbow' to recommend."
            )
        if gain_per_usd > 0.0:
            parts.append(
                f"Across the frontier, quality rises at "
                f"{gain_per_usd:.6g} points per USD "
                f"(${safe_div(1.0, gain_per_usd):.6f} per quality point)."
            )
        else:
            parts.append(
                "The frontier's quality range is flat relative to its cost range, "
                "so there is no measurable price of quality on these points."
            )
    if baseline_label is not None:
        parts.append(f"Baseline for the ablation table is {baseline_label}.")
    return " ".join(parts)


def pareto_frontier(
    results: list[TaskResult],
    price_book: PriceBook,
    *,
    baseline_strategy: str | None = None,
) -> ParetoFrontier:
    """Cost/quality Pareto frontier over ``(model, strategy)`` groups.

    Returns a valid, explanatory :class:`~tokbench.report_schema.ParetoFrontier`
    for any input, including an empty one — a report must never raise because a
    run produced no data.

    The ``baseline_strategy`` is resolved with
    :func:`tokbench.analysis.aggregate.resolve_baseline_strategy`, so an unknown
    strategy id silently falls back to the cheapest strategy rather than leaving
    the ablation table without a reference point.
    """
    if not results:
        return ParetoFrontier(
            explanation=(
                "No results to analyse: the run contains zero completed cells, so "
                "there is no cost/quality plane to compute a frontier over."
            )
        )

    stats = group_stats(results, price_book)
    resolved_baseline = resolve_baseline_strategy(results, baseline_strategy)

    # ---------------------------------------------------------- dominance
    dominators: dict[str, list[str]] = {g.label: [] for g in stats}
    for candidate in stats:
        for other in stats:
            if other is candidate:
                continue
            if dominates(other, candidate):
                dominators[candidate.label].append(other.label)
    for label in dominators:
        dominators[label].sort()

    frontier = [g for g in stats if not dominators[g.label]]
    # Cheapest -> best.  Ties on cost are broken by higher quality first.
    frontier.sort(key=lambda g: (g.cost_per_task_usd, -g.quality, g.label))

    # ------------------------------------------------------- value scores
    raw_values = {g.label: g.quality / max(g.cost_per_task_usd, COST_EPSILON) for g in stats}
    best_raw = max(raw_values.values()) if raw_values else 0.0

    points: list[ParetoPoint] = []
    for group in sorted(stats, key=lambda g: (g.cost_per_task_usd, -g.quality, g.label)):
        points.append(
            ParetoPoint(
                label=group.label,
                model_id=group.model_id,
                strategy_id=group.strategy_id,
                cost_usd=group.cost_usd,
                cost_per_task_usd=group.cost_per_task_usd,
                quality=group.quality,
                billable_tokens_per_task=group.billable_tokens_per_task,
                on_frontier=not dominators[group.label],
                dominated_by=list(dominators[group.label]),
                value_score=safe_div(raw_values[group.label], best_raw),
            )
        )

    # ----------------------------------------------------------- headline
    frontier_labels = [g.label for g in frontier]
    cheapest_label = frontier_labels[0] if frontier_labels else None
    # Value is measured on the frontier only: a dominated point can never win it
    # (its dominator is at least as cheap and at least as good), but restricting
    # the comparison keeps the definition unambiguous.
    best_value_label = (
        min(
            (p for p in points if p.on_frontier),
            key=lambda p: (-p.value_score, -p.quality, p.cost_per_task_usd, p.label),
        ).label
        if frontier
        else None
    )
    best_quality_label = (
        min(
            (p for p in points if p.on_frontier),
            key=lambda p: (-p.quality, p.cost_per_task_usd, p.label),
        ).label
        if frontier
        else None
    )

    if len(frontier) >= 2:
        cost_span = frontier[-1].cost_per_task_usd - frontier[0].cost_per_task_usd
        quality_span = max(g.quality for g in frontier) - min(g.quality for g in frontier)
        gain_per_usd = safe_div(quality_span, cost_span)
    else:
        gain_per_usd = 0.0

    knee = _knee_point(frontier)
    baseline_label = _pick_baseline_label(points, resolved_baseline)

    return ParetoFrontier(
        points=points,
        frontier_labels=frontier_labels,
        frontier_quality_gain_per_usd=gain_per_usd,
        baseline_label=baseline_label,
        best_value_label=best_value_label,
        best_quality_label=best_quality_label,
        cheapest_label=cheapest_label,
        # Published explicitly so renderers never re-derive it: the HTML report,
        # the Streamlit panel and the README figure all read this one field.
        knee_label=knee.label,
        explanation=_explain(
            frontier,
            knee,
            gain_per_usd,
            cheapest_label,
            best_value_label,
            best_quality_label,
            baseline_label,
        ),
    )


__all__ = [
    "COST_EPSILON",
    "dominates",
    "label_of",
    "pareto_frontier",
]
