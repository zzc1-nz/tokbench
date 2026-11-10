"""Run-level aggregation: headline summary, token-efficiency leaderboard, and the
chart-shaped views (per-category breakdown, stacked token decomposition, time
series).

Everything here is a **pure function over lists** of
:class:`~tokbench.models.CallRecord` / :class:`~tokbench.models.TaskResult`; no
file I/O happens outside :mod:`tokbench.analysis.report`.  That is what lets the
same numbers be recomputed from a JSONL dump, a notebook, or a test fixture.

Token-efficiency leaderboard — the headline ranking
---------------------------------------------------
The project's core question is *"which (model, strategy) pair buys the most
quality per token?"*.  Cost alone is the wrong answer (a cheap model that answers
badly is not efficient) and quality alone is the wrong answer (a huge model that
maxes the grader is not efficient either).  So each group is scored on the ratio

    Q_i   = mean quality of group i's cells            (in [0, 1])
    T_i   = billable tokens per task for group i
          = sum(Usage.billable_tokens) / cells
    raw_i = Q_i / T_i                                  (quality per billable token)

and then **normalised to the best group** so the winner scores exactly 1.0 and
every other row is a fraction of the winner:

    efficiency_score_i = raw_i / max_j(raw_j)          (in [0, 1])

``Usage.billable_tokens`` is uncached-input + output: a cache hit is genuinely
cheaper *and* cheaper in effective context spend, so it belongs in the ranking.
``T_i`` is floored at ``1e-9`` tokens before the division, which makes a
zero-token row the efficiency *ceiling* instead of a ``ZeroDivisionError``.

Ties (equal score to 1e-12) are broken by lower ``cost_per_task_usd``, then by
``model_id`` and ``strategy_id`` ascending — a total order, so ranking is
reproducible run to run.

Cost basis
----------
Every cost in this module is re-derived from the :class:`~tokbench.pricing.PriceBook`
when the model is priced, falling back to the cost recorded on the record when it
is not (:func:`cost_of`).  Re-deriving is deliberate: it is what makes
``tokbench report --prices my_prices.yaml`` able to restate a whole run without
re-running a single API call.  The recorded value is only a fallback for models
missing from the book (the report notes that case explicitly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import CallRecord, CallStatus, RunManifest, TaskResult, Usage
from ..pricing import PriceBook
from ..report_schema import (
    CategoryRow,
    EfficiencyRow,
    RunSummary,
    SeriesPoint,
    StackedBarRow,
)

#: Smallest token denominator used when normalising the efficiency ratio.  A
#: group with zero billable tokens must not raise; it should win the ranking.
MIN_TOKEN_DENOMINATOR = 1e-9

#: Call statuses that count as a hard failure.  ``BUDGET_EXCEEDED`` and
#: ``SKIPPED`` mean "never attempted", which is not an adapter failure, so they
#: are excluded from the failure count.
_FAILED_STATUSES = (CallStatus.ERROR, CallStatus.TIMEOUT)


# --------------------------------------------------------------------------- #
# Numeric helpers (every division in this package goes through these)
# --------------------------------------------------------------------------- #
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` for a zero or non-finite denominator.

    Guards the empty/one-cell run: a report with no data must produce a valid
    payload, never a ``ZeroDivisionError`` or a ``nan`` that leaks into JSON.
    """
    if denominator == 0.0 or not math.isfinite(denominator):
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def safe_pct(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Percentage form of :func:`safe_div` (``0.5 -> 50.0``)."""
    return safe_div(numerator, denominator, default / 100.0) * 100.0


def label_of(model_id: str, strategy_id: str) -> str:
    """Canonical ``(model, strategy)`` label used by every chart and table.

    One definition, used by the leaderboard, the Pareto points and the renderers,
    so the same group can never appear under two spellings in one report.
    """
    return f"{model_id} / {strategy_id}"


def cost_of(
    price_book: PriceBook,
    model_id: str,
    usage: Usage,
    recorded: float = 0.0,
) -> float:
    """USD for ``usage`` priced with ``price_book``.

    Falls back to the ``recorded`` cost when the model has no price entry, so an
    unpriced local model degrades to "whatever the run recorded" instead of
    crashing the whole report.  :func:`tokbench.analysis.report.build_report`
    surfaces that fallback in ``ReportMeta.notes``.
    """
    if price_book.has(model_id):
        return price_book.cost(model_id, usage)
    return recorded


def mean_quality(results: list[TaskResult]) -> float:
    """Mean graded quality, ignoring ungraded cells.

    Ungraded cells are *excluded* rather than scored 0: a grader that did not run
    is missing data, not evidence of a bad answer.  With no graded cell at all
    the mean is 0.0.
    """
    scores = [r.quality.score for r in results if r.quality is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def sum_usage(results: list[TaskResult]) -> Usage:
    """Element-wise sum of every cell's total usage."""
    return sum((r.usage_total for r in results), Usage())


# --------------------------------------------------------------------------- #
# (model, strategy) grouping — shared by the leaderboard and the Pareto module
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GroupStats:
    """Aggregated numbers for one ``(model_id, strategy_id)`` group.

    Kept as an intermediate value object so the leaderboard, the Pareto frontier
    and the retry analysis all derive their points from *one* aggregation rule
    (same cells, same cost basis, same quality mean).  Divergent definitions
    between charts are the classic way a benchmark report stops being credible.
    """

    model_id: str
    strategy_id: str
    cells: int
    successes: int
    usage: Usage
    cost_usd: float
    quality: float
    mean_latency_ms: float

    @property
    def label(self) -> str:
        """Stable display label, e.g. ``"deepseek-chat / cot"``."""
        return label_of(self.model_id, self.strategy_id)

    @property
    def success_rate(self) -> float:
        return safe_div(float(self.successes), float(self.cells))

    @property
    def billable_tokens(self) -> int:
        return self.usage.billable_tokens

    @property
    def tokens_per_task(self) -> float:
        return safe_div(float(self.usage.total_tokens), float(self.cells))

    @property
    def billable_tokens_per_task(self) -> float:
        return safe_div(float(self.usage.billable_tokens), float(self.cells))

    @property
    def output_tokens_per_task(self) -> float:
        return safe_div(float(self.usage.output_tokens), float(self.cells))

    @property
    def cost_per_task_usd(self) -> float:
        return safe_div(self.cost_usd, float(self.cells))


def group_stats(results: list[TaskResult], price_book: PriceBook) -> list[GroupStats]:
    """Aggregate ``results`` into one :class:`GroupStats` per (model, strategy).

    Output is sorted by ``(model_id, strategy_id)``; sets are never iterated, so
    the order is identical on every run and every machine.
    """
    buckets: dict[tuple[str, str], list[TaskResult]] = {}
    for r in results:
        buckets.setdefault((r.model_id, r.strategy_id), []).append(r)

    stats: list[GroupStats] = []
    for model_id, strategy_id in sorted(buckets):
        rows = buckets[(model_id, strategy_id)]
        usage = sum_usage(rows)
        recorded = sum(r.cost_usd for r in rows)
        stats.append(
            GroupStats(
                model_id=model_id,
                strategy_id=strategy_id,
                cells=len(rows),
                successes=sum(1 for r in rows if r.final_status.succeeded),
                usage=usage,
                cost_usd=cost_of(price_book, model_id, usage, recorded),
                quality=mean_quality(rows),
                mean_latency_ms=safe_div(sum(r.latency_ms for r in rows), float(len(rows))),
            )
        )
    return stats


def resolve_baseline_strategy(results: list[TaskResult], baseline: str | None = None) -> str | None:
    """Pick the strategy every ablation is measured against.

    If ``baseline`` names a strategy that is actually present, it wins.  Otherwise
    we auto-pick the strategy with the **lowest mean billable tokens per task** —
    the cheapest way to answer, which is the natural "do nothing clever" control.
    Ties break alphabetically so the choice is reproducible.

    Returns ``None`` for an empty run.  A caller-supplied baseline that does not
    exist falls back silently here; :func:`build_report` records that in the
    report notes so it can never be mistaken for an honoured request.
    """
    buckets: dict[str, list[TaskResult]] = {}
    for r in results:
        buckets.setdefault(r.strategy_id, []).append(r)
    if not buckets:
        return None
    if baseline is not None and baseline in buckets:
        return baseline

    def mean_billable(strategy: str) -> float:
        rows = buckets[strategy]
        return safe_div(float(sum_usage(rows).billable_tokens), float(len(rows)))

    return min(sorted(buckets), key=mean_billable)


# --------------------------------------------------------------------------- #
# Headline summary
# --------------------------------------------------------------------------- #
def _wall_clock_s(calls: list[CallRecord]) -> float:
    """Observed span of the call log, used when the manifest has no timing."""
    if len(calls) < 2:
        return 0.0
    start = min(c.started_at.timestamp() for c in calls)
    end = max(c.ended_at.timestamp() for c in calls)
    return max(end - start, 0.0)


def summarize(
    calls: list[CallRecord],
    results: list[TaskResult],
    manifest: RunManifest | None,
    *,
    price_book: PriceBook | None = None,
) -> RunSummary:
    """Headline numbers for the whole run.

    Resolution order is *results first, call log second, manifest third*: results
    are the logical cells (retries already folded in), the call log is the raw
    attempt stream, and the manifest is the runner's own bookkeeping for runs that
    crashed before writing results.

    ``planned_cells`` is derived from the observed matrix
    (models x strategies x tasks x repeats) because a run directory does not
    persist the task set; for a budget-aborted run that is a lower bound.  With no
    results at all we fall back to ``manifest.planned_calls`` so an aborted run
    still reports the plan it was measured against.

    ``success_rate`` is **cell-level** (final status ok / cells), not call-level:
    retries deliberately mask call failures, and a cell that eventually succeeded
    is a success for benchmark purposes.
    """
    models = sorted({r.model_id for r in results})
    strategies = sorted({r.strategy_id for r in results})
    tasks = sorted({r.task_id for r in results})
    repeats = len(sorted({r.repeat for r in results}))
    cells = len(results)

    planned_cells = len(models) * len(strategies) * len(tasks) * repeats
    if planned_cells == 0 and manifest is not None:
        planned_cells = manifest.planned_calls

    if calls:
        total_calls = len(calls)
        failed_calls = sum(1 for c in calls if c.status in _FAILED_STATUSES)
    elif manifest is not None:
        total_calls = manifest.completed_calls
        failed_calls = manifest.failed_calls
    else:
        total_calls = 0
        failed_calls = 0

    successes = sum(1 for r in results if r.final_status.succeeded)
    success_rate = safe_div(
        float(successes), float(cells), manifest.success_rate if manifest else 0.0
    )

    if results:
        usage = sum_usage(results)
        if price_book is not None:
            total_cost = sum(
                cost_of(price_book, r.model_id, r.usage_total, r.cost_usd) for r in results
            )
        else:
            total_cost = sum(r.cost_usd for r in results)
    elif calls:
        usage = sum((c.usage for c in calls), Usage())
        if price_book is not None:
            total_cost = sum(cost_of(price_book, c.model_id, c.usage, c.cost_usd) for c in calls)
        else:
            total_cost = sum(c.cost_usd for c in calls)
    elif manifest is not None:
        usage = manifest.total_usage
        total_cost = manifest.total_cost_usd
    else:
        usage = Usage()
        total_cost = 0.0

    if manifest is not None and manifest.wall_clock_s > 0:
        wall_clock = manifest.wall_clock_s
    else:
        wall_clock = _wall_clock_s(calls)

    billable = usage.billable_tokens
    return RunSummary(
        models=len(models),
        strategies=len(strategies),
        tasks=len(tasks),
        repeats=repeats,
        planned_cells=planned_cells,
        completed_cells=cells,
        total_calls=total_calls,
        failed_calls=failed_calls,
        retried_cells=sum(1 for r in results if r.retried),
        success_rate=success_rate,
        total_usage=usage,
        total_cost_usd=total_cost,
        total_billable_tokens=billable,
        wall_clock_s=wall_clock,
        cost_per_1k_billable_tokens=safe_div(total_cost * 1000.0, float(billable)),
        cache_hit_rate=usage.cache_hit_rate,
    )


# --------------------------------------------------------------------------- #
# Token-efficiency leaderboard
# --------------------------------------------------------------------------- #
def leaderboard(
    results: list[TaskResult],
    price_book: PriceBook,
    *,
    min_cells: int = 1,
) -> list[EfficiencyRow]:
    """Rank (model, strategy) pairs by quality per billable token.

    Formula (see the module docstring for the derivation)::

        raw_i   = mean_quality_i / max(billable_tokens_per_task_i, 1e-9)
        score_i = raw_i / max_j(raw_j)

    The best row therefore scores ``1.0`` and every other row is "fraction of the
    winner's efficiency".  Rows with fewer than ``min_cells`` cells are dropped
    (one lucky cell should not top a leaderboard); ``min_cells`` is clamped to at
    least 1.

    ``tokens_per_quality_point`` is the inverse view — how many billable tokens
    one full quality point costs this group (``None`` when quality is 0, where the
    ratio is undefined).  ``provider`` stays empty because :class:`TaskResult`
    does not carry it; :func:`build_report` fills it from the call log.
    """
    threshold = max(min_cells, 1)
    stats = [g for g in group_stats(results, price_book) if g.cells >= threshold]

    raws = [g.quality / max(g.billable_tokens_per_task, MIN_TOKEN_DENOMINATOR) for g in stats]
    best_raw = max(raws) if raws else 0.0

    rows: list[EfficiencyRow] = []
    for group, raw in zip(stats, raws, strict=False):
        rows.append(
            EfficiencyRow(
                model_id=group.model_id,
                strategy_id=group.strategy_id,
                cells=group.cells,
                success_rate=group.success_rate,
                total_usage=group.usage,
                billable_tokens=group.billable_tokens,
                cost_usd=group.cost_usd,
                tokens_per_task=group.tokens_per_task,
                billable_tokens_per_task=group.billable_tokens_per_task,
                cost_per_task_usd=group.cost_per_task_usd,
                output_tokens_per_task=group.output_tokens_per_task,
                mean_quality=group.quality,
                mean_latency_ms=group.mean_latency_ms,
                cache_hit_rate=group.usage.cache_hit_rate,
                tokens_per_quality_point=(
                    safe_div(group.billable_tokens_per_task, group.quality)
                    if group.quality > 0.0
                    else None
                ),
                efficiency_score=safe_div(raw, best_raw),
            )
        )

    # Total order: score desc, then cheaper, then stable ids.  Never rely on the
    # incoming list order — parallel runners write results out of order.
    rows.sort(
        key=lambda r: (
            -round(r.efficiency_score, 12),
            r.cost_per_task_usd,
            r.model_id,
            r.strategy_id,
        )
    )
    for index, row in enumerate(rows, start=1):
        row.rank = index
    return rows


# --------------------------------------------------------------------------- #
# Per-category breakdown
# --------------------------------------------------------------------------- #
def category_breakdown(
    results: list[TaskResult],
    price_book: PriceBook | None = None,
) -> list[CategoryRow]:
    """Quality / cost / token economy sliced by task category.

    The point is diagnosis: a strategy can win overall and still lose on the
    category that matters.  Pass ``price_book`` to re-derive cost from token counts
    (what :func:`tokbench.analysis.report.build_report` does, so this table agrees
    with the leaderboard); omit it to use the cost recorded on each cell.
    """
    buckets: dict[tuple[str, str, str], list[TaskResult]] = {}
    for r in results:
        buckets.setdefault((r.model_id, r.strategy_id, r.category), []).append(r)

    rows: list[CategoryRow] = []
    for model_id, strategy_id, category in sorted(buckets):
        rows_in = buckets[(model_id, strategy_id, category)]
        usage = sum_usage(rows_in)
        cells = len(rows_in)
        recorded = sum(r.cost_usd for r in rows_in)
        cost = (
            cost_of(price_book, model_id, usage, recorded) if price_book is not None else recorded
        )
        rows.append(
            CategoryRow(
                model_id=model_id,
                strategy_id=strategy_id,
                category=category,
                cells=cells,
                success_rate=safe_div(
                    float(sum(1 for r in rows_in if r.final_status.succeeded)),
                    float(cells),
                ),
                billable_tokens_per_task=safe_div(float(usage.billable_tokens), float(cells)),
                cost_per_task_usd=safe_div(cost, float(cells)),
                mean_quality=mean_quality(rows_in),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Stacked token decomposition
# --------------------------------------------------------------------------- #
def stacked_bars(
    results: list[TaskResult],
    price_book: PriceBook | None = None,
) -> list[StackedBarRow]:
    """Token-type decomposition per (model, strategy) — the "where did the tokens
    go?" chart.

    Splits total input into uncached / cache-read / cache-write so a strategy's
    token bill can be attributed: a few-shot strategy inflates uncached input, a
    cached-prefix strategy shifts it into cache reads, chain-of-thought inflates
    output.  ``price_book`` re-derives cost from those token counts; omit it to use
    the recorded cell cost.
    """
    buckets: dict[tuple[str, str], list[TaskResult]] = {}
    for r in results:
        buckets.setdefault((r.model_id, r.strategy_id), []).append(r)

    rows: list[StackedBarRow] = []
    for model_id, strategy_id in sorted(buckets):
        rows_in = buckets[(model_id, strategy_id)]
        usage = sum_usage(rows_in)
        recorded = sum(r.cost_usd for r in rows_in)
        cost = (
            cost_of(price_book, model_id, usage, recorded) if price_book is not None else recorded
        )
        rows.append(
            StackedBarRow(
                model_id=model_id,
                strategy_id=strategy_id,
                uncached_input_tokens=usage.uncached_input_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cost_usd=cost,
                cells=len(rows_in),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Chronological time series
# --------------------------------------------------------------------------- #
def time_series(calls: list[CallRecord]) -> list[SeriesPoint]:
    """Cumulative token / cost / cache-hit trajectory of the run.

    Sorted by ``(started_at, record_id)``.  The record id is the tie-breaker
    because a concurrent runner can emit several calls with an identical
    timestamp, and a chart that reshuffles between renders is a bug.

    ``cumulative_cache_hit_rate`` is a running ``cache_read / cache_eligible``
    ratio, not a mean of per-call rates: mean-of-ratios overweights small calls
    and is the wrong statistic for a spend question.  ``cache_eligible`` is
    ``input_tokens - cache_write_tokens`` so the running ratio uses the same
    definition as :attr:`Usage.cache_hit_rate` and the cache-sensitivity curve —
    one run must never report two different hit rates.
    """
    ordered = sorted(calls, key=lambda c: (c.started_at.timestamp(), c.record_id))

    points: list[SeriesPoint] = []
    cum_tokens = 0
    cum_billable = 0
    cum_cost = 0.0
    cum_cache_read = 0
    cum_eligible = 0

    for index, call in enumerate(ordered):
        usage = call.usage
        cum_tokens += usage.total_tokens
        cum_billable += usage.billable_tokens
        cum_cost += call.cost_usd
        cum_cache_read += usage.cache_read_tokens
        cum_eligible += usage.cache_eligible_tokens
        points.append(
            SeriesPoint(
                t=call.started_at,
                index=index,
                model_id=call.model_id,
                strategy_id=call.strategy_id,
                tokens=usage.total_tokens,
                billable_tokens=usage.billable_tokens,
                cost_usd=call.cost_usd,
                cumulative_tokens=cum_tokens,
                cumulative_billable_tokens=cum_billable,
                cumulative_cost_usd=cum_cost,
                cumulative_cache_hit_rate=min(
                    safe_div(float(cum_cache_read), float(cum_eligible)), 1.0
                ),
            )
        )
    return points


__all__ = [
    "GroupStats",
    "MIN_TOKEN_DENOMINATOR",
    "category_breakdown",
    "cost_of",
    "group_stats",
    "label_of",
    "leaderboard",
    "mean_quality",
    "resolve_baseline_strategy",
    "safe_div",
    "safe_pct",
    "stacked_bars",
    "sum_usage",
    "summarize",
    "time_series",
]
