"""Retry overhead: how much of the bill is pure waste?

This is the report's headline "honesty" metric.  A benchmark that silently
retries failed calls looks cheap and reliable while quietly burning budget, so we
attribute the retry spend explicitly.

Where the number comes from
---------------------------
:class:`~tokbench.models.TaskResult` already separates the two halves of a cell::

    usage_total        tokens across every attempt
    usage_first_attempt  tokens of attempt 1 only
    retry_overhead_usage = usage_total - usage_first_attempt   (per component)

so retry overhead is **measured**, not inferred from ``attempts`` counts.  The
retry cost is that usage priced with the same :class:`~tokbench.pricing.PriceBook`
as everything else — which matters, because retries of a cache-heavy call are
cheaper than retries of a cold one, and a flat "retries x average cost" estimate
would hide that.

Definitions (all per ``(model, strategy)`` group, then rolled up)
-----------------------------------------------------------------
    retry_rate        = cells with attempts > 1 / cells
    extra_attempts    = sum(attempts - 1)
    retry_cost_usd    = price_book.cost(retry_overhead_usage)
    retry_cost_share  = retry_cost_usd / group total cost
    retry_token_share = retry billable tokens / group total billable tokens

Billable tokens (uncached input + output) are used for the token share so the
token and cost shares answer the same question on the same basis; raw
``total_tokens`` would count cache reads that were nearly free.

``wasted_pct_of_budget`` is the spend-relative form of the headline
(``overall_retry_cost_share * 100``) because the analysis layer never sees the
configured budget ceiling — the budget lives in the run config, not in the
results.  It is therefore "percent of *actual* spend wasted", which is the honest
number; a run that aborted on budget is reported by the summary, not here.

``dominant_error_kind`` is the most frequent ``error_kind`` among failed calls of
that group (ties break alphabetically for determinism).  Failed calls are matched
by ``(model_id, strategy_id)`` straight from the call log, so a group with retries
but a lost call log simply reports no error breakdown rather than failing.
"""

from __future__ import annotations

from ..models import CallRecord, CallStatus, TaskResult, Usage
from ..pricing import PriceBook
from ..report_schema import RetryAnalysis, RetryRow
from .aggregate import cost_of, safe_div, sum_usage

#: Statuses that represent a failed *attempt* (budget/skip means never tried).
_FAILED_STATUSES = (CallStatus.ERROR, CallStatus.TIMEOUT)

#: Fallback bucket for failed calls whose adapter did not set ``error_kind``.
UNKNOWN_ERROR_KIND = "unknown"


def _retry_cost(
    price_book: PriceBook,
    model_id: str,
    retry_usage: Usage,
    total_usage: Usage,
    recorded_cost: float,
) -> float:
    """Price the retry overhead, falling back to a proportional slice of the
    recorded cell cost for models that are not in the price book."""
    if price_book.has(model_id):
        return price_book.cost(model_id, retry_usage)
    total_tokens = total_usage.total_tokens
    if total_tokens <= 0:
        return 0.0
    return recorded_cost * (retry_usage.total_tokens / total_tokens)


def _error_breakdown(calls: list[CallRecord]) -> dict[str, int]:
    """Count failed calls per normalised ``error_kind`` (sorted keys)."""
    counts: dict[str, int] = {}
    for call in calls:
        if call.status not in _FAILED_STATUSES:
            continue
        kind = call.error_kind or UNKNOWN_ERROR_KIND
        counts[kind] = counts.get(kind, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def _dominant_error(counts: dict[str, int]) -> str | None:
    """Most frequent error kind; ties break alphabetically."""
    if not counts:
        return None
    return min(counts, key=lambda k: (-counts[k], k))


def retry_analysis(
    calls: list[CallRecord],
    results: list[TaskResult],
    price_book: PriceBook,
) -> RetryAnalysis:
    """Retry waste broken down per ``(model, strategy)`` plus run totals.

    Empty-safe: with no results (and no calls) the returned analysis is a valid
    object whose ``explanation`` says the run has nothing to analyse.  With results
    but no failed calls it reports a 0% share and says so explicitly — "no retries
    observed" is a real finding, not a missing number.
    """
    if not results and not calls:
        return RetryAnalysis(
            explanation=(
                "No results or calls in this run: there is no retry activity to "
                "attribute. A retry-overhead figure only exists once at least one "
                "cell has been attempted."
            )
        )

    # ------------------------------------------------------------ per group
    buckets: dict[tuple[str, str], list[TaskResult]] = {}
    for r in results:
        buckets.setdefault((r.model_id, r.strategy_id), []).append(r)

    calls_by_group: dict[tuple[str, str], list[CallRecord]] = {}
    for call in calls:
        calls_by_group.setdefault((call.model_id, call.strategy_id), []).append(call)

    # Union of both sources so a group with calls but no results (every attempt
    # failed and the runner wrote no cell) still shows up as waste.
    keys = sorted(set(buckets) | set(calls_by_group))

    rows: list[RetryRow] = []
    for model_id, strategy_id in keys:
        group_results = buckets.get((model_id, strategy_id), [])
        group_calls = calls_by_group.get((model_id, strategy_id), [])

        retry_usage = sum((r.retry_overhead_usage for r in group_results), Usage())
        total_usage = sum_usage(group_results)
        recorded_cost = sum(r.cost_usd for r in group_results)
        total_cost = cost_of(price_book, model_id, total_usage, recorded_cost)
        retry_cost = _retry_cost(price_book, model_id, retry_usage, total_usage, recorded_cost)
        errors = _error_breakdown(group_calls)

        cells = len(group_results)
        retried_cells = sum(1 for r in group_results if r.retried)
        rows.append(
            RetryRow(
                model_id=model_id,
                strategy_id=strategy_id,
                cells=cells,
                retried_cells=retried_cells,
                retry_rate=safe_div(float(retried_cells), float(cells)),
                extra_attempts=sum(max(r.attempts - 1, 0) for r in group_results),
                retry_usage=retry_usage,
                retry_cost_usd=retry_cost,
                total_cost_usd=total_cost,
                retry_cost_share=safe_div(retry_cost, total_cost),
                retry_token_share=safe_div(
                    float(retry_usage.billable_tokens), float(total_usage.billable_tokens)
                ),
                dominant_error_kind=_dominant_error(errors),
                error_breakdown=errors,
            )
        )

    # --------------------------------------------------------------- totals
    total_retry_cost = sum(r.retry_cost_usd for r in rows)
    total_cost = sum(r.total_cost_usd for r in rows)
    # Total billable tokens = every cell's billable usage; the retry slice is the
    # part of it that bought nothing.
    total_tokens = sum(sum_usage(group).billable_tokens for _, group in sorted(buckets.items()))
    total_retry_tokens = sum(r.retry_usage.billable_tokens for r in rows)

    # Roll the per-group error breakdowns up for the run-level view.
    overall_errors: dict[str, int] = {}
    for row in rows:
        for kind, count in row.error_breakdown.items():
            overall_errors[kind] = overall_errors.get(kind, 0) + count
    overall_errors = {k: overall_errors[k] for k in sorted(overall_errors)}

    overall_share = safe_div(total_retry_cost, total_cost)
    retry_token_share = safe_div(float(total_retry_tokens), float(total_tokens))

    worst: RetryRow | None = None
    if rows:
        worst = min(
            rows,
            key=lambda r: (-r.retry_cost_usd, -r.retry_cost_share, r.model_id, r.strategy_id),
        )
        if worst.retry_cost_usd <= 0.0:
            worst = None

    return RetryAnalysis(
        rows=rows,
        total_retry_cost_usd=total_retry_cost,
        total_cost_usd=total_cost,
        overall_retry_cost_share=overall_share,
        total_retry_tokens=total_retry_tokens,
        total_tokens=total_tokens,
        retry_token_share=retry_token_share,
        wasted_pct_of_budget=overall_share * 100.0,
        error_breakdown=overall_errors,
        worst_offender=(f"{worst.model_id} / {worst.strategy_id}" if worst is not None else None),
        explanation=_explain(
            rows, overall_share, retry_token_share, total_retry_cost, total_cost, worst
        ),
    )


def _explain(
    rows: list[RetryRow],
    overall_share: float,
    retry_token_share: float,
    total_retry_cost: float,
    total_cost: float,
    worst: RetryRow | None,
) -> str:
    """Headline sentence: what fraction of spend is pure retry waste, and who."""
    retried = sum(r.retried_cells for r in rows)
    cells = sum(r.cells for r in rows)
    attempts = sum(r.extra_attempts for r in rows)

    if total_cost <= 0.0 and total_retry_cost <= 0.0 and cells == 0:
        return "No completed cells and no priced calls: there is no spend to attribute to retries."
    if retried == 0:
        return (
            f"No retries observed across {cells} cell(s): 0.0% of the "
            f"${total_cost:.6f} total spend was lost to failed first attempts. "
            "The reported cost is all productive spend."
        )

    parts = [
        f"Retries burned ${total_retry_cost:.6f} of ${total_cost:.6f} total spend "
        f"({overall_share * 100:.2f}% of the bill, {retry_token_share * 100:.2f}% of "
        "billable tokens) — that share is pure waste: it bought no additional "
        "graded output."
    ]
    if worst is not None:
        parts.append(
            f"Worst offender is {worst.model_id} / {worst.strategy_id}: "
            f"${worst.retry_cost_usd:.6f} of retry spend "
            f"({worst.retry_cost_share * 100:.2f}% of that group's cost) from "
            f"{worst.retried_cells}/{worst.cells} retried cells"
            + (
                f", dominated by '{worst.dominant_error_kind}' errors."
                if worst.dominant_error_kind
                else "."
            )
        )
    parts.append(f"{retried}/{cells} cells needed a retry, {attempts} extra attempt(s) in total.")
    return " ".join(parts)


__all__ = ["UNKNOWN_ERROR_KIND", "retry_analysis"]
