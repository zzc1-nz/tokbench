"""Strategy ablation deltas: is a prompt strategy worth what it costs?

The leaderboard and frontier say *which* configuration wins.  This module answers
the causal question the experiment was actually built to ask: **holding the model
fixed, what does switching strategy X instead of baseline Y buy, and what does it
cost?**  That is the difference between a benchmark and an ablation table.

Each row compares one strategy against the baseline, first aggregated across all
models (``model_id is None``) and then once per model (``model_id`` set) so a
strategy that helps small models and hurts large ones is visible instead of
averaged away.

Definitions
-----------
    token_delta_pct = (strategy_billable_per_task - baseline_billable_per_task)
                      / baseline_billable_per_task * 100
    cost_delta_pct  = same, on cost per task (USD)
    quality_delta   = strategy_mean_quality - baseline_mean_quality   (absolute)
    tokens_per_quality_delta
                    = (strategy_billable_per_task - baseline_billable_per_task)
                      / (quality_delta / 0.01)

``tokens_per_quality_delta`` is the *price of quality*: extra billable tokens
needed to move mean quality by +0.01.  It is ``None`` when quality did not
improve, because "tokens per quality point" is undefined (or meaningless) when the
denominator is zero or negative — reporting ``inf`` there would be worse than
reporting nothing.

Both deltas are **signed**: negative means the strategy is cheaper/token-lighter
than the baseline, which is the interesting case for a compression strategy.

Verdict thresholds (documented, not magic numbers)
--------------------------------------------------
One quality threshold is used in both directions: ``QUALITY_STEP``-scale moves
below ``WORTH_IT_MIN_QUALITY_GAIN`` (``+0.02``, i.e. two grader points) are noise
on a run of this size.

``worth it`` — the strategy is no worse on quality and no more expensive, *with at
least one strict improvement*; or it buys at least two quality points for at most
a 2x cost premium (``cost_delta_pct <= 100``).  Doubling the bill is defensible
when the answer actually gets better; tripling it is not, however good the answer.
``wasteful`` — it costs at least 10% more and buys less than two quality points:
the prompt-bloat signature, paying for tokens that did not improve the answer.
Also ``wasteful`` when it is cheaper but loses at least two quality points, i.e.
the token saving was paid for with quality the run actually measured.
``neutral`` — everything else: a no-op, or a small trade on one axis for a small
gain on the other, which this run is not powered to call.

A strategy that is both cheaper and better is ``worth it`` by the first rule.

Baseline selection
------------------
``baseline`` wins when it names a strategy present in the results.  Otherwise the
baseline is auto-picked as the strategy with the lowest mean billable tokens per
task: the cheapest way to answer is the natural "do nothing clever" control, and
it makes the table read as "X costs N% more than the leanest option".  Ties break
alphabetically for determinism.
"""

from __future__ import annotations

from ..models import TaskResult
from ..pricing import PriceBook
from ..report_schema import StrategyDeltaRow
from .aggregate import (
    cost_of,
    mean_quality,
    resolve_baseline_strategy,
    safe_div,
    safe_pct,
    sum_usage,
)

#: Quality gain that counts as a real improvement (+0.02 = two grader points).
#: Moves smaller than this are noise at the run sizes tokbench targets, and the
#: same threshold is used to judge a quality *loss*.
WORTH_IT_MIN_QUALITY_GAIN = 0.02

#: Cost premium (percent) still acceptable when quality genuinely improves.
#: 100% = "you may double the bill for a real quality gain, not more".
WORTH_IT_MAX_COST_PREMIUM_PCT = 100.0

#: Cost premium above which spending without a quality gain is "wasteful".
WASTEFUL_MIN_COST_PREMIUM_PCT = 10.0

#: Quality loss (negative) that makes a token saving a bad trade.
WASTEFUL_MAX_QUALITY_LOSS = -WORTH_IT_MIN_QUALITY_GAIN

#: The quality step the "tokens per quality point" figure is expressed in.
QUALITY_STEP = 0.01


def _verdict(token_delta_pct: float, cost_delta_pct: float, quality_delta: float) -> str:
    """Classify one ablation comparison (thresholds documented in the module doc)."""
    cheaper = cost_delta_pct < 0.0
    pricier = cost_delta_pct > 0.0
    better = quality_delta > 0.0
    worse = quality_delta < 0.0

    # Strictly better on one axis and no worse on the other.
    if better and not pricier:
        return "worth it"
    if cheaper and not worse:
        return "worth it"
    # A real quality gain for a bounded price premium.
    if quality_delta >= WORTH_IT_MIN_QUALITY_GAIN and (
        cost_delta_pct <= WORTH_IT_MAX_COST_PREMIUM_PCT
    ):
        return "worth it"
    # Paying materially more for a quality move that is within noise.
    if cost_delta_pct >= WASTEFUL_MIN_COST_PREMIUM_PCT and (
        quality_delta < WORTH_IT_MIN_QUALITY_GAIN
    ):
        return "wasteful"
    # Saving tokens by losing real quality.
    if quality_delta <= WASTEFUL_MAX_QUALITY_LOSS:
        return "wasteful"
    return "neutral"


def _group_metrics(
    results: list[TaskResult], price_book: PriceBook | None
) -> tuple[float, float, float]:
    """``(billable_tokens_per_task, cost_per_task_usd, mean_quality)`` for a group."""
    cells = len(results)
    usage = sum_usage(results)
    if price_book is not None:
        model_id = results[0].model_id
        cost = cost_of(price_book, model_id, usage, sum(r.cost_usd for r in results))
    else:
        cost = sum(r.cost_usd for r in results)
    return (
        safe_div(float(usage.billable_tokens), float(cells)),
        safe_div(cost, float(cells)),
        mean_quality(results),
    )


def _row(
    baseline_strategy: str,
    strategy_id: str,
    model_id: str | None,
    baseline: tuple[float, float, float],
    candidate: tuple[float, float, float],
) -> StrategyDeltaRow:
    """Build one comparison row from two ``(tokens, cost, quality)`` triples."""
    base_tokens, base_cost, base_quality = baseline
    tokens, cost, quality = candidate
    token_delta_pct = safe_pct(tokens - base_tokens, base_tokens)
    cost_delta_pct = safe_pct(cost - base_cost, base_cost)
    quality_delta = quality - base_quality
    tokens_per_quality = (
        (tokens - base_tokens) / (quality_delta / QUALITY_STEP) if quality_delta > 0.0 else None
    )
    return StrategyDeltaRow(
        baseline_strategy=baseline_strategy,
        strategy_id=strategy_id,
        model_id=model_id,
        baseline_billable_per_task=base_tokens,
        strategy_billable_per_task=tokens,
        token_delta_pct=token_delta_pct,
        baseline_cost_per_task_usd=base_cost,
        strategy_cost_per_task_usd=cost,
        cost_delta_pct=cost_delta_pct,
        baseline_quality=base_quality,
        quality=quality,
        quality_delta=quality_delta,
        tokens_per_quality_delta=tokens_per_quality,
        verdict=_verdict(token_delta_pct, cost_delta_pct, quality_delta),
    )


def strategy_deltas(
    results: list[TaskResult],
    *,
    baseline: str | None = None,
    price_book: PriceBook | None = None,
) -> list[StrategyDeltaRow]:
    """Ablation deltas of every strategy against ``baseline``.

    Returns the global comparison first (``model_id is None``), then one block per
    model sorted by model id, each block sorted by ``strategy_id`` — so the table
    reads top-down without a sort step in the renderer.

    ``price_book`` is optional: pass it to re-price the cost columns from token
    counts (the report does), omit it to use the cost recorded on each cell.  The
    ``baseline`` strategy itself never appears as a candidate row, and a strategy
    with no cells in a given model simply produces no row for that model.

    Empty-safe: returns ``[]`` when there are no results or no baseline can be
    determined.
    """
    if not results:
        return []

    resolved = resolve_baseline_strategy(results, baseline)
    if resolved is None:
        return []

    strategies = sorted({r.strategy_id for r in results})
    if len(strategies) < 2:
        return []

    # ------------------------------------------------------------- baseline
    baseline_cells = [r for r in results if r.strategy_id == resolved]
    if not baseline_cells:
        return []
    baseline_global = _group_metrics(baseline_cells, price_book)

    baseline_by_model: dict[str, tuple[float, float, float]] = {}
    for model_id in sorted({r.model_id for r in baseline_cells}):
        baseline_by_model[model_id] = _group_metrics(
            [r for r in baseline_cells if r.model_id == model_id], price_book
        )

    rows: list[StrategyDeltaRow] = []

    # --------------------------------------------------------------- global
    for strategy_id in strategies:
        if strategy_id == resolved:
            continue
        cells = [r for r in results if r.strategy_id == strategy_id]
        if not cells:
            continue
        rows.append(
            _row(
                resolved,
                strategy_id,
                None,
                baseline_global,
                _group_metrics(cells, price_book),
            )
        )

    # ------------------------------------------------------------ per model
    models = sorted({r.model_id for r in results})
    for model_id in models:
        if model_id not in baseline_by_model:
            continue
        model_cells = [r for r in results if r.model_id == model_id]
        for strategy_id in sorted({r.strategy_id for r in model_cells}):
            if strategy_id == resolved:
                continue
            cells = [r for r in model_cells if r.strategy_id == strategy_id]
            if not cells:
                continue
            rows.append(
                _row(
                    resolved,
                    strategy_id,
                    model_id,
                    baseline_by_model[model_id],
                    _group_metrics(cells, price_book),
                )
            )
    return rows


__all__ = [
    "QUALITY_STEP",
    "WASTEFUL_MIN_COST_PREMIUM_PCT",
    "WORTH_IT_MAX_COST_PREMIUM_PCT",
    "WORTH_IT_MIN_QUALITY_GAIN",
    "strategy_deltas",
]
