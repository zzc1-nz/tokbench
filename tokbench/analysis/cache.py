"""Cache sensitivity: what would this run cost at a different cache hit rate?

Prompt caching is the single biggest lever on input cost — vendors charge roughly
10% of the input price for a cache *read* — so the report needs more than "here is
the hit rate we got".  It needs the counterfactual: *what does the bill look like
at 0%, at 40%, at 100%?*  That is what turns "we should cache more" from an
opinion into a number with a dollar sign on it.

Projection model
----------------
For each ``(model, strategy)`` group, hold **total prompt tokens and output tokens
fixed** and redistribute the input side between full-price and cache-read tokens:

    eligible_input = input_tokens - cache_write_tokens        (floored at 0)
    cache_read(h)  = h * eligible_input
    uncached(h)    = (1 - h) * eligible_input

    cost(h) = uncached(h) * price.input
            + cache_read(h) * price.cache_read
            + cache_write_tokens * price.cache_write          (held at observed)
            + output_tokens * price.output                    (held at observed)
            + reasoning_tokens * price.reasoning              (held at observed)
            all divided by 1e6

``cache_write_tokens`` is held fixed rather than scaled down as reads rise: a
cache write is a *property of the request shape*, and assuming it vanishes at
100% hit rate would silently invent a discount.  Holding it fixed is the
conservative choice — it under-states the savings, which is the right direction
for a benchmark.

Two consequences worth stating because they make the numbers auditable:

* The curve is **anchored at the observed point for a single group**: when every
  group caches at the same rate, ``cost(observed_hit_rate)`` reproduces the run's
  actual cost exactly (same tokens, same prices).  With several groups caching at
  *different* rates, the sweep applies one global rate to all of them, so the
  curve at the observed global rate lands within a fraction of a percent of the
  real bill rather than exactly on it — the projection redistributes caching
  evenly, which is the question "what if the whole run cached at rate h" answers.
  ``observed_cost_usd`` is always the *actual* cost, never the curve's estimate.
* ``cost(h)`` is affine in ``h`` with slope
  ``-sum(eligible_input * (input_price - cache_read_price)) / 1e6`` per unit hit
  rate, which is why ``marginal_usd_per_10pp`` (a 10pp finite difference of the
  curve) is exact rather than an approximation.

``observed_hit_rate`` is token-weighted — ``sum(cache_read) / sum(eligible_input)``
— not a mean of per-call rates.  Mean-of-ratios would let one tiny call with a
lucky hit dominate the headline; spend is what we are projecting, so tokens are
the right weights.

Models missing from the price book cannot be projected (we do not know their
cache_read price), so their observed cost is held constant across the curve and
the explanation says how many groups were excluded.
"""

from __future__ import annotations

from ..models import TaskResult, Usage
from ..pricing import MILLION, PriceBook
from ..report_schema import CachePoint, CacheSensitivity
from .aggregate import cost_of, safe_div, safe_pct, sum_usage

#: Minimum number of curve points; a 1-point "curve" is not a curve.
MIN_STEPS = 2

#: Width of the finite-difference window used for ``marginal_usd_per_10pp``.
_MARGINAL_WINDOW = 0.10


def _project_group(
    price_book: PriceBook,
    model_id: str,
    usage: Usage,
    recorded_cost: float,
    hit_rate: float,
) -> float:
    """Projected USD cost of one group at a hypothetical cache ``hit_rate``.

    Returns the recorded cost unchanged for models absent from the price book —
    a projection without prices would be fiction.
    """
    if not price_book.has(model_id):
        return recorded_cost
    price = price_book.get(model_id)
    eligible = max(usage.input_tokens - usage.cache_write_tokens, 0)
    cache_read = hit_rate * eligible
    uncached = max(eligible - cache_read, 0.0)
    total = (
        uncached * price.input
        + cache_read * (price.cache_read or 0.0)
        + usage.cache_write_tokens * (price.cache_write or 0.0)
        + usage.output_tokens * price.output
        + usage.reasoning_tokens * (price.reasoning or 0.0)
    )
    return total / MILLION


def cache_sensitivity(
    results: list[TaskResult],
    price_book: PriceBook,
    *,
    steps: int = 21,
) -> CacheSensitivity:
    """Projected total cost as a function of cache hit rate, plus the marginal
    value of another 10 percentage points.

    ``steps`` is clamped to at least :data:`MIN_STEPS`; the curve always spans
    ``0.0 .. 1.0`` inclusive, so ``curve[0]`` is the zero-cache cost and
    ``curve[-1]`` the full-cache cost.  The curve is sorted by hit rate ascending
    and therefore by cost descending — the shape a reader expects from "cost of
    caching".

    Empty-safe: with no results the returned object carries an explanation and an
    empty curve rather than raising.
    """
    step_count = max(int(steps), MIN_STEPS)
    if not results:
        return CacheSensitivity(
            explanation=(
                "No results to project: cache sensitivity needs at least one "
                "completed cell, because it re-prices that cell's tokens at "
                "hypothetical hit rates."
            )
        )

    # --------------------------------------------------------- group totals
    buckets: dict[tuple[str, str], list[TaskResult]] = {}
    for r in results:
        buckets.setdefault((r.model_id, r.strategy_id), []).append(r)

    groups: list[tuple[str, Usage, float]] = []
    for model_id, _strategy_id in sorted(buckets):
        rows = buckets[(model_id, _strategy_id)]
        groups.append(
            (
                model_id,
                sum_usage(rows),
                cost_of(price_book, model_id, sum_usage(rows), sum(r.cost_usd for r in rows)),
            )
        )

    observed_cost = sum(g[2] for g in groups)
    cells = len(results)

    # ------------------------------------------------------------ hit rates
    total_eligible = sum(max(g[1].input_tokens - g[1].cache_write_tokens, 0) for g in groups)
    total_cache_read = sum(g[1].cache_read_tokens for g in groups)
    observed_hit_rate = safe_div(float(total_cache_read), float(total_eligible))

    excluded = sum(1 for g in groups if not price_book.has(g[0]))

    def total_at(hit_rate: float) -> float:
        return sum(
            _project_group(price_book, model_id, usage, recorded, hit_rate)
            for model_id, usage, recorded in groups
        )

    zero_cost = total_at(0.0)
    full_cost = total_at(1.0)

    # ----------------------------------------------------------------- curve
    curve: list[CachePoint] = []
    for index in range(step_count):
        hit_rate = index / (step_count - 1)
        projected = total_at(hit_rate)
        savings = zero_cost - projected
        curve.append(
            CachePoint(
                cache_hit_rate=hit_rate,
                projected_cost_usd=projected,
                projected_cost_per_task_usd=safe_div(projected, float(cells)),
                savings_vs_zero_cache_usd=savings,
                savings_vs_zero_cache_pct=safe_pct(savings, zero_cost),
            )
        )

    # ------------------------------------------------------------- marginal
    # USD per +10pp of hit rate, as a 10pp-wide finite difference clipped to the
    # domain.  cost(h) is affine in h, so this is the exact local slope — no
    # approximation error from the window width.
    window = _MARGINAL_WINDOW
    lo = min(max(observed_hit_rate - window / 2.0, 0.0), 1.0 - window)
    hi = lo + window
    marginal = total_at(hi) - total_at(lo)  # already per 10pp: hi - lo == 0.10

    savings_at_observed = zero_cost - observed_cost
    return CacheSensitivity(
        curve=curve,
        observed_hit_rate=observed_hit_rate,
        observed_cost_usd=observed_cost,
        zero_cache_cost_usd=zero_cost,
        full_cache_cost_usd=full_cost,
        savings_at_observed_usd=savings_at_observed,
        savings_at_observed_pct=safe_pct(savings_at_observed, zero_cost),
        marginal_usd_per_10pp=marginal,
        explanation=_explain(
            observed_hit_rate=observed_hit_rate,
            observed_cost=observed_cost,
            zero_cost=zero_cost,
            full_cost=full_cost,
            savings_at_observed=savings_at_observed,
            marginal=marginal,
            cells=cells,
            groups=len(groups),
            excluded=excluded,
        ),
    )


def _explain(
    *,
    observed_hit_rate: float,
    observed_cost: float,
    zero_cost: float,
    full_cost: float,
    savings_at_observed: float,
    marginal: float,
    cells: int,
    groups: int,
    excluded: int,
) -> str:
    """Dollar-and-percent statement of what caching did and what it could do."""
    if zero_cost <= 0.0 and observed_cost <= 0.0:
        return (
            f"All {groups} group(s) price at $0 (free or unpriced models), so "
            "caching cannot change the bill: the projected curve is flat at zero."
        )

    parts = [
        f"Caching already saved ${savings_at_observed:.6f} "
        f"({safe_pct(savings_at_observed, zero_cost):.2f}% of the "
        f"${zero_cost:.6f} no-cache bill) at the observed token-weighted hit rate "
        f"of {observed_hit_rate * 100:.1f}% across {cells} cell(s)."
    ]
    if observed_hit_rate < 0.999:
        potential = observed_cost - full_cost
        parts.append(
            f"Pushing every prompt to a 100% hit rate would take the bill to "
            f"${full_cost:.6f}, a further ${potential:.6f} "
            f"({safe_pct(potential, observed_cost):.2f}%) below the observed "
            f"${observed_cost:.6f}."
        )
    parts.append(
        f"Marginal value: ${abs(marginal):.6f} "
        f"{'saved' if marginal < 0 else 'added'} per +10 percentage points of "
        "cache hit rate."
    )
    if excluded:
        parts.append(
            f"{excluded} group(s) were held at their recorded cost because the "
            "price book has no entry for that model, so their cache prices are "
            "unknown."
        )
    return " ".join(parts)


__all__ = ["MIN_STEPS", "cache_sensitivity"]
