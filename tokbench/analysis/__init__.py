"""Analytics layer: turn a run directory into a :class:`ReportPayload`.

Public surface
--------------
The pipeline is a straight line, and every stage is a pure function over lists so
it can be unit-tested and replayed without a filesystem::

    calls / results / manifest
        -> summarize()          headline numbers
        -> leaderboard()        quality per billable token, normalised
        -> pareto_frontier()    non-dominated cost/quality points + knee
        -> retry_analysis()     how much of the bill was pure retry waste
        -> cache_sensitivity()  cost as a function of cache hit rate
        -> strategy_deltas()    what each strategy buys for its extra tokens
        -> build_report()       the frozen ReportPayload contract

Only :mod:`tokbench.analysis.report` performs file I/O.  Importing this package
therefore has no side effects and does not require a run directory to exist.
"""

from __future__ import annotations

from .aggregate import (
    MIN_TOKEN_DENOMINATOR,
    GroupStats,
    category_breakdown,
    cost_of,
    group_stats,
    leaderboard,
    mean_quality,
    resolve_baseline_strategy,
    safe_div,
    safe_pct,
    stacked_bars,
    sum_usage,
    summarize,
    time_series,
)
from .cache import MIN_STEPS, cache_sensitivity
from .deltas import (
    QUALITY_STEP,
    WASTEFUL_MIN_COST_PREMIUM_PCT,
    WORTH_IT_MAX_COST_PREMIUM_PCT,
    WORTH_IT_MIN_QUALITY_GAIN,
    strategy_deltas,
)
from .pareto import COST_EPSILON, dominates, label_of, pareto_frontier
from .report import (
    SUPPORTED_FORMATS,
    build_and_write,
    build_report,
    dumps,
    load_report,
    write_report,
)
from .retry import retry_analysis

__all__ = [
    "COST_EPSILON",
    "GroupStats",
    "MIN_STEPS",
    "MIN_TOKEN_DENOMINATOR",
    "QUALITY_STEP",
    "SUPPORTED_FORMATS",
    "WASTEFUL_MIN_COST_PREMIUM_PCT",
    "WORTH_IT_MAX_COST_PREMIUM_PCT",
    "WORTH_IT_MIN_QUALITY_GAIN",
    "build_and_write",
    "build_report",
    "cache_sensitivity",
    "category_breakdown",
    "cost_of",
    "dominates",
    "dumps",
    "group_stats",
    "label_of",
    "leaderboard",
    "load_report",
    "mean_quality",
    "pareto_frontier",
    "resolve_baseline_strategy",
    "retry_analysis",
    "safe_div",
    "safe_pct",
    "stacked_bars",
    "strategy_deltas",
    "sum_usage",
    "summarize",
    "time_series",
    "write_report",
]
