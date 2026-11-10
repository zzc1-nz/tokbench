"""Tests for the analytics layer.

Every fixture here is a hand-built list of :class:`~tokbench.models.TaskResult` /
:class:`~tokbench.models.CallRecord` objects with **arithmetic done by hand in the
comments**, so a failure points at a specific formula rather than at "the numbers
changed".  No runner, no adapters, no filesystem except the one round-trip test
that exercises :func:`tokbench.analysis.report.build_report` against a real
:class:`~tokbench.store.RunStore`.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokbench.analysis import (
    build_report,
    cache_sensitivity,
    category_breakdown,
    dumps,
    leaderboard,
    load_report,
    pareto_frontier,
    retry_analysis,
    stacked_bars,
    strategy_deltas,
    summarize,
    time_series,
    write_report,
)
from tokbench.analysis.pareto import dominates
from tokbench.models import (
    CallRecord,
    CallStatus,
    GraderKind,
    QualityScore,
    RunManifest,
    TaskResult,
    Usage,
)
from tokbench.pricing import ModelPrice, PriceBook
from tokbench.store import RunStore

# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
#: A price book with round numbers so every expected cost below is mental math:
#:   input  = $1.00 / 1M      output = $2.00 / 1M
#:   cache_read = $0.10 / 1M  cache_write = $0.00 / 1M
#: 1000 uncached input + 1000 output  =>  0.001 + 0.002 = $0.003
#: ``A``-``D`` share cheap's rates so the Pareto fixture stays hand-computable.
_TEST_PRICE = ModelPrice(input=1.00, output=2.00, cache_read=0.10, cache_write=0.0)
TEST_BOOK = PriceBook(
    name="test-book",
    effective_date="2026-01-01",
    models={
        "cheap": _TEST_PRICE,
        "pricey": ModelPrice(input=2.00, output=4.00, cache_read=0.20, cache_write=0.0),
        "free": ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        "A": _TEST_PRICE,
        "B": _TEST_PRICE,
        "C": _TEST_PRICE,
        "D": _TEST_PRICE,
    },
)

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def quality(score: float) -> QualityScore:
    return QualityScore(score=score, passed=score >= 0.5, grader=GraderKind.HEURISTIC)


def usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def result(
    model_id: str,
    strategy_id: str,
    *,
    task_id: str = "t1",
    category: str = "codegen",
    score: float | None = 1.0,
    usage_total: Usage | None = None,
    usage_first: Usage | None = None,
    attempts: int = 1,
    status: CallStatus = CallStatus.OK,
    cost_usd: float = 0.0,
    latency_ms: float = 100.0,
    repeat: int = 0,
) -> TaskResult:
    """Build one cell; ``usage_first`` defaults to the whole usage (no retries)."""
    total = usage_total if usage_total is not None else usage(1000, 500)
    return TaskResult(
        result_id=f"{model_id}|{strategy_id}|{task_id}|r{repeat}",
        experiment="test-experiment",
        model_id=model_id,
        strategy_id=strategy_id,
        task_id=task_id,
        category=category,
        repeat=repeat,
        attempts=attempts,
        final_status=status,
        usage_total=total,
        usage_first_attempt=usage_first if usage_first is not None else total,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        quality=None if score is None else quality(score),
    )


def call(
    model_id: str,
    strategy_id: str,
    *,
    record_id: str = "c1",
    attempt: int = 1,
    status: CallStatus = CallStatus.OK,
    error_kind: str | None = None,
    usage_: Usage | None = None,
    started_offset_s: float = 0.0,
    cost_usd: float = 0.0,
) -> CallRecord:
    started = BASE_TIME + timedelta(seconds=started_offset_s)
    return CallRecord(
        record_id=record_id,
        experiment="test-experiment",
        model_id=model_id,
        provider="mock",
        strategy_id=strategy_id,
        task_id="t1",
        category="codegen",
        attempt=attempt,
        is_final_attempt=status is CallStatus.OK,
        status=status,
        started_at=started,
        ended_at=started + timedelta(milliseconds=250),
        latency_ms=250.0,
        usage=usage_ if usage_ is not None else usage(1000, 500),
        cost_usd=cost_usd,
        error_kind=error_kind,
    )


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_counts_and_totals() -> None:
    """2 models x 2 strategies x 2 tasks = 8 cells; verify every headline field."""
    results = [
        result("cheap", "baseline", task_id="t1"),
        result("cheap", "baseline", task_id="t2"),
        result("cheap", "cot", task_id="t1"),
        result("cheap", "cot", task_id="t2"),
        result("pricey", "baseline", task_id="t1"),
        result("pricey", "baseline", task_id="t2"),
        result("pricey", "cot", task_id="t1"),
        result("pricey", "cot", task_id="t2"),
    ]
    calls = [call("cheap", "baseline", record_id=f"c{i}") for i in range(8)]
    summary = summarize(calls, results, None, price_book=TEST_BOOK)

    assert summary.models == 2
    assert summary.strategies == 2
    assert summary.tasks == 2
    assert summary.repeats == 1
    assert summary.planned_cells == 2 * 2 * 2 * 1
    assert summary.completed_cells == 8
    assert summary.total_calls == 8
    assert summary.failed_calls == 0
    assert summary.success_rate == 1.0

    # Each cell: input 1000, output 500 -> billable 1500; 8 cells -> 12000.
    assert summary.total_billable_tokens == 8 * 1500
    assert summary.total_usage.total_tokens == 8 * 1500
    # cheap cells cost 0.001+0.001=0.002 each, pricey 0.002+0.002=0.004 each.
    # 4 cheap + 4 pricey = 0.008 + 0.016 = 0.024
    assert summary.total_cost_usd == pytest.approx(0.024)
    # 0.024 USD / 12000 billable tokens * 1000 = 0.002 USD per 1k billable.
    assert summary.cost_per_1k_billable_tokens == pytest.approx(0.002)
    assert summary.cache_hit_rate == 0.0


def test_summarize_empty_is_safe() -> None:
    summary = summarize([], [], None)
    assert summary.completed_cells == 0
    assert summary.total_cost_usd == 0.0
    assert summary.success_rate == 0.0
    assert summary.cost_per_1k_billable_tokens == 0.0
    assert summary.cache_hit_rate == 0.0
    assert summary.total_usage.total_tokens == 0


def test_summarize_single_result_is_safe() -> None:
    summary = summarize([], [result("cheap", "baseline")], None, price_book=TEST_BOOK)
    assert summary.completed_cells == 1
    assert summary.models == 1
    assert summary.planned_cells == 1
    assert summary.success_rate == 1.0
    assert summary.total_cost_usd == pytest.approx(0.002)


def test_summarize_uses_manifest_for_plan_and_timing() -> None:
    manifest = RunManifest(
        run_id="run-1",
        experiment_name="exp",
        planned_calls=100,
        completed_calls=40,
        failed_calls=3,
        wall_clock_s=12.5,
    )
    summary = summarize([], [], manifest)
    assert summary.planned_cells == 100
    assert summary.total_calls == 40
    assert summary.failed_calls == 3
    assert summary.wall_clock_s == 12.5
    # No calls and no results -> manifest success_rate fallback: 1 - 3/100.
    assert summary.success_rate == pytest.approx(0.97)


# --------------------------------------------------------------------------- #
# leaderboard
# --------------------------------------------------------------------------- #
def test_leaderboard_efficiency_formula_is_hand_computable() -> None:
    """Two groups, quality-per-billable-token worked out by hand.

    cheap / baseline : quality 1.0, billable 1000/task -> raw = 1.0/1000 = 0.001
    pricey / baseline: quality 0.5, billable 2000/task -> raw = 0.5/2000 = 0.00025
    best raw = 0.001, so scores are 1.0 and 0.25; rank 1 is cheap.
    """
    results = [
        result("cheap", "baseline", usage_total=usage(800, 200), score=1.0),
        result("pricey", "baseline", usage_total=usage(1600, 400), score=0.5),
    ]
    rows = leaderboard(results, TEST_BOOK)

    assert [r.model_id for r in rows] == ["cheap", "pricey"]
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].billable_tokens_per_task == pytest.approx(1000.0)
    assert rows[1].billable_tokens_per_task == pytest.approx(2000.0)
    assert rows[0].efficiency_score == pytest.approx(1.0)
    assert rows[1].efficiency_score == pytest.approx(0.25)
    # tokens_per_quality_point = billable/task / quality
    assert rows[0].tokens_per_quality_point == pytest.approx(1000.0)
    assert rows[1].tokens_per_quality_point == pytest.approx(2000.0 / 0.5)


def test_leaderboard_zero_quality_gives_none_tokens_per_point() -> None:
    rows = leaderboard([result("cheap", "baseline", score=0.0)], TEST_BOOK)
    assert rows[0].efficiency_score == 0.0
    assert rows[0].tokens_per_quality_point is None


def test_leaderboard_ties_break_by_lower_cost() -> None:
    """Identical quality and tokens -> identical score -> cheaper cost wins."""
    cheap = result("cheap", "baseline", usage_total=usage(1000, 0), score=0.8)
    pricey = result("pricey", "baseline", usage_total=usage(1000, 0), score=0.8)
    rows = leaderboard([pricey, cheap], TEST_BOOK)
    assert [r.model_id for r in rows] == ["cheap", "pricey"]
    assert rows[0].efficiency_score == pytest.approx(rows[1].efficiency_score)


def test_leaderboard_min_cells_filters_thin_groups() -> None:
    results = [
        result("cheap", "baseline", task_id="t1"),
        result("cheap", "baseline", task_id="t2"),
        result("pricey", "cot", task_id="t1"),
    ]
    assert len(leaderboard(results, TEST_BOOK)) == 2
    kept = leaderboard(results, TEST_BOOK, min_cells=2)
    assert len(kept) == 1
    assert kept[0].model_id == "cheap"


def test_leaderboard_empty_and_zero_token_are_safe() -> None:
    assert leaderboard([], TEST_BOOK) == []
    # Zero-token cell: the efficiency denominator is floored, no ZeroDivisionError.
    rows = leaderboard([result("cheap", "baseline", usage_total=usage(0, 0), score=1.0)], TEST_BOOK)
    assert rows[0].billable_tokens_per_task == 0.0
    assert rows[0].efficiency_score == 1.0


def test_leaderboard_is_deterministic_regardless_of_input_order() -> None:
    forward = [
        result("a", "s1", task_id="t1"),
        result("b", "s2", task_id="t1"),
        result("a", "s2", task_id="t1"),
    ]
    backward = list(reversed(forward))
    assert leaderboard(forward, TEST_BOOK) == leaderboard(backward, TEST_BOOK)


# --------------------------------------------------------------------------- #
# Pareto frontier 鈥?the hand-computable 4-point example
# --------------------------------------------------------------------------- #
def _pareto_results() -> list[TaskResult]:
    """Four points whose frontier membership is decidable by eye.

    Rates are TEST_BOOK's: input $1/1M, output $2/1M.

    Model  usage (in, out)   cost/task   quality   verdict
    A      (1000, 1000)      $0.003      0.90      on frontier
    B      (0,    1000)      $0.002      0.60      on frontier (cheapest)
    C      (1000,  500)      $0.002      0.50      DOMINATED by B
    D      (1000, 1500)      $0.004      0.95      on frontier (best)

    B (0.002, 0.60) beats C (0.002, 0.50) on quality at equal cost => C is off.
    A, B, D form the staircase: cheap-low -> mid-high -> expensive-best.
    """
    return [
        result("A", "s", usage_total=usage(1000, 1000), score=0.90),
        result("B", "s", usage_total=usage(0, 1000), score=0.60),
        result("C", "s", usage_total=usage(1000, 500), score=0.50),
        result("D", "s", usage_total=usage(1000, 1500), score=0.95),
    ]


def test_pareto_cost_per_task_is_hand_computable() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    by_label = {p.label: p for p in frontier.points}
    assert by_label["A / s"].cost_per_task_usd == pytest.approx(0.003)
    assert by_label["B / s"].cost_per_task_usd == pytest.approx(0.002)
    assert by_label["C / s"].cost_per_task_usd == pytest.approx(0.002)
    assert by_label["D / s"].cost_per_task_usd == pytest.approx(0.004)
    assert by_label["A / s"].quality == pytest.approx(0.90)


def test_pareto_frontier_membership() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    # Sorted by cost ascending: B (0.002) and C (0.002) tie on cost, then A, D.
    assert "C / s" not in frontier.frontier_labels
    assert set(frontier.frontier_labels) == {"A / s", "B / s", "D / s"}
    assert frontier.frontier_labels[-1] == "D / s"

    by_label = {p.label: p for p in frontier.points}
    assert by_label["C / s"].on_frontier is False
    assert by_label["C / s"].dominated_by == ["B / s"]
    for label in ("A / s", "B / s", "D / s"):
        assert by_label[label].on_frontier is True
        assert by_label[label].dominated_by == []


def test_pareto_frontier_is_sorted_by_cost() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    costs = [p.cost_per_task_usd for p in frontier.points]
    assert costs == sorted(costs)
    frontier_costs = [
        p.cost_per_task_usd for p in frontier.points if p.label in frontier.frontier_labels
    ]
    assert frontier_costs == sorted(frontier_costs)


def test_pareto_dominance_relation() -> None:
    """Strict dominance: equal on both axes is a tie, not dominance."""
    frontier = pareto_frontier(
        [
            result("A", "s", usage_total=usage(0, 1000), score=0.5),
            result("B", "s", usage_total=usage(0, 1000), score=0.5),
            result("C", "s", usage_total=usage(0, 2000), score=0.5),
        ],
        TEST_BOOK,
    )
    by_label = {p.label: p for p in frontier.points}
    # A and B are identical -> both stay on the frontier.
    assert by_label["A / s"].on_frontier is True
    assert by_label["B / s"].on_frontier is True
    # C is strictly more expensive at equal quality -> dominated by both.
    assert by_label["C / s"].dominated_by == ["A / s", "B / s"]


def test_pareto_does_not_report_dominance_on_equal_cost_equal_quality() -> None:
    cheap = result("cheap", "s", usage_total=usage(0, 1000), score=0.7)
    pricey = result("pricey", "s", usage_total=usage(0, 1000), score=0.7)
    stats = {}
    from tokbench.analysis.aggregate import group_stats

    for group in group_stats([cheap, pricey], TEST_BOOK):
        stats[group.model_id] = group
    # 'cheap' and 'pricey' happen to price differently here, so compare a group
    # with itself: a point never strictly dominates its own duplicate.
    assert dominates(stats["cheap"], stats["cheap"]) is False


def test_pareto_value_score_and_headline_labels() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    # value = quality / cost; B: 0.60/0.002 = 300, A: 0.90/0.003 = 300,
    # D: 0.95/0.004 = 237.5 -> A and B tie at 300, normalised to 1.0.
    by_label = {p.label: p for p in frontier.points}
    assert by_label["A / s"].value_score == pytest.approx(1.0)
    assert by_label["B / s"].value_score == pytest.approx(1.0)
    assert by_label["D / s"].value_score == pytest.approx(237.5 / 300.0)
    assert frontier.cheapest_label == "B / s"
    assert frontier.best_quality_label == "D / s"
    assert frontier.best_value_label in {"A / s", "B / s"}


def test_pareto_gain_per_usd_and_knee() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    # Frontier: B(0.002, 0.60) -> A(0.003, 0.90) -> D(0.004, 0.95).
    # gain = (0.95 - 0.60) / (0.004 - 0.002) = 0.35 / 0.002 = 175 points/USD.
    assert frontier.frontier_quality_gain_per_usd == pytest.approx(175.0)
    # Knee: normalise -> B(0,0), A(0.5,0.857), D(1,1). Line B->D is y=x.
    # distance(A) = |0.857 - 0.5| / sqrt(2) = 0.2525; A is the knee.
    assert "A / s" in frontier.explanation
    assert "Knee" in frontier.explanation


def test_pareto_knee_label_is_published_and_matches_explanation() -> None:
    """The knee must be a field, not prose the renderers have to re-derive.

    ``ParetoFrontier.knee_label`` is the authoritative value: the HTML report, the
    Streamlit panel and the README figure all read it.  If a renderer instead
    recomputed the elbow, a change to the heuristic here would silently make the
    three artifacts disagree.  This test pins the field to the explanation so that
    contract cannot drift.
    """
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    assert frontier.knee_label == "A / s"
    assert frontier.knee_label in frontier.explanation

    # Degenerate frontiers (fewer than 3 points) have no interior elbow.
    single = pareto_frontier(_pareto_results()[:1], TEST_BOOK)
    assert single.knee_label is None


def test_pareto_knee_label_survives_json_round_trip() -> None:
    """The published knee must serialise — reports are read back from disk."""
    from tokbench.report_schema import ParetoFrontier

    frontier = pareto_frontier(_pareto_results(), TEST_BOOK)
    restored = ParetoFrontier.model_validate_json(frontier.model_dump_json())
    assert restored.knee_label == frontier.knee_label


def test_pareto_baseline_label_and_explanation() -> None:
    frontier = pareto_frontier(_pareto_results(), TEST_BOOK, baseline_strategy="s")
    assert frontier.baseline_label == "D / s"  # strongest representative of 's'
    assert frontier.explanation
    assert "Baseline" in frontier.explanation


def test_pareto_empty_and_single_point_are_safe() -> None:
    empty = pareto_frontier([], TEST_BOOK)
    assert empty.points == []
    assert empty.frontier_labels == []
    assert empty.baseline_label is None
    assert "No results" in empty.explanation

    single = pareto_frontier([result("A", "s")], TEST_BOOK)
    assert single.frontier_labels == ["A / s"]
    assert single.points[0].on_frontier is True
    assert single.cheapest_label == "A / s"
    assert single.best_quality_label == "A / s"
    assert "Only one candidate" in single.explanation


def test_pareto_free_model_does_not_divide_by_zero() -> None:
    frontier = pareto_frontier(
        [result("free", "s", score=0.9), result("cheap", "s", score=0.5)], TEST_BOOK
    )
    free_point = next(p for p in frontier.points if p.model_id == "free")
    assert free_point.cost_per_task_usd == 0.0
    # Free + good is the best possible value -> normalised to 1.0.
    assert free_point.value_score == pytest.approx(1.0)
    assert frontier.cheapest_label == "free / s"


# --------------------------------------------------------------------------- #
# Retry analysis 鈥?exact arithmetic
# --------------------------------------------------------------------------- #
def test_retry_cost_share_arithmetic() -> None:
    """One retried cell, one clean cell, hand-computed shares.

    cheap/baseline:
      clean cell   : 1000 in, 500 out             -> total usage 1000/500
      retried cell : attempt1 1000 in / 500 out
                     attempt2 1000 in / 500 out   -> total 2000/1000
      retry overhead = total - first = 1000 in / 500 out
      total cost  = price(3000 in, 1500 out) = 0.003 + 0.003 = $0.006
      retry cost  = price(1000 in,  500 out) = 0.001 + 0.001 = $0.002
      retry_cost_share = 0.002 / 0.006 = 1/3
      billable tokens  = 4500 total, 1500 retry -> retry_token_share = 1/3
    """
    first = usage(1000, 500)
    retried_total = usage(2000, 1000)
    results = [
        result("cheap", "baseline", task_id="t1", usage_total=first),
        result(
            "cheap",
            "baseline",
            task_id="t2",
            usage_total=retried_total,
            usage_first=first,
            attempts=2,
        ),
    ]
    analysis = retry_analysis([], results, TEST_BOOK)

    assert len(analysis.rows) == 1
    row = analysis.rows[0]
    assert row.cells == 2
    assert row.retried_cells == 1
    assert row.retry_rate == pytest.approx(0.5)
    assert row.extra_attempts == 1
    assert row.retry_usage.input_tokens == 1000
    assert row.retry_usage.output_tokens == 500
    assert row.retry_cost_usd == pytest.approx(0.002)
    assert row.total_cost_usd == pytest.approx(0.006)
    assert row.retry_cost_share == pytest.approx(1.0 / 3.0)
    assert row.retry_token_share == pytest.approx(1.0 / 3.0)

    assert analysis.total_retry_cost_usd == pytest.approx(0.002)
    assert analysis.total_cost_usd == pytest.approx(0.006)
    assert analysis.overall_retry_cost_share == pytest.approx(1.0 / 3.0)
    assert analysis.wasted_pct_of_budget == pytest.approx(100.0 / 3.0)
    assert analysis.total_retry_tokens == 1500
    assert analysis.total_tokens == 4500
    assert analysis.worst_offender == "cheap / baseline"
    assert "33.33%" in analysis.explanation


def test_retry_error_breakdown_and_worst_offender() -> None:
    """Two groups; the one with more retry spend must be the worst offender."""
    first = usage(1000, 500)
    results = [
        # group 1: 1 retried cell -> retry cost $0.002
        result(
            "cheap",
            "s1",
            task_id="t1",
            usage_total=usage(2000, 1000),
            usage_first=first,
            attempts=2,
        ),
        result("cheap", "s1", task_id="t2", usage_total=first),
        # group 2: 1 retried cell with 2x the overhead -> $0.004
        result(
            "pricey",
            "s2",
            task_id="t1",
            usage_total=usage(2000, 1000),
            usage_first=first,
            attempts=2,
        ),
    ]
    calls = [
        call(
            "cheap",
            "s1",
            record_id="c1",
            attempt=1,
            status=CallStatus.ERROR,
            error_kind="rate_limit",
        ),
        call("cheap", "s1", record_id="c2", attempt=2),
        call(
            "pricey",
            "s2",
            record_id="c3",
            attempt=1,
            status=CallStatus.TIMEOUT,
            error_kind="timeout",
        ),
        call(
            "pricey",
            "s2",
            record_id="c4",
            attempt=1,
            status=CallStatus.ERROR,
            error_kind="rate_limit",
        ),
        call("pricey", "s2", record_id="c5", attempt=2),
    ]
    analysis = retry_analysis(calls, results, TEST_BOOK)

    rows = {f"{r.model_id} / {r.strategy_id}": r for r in analysis.rows}
    assert rows["cheap / s1"].dominant_error_kind == "rate_limit"
    assert rows["cheap / s1"].error_breakdown == {"rate_limit": 1}
    assert rows["pricey / s2"].error_breakdown == {"rate_limit": 1, "timeout": 1}
    # Tie on count -> alphabetical winner.
    assert rows["pricey / s2"].dominant_error_kind == "rate_limit"

    # pricey retry: 1000 in @2.00 + 500 out @4.00 = 0.002 + 0.002 = $0.004
    assert rows["pricey / s2"].retry_cost_usd == pytest.approx(0.004)
    assert analysis.worst_offender == "pricey / s2"
    assert analysis.error_breakdown == {"rate_limit": 2, "timeout": 1}


def test_retry_no_retries_is_a_real_zero() -> None:
    analysis = retry_analysis([call("cheap", "s")], [result("cheap", "s")], TEST_BOOK)
    assert analysis.overall_retry_cost_share == 0.0
    assert analysis.total_retry_cost_usd == 0.0
    assert analysis.worst_offender is None
    assert "No retries observed" in analysis.explanation


def test_retry_empty_is_safe() -> None:
    analysis = retry_analysis([], [], TEST_BOOK)
    assert analysis.rows == []
    assert analysis.overall_retry_cost_share == 0.0
    assert analysis.worst_offender is None
    assert analysis.explanation


# --------------------------------------------------------------------------- #
# Cache sensitivity
# --------------------------------------------------------------------------- #
def _cache_results() -> list[TaskResult]:
    """One group: 1000 input tokens of which 0 are cached, 500 output.

    cheap: uncached = 1000, output = 500.
      h=0.0 -> 1000*1.00 + 500*2.00 = 0.001 + 0.001 = $0.002
      h=0.5 ->  500*1.00 + 500*0.10 + 500*2.00 = 0.0005+0.00005+0.001 = $0.00155
      h=1.0 ->    0*1.00 + 1000*0.10 + 500*2.00 = 0.0001 + 0.001 = $0.0011
    """
    return [result("cheap", "baseline", usage_total=usage(1000, 500), score=1.0)]


def test_cache_curve_endpoints_are_hand_computable() -> None:
    sensitivity = cache_sensitivity(_cache_results(), TEST_BOOK, steps=21)
    assert len(sensitivity.curve) == 21
    assert sensitivity.curve[0].cache_hit_rate == 0.0
    assert sensitivity.curve[-1].cache_hit_rate == 1.0
    assert sensitivity.zero_cache_cost_usd == pytest.approx(0.002)
    assert sensitivity.full_cache_cost_usd == pytest.approx(0.0011)
    assert sensitivity.observed_hit_rate == 0.0
    assert sensitivity.observed_cost_usd == pytest.approx(0.002)
    # Observed hit rate is 0, so caching saved nothing yet.
    assert sensitivity.savings_at_observed_usd == pytest.approx(0.0)
    # Marginal: moving 10pp of the 1000 eligible tokens from input ($1.00/1M) to
    # cache_read ($0.10/1M) saves 100 tokens * 0.90 / 1M = $0.00009.
    assert sensitivity.marginal_usd_per_10pp == pytest.approx(-0.00009, rel=1e-6)


def test_cache_curve_is_monotonic_decreasing() -> None:
    sensitivity = cache_sensitivity(_cache_results(), TEST_BOOK, steps=11)
    costs = [p.projected_cost_usd for p in sensitivity.curve]
    assert all(later <= earlier + 1e-15 for earlier, later in zip(costs, costs[1:], strict=False))
    rates = [p.cache_hit_rate for p in sensitivity.curve]
    assert rates == sorted(rates)
    # Savings vs zero-cache must rise monotonically as the hit rate climbs.
    savings = [p.savings_vs_zero_cache_usd for p in sensitivity.curve]
    assert all(
        later >= earlier - 1e-15 for earlier, later in zip(savings, savings[1:], strict=False)
    )


def test_cache_observed_hit_rate_anchors_the_curve() -> None:
    """A run that already caches must reproduce its own cost at h=observed."""
    results = [
        result("cheap", "baseline", usage_total=usage(1000, 500, cache_read_tokens=400)),
    ]
    sensitivity = cache_sensitivity(results, TEST_BOOK, steps=21)
    # 600 uncached * 1.00 + 400 * 0.10 + 500 * 2.00
    # = 0.0006 + 0.00004 + 0.001 = $0.00164
    assert sensitivity.observed_hit_rate == pytest.approx(0.4)
    assert sensitivity.observed_cost_usd == pytest.approx(0.00164)
    assert sensitivity.zero_cache_cost_usd == pytest.approx(0.002)
    assert sensitivity.savings_at_observed_usd == pytest.approx(0.002 - 0.00164)
    assert sensitivity.savings_at_observed_pct == pytest.approx(18.0)
    assert sensitivity.marginal_usd_per_10pp < 0.0
    assert "saved" in sensitivity.explanation


def test_cache_empty_and_zero_cost_are_safe() -> None:
    empty = cache_sensitivity([], TEST_BOOK)
    assert empty.curve == []
    assert empty.observed_cost_usd == 0.0
    assert empty.explanation

    free = cache_sensitivity([result("free", "s")], TEST_BOOK)
    assert free.zero_cache_cost_usd == 0.0
    assert free.observed_cost_usd == 0.0
    assert free.marginal_usd_per_10pp == 0.0
    assert free.explanation


def test_cache_steps_are_clamped() -> None:
    sensitivity = cache_sensitivity(_cache_results(), TEST_BOOK, steps=1)
    assert len(sensitivity.curve) == 2


# --------------------------------------------------------------------------- #
# Strategy deltas
# --------------------------------------------------------------------------- #
def test_strategy_deltas_hand_computable() -> None:
    """baseline 1000 billable/task, cot 2000 billable/task.

    token_delta_pct = (2000-1000)/1000 * 100 = +100%
    cost: baseline 0.002/task, cot 0.004/task -> +100%
    quality: 0.5 -> 0.7 => +0.2
    tokens_per_quality_delta = (2000-1000) / (0.2/0.01) = 1000/20 = 50
    """
    results = [
        result("cheap", "baseline", task_id="t1", usage_total=usage(800, 200), score=0.5),
        result("cheap", "cot", task_id="t1", usage_total=usage(1600, 400), score=0.7),
    ]
    rows = strategy_deltas(results, price_book=TEST_BOOK)
    assert len(rows) == 2  # one global, one per-model
    global_row = rows[0]
    assert global_row.model_id is None
    assert global_row.baseline_strategy == "baseline"
    assert global_row.strategy_id == "cot"
    assert global_row.baseline_billable_per_task == pytest.approx(1000.0)
    assert global_row.strategy_billable_per_task == pytest.approx(2000.0)
    assert global_row.token_delta_pct == pytest.approx(100.0)
    assert global_row.cost_delta_pct == pytest.approx(100.0)
    assert global_row.quality_delta == pytest.approx(0.2)
    assert global_row.tokens_per_quality_delta == pytest.approx(50.0)
    assert global_row.verdict == "worth it"

    per_model = rows[1]
    assert per_model.model_id == "cheap"
    assert per_model.token_delta_pct == pytest.approx(100.0)


def test_strategy_deltas_verdicts() -> None:
    """Thresholds from the module docstring, exercised at their boundaries."""
    # +60% tokens, +1% quality -> over the 50% premium, under the gain -> wasteful.
    wasteful = strategy_deltas(
        [
            result("cheap", "baseline", usage_total=usage(1000, 0), score=0.50),
            result("cheap", "bloat", usage_total=usage(1600, 0), score=0.51),
        ],
        price_book=TEST_BOOK,
    )[0]
    assert wasteful.verdict == "wasteful"
    assert wasteful.tokens_per_quality_delta is not None

    # +10% tokens, +5% quality -> within the premium, above the gain -> worth it.
    worth = strategy_deltas(
        [
            result("cheap", "baseline", usage_total=usage(1000, 0), score=0.50),
            result("cheap", "tuned", usage_total=usage(1100, 0), score=0.55),
        ],
        price_book=TEST_BOOK,
    )[0]
    assert worth.verdict == "worth it"

    # Same tokens, same quality -> a no-op, within noise.
    neutral = strategy_deltas(
        [
            result("cheap", "baseline", usage_total=usage(1000, 0), score=0.50),
            result("cheap", "same", usage_total=usage(1000, 0), score=0.50),
        ],
        price_book=TEST_BOOK,
    )[0]
    assert neutral.verdict == "neutral"
    assert neutral.token_delta_pct == pytest.approx(0.0)
    assert neutral.quality_delta == pytest.approx(0.0)


def test_strategy_deltas_negative_quality_delta_has_no_tokens_per_point() -> None:
    rows = strategy_deltas(
        [
            result("cheap", "baseline", usage_total=usage(1000, 0), score=0.8),
            result("cheap", "cot", usage_total=usage(2000, 0), score=0.6),
        ],
        price_book=TEST_BOOK,
    )
    row = rows[0]
    assert row.quality_delta == pytest.approx(-0.2)
    assert row.tokens_per_quality_delta is None


def test_strategy_deltas_baseline_autopick_and_explicit() -> None:
    results = [
        result("cheap", "lean", usage_total=usage(500, 0), score=0.5),
        result("cheap", "fat", usage_total=usage(1500, 0), score=0.6),
    ]
    auto = strategy_deltas(results, price_book=TEST_BOOK)
    # 'lean' has fewer billable tokens/task -> auto-picked as baseline.
    assert auto[0].baseline_strategy == "lean"
    assert auto[0].strategy_id == "fat"

    explicit = strategy_deltas(results, baseline="fat", price_book=TEST_BOOK)
    assert explicit[0].baseline_strategy == "fat"
    assert explicit[0].strategy_id == "lean"
    # Going from fat to lean is cheaper -> negative token delta.
    assert explicit[0].token_delta_pct == pytest.approx(-(1000 / 1500) * 100)


def test_strategy_deltas_empty_and_single_strategy_are_safe() -> None:
    assert strategy_deltas([]) == []
    assert strategy_deltas([result("cheap", "only")]) == []


# --------------------------------------------------------------------------- #
# Category breakdown / stacked bars / time series
# --------------------------------------------------------------------------- #
def test_category_breakdown_groups_and_sorts() -> None:
    results = [
        result("cheap", "s", task_id="t1", category="codegen", score=1.0),
        result("cheap", "s", task_id="t2", category="codegen", score=0.0),
        result("cheap", "s", task_id="t3", category="sql", score=1.0),
    ]
    rows = category_breakdown(results)
    assert [(r.category) for r in rows] == ["codegen", "sql"]
    codegen = rows[0]
    assert codegen.cells == 2
    assert codegen.mean_quality == pytest.approx(0.5)
    assert codegen.billable_tokens_per_task == pytest.approx(1500.0)
    assert rows[1].mean_quality == pytest.approx(1.0)
    assert category_breakdown([]) == []


def test_stacked_bars_token_decomposition() -> None:
    results = [
        result(
            "cheap",
            "s",
            usage_total=usage(1000, 500, cache_read_tokens=300, reasoning_tokens=50),
        )
    ]
    rows = stacked_bars(results)
    assert len(rows) == 1
    row = rows[0]
    assert row.uncached_input_tokens == 700  # 1000 - 300
    assert row.cache_read_tokens == 300
    assert row.output_tokens == 500
    assert row.reasoning_tokens == 50
    assert row.cells == 1
    assert stacked_bars([]) == []


def test_cost_sections_agree_on_one_price_book() -> None:
    """Leaderboard, categories and stacked bars must reconcile.

    Every cost section re-derives from the same price book, so the totals have to
    add up.  Divergent cost bases between charts are the classic way a benchmark
    report stops being trustworthy.
    """
    results = [
        result("cheap", "s", task_id="t1", category="codegen", usage_total=usage(1000, 500)),
        result("cheap", "s", task_id="t2", category="sql", usage_total=usage(2000, 500)),
        result("pricey", "s", task_id="t1", category="codegen", usage_total=usage(1000, 500)),
    ]
    board_total = sum(r.cost_usd for r in leaderboard(results, TEST_BOOK))
    category_total = sum(
        r.cost_per_task_usd * r.cells for r in category_breakdown(results, TEST_BOOK)
    )
    stacked_total = sum(r.cost_usd for r in stacked_bars(results, TEST_BOOK))

    assert board_total == pytest.approx(category_total)
    assert board_total == pytest.approx(stacked_total)
    # And the recorded cost (all zero here) would NOT have matched, proving the
    # price book is actually being used.
    assert board_total > 0.0


def test_time_series_is_chronological_and_cumulative() -> None:
    calls = [
        call("cheap", "s", record_id="c3", started_offset_s=3.0, usage_=usage(100, 0)),
        call("cheap", "s", record_id="c1", started_offset_s=1.0, usage_=usage(200, 0)),
        call("cheap", "s", record_id="c2", started_offset_s=2.0, usage_=usage(300, 0)),
    ]
    points = time_series(calls)
    assert [p.index for p in points] == [0, 1, 2]
    assert [p.t for p in points] == sorted(p.t for p in points)
    assert [p.tokens for p in points] == [200, 300, 100]
    assert [p.cumulative_tokens for p in points] == [200, 500, 600]
    assert [p.cumulative_billable_tokens for p in points] == [200, 500, 600]
    assert points[-1].cumulative_cost_usd == pytest.approx(sum(c.cost_usd for c in calls))
    assert time_series([]) == []


def test_time_series_breaks_timestamp_ties_by_record_id() -> None:
    calls = [
        call("cheap", "s", record_id="b", started_offset_s=1.0),
        call("cheap", "s", record_id="a", started_offset_s=1.0),
    ]
    points = time_series(calls)
    assert [p.index for p in points] == [0, 1]
    # Both share a timestamp; ordering falls back to record_id ascending.
    assert time_series(list(reversed(calls))) == points


def test_time_series_cumulative_cache_hit_rate() -> None:
    calls = [
        call(
            "cheap",
            "s",
            record_id="c1",
            started_offset_s=1.0,
            usage_=usage(1000, 0, cache_read_tokens=0),
        ),
        call(
            "cheap",
            "s",
            record_id="c2",
            started_offset_s=2.0,
            usage_=usage(1000, 0, cache_read_tokens=500),
        ),
    ]
    points = time_series(calls)
    assert points[0].cumulative_cache_hit_rate == pytest.approx(0.0)
    # Running ratio: 500 cache reads / 2000 total input = 0.25 (not the 0.5 mean).
    assert points[1].cumulative_cache_hit_rate == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Report assembly (the only file I/O in the package)
# --------------------------------------------------------------------------- #
#: Workspace-local scratch root for the report tests.  ``tmp_path`` points outside
#: the session workspace, which the file sandbox denies, so run directories are
#: created next to the tests and removed afterwards.
_SCRATCH = Path(__file__).resolve().parent / "_analysis_scratch"


@pytest.fixture()
def run_root():
    """A clean directory for :class:`RunStore` fixtures."""
    if _SCRATCH.exists():
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        yield _SCRATCH
    finally:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


def _seed_store(root) -> RunStore:
    store = RunStore.create(root, "20260101-120000-test")
    manifest = RunManifest(
        run_id=store.run_id,
        experiment_name="test-experiment",
        adapter_mode="mock",
        planned_calls=4,
        completed_calls=4,
        total_cost_usd=0.008,
        wall_clock_s=1.5,
    )
    store.write_manifest(manifest)
    for model_id, score in (("cheap", 0.9), ("pricey", 0.6)):
        for task_id in ("t1", "t2"):
            store.append_call(
                call(
                    model_id, "baseline", record_id=f"{model_id}-{task_id}", usage_=usage(1000, 500)
                )
            )
            store.append_result(
                result(
                    model_id, "baseline", task_id=task_id, score=score, usage_total=usage(1000, 500)
                )
            )
    return store


def test_build_report_fills_every_section(run_root) -> None:
    store = _seed_store(run_root)
    payload = build_report(store)

    assert payload.meta.run_id == store.run_id
    assert payload.meta.experiment == "test-experiment"
    assert payload.meta.currency == "USD"
    assert payload.meta.price_book  # default book name
    assert any("mock" in note for note in payload.meta.notes)

    assert payload.summary.completed_cells == 4
    assert payload.summary.models == 2
    assert len(payload.leaderboard) == 2
    assert payload.leaderboard[0].provider == "mock"
    assert len(payload.categories) == 2
    assert len(payload.pareto.points) == 2
    assert len(payload.retry.rows) == 2
    assert len(payload.cache.curve) == 21
    assert len(payload.series) == 4
    assert len(payload.stacked) == 2
    assert payload.manifest is not None
    assert payload.raw["cells"] == 4
    assert payload.raw["baseline_strategy"] == "baseline"


def test_build_report_empty_run_is_valid(run_root) -> None:
    store = RunStore.create(run_root, "empty-run")
    payload = build_report(store)
    assert payload.summary.completed_cells == 0
    assert payload.leaderboard == []
    assert payload.pareto.points == []
    assert payload.retry.rows == []
    assert payload.cache.curve == []
    assert payload.meta.notes  # explains the emptiness


def test_build_report_records_unpriced_models(run_root) -> None:
    store = RunStore.create(run_root, "unpriced-run")
    store.append_result(result("not-in-book", "s", usage_total=usage(10, 5)))
    payload = build_report(store)
    assert payload.raw["unpriced_models"] == ["not-in-book"]
    assert any("not-in-book" in note for note in payload.meta.notes)


def test_build_report_records_ignored_baseline(run_root) -> None:
    store = _seed_store(run_root)
    payload = build_report(store, baseline_strategy="does-not-exist")
    assert payload.raw["baseline_strategy"] == "baseline"
    assert payload.raw["baseline_source"] == "auto"
    assert any("does-not-exist" in note for note in payload.meta.notes)


def test_dumps_is_compact_and_sorted(run_root) -> None:
    store = RunStore.create(run_root, "dumps-run")
    store.append_result(result("cheap", "baseline"))
    payload = build_report(store)
    text = dumps(payload)
    # Compact: no newlines and no space after the separators.
    assert "\n" not in text
    assert '": ' not in text
    assert text.startswith('{"cache":')  # sort_keys=True
    assert len(text) > 100


def test_write_and_load_report_round_trip(run_root) -> None:
    store = _seed_store(run_root)
    payload = build_report(store)
    paths = write_report(payload, store, formats=("json",))
    assert len(paths) == 1
    assert paths[0].name == "report.json"

    loaded = load_report(paths[0])
    assert loaded.meta.run_id == payload.meta.run_id
    assert loaded.summary.completed_cells == payload.summary.completed_cells
    assert loaded.leaderboard[0].efficiency_score == pytest.approx(
        payload.leaderboard[0].efficiency_score
    )


def test_write_report_markdown(run_root) -> None:
    store = _seed_store(run_root)
    payload = build_report(store)
    paths = write_report(payload, store, formats=("json", "md"))
    assert [p.suffix for p in paths] == [".json", ".md"]
    markdown = paths[1].read_text(encoding="utf-8")
    assert "# tokbench report" in markdown
    assert "Token-efficiency leaderboard" in markdown
    assert "Retry overhead" in markdown


def test_write_report_rejects_unknown_format(run_root) -> None:
    from tokbench.errors import ReportError

    store = _seed_store(run_root)
    payload = build_report(store)
    with pytest.raises(ReportError, match="unknown report format"):
        write_report(payload, store, formats=("pdf",))
    with pytest.raises(ReportError, match="no report formats"):
        write_report(payload, store, formats=())


# --------------------------------------------------------------------------- #
# Determinism / robustness sweep
# --------------------------------------------------------------------------- #
def test_all_sections_are_deterministic(run_root) -> None:
    """Same run directory -> byte-identical metrics.

    ``meta.generated_at`` is excluded on purpose: it is the one field that *must*
    change between builds (it records when the report was produced), so it is
    stamped to a fixed value before comparing.
    """
    store = _seed_store(run_root)
    first = build_report(store)
    second = build_report(store)
    first.meta.generated_at = BASE_TIME
    second.meta.generated_at = BASE_TIME
    assert dumps(first) == dumps(second)


def test_single_cell_run_never_raises(run_root) -> None:
    store = RunStore.create(run_root, "single-cell")
    store.append_call(call("cheap", "baseline", record_id="only"))
    store.append_result(result("cheap", "baseline"))
    payload = build_report(store)
    assert payload.summary.completed_cells == 1
    assert len(payload.leaderboard) == 1
    assert len(payload.pareto.frontier_labels) == 1
    assert payload.retry.overall_retry_cost_share == 0.0
    assert len(payload.cache.curve) == 21


def test_zero_token_and_zero_cost_run_is_safe() -> None:
    results = [result("free", "s", usage_total=usage(0, 0), score=1.0)]
    assert leaderboard(results, TEST_BOOK)[0].efficiency_score == 1.0
    frontier = pareto_frontier(results, TEST_BOOK)
    assert frontier.points[0].cost_per_task_usd == 0.0
    assert pareto_frontier([], TEST_BOOK).explanation
    assert cache_sensitivity(results, TEST_BOOK).observed_cost_usd == 0.0
    assert retry_analysis([], results, TEST_BOOK).overall_retry_cost_share == 0.0


# --------------------------------------------------------------------------- #
# Self-check: ~200 synthetic cells, plausible-number assertions
# --------------------------------------------------------------------------- #
#: Model lineup for the self-check fixture, mirroring the flagship config:
#: (model_id, provider, quality floor, quality span, is_reasoning).
_SELFCHECK_MODELS = (
    ("deepseek-chat", "deepseek", 0.55, 0.40, False),
    ("deepseek-reasoner", "deepseek", 0.70, 0.28, True),
    ("gpt-4o-mini", "openai", 0.50, 0.45, False),
    ("gpt-4.1-nano", "openai", 0.42, 0.48, False),
    ("mock-small", "mock", 0.35, 0.50, False),
)

#: strategy -> (input multiplier, output multiplier, retry probability).
_SELFCHECK_STRATEGIES = (
    ("baseline", 1.00, 1.00, 0.02),
    ("cot", 1.05, 1.85, 0.03),
    ("fewshot-compress", 1.55, 0.90, 0.05),
)

_SELFCHECK_CATEGORIES = (
    "codegen",
    "debug",
    "refactor",
    "test-gen",
    "sql",
    "explain",
    "tool-use",
    "reasoning",
)

#: Categories that carry a cacheable prefix (repo context, tool transcripts).
_CACHEABLE = frozenset({"refactor", "tool-use", "explain"})


def _unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from a stable string key."""
    import hashlib

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _selfcheck_fixture() -> tuple[list[CallRecord], list[TaskResult], RunManifest]:
    """5 models x 3 strategies x 13 tasks = 195 cells (~8% retried).

    Token counts and qualities are derived from stable hashes rather than a random
    generator so the numbers are reproducible on every machine and the printed
    self-check output is comparable between runs.
    """
    from tokbench.pricing import DEFAULT_PRICE_BOOK

    calls: list[CallRecord] = []
    results: list[TaskResult] = []
    task_ids = [f"cb-{i:03d}" for i in range(1, 14)]

    for model_id, _provider, q_floor, q_span, reasoning_model in _SELFCHECK_MODELS:
        for strategy_id, in_mult, out_mult, retry_prob in _SELFCHECK_STRATEGIES:
            for task_index, task_id in enumerate(task_ids):
                category = _SELFCHECK_CATEGORIES[task_index % len(_SELFCHECK_CATEGORIES)]
                noise = _unit(model_id, strategy_id, task_id)

                input_tokens = int(1800 * in_mult * (0.85 + 0.30 * noise))
                cache_share = 0.55 if category in _CACHEABLE else 0.15
                cache_read = int(input_tokens * cache_share * (0.7 + 0.6 * noise))
                output_tokens = int(420 * out_mult * (0.7 + 0.6 * noise))
                reasoning_tokens = int(output_tokens * 0.45) if reasoning_model else 0

                first_usage = usage(
                    input_tokens, output_tokens, cache_read, reasoning_tokens=reasoning_tokens
                )

                strategy_bonus = {"baseline": 0.0, "cot": 0.09, "fewshot-compress": 0.05}[
                    strategy_id
                ]
                difficulty = 0.15 * ((task_index % 5) / 4.0)
                score = min(
                    max(
                        q_floor + q_span * (0.35 + 0.65 * noise) + strategy_bonus - difficulty,
                        0.0,
                    ),
                    1.0,
                )

                attempts = 1
                total_usage = first_usage
                final_usage = first_usage
                record_ids: list[str] = []
                if _unit(model_id, strategy_id, task_id, "retry") < retry_prob:
                    attempts = 2
                    final_usage = usage(
                        int(input_tokens * 0.98),
                        int(output_tokens * 1.02),
                        cache_read,
                        reasoning_tokens=reasoning_tokens,
                    )
                    total_usage = first_usage + final_usage
                    error_kind = ("rate_limit", "timeout", "server_error")[
                        int(_unit(model_id, task_id, "kind") * 3) % 3
                    ]
                    failed_call = call(
                        model_id,
                        strategy_id,
                        record_id=f"{model_id}-{strategy_id}-{task_id}-a1",
                        status=CallStatus.ERROR,
                        error_kind=error_kind,
                        usage_=first_usage,
                    )
                    failed_call.is_final_attempt = False
                    calls.append(failed_call)
                    record_ids.append(failed_call.record_id)

                final_call = call(
                    model_id,
                    strategy_id,
                    record_id=f"{model_id}-{strategy_id}-{task_id}-a{attempts}",
                    attempt=attempts,
                    usage_=final_usage,
                )
                calls.append(final_call)
                record_ids.append(final_call.record_id)

                results.append(
                    result(
                        model_id,
                        strategy_id,
                        task_id=task_id,
                        category=category,
                        score=score,
                        usage_total=total_usage,
                        usage_first=first_usage,
                        attempts=attempts,
                    )
                )

    usage_total = sum((r.usage_total for r in results), Usage())
    manifest = RunManifest(
        run_id="selfcheck-flagship",
        experiment_name="selfcheck-flagship",
        adapter_mode="mock",
        planned_calls=len(results),
        completed_calls=len(calls),
        failed_calls=sum(1 for c in calls if c.status in (CallStatus.ERROR, CallStatus.TIMEOUT)),
        retried_cells=sum(1 for r in results if r.retried),
        total_usage=usage_total,
        total_cost_usd=sum(DEFAULT_PRICE_BOOK.cost(r.model_id, r.usage_total) for r in results),
    )
    return calls, results, manifest


def test_selfcheck_200_cells_headline_numbers_are_plausible(run_root, capsys) -> None:
    """Build ~200 synthetic cells and assert every headline number is plausible.

    This is the "does the arithmetic make sense at scale?" check the unit tests
    cannot provide: they prove the formulas on toy inputs, this proves the
    pipeline produces sane magnitudes on a flagship-shaped run.  The printed
    block is the self-check output.
    """
    calls, results, manifest = _selfcheck_fixture()
    assert 190 <= len(results) <= 200, f"expected ~200 cells, got {len(results)}"

    store = RunStore.create(run_root, "selfcheck")
    store.write_manifest(manifest)
    for c in calls:
        store.append_call(c)
    for r in results:
        store.append_result(r)
    payload = build_report(store)

    summary = payload.summary
    retry = payload.retry
    cache = payload.cache

    # ---- magnitudes that must hold for any plausible 200-cell run ----------
    assert summary.completed_cells == len(results)
    assert summary.models == 5
    assert summary.strategies == 3
    assert summary.tasks == 13
    assert summary.total_usage.total_tokens > 0
    assert summary.total_cost_usd > 0.0
    # ~1500-6000 billable tokens per cell is the documented design target.
    tokens_per_cell = summary.total_billable_tokens / summary.completed_cells
    assert 1000 <= tokens_per_cell <= 8000, f"implausible {tokens_per_cell:.0f} tok/cell"

    # ---- leaderboard: normalised, ranked, one winner -----------------------
    assert len(payload.leaderboard) == 15  # 5 models x 3 strategies
    assert [r.rank for r in payload.leaderboard] == list(range(1, 16))
    assert all(0.0 <= r.efficiency_score <= 1.0 for r in payload.leaderboard)
    assert payload.leaderboard[0].efficiency_score == pytest.approx(1.0)
    assert payload.leaderboard[0].mean_quality > 0.0
    # Scores must be non-increasing down the ranking.
    scores = [r.efficiency_score for r in payload.leaderboard]
    assert scores == sorted(scores, reverse=True)

    # ---- pareto: frontier is a subset, dominated points have dominators ---
    assert payload.pareto.points
    assert 1 <= len(payload.pareto.frontier_labels) <= len(payload.pareto.points)
    for point in payload.pareto.points:
        if point.on_frontier:
            assert point.dominated_by == []
        else:
            assert point.dominated_by
    # Frontier labels are cost-ordered and every one is a real point.
    labels = {p.label: p for p in payload.pareto.points}
    frontier_costs = [labels[lbl].cost_per_task_usd for lbl in payload.pareto.frontier_labels]
    assert frontier_costs == sorted(frontier_costs)

    # ---- retry: the headline share is a real percentage of spend ----------
    assert 0.0 < retry.overall_retry_cost_share < 0.5, (
        f"retry share {retry.overall_retry_cost_share:.3f} is implausible for an ~8% "
        "retry-rate fixture"
    )
    assert retry.total_retry_cost_usd > 0.0
    assert retry.total_retry_cost_usd < retry.total_cost_usd
    assert retry.worst_offender is not None
    assert retry.worst_offender in {f"{r.model_id} / {r.strategy_id}" for r in retry.rows}
    # Retry spend cannot exceed the sum of its parts.
    assert retry.total_retry_cost_usd == pytest.approx(sum(r.retry_cost_usd for r in retry.rows))

    # ---- cache: monotone curve, anchored on the observed point ------------
    assert len(cache.curve) == 21
    assert cache.zero_cache_cost_usd > cache.full_cache_cost_usd > 0.0
    assert 0.0 <= cache.observed_hit_rate <= 1.0
    assert cache.zero_cache_cost_usd >= cache.observed_cost_usd >= cache.full_cache_cost_usd
    assert cache.marginal_usd_per_10pp < 0.0  # caching saves money
    curve_costs = [p.projected_cost_usd for p in cache.curve]
    assert curve_costs == sorted(curve_costs, reverse=True)

    # ---- deltas: cot costs more tokens than baseline, globally ------------
    global_rows = {d.strategy_id: d for d in payload.strategy_deltas if d.model_id is None}
    assert set(global_rows) == {"cot", "fewshot-compress"}
    assert global_rows["cot"].token_delta_pct > 0.0
    assert global_rows["cot"].cost_delta_pct > 0.0
    assert global_rows["cot"].quality_delta > 0.0
    assert global_rows["cot"].tokens_per_quality_delta is not None
    assert global_rows["cot"].verdict in {"worth it", "wasteful", "neutral"}

    # ---- print the self-check headline block ------------------------------
    with capsys.disabled():
        print("\n" + "=" * 74)
        print(f"tokbench analysis self-check — {len(results)} cells, {len(calls)} calls")
        print("=" * 74)
        print(f"  total spend     ${summary.total_cost_usd:,.6f}")
        print(f"  billable tokens {summary.total_billable_tokens:,} ({tokens_per_cell:,.0f}/cell)")
        print(f"  cache hit rate  {summary.cache_hit_rate * 100:.1f}%")
        print(
            f"  RETRY WASTE     {retry.overall_retry_cost_share * 100:.2f}% of spend "
            f"= ${retry.total_retry_cost_usd:,.6f}  (worst: {retry.worst_offender})"
        )
        print(f"  frontier        {' -> '.join(payload.pareto.frontier_labels)}")
        print(f"  best value      {payload.pareto.best_value_label}")
        print(
            f"  cache savings   ${cache.savings_at_observed_usd:,.6f} "
            f"({cache.savings_at_observed_pct:.2f}%) at observed hit rate; "
            f"${cache.marginal_usd_per_10pp:,.6f} per +10pp"
        )
        print(
            f"  top efficiency  {payload.leaderboard[0].model_id} / "
            f"{payload.leaderboard[0].strategy_id} "
            f"(score 1.000, {payload.leaderboard[0].billable_tokens_per_task:,.0f} tok/task)"
        )
        print("=" * 74)
