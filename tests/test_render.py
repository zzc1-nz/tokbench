"""Tests for the rendering layer: HTML report + Streamlit panel plumbing.

Two things are being verified here:

1. **The renderer is safe and total.**  ``render_html`` must produce a complete,
   self-contained document for any schema-valid payload —?including hostile
   model ids —?and must never leak Python literals (``None``/``nan``/``inf``)
   into the output.  Those are the failure modes that make a report look broken
   or dangerous, so they get explicit assertions.

2. **The panel is importable and renderable without a Streamlit runtime.**  We
   do not launch a server in tests; we exercise ``load_payload`` and the pure
   ``build_*`` chart-frame builders that the tabs call.  That catches the real
   regression (an import or signature break) in under a second.

The fixture is built **by hand** rather than from a real run so the tests are
deterministic and do not depend on the analysis layer, which is developed in
parallel.  Its numbers are chosen to be plausible: a believable Pareto frontier
with a genuine knee, retry waste concentrated in one pair, and a cache curve
that is linear in the hit rate (matching docs/METHODOLOGY.md →7).
"""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokbench.models import RunManifest, Usage
from tokbench.render import render_html, write_html
from tokbench.report_schema import (
    CachePoint,
    CacheSensitivity,
    CategoryRow,
    EfficiencyRow,
    ParetoFrontier,
    ParetoPoint,
    ReportMeta,
    ReportPayload,
    RetryAnalysis,
    RetryRow,
    RunSummary,
    SeriesPoint,
    StackedBarRow,
    StrategyDeltaRow,
)

# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #
#: Model tiers used by the fixture.  Chosen so the cost/quality relationship has
#: a real frontier shape: cheap-and-weak, mid-and-good, expensive-and-best.
_MODELS: list[tuple[str, str, float, float]] = [
    # (model_id, provider, cost_per_task_usd, mean_quality)
    ("mock-small", "mock", 0.0009, 0.412),
    ("gpt-4.1-nano", "openai", 0.0014, 0.508),
    ("gemini-2.0-flash", "google", 0.0021, 0.561),
    ("deepseek-chat", "deepseek", 0.0032, 0.642),
    ("gpt-4o-mini", "openai", 0.0048, 0.688),
    ("claude-3-5-haiku-20241022", "anthropic", 0.0071, 0.741),
    ("mock-medium", "mock", 0.0114, 0.779),
    ("gpt-4o", "openai", 0.0186, 0.804),
    ("claude-sonnet-4-20250514", "anthropic", 0.0297, 0.842),
    ("mock-large", "mock", 0.0463, 0.851),
]

_STRATEGIES: list[tuple[str, float, float, float]] = [
    # (strategy_id, token_multiplier, quality_bonus, retry_rate)
    ("baseline", 1.0, 0.0, 0.02),
    ("cot", 1.38, 0.031, 0.045),
    ("few-shot", 1.72, 0.019, 0.028),
]

_BASE_INPUT_TOKENS = 1180
_BASE_OUTPUT_TOKENS = 232
_CATEGORIES = ["codegen", "refactor", "explain", "debug", "tests"]
_N_CELLS = 45
_N_SERIES = 120


def _usage_for(model_idx: int, strat_idx: int, *, attempt: int = 1) -> Usage:
    """Plausible per-call usage for one fixture cell.

    Bigger models get longer outputs (they ramble) and better cache behaviour
    (they are the ones people pay to cache), which makes the stacked-bar chart
    show a realistic spread instead of uniform blocks.
    """
    token_mult = _STRATEGIES[strat_idx][1]
    input_tokens = int(_BASE_INPUT_TOKENS * token_mult * (1.0 + 0.05 * model_idx))
    output_tokens = int(_BASE_OUTPUT_TOKENS * token_mult * (1.0 + 0.09 * model_idx))
    cache_read = int(input_tokens * (0.18 + 0.032 * model_idx))
    cache_write = int(input_tokens * (0.05 + 0.004 * model_idx))
    reasoning = int(output_tokens * 0.42) if strat_idx == 1 else 0
    if attempt > 1:
        # a retry re-sends the prompt and gets cut short: more input, less output
        return Usage(
            input_tokens=int(input_tokens * 1.06),
            output_tokens=int(output_tokens * 0.35),
            cache_read_tokens=int(cache_read * 1.06),
            cache_write_tokens=0,
            reasoning_tokens=int(reasoning * 0.3),
        )
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
    )


def _on_frontier_flags(
    points: list[ParetoPoint],
) -> tuple[list[ParetoPoint], list[str]]:
    """Mark the non-dominated set and return (points, frontier labels in order).

    Uses the documented dominance rule (METHODOLOGY →5) so the fixture exercises
    the same semantics the analysis layer will produce.
    """
    frontier: list[ParetoPoint] = []
    for a in points:
        dominated_by = [
            b.label
            for b in points
            if b is not a
            and b.cost_per_task_usd <= a.cost_per_task_usd
            and b.quality >= a.quality
            and (b.cost_per_task_usd < a.cost_per_task_usd or b.quality > a.quality)
        ]
        a.dominated_by = dominated_by
        a.on_frontier = not dominated_by
        if a.on_frontier:
            frontier.append(a)
    frontier.sort(key=lambda p: (p.cost_per_task_usd, p.quality))
    return points, [p.label for p in frontier]


def build_fixture_payload(
    *,
    hostile_model_id: str | None = None,
) -> ReportPayload:
    """Construct a realistic :class:`ReportPayload` without touching disk.

    Args:
        hostile_model_id: When given, the first model id is replaced with this
            string.  Used by the XSS test to prove every insertion point escapes;
            kept out of the default fixture so the sample HTML stays readable.

    Returns:
        A fully populated payload: 10 models · 3 strategies, a 120-point series,
        retry rows with error breakdowns, a linear cache curve and strategy
        deltas against ``baseline``.
    """
    models = list(_MODELS)
    if hostile_model_id is not None:
        models[0] = (hostile_model_id, models[0][1], models[0][2], models[0][3])

    leaderboard: list[EfficiencyRow] = []
    pareto_points: list[ParetoPoint] = []
    retry_rows: list[RetryRow] = []
    stacked: list[StackedBarRow] = []
    categories: list[CategoryRow] = []
    deltas: list[StrategyDeltaRow] = []

    total_cost = 0.0
    total_retry_cost = 0.0
    total_retry_tokens = 0
    total_tokens_all = 0
    total_usage = Usage()
    error_totals: dict[str, int] = {}

    for mi, (model_id, provider, cost_per_task, quality) in enumerate(models):
        for si, (strategy_id, token_mult, quality_bonus, retry_rate) in enumerate(_STRATEGIES):
            cells = _N_CELLS
            first = _usage_for(mi, si)
            retried = max(1, int(round(cells * retry_rate))) if retry_rate else 0
            extra_attempts = retried
            retry_usage = Usage()
            for _ in range(retried):
                retry_usage = retry_usage + _usage_for(mi, si, attempt=2)
            usage_total = Usage(
                input_tokens=first.input_tokens * cells + retry_usage.input_tokens,
                output_tokens=first.output_tokens * cells + retry_usage.output_tokens,
                cache_read_tokens=first.cache_read_tokens * cells + retry_usage.cache_read_tokens,
                cache_write_tokens=first.cache_write_tokens * cells
                + retry_usage.cache_write_tokens,
                reasoning_tokens=first.reasoning_tokens * cells + retry_usage.reasoning_tokens,
            )
            pair_quality = min(quality + quality_bonus, 0.995)
            # Cost scales with tokens and with the model's price tier.
            pair_cost = cost_per_task * cells * token_mult
            retry_cost = pair_cost * (retry_usage.total_tokens / max(usage_total.total_tokens, 1))
            pair_cost_total = pair_cost

            total_cost += pair_cost_total
            total_retry_cost += retry_cost
            total_retry_tokens += retry_usage.total_tokens
            total_tokens_all += usage_total.total_tokens
            total_usage = total_usage + usage_total

            billable = usage_total.uncached_input_tokens + usage_total.output_tokens
            lb = EfficiencyRow(
                model_id=model_id,
                strategy_id=strategy_id,
                provider=provider,
                cells=cells,
                success_rate=1.0 - retry_rate * 0.5,
                total_usage=usage_total,
                billable_tokens=billable,
                cost_usd=pair_cost_total,
                tokens_per_task=usage_total.total_tokens / cells,
                billable_tokens_per_task=billable / cells,
                cost_per_task_usd=pair_cost_total / cells,
                output_tokens_per_task=usage_total.output_tokens / cells,
                mean_quality=pair_quality,
                mean_latency_ms=420.0 + 130.0 * mi + 60.0 * si,
                cache_hit_rate=usage_total.cache_hit_rate,
                tokens_per_quality_point=(billable / cells) / max(pair_quality, 1e-9),
                efficiency_score=0.0,
            )
            leaderboard.append(lb)

            pareto_points.append(
                ParetoPoint(
                    label=f"{model_id}/{strategy_id}",
                    model_id=model_id,
                    strategy_id=strategy_id,
                    cost_usd=pair_cost_total,
                    cost_per_task_usd=pair_cost_total / cells,
                    quality=pair_quality,
                    billable_tokens_per_task=billable / cells,
                    value_score=pair_quality / max(pair_cost_total / cells, 1e-9),
                )
            )

            if retry_rate:
                breakdown = {
                    "rate_limit": max(retried - 2, 0),
                    "timeout": 1 if retried > 2 else 0,
                    "server_error": 1 if retried > 4 else 0,
                }
                breakdown = {k: v for k, v in breakdown.items() if v}
                for kind, count in breakdown.items():
                    error_totals[kind] = error_totals.get(kind, 0) + count
                retry_rows.append(
                    RetryRow(
                        model_id=model_id,
                        strategy_id=strategy_id,
                        cells=cells,
                        retried_cells=retried,
                        retry_rate=retried / cells,
                        extra_attempts=extra_attempts,
                        retry_usage=retry_usage,
                        retry_cost_usd=retry_cost,
                        total_cost_usd=pair_cost_total,
                        retry_cost_share=retry_cost / max(pair_cost_total, 1e-9),
                        retry_token_share=retry_usage.total_tokens
                        / max(usage_total.total_tokens, 1),
                        dominant_error_kind=(
                            max(breakdown, key=breakdown.get) if breakdown else None
                        ),
                        error_breakdown=breakdown,
                    )
                )

            stacked.append(
                StackedBarRow(
                    model_id=model_id,
                    strategy_id=strategy_id,
                    uncached_input_tokens=usage_total.uncached_input_tokens,
                    cache_read_tokens=usage_total.cache_read_tokens,
                    cache_write_tokens=usage_total.cache_write_tokens,
                    output_tokens=usage_total.output_tokens,
                    reasoning_tokens=usage_total.reasoning_tokens,
                    cost_usd=pair_cost_total,
                    cells=cells,
                )
            )

            if si > 0:
                base = next(
                    r for r in leaderboard if r.model_id == model_id and r.strategy_id == "baseline"
                )
                tok_delta = (lb.billable_tokens_per_task - base.billable_tokens_per_task) / max(
                    base.billable_tokens_per_task, 1e-9
                )
                cost_delta = (lb.cost_per_task_usd - base.cost_per_task_usd) / max(
                    base.cost_per_task_usd, 1e-9
                )
                q_delta = lb.mean_quality - base.mean_quality
                if q_delta > 0.02 and cost_delta < 0.25:
                    verdict = "worth it"
                elif q_delta < 0.005 and cost_delta > 0.10:
                    verdict = "wasteful"
                else:
                    verdict = "neutral"
                deltas.append(
                    StrategyDeltaRow(
                        baseline_strategy="baseline",
                        strategy_id=strategy_id,
                        model_id=model_id,
                        baseline_billable_per_task=base.billable_tokens_per_task,
                        strategy_billable_per_task=lb.billable_tokens_per_task,
                        token_delta_pct=tok_delta,
                        baseline_cost_per_task_usd=base.cost_per_task_usd,
                        strategy_cost_per_task_usd=lb.cost_per_task_usd,
                        cost_delta_pct=cost_delta,
                        baseline_quality=base.mean_quality,
                        quality=q_delta + base.mean_quality,
                        quality_delta=q_delta,
                        tokens_per_quality_delta=(
                            (lb.billable_tokens_per_task - base.billable_tokens_per_task)
                            / (q_delta * 100)
                            if q_delta
                            else None
                        ),
                        verdict=verdict,
                    )
                )

            for ci, category in enumerate(_CATEGORIES[:2]):
                categories.append(
                    CategoryRow(
                        model_id=model_id,
                        strategy_id=strategy_id,
                        category=category,
                        cells=max(cells // len(_CATEGORIES), 1),
                        success_rate=1.0 - retry_rate * 0.6,
                        billable_tokens_per_task=billable / cells * (1.0 + 0.08 * ci),
                        cost_per_task_usd=pair_cost_total / cells * (1.0 + 0.08 * ci),
                        mean_quality=max(pair_quality - 0.03 * ci, 0.0),
                    )
                )

    # ---- rank the leaderboard on quality per billable token ----------------
    for row in leaderboard:
        raw = row.mean_quality / max(row.tokens_per_task, 1.0)
        row.efficiency_score = raw
    best = max(r.efficiency_score for r in leaderboard) or 1.0
    for row in leaderboard:
        row.efficiency_score = row.efficiency_score / best
    leaderboard.sort(key=lambda r: (-r.efficiency_score, r.cost_per_task_usd))
    for i, row in enumerate(leaderboard, 1):
        row.rank = i

    pareto_points, frontier_labels = _on_frontier_flags(pareto_points)
    frontier = [p for p in pareto_points if p.on_frontier]
    gain = 0.0
    if len(frontier) >= 2:
        dx = frontier[-1].cost_per_task_usd - frontier[0].cost_per_task_usd
        dy = frontier[-1].quality - frontier[0].quality
        gain = dy / dx if dx > 0 else 0.0

    best_value = max(pareto_points, key=lambda p: p.value_score)
    best_quality = max(pareto_points, key=lambda p: p.quality)
    cheapest = min(pareto_points, key=lambda p: p.cost_per_task_usd)

    # ---- cache sensitivity: linear in hit rate (METHODOLOGY →7) ------------
    zero_cost = total_cost / max(1.0 - 0.30, 1e-9) * (1.0 - 0.0)
    observed_hit = total_usage.cache_hit_rate
    # A run with ~18% cache reads pays proportionally less than the no-cache case.
    zero_cost = total_cost / max(1.0 - 0.62 * observed_hit, 1e-9)
    full_cost = zero_cost * 0.24
    curve: list[CachePoint] = []
    for i in range(0, 21):
        h = i / 20.0
        projected = zero_cost + (full_cost - zero_cost) * h
        curve.append(
            CachePoint(
                cache_hit_rate=h,
                projected_cost_usd=projected,
                projected_cost_per_task_usd=projected / (len(leaderboard) * _N_CELLS),
                savings_vs_zero_cache_usd=zero_cost - projected,
                savings_vs_zero_cache_pct=(zero_cost - projected) / max(zero_cost, 1e-9),
            )
        )
    observed_cost = zero_cost + (full_cost - zero_cost) * observed_hit

    # ---- series: 120 cumulative points with a retry spike ------------------
    series: list[SeriesPoint] = []
    cum_tokens = 0
    cum_cost = 0.0
    start = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
    per_cell = max(total_tokens_all // _N_SERIES, 1)
    for i in range(_N_SERIES):
        model_id = models[i % len(models)][0]
        strategy_id = _STRATEGIES[i % len(_STRATEGIES)][0]
        bump = 1.0 + (0.42 if i % 17 == 0 else 0.0)  # occasional retry spike
        tokens = int(per_cell * bump)
        cum_tokens += tokens
        cum_cost += total_cost / _N_SERIES * bump
        series.append(
            SeriesPoint(
                t=start + timedelta(minutes=3 * i),
                index=i,
                model_id=model_id,
                strategy_id=strategy_id,
                tokens=tokens,
                billable_tokens=int(tokens * 0.78),
                cost_usd=total_cost / _N_SERIES * bump,
                cumulative_tokens=cum_tokens,
                cumulative_billable_tokens=int(cum_tokens * 0.78),
                cumulative_cost_usd=cum_cost,
                cumulative_cache_hit_rate=observed_hit,
            )
        )

    retry_share = total_retry_cost / max(total_cost, 1e-9)
    worst = max(retry_rows, key=lambda r: r.retry_cost_usd) if retry_rows else None

    return ReportPayload(
        meta=ReportMeta(
            run_id="20261110-0900-render-fixture",
            experiment="token-economics / render fixture",
            generated_at=datetime(2026, 11, 10, 13, 38, tzinfo=timezone.utc),
            price_book="tokbench-default-2026",
            price_book_date="2026-01-01",
            currency="USD",
            notes=[
                "Hand-built fixture payload —?not a real run.",
                "Numbers are synthetic but internally consistent.",
            ],
        ),
        summary=RunSummary(
            models=len(models),
            strategies=len(_STRATEGIES),
            tasks=5,
            repeats=1,
            planned_cells=len(leaderboard) * _N_CELLS,
            completed_cells=len(leaderboard) * _N_CELLS,
            total_calls=len(leaderboard) * _N_CELLS + total_retry_tokens // 400,
            failed_calls=len(retry_rows),
            retried_cells=len(retry_rows),
            success_rate=0.982,
            total_usage=total_usage,
            total_cost_usd=total_cost,
            total_billable_tokens=(total_usage.uncached_input_tokens + total_usage.output_tokens),
            wall_clock_s=248.6,
            cost_per_1k_billable_tokens=(
                total_cost
                / max(total_usage.uncached_input_tokens + total_usage.output_tokens, 1)
                * 1000
            ),
            cache_hit_rate=observed_hit,
        ),
        leaderboard=leaderboard,
        categories=categories,
        pareto=ParetoFrontier(
            points=pareto_points,
            frontier_labels=frontier_labels,
            frontier_quality_gain_per_usd=gain,
            baseline_label=f"{models[0][0]}/baseline",
            best_value_label=best_value.label,
            best_quality_label=best_quality.label,
            cheapest_label=cheapest.label,
            explanation=(
                "The frontier trades cost for quality; the knee is the pragmatic "
                "recommendation where marginal quality per dollar collapses."
            ),
        ),
        retry=RetryAnalysis(
            rows=retry_rows,
            total_retry_cost_usd=total_retry_cost,
            total_cost_usd=total_cost,
            overall_retry_cost_share=retry_share,
            total_retry_tokens=total_retry_tokens,
            total_tokens=total_tokens_all,
            retry_token_share=total_retry_tokens / max(total_tokens_all, 1),
            wasted_pct_of_budget=retry_share,
            error_breakdown=error_totals,
            worst_offender=(f"{worst.model_id}/{worst.strategy_id}" if worst is not None else None),
            explanation="Retry overhead is every token spent after attempt one.",
        ),
        cache=CacheSensitivity(
            curve=curve,
            observed_hit_rate=observed_hit,
            observed_cost_usd=observed_cost,
            zero_cache_cost_usd=zero_cost,
            full_cache_cost_usd=full_cost,
            savings_at_observed_usd=zero_cost - observed_cost,
            savings_at_observed_pct=(zero_cost - observed_cost) / max(zero_cost, 1e-9),
            marginal_usd_per_10pp=(zero_cost - full_cost) * 0.10,
            explanation="Cost is linear in hit rate when token volume is held fixed.",
        ),
        series=series,
        stacked=stacked,
        strategy_deltas=deltas,
        manifest=RunManifest(
            run_id="20261110-0900-render-fixture",
            experiment_name="token-economics / render fixture",
            started_at=start,
            finished_at=start + timedelta(seconds=248),
            engine_version="0.1.0",
            host="fixture",
            python="3.14.5",
            config_hash="deadbeefcafe1234",
            planned_calls=len(leaderboard) * _N_CELLS,
            completed_calls=len(leaderboard) * _N_CELLS,
            failed_calls=len(retry_rows),
            retried_cells=len(retry_rows),
            total_usage=total_usage,
            total_cost_usd=total_cost,
            wall_clock_s=248.6,
            seed=42,
            adapter_mode="mock",
        ),
    )


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def payload() -> ReportPayload:
    """The default realistic fixture payload (module-scoped: it is read-only)."""
    return build_fixture_payload()


@pytest.fixture(scope="module")
def rendered(payload: ReportPayload) -> str:
    """The default fixture rendered once, reused by the structural assertions."""
    return render_html(payload)


@pytest.fixture()
def run_tmp() -> Iterator[Path]:
    """A throwaway directory *inside the workspace*.

    Three constraints shaped this, all discovered empirically:

    1. **Not pytest's ``tmp_path``.**  That resolves under the OS temp
       directory, which is not writable in the confined sandbox this project is
       developed in, so every ``tmp_path`` test errored out.
    2. **Not ``tempfile.mkdtemp``.**  It applies a restrictive ACL that leaves
       the directory readable but not writable, so creating children inside it
       fails with ``WinError 5``.
    3. **The parent must start with a dot.**  Pytest's default
       ``norecursedirs`` includes ``.*`` but *not* ``_*``, so a scratch parent
       named ``_tmp_render`` gets collected as a test package — which made a
       full-suite run fail at collection time with a permission error.  Naming
       it ``.tmp_render`` keeps it invisible to collection.

    Each test gets its own unique subdirectory (uuid-suffixed), so tests never
    share state, and the directory is removed even when the test fails.
    """
    base = Path(__file__).resolve().parent / ".tmp_render"
    base.mkdir(parents=True, exist_ok=True)
    work = base / f"run-{uuid.uuid4().hex[:12]}"
    work.mkdir()
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Payload round-trip
# --------------------------------------------------------------------------- #
def test_payload_round_trips_through_json(payload: ReportPayload) -> None:
    """The payload must survive a JSON hop —?it doubles as the machine report.

    ``exclude_computed_fields=True`` is required: :class:`Usage` exposes derived
    ``computed_field`` values (``total_tokens``, ``billable_tokens``, ...) that
    are serialised by default but rejected on the way back in, because the
    models are ``extra="forbid"``.  Writing the report with the computed fields
    included would produce a ``report.json`` that cannot be re-read.
    """
    text = payload.model_dump_json(exclude_computed_fields=True)
    restored = ReportPayload.model_validate_json(text)
    assert restored == payload
    # And the computed fields still work after the round-trip.
    assert restored.summary.total_usage.billable_tokens >= 0


def test_fixture_is_realistic(payload: ReportPayload) -> None:
    """Guard the fixture itself: a degenerate fixture proves nothing."""
    assert len(payload.leaderboard) == 30  # 10 models x 3 strategies
    assert len(payload.pareto.points) == 30
    assert len(payload.series) >= 100
    assert len(payload.retry.rows) >= 5
    assert len(payload.cache.curve) >= 10
    assert len(payload.strategy_deltas) >= 10
    assert payload.summary.total_cost_usd > 0
    # A believable frontier: more than two points, and not everything is on it.
    assert len(payload.pareto.frontier_labels) >= 3
    assert len(payload.pareto.frontier_labels) < len(payload.pareto.points)
    # Retry waste must be a visible but non-absurd fraction of spend.
    assert 0.0 < payload.retry.overall_retry_cost_share < 0.5


# --------------------------------------------------------------------------- #
# HTML structure
# --------------------------------------------------------------------------- #
def test_html_is_a_complete_document(rendered: str) -> None:
    """A single, well-formed, standalone HTML document."""
    assert rendered.startswith("<!doctype html>")
    assert rendered.rstrip().endswith("</html>")
    assert rendered.count("<body") == 1
    assert rendered.count("</body>") == 1
    # Exactly one inline stylesheet, no external ones.
    assert rendered.count("<style>") == 1
    assert "<link" not in rendered


def test_html_contains_every_section(rendered: str) -> None:
    """Every chart/table section the brief requires is present and anchored."""
    for section_id in (
        "section-pareto",
        "section-retry",
        "section-cache",
        "section-series",
        "section-stacked",
        "section-leaderboard",
        "section-strategy",
    ):
        assert f'id="{section_id}"' in rendered, f"missing {section_id}"


def test_html_contains_every_chart(rendered: str) -> None:
    """Charts are inline SVG with an accessible name, not empty placeholders."""
    assert rendered.count("<svg") >= 5
    assert rendered.count("</svg>") == rendered.count("<svg")
    assert rendered.count("<title>") >= rendered.count("<svg")
    # The hero and the two supporting charts must not have fallen back to the
    # empty state (the fixture always has data for them).
    for anchor in ("pareto-empty", "cache-empty", "series-empty", "stacked-empty"):
        assert f'id="{anchor}"' not in rendered


def test_hero_pareto_has_axes_frontier_and_knee(rendered: str) -> None:
    """The hero chart must carry the specific furniture the brief calls for.

    Assertions are case-insensitive on purpose: the SVG axis titles are set in
    lowercase (a deliberate micro-caps style, uppercased again by CSS
    ``text-transform``), while the figcaption beneath the chart is sentence
    case.  Both are correct; pinning an exact casing here would test
    typography rather than structure, and would break the moment either is
    restyled.  The strings asserted are the ones actually emitted.
    """
    pareto_start = rendered.index('id="section-pareto"')
    pareto = rendered[pareto_start : rendered.index('id="section-retry"')]
    lowered = pareto.lower()
    # axis labels with units — the x title gains ", log scale" when the cost
    # span crosses a decade, so assert on the stable prefix.
    assert "cost per task (usd" in lowered
    assert "mean quality (0-1)" in lowered
    # gridlines + ticks + axis rules
    assert 'class="grid' in pareto
    assert "tick--x" in pareto and "tick--y" in pareto
    assert 'class="axis"' in pareto
    # frontier line and markers
    assert "fline" in pareto
    assert "p-front" in pareto
    # dominated points are dimmed, not hidden
    assert "p-dom" in pareto
    # per-point labels and inline legend
    assert "plabel" in pareto
    assert "pareto frontier" in lowered
    # knee annotation
    assert "knee" in lowered
    assert "knee point" in lowered
    # tooltips
    assert "<title>" in pareto
    # label placement must not have fallen back to a broken state
    assert "leader" in lowered or "plabel" in pareto


def test_retry_share_is_unmissable(rendered: str) -> None:
    """The retry panel needs the big number, the meter and per-pair bars."""
    assert 'class="waste-num"' in rendered
    assert 'class="meter"' in rendered
    assert "retry cost share" in rendered
    assert "seg-waste" in rendered  # per-model stacked bar waste segment
    assert "seg-first" in rendered
    assert 'id="retry-table"' in rendered


def test_cache_chart_marks_observed_rate(rendered: str) -> None:
    """The observed hit rate must be pinned on the curve, not just tabulated."""
    assert "p-observed" in rendered
    assert "obs-line" in rendered
    assert "observed" in rendered
    assert "curve" in rendered


def test_series_and_stacked_charts_have_data(rendered: str) -> None:
    """The area chart and the stacked decomposition both drew real geometry."""
    assert "area-fill" in rendered
    assert "series-line" in rendered
    assert "seg-uncached" in rendered
    assert "seg-cache-read" in rendered
    assert "seg-output" in rendered


def test_tables_present_with_medals(rendered: str) -> None:
    """Leaderboard (with rank medals) and the ablation delta table are rendered."""
    assert 'id="leaderboard-table"' in rendered
    assert 'id="delta-table"' in rendered
    assert "rank--1" in rendered
    assert "rank--2" in rendered
    assert "rank--3" in rendered
    # the verdict column resolves to a coloured badge
    assert "badge--good" in rendered or "badge--warn" in rendered


def test_kpi_cards_present(rendered: str) -> None:
    """The headline KPI row is there, including the retry-waste tile."""
    for kpi in ("kpi-spend", "kpi-retry", "kpi-tokens", "kpi-cache", "kpi-cells"):
        assert f'id="{kpi}"' in rendered


def test_footer_documents_methodology(rendered: str) -> None:
    """The report must explain how its numbers were produced."""
    assert "<footer>" in rendered
    assert "methodology" in rendered
    assert "Pareto" in rendered


def test_print_stylesheet_present(rendered: str) -> None:
    """Print-friendly output is part of the brief."""
    assert "@media print" in rendered
    assert "break-inside" in rendered


def test_theme_switch_produces_light_palette(payload: ReportPayload) -> None:
    """``theme="light"`` changes tokens without changing structure."""
    light = render_html(payload, theme="light")
    assert 'data-theme="light"' in light
    assert "--bg:#ffffff" in light
    with pytest.raises(ValueError):
        render_html(payload, theme="solarized")


# --------------------------------------------------------------------------- #
# Self-containment + hygiene
# --------------------------------------------------------------------------- #
def test_no_external_references(rendered: str) -> None:
    """Zero network requests when opened offline.

    Checked by looking for *any* scheme-qualified URL or fetchable attribute.
    The SVG namespace URI is deliberately omitted from inline ``<svg>`` tags for
    exactly this reason, so this assertion stays honest rather than exempting
    known-benign hits.

    ``url(#id)`` is permitted: it is an intra-document reference to the one
    inline gradient definition, which resolves without a request.  Anything
    that could reach the network —?``url(http...)``, ``url(//...)`` —?is not.
    """
    lowered = rendered.lower()
    for needle in (
        "http://",
        "https://",
        "//cdn",
        "<script",
        "<link",
        "<iframe",
        "@import",
        "src=",
        "javascript:",
        "onload=",
    ):
        assert needle not in lowered, f"external reference found: {needle!r}"
    # Every url(...) must be a same-document fragment reference.
    for match in re.finditer(r"url\(\s*([^)]*)\)", lowered):
        assert match.group(1).strip().startswith("#"), (
            f"non-fragment url() reference: {match.group(0)!r}"
        )


def test_no_python_literals_leak(rendered: str) -> None:
    """``None``/``nan``/``inf`` in a report reads like a bug —?forbid them.

    Matched as standalone tokens rather than substrings, because real model ids
    legitimately contain these letters (``gpt-4.1-nano`` is not a NaN).
    """
    assert not re.search(r"(?<![\w.-])None(?![\w-])", rendered), "None leaked"
    assert not re.search(r"(?<![\w.-])nan(?![\w-])", rendered, re.I), "nan leaked"
    assert not re.search(r"(?<![\w.-])inf(?![\w-])", rendered, re.I), "inf leaked"
    assert not re.search(r"(?<![\w.-])-?infinity(?![\w-])", rendered, re.I)
    # And the document is still complete.
    assert rendered.rstrip().endswith("</html>")


def test_missing_values_render_as_dash_not_none() -> None:
    """A sparse payload must degrade to em dashes, never to ``None``."""
    sparse = ReportPayload(
        meta=ReportMeta(
            run_id="sparse",
            experiment="sparse run",
            price_book_date="",
            notes=[],
        ),
        pareto=ParetoFrontier(
            points=[
                ParetoPoint(
                    label="solo/model",
                    model_id="solo",
                    strategy_id="model",
                    cost_per_task_usd=float("nan"),
                    quality=float("nan"),
                    on_frontier=True,
                )
            ]
        ),
        series=[],
        stacked=[],
        leaderboard=[],
        strategy_deltas=[],
    )
    out = render_html(sparse)
    assert "None" not in out
    assert "nan" not in out.lower()
    assert "\u2014" in out  # em dash used for the absent values
    # Empty states are announced rather than silently skipped.
    assert 'id="leaderboard-empty"' in out
    assert 'id="deltas-empty"' in out


# --------------------------------------------------------------------------- #
# XSS / escaping
# --------------------------------------------------------------------------- #
_HOSTILE = '<script>alert("xss")</script><img src=x onerror=alert(1)>'

#: Every tag the renderer is allowed to emit.  Used by the XSS test to prove an
#: injected ``<script>``/``<img>`` never survives as real markup.
_ALLOWED_TAGS = {
    "a",
    "body",
    "br",
    "caption",
    "code",
    "desc",
    "div",
    "defs",
    "footer",
    "g",
    "h1",
    "h2",
    "h3",
    "head",
    "header",
    "html",
    "li",
    "line",
    "linearGradient",
    "main",
    "meta",
    "path",
    "polygon",
    "polyline",
    "rect",
    "section",
    "span",
    "stop",
    "style",
    "svg",
    "table",
    "tbody",
    "td",
    "text",
    "th",
    "thead",
    "title",
    "tr",
    "ul",
    "circle",
    "figcaption",
    "figure",
    "p",
    "b",
}


def test_hostile_model_id_is_escaped_everywhere() -> None:
    """A model id containing markup must never become live markup.

    The payload is a *document*: model ids flow into table cells, SVG ``<text>``
    nodes, ``<title>`` tooltips, attributes and the ``<h1>``.  One unescaped
    insertion point is a stored-XSS hole in a file people open in a browser, so
    this asserts on the raw tags rather than on a single location.

    Note the subtlety: the *escaped* text still contains the literal substring
    ``onerror=alert(1)`` —?that is expected and harmless, because it sits inside
    ``&lt;img ...&gt;`` and is never parsed as a tag.  What must not exist is a
    real ``<img`` tag, so the assertions look for markup delimiters, not for the
    attribute name.
    """
    hostile = build_fixture_payload(hostile_model_id=_HOSTILE)
    out = render_html(hostile)

    # No live markup from the injected payload.
    assert "<script>" not in out
    assert "</script>" not in out
    assert "<img" not in out
    # The only tags in the document are the ones the renderer itself emits.
    assert set(re.findall(r"<\s*([a-zA-Z][\w-]*)", out)) <= _ALLOWED_TAGS
    # ...while its escaped form is present, proving it was rendered (not dropped).
    assert "&lt;script&gt;" in out
    assert "&lt;img" in out
    # The document still closes correctly: no tag was broken open.
    assert out.rstrip().endswith("</html>")


def test_escaping_covers_quotes_in_attributes() -> None:
    """``html.escape(quote=True)`` matters: ids land inside ``title="..."``."""
    hostile = build_fixture_payload(hostile_model_id='x" onmouseover="alert(1)')
    out = render_html(hostile)
    # A raw, unescaped attribute injection would produce a second onmouseover.
    assert '" onmouseover="' not in out
    assert "&quot;" in out


def test_hostile_error_kind_is_escaped() -> None:
    """Error strings come from provider responses —?also untrusted."""
    payload = build_fixture_payload()
    payload.retry.error_breakdown = {"<script>bad()</script>": 3}
    payload.retry.worst_offender = "<b>evil</b>"
    out = render_html(payload)
    assert "<script>" not in out
    assert "<b>evil</b>" not in out
    assert "&lt;script&gt;" in out


# --------------------------------------------------------------------------- #
# write_html
# --------------------------------------------------------------------------- #
def test_write_html_writes_the_same_bytes(run_tmp: Path, payload: ReportPayload) -> None:
    """``write_html`` is the file-system face of ``render_html``."""
    target = run_tmp / "nested" / "report.html"
    written = write_html(payload, target, run_dir=run_tmp)
    assert written.exists()
    assert written == target.resolve()
    assert written.read_text(encoding="utf-8") == render_html(payload, run_dir=run_tmp)
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_render_is_deterministic(payload: ReportPayload) -> None:
    """Same payload —?byte-identical output (reports must be diff-able)."""
    assert render_html(payload) == render_html(payload)


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def test_panel_module_imports() -> None:
    """Importing the panel must not require a Streamlit server or a run on disk."""
    from tokbench import panel

    assert callable(panel.main)
    assert callable(panel.load_payload)
    assert callable(panel.main_payload)
    assert callable(panel.build_pareto_figure)
    assert callable(panel.build_series_figure)
    assert callable(panel.build_stacked_figure)
    assert callable(panel.build_retry_figure)
    assert callable(panel.build_cache_figure)
    assert callable(panel.leaderboard_frame)
    assert callable(panel.raw_frames)


def test_panel_loads_payload_from_report_json(run_tmp: Path) -> None:
    """``load_payload`` prefers the cached ``report.json`` in the run dir."""
    from tokbench.panel import load_payload

    run_dir = run_tmp / "run-1"
    run_dir.mkdir()
    fixture = build_fixture_payload()
    (run_dir / "report.json").write_text(
        fixture.model_dump_json(exclude_computed_fields=True), encoding="utf-8"
    )
    loaded = load_payload(run_dir)
    assert loaded.meta.run_id == fixture.meta.run_id
    assert len(loaded.leaderboard) == len(fixture.leaderboard)


def test_panel_falls_back_to_analysis(run_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``report.json`` the panel computes one via the analysis layer.

    The analysis module is built in parallel, so this test injects a fake
    ``tokbench.analysis.report`` —?the point is that ``load_payload`` *tries*
    that path (lazily) instead of crashing when the file is absent.
    """
    import sys
    import types

    from tokbench.panel import load_payload

    run_dir = run_tmp / "run-2"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    fixture = build_fixture_payload()
    fake_mod = types.ModuleType("tokbench.analysis.report")

    def build_report(store, price_book=None, baseline_strategy=None):  # noqa: ANN001
        return fixture

    fake_mod.build_report = build_report  # type: ignore[attr-defined]
    pkg = types.ModuleType("tokbench.analysis")
    pkg.report = fake_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokbench.analysis", pkg)
    monkeypatch.setitem(sys.modules, "tokbench.analysis.report", fake_mod)

    loaded = load_payload(run_dir)
    assert loaded.meta.run_id == fixture.meta.run_id


def test_panel_figures_build_from_fixture(payload: ReportPayload) -> None:
    """Every plotly builder returns a figure for the fixture (no exceptions)."""
    from tokbench import panel

    for builder in (
        panel.build_pareto_figure,
        panel.build_series_figure,
        panel.build_stacked_figure,
        panel.build_retry_figure,
        panel.build_cache_figure,
    ):
        fig = builder(payload)
        assert fig is not None
        assert fig.data, f"{builder.__name__} produced an empty figure"


def test_panel_frames_are_dataframes(payload: ReportPayload) -> None:
    """The tabular helpers return tidy pandas frames with stable columns."""
    from tokbench import panel

    lb = panel.leaderboard_frame(payload)
    assert len(lb) == len(payload.leaderboard)
    assert "model" in lb.columns and "score" in lb.columns
    calls, results = panel.raw_frames(payload)
    assert list(calls.columns) == list(results.columns)


def test_raw_frames_are_arrow_serialisable(payload: ReportPayload) -> None:
    """Every Raw-tab frame must survive pyarrow conversion.

    ``st.dataframe`` serialises through pyarrow.  A frame whose column mixes
    ints with JSON strings becomes ``object``-dtype and raises
    ``ArrowTypeError: Expected bytes, got a 'int' object`` at render time — a
    bug that is invisible to a plain ``isinstance(df, DataFrame)`` assertion and
    only shows up when the tab actually draws.  This pins the invariant.
    """
    pa = pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    from tokbench import panel

    # The real payload carries scalar diagnostics under ``raw``; make sure the
    # fallback path (no calls/results lists) is exercised too.
    sparse = ReportPayload(
        meta=ReportMeta(run_id="raw", experiment="raw"),
        raw={
            "baseline_strategy": "baseline",
            "cells": 2250,
            "calls": 2400,
            "quality_histogram": {"0.5": 10, "0.6": 22},
            "unpriced_models": [],
        },
    )
    for source in (payload, sparse):
        calls, results = panel.raw_frames(source)
        for frame in (calls, results):
            assert isinstance(frame, pd.DataFrame)
            # This is the assertion that would have caught the mixed-type bug.
            pa.Table.from_pandas(frame)


def test_panel_figure_builders_tolerate_empty_payload() -> None:
    """A payload with no data must not raise inside a chart builder."""
    from tokbench import panel

    empty = ReportPayload(meta=ReportMeta(run_id="e", experiment="empty"))
    for builder in (
        panel.build_pareto_figure,
        panel.build_series_figure,
        panel.build_stacked_figure,
        panel.build_retry_figure,
        panel.build_cache_figure,
    ):
        assert builder(empty) is not None


def test_panel_runs_as_a_bare_script() -> None:
    """``streamlit run app.py`` must work, not just ``import tokbench.panel``.

    Streamlit executes the app file as ``__main__`` with no package context, so
    ``from ..models import ...`` raises "attempted relative import with no known
    parent package".  That failure mode is invisible to an import test, which is
    exactly how it shipped once.  Run the file the way Streamlit does — as a
    standalone script via ``runpy`` — and assert it defines its entry points.
    """
    import runpy
    import sys

    import tokbench

    app_path = Path(tokbench.__file__).parent / "panel" / "app.py"
    assert app_path.exists()

    # Simulate `streamlit run`: __main__ semantics, no package set on the module.
    saved = sys.argv
    sys.argv = [str(app_path)]
    try:
        namespace = runpy.run_path(str(app_path), run_name="__main__")
    finally:
        sys.argv = saved

    for name in (
        "main",
        "load_payload",
        "build_pareto_figure",
        "build_retry_figure",
        "build_cache_figure",
        "leaderboard_frame",
    ):
        assert callable(namespace[name]), f"{name} missing when run as a script"

    # The script branch must have put the project root on sys.path so that
    # `import tokbench` works inside the panel at runtime.
    assert "tokbench" in sys.modules
