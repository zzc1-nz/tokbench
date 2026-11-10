"""Report payload schema — the contract between analysis and rendering.

``tokbench.analysis`` *produces* a :class:`ReportPayload`.
``tokbench.render`` and ``tokbench.panel`` *consume* it.

Freezing this schema lets the analysis, HTML renderer and Streamlit panel be
developed independently: both sides only need this file to agree.  Everything is
JSON-serialisable, so the payload doubles as the machine-readable report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import RunManifest, Usage, utcnow

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Meta / summary
# --------------------------------------------------------------------------- #
class ReportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    experiment: str
    generated_at: datetime = Field(default_factory=utcnow)
    generator: str = "tokbench"
    generator_version: str = "0.1.0"
    price_book: str = "default"
    price_book_date: str = ""
    currency: str = "USD"
    notes: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Headline numbers for the whole run."""

    model_config = ConfigDict(extra="forbid")

    models: int = 0
    strategies: int = 0
    tasks: int = 0
    repeats: int = 0
    planned_cells: int = 0
    completed_cells: int = 0
    total_calls: int = 0
    failed_calls: int = 0
    retried_cells: int = 0
    success_rate: float = 0.0
    total_usage: Usage = Field(default_factory=Usage)
    total_cost_usd: float = 0.0
    total_billable_tokens: int = 0
    wall_clock_s: float = 0.0
    cost_per_1k_billable_tokens: float = 0.0
    cache_hit_rate: float = 0.0


# --------------------------------------------------------------------------- #
# Token efficiency ranking
# --------------------------------------------------------------------------- #
class EfficiencyRow(BaseModel):
    """One (model, strategy) pair ranked on token economy."""

    model_config = ConfigDict(extra="forbid")

    rank: int = 0
    model_id: str
    strategy_id: str
    provider: str = ""
    cells: int = 0
    success_rate: float = 0.0
    total_usage: Usage = Field(default_factory=Usage)
    billable_tokens: int = 0
    cost_usd: float = 0.0
    tokens_per_task: float = 0.0
    billable_tokens_per_task: float = 0.0
    cost_per_task_usd: float = 0.0
    output_tokens_per_task: float = 0.0
    mean_quality: float = 0.0
    mean_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    tokens_per_quality_point: float | None = None
    efficiency_score: float = 0.0


class CategoryRow(BaseModel):
    """Per-task-category breakdown, useful for spotting hard categories."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    strategy_id: str
    category: str
    cells: int = 0
    success_rate: float = 0.0
    billable_tokens_per_task: float = 0.0
    cost_per_task_usd: float = 0.0
    mean_quality: float = 0.0


# --------------------------------------------------------------------------- #
# Pareto frontier (cost vs quality)
# --------------------------------------------------------------------------- #
class ParetoPoint(BaseModel):
    """One candidate on the cost/quality plane."""

    model_config = ConfigDict(extra="forbid")

    label: str
    model_id: str
    strategy_id: str
    cost_usd: float = 0.0
    cost_per_task_usd: float = 0.0
    quality: float = 0.0
    billable_tokens_per_task: float = 0.0
    on_frontier: bool = False
    dominated_by: list[str] = Field(default_factory=list)
    #: Relative quality per dollar, higher is better.
    value_score: float = 0.0


class ParetoFrontier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[ParetoPoint] = Field(default_factory=list)
    frontier_labels: list[str] = Field(default_factory=list)
    frontier_quality_gain_per_usd: float = 0.0
    baseline_label: str | None = None
    best_value_label: str | None = None
    best_quality_label: str | None = None
    cheapest_label: str | None = None
    #: The "elbow" of the frontier — the point where extra spend stops buying a
    #: proportionate amount of quality (max perpendicular distance from the chord
    #: joining the cheapest and best-quality points).  Authoritative: the analysis
    #: layer computes it once and every renderer reads it, so the HTML report, the
    #: Streamlit panel and the README figure can never disagree about where the
    #: knee is.
    knee_label: str | None = None
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Retry overhead
# --------------------------------------------------------------------------- #
class RetryRow(BaseModel):
    """Retry waste for one (model, strategy) pair."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    strategy_id: str
    cells: int = 0
    retried_cells: int = 0
    retry_rate: float = 0.0
    extra_attempts: int = 0
    retry_usage: Usage = Field(default_factory=Usage)
    retry_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    retry_cost_share: float = 0.0
    retry_token_share: float = 0.0
    dominant_error_kind: str | None = None
    error_breakdown: dict[str, int] = Field(default_factory=dict)


class RetryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[RetryRow] = Field(default_factory=list)
    total_retry_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    overall_retry_cost_share: float = 0.0
    total_retry_tokens: int = 0
    total_tokens: int = 0
    retry_token_share: float = 0.0
    wasted_pct_of_budget: float = 0.0
    error_breakdown: dict[str, int] = Field(default_factory=dict)
    worst_offender: str | None = None
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Cache sensitivity
# --------------------------------------------------------------------------- #
class CachePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_hit_rate: float
    projected_cost_usd: float
    projected_cost_per_task_usd: float
    savings_vs_zero_cache_usd: float
    savings_vs_zero_cache_pct: float


class CacheSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curve: list[CachePoint] = Field(default_factory=list)
    observed_hit_rate: float = 0.0
    observed_cost_usd: float = 0.0
    zero_cache_cost_usd: float = 0.0
    full_cache_cost_usd: float = 0.0
    savings_at_observed_usd: float = 0.0
    savings_at_observed_pct: float = 0.0
    #: USD saved per +10 percentage points of cache hit rate.
    marginal_usd_per_10pp: float = 0.0
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Time series / stacked bars
# --------------------------------------------------------------------------- #
class SeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: datetime
    index: int
    model_id: str
    strategy_id: str
    tokens: int = 0
    billable_tokens: int = 0
    cost_usd: float = 0.0
    cumulative_tokens: int = 0
    cumulative_billable_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    cumulative_cache_hit_rate: float = 0.0


class StackedBarRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    strategy_id: str
    uncached_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cells: int = 0


# --------------------------------------------------------------------------- #
# Strategy ablation deltas
# --------------------------------------------------------------------------- #
class StrategyDeltaRow(BaseModel):
    """'Strategy X costs N% more than baseline Y' — the headline ablation table."""

    model_config = ConfigDict(extra="forbid")

    baseline_strategy: str
    strategy_id: str
    model_id: str | None = None
    baseline_billable_per_task: float = 0.0
    strategy_billable_per_task: float = 0.0
    token_delta_pct: float = 0.0
    baseline_cost_per_task_usd: float = 0.0
    strategy_cost_per_task_usd: float = 0.0
    cost_delta_pct: float = 0.0
    baseline_quality: float = 0.0
    quality: float = 0.0
    quality_delta: float = 0.0
    tokens_per_quality_delta: float | None = None
    verdict: str = ""


# --------------------------------------------------------------------------- #
# Root payload
# --------------------------------------------------------------------------- #
class ReportPayload(BaseModel):
    """Complete, self-contained report document."""

    model_config = ConfigDict(extra="forbid")

    meta: ReportMeta
    summary: RunSummary = Field(default_factory=RunSummary)
    leaderboard: list[EfficiencyRow] = Field(default_factory=list)
    categories: list[CategoryRow] = Field(default_factory=list)
    pareto: ParetoFrontier = Field(default_factory=ParetoFrontier)
    retry: RetryAnalysis = Field(default_factory=RetryAnalysis)
    cache: CacheSensitivity = Field(default_factory=CacheSensitivity)
    series: list[SeriesPoint] = Field(default_factory=list)
    stacked: list[StackedBarRow] = Field(default_factory=list)
    strategy_deltas: list[StrategyDeltaRow] = Field(default_factory=list)
    manifest: RunManifest | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


ChartKind = Literal["pareto", "retry", "cache", "series", "stacked", "leaderboard"]


__all__ = [
    "CachePoint",
    "CacheSensitivity",
    "CategoryRow",
    "ChartKind",
    "EfficiencyRow",
    "ParetoFrontier",
    "ParetoPoint",
    "ReportMeta",
    "ReportPayload",
    "RetryAnalysis",
    "RetryRow",
    "RunSummary",
    "SCHEMA_VERSION",
    "SeriesPoint",
    "StackedBarRow",
    "StrategyDeltaRow",
]
