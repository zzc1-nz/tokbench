"""Core data contracts for tokbench.

Every other module (runner, adapters, analysis, renderers, panel) speaks in
these types.  They are deliberately *serialisable and self-describing* so a run
directory is a complete, replayable artefact: JSONL call log + task results +
manifest.

Token accounting convention (important for the cost model):

``Usage.input_tokens`` is the **total** prompt token count reported by the
provider, *including* cached prompt tokens.  ``cache_read_tokens`` and
``cache_write_tokens`` are subsets of it.  The uncached (full-price) portion is
therefore ``input_tokens - cache_read_tokens - cache_write_tokens``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


# --------------------------------------------------------------------------- #
# Usage / cost primitives
# --------------------------------------------------------------------------- #
class Usage(BaseModel):
    """Token counters for one API call (or a sum of calls).

    ``extra="ignore"`` is deliberate: the ``@computed_field`` properties below
    (``uncached_input_tokens``, ``total_tokens``, ``billable_tokens``,
    ``cache_hit_rate``) are written into ``model_dump()``/JSON by design so the
    derived numbers are visible in the call log, but they must not be required
    when re-validating that same JSONL on read. Forbidding extras here would
    make every run directory unreadable by its own analysis layer.
    """

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uncached_input_tokens(self) -> int:
        return max(self.input_tokens - self.cache_read_tokens - self.cache_write_tokens, 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """All tokens that crossed the wire (cache reads included)."""
        return self.input_tokens + self.output_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def billable_tokens(self) -> int:
        """Tokens after the cache discount is applied to the input side.

        Used for "token efficiency" ranking, because a cache hit really is
        cheaper in both money *and* effective context spend.
        """
        return self.uncached_input_tokens + self.output_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_eligible_tokens(self) -> int:
        """Prompt tokens that *could* have been served from cache.

        A cache write is, by definition, a miss that populates the cache — it is
        not eligible for a read on the same call.  So the denominator of a hit
        rate is ``input_tokens - cache_write_tokens``, not ``input_tokens``.
        Counting writes as misses would make a run that aggressively *warms* a
        cache look worse than one that never caches at all.
        """
        return max(self.input_tokens - self.cache_write_tokens, 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cache-eligible prompt tokens served from cache.

        This is the single definition used everywhere — summary KPIs, the
        time-series running ratio, and the cache-sensitivity curve — so the same
        run can never report two different hit rates in one report.
        """
        eligible = self.cache_eligible_tokens
        if eligible <= 0:
            return 0.0
        return min(self.cache_read_tokens / eligible, 1.0)

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):  # pragma: no cover - defensive
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __radd__(self, other: object) -> Usage:
        if other == 0:  # sum() seed
            return self.model_copy()
        if isinstance(other, Usage):
            return self + other
        return NotImplemented  # pragma: no cover

    def scaled(self, factor: float) -> Usage:
        return Usage(
            input_tokens=int(round(self.input_tokens * factor)),
            output_tokens=int(round(self.output_tokens * factor)),
            cache_read_tokens=int(round(self.cache_read_tokens * factor)),
            cache_write_tokens=int(round(self.cache_write_tokens * factor)),
            reasoning_tokens=int(round(self.reasoning_tokens * factor)),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


# --------------------------------------------------------------------------- #
# Status / quality
# --------------------------------------------------------------------------- #
class CallStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    SKIPPED = "skipped"

    @property
    def succeeded(self) -> bool:
        return self is CallStatus.OK


class GraderKind(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    HEURISTIC = "heuristic"
    LLM_JUDGE = "llm_judge"
    NONE = "none"


class QualityScore(BaseModel):
    """Normalised quality signal in ``[0, 1]`` for one model output."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0, default=0.0)
    passed: bool = False
    grader: GraderKind = GraderKind.NONE
    rationale: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# One API call
# --------------------------------------------------------------------------- #
class CallRecord(BaseModel):
    """A single HTTP call attempt — the atomic unit of the JSONL call log."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    experiment: str
    model_id: str
    provider: str
    strategy_id: str
    task_id: str
    category: str
    repeat: int = 0
    attempt: int = 1
    is_final_attempt: bool = True
    status: CallStatus = CallStatus.OK
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime = Field(default_factory=utcnow)
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    finish_reason: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
    output_text: str = ""
    output_sha: str = ""
    prompt_tokens_est: int = 0
    quality: QualityScore | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def is_retry(self) -> bool:
        return self.attempt > 1


# --------------------------------------------------------------------------- #
# Logical unit: model x strategy x task x repeat (may contain retries)
# --------------------------------------------------------------------------- #
class TaskResult(BaseModel):
    """Aggregated outcome of all attempts for one experiment cell."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    experiment: str
    model_id: str
    strategy_id: str
    task_id: str
    category: str
    repeat: int = 0
    attempts: int = 1
    final_status: CallStatus = CallStatus.OK
    usage_total: Usage = Field(default_factory=Usage)
    usage_first_attempt: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    quality: QualityScore | None = None
    record_ids: list[str] = Field(default_factory=list)

    @property
    def retry_overhead_usage(self) -> Usage:
        """Tokens burned purely because the first attempt failed."""
        t = self.usage_total
        f = self.usage_first_attempt
        return Usage(
            input_tokens=max(t.input_tokens - f.input_tokens, 0),
            output_tokens=max(t.output_tokens - f.output_tokens, 0),
            cache_read_tokens=max(t.cache_read_tokens - f.cache_read_tokens, 0),
            cache_write_tokens=max(t.cache_write_tokens - f.cache_write_tokens, 0),
            reasoning_tokens=max(t.reasoning_tokens - f.reasoning_tokens, 0),
        )

    @property
    def retried(self) -> bool:
        return self.attempts > 1


# --------------------------------------------------------------------------- #
# Run-level manifest
# --------------------------------------------------------------------------- #
class RunManifest(BaseModel):
    """Everything needed to interpret a run directory without the config file."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_name: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    engine_version: str = "0.1.0"
    host: str = ""
    python: str = ""
    config_hash: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    planned_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    retried_cells: int = 0
    total_usage: Usage = Field(default_factory=Usage)
    total_cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    seed: int | None = None
    adapter_mode: str = "mock"  # "live" | "mock" | "replay"
    notes: list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.planned_calls <= 0:
            return 0.0
        return max(0.0, 1.0 - self.failed_calls / self.planned_calls)


# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    """One benchmark item loaded from ``tasks/*.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str = "general"
    prompt: str
    reference: str | None = None
    must_include: list[str] = Field(default_factory=list)
    tool_payload: str | None = None
    max_output_tokens: int | None = None
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "unnamed"
    version: str = "1"
    tasks: list[Task] = Field(default_factory=list)
    source_path: str = ""

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.tasks)

    @property
    def categories(self) -> list[str]:
        return sorted({t.category for t in self.tasks})


__all__ = [
    "CallRecord",
    "CallStatus",
    "GraderKind",
    "QualityScore",
    "RunManifest",
    "Task",
    "TaskResult",
    "TaskSet",
    "Usage",
    "short_hash",
    "utcnow",
]
