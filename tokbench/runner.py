"""Execution engine: task loading, cell scheduling, retry, budget, persistence.

This is the only module that knows how the pieces fit together.  Its job is to
turn an :class:`~tokbench.config.ExperimentConfig` plus a task set into a *run
directory* that is a complete, replayable artefact — and to do so in a way that
survives the two things that always happen in a benchmark: a provider misbehaves,
and the budget runs out mid-flight.

Design decisions and their reasons:

**Append-only persistence, written as work completes.**  Every attempt is a
:class:`~tokbench.models.CallRecord` appended to ``calls.jsonl`` the moment it
finishes.  A crashed or budget-aborted run therefore still leaves a valid partial
dataset, and no completed (and paid-for) call is ever lost to an in-memory buffer.

**Retry is a runner concern, not an adapter concern.**  Adapters normalise
failures into :class:`~tokbench.errors.ErrorKind`; the runner decides whether to
retry, and every attempt is recorded.  This is what makes the retry-overhead
analysis possible: the wasted tokens are visible in the log instead of being
hidden inside an SDK's internal retry loop.

**Budget breaches stop scheduling, they do not fail the run.**  A ceiling is a
normal outcome of an experiment ("we spent what we said we would"), so the engine
drains in-flight work, writes a manifest with ``aborted_reason``, and returns
normally.

**A cell failure never crashes the run.**  Exceptions are caught per cell and
converted into a ``TaskResult`` with a failed status, because one bad prompt must
not destroy the other 749 measurements.

**Concurrency without nondeterministic logs.**  Work is submitted to a
``ThreadPoolExecutor`` for throughput, but every store append is serialised under
one lock and stamped with a monotonic sequence number in ``record_id``, so the
log order is a deterministic function of completion order rather than a race.
"""

from __future__ import annotations

import contextlib
import json
import platform
import random
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.base import AdapterRequest, AdapterResponse, BaseAdapter
from .adapters.registry import get_adapter, mode_notes, resolve_mode
from .config import Cell, ExperimentConfig, ModelSpec, RetryPolicy
from .errors import AdapterError, ConfigError, ErrorKind, TaskSetError
from .grader import grade
from .models import (
    CallRecord,
    CallStatus,
    QualityScore,
    RunManifest,
    Task,
    TaskResult,
    TaskSet,
    Usage,
    short_hash,
    utcnow,
)
from .pricing import PriceBook, load_price_book
from .prompt import AssembledPrompt, assemble_prompt
from .store import RunStore, new_run_id

ENGINE_VERSION = "0.1.0"

#: Default output root when neither the caller nor the config names one.
DEFAULT_OUTPUT_BASE = "reports"


# --------------------------------------------------------------------------- #
# Task-set loading
# --------------------------------------------------------------------------- #
def load_taskset(path: str | Path, limit: int | None = None) -> TaskSet:
    """Load and validate a task set from JSONL or a JSON document.

    Two formats are accepted because both are natural for benchmark data:

    * **JSONL** — one :class:`~tokbench.models.Task` object per line (the
      streaming-friendly default for ``tasks/*.jsonl``).
    * **JSON** — either ``{"tasks": [...]}`` or a bare ``[...]`` list, optionally
      with ``name``/``version`` metadata.

    Args:
        path: file to read.
        limit: keep only the first ``limit`` tasks (applied *after* validation so
            a truncated file is still an error).

    Raises:
        TaskSetError: file missing, unparseable, empty, or containing an invalid
            task.  A benchmark that silently skipped a malformed line would
            produce incomparable results, so every defect is fatal here.

    Note:
        Task ids must be unique: reports and resume logic key off them, and a
        duplicate would make two different cells look like one.

        Files are read as ``utf-8-sig`` so a leading byte-order mark is stripped.
        Windows editors (Notepad, ``Set-Content``, some spreadsheet exports)
        routinely emit a BOM, and a BOM would otherwise make the very first line
        unparseable — a failure that looks like corrupt data but is really just an
        encoding artefact.
    """
    p = Path(path)
    if not p.exists():
        raise TaskSetError(f"task set not found: {p}")

    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TaskSetError(f"{p.name}: not valid UTF-8 ({exc}); re-save the file as UTF-8") from exc
    if not text.strip():
        raise TaskSetError(f"task set is empty: {p}")

    name = p.stem
    version = "1"
    raw_tasks: list[Any]

    if p.suffix.lower() == ".jsonl":
        raw_tasks = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw_tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TaskSetError(f"{p.name}:{lineno}: invalid JSON: {exc}") from exc
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TaskSetError(f"{p.name}: invalid JSON: {exc}") from exc
        if isinstance(payload, dict):
            raw_tasks = list(payload.get("tasks") or [])
            name = str(payload.get("name") or name)
            version = str(payload.get("version") or version)
        elif isinstance(payload, list):
            raw_tasks = payload
        else:
            raise TaskSetError(
                f"{p.name}: expected a JSON list or an object with a 'tasks' key, "
                f"got {type(payload).__name__}"
            )

    if not raw_tasks:
        raise TaskSetError(f"task set contains no tasks: {p}")

    tasks: list[Task] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise TaskSetError(f"{p.name}: task #{index} is not a JSON object")
        try:
            task = Task.model_validate(item)
        except Exception as exc:
            raise TaskSetError(f"{p.name}: task #{index} is invalid: {exc}") from exc
        if task.id in seen:
            raise TaskSetError(f"{p.name}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)

    if limit is not None:
        if limit < 1:
            raise TaskSetError(f"limit must be >= 1, got {limit}")
        tasks = tasks[:limit]

    return TaskSet(name=name, version=version, tasks=tasks, source_path=str(p.resolve()))


# --------------------------------------------------------------------------- #
# Progress / outcome DTOs
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ProgressEvent:
    """One progress tick, emitted after each cell finishes.

    Deliberately a plain dataclass (not a pydantic model): it is a *signal*, not
    persisted data, and the CLI/panel should be able to consume it with no import
    cost.
    """

    index: int
    total: int
    model_id: str
    strategy_id: str
    task_id: str
    status: CallStatus
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    message: str = ""

    @property
    def fraction(self) -> float:
        """Completion ratio in ``[0, 1]`` for progress bars."""
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.index / self.total))


@dataclass(slots=True)
class RunOutcome:
    """Everything a caller needs after :func:`run_experiment` returns."""

    run_id: str
    store: RunStore
    manifest: RunManifest
    planned_cells: int
    completed_cells: int
    aborted_reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self.aborted_reason is not None

    @property
    def run_dir(self) -> Path:
        return self.store.root


#: Callback signature for progress reporting; an alias purely for readability.
ProgressCallback = Callable[[ProgressEvent], None]


# --------------------------------------------------------------------------- #
# Retry policy helper
# --------------------------------------------------------------------------- #
class RetryDecision:
    """Answers "should this failure be retried, and after how long?".

    Encapsulated as a class (rather than inline in the cell worker) because the
    retry policy is a *measured* behaviour of the experiment: the tests assert on
    it directly, and the report's retry section depends on it behaving identically
    for every adapter.
    """

    def __init__(self, policy: RetryPolicy, *, rng: random.Random) -> None:
        self.policy = policy
        self._rng = rng
        self._retry_on = {k.strip().lower() for k in policy.retry_on if k.strip()}

    def should_retry(self, kind: ErrorKind, attempt: int) -> bool:
        """Whether ``attempt`` (1-based) may be retried after failing with ``kind``.

        A kind is retried when it is inherently retryable *or* explicitly listed
        in ``policy.retry_on``.  The explicit list exists so an experiment can opt
        into retrying e.g. ``content_filter`` if its provider is flaky about it,
        without changing the global taxonomy.
        """
        if attempt >= self.policy.max_attempts:
            return False
        return kind.retryable or kind.value in self._retry_on

    def delay_s(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped.

        Full jitter (``uniform(0, backoff)``) is used instead of fixed backoff
        because it de-synchronises concurrent workers after a shared outage — a
        thundering herd would otherwise turn one rate-limit event into a cascade
        that dominates the retry-cost metric it is supposed to measure.
        """
        base = self.policy.backoff_base_s * (2 ** max(attempt - 1, 0))
        capped = min(base, self.policy.backoff_max_s)
        if capped <= 0:
            return 0.0
        if not self.policy.jitter:
            return capped
        return self._rng.uniform(0.0, capped)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BudgetLedger:
    """Running budget ledger, guarded by the engine's lock.

    The engine charges *after* each completed call rather than before: a call
    that has already been paid for must be recorded even if it tips the run over
    the ceiling, otherwise the manifest would under-report real spend.
    """

    max_calls: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    breached: str | None = None

    def record(self, usage: Usage, cost_usd: float) -> None:
        self.calls += 1
        self.tokens += usage.total_tokens
        self.cost_usd += cost_usd

    def check(self) -> str | None:
        """Return a human-readable breach reason, or ``None`` if within budget."""
        if self.breached is not None:
            return self.breached
        if self.max_calls is not None and self.calls >= self.max_calls:
            self.breached = f"budget: max_calls reached ({self.calls}/{self.max_calls})"
        elif self.max_total_tokens is not None and self.tokens >= self.max_total_tokens:
            self.breached = (
                f"budget: max_total_tokens reached ({self.tokens}/{self.max_total_tokens})"
            )
        elif self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            self.breached = (
                f"budget: max_cost_usd reached (${self.cost_usd:.6f}/${self.max_cost_usd:.6f})"
            )
        return self.breached


# --------------------------------------------------------------------------- #
# Per-cell inputs
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _CellContext:
    """Immutable per-cell inputs shared by the worker (keeps signatures short)."""

    cell: Cell
    prompt: AssembledPrompt
    request: AdapterRequest
    adapter: BaseAdapter


@dataclass(slots=True)
class _AttemptOutcome:
    """Result of one attempt, before the engine decides whether it is final."""

    record: CallRecord
    response: AdapterResponse | None
    error: AdapterError | None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RunEngine:
    """Mutable state for one :func:`run_experiment` invocation.

    A class (rather than closures) because the state — store, budget, sequence
    counter, abort flag, adapters — is genuinely shared between the scheduler and
    the worker threads, and naming it makes the locking discipline auditable.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        store: RunStore,
        price_book: PriceBook,
        manifest: RunManifest,
        progress: ProgressCallback | None,
        keep_outputs: bool,
    ) -> None:
        self.config = config
        self.store = store
        self.price_book = price_book
        self.manifest = manifest
        self.progress = progress
        self.keep_outputs = keep_outputs

        self._lock = threading.Lock()
        self._adapter_lock = threading.Lock()
        self._seq = 0
        self._abort_reason: str | None = None
        self.retry = RetryDecision(config.retry, rng=random.Random(config.experiment.seed))
        budget = config.budget
        self.budget = BudgetLedger(
            max_calls=budget.max_calls,
            max_total_tokens=budget.max_total_tokens,
            max_cost_usd=budget.max_cost_usd,
        )
        #: Aggregates for the manifest, updated under ``_lock``.
        self.total_usage = Usage()
        self.total_cost_usd = 0.0
        self.completed_calls = 0
        self.failed_calls = 0
        self.retried_cells = 0
        self.completed_cells = 0
        self._adapters: dict[str, BaseAdapter] = {}

    # ------------------------------------------------------------- scheduling
    @property
    def aborted(self) -> bool:
        """Whether new cells (and new retries) should stop being scheduled."""
        return self._abort_reason is not None

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def abort(self, reason: str) -> None:
        """Set the abort reason once; later calls do not overwrite the first."""
        with self._lock:
            if self._abort_reason is None:
                self._abort_reason = reason

    def next_sequence(self) -> int:
        """Monotonic, gap-free sequence number used to order ``record_id``s."""
        with self._lock:
            self._seq += 1
            return self._seq

    def price_key(self, model: ModelSpec) -> str:
        """Price-book lookup key for a model.

        The *provider* model name is what vendors publish prices under, while the
        model ``id`` is the experiment-local alias; pricing by the alias would
        silently miss the price book whenever they differ.
        """
        return model.price_key

    def adapter_for(self, model: ModelSpec) -> BaseAdapter:
        """Create (once) and return the adapter for ``model``.

        One adapter per model keeps ``httpx`` connection pooling effective across
        the thread pool and makes ``close()`` at the end of the run trivial.
        """
        with self._adapter_lock:
            adapter = self._adapters.get(model.id)
            if adapter is not None:
                return adapter
            _effective, note = resolve_mode(model, self.config.adapter_mode)
            if note:
                self._add_note(note)
            adapter = get_adapter(
                model,
                mode=self.config.adapter_mode,
                seed=self.config.experiment.seed,
            )
            self._adapters[model.id] = adapter
            return adapter

    def close_adapters(self) -> None:
        """Close every adapter exactly once (idempotent, never raises)."""
        with self._adapter_lock:
            adapters = list(self._adapters.values())
            self._adapters.clear()
        for adapter in adapters:
            with contextlib.suppress(Exception):  # pragma: no cover - close must not break a run
                adapter.close()

    def _add_note(self, note: str) -> None:
        """Append a manifest note without duplicating it."""
        with self._lock:
            if note not in self.manifest.notes:
                self.manifest.notes.append(note)

    # --------------------------------------------------------------- writes
    def append_call(self, record: CallRecord) -> None:
        """Persist one attempt and fold it into the run aggregates."""
        with self._lock:
            self.store.append_call(record)
            self.completed_calls += 1
            if not record.status.succeeded:
                self.failed_calls += 1
            self.total_usage = self.total_usage + record.usage
            self.total_cost_usd += record.cost_usd

    def append_result(self, result: TaskResult) -> None:
        """Persist one cell result and update the cell counters."""
        with self._lock:
            self.store.append_result(result)
            self.completed_cells += 1
            if result.retried:
                self.retried_cells += 1

    def charge_budget(self, usage: Usage, cost_usd: float) -> str | None:
        """Charge the budget ledger; abort the run if a ceiling is newly hit."""
        with self._lock:
            self.budget.record(usage, cost_usd)
            reason = self.budget.check()
        if reason is not None:
            self.abort(reason)
        return reason

    def emit(self, event: ProgressEvent) -> None:
        """Invoke the progress callback defensively.

        A broken progress bar must never abort a run that is otherwise healthy, so
        callback errors are swallowed rather than propagating into the pool.
        """
        if self.progress is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - an observer must not break a run
            self.progress(event)

    # -------------------------------------------------------------- manifest
    def snapshot_manifest(self, *, finished: bool, wall_clock_s: float) -> RunManifest:
        """Copy current aggregates into the manifest (the caller writes it)."""
        with self._lock:
            self.manifest.completed_calls = self.completed_calls
            self.manifest.failed_calls = self.failed_calls
            self.manifest.retried_cells = self.retried_cells
            self.manifest.total_usage = self.total_usage.model_copy()
            self.manifest.total_cost_usd = round(self.total_cost_usd, 10)
            self.manifest.wall_clock_s = round(wall_clock_s, 3)
            if finished:
                self.manifest.finished_at = utcnow()
                if self._abort_reason:
                    self.manifest.notes.append(self._abort_reason)
            return self.manifest


# --------------------------------------------------------------------------- #
# One attempt
# --------------------------------------------------------------------------- #
def _build_request(cell: Cell, prompt: AssembledPrompt, *, cache_key: str) -> AdapterRequest:
    """Translate an assembled prompt into the adapter's request DTO.

    ``cache_key`` is the cell identity (model|strategy|task|repeat): stable across
    processes, which is what lets the mock reproduce identical numbers and what a
    caching provider would use to recognise a reusable prefix.

    The task's grading targets are passed through ``params["hints"]`` /
    ``params["reference"]``.  This exists for the *mock* adapter, which simulates a
    model rather than being one: to simulate "a competent model answered this
    correctly" it needs to know what a correct answer contains, and the only place
    that information exists is the task's ``must_include`` / ``reference``.  Real
    adapters ignore these keys (they are not forwarded to any provider payload —
    each adapter merges only its own recognised options), so live behaviour is
    unchanged.  Model-level ``params`` are merged underneath, never clobbered.
    """
    applied = prompt.applied
    params: dict[str, Any] = dict(cell.model.params or {})
    params.update(
        {
            "hints": list(cell.task.must_include),
            "reference": cell.task.reference or "",
        }
    )
    return AdapterRequest(
        messages=prompt.messages,
        model=cell.model.model,
        max_output_tokens=int(applied.get("max_output_tokens") or 512),
        temperature=float(applied.get("temperature") or 0.0),
        params=params,
        cache_key=cache_key,
        prompt_tokens_est=prompt.total_prompt_tokens_est,
    )


def _make_record(
    *,
    seq: int,
    experiment: str,
    cell: Cell,
    attempt: int,
    status: CallStatus,
    started_at: Any,
    ended_at: Any,
    latency_ms: float,
    usage: Usage,
    cost_usd: float,
    prompt: AssembledPrompt,
    output_text: str = "",
    finish_reason: str | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    quality: QualityScore | None = None,
    usage_estimated: bool = False,
) -> CallRecord:
    """Assemble one :class:`CallRecord` for an attempt.

    ``record_id`` embeds a zero-padded monotonic sequence number, so the JSONL log
    is deterministically ordered even when produced by a thread pool: the id is
    the sort key, and it never collides across models or retries.
    """
    task = cell.task
    record_id = (
        f"{seq:06d}-{cell.model.id}-{cell.strategy.id}-"
        f"{getattr(task, 'id', 'task')}-r{cell.repeat}-a{attempt}"
    )
    tags = {
        "task_category": str(getattr(task, "category", "general")),
        "strategy_preset": cell.strategy.system_prompt_preset,
        "strategy_cot": "true" if cell.strategy.cot else "false",
        "strategy_few_shot": str(cell.strategy.few_shot),
    }
    if usage_estimated:
        tags["usage_estimated"] = "true"
    return CallRecord(
        record_id=record_id,
        experiment=experiment,
        model_id=cell.model.id,
        provider=cell.model.provider,
        strategy_id=cell.strategy.id,
        task_id=str(getattr(task, "id", "task")),
        category=str(getattr(task, "category", "general")),
        repeat=cell.repeat,
        attempt=attempt,
        is_final_attempt=True,  # corrected by the caller before persisting
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=round(latency_ms, 3),
        usage=usage,
        cost_usd=round(cost_usd, 10),
        finish_reason=finish_reason,
        error_kind=error_kind,
        error_message=error_message,
        output_text=output_text,
        output_sha=short_hash(output_text) if output_text else "",
        prompt_tokens_est=prompt.total_prompt_tokens_est,
        quality=quality,
        tags=tags,
    )


def _run_attempt(
    *,
    engine: RunEngine,
    cell: Cell,
    prompt: AssembledPrompt,
    request: AdapterRequest,
    adapter: BaseAdapter,
    attempt: int,
) -> _AttemptOutcome:
    """Execute exactly one attempt and build its (not yet persisted) record.

    The adapter is called outside the store lock: network I/O must never serialise
    the pool.  Cost is applied here, from the price book — never by the adapter.
    """
    seq = engine.next_sequence()
    started_at = utcnow()
    started_perf = time.perf_counter()
    response: AdapterResponse | None = None
    error: AdapterError | None = None

    try:
        response = adapter.complete(request)
    except AdapterError as exc:
        error = exc
    except Exception as exc:  # an adapter bug must not kill the cell
        error = BaseAdapter.wrap_error(exc, kind=ErrorKind.UNKNOWN)

    latency_ms = (time.perf_counter() - started_perf) * 1000.0
    ended_at = utcnow()

    if response is not None:
        usage = response.usage
        # The mock reports a simulated latency; prefer it over the measured wall
        # clock, which is meaningless (~0 ms) for a synthetic call.
        simulated = (response.extra or {}).get("latency_ms")
        if simulated is not None:
            latency_ms = float(simulated)
        cost = engine.price_book.cost(engine.price_key(cell.model), usage)
        record = _make_record(
            seq=seq,
            experiment=engine.config.experiment.name,
            cell=cell,
            attempt=attempt,
            status=CallStatus.OK,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost,
            prompt=prompt,
            output_text=response.text if engine.keep_outputs else "",
            finish_reason=response.finish_reason,
            quality=grade(cell.task, response.text),
            usage_estimated=response.usage_estimated,
        )
        return _AttemptOutcome(record=record, response=response, error=None)

    assert error is not None  # both failure branches above set it
    status = CallStatus.TIMEOUT if error.kind is ErrorKind.TIMEOUT else CallStatus.ERROR
    record = _make_record(
        seq=seq,
        experiment=engine.config.experiment.name,
        cell=cell,
        attempt=attempt,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        usage=Usage(),
        cost_usd=0.0,
        prompt=prompt,
        error_kind=error.kind.value,
        error_message=str(error.message)[:500],
    )
    return _AttemptOutcome(record=record, response=None, error=error)


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def _execute_cell(engine: RunEngine, ctx: _CellContext) -> TaskResult:
    """Run one cell with the configured retry policy.

    Contract:

    * every attempt produces its own ``CallRecord``, persisted immediately;
    * exactly one attempt is marked ``is_final_attempt`` — the last one actually
      made, which is decided *after* the retry/budget decision so a failed call
      that will be retried is never mislabelled;
    * ``usage_total`` sums every attempt, ``usage_first_attempt`` is attempt 1;
    * quality is graded on the final successful text;
    * a budget breach stops further attempts, but the cell still yields a valid
      ``TaskResult``.
    """
    cell = ctx.cell
    usage_total = Usage()
    usage_first: Usage | None = None
    record_ids: list[str] = []
    total_cost = 0.0
    total_latency = 0.0
    final_quality: QualityScore | None = None
    final_status = CallStatus.ERROR
    succeeded = False
    attempts = 0

    for attempt in range(1, engine.config.retry.max_attempts + 1):
        attempts = attempt
        outcome = _run_attempt(
            engine=engine,
            cell=cell,
            prompt=ctx.prompt,
            request=ctx.request,
            adapter=ctx.adapter,
            attempt=attempt,
        )
        record = outcome.record
        succeeded = outcome.error is None

        # Aggregate before the budget charge so the manifest totals include this
        # attempt even when it is the one that trips a ceiling.
        usage_total = usage_total + record.usage
        total_cost += record.cost_usd
        total_latency += record.latency_ms
        if attempt == 1:
            usage_first = record.usage
        engine.charge_budget(record.usage, record.cost_usd)

        retryable = (
            not succeeded
            and outcome.error is not None
            and engine.retry.should_retry(outcome.error.kind, attempt)
            and not engine.aborted
        )

        # Persist before backing off: the record of a failed (possibly billed)
        # attempt must not sit in memory while the worker sleeps for up to
        # ``backoff_max_s``.  ``is_final_attempt`` records the retry decision as
        # scheduled; if another worker trips the budget *during* the backoff the
        # retry is cancelled and this flag is one short of the truth — which is
        # why ``TaskResult.record_ids`` is the authoritative list of attempts.
        record.is_final_attempt = not retryable
        engine.append_call(record)
        record_ids.append(record.record_id)

        if succeeded:
            final_quality = record.quality
            final_status = CallStatus.OK
            break

        final_status = record.status
        if not retryable:
            break
        delay = engine.retry.delay_s(attempt)
        if delay > 0:
            time.sleep(delay)
        if engine.aborted:
            # The budget ran out while we waited: do not spend on the retry.
            final_status = CallStatus.BUDGET_EXCEEDED
            break

    if final_quality is None and not succeeded:
        # The cell never succeeded: report the budget as the *reason* when that is
        # what stopped it, otherwise the last real failure kind.  A failed cell
        # still gets a graded (zero-score) quality record so reports never have a
        # silent gap.
        if engine.aborted:
            final_status = CallStatus.BUDGET_EXCEEDED
        final_quality = grade(cell.task, "")

    result = TaskResult(
        result_id=(f"{cell.model.id}|{cell.strategy.id}|{cell.task_id}|r{cell.repeat}"),
        experiment=engine.config.experiment.name,
        model_id=cell.model.id,
        strategy_id=cell.strategy.id,
        task_id=cell.task_id,
        category=cell.category,
        repeat=cell.repeat,
        attempts=attempts,
        final_status=final_status,
        usage_total=usage_total,
        usage_first_attempt=usage_first or Usage(),
        latency_ms=round(total_latency, 3),
        cost_usd=round(total_cost, 10),
        quality=final_quality,
        record_ids=record_ids,
    )
    engine.append_result(result)
    return result


def _failed_result(engine: RunEngine, cell: Cell, message: str) -> TaskResult:
    """Build and persist a ``TaskResult`` for a cell that could not even be built.

    This is the last line of defence for the "a cell failure never crashes the
    run" contract: prompt assembly, adapter construction or scheduling may raise,
    and the run must still account for the cell instead of silently shrinking the
    planned matrix.
    """
    result = TaskResult(
        result_id=(f"{cell.model.id}|{cell.strategy.id}|{cell.task_id}|r{cell.repeat}"),
        experiment=engine.config.experiment.name,
        model_id=cell.model.id,
        strategy_id=cell.strategy.id,
        task_id=cell.task_id,
        category=cell.category,
        repeat=cell.repeat,
        attempts=0,
        final_status=CallStatus.ERROR,
        quality=QualityScore(
            score=0.0,
            passed=False,
            rationale=f"cell failed before any call: {message}",
        ),
    )
    engine.append_result(result)
    return result


def _prepare_cell(engine: RunEngine, cell: Cell) -> _CellContext:
    """Assemble the prompt and build the adapter request for one cell.

    Prompt assembly runs on the scheduler thread rather than inside the worker so
    a prompt-assembly bug fails fast and identically for every cell, and so the
    pure-but-not-free work is not repeated by a retry.
    """
    prompt = assemble_prompt(cell.task, cell.strategy, cell.model)
    cache_key = f"{cell.model.id}|{cell.strategy.id}|{cell.task_id}|r{cell.repeat}"
    return _CellContext(
        cell=cell,
        prompt=prompt,
        request=_build_request(cell, prompt, cache_key=cache_key),
        adapter=engine.adapter_for(cell.model),
    )


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def _resolve_output_base(config: ExperimentConfig, output_base: str | Path | None) -> Path:
    """Output root: explicit argument > config ``output.dir`` > default.

    Relative paths in the config are resolved against the config file's directory
    so a run behaves the same from any working directory.
    """
    if output_base is not None:
        return Path(output_base)
    configured = config.output.dir or DEFAULT_OUTPUT_BASE
    if Path(configured).is_absolute():
        return Path(configured)
    if config.config_path:
        return (Path(config.config_path).parent / configured).resolve()
    return Path(configured)


def _completed_cell_keys(resume_from: str | Path | None) -> set[str]:
    """Cell keys a previous run already completed successfully.

    Resuming is *additive*: the new run gets a new directory and a new manifest and
    only executes the missing cells.  Reusing the old directory would make the two
    manifests disagree about what the run contains, which is worse than a slightly
    larger output tree.
    """
    if resume_from is None:
        return set()
    source = Path(resume_from)
    if not source.exists():
        raise ConfigError(f"resume source not found: {source}")
    if not (source / "results.jsonl").exists():
        raise ConfigError(f"resume source has no results.jsonl: {source}")
    done: set[str] = set()
    for result in RunStore(source).read_results():
        if result.final_status is CallStatus.OK:
            done.add(f"{result.model_id}|{result.strategy_id}|{result.task_id}|r{result.repeat}")
    return done


def _check_prices(config: ExperimentConfig, price_book: PriceBook) -> None:
    """Fail fast when a model has no price entry.

    Discovering this mid-run would leave a half-written run directory behind, so
    the check happens before anything is created.
    """
    missing = [
        f"{model.id!r} (looked up as {model.price_key!r})"
        for model in config.models
        if not price_book.has(model.price_key)
    ]
    if missing:
        raise ConfigError(
            "no price entry for model(s) "
            + ", ".join(missing)
            + f"; add them to the price book (known: {sorted(price_book.models)[:8]}...)"
        )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_experiment(
    config: ExperimentConfig,
    *,
    taskset: TaskSet | None = None,
    run_id: str | None = None,
    output_base: str | Path | None = None,
    progress: ProgressCallback | None = None,
    resume_from: str | Path | None = None,
) -> RunOutcome:
    """Execute an experiment and return the finished run artefact.

    Args:
        config: the fully validated experiment declaration.
        taskset: tasks to run; loaded from ``config.taskset`` when omitted.
        run_id: directory name for this run; generated from a timestamp when
            omitted.
        output_base: root directory for run directories; defaults to the config's
            ``output.dir``.
        progress: optional callback invoked once per completed cell (used by the
            CLI progress bar and the Streamlit panel).
        resume_from: an existing run directory whose *successful* cells should be
            skipped.  Missing cells are executed into the new run directory.

    Returns:
        A :class:`RunOutcome` pointing at a run directory that always contains a
        valid ``manifest.json``, plus ``calls.jsonl`` / ``results.jsonl`` for every
        attempt that was actually made — including partial and aborted runs.

    Raises:
        ConfigError: the config and task set are inconsistent (e.g. no tasks, no
            price entry for a model).
        TaskSetError: the configured task file is missing or malformed.

    Notes:
        Budget breaches and per-cell failures are *not* raised: they are recorded
        as data.  A benchmark harness that threw away a half-finished, already
        paid-for run would be actively harmful.
    """
    if taskset is None:
        taskset = load_taskset(
            config.resolve_path(config.taskset.path),
            limit=config.taskset.limit,
        )
    if not taskset.tasks:
        raise TaskSetError("task set contains no tasks")

    cells = config.expanded_cells(list(taskset.tasks))
    if not cells:
        raise ConfigError("experiment expands to zero cells; check models/strategies")

    price_book = load_price_book(str(config.resolve_path(config.prices)) if config.prices else None)
    _check_prices(config, price_book)

    store = RunStore.create(
        _resolve_output_base(config, output_base), run_id or new_run_id(config.experiment.name)
    )
    started_perf = time.perf_counter()

    manifest = RunManifest(
        run_id=store.run_id,
        experiment_name=config.experiment.name,
        engine_version=ENGINE_VERSION,
        host=socket.gethostname(),
        python=platform.python_version(),
        config_hash=config.config_hash(),
        config_snapshot=config.model_dump(mode="json", exclude={"config_path"}),
        planned_calls=len(cells),
        seed=config.experiment.seed,
        adapter_mode=config.adapter_mode,
    )
    manifest.notes.extend(mode_notes(config.models, config.adapter_mode))
    store.write_config_snapshot(config.to_yaml())
    store.write_manifest(manifest)

    engine = RunEngine(
        config=config,
        store=store,
        price_book=price_book,
        manifest=manifest,
        progress=progress,
        keep_outputs=config.output.keep_outputs,
    )

    already_done = _completed_cell_keys(resume_from)
    pending = [cell for cell in cells if cell.cell_key not in already_done]
    total = len(pending)
    if already_done:
        engine._add_note(f"resumed: skipped {len(cells) - total} already-successful cells")

    if not pending:
        engine._add_note("nothing to do: every cell was already completed")
        finished = engine.snapshot_manifest(
            finished=True, wall_clock_s=time.perf_counter() - started_perf
        )
        store.write_manifest(finished)
        return RunOutcome(
            run_id=store.run_id,
            store=store,
            manifest=finished,
            planned_cells=len(cells),
            completed_cells=0,
            aborted_reason=None,
        )

    concurrency = max(1, min(config.experiment.concurrency, total))
    finished: RunManifest
    try:
        # A bounded in-flight window is what makes the budget a *ceiling* rather
        # than a post-mortem.  Submitting all cells up front (the naive approach)
        # means every one of them has already been handed to a worker by the time
        # the first budget check runs, so a max_calls=3 ceiling would still execute
        # the whole matrix.  Keeping at most ``concurrency`` cells in flight bounds
        # the overshoot to the work already running when the ceiling is crossed.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending_iter = iter(pending)
            in_flight: list[tuple[Cell, Any]] = []

            def fill_window() -> None:
                """Top the in-flight set back up to ``concurrency`` cells."""
                while len(in_flight) < concurrency and not engine.aborted:
                    cell = next(pending_iter, None)
                    if cell is None:
                        return
                    try:
                        ctx = _prepare_cell(engine, cell)
                    except Exception as exc:  # prompt/adapter construction failure
                        _failed_result(engine, cell, f"{type(exc).__name__}: {exc}")
                        continue
                    in_flight.append((cell, pool.submit(_execute_cell, engine, ctx)))

            fill_window()
            emitted = 0
            while in_flight:
                cell, future = in_flight.pop(0)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - worker must not leak
                    result = None
                    message = f"{type(exc).__name__}: {exc}"
                else:
                    message = engine.abort_reason or ""
                emitted += 1
                engine.emit(
                    ProgressEvent(
                        index=emitted,
                        total=total,
                        model_id=result.model_id if result else cell.model.id,
                        strategy_id=result.strategy_id if result else cell.strategy.id,
                        task_id=result.task_id if result else cell.task_id,
                        status=result.final_status if result else CallStatus.ERROR,
                        usage=result.usage_total if result else Usage(),
                        cost_usd=result.cost_usd if result else 0.0,
                        message=message,
                    )
                )
                fill_window()
    finally:
        engine.close_adapters()
        finished = engine.snapshot_manifest(
            finished=True, wall_clock_s=time.perf_counter() - started_perf
        )
        store.write_manifest(finished)

    return RunOutcome(
        run_id=store.run_id,
        store=store,
        manifest=finished,
        planned_cells=len(cells),
        completed_cells=engine.completed_cells,
        aborted_reason=engine.abort_reason,
    )


__all__ = [
    "BudgetLedger",
    "DEFAULT_OUTPUT_BASE",
    "ENGINE_VERSION",
    "ProgressCallback",
    "ProgressEvent",
    "RetryDecision",
    "RunEngine",
    "RunOutcome",
    "load_taskset",
    "run_experiment",
]
