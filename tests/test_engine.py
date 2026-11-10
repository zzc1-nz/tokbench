"""Tests for the execution engine: prompt assembly, grading, mock adapter, runner.

What is worth testing here, and why:

* **Determinism** — the project's central claim is that a run is reproducible.  If
  the mock adapter or the record ordering drifted between runs, every report
  number would be unfalsifiable, so determinism gets direct assertions rather than
  a comment.
* **Attribution monotonicity** — "more few-shot costs more tokens" must hold for
  *every* task, otherwise the strategy ablation table would have exceptions that
  are really bugs in prompt assembly.
* **Retry semantics** — that only retryable kinds are retried, that every attempt
  is logged, and that ``usage_first_attempt`` is genuinely attempt 1.
* **Budget abort** — that hitting a ceiling leaves a *valid* run directory with a
  manifest explaining itself, rather than an exception and a partial directory.
* **Grading spread** — that quality is not a constant, because a constant quality
  axis would make the cost/quality Pareto chart meaningless.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from tokbench.adapters.base import AdapterRequest, AdapterResponse, BaseAdapter
from tokbench.adapters.mock import MockAdapter
from tokbench.adapters.registry import (
    ADAPTERS,
    available_providers,
    get_adapter,
    has_credentials,
    mode_notes,
    resolve_mode,
)
from tokbench.config import (
    BudgetSpec,
    ExperimentConfig,
    ExperimentSpec,
    ModelSpec,
    OutputSpec,
    RetryPolicy,
    StrategySpec,
)
from tokbench.errors import AdapterError, ConfigError, ErrorKind, TaskSetError
from tokbench.grader import grade, resolve_kind
from tokbench.models import CallStatus, GraderKind, Task, Usage
from tokbench.prompt import (
    SYSTEM_PROMPTS,
    assemble_prompt,
    estimate_tokens,
    resolve_max_output_tokens,
    summarize_tool_payload,
)
from tokbench.runner import (
    RetryDecision,
    load_taskset,
    run_experiment,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: Workspace-local scratch root for tests.
#:
#: Why not pytest's ``tmp_path``: on this Windows sandbox, directories created via
#: ``tempfile.mkdtemp`` (which pytest's tmpdir fixture uses) are not listable by
#: the confined token, so every test using ``tmp_path`` errored during fixture
#: setup with ``PermissionError: [WinError 5]``.  Plain ``os.makedirs`` under the
#: workspace is readable, so the suite provisions its own scratch directories.  It
#: stays dependency-free and self-cleaning, and the tests are otherwise ordinary.
_SCRATCH_ROOT = Path(__file__).resolve().parent / ".scratch"


@pytest.fixture()
def tmp_path() -> Any:
    """Workspace-local replacement for pytest's ``tmp_path``.

    Yields a fresh empty directory and removes it afterwards.  Shadows the
    built-in fixture for this module only, so the test bodies read identically to
    a normal pytest suite.
    """
    root = _SCRATCH_ROOT / f"case-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def make_task(
    task_id: str = "t-1",
    *,
    category: str = "codegen",
    reference: str | None = None,
    must_include: list[str] | None = None,
    tool_payload: str | None = None,
    difficulty: float = 0.5,
) -> Task:
    return Task(
        id=task_id,
        category=category,
        prompt="Write a Python function that groups records by key and returns a dict.",
        reference=reference,
        must_include=must_include or [],
        tool_payload=tool_payload,
        difficulty=difficulty,
    )


def make_config(
    *,
    models: list[ModelSpec] | None = None,
    strategies: list[StrategySpec] | None = None,
    repeats: int = 1,
    concurrency: int = 2,
    budget: BudgetSpec | None = None,
    retry: RetryPolicy | None = None,
    adapter_mode: str = "mock",
    name: str = "test-experiment",
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSpec(name=name, seed=7, repeats=repeats, concurrency=concurrency),
        models=models
        or [
            ModelSpec(id="mock-small", provider="mock"),
            ModelSpec(id="mock-large", provider="mock"),
        ],
        strategies=strategies
        or [
            StrategySpec(id="baseline", system_prompt_preset="short"),
            StrategySpec(id="long", system_prompt_preset="long", cot=True),
        ],
        retry=retry or RetryPolicy(max_attempts=3, backoff_base_s=0.0, backoff_max_s=0.0),
        budget=budget or BudgetSpec(),
        output=OutputSpec(dir="reports"),
        adapter_mode=adapter_mode,  # type: ignore[arg-type]
    )


def write_taskset(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "tasks.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"t-{i}",
                "category": "codegen" if i % 2 else "sql",
                "prompt": f"Implement feature number {i} in Python with tests.",
                "must_include": ["def "],
                "difficulty": round(0.2 + 0.05 * (i % 12), 3),
            }
        )
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
class TestPromptAssembly:
    def test_system_prompt_presets_are_ordered_by_length(self) -> None:
        """The preset is a cost lever, so its lengths must actually be ordered."""
        lengths = [len(SYSTEM_PROMPTS[name]) for name in ("none", "short", "medium", "long")]
        assert lengths == sorted(lengths)
        assert lengths[0] == 0
        # 'long' must be a genuinely large prompt (~1200+ tokens by estimate).
        assert estimate_tokens(SYSTEM_PROMPTS["long"]) >= 1200

    def test_long_system_prompt_costs_more_than_short(self) -> None:
        task = make_task()
        model = ModelSpec(id="mock-medium", provider="mock")
        short = assemble_prompt(task, StrategySpec(id="s", system_prompt_preset="short"), model)
        long = assemble_prompt(task, StrategySpec(id="l", system_prompt_preset="long"), model)
        assert long.system_tokens > short.system_tokens
        assert long.total_prompt_tokens_est > short.total_prompt_tokens_est

    def test_more_few_shot_is_monotonically_more_expensive(self) -> None:
        """Every extra exemplar must add tokens — no task-dependent exceptions."""
        task = make_task(category="sql")
        model = ModelSpec(id="mock-medium", provider="mock")
        totals = [
            assemble_prompt(
                task, StrategySpec(id=f"fs{n}", few_shot=n), model
            ).total_prompt_tokens_est
            for n in range(0, 4)
        ]
        assert totals == sorted(totals)
        assert totals[-1] > totals[0]

    def test_few_shot_selection_is_category_aware_and_deterministic(self) -> None:
        task = make_task(category="sql")
        model = ModelSpec(id="mock-medium", provider="mock")
        strategy = StrategySpec(id="fs", few_shot=2)
        first = assemble_prompt(task, strategy, model)
        second = assemble_prompt(task, strategy, model)
        assert first.messages == second.messages
        assert first.applied["few_shot_applied"] == 2
        # The SQL exemplar should be preferred for a SQL task.
        assert any("sql" in example_id for example_id in first.applied["few_shot_ids"])

    def test_cot_adds_instruction_and_is_attributed(self) -> None:
        task = make_task()
        model = ModelSpec(id="mock-medium", provider="mock")
        off = assemble_prompt(task, StrategySpec(id="a", cot=False), model)
        on = assemble_prompt(task, StrategySpec(id="b", cot=True), model)
        assert on.breakdown["cot"] > 0
        assert off.breakdown["cot"] == 0
        assert on.total_prompt_tokens_est > off.total_prompt_tokens_est

    def test_tool_payload_compression_reduces_tokens(self) -> None:
        payload = json.dumps(
            {"files": [f"src/module_{i}.py" for i in range(120)], "stdout": "x" * 2000}
        )
        task = make_task(tool_payload=payload, category="tool-use")
        model = ModelSpec(id="mock-medium", provider="mock")
        verbatim = assemble_prompt(task, StrategySpec(id="v", compress_tool_output=False), model)
        compressed = assemble_prompt(task, StrategySpec(id="c", compress_tool_output=True), model)
        assert verbatim.tool_tokens > compressed.tool_tokens
        assert compressed.total_prompt_tokens_est < verbatim.total_prompt_tokens_est
        # The compression must be labelled in the prompt, not silent.
        user_message = compressed.messages[-1]["content"]
        assert "compressed" in user_message.lower()

    def test_breakdown_sums_to_total(self) -> None:
        """The explainability contract: no tokens may be unaccounted for."""
        task = make_task(tool_payload='{"a": 1}', category="tool-use")
        model = ModelSpec(id="mock-large", provider="mock")
        prompt = assemble_prompt(
            task,
            StrategySpec(id="x", system_prompt_preset="long", few_shot=2, cot=True),
            model,
        )
        assert sum(prompt.breakdown.values()) == prompt.total_prompt_tokens_est
        assert set(prompt.breakdown) >= {"system", "few_shot", "cot", "task", "tool"}

    def test_system_prompt_override_wins_over_preset(self) -> None:
        task = make_task()
        model = ModelSpec(id="mock-medium", provider="mock")
        prompt = assemble_prompt(
            task,
            StrategySpec(id="override", system_prompt_preset="long", system_prompt="Be terse."),
            model,
        )
        assert prompt.messages[0]["content"] == "Be terse."
        assert prompt.applied["system_source"] == "override"

    def test_none_preset_omits_system_message(self) -> None:
        prompt = assemble_prompt(
            make_task(),
            StrategySpec(id="bare", system_prompt_preset="none"),
            ModelSpec(id="mock-medium", provider="mock"),
        )
        assert all(m["role"] != "system" for m in prompt.messages)

    def test_max_output_tokens_resolution_order(self) -> None:
        model = ModelSpec(id="m", provider="mock", max_output_tokens=256)
        assert resolve_max_output_tokens(StrategySpec(id="a"), model) == 256
        assert (
            resolve_max_output_tokens(StrategySpec(id="b", max_output_tokens=1024), model) == 1024
        )

    def test_estimate_tokens_handles_cjk(self) -> None:
        latin = estimate_tokens("hello world this is a test")
        cjk = estimate_tokens("你好世界这是一个测试你好世界")
        assert cjk >= len("你好世界这是一个测试你好世界")
        assert latin >= 1
        assert estimate_tokens("") == 0

    def test_summarize_tool_payload_reports_shape(self) -> None:
        summary = summarize_tool_payload('{"files": ["a.py", "b.py"], "count": 2}')
        assert "compressed" in summary
        assert "files" in summary


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
class TestGrader:
    def test_exact_match_full_and_partial(self) -> None:
        task = make_task(reference="return x + 1")
        assert grade(task, "return x + 1").score == 1.0
        assert grade(task, "  Return X + 1 ").score == 1.0
        partial = grade(task, "return x + 2")
        assert 0.0 < partial.score < 1.0
        assert partial.passed is False

    def test_contains_scores_coverage(self) -> None:
        task = make_task(must_include=["def ", "return", "sorted("])
        full = grade(task, "def f(x):\n    return sorted(x)\n" * 3)
        half = grade(task, "def f(x):\n    pass\n" * 3)
        assert full.score > half.score
        assert full.passed is True
        assert half.passed is False
        assert "sorted(" in half.details["missing"]

    def test_heuristic_penalises_refusals(self) -> None:
        task = make_task()
        refusal = grade(task, "I'm sorry, but I cannot help with that request.")
        answer = grade(
            task,
            "Here is the implementation:\n\n```python\ndef f(x):\n    return x\n```\n\n"
            "It preserves order and does not mutate the input.\n\n"
            "Edge cases: empty input returns an empty dict.",
        )
        assert refusal.score < 0.3
        assert answer.score > refusal.score

    def test_heuristic_penalises_empty(self) -> None:
        assert grade(make_task(), "").score == 0.0

    def test_grader_kind_auto_resolution(self) -> None:
        assert resolve_kind(make_task(reference="SELECT 1")) is GraderKind.EXACT_MATCH
        assert resolve_kind(make_task(must_include=["def "])) is GraderKind.CONTAINS
        assert resolve_kind(make_task()) is GraderKind.HEURISTIC
        long_reference = "x" * 200
        assert resolve_kind(make_task(reference=long_reference)) is GraderKind.HEURISTIC

    def test_grade_is_deterministic(self) -> None:
        task = make_task(must_include=["def "])
        text = "def f(): pass\n\nSome explanation follows here in a second paragraph."
        assert grade(task, text).score == grade(task, text).score

    def test_explicit_kind_none(self) -> None:
        result = grade(make_task(), "anything", kind="none")
        assert result.grader is GraderKind.NONE

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            grade(make_task(), "x", kind="telepathy")


# --------------------------------------------------------------------------- #
# Mock adapter
# --------------------------------------------------------------------------- #
class TestMockAdapter:
    def _request(self, key: str = "k-1", **overrides: Any) -> AdapterRequest:
        defaults: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "You are a senior engineer."},
                {"role": "user", "content": "Write a function that sums a list."},
            ],
            "model": "mock-medium",
            "max_output_tokens": 512,
            "prompt_tokens_est": 400,
            "cache_key": key,
        }
        defaults.update(overrides)
        return AdapterRequest(**defaults)

    def test_identical_requests_produce_identical_responses(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=3)
        first = adapter.complete(self._request())
        second = adapter.complete(self._request())
        assert first.text == second.text
        assert first.usage == second.usage
        assert first.finish_reason == second.finish_reason

    def test_different_cache_keys_differ(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=3)
        a = adapter.complete(self._request("cell-a"))
        b = adapter.complete(self._request("cell-b"))
        assert (a.usage, a.text) != (b.usage, b.text)

    def test_two_adapters_with_same_seed_agree(self) -> None:
        spec = ModelSpec(id="mock-large")
        a = MockAdapter(spec, fail_rate=0.0, seed=11).complete(self._request())
        b = MockAdapter(spec, fail_rate=0.0, seed=11).complete(self._request())
        assert a.usage == b.usage

    def test_input_tokens_track_prompt_estimate(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=5)
        small = adapter.complete(self._request(prompt_tokens_est=200))
        large = adapter.complete(self._request("k-2", prompt_tokens_est=4000))
        assert large.usage.input_tokens > small.usage.input_tokens
        assert 180 <= small.usage.input_tokens <= 240

    def test_cache_counters_are_subsets_of_input(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=5)
        for index in range(20):
            usage = adapter.complete(self._request(f"key-{index}")).usage
            assert usage.cache_read_tokens <= usage.input_tokens
            assert usage.cache_write_tokens <= usage.input_tokens
            assert usage.cache_read_tokens + usage.cache_write_tokens <= usage.input_tokens

    def test_output_respects_max_output_tokens(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-large"), fail_rate=0.0, seed=9)
        for index in range(20):
            response = adapter.complete(self._request(f"cap-{index}", max_output_tokens=64))
            assert response.usage.output_tokens <= 64

    def test_injected_failures_are_retryable_kinds(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=1.0, seed=1)
        kinds = set()
        for index in range(30):
            with pytest.raises(AdapterError) as excinfo:
                adapter.complete(self._request(f"fail-{index}"))
            kinds.add(excinfo.value.kind)
        assert kinds
        assert all(kind.retryable for kind in kinds)
        assert kinds <= {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.SERVER_ERROR}

    def test_failure_rate_zero_never_fails(self) -> None:
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=1)
        for index in range(50):
            assert adapter.complete(self._request(f"ok-{index}")).text

    def test_failure_injection_is_deterministic(self) -> None:
        def failures(seed: int) -> list[bool]:
            adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.3, seed=seed)
            out = []
            for index in range(40):
                try:
                    adapter.complete(self._request(f"det-{index}"))
                except AdapterError:
                    out.append(True)
                else:
                    out.append(False)
            return out

        assert failures(4) == failures(4)

    def test_retried_cell_can_succeed_after_a_failure(self) -> None:
        """Regression: failure draws must vary per attempt, not per cell.

        Seeding the failure verdict on ``cache_key`` alone made every retry of a
        failing cell fail identically — retries were provably useless and the
        retry-overhead metric became a constant instead of a measurement.
        """
        recovered = 0
        for index in range(300):
            adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.05, seed=42)
            request = self._request(f"retry-{index}")
            outcomes: list[str] = []
            for _ in range(3):
                try:
                    adapter.complete(request)
                except AdapterError as exc:
                    outcomes.append(exc.kind.value)
                else:
                    outcomes.append("ok")
            if outcomes[0] != "ok" and outcomes[1] == "ok":
                recovered += 1
        assert recovered > 0, "a failed cell never succeeded on retry"

    def test_failure_rate_is_approximately_three_percent(self) -> None:
        """The headline retry data must actually materialise at scale."""
        adapter = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.03, seed=42)
        total, failures = 2000, 0
        kinds: set[str] = set()
        for index in range(total):
            try:
                adapter.complete(self._request(f"rate-{index}"))
            except AdapterError as exc:
                failures += 1
                kinds.add(exc.kind.value)
        rate = failures / total
        assert 0.01 <= rate <= 0.06, f"injected failure rate {rate:.3%} is off target"
        # All three transient classes must appear, or the retry analysis has no
        # variety to report.
        assert {"rate_limit", "timeout", "server_error"} <= kinds

    def test_successful_token_counts_unaffected_by_fail_rate(self) -> None:
        """Changing the failure rate must not change the body numbers."""
        request = self._request("body-stability")
        clean = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=42)
        expected = clean.complete(request)
        flaky = MockAdapter(ModelSpec(id="mock-medium"), fail_rate=0.0, seed=42)
        assert flaky.complete(request).usage == expected.usage

    def test_response_text_varies_by_tier(self) -> None:
        request = self._request("tier")
        lengths = {
            tier: len(
                MockAdapter(ModelSpec(id=f"mock-{tier}"), fail_rate=0.0, seed=2)
                .complete(request)
                .text
            )
            for tier in ("small", "medium", "large")
        }
        assert lengths["small"] < lengths["large"]

    def test_graded_quality_separates_tiers(self, tmp_path: Path) -> None:
        """Higher tiers must earn higher *graded* scores, not just longer text.

        This is the invariant the whole cost/quality analysis rests on: if every
        tier graded the same, the Pareto frontier would collapse to a single point
        and "is the expensive model worth it?" would have no answer.  Grading is the
        real grader, not a mock shortcut, so this also proves the mock's answers are
        task-aware enough for the grader to differentiate.
        """
        taskset = load_taskset(write_taskset(tmp_path, count=8))
        config = make_config(
            models=[
                ModelSpec(id="mock-small", provider="mock"),
                ModelSpec(id="mock-medium", provider="mock"),
                ModelSpec(id="mock-large", provider="mock"),
            ],
            strategies=[StrategySpec(id="baseline", system_prompt_preset="short")],
            concurrency=4,
            retry=RetryPolicy(max_attempts=1, backoff_base_s=0.0, backoff_max_s=0.0),
        )
        outcome = run_experiment(config, taskset=taskset, run_id="tiers", output_base=tmp_path)
        scores: dict[str, list[float]] = {}
        for result in outcome.store.read_results():
            if result.quality is not None:
                scores.setdefault(result.model_id, []).append(result.quality.score)
        means = {model: sum(v) / len(v) for model, v in scores.items()}
        assert means["mock-small"] < means["mock-large"], means
        assert means["mock-small"] <= means["mock-medium"] <= means["mock-large"], means
        # And the scores must not be a constant.
        assert len({round(s, 3) for v in scores.values() for s in v}) > 1


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_all_providers_registered(self) -> None:
        assert set(available_providers()) == {
            "mock",
            "openai",
            "anthropic",
            "deepseek",
            "ollama",
        }
        assert ADAPTERS["mock"] is MockAdapter

    def test_mock_mode_always_returns_mock(self) -> None:
        spec = ModelSpec(id="gpt-4o-mini", provider="openai")
        assert isinstance(get_adapter(spec, mode="mock"), MockAdapter)

    def test_auto_falls_back_to_mock_without_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        spec = ModelSpec(id="gpt-4o-mini", provider="openai", api_key=None)
        adapter = get_adapter(spec, mode="auto")
        assert isinstance(adapter, MockAdapter)
        provider, note = resolve_mode(spec, "auto")
        assert provider == "mock"
        assert note is not None and "OPENAI_API_KEY" in note

    def test_auto_goes_live_with_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        spec = ModelSpec(id="gpt-4o-mini", provider="openai")
        assert has_credentials(spec) is True
        adapter = get_adapter(spec, mode="auto")
        try:
            assert adapter.provider == "openai"
        finally:
            adapter.close()

    def test_live_mode_unknown_provider_raises(self) -> None:
        spec = ModelSpec(id="x", provider="telepathy")
        with pytest.raises(ConfigError):
            get_adapter(spec, mode="live")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ConfigError):
            get_adapter(ModelSpec(id="m"), mode="sideways")

    def test_mode_notes_only_reports_downgrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        notes = mode_notes(
            [ModelSpec(id="mock-small"), ModelSpec(id="ds", provider="deepseek")], "auto"
        )
        assert len(notes) == 1
        assert "deepseek" in notes[0]


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
class TestRetryDecision:
    def _policy(self, **overrides: Any) -> RetryPolicy:
        defaults: dict[str, Any] = {
            "max_attempts": 3,
            "backoff_base_s": 0.5,
            "backoff_max_s": 4.0,
            "jitter": False,
            "retry_on": ["rate_limit", "timeout", "server_error", "network"],
        }
        defaults.update(overrides)
        return RetryPolicy(**defaults)

    def test_retryable_kinds_retried(self) -> None:
        import random

        decision = RetryDecision(self._policy(), rng=random.Random(0))
        for kind in (
            ErrorKind.RATE_LIMIT,
            ErrorKind.TIMEOUT,
            ErrorKind.SERVER_ERROR,
            ErrorKind.NETWORK,
        ):
            assert decision.should_retry(kind, 1) is True

    def test_non_retryable_kinds_not_retried(self) -> None:
        import random

        decision = RetryDecision(self._policy(), rng=random.Random(0))
        for kind in (ErrorKind.AUTH, ErrorKind.BAD_REQUEST, ErrorKind.CONTEXT_OVERFLOW):
            assert decision.should_retry(kind, 1) is False

    def test_explicit_retry_on_extends_policy(self) -> None:
        import random

        decision = RetryDecision(self._policy(retry_on=["content_filter"]), rng=random.Random(0))
        assert decision.should_retry(ErrorKind.CONTENT_FILTER, 1) is True
        assert decision.should_retry(ErrorKind.AUTH, 1) is False

    def test_attempt_ceiling(self) -> None:
        import random

        decision = RetryDecision(self._policy(max_attempts=2), rng=random.Random(0))
        assert decision.should_retry(ErrorKind.RATE_LIMIT, 1) is True
        assert decision.should_retry(ErrorKind.RATE_LIMIT, 2) is False

    def test_backoff_is_exponential_and_capped(self) -> None:
        import random

        decision = RetryDecision(self._policy(), rng=random.Random(0))
        assert decision.delay_s(1) == pytest.approx(0.5)
        assert decision.delay_s(2) == pytest.approx(1.0)
        assert decision.delay_s(3) == pytest.approx(2.0)
        assert decision.delay_s(10) == pytest.approx(4.0)

    def test_jitter_stays_within_bounds(self) -> None:
        import random

        decision = RetryDecision(self._policy(jitter=True), rng=random.Random(1))
        for _ in range(50):
            delay = decision.delay_s(2)
            assert 0.0 <= delay <= 1.0


# --------------------------------------------------------------------------- #
# Task-set loading
# --------------------------------------------------------------------------- #
class TestLoadTaskset:
    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        path = write_taskset(tmp_path, count=3)
        taskset = load_taskset(path)
        assert len(taskset.tasks) == 3
        assert taskset.tasks[0].id == "t-0"
        assert taskset.source_path.endswith("tasks.jsonl")

    def test_limit_applies(self, tmp_path: Path) -> None:
        path = write_taskset(tmp_path, count=5)
        assert len(load_taskset(path, limit=2).tasks) == 2

    def test_json_document_with_tasks_key(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text(
            json.dumps(
                {
                    "name": "mini",
                    "version": "2",
                    "tasks": [{"id": "a", "prompt": "do it"}],
                }
            ),
            encoding="utf-8",
        )
        taskset = load_taskset(path)
        assert taskset.name == "mini"
        assert taskset.version == "2"
        assert len(taskset.tasks) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TaskSetError):
            load_taskset(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(TaskSetError):
            load_taskset(path)

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "prompt": "x"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(TaskSetError):
            load_taskset(path)

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "dupe.jsonl"
        path.write_text(
            '{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n',
            encoding="utf-8",
        )
        with pytest.raises(TaskSetError):
            load_taskset(path)

    def test_invalid_task_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.jsonl"
        path.write_text('{"id": "a"}\n', encoding="utf-8")  # prompt is required
        with pytest.raises(TaskSetError):
            load_taskset(path)

    def test_utf8_bom_jsonl_is_tolerated(self, tmp_path: Path) -> None:
        """Regression: Windows editors emit a BOM; it must not corrupt line 1.

        Notepad and ``Set-Content`` write ``\\ufeff`` before the first character.
        Read as plain UTF-8 that byte becomes part of the first line and the JSON
        parse fails — a failure that looks like corrupt data but is purely an
        encoding artefact.
        """
        path = tmp_path / "bom.jsonl"
        payload = '{"id": "bom-1", "prompt": "do the thing"}\n'
        path.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        taskset = load_taskset(path)
        assert len(taskset.tasks) == 1
        assert taskset.tasks[0].id == "bom-1"
        assert taskset.tasks[0].prompt == "do the thing"

    def test_utf8_bom_json_document_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.json"
        payload = json.dumps({"name": "bommed", "tasks": [{"id": "b1", "prompt": "x"}]})
        path.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
        taskset = load_taskset(path)
        assert taskset.name == "bommed"
        assert taskset.tasks[0].id == "b1"

    def test_non_utf8_file_reports_actionable_error(self, tmp_path: Path) -> None:
        path = tmp_path / "latin.jsonl"
        path.write_bytes('{"id": "a", "prompt": "café"}'.encode("latin-1"))
        with pytest.raises(TaskSetError) as excinfo:
            load_taskset(path)
        assert "UTF-8" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Runner — end to end
# --------------------------------------------------------------------------- #
class TestRunExperiment:
    def test_end_to_end_mock_run(self, tmp_path: Path) -> None:
        """2 models x 2 strategies x 3 tasks x 1 repeat leaves valid artefacts."""
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config()
        outcome = run_experiment(
            config, taskset=taskset, run_id="e2e", output_base=tmp_path / "runs"
        )

        assert outcome.planned_cells == 12
        assert outcome.completed_cells == 12
        assert outcome.aborted_reason is None

        store = outcome.store
        calls = store.read_calls()
        results = store.read_results()
        manifest = store.read_manifest()

        assert manifest is not None
        assert manifest.run_id == "e2e"
        assert manifest.planned_calls == 12
        assert manifest.finished_at is not None
        assert manifest.config_hash == config.config_hash()
        assert (store.root / "config.yaml").exists()

        assert len(results) == 12
        assert all(r.final_status is CallStatus.OK for r in results)
        assert all(r.quality is not None for r in results)
        # Retries mean more calls than cells; never fewer.
        assert len(calls) >= len(results)
        assert manifest.completed_calls == len(calls)
        assert manifest.total_cost_usd > 0.0
        assert manifest.total_usage.input_tokens > 0

        # Every record is well formed and references a real cell.
        for record in calls:
            assert record.experiment == config.experiment.name
            assert record.record_id
            assert record.cost_usd >= 0.0
            if record.status is CallStatus.OK:
                assert record.output_text
                assert record.usage.input_tokens > 0
                assert record.quality is not None

        # record_ids on results must point at real calls.
        known = {c.record_id for c in calls}
        for result in results:
            assert result.record_ids
            assert set(result.record_ids) <= known
            assert result.attempts == len(result.record_ids)

    def test_usage_first_attempt_is_attempt_one(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config(
            models=[ModelSpec(id="mock-medium", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
            retry=RetryPolicy(max_attempts=5, backoff_base_s=0.0, backoff_max_s=0.0),
        )
        outcome = run_experiment(config, taskset=taskset, run_id="retry", output_base=tmp_path)
        calls_by_cell: dict[str, list[Any]] = {}
        for record in outcome.store.read_calls():
            key = f"{record.model_id}|{record.strategy_id}|{record.task_id}|r{record.repeat}"
            calls_by_cell.setdefault(key, []).append(record)

        for result in outcome.store.read_results():
            key = f"{result.model_id}|{result.strategy_id}|{result.task_id}|r{result.repeat}"
            records = sorted(calls_by_cell[key], key=lambda r: r.attempt)
            assert result.attempts == len(records)
            assert result.usage_first_attempt == records[0].usage
            expected_total = sum((r.usage for r in records), Usage())
            assert result.usage_total == expected_total
            # Exactly one attempt per cell is final.
            assert sum(1 for r in records if r.is_final_attempt) == 1

    def test_budget_abort_still_writes_valid_manifest(self, tmp_path: Path) -> None:
        """A ceiling must truncate the run gracefully, not raise."""
        taskset = load_taskset(write_taskset(tmp_path, count=4))
        config = make_config(budget=BudgetSpec(max_calls=3), concurrency=1)
        outcome = run_experiment(config, taskset=taskset, run_id="budget", output_base=tmp_path)

        assert outcome.aborted_reason is not None
        assert "max_calls" in outcome.aborted_reason
        assert outcome.completed_cells < outcome.planned_cells
        manifest = outcome.store.read_manifest()
        assert manifest is not None
        assert manifest.finished_at is not None
        assert any("max_calls" in note for note in manifest.notes)
        assert outcome.store.read_calls()

    def test_token_budget_abort(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config(budget=BudgetSpec(max_total_tokens=1), concurrency=1)
        outcome = run_experiment(
            config, taskset=taskset, run_id="tokenbudget", output_base=tmp_path
        )
        assert outcome.aborted_reason is not None
        assert "max_total_tokens" in outcome.aborted_reason

    def test_cost_budget_abort(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config(budget=BudgetSpec(max_cost_usd=1e-9), concurrency=1)
        outcome = run_experiment(config, taskset=taskset, run_id="costbudget", output_base=tmp_path)
        assert outcome.aborted_reason is not None
        assert "max_cost_usd" in outcome.aborted_reason

    def test_cell_failure_does_not_crash_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An adapter raising a non-retryable error yields a failed cell, not a crash."""
        taskset = load_taskset(write_taskset(tmp_path, count=2))
        config = make_config(
            models=[ModelSpec(id="mock-medium", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
            retry=RetryPolicy(max_attempts=2, backoff_base_s=0.0, backoff_max_s=0.0),
        )

        class ExplodingAdapter(BaseAdapter):
            provider = "mock"

            def complete(self, request: AdapterRequest) -> AdapterResponse:
                raise AdapterError("boom", kind=ErrorKind.AUTH)

        monkeypatch.setattr(
            "tokbench.runner.get_adapter", lambda spec, **kwargs: ExplodingAdapter(spec)
        )
        outcome = run_experiment(config, taskset=taskset, run_id="boom", output_base=tmp_path)
        results = outcome.store.read_results()
        assert len(results) == 2
        assert all(r.final_status is CallStatus.ERROR for r in results)
        # AUTH is not retryable, so exactly one attempt per cell.
        assert all(r.attempts == 1 for r in results)
        assert outcome.manifest.failed_calls == 2
        assert outcome.aborted_reason is None

    def test_retryable_failure_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=1))
        config = make_config(
            models=[ModelSpec(id="mock-medium", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
            retry=RetryPolicy(max_attempts=3, backoff_base_s=0.0, backoff_max_s=0.0),
        )

        class FlakyAdapter(BaseAdapter):
            provider = "mock"
            calls = 0

            def complete(self, request: AdapterRequest) -> AdapterResponse:
                FlakyAdapter.calls += 1
                if FlakyAdapter.calls == 1:
                    raise AdapterError("rate limited", kind=ErrorKind.RATE_LIMIT)
                return AdapterResponse(
                    text="def f():\n    return 1\n\n" * 5,
                    usage=Usage(input_tokens=100, output_tokens=50),
                )

        monkeypatch.setattr(
            "tokbench.runner.get_adapter", lambda spec, **kwargs: FlakyAdapter(spec)
        )
        outcome = run_experiment(config, taskset=taskset, run_id="flaky", output_base=tmp_path)
        result = outcome.store.read_results()[0]
        assert result.attempts == 2
        assert result.final_status is CallStatus.OK
        assert result.retried is True
        assert result.usage_total.input_tokens == 100
        assert result.usage_first_attempt.input_tokens == 0
        assert result.retry_overhead_usage.input_tokens == 100

        calls = outcome.store.read_calls()
        assert len(calls) == 2
        assert calls[0].error_kind == ErrorKind.RATE_LIMIT.value
        assert calls[0].is_final_attempt is False
        assert calls[1].is_final_attempt is True

    def test_determinism_across_runs(self, tmp_path: Path) -> None:
        """Same config + same seed => same token numbers, byte for byte."""
        taskset = load_taskset(write_taskset(tmp_path, count=2))

        def run(run_id: str) -> list[tuple[str, int, int, int]]:
            config = make_config(
                concurrency=1,
                models=[ModelSpec(id="mock-medium", provider="mock")],
                strategies=[StrategySpec(id="baseline")],
            )
            outcome = run_experiment(config, taskset=taskset, run_id=run_id, output_base=tmp_path)
            return sorted(
                (
                    record.task_id,
                    record.attempt,
                    record.usage.input_tokens,
                    record.usage.output_tokens,
                )
                for record in outcome.store.read_calls()
            )

        assert run("det-a") == run("det-b")

    def test_record_ids_are_ordered_and_unique(self, tmp_path: Path) -> None:
        """Sequence numbers must be unique and gap-free.

        Note the assertion is on the *set*, not on log order: workers append under
        a lock as they finish, so the JSONL order is completion order while the
        embedded sequence is allocation order.  Both properties matter — uniqueness
        prevents collisions, and gap-freeness proves every allocated id was written.
        """
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config(concurrency=4)
        outcome = run_experiment(config, taskset=taskset, run_id="ordered", output_base=tmp_path)
        calls = outcome.store.read_calls()
        ids = [c.record_id for c in calls]
        assert len(ids) == len(set(ids))
        sequences = sorted(int(record_id.split("-")[0]) for record_id in ids)
        assert sequences == list(range(1, len(ids) + 1))
        # Log order must still be a stable, replayable order (completion order).
        assert len(calls) == len(ids)

    def test_progress_events_are_emitted(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=2))
        events: list[Any] = []
        config = make_config(
            models=[ModelSpec(id="mock-medium", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
        )
        run_experiment(
            config,
            taskset=taskset,
            run_id="progress",
            output_base=tmp_path,
            progress=events.append,
        )
        assert len(events) == 2
        assert [e.index for e in events] == [1, 2]
        assert all(e.total == 2 for e in events)
        assert events[-1].fraction == pytest.approx(1.0)

    def test_missing_price_entry_raises_before_creating_run_dir(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=1))
        config = make_config(models=[ModelSpec(id="unpriced-model", provider="mock")])
        with pytest.raises(ConfigError):
            run_experiment(config, taskset=taskset, run_id="unpriced", output_base=tmp_path)
        assert not (tmp_path / "unpriced").exists()

    def test_resume_skips_completed_cells(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=2))
        config = make_config(
            models=[ModelSpec(id="mock-medium", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
        )
        first = run_experiment(config, taskset=taskset, run_id="first", output_base=tmp_path)
        assert first.completed_cells == 2

        resumed = run_experiment(
            config,
            taskset=taskset,
            run_id="second",
            output_base=tmp_path,
            resume_from=first.run_dir,
        )
        assert resumed.completed_cells == 0
        assert resumed.store.read_manifest() is not None
        assert any("nothing to do" in note for note in resumed.manifest.notes)

    def test_auto_mode_records_downgrade_notes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        taskset = load_taskset(write_taskset(tmp_path, count=1))
        config = make_config(
            # Priced model name, unpriced alias id: the note is about the provider
            # downgrade, and a price entry must exist for the run to start at all.
            models=[ModelSpec(id="ds-chat", provider="deepseek", model="deepseek-chat")],
            strategies=[StrategySpec(id="baseline")],
            adapter_mode="auto",
        )
        outcome = run_experiment(config, taskset=taskset, run_id="auto", output_base=tmp_path)
        manifest = outcome.store.read_manifest()
        assert manifest is not None
        assert any("deepseek" in note and "mock" in note for note in manifest.notes)
        assert manifest.adapter_mode == "auto"


# --------------------------------------------------------------------------- #
# Engine-level invariants
# --------------------------------------------------------------------------- #
class TestEngineInvariants:
    def test_cost_comes_from_price_book_not_adapter(self, tmp_path: Path) -> None:
        """Adapters return tokens only; the runner applies the price book."""
        taskset = load_taskset(write_taskset(tmp_path, count=2))
        config = make_config(
            models=[ModelSpec(id="mock-large", provider="mock")],
            strategies=[StrategySpec(id="baseline")],
        )
        outcome = run_experiment(config, taskset=taskset, run_id="pricing", output_base=tmp_path)
        from tokbench.pricing import load_price_book

        book = load_price_book(None)
        for record in outcome.store.read_calls():
            expected = book.cost("mock-large", record.usage)
            assert record.cost_usd == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_manifest_totals_match_call_log(self, tmp_path: Path) -> None:
        taskset = load_taskset(write_taskset(tmp_path, count=3))
        config = make_config()
        outcome = run_experiment(config, taskset=taskset, run_id="totals", output_base=tmp_path)
        calls = outcome.store.read_calls()
        manifest = outcome.store.read_manifest()
        assert manifest is not None
        assert manifest.total_usage == sum((c.usage for c in calls), Usage())
        assert manifest.total_cost_usd == pytest.approx(
            sum(c.cost_usd for c in calls), rel=1e-9, abs=1e-9
        )
        assert manifest.completed_calls == len(calls)
        assert manifest.failed_calls == sum(1 for c in calls if not c.status.succeeded)
        assert manifest.retried_cells == sum(1 for r in outcome.store.read_results() if r.retried)

    def test_retry_injection_produces_retries_at_scale(self, tmp_path: Path) -> None:
        """The ~3% injection must actually show up, or retry analysis has no data.

        The sample is deliberately large (5 models x 2 strategies x 40 tasks = 400
        cells).  At 3% per attempt a 60-cell run expects ~1.8 failures, so a
        zero-retry outcome there is a coin flip rather than a signal — the earlier
        version of this test was flaky for exactly that reason.  400 cells makes the
        expectation ~12 failures, so "no retries at all" would indicate a real
        regression instead of bad luck.
        """
        taskset = load_taskset(write_taskset(tmp_path, count=40))
        config = make_config(
            concurrency=8,
            models=[
                ModelSpec(id="mock-small", provider="mock"),
                ModelSpec(id="mock-medium", provider="mock"),
                ModelSpec(id="mock-large", provider="mock"),
                ModelSpec(id="deepseek-chat", provider="mock"),
                ModelSpec(id="gpt-4o-mini", provider="mock"),
            ],
            strategies=[StrategySpec(id="baseline"), StrategySpec(id="cot", cot=True)],
            retry=RetryPolicy(max_attempts=4, backoff_base_s=0.0, backoff_max_s=0.0),
        )
        outcome = run_experiment(config, taskset=taskset, run_id="scale", output_base=tmp_path)
        results = outcome.store.read_results()
        calls = outcome.store.read_calls()
        assert len(results) == 400

        retried = [r for r in results if r.retried]
        assert retried, "expected retries from the injected ~3% failure rate"
        failed = [c for c in calls if not c.status.succeeded]
        assert failed, "expected failed calls in the log"
        rate = len(failed) / len(calls)
        assert 0.005 <= rate <= 0.10, f"observed failure rate {rate:.2%} is off target"
        # Every injected kind must be represented across a sample this size.
        kinds = {c.error_kind for c in failed}
        assert kinds <= {"rate_limit", "timeout", "server_error"}
        for result in retried:
            assert result.retry_overhead_usage.total_tokens >= 0
            assert result.usage_total.total_tokens >= result.usage_first_attempt.total_tokens
