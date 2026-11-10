"""Contract tests for the benchmark task set and the shipped example configs.

These tests guard the *substrate* of every experiment: the task file, the two
example experiment configs, and the price-book overlay.  If any of them drifts,
every number in a report becomes uncomparable, so the assertions here are
deliberately strict and are expected to run in CI on every commit.

What is covered
---------------
1. ``tasks/coding_bench.jsonl`` parses into exactly 50 pydantic ``Task`` objects
   through the real loader (``tokbench.runner.load_taskset``), so this test also
   exercises the runner's JSONL path rather than a private copy of it.
2. Task ids are unique and form the contiguous ``cb-001``..``cb-050`` series.
3. The category distribution matches the documented experimental design.
4. All six ``tool-use`` tasks carry a substantial, parseable JSON ``tool_payload``
   (the blob the ``compress_tool_output`` strategy ablates).
5. Every ``must_include`` entry is genuinely checkable: it is a non-trivial
   substring, and the grader's containment tier can find it in a synthetic
   correct answer built from the task's own needles.
6. Both example YAMLs load via ``ExperimentConfig.from_yaml`` and expand to the
   advertised number of cells.
7. The price-book overlay parses into ``PriceBook`` and, when merged onto the
   bundled book, actually overrides the rate it claims to override.
8. The flagship token arithmetic is internally consistent (cells x 2,000
   prompt-token planning constant).

A note on the flagship arithmetic
---------------------------------
The design brief describes the flagship as "5 models x 3 strategies x 50 tasks x
3 repeats = 4500 cells".  That product is **2,250**, not 4,500:

    5 * 3 * 50 * 3 = 2,250

The 4.5M token figure in the same brief is only consistent with 2,250 cells at
the documented 2,000-token-per-call planning constant (2,250 * 2,000 = 4.5M), so
the "4500" is a slip in the brief, not a second experiment.  These tests assert
the arithmetic that is actually true and encode the reasoning here so the
discrepancy is visible to a reviewer instead of hidden inside a magic number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokbench.config import ExperimentConfig
from tokbench.grader import GraderKind, grade, resolve_kind
from tokbench.models import Task
from tokbench.pricing import DEFAULT_PRICE_BOOK, PriceBook, load_price_book
from tokbench.prompt import estimate_tokens
from tokbench.runner import load_taskset

# --------------------------------------------------------------------------- #
# Fixtures: paths resolved from this file, so pytest works from any cwd.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKSET_PATH = PROJECT_ROOT / "tasks" / "coding_bench.jsonl"
FLAGSHIP_PATH = PROJECT_ROOT / "examples" / "experiment.yaml"
SMOKE_PATH = PROJECT_ROOT / "examples" / "experiment_smoke.yaml"
PRICES_PATH = PROJECT_ROOT / "examples" / "prices.example.yaml"

#: The experimental design, restated as a constant so a change to either the
#: data or the documentation has to be made in two places on purpose.
EXPECTED_CATEGORIES = {
    "codegen": 10,
    "debug": 8,
    "refactor": 6,
    "test-gen": 6,
    "sql": 5,
    "explain": 5,
    "tool-use": 6,
    "reasoning": 4,
}
EXPECTED_TASK_COUNT = 50
EXPECTED_IDS = [f"cb-{n:03d}" for n in range(1, EXPECTED_TASK_COUNT + 1)]

#: Tool payload size band, in estimated tokens.
MIN_TOOL_TOKENS = 800
MAX_TOOL_TOKENS = 1500

#: Flagship shape: 5 models x 3 strategies x 50 tasks x 3 repeats.
FLAGSHIP_MODELS = 5
FLAGSHIP_STRATEGIES = 3
FLAGSHIP_REPEATS = 3
FLAGSHIP_CELLS = (
    FLAGSHIP_MODELS * FLAGSHIP_STRATEGIES * EXPECTED_TASK_COUNT * FLAGSHIP_REPEATS
)  # = 2,250
PLANNING_TOKENS_PER_CALL = 2_000
FLAGSHIP_PLANNED_TOKENS = FLAGSHIP_CELLS * PLANNING_TOKENS_PER_CALL  # = 4,500,000


@pytest.fixture(scope="module")
def taskset():
    """The benchmark task set, loaded through the production loader."""
    return load_taskset(TASKSET_PATH)


@pytest.fixture(scope="module")
def tasks(taskset) -> list[Task]:
    return list(taskset.tasks)


@pytest.fixture(scope="module")
def flagship() -> ExperimentConfig:
    return ExperimentConfig.from_yaml(FLAGSHIP_PATH)


@pytest.fixture(scope="module")
def smoke() -> ExperimentConfig:
    return ExperimentConfig.from_yaml(SMOKE_PATH)


# --------------------------------------------------------------------------- #
# 1. The task file loads and is the right size
# --------------------------------------------------------------------------- #
def test_files_exist():
    """All committed artefacts are present — a missing file must fail loudly."""
    for path in (TASKSET_PATH, FLAGSHIP_PATH, SMOKE_PATH, PRICES_PATH):
        assert path.exists(), f"missing deliverable: {path}"


def test_taskset_loads_exactly_50_tasks(taskset, tasks):
    assert len(taskset.tasks) == EXPECTED_TASK_COUNT
    assert len(tasks) == EXPECTED_TASK_COUNT
    assert all(isinstance(task, Task) for task in tasks)


def test_every_line_is_independently_parseable():
    """Each JSONL line must stand alone (no reliance on load order or comments)."""
    lines = [line for line in TASKSET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == EXPECTED_TASK_COUNT
    for lineno, line in enumerate(lines, 1):
        payload = json.loads(line)
        assert isinstance(payload, dict), f"line {lineno} is not a JSON object"
        Task.model_validate(payload)  # raises on any schema violation


def test_unknown_field_is_rejected():
    """``Task`` forbids extra fields; the dataset must not smuggle any in."""
    from pydantic import ValidationError

    raw = json.loads(TASKSET_PATH.read_text(encoding="utf-8").splitlines()[0])
    raw["not_a_real_field"] = True
    # Assert the specific failure, not a blind Exception: if ``extra`` were ever
    # relaxed to "ignore", this test must fail rather than silently pass.
    with pytest.raises(ValidationError):
        Task.model_validate(raw)


# --------------------------------------------------------------------------- #
# 2. Identity and design
# --------------------------------------------------------------------------- #
def test_ids_are_unique(tasks):
    ids = [task.id for task in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids make two cells look like one"


def test_ids_form_contiguous_series(tasks):
    assert [task.id for task in tasks] == EXPECTED_IDS


def test_category_distribution_matches_design(tasks):
    actual: dict[str, int] = {}
    for task in tasks:
        actual[task.category] = actual.get(task.category, 0) + 1
    assert actual == EXPECTED_CATEGORIES


def test_every_task_has_a_substantial_prompt(tasks):
    for task in tasks:
        words = len(task.prompt.split())
        assert task.prompt.strip(), f"{task.id}: empty prompt"
        assert 30 <= words <= 120, f"{task.id}: prompt is {words} words, expected 30-120"


def test_prompts_are_not_duplicates(tasks):
    normalized = [" ".join(task.prompt.lower().split()) for task in tasks]
    assert len(normalized) == len(set(normalized)), "duplicate prompt text"


def test_difficulty_is_in_range_and_varied(tasks):
    for task in tasks:
        assert 0.0 <= task.difficulty <= 1.0, f"{task.id}: difficulty out of range"
    distinct = {round(task.difficulty, 3) for task in tasks}
    assert len(distinct) >= 5, "difficulty must vary so quality scores can spread"
    assert min(distinct) <= 0.3 and max(distinct) >= 0.8, "difficulty range too narrow"


def test_language_coverage(tasks):
    languages = {str(task.metadata.get("language", "")).lower() for task in tasks}
    for required in ("python", "typescript", "go", "rust", "sql", "shell"):
        assert required in languages, f"no task written in {required}"


# --------------------------------------------------------------------------- #
# 3. Tool payloads — the thing the compression strategy ablates
# --------------------------------------------------------------------------- #
def test_tool_use_tasks_all_have_payloads(tasks):
    tool_tasks = [task for task in tasks if task.category == "tool-use"]
    assert len(tool_tasks) == EXPECTED_CATEGORIES["tool-use"]
    for task in tool_tasks:
        assert task.tool_payload, f"{task.id}: tool-use task has no tool_payload"
        assert task.tool_payload.strip(), f"{task.id}: tool_payload is blank"


def test_tool_payloads_are_substantial(tasks):
    """Each blob must be large enough that compressing it changes the bill."""
    for task in tasks:
        if not task.tool_payload:
            continue
        tokens = estimate_tokens(task.tool_payload)
        assert MIN_TOOL_TOKENS <= tokens <= MAX_TOOL_TOKENS, (
            f"{task.id}: tool_payload ~{tokens} tokens, expected "
            f"{MIN_TOOL_TOKENS}-{MAX_TOOL_TOKENS}"
        )


def test_tool_payloads_are_valid_json(tasks):
    """Compression summarises JSON structurally; a malformed blob hits the fallback."""
    for task in tasks:
        if not task.tool_payload:
            continue
        parsed = json.loads(task.tool_payload)
        assert isinstance(parsed, (dict, list)), f"{task.id}: payload is a bare scalar"


def test_only_tool_use_tasks_carry_payloads(tasks):
    for task in tasks:
        if task.category != "tool-use":
            assert not task.tool_payload, (
                f"{task.id}: payload on a {task.category!r} task would be ignored"
            )


# --------------------------------------------------------------------------- #
# 4. Grading is actually possible for every task
# --------------------------------------------------------------------------- #
def test_every_task_has_a_grading_path(tasks):
    """No task may fall through to a tier that cannot score it."""
    for task in tasks:
        kind = resolve_kind(task)
        assert kind is not GraderKind.NONE, f"{task.id}: no grading tier resolved"
        has_reference = bool((task.reference or "").strip())
        assert has_reference or task.must_include, (
            f"{task.id}: neither a reference nor must_include entries"
        )


def test_must_include_entries_are_checkable(tasks):
    """A needle must be findable in an answer that contains it, per the grader."""
    for task in tasks:
        for needle in task.must_include:
            compact = "".join(needle.split())
            assert len(compact) >= 2, f"{task.id}: needle {needle!r} is too short"


def test_contains_grader_scores_a_correct_answer_highly(tasks):
    """Build an answer from the task's own needles and confirm it passes.

    This is the end-to-end proof that ``must_include`` is not decorative: if the
    grader could not find its own needles, every containment-graded task would be
    scoring 0 for correct answers.
    """
    contains_tasks = [task for task in tasks if resolve_kind(task) is GraderKind.CONTAINS]
    assert contains_tasks, "expected most tasks to be graded by containment"

    for task in contains_tasks:
        answer = (
            "Here is the implementation.\n\n"
            + "\n".join(f"- {needle}" for needle in task.must_include)
            + "\n\nThat covers the required behaviour, including the error path."
        )
        score = grade(task, answer)
        assert score.grader is GraderKind.CONTAINS
        assert score.passed, f"{task.id}: correct answer scored {score.score} ({score.rationale})"
        assert score.score >= 0.9


def test_contains_grader_scores_an_empty_answer_zero(tasks):
    """The same tier must give an empty answer no credit at all."""
    contains_tasks = [task for task in tasks if resolve_kind(task) is GraderKind.CONTAINS]
    for task in contains_tasks[:10]:
        score = grade(task, "")
        assert score.score == 0.0
        assert not score.passed


def test_exact_match_tasks_have_short_references(tasks):
    """A reference longer than the exact-match ceiling would be silently ignored."""
    referenced = [task for task in tasks if (task.reference or "").strip()]
    assert len(referenced) >= 3, "expected at least three exact-match tasks"
    for task in referenced:
        assert len(task.reference or "") <= 64, f"{task.id}: reference too long for exact match"
        assert "\n" not in (task.reference or ""), f"{task.id}: reference must be one line"


def test_exact_match_tasks_grade_their_own_reference(tasks):
    """Feeding the reference back must earn a full pass."""
    for task in tasks:
        if resolve_kind(task) is not GraderKind.EXACT_MATCH:
            continue
        score = grade(task, task.reference or "")
        assert score.passed and score.score == 1.0, f"{task.id}: {score.rationale}"


# --------------------------------------------------------------------------- #
# 5. Example configs load and expand to the advertised shape
# --------------------------------------------------------------------------- #
def test_flagship_config_loads(flagship):
    assert flagship.version == 1
    assert flagship.adapter_mode == "mock"
    assert len(flagship.models) == FLAGSHIP_MODELS
    assert len(flagship.strategies) == FLAGSHIP_STRATEGIES
    assert flagship.experiment.repeats == FLAGSHIP_REPEATS
    assert flagship.experiment.seed == 42
    assert flagship.experiment.concurrency == 8
    assert flagship.retry.max_attempts == 3
    assert flagship.budget.max_total_tokens == 5_000_000
    assert flagship.output.dir == "reports"
    assert flagship.taskset.path.endswith("tasks/coding_bench.jsonl")


def test_flagship_model_lineup(flagship):
    ids = [model.id for model in flagship.models]
    assert ids == [
        "deepseek-chat",
        "deepseek-reasoner",
        "gpt-4o-mini",
        "gpt-4.1-nano",
        "mock-small",
    ]
    for model in flagship.models:
        assert model.provider == "mock"
        # price_key must resolve in the price book, else the run cannot cost itself
        assert model.price_key == model.id
        assert DEFAULT_PRICE_BOOK.has(model.price_key), (
            f"{model.id}: no price entry for {model.price_key!r}"
        )


def test_flagship_strategies_encode_the_ablation(flagship):
    by_id = {strategy.id: strategy for strategy in flagship.strategies}
    assert set(by_id) == {"baseline", "cot", "fewshot-compress"}

    baseline = by_id["baseline"]
    assert baseline.system_prompt_preset == "short"
    assert baseline.few_shot == 0
    assert baseline.cot is False
    assert baseline.compress_tool_output is False

    cot = by_id["cot"]
    assert cot.cot is True
    # Single-variable discipline: only `cot` differs from the control.
    assert cot.system_prompt_preset == baseline.system_prompt_preset
    assert cot.few_shot == baseline.few_shot
    assert cot.compress_tool_output == baseline.compress_tool_output

    fewshot = by_id["fewshot-compress"]
    assert fewshot.few_shot == 3
    assert fewshot.compress_tool_output is True
    assert fewshot.cot is False
    assert fewshot.system_prompt_preset == baseline.system_prompt_preset


def test_flagship_taskset_path_resolves_to_the_real_file(flagship):
    resolved = flagship.resolve_path(flagship.taskset.path)
    assert resolved.exists(), f"flagship taskset path does not resolve: {resolved}"
    assert resolved.resolve() == TASKSET_PATH.resolve()


def test_flagship_prices_path_resolves(flagship):
    assert flagship.prices is not None
    resolved = flagship.resolve_path(flagship.prices)
    assert resolved.exists(), f"flagship prices path does not resolve: {resolved}"
    assert resolved.resolve() == PRICES_PATH.resolve()


def test_smoke_config_loads(smoke):
    assert smoke.adapter_mode == "mock"
    assert len(smoke.models) == 2
    assert len(smoke.strategies) == 2
    assert smoke.experiment.repeats == 1
    assert smoke.taskset.limit == 5
    assert smoke.prices is None, "smoke run must work with no extra files present"


def test_smoke_taskset_limit_yields_five_tasks(smoke, taskset):
    limited = load_taskset(smoke.resolve_path(smoke.taskset.path), limit=smoke.taskset.limit)
    assert len(limited.tasks) == 5
    assert [task.id for task in limited.tasks] == EXPECTED_IDS[:5]


def test_smoke_cell_count(smoke, taskset):
    limited = load_taskset(smoke.resolve_path(smoke.taskset.path), limit=smoke.taskset.limit)
    # 2 models x 2 strategies x 5 tasks x 1 repeat
    assert smoke.planned_cells(len(limited.tasks)) == 20


# --------------------------------------------------------------------------- #
# 6. The flagship arithmetic
# --------------------------------------------------------------------------- #
def test_flagship_cell_arithmetic(flagship, taskset):
    """5 x 3 x 50 x 3 = 2,250 cells (NOT 4,500 — see the module docstring)."""
    cells = flagship.planned_cells(len(taskset.tasks))
    assert FLAGSHIP_CELLS == 2_250
    assert cells == FLAGSHIP_CELLS


def test_flagship_expands_to_the_planned_cells(flagship, taskset):
    """The expansion must actually produce one cell per matrix position."""
    expanded = flagship.expanded_cells(list(taskset.tasks))
    assert len(expanded) == FLAGSHIP_CELLS
    assert len({cell.cell_key for cell in expanded}) == FLAGSHIP_CELLS

    # Every (model, strategy, task, repeat) combination appears exactly once.
    combos = {(cell.model.id, cell.strategy.id, cell.task_id, cell.repeat) for cell in expanded}
    assert len(combos) == FLAGSHIP_CELLS


def test_flagship_token_plan_is_4_5_million():
    """2,250 cells x 2,000 prompt tokens = 4.5M — the number the budget covers."""
    assert FLAGSHIP_PLANNED_TOKENS == 4_500_000
    assert FLAGSHIP_PLANNED_TOKENS == 2_250 * 2_000


def test_flagship_budget_covers_the_plan(flagship):
    assert flagship.budget.max_total_tokens is not None
    assert flagship.budget.max_total_tokens >= FLAGSHIP_PLANNED_TOKENS


def test_prompt_tokens_are_within_the_planning_envelope(flagship, taskset):
    """Real assembled prompts must not blow past the 2,000-token planning constant.

    The constant is a planning assumption, not a measurement: these 50 tasks
    assemble to a few hundred tokens each, well under it.  Asserting the *bound*
    (rather than equality) keeps the 4.5M budget honest while letting the dataset
    evolve.
    """
    from tokbench.prompt import assemble_prompt

    model = flagship.models[0]
    for strategy in flagship.strategies:
        for task in taskset.tasks:
            assembled = assemble_prompt(task, strategy, model)
            assert assembled.total_prompt_tokens_est <= PLANNING_TOKENS_PER_CALL, (
                f"{task.id}/{strategy.id}: {assembled.total_prompt_tokens_est} tokens "
                f"exceeds the {PLANNING_TOKENS_PER_CALL}-token planning constant"
            )


def test_compression_reduces_prompt_tokens(flagship, taskset):
    """The fewshot-compress arm must actually shrink tool payloads.

    Otherwise the arm would be measuring few-shot cost alone while claiming to
    measure compression, and the strategy-delta table would be misleading.
    """
    from tokbench.prompt import assemble_prompt

    model = flagship.models[0]
    strategies = {strategy.id: strategy for strategy in flagship.strategies}
    baseline = strategies["baseline"]
    compress = strategies["fewshot-compress"]

    tool_tasks = [task for task in taskset.tasks if task.tool_payload]
    assert len(tool_tasks) == EXPECTED_CATEGORIES["tool-use"]

    for task in tool_tasks:
        raw = assemble_prompt(task, baseline, model)
        compressed = assemble_prompt(task, compress, model)
        # Compression must shrink the tool block itself, even though the extra
        # few-shot messages make the *total* larger.
        assert compressed.tool_tokens < raw.tool_tokens, (
            f"{task.id}: compression did not reduce tool tokens "
            f"({compressed.tool_tokens} vs {raw.tool_tokens})"
        )


# --------------------------------------------------------------------------- #
# 7. Price-book overlay
# --------------------------------------------------------------------------- #
def test_price_overlay_parses():
    book = PriceBook.from_yaml(str(PRICES_PATH))
    assert book.name == "my-rates"
    assert book.currency == "USD"
    assert "deepseek-chat" in book.models
    assert book.models["deepseek-chat"].input == pytest.approx(0.27)
    assert book.models["deepseek-chat"].output == pytest.approx(1.10)


def test_price_overlay_fills_cache_defaults():
    """Omitted cache prices must be derived, not left as None."""
    book = PriceBook.from_yaml(str(PRICES_PATH))
    price = book.models["gpt-4o-mini"]
    assert price.cache_read == pytest.approx(price.input * 0.1)
    assert price.cache_write == pytest.approx(price.input * 1.25)


def test_price_overlay_explicit_cache_prices_win():
    book = PriceBook.from_yaml(str(PRICES_PATH))
    price = book.models["deepseek-chat"]
    # 0.07 is not 10% of 0.27 (0.027), so this proves the explicit value survived.
    assert price.cache_read == pytest.approx(0.07)
    assert price.cache_write == pytest.approx(0.27)


def test_price_overlay_merges_over_the_bundled_book():
    merged = load_price_book(str(PRICES_PATH))
    # Overridden entry.
    assert merged.get("deepseek-chat").input == pytest.approx(0.27)
    # Untouched bundled entry survives the merge.
    assert merged.has("gpt-4o")
    assert merged.get("gpt-4o").input == pytest.approx(2.50)


def test_price_overlay_covers_every_flagship_model(flagship):
    """Every model in the flagship must be priceable through the overlay."""
    merged = load_price_book(str(PRICES_PATH))
    for model in flagship.models:
        assert merged.has(model.price_key), f"no price for {model.price_key}"


def test_price_book_costs_a_known_usage():
    """A hand-computable cost check: 1M uncached in + 1M out at deepseek-chat rates."""
    from tokbench.models import Usage

    book = load_price_book(str(PRICES_PATH))
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    # 0.27 + 1.10 = 1.37 USD
    assert book.cost("deepseek-chat", usage) == pytest.approx(1.37)
