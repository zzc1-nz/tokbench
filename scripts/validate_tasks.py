"""Validate ``tasks/coding_bench.jsonl`` against the frozen ``Task`` contract.

This is the gate that keeps the benchmark dataset trustworthy.  A benchmark whose
task file is silently malformed produces numbers that look fine and mean nothing,
so every constraint below is fatal: the script exits ``1`` on the first class of
violation it finds (reporting *all* instances of it, not just the first) rather
than warning and continuing.

Checks
------
1. Every line is a JSON object and validates as :class:`tokbench.models.Task`
   (``extra="forbid"``, so a typo'd field is an error, not dead weight).
2. Exactly :data:`EXPECTED_COUNT` tasks; ids unique and matching ``cb-NNN``.
3. Category distribution matches :data:`EXPECTED_CATEGORIES` exactly.
4. Every task has a non-empty prompt of 30-120 words, and a difficulty in
   ``[0, 1]`` that is actually varied across the set.
5. Every ``must_include`` entry is a non-trivial substring that can plausibly
   appear in an answer (not whitespace, not absurdly long).
6. All six ``tool-use`` tasks carry a substantial ``tool_payload`` inside the
   documented 800-1500 estimated-token band — that payload is what the
   ``compress_tool_output`` strategy ablates, so a thin one would make the
   ablation meaningless.
7. ``reference`` is either absent or short enough for exact-match grading, and at
   least a few tasks carry one.
8. No duplicate prompts (a benchmark that asks the same thing twice weights that
   behaviour twice).

Run it from the ``tokbench`` project root::

    $env:PYTHONPATH="...\\.pylibs"
    python scripts/validate_tasks.py

It prints a category distribution table plus token statistics and exits ``0``
only when every check passes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Import the *project* modules, not copies: the point is to validate against the
# same frozen schema the runner will use at run time.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from tokbench.models import Task  # noqa: E402
from tokbench.prompt import estimate_tokens  # noqa: E402

# --------------------------------------------------------------------------- #
# Expected shape of the dataset
# --------------------------------------------------------------------------- #
TASKSET_PATH = ROOT / "tasks" / "coding_bench.jsonl"

EXPECTED_COUNT = 50

#: The eight categories and their required sizes.  These numbers are the
#: experimental design: they decide how much weight each skill area carries in
#: the aggregate score, so they are asserted rather than merely documented.
EXPECTED_CATEGORIES: dict[str, int] = {
    "codegen": 10,
    "debug": 8,
    "refactor": 6,
    "test-gen": 6,
    "sql": 5,
    "explain": 5,
    "tool-use": 6,
    "reasoning": 4,
}

#: Prompt length window, in words.  Below 30 words a request is a toy question;
#: above 120 it stops reading like a ticket and starts rewarding verbosity.
MIN_PROMPT_WORDS = 30
MAX_PROMPT_WORDS = 120

#: Tool payload band, in *estimated* tokens.  Large enough that compressing it
#: changes the bill materially, small enough that one task does not dominate the
#: run's token budget.
MIN_TOOL_TOKENS = 800
MAX_TOOL_TOKENS = 1500

#: A reference longer than this cannot be graded by exact match, so it would be
#: silently ignored by ``grader.resolve_kind``.  Rejecting it here prevents a
#: task author from believing their reference is doing work when it is not.
MAX_REFERENCE_CHARS = 64

#: Substrings shorter than this are too generic to prove anything (``a`` would
#: "match" every answer), so ``must_include`` entries must clear it *after*
#: whitespace compaction.  Two characters is deliberate: ``"fn"`` and ``"?"``
#: differ enormously in meaning, so the length floor is a floor, not a proxy for
#: specificity — the grader's own coverage fraction supplies the rest.
MIN_NEEDLE_CHARS = 2

#: How many tasks must carry a ``reference`` for exact-match grading to have any
#: coverage at all.
MIN_REFERENCED_TASKS = 3

#: Distinct difficulty levels required: a flat difficulty column would make the
#: quality spread unexplainable.
MIN_DISTINCT_DIFFICULTIES = 5

REQUIRED_LANGUAGES = {"python", "typescript", "go", "rust", "sql", "shell"}


class Report:
    """Collects failures so one run reports every problem, not just the first."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.fail(message)
        return condition

    @property
    def ok(self) -> bool:
        return not self.errors


def _rule(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{title}")
        print("-" * width)
    else:
        print("-" * width)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path, report: Report) -> list[dict]:
    """Read the JSONL file into raw dicts, reporting parse failures by line."""
    if not path.exists():
        report.fail(f"task set not found: {path}")
        return []

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        report.fail(f"task set is empty: {path}")
        return []

    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            report.fail(f"line {lineno}: blank line (JSONL must be one object per line)")
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            report.fail(f"line {lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(obj, dict):
            report.fail(f"line {lineno}: expected a JSON object, got {type(obj).__name__}")
            continue
        rows.append(obj)
    return rows


def parse_tasks(rows: list[dict], report: Report) -> list[Task]:
    """Validate each raw dict against the frozen :class:`Task` model."""
    tasks: list[Task] = []
    for index, row in enumerate(rows, 1):
        try:
            tasks.append(Task.model_validate(row))
        except ValidationError as exc:
            # Collapse pydantic's multi-line message to one readable line.
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            )
            ident = row.get("id", f"line {index}")
            report.fail(f"{ident}: does not validate as Task: {details}")
    return tasks


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_count(tasks: list[Task], report: Report) -> None:
    report.check(
        len(tasks) == EXPECTED_COUNT,
        f"expected exactly {EXPECTED_COUNT} tasks, found {len(tasks)}",
    )


def check_ids(tasks: list[Task], report: Report) -> None:
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    report.check(not duplicates, f"duplicate task ids: {duplicates}")

    expected = [f"cb-{n:03d}" for n in range(1, EXPECTED_COUNT + 1)]
    missing = [i for i in expected if i not in set(ids)]
    unexpected = [i for i in ids if i not in set(expected)]
    report.check(not missing, f"missing ids: {missing}")
    report.check(not unexpected, f"unexpected ids: {unexpected}")


def check_categories(tasks: list[Task], report: Report) -> None:
    actual = Counter(t.category for t in tasks)
    expected = Counter(EXPECTED_CATEGORIES)

    for category, want in sorted(expected.items()):
        got = actual.get(category, 0)
        report.check(
            got == want,
            f"category {category!r}: expected {want} tasks, found {got}",
        )
    for category in sorted(set(actual) - set(expected)):
        report.fail(
            f"category {category!r} is not part of the design "
            f"({actual[category]} tasks); expected one of {sorted(expected)}"
        )


def check_prompts(tasks: list[Task], report: Report) -> None:
    for task in tasks:
        prompt = task.prompt.strip()
        if not prompt:
            report.fail(f"{task.id}: empty prompt")
            continue
        words = len(prompt.split())
        report.check(
            MIN_PROMPT_WORDS <= words <= MAX_PROMPT_WORDS,
            f"{task.id}: prompt is {words} words, expected {MIN_PROMPT_WORDS}-{MAX_PROMPT_WORDS}",
        )

    # Duplicate prompts would silently double-weight one behaviour.
    seen: dict[str, str] = {}
    for task in tasks:
        key = " ".join(task.prompt.lower().split())
        if key in seen:
            report.fail(f"{task.id}: prompt duplicates {seen[key]}")
        else:
            seen[key] = task.id


def check_difficulty(tasks: list[Task], report: Report) -> None:
    for task in tasks:
        # The model already enforces the range; re-asserting documents intent and
        # catches a schema relaxation elsewhere.
        report.check(
            0.0 <= task.difficulty <= 1.0,
            f"{task.id}: difficulty {task.difficulty} outside [0, 1]",
        )
    distinct = len({round(t.difficulty, 3) for t in tasks})
    report.check(
        distinct >= MIN_DISTINCT_DIFFICULTIES,
        f"difficulty has only {distinct} distinct values, expected at least "
        f"{MIN_DISTINCT_DIFFICULTIES} so quality scores can spread",
    )


def check_must_include(tasks: list[Task], report: Report) -> None:
    for task in tasks:
        if task.reference and len(task.reference.strip()) <= MAX_REFERENCE_CHARS:
            # Exact-match grading wins, so must_include is unused here.
            continue
        report.check(
            bool(task.must_include),
            f"{task.id}: no must_include entries and no short reference, so the "
            "task can only be graded by the generic heuristic",
        )
        for needle in task.must_include:
            if not needle.strip():
                report.fail(f"{task.id}: blank must_include entry")
                continue
            # ``grade_contains`` compares after removing *all* whitespace, so a
            # trailing space ("def ", "func ") is not a defect in the needle: it
            # is how the author says "this token, not a substring of a longer
            # identifier".  Judge the compacted form instead.
            compact = "".join(needle.split())
            if len(compact) < MIN_NEEDLE_CHARS:
                report.fail(
                    f"{task.id}: must_include {needle!r} compacts to {compact!r}, "
                    "which is too short to be evidence"
                )


def check_references(tasks: list[Task], report: Report) -> None:
    referenced = [t for t in tasks if (t.reference or "").strip()]
    report.check(
        len(referenced) >= MIN_REFERENCED_TASKS,
        f"only {len(referenced)} tasks carry a reference, expected at least "
        f"{MIN_REFERENCED_TASKS} so exact-match grading has coverage",
    )
    for task in referenced:
        reference = task.reference or ""
        report.check(
            len(reference) <= MAX_REFERENCE_CHARS,
            f"{task.id}: reference is {len(reference)} chars; exact-match grading "
            f"only applies at <= {MAX_REFERENCE_CHARS}, so this reference would be "
            "ignored by the grader",
        )
        report.check(
            "\n" not in reference,
            f"{task.id}: reference must be a single line for exact matching",
        )


def check_tool_payloads(tasks: list[Task], report: Report) -> None:
    tool_tasks = [t for t in tasks if t.category == "tool-use"]
    report.check(
        len(tool_tasks) == EXPECTED_CATEGORIES["tool-use"],
        f"expected {EXPECTED_CATEGORIES['tool-use']} tool-use tasks, found {len(tool_tasks)}",
    )

    for task in tool_tasks:
        payload = task.tool_payload or ""
        if not payload.strip():
            report.fail(f"{task.id}: tool-use task has no tool_payload")
            continue
        tokens = estimate_tokens(payload)
        report.check(
            MIN_TOOL_TOKENS <= tokens <= MAX_TOOL_TOKENS,
            f"{task.id}: tool_payload is ~{tokens} tokens ({len(payload)} chars), "
            f"expected {MIN_TOOL_TOKENS}-{MAX_TOOL_TOKENS}",
        )
        # The payload must be parseable JSON: the compression strategy summarises
        # it structurally, and a malformed blob would exercise the fallback path
        # instead of the intended one.
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            report.fail(f"{task.id}: tool_payload is not valid JSON: {exc}")

    # A tool payload on a non-tool task would be silently ignored by the prompt
    # builder, which is confusing rather than harmful — still worth flagging.
    for task in tasks:
        if task.category != "tool-use" and task.tool_payload:
            report.fail(
                f"{task.id}: has a tool_payload but category is {task.category!r}, not 'tool-use'"
            )


def check_languages(tasks: list[Task], report: Report) -> None:
    seen = {str(t.metadata.get("language", "unknown")) for t in tasks}
    missing = REQUIRED_LANGUAGES - seen
    report.check(not missing, f"no tasks for required languages: {sorted(missing)}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_distribution(tasks: list[Task]) -> None:
    counts = Counter(t.category for t in tasks)
    total = len(tasks) or 1
    print(f"{'category':<12} {'count':>5} {'share':>7}  {'prompt tok (avg)':>16}")
    print("-" * 46)
    for category, expected in sorted(EXPECTED_CATEGORIES.items()):
        count = counts.get(category, 0)
        subset = [t for t in tasks if t.category == category]
        avg_tokens = sum(estimate_tokens(t.prompt) for t in subset) / len(subset) if subset else 0.0
        marker = "" if count == expected else f"  <-- expected {expected}"
        print(f"{category:<12} {count:>5} {count / total:>6.0%}  {avg_tokens:>16.0f}{marker}")
    print("-" * 46)
    print(f"{'TOTAL':<12} {len(tasks):>5} {1.0:>6.0%}")


def print_token_stats(tasks: list[Task]) -> None:
    prompt_tokens = [estimate_tokens(t.prompt) for t in tasks]
    prompt_chars = [len(t.prompt) for t in tasks]
    tool_tokens = [estimate_tokens(t.tool_payload) for t in tasks if t.tool_payload]

    print(f"{'metric':<34} {'value':>12}")
    print("-" * 48)
    print(f"{'prompt chars (avg)':<34} {sum(prompt_chars) / len(prompt_chars):>12.0f}")
    print(f"{'prompt tokens (avg, chars/4)':<34} {sum(prompt_tokens) / len(prompt_tokens):>12.1f}")
    print(f"{'prompt tokens (min)':<34} {min(prompt_tokens):>12d}")
    print(f"{'prompt tokens (max)':<34} {max(prompt_tokens):>12d}")
    if tool_tokens:
        print(f"{'tool payload tokens (avg)':<34} {sum(tool_tokens) / len(tool_tokens):>12.0f}")
        print(f"{'tool payload tokens (min)':<34} {min(tool_tokens):>12d}")
        print(f"{'tool payload tokens (max)':<34} {max(tool_tokens):>12d}")
        print(
            f"{'tool payload chars (avg)':<34} "
            f"{sum(len(t.tool_payload or '') for t in tasks) / len(tool_tokens):>12.0f}"
        )
    total_chars = sum(prompt_chars) + sum(len(t.tool_payload or "") for t in tasks)
    print(f"{'dataset chars (prompt + payload)':<34} {total_chars:>12,}")
    print(f"{'dataset est. tokens':<34} {total_chars / 4:>12,.0f}")


def print_difficulty_histogram(tasks: list[Task]) -> None:
    buckets = Counter(min(int(t.difficulty * 5), 4) for t in tasks)
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    print(f"{'difficulty band':<16} {'count':>6}  bar")
    print("-" * 48)
    for index, label in enumerate(labels):
        count = buckets.get(index, 0)
        print(f"{label:<16} {count:>6}  {'#' * count}")
    ambiguous = [t.id for t in tasks if "ambiguity" in t.metadata]
    if ambiguous:
        print(f"\ndeliberately ambiguous ({len(ambiguous)}): {', '.join(ambiguous)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    report = Report()

    _rule()
    print("tokbench task-set validator")
    print(f"file: {TASKSET_PATH}")
    _rule()

    rows = load_raw(TASKSET_PATH, report)
    tasks = parse_tasks(rows, report)

    check_count(tasks, report)
    check_ids(tasks, report)
    check_categories(tasks, report)
    check_prompts(tasks, report)
    check_difficulty(tasks, report)
    check_must_include(tasks, report)
    check_references(tasks, report)
    check_tool_payloads(tasks, report)
    check_languages(tasks, report)

    if tasks:
        _rule("category distribution")
        print_distribution(tasks)

        _rule("token statistics")
        print_token_stats(tasks)

        _rule("difficulty spread")
        print_difficulty_histogram(tasks)

    _rule("result")
    if report.ok:
        print(
            f"PASS - {len(tasks)} tasks, {len(EXPECTED_CATEGORIES)} categories, "
            f"all constraints satisfied"
        )
        return 0

    print(f"FAIL - {len(report.errors)} violation(s):")
    for message in report.errors:
        print(f"  * {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
