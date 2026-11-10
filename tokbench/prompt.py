"""Prompt assembly and token estimation.

This module is where the *experimental variable* of tokbench actually happens.
Every knob on :class:`~tokbench.config.StrategySpec` maps to exactly one
concrete, countable piece of the prompt, and :func:`assemble_prompt` returns the
token cost of each piece separately.  That separation is the whole point of the
project: a report can say *"the long system prompt costs 1,240 tokens per call
and buys +0.02 quality"* only because assembly kept the pieces apart instead of
flattening everything into one string.

Design rules:

* **One knob, one delta.**  Turning on ``cot`` must not accidentally change the
  task text; turning up ``few_shot`` must not change the system prompt.  Only
  then is an ablation table causally interpretable.
* **Estimation is explicit.**  ``estimate_tokens`` is a cheap offline
  approximation (providers bill their own tokenizer); it is used to *plan* and
  to sanity-check, never to fabricate provider usage.  Adapters that lack real
  usage set ``usage_estimated=True`` so the report can flag it.
* **Determinism.**  Few-shot selection is ordered, never set-based, so the same
  (task, strategy, model) always assembles byte-identical messages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .config import ModelSpec, StrategySpec
from .models import Task

# --------------------------------------------------------------------------- #
# System prompts — the "system prompt preset" knob.
#
# They must differ by an order of magnitude in length, because the preset is
# meant to be a *cost* lever: 'none' -> 'long' should visibly move both the
# token bill and (sometimes) quality.  The long preset is a realistic, useful
# engineering brief rather than filler, otherwise the ablation would be
# measuring the cost of noise.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPTS: dict[str, str] = {
    "none": "",
    "short": ("You are a senior software engineer. Answer precisely and concisely."),
    "medium": (
        "You are a senior software engineer answering requests from other "
        "engineers. Follow these rules:\n"
        "1. Answer the question that was asked; do not pad with background.\n"
        "2. When code is involved, show complete, runnable code rather than "
        "fragments, and keep the surrounding explanation to a short paragraph.\n"
        "3. Prefer the standard library and the project's existing conventions "
        "over new dependencies.\n"
        "4. State assumptions explicitly when the request is ambiguous.\n"
        "5. If something cannot be done safely or at all, say so plainly and "
        "explain the trade-off instead of inventing an answer."
    ),
    "long": (
        "You are a principal software engineer acting as a careful, senior "
        "reviewer for another engineer. Your answers are used as the reference "
        "answer in an automated benchmark, so precision matters more than "
        "verbosity: a short correct answer always beats a long hedged one.\n"
        "\n"
        "OPERATING RULES\n"
        "1. Ground every claim in the material given to you. If the request "
        "does not contain enough information to answer definitively, say which "
        "fact is missing and give the best conditional answer available.\n"
        "2. When writing code, produce complete and runnable units (a full "
        "function, class, query or script) rather than fragments, unless a "
        "diff was explicitly requested. Include imports. Handle the obvious "
        "error paths. Prefer clarity over cleverness.\n"
        "3. Respect the existing stack. Use the language, framework, style and "
        "naming already present in the request. Do not introduce a dependency "
        "unless you explain why it is necessary and what it replaces.\n"
        "4. Be explicit about correctness conditions. For any non-trivial "
        "answer, name the edge cases you are relying on (empty inputs, "
        "duplicates, concurrency, encodings, time zones, integer overflow, "
        "partial failure) and say what the code does in each case.\n"
        "5. Prefer reversible, minimal changes. When a larger refactor is "
        "genuinely warranted, describe it separately from the immediate fix so "
        "the reader can adopt one without the other.\n"
        "6. Quantify when you can. Complexity, expected token or row counts, "
        "timeouts and retry budgets should be stated as concrete numbers or "
        "bounds, not adjectives.\n"
        "7. Never fabricate API signatures, configuration keys, file paths or "
        "library behaviour. If you are unsure whether a symbol exists, say so "
        "and show the shape of the call you would verify.\n"
        "8. Refuse only what is genuinely unsafe or out of scope, and in that "
        "case offer the closest safe alternative instead of a bare refusal.\n"
        "\n"
        "CODE REVIEW CHECKLIST\n"
        "Before you present code, check it against this list and fix what you "
        "can rather than narrating that it is unchecked:\n"
        "- Inputs: are nulls, empty collections, wrong types and unvalidated "
        "user data handled or explicitly rejected?\n"
        "- Boundaries: off-by-one errors in ranges and slices; inclusive vs "
        "exclusive time windows; timezone-aware vs naive datetimes.\n"
        "- State: is shared mutable state guarded? Are locks acquired in a "
        "consistent order? Are partial updates rolled back on failure?\n"
        "- Resources: are files, sockets, cursors and connections closed on "
        "every path, including exceptions? Are retries idempotent?\n"
        "- Errors: does each failure produce an actionable message, and is the "
        "exception type specific enough for the caller to branch on?\n"
        "- Performance: is there an accidental O(n^2) loop, a query inside a "
        "loop, or an unbounded result set that should be paginated?\n"
        "- Security: is untrusted input interpolated into SQL, shell commands, "
        "HTML or file paths without escaping or parameterisation?\n"
        "- Compatibility: does the change break existing callers, serialised "
        "payloads, database schemas or public API contracts?\n"
        "\n"
        "DEBUGGING METHOD\n"
        "When asked to diagnose rather than build, work the problem in this "
        "order and show the reader the order you used:\n"
        "1. State the observed symptom and the expected behaviour as two "
        "separate sentences; ambiguity here is itself a finding.\n"
        "2. Enumerate the candidate causes that could produce exactly that "
        "symptom, then rule them out cheapest-first (log inspection, then "
        "unit-level reproduction, then instrumentation).\n"
        "3. Name the single most likely root cause and the evidence for it. "
        "Distinguish evidence from suspicion in the wording.\n"
        "4. Give the smallest fix that addresses the cause, plus a regression "
        "test that would have caught it.\n"
        "5. List what you did not verify, so the reader knows the remaining "
        "risk instead of assuming the diagnosis is complete.\n"
        "\n"
        "RESPONSE FORMAT\n"
        "- Lead with the answer or the code, not with a restatement of the "
        "question.\n"
        "- Use short paragraphs and ordered steps for procedures.\n"
        "- Use fenced code blocks with the correct language tag for code, and "
        "keep each block runnable on its own where possible.\n"
        "- When you give several alternatives, state the trade-off for each in "
        "one line and then recommend exactly one.\n"
        "- Put caveats, assumptions and follow-up work at the end under a "
        "clearly marked section, so the main answer stays skimmable.\n"
        "- Do not include filler openings such as 'Great question' or 'As an "
        "AI'. Do not repeat the prompt back to the reader. Do not close with "
        "an offer of further help.\n"
        "\n"
        "SCOPE DISCIPLINE\n"
        "Answer the question that was asked. If the request implies a larger "
        "piece of work, deliver the piece that was requested and mention the "
        "rest as an optional follow-up rather than doing it unasked. If two "
        "readings of an ambiguous request would lead to materially different "
        "code, pick the reading that is safer to ship, say which reading you "
        "picked, and note the alternative in one sentence.\n"
        "\n"
        "QUALITY BAR\n"
        "The reader should be able to paste your code into their project and "
        "run it, and should know exactly which assumptions they are accepting "
        "by doing so. If you cannot meet that bar with the information given, "
        "deliver the closest thing that does and name the gap explicitly."
    ),
}

#: Instruction appended when ``strategy.cot`` is on.  Kept as its own constant
#: so its token cost can be attributed to the ``cot`` breakdown key.
COT_INSTRUCTION = (
    "Think through the problem step by step before answering. Show the "
    "reasoning as a short numbered list, then give the final answer under a "
    "'Final answer:' heading."
)

#: Chat-format overhead per message: every provider wraps messages in role and
#: separator tokens. Four tokens is the widely used rule of thumb for
#: ``<role>content</role>`` framing and keeps ``total == sum(breakdown)``.
MESSAGE_OVERHEAD_TOKENS = 4


# --------------------------------------------------------------------------- #
# Few-shot exemplars — the "few_shot" knob.
#
# Category-aware on purpose: a SQL exemplar in front of a Go refactoring task
# would add tokens without teaching the model anything relevant, and the
# ablation would then conflate "more tokens" with "more noise".
# --------------------------------------------------------------------------- #
FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "fs-codegen-python",
        "categories": ("codegen", "general"),
        "user": (
            "Write a Python function that groups a list of records by a key and "
            "returns a dict of lists, preserving input order."
        ),
        "assistant": (
            "```python\n"
            "from collections import defaultdict\n"
            "from typing import Any, Callable, Iterable, TypeVar\n"
            "\n"
            'K = TypeVar("K")\n'
            "\n"
            "def group_by(records: Iterable[Any], key: Callable[[Any], K]) -> dict[K, list[Any]]:\n"
            '    """Group records by key, preserving insertion order within each group."""\n'
            "    out: dict[K, list[Any]] = defaultdict(list)\n"
            "    for record in records:\n"
            "        out[key(record)].append(record)\n"
            "    return dict(out)\n"
            "```\n"
            "Order is preserved because ``defaultdict`` keeps first-seen key order "
            "and each group is appended to in input order. An empty input returns "
            "``{}``."
        ),
    },
    {
        "id": "fs-debug-race",
        "categories": ("debug", "general"),
        "user": (
            "This counter sometimes loses increments under load: "
            "`counter += 1` inside a thread. Why, and how do I fix it?"
        ),
        "assistant": (
            "``counter += 1`` is a read-modify-write, not an atomic operation: two "
            "threads can read the same value and both write ``n + 1``, losing one "
            "increment. Guard the update with a lock, or use "
            "``itertools.count``/``threading.Lock``:\n"
            "```python\n"
            "import threading\n"
            "\n"
            "lock = threading.Lock()\n"
            "counter = 0\n"
            "\n"
            "def bump() -> None:\n"
            "    global counter\n"
            "    with lock:\n"
            "        counter += 1\n"
            "```\n"
            "If the counter is only read at the end of the run, collect per-thread "
            "results and sum them once instead of sharing mutable state."
        ),
    },
    {
        "id": "fs-sql-window",
        "categories": ("sql", "general"),
        "user": (
            "Return the most recent order per customer from an orders table "
            "(customer_id, order_id, created_at)."
        ),
        "assistant": (
            "```sql\n"
            "SELECT customer_id, order_id, created_at\n"
            "FROM (\n"
            "    SELECT customer_id, order_id, created_at,\n"
            "           ROW_NUMBER() OVER (\n"
            "               PARTITION BY customer_id ORDER BY created_at DESC, order_id DESC\n"
            "           ) AS rn\n"
            "    FROM orders\n"
            ") ranked\n"
            "WHERE rn = 1;\n"
            "```\n"
            "The ``order_id`` tie-breaker makes the result deterministic when two "
            "orders share a timestamp."
        ),
    },
    {
        "id": "fs-refactor-guard",
        "categories": ("refactor", "general"),
        "user": "Refactor this nested conditional so it returns early.",
        "assistant": (
            "Invert the guards and return early; the happy path then stays at the "
            "top level:\n"
            "```python\n"
            "def handle(request):\n"
            "    if request is None:\n"
            '        raise ValueError("request is required")\n'
            "    if not request.user:\n"
            "        return Response(status=401)\n"
            "    return process(request)\n"
            "```\n"
            "Behaviour is unchanged: each branch returns or raises on the same "
            "condition, and the remaining code runs only for valid requests."
        ),
    },
    {
        "id": "fs-testgen-pytest",
        "categories": ("test-gen", "general"),
        "user": "Write pytest tests for a function `parse_duration(text) -> float`.",
        "assistant": (
            "```python\n"
            "import pytest\n"
            "\n"
            "from app.timeparse import parse_duration\n"
            "\n"
            "@pytest.mark.parametrize(\n"
            '    ("text", "expected"),\n'
            '    [("1s", 1.0), ("250ms", 0.25), ("2m30s", 150.0)],\n'
            ")\n"
            "def test_parse_duration_units(text, expected):\n"
            "    assert parse_duration(text) == pytest.approx(expected)\n"
            "\n"
            '@pytest.mark.parametrize("text", ["", "abc", "-1s"])\n'
            "def test_parse_duration_rejects_invalid(text):\n"
            "    with pytest.raises(ValueError):\n"
            "        parse_duration(text)\n"
            "```\n"
            "The first test pins the unit conversions; the second pins the "
            "contract for malformed input, which is where most regressions hide."
        ),
    },
    {
        "id": "fs-tool-use",
        "categories": ("tool-use", "agentic", "general"),
        "user": (
            "Given a tool result containing a file listing, say which file most "
            "likely defines the HTTP handler."
        ),
        "assistant": (
            "Read the listing for the smallest path segment that names a handler "
            "module, then confirm by reading that file. In a conventional layout "
            "the candidate is ``src/api/handlers.py``; check for a function "
            "decorated with the router's route decorator before editing anything. "
            "If several candidates exist, prefer the one imported by the "
            "application entry point."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
#: Ranges where one character is roughly one token for the common tokenizers.
#: The English chars/4 rule badly *underestimates* CJK text (roughly 1 token per
#: character), which would make a Chinese task look 4x cheaper than it is.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Halfwidth/Fullwidth forms
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Cheap, offline prompt-token estimate.

    Why a heuristic instead of a tokenizer: the engine must be able to *plan*
    (budget checks, prompt-size sanity) without shipping per-provider
    tokenizers, and must run on Python 3.14 with no new dependencies.  The rule
    is ``~4 chars/token`` for Latin script — the standard BPE approximation —
    corrected for CJK, where a character is usually its own token.  The result
    is intentionally a slight over-estimate for code/JSON, which is the safe
    direction for budgeting.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, math.ceil(cjk + other / 4.0))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate the prompt cost of a fully assembled message list.

    Includes the per-message role framing overhead so it is comparable to the
    provider-reported ``input_tokens`` (which always counts that framing).
    """
    return sum(estimate_tokens(m.get("content", "")) + MESSAGE_OVERHEAD_TOKENS for m in messages)


# --------------------------------------------------------------------------- #
# Assembled prompt
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AssembledPrompt:
    """One fully assembled request plus its token decomposition.

    ``breakdown`` is the explainability contract: ``total_prompt_tokens_est``
    always equals ``sum(breakdown.values())``, so a report can attribute the
    cost difference between two strategies to named knobs instead of guessing.
    """

    messages: list[dict[str, str]]
    system_tokens: int = 0
    few_shot_tokens: int = 0
    task_tokens: int = 0
    tool_tokens: int = 0
    total_prompt_tokens_est: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)
    applied: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_system_prompt(strategy: StrategySpec) -> tuple[str, str]:
    """Return ``(text, source)`` for the strategy's system prompt.

    An inline ``system_prompt`` override wins over the preset: it is the escape
    hatch for testing a specific brief without inventing a new preset name.
    """
    if strategy.system_prompt is not None:
        return strategy.system_prompt, "override"
    preset = strategy.system_prompt_preset
    return SYSTEM_PROMPTS.get(preset, ""), f"preset:{preset}"


def select_few_shot_examples(category: str, count: int) -> list[dict[str, Any]]:
    """Pick ``count`` exemplars, category-relevant ones first.

    Selection is deterministic (file order) so two runs of the same experiment
    assemble identical prompts.  Category matches come first, then the general
    pool fills the remainder — a partially relevant exemplar is still a better
    teacher than dropping the knob entirely.
    """
    if count <= 0:
        return []
    matched = [e for e in FEW_SHOT_EXAMPLES if category in e["categories"]]
    rest = [e for e in FEW_SHOT_EXAMPLES if e not in matched]
    return (matched + rest)[:count]


def resolve_max_output_tokens(strategy: StrategySpec, model: ModelSpec) -> int:
    """Output-token cap: a strategy override wins, otherwise the model default.

    The cap is part of the ablation surface because raising it changes both the
    answer length and the bill, and truncation is a quality confounder that
    should be visible in the config rather than implicit in the provider.
    """
    if strategy.max_output_tokens is not None:
        return strategy.max_output_tokens
    return model.max_output_tokens


def summarize_tool_payload(payload: str, *, max_chars: int = 360) -> str:
    """Compress a tool/context blob into a structural summary.

    JSON is summarised by shape (keys, sizes, first values) instead of being
    truncated mid-token: a truncated JSON blob is both misleading and expensive,
    while a shape summary keeps the information an agent actually needs.  A note
    records that compression happened so a reader never mistakes the summary for
    the real payload.
    """
    original_tokens = estimate_tokens(payload)
    summary: str
    try:
        parsed: Any = json.loads(payload)
    except (ValueError, TypeError):
        lines = [ln for ln in payload.splitlines() if ln.strip()]
        head = lines[:3]
        tail = lines[-1:] if len(lines) > 3 else []
        body = "\n".join(head + (["..."] if tail else []) + tail)
        summary = body[:max_chars]
    else:
        if isinstance(parsed, dict):
            parts: list[str] = []
            for key, value in list(parsed.items())[:8]:
                parts.append(f"- {key}: {_shape_of(value)}")
            summary = "object with keys:\n" + "\n".join(parts)
        elif isinstance(parsed, list):
            preview = ", ".join(_shape_of(v) for v in parsed[:5])
            summary = f"array of {len(parsed)} items; first items: {preview}"
        else:
            summary = f"{type(parsed).__name__}: {str(parsed)[:max_chars]}"
        if len(summary) > max_chars:
            summary = summary[: max_chars - 1].rstrip() + "…"
    return (
        "[tool output compressed: "
        f"{len(payload)} chars ≈ {original_tokens} tokens -> summary]\n" + summary
    )


def _shape_of(value: Any) -> str:
    """One-line structural description of a JSON value (no contents dump)."""
    if isinstance(value, dict):
        return f"object({len(value)} keys: {', '.join(list(value)[:4])})"
    if isinstance(value, list):
        return f"array({len(value)})"
    if isinstance(value, str):
        flat = value.replace("\n", " ").strip()
        return f"str({len(value)} chars: {flat[:40]!r})"
    return f"{type(value).__name__}({value})"


def _build_tool_block(task: Task, *, compress: bool) -> str | None:
    """Render ``task.tool_payload`` as a prompt block, optionally compressed."""
    payload = task.tool_payload
    if not payload:
        return None
    body = summarize_tool_payload(payload) if compress else payload
    return "TOOL OUTPUT (verbatim):\n```\n" + body + "\n```"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assemble_prompt(
    task: Task,
    strategy: StrategySpec,
    model: ModelSpec,
) -> AssembledPrompt:
    """Build the exact message list for one experiment cell.

    Every :class:`~tokbench.config.StrategySpec` knob is applied here and
    nowhere else, which is what makes the strategy ablation trustworthy: the
    runner never edits messages, so the only difference between two strategies
    for the same task is the difference this function produced.

    Returns an :class:`AssembledPrompt` whose ``breakdown`` sums to
    ``total_prompt_tokens_est`` and whose ``applied`` dict records every
    decision (including the ones that were *not* taken), so a report can explain
    a cost delta down to the knob.
    """
    breakdown: dict[str, int] = {
        "system": 0,
        "few_shot": 0,
        "cot": 0,
        "task": 0,
        "tool": 0,
        "overhead": 0,
    }
    applied: dict[str, Any] = {}

    # --- system prompt -----------------------------------------------------
    system_text, system_source = resolve_system_prompt(strategy)
    system_tokens = estimate_tokens(system_text)
    applied["system_source"] = system_source
    applied["system_preset"] = strategy.system_prompt_preset
    applied["system_chars"] = len(system_text)

    # --- chain-of-thought instruction --------------------------------------
    cot_text = COT_INSTRUCTION if strategy.cot else ""
    cot_tokens = estimate_tokens(cot_text)
    applied["cot"] = strategy.cot

    # --- few-shot ----------------------------------------------------------
    examples = select_few_shot_examples(task.category, strategy.few_shot)
    few_shot_messages: list[dict[str, str]] = []
    few_shot_tokens = 0
    for example in examples:
        for role in ("user", "assistant"):
            content = example[role]
            few_shot_messages.append({"role": role, "content": content})
            few_shot_tokens += estimate_tokens(content)
    applied["few_shot_requested"] = strategy.few_shot
    applied["few_shot_applied"] = len(examples)
    applied["few_shot_ids"] = [e["id"] for e in examples]

    # --- task + tool payload ----------------------------------------------
    task_text = task.prompt
    tool_block = _build_tool_block(task, compress=strategy.compress_tool_output)
    user_text = task_text if tool_block is None else f"{task_text}\n\n{tool_block}"
    task_tokens = estimate_tokens(task_text)
    tool_tokens = estimate_tokens(tool_block) if tool_block else 0
    applied["compress_tool_output"] = strategy.compress_tool_output
    applied["tool_payload_chars"] = len(task.tool_payload or "")
    applied["tool_block_chars"] = len(tool_block or "")

    # --- messages ----------------------------------------------------------
    messages: list[dict[str, str]] = []
    system_parts = [p for p in (system_text, cot_text) if p]
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_text})

    # --- accounting --------------------------------------------------------
    breakdown["system"] = system_tokens
    breakdown["few_shot"] = few_shot_tokens
    breakdown["cot"] = cot_tokens
    breakdown["task"] = task_tokens
    breakdown["tool"] = tool_tokens
    breakdown["overhead"] = MESSAGE_OVERHEAD_TOKENS * len(messages)
    total = sum(breakdown.values())

    # --- sampling knobs ----------------------------------------------------
    max_output_tokens = resolve_max_output_tokens(strategy, model)
    if strategy.max_output_tokens is None and task.max_output_tokens is not None:
        # A task-level cap is a hard requirement (the answer must fit the
        # grader's expectation), so it applies unless the strategy explicitly
        # overrides it.
        max_output_tokens = task.max_output_tokens
        output_source = "task"
    elif strategy.max_output_tokens is not None:
        output_source = "strategy"
    else:
        output_source = "model"
    applied["max_output_tokens"] = max_output_tokens
    applied["max_output_tokens_source"] = output_source

    temperature = strategy.temperature if strategy.temperature is not None else model.temperature
    applied["temperature"] = temperature
    applied["temperature_source"] = "strategy" if strategy.temperature is not None else "model"
    applied["task_id"] = task.id
    applied["task_category"] = task.category

    return AssembledPrompt(
        messages=messages,
        system_tokens=system_tokens,
        few_shot_tokens=few_shot_tokens,
        task_tokens=task_tokens,
        tool_tokens=tool_tokens,
        total_prompt_tokens_est=total,
        breakdown=breakdown,
        applied=applied,
    )


__all__ = [
    "COT_INSTRUCTION",
    "FEW_SHOT_EXAMPLES",
    "MESSAGE_OVERHEAD_TOKENS",
    "SYSTEM_PROMPTS",
    "AssembledPrompt",
    "assemble_prompt",
    "estimate_messages_tokens",
    "estimate_tokens",
    "resolve_max_output_tokens",
    "resolve_system_prompt",
    "select_few_shot_examples",
    "summarize_tool_payload",
]
