"""Deterministic offline adapter — the engine's test double and demo driver.

Why this is not a throwaway stub: every number in a tokbench report must be
reproducible.  A mock that returned ``random.randint`` would make the Pareto
frontier move between runs and destroy the project's central claim.  So the mock
derives *everything* — token counts, latency, injected failures, answer quality —
from ``sha256(seed + cache_key)``.  Identical inputs therefore produce
byte-identical records, while different (model, strategy, task) cells produce
different, plausible numbers.

What it emulates, and why each ratio was chosen:

* **input tokens** track the assembled prompt estimate with a provider-like
  drift (±6%), so the "long system prompt costs more" story shows up in the call
  log the same way it would against a real API.
* **cache behaviour**: a prompt prefix is "cacheable" when the cell has a system
  prompt or few-shot examples (i.e. a reusable prefix exists).  Cache reads land
  at 40-80% of input — the band real providers report for warm prefixes — and
  the first repeat of a cell pays cache *writes* instead, because that is what
  actually happens on a cold cache.
* **output tokens** are drawn from a normal distribution around a tier- and
  difficulty-dependent mean, clamped to the requested cap.  A real model
  truncates at ``max_output_tokens``, so the mock does too.
* **~3% retryable failures** are injected because the retry-overhead analysis is
  a headline feature of the report: without failures there is no retry data to
  analyse, and the retry code path would be untested in practice.  The draw is
  keyed on ``(cache_key, attempt_number)``, not on ``cache_key`` alone — otherwise
  a retried cell would re-draw the *same* verdict and either always fail or never
  fail, which would make retries either pointless or invisible.  See
  :meth:`MockAdapter._maybe_fail`.
* **answer text varies by tier** (``mock-small``/``medium``/``large``) so the
  grader produces a *spread* of quality scores.  A mock where every model scores
  1.0 would make the cost/quality Pareto chart a flat line.

The adapter never computes cost — that is :mod:`tokbench.pricing`'s job — and it
never touches the network.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from dataclasses import dataclass
from typing import Any

from ..errors import AdapterError, ErrorKind
from ..models import Usage
from ..prompt import estimate_tokens
from .base import AdapterRequest, AdapterResponse, BaseAdapter

#: Injected failure kinds, weighted by how often they occur in practice:
#: rate limits dominate, timeouts are next, 5xx is rare.
_FAILURE_KINDS: tuple[tuple[ErrorKind, float], ...] = (
    (ErrorKind.RATE_LIMIT, 0.55),
    (ErrorKind.TIMEOUT, 0.25),
    (ErrorKind.SERVER_ERROR, 0.20),
)

#: Model-tier behaviour.  ``output_ratio`` scales how much the model writes,
#: ``quality`` how complete/correct the generated answer looks to the grader,
#: ``latency`` how slow the call is.  The three tiers exist so quality scores
#: spread across the leaderboard instead of tying at 1.0.
_TIERS: dict[str, dict[str, float]] = {
    "small": {"output_ratio": 0.55, "quality": 0.55, "latency": 0.6},
    "medium": {"output_ratio": 0.85, "quality": 0.80, "latency": 1.0},
    "large": {"output_ratio": 1.15, "quality": 0.98, "latency": 1.6},
}

_TIER_ORDER: tuple[str, ...] = ("small", "medium", "large")

#: Identifier-like tokens that carry no topical signal, filtered out of the
#: prompt-identifier echo so the mock's answers do not read as a token dump.
_GENERIC_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "but",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "must",
        "should",
        "callers",
        "caller",
        "example",
        "note",
        "here",
        "there",
        "show",
        "write",
        "implement",
        "explain",
        "why",
        "how",
        "what",
        "use",
        "using",
        "avoid",
        "make",
        "sure",
        "only",
        "also",
        "then",
        "when",
        "where",
        "which",
        "while",
        "each",
        "every",
        "some",
        "any",
        "all",
        "not",
        "our",
        "your",
        "its",
        "it",
        "we",
        "you",
        "they",
        "one",
        "two",
        "three",
    }
)

#: Rough seconds-per-output-token by tier, used only for plausible latency.
_SECONDS_PER_OUTPUT_TOKEN = 0.004


#: Substring -> tier rules, checked in order.  Explicit capability classes beat a
#: hash: a benchmark whose "large" tier silently landed on ``gpt-4.1-nano`` would
#: produce a leaderboard ordered by noise.  Order matters — the first match wins,
#: so the most specific family markers come first.
_TIER_MARKERS: tuple[tuple[str, str], ...] = (
    # explicitly small / cheap / distilled variants
    ("nano", "small"),
    ("mini", "small"),
    ("small", "small"),
    ("haiku", "small"),
    ("flash", "small"),
    ("lite", "small"),
    ("tiny", "small"),
    ("7b", "small"),
    ("8b", "small"),
    # explicitly large / frontier variants
    ("opus", "large"),
    ("sonnet", "large"),
    ("large", "large"),
    ("pro", "large"),
    ("reasoner", "large"),
    ("reasoning", "large"),
    ("70b", "large"),
    ("405b", "large"),
    # mid-tier standard models
    ("medium", "medium"),
    ("chat", "medium"),
    ("turbo", "medium"),
    ("4o", "medium"),
)


def _tier_for_model(model: str) -> str:
    """Infer a behavioural tier from the model name.

    Resolution order:

    1. An explicit ``mock-small`` / ``mock-medium`` / ``mock-large`` name wins.
    2. Known family markers (``nano``/``mini`` -> small, ``opus``/``reasoner`` ->
       large, ``chat``/``turbo`` -> medium), most specific first.
    3. Otherwise the name is hashed, so an unrecognised alias still maps to a
       *stable* tier instead of changing behaviour between runs.

    Step 2 matters for the report's credibility: a hash-only mapping put
    ``gpt-4.1-nano`` in the "large" tier and ``deepseek-chat`` in "small", which
    would have made the leaderboard order an artefact of the hash.
    """
    lowered = (model or "").lower()
    for tier in _TIER_ORDER:
        if f"mock-{tier}" in lowered:
            return tier
    for marker, tier in _TIER_MARKERS:
        if marker in lowered:
            return tier
    digest = hashlib.sha256(lowered.encode("utf-8")).digest()
    return _TIER_ORDER[digest[0] % len(_TIER_ORDER)]


def _stable_seed(seed: int, cache_key: str) -> int:
    """Fold the adapter seed and the request identity into one integer seed.

    ``cache_key`` is the only identity a request carries that is stable across
    processes, so it must dominate the seed.  The adapter ``seed`` lets two
    experiments share a task set but not an identical noise stream.
    """
    payload = f"{seed}\x00{cache_key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(slots=True)
class _DeterministicNoise:
    """A tiny seeded noise source built from sha256.

    ``random.Random`` is deterministic per-seed, but its stream depends on the
    *order* of calls; deriving each draw from an explicit label makes the mock
    robust to future code reordering (drawing a failure first vs. last cannot
    change the token counts).
    """

    seed: int

    def unit(self, label: str) -> float:
        """Deterministic float in ``[0, 1)`` for a named draw."""
        digest = hashlib.sha256(f"{self.seed}\x00{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def normal(self, label: str, mean: float, sd: float) -> float:
        """Deterministic normal draw via Box-Muller on two labelled uniforms."""
        u1 = max(self.unit(f"{label}:a"), 1e-12)
        u2 = self.unit(f"{label}:b")
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + sd * z

    def rng(self, label: str) -> random.Random:
        """Seeded ``Random`` for text generation (order-stable within a label)."""
        return random.Random(_stable_seed(self.seed, label))


# --------------------------------------------------------------------------- #
# Answer synthesis
# --------------------------------------------------------------------------- #
_CODE_HINTS: tuple[tuple[str, str], ...] = (
    ("sql", "sql"),
    ("select", "sql"),
    ("typescript", "ts"),
    ("python", "python"),
    ("rust", "rust"),
    ("golang", "go"),
    ("go ", "go"),
    ("bash", "bash"),
    ("shell", "bash"),
)


def _guess_language(text: str) -> str:
    lowered = (text or "").lower()
    for needle, language in _CODE_HINTS:
        if needle in lowered:
            return language
    return "python"


def _keyword_list(text: str, limit: int) -> list[str]:
    """Stable content keywords from the prompt, used to make the answer topical."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text or "")
    seen: dict[str, None] = {}
    for word in words:
        seen.setdefault(word.lower(), None)
    return list(seen)[:limit]


def _answer_skeleton(language: str) -> str:
    """An idiomatic code body for the requested language.

    These are deliberately *plausible* rather than generic: each scaffold uses the
    constructs a competent answer in that language actually contains (``except``
    and ``raise`` in Python, ``context.Context`` and ``chan`` in Go, ``set -euo
    pipefail`` in bash).  A grader that checks required substrings therefore gives
    a language-appropriate answer real credit instead of scoring every model as
    if it had written the wrong language.
    """
    if language == "sql":
        return (
            "```sql\n"
            "WITH ranked AS (\n"
            "    SELECT t.id, t.created_at,\n"
            "           ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY t.created_at DESC) AS rn\n"
            "    FROM target_table AS t\n"
            "    JOIN related_table AS r ON r.target_id = t.id\n"
            "    WHERE t.created_at >= :since\n"
            ")\n"
            "SELECT id, created_at\n"
            "FROM ranked\n"
            "WHERE rn = 1\n"
            "ORDER BY created_at DESC\n"
            "LIMIT 100;\n"
            "```"
        )
    if language == "ts":
        return (
            "```ts\n"
            "export interface Options {\n"
            "  leading?: boolean;\n"
            "  cancel(): void;\n"
            "  flush(): void;\n"
            "}\n"
            "\n"
            "export function debounce<F extends (...args: any[]) => void>(\n"
            "  fn: F,\n"
            "  waitMs: number,\n"
            "): F & Options {\n"
            "  let timer: ReturnType<typeof setTimeout> | null = null;\n"
            "  let lastArgs: Parameters<F> | null = null;\n"
            "\n"
            "  const cancel = (): void => {\n"
            "    if (timer !== null) {\n"
            "      clearTimeout(timer);\n"
            "      timer = null;\n"
            "    }\n"
            "    lastArgs = null;\n"
            "  };\n"
            "\n"
            "  const flush = (): void => {\n"
            "    if (lastArgs !== null) {\n"
            "      const args = lastArgs;\n"
            "      cancel();\n"
            "      fn(...args);\n"
            "    }\n"
            "  };\n"
            "\n"
            "  const debounced = function (this: unknown, ...args: Parameters<F>): void {\n"
            "    lastArgs = args;\n"
            "    if (timer !== null) clearTimeout(timer);\n"
            "    timer = setTimeout(() => {\n"
            "      timer = null;\n"
            "      lastArgs = null;\n"
            "      fn.apply(this, args);\n"
            "    }, waitMs);\n"
            "  } as F & Options;\n"
            "\n"
            "  debounced.cancel = cancel;\n"
            "  debounced.flush = flush;\n"
            "  return debounced;\n"
            "}\n"
            "```"
        )
    if language == "go":
        return (
            "```go\n"
            "package pool\n"
            "\n"
            "import (\n"
            '\t"context"\n'
            '\t"sync"\n'
            ")\n"
            "\n"
            "type Job func(context.Context, any) error\n"
            "\n"
            "type Pool struct {\n"
            "\tjobs    chan Job\n"
            "\twg      sync.WaitGroup\n"
            "\tmu      sync.Mutex\n"
            "\terrs    []error\n"
            "}\n"
            "\n"
            "func New(ctx context.Context, workers, queue int) *Pool {\n"
            "\tp := &Pool{jobs: make(chan Job, queue)}\n"
            "\tfor i := 0; i < workers; i++ {\n"
            "\t\tp.wg.Add(1)\n"
            "\t\tgo func() {\n"
            "\t\t\tdefer p.wg.Done()\n"
            "\t\t\tfor job := range p.jobs {\n"
            "\t\t\t\tif err := job(ctx, nil); err != nil {\n"
            "\t\t\t\t\tp.mu.Lock()\n"
            "\t\t\t\t\tp.errs = append(p.errs, err)\n"
            "\t\t\t\t\tp.mu.Unlock()\n"
            "\t\t\t\t}\n"
            "\t\t\t}\n"
            "\t\t}()\n"
            "\t}\n"
            "\treturn p\n"
            "}\n"
            "\n"
            "func (p *Pool) Submit(ctx context.Context, job Job) error {\n"
            "\tselect {\n"
            "\tcase p.jobs <- job:\n"
            "\t\treturn nil\n"
            "\tcase <-ctx.Done():\n"
            "\t\treturn ctx.Err()\n"
            "\t}\n"
            "}\n"
            "\n"
            "func (p *Pool) Close() []error {\n"
            "\tclose(p.jobs)\n"
            "\tp.wg.Wait()\n"
            "\treturn p.errs\n"
            "}\n"
            "```"
        )
    if language == "rust":
        return (
            "```rust\n"
            "use std::path::Path;\n"
            "\n"
            "use serde::Deserialize;\n"
            "use thiserror::Error;\n"
            "\n"
            "#[derive(Debug, Error)]\n"
            "pub enum ConfigError {\n"
            '    #[error("missing config file: {0}")]\n'
            "    Missing(String),\n"
            '    #[error("parse failure: {0}")]\n'
            "    Parse(#[from] toml::de::Error),\n"
            '    #[error("validation failure: {0}")]\n'
            "    Validation(String),\n"
            "}\n"
            "\n"
            "#[derive(Debug, Deserialize)]\n"
            "pub struct Config {\n"
            "    pub timeout_secs: u64,\n"
            "    pub log_level: String,\n"
            "}\n"
            "\n"
            "impl Config {\n"
            "    pub fn load(path: &Path) -> Result<Config, ConfigError> {\n"
            "        let text = std::fs::read_to_string(path)\n"
            "            .map_err(|_| ConfigError::Missing(path.display().to_string()))?;\n"
            "        let cfg: Config = toml::from_str(&text)?;\n"
            "        if cfg.timeout_secs == 0 {\n"
            '            return Err(ConfigError::Validation("timeout must be > 0".into()));\n'
            "        }\n"
            "        Ok(cfg)\n"
            "    }\n"
            "}\n"
            "```"
        )
    if language == "bash":
        return (
            "```bash\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "\n"
            'log_dir="${1:?usage: $0 <log-dir>}"\n'
            'exec 9>"$log_dir/.rotate.lock"\n'
            'flock -n 9 || { echo "rotation already running" >&2; exit 1; }\n'
            "\n"
            "find \"$log_dir\" -maxdepth 1 -type f -name '*.log' -mtime +1 -print0 \\\n"
            "  | xargs -0 -r gzip -f\n"
            "find \"$log_dir\" -maxdepth 1 -type f -name '*.log.gz' -mtime +14 -delete\n"
            "\n"
            "pkill -HUP -f 'app-server' || true\n"
            "```"
        )
    return (
        "```python\n"
        "import time\n"
        "from typing import Any, Callable, TypeVar\n"
        "\n"
        'T = TypeVar("T")\n'
        "\n"
        "def retry(\n"
        "    retry_on: tuple[type[BaseException], ...] = (Exception,),\n"
        "    attempts: int = 3,\n"
        "    base_s: float = 0.5,\n"
        "    max_elapsed_s: float = 30.0,\n"
        ") -> Callable[[Callable[..., T]], Callable[..., T]]:\n"
        '    """Retry only the given exception types, with exponential backoff."""\n'
        "    def decorate(fn: Callable[..., T]) -> Callable[..., T]:\n"
        "        def wrapper(*args: Any, **kwargs: Any) -> T:\n"
        "            started = time.monotonic()\n"
        "            last: BaseException | None = None\n"
        "            for attempt in range(1, attempts + 1):\n"
        "                try:\n"
        "                    return fn(*args, **kwargs)\n"
        "                except retry_on as exc:\n"
        "                    last = exc\n"
        "                    if attempt == attempts:\n"
        "                        break\n"
        "                    remaining = max_elapsed_s - (time.monotonic() - started)\n"
        "                    if remaining <= 0:\n"
        "                        break\n"
        "                    time.sleep(min(base_s * 2 ** (attempt - 1), remaining))\n"
        '            raise RuntimeError(f"retry exhausted after {attempts}") from last\n'
        "        return wrapper\n"
        "    return decorate\n"
        "```"
    )


def _prompt_identifiers(text: str, limit: int) -> list[str]:
    """Identifier-like tokens from the prompt, most distinctive first.

    A real model reuses the vocabulary of the request it was given (function
    names, type names, library names).  Echoing those identifiers is how the mock
    stays *task-aware* without pretending to reason: it raises the answer's
    topical coverage, which is exactly the signal a substring grader measures.
    """
    candidates = re.findall(r"`([A-Za-z_][\w.:]*)`|\b([A-Z][A-Za-z0-9_]{2,})\b", text or "")
    seen: dict[str, None] = {}
    for backticked, capitalised in candidates:
        token = backticked or capitalised
        if token and token.lower() not in _GENERIC_IDENTIFIERS:
            seen.setdefault(token, None)
    return list(seen)[:limit]


def _competence(tier: str, quality: float, leverage: float = 0.0) -> float:
    """Combine the tier prior, the per-call draw and the prompt's leverage.

    Why all three: the tier says what the *model class* can do, ``quality`` says
    how well this particular call went (a strong model can still have a bad call),
    and ``leverage`` says how much the *strategy* helped.  The tier sets the
    centre, the draw perturbs it, and the strategy effect shifts it — then the
    result is clamped to a plausible band per tier so a "large" model never
    randomly looks incompetent, and a great prompt cannot make a weak model
    frontier-class.
    """
    base = _TIERS[tier]["quality"]
    # quality is drawn around ``base`` with sd 0.12; re-centre on the tier and let
    # the draw supply ±0.15 of variation around it.
    blended = base + (quality - base) * 0.6 + leverage
    floor, ceiling = _COMPETENCE_BANDS[tier]
    return max(floor, min(ceiling, blended))


#: Hard per-tier bounds on simulated competence.  These are what guarantee a
#: monotonic leaderboard: whatever the per-call draw does, a small model cannot
#: out-score a large one on average.  The bands deliberately overlap slightly at
#: the edges so an occasional bad call by a strong model is representable.
_COMPETENCE_BANDS: dict[str, tuple[float, float]] = {
    "small": (0.25, 0.62),
    "medium": (0.58, 0.88),
    "large": (0.85, 1.00),
}


def _select_hints(hints: list[str], competence: float, rng: random.Random) -> list[str]:
    """Choose which grading targets this answer will actually contain.

    The count is ``round(competence * len(hints))``, then jittered by one so two
    calls at the same competence do not produce byte-identical answers.  Selection
    is shuffled but seeded, so it is reproducible.

    This is the honest way to model a real model's behaviour: a strong model
    mentions nearly all the required constructs, a weak one misses some.  The
    mock is not *finding* the answers — the runner supplies them as simulation
    hints (see ``runner._build_request``) precisely because a behavioural
    simulator of "a model that answered correctly" needs to know the target.
    """
    if not hints:
        return []
    target = int(round(competence * len(hints)))
    # ±1 jitter, but never below zero or above the full set.
    target += rng.choice((-1, 0, 0, 1))
    target = max(0, min(len(hints), target))
    if target == 0:
        return []
    order = list(hints)
    rng.shuffle(order)
    return order[:target]


def _weave_hints(hints: list[str], language: str) -> str:
    """Render selected grading targets as a natural "key points" section.

    Deliberately *not* a bare list dump: the hints are phrased as the concrete
    constructs the answer relies on, grouped into a sentence that reads like a
    model summarising its own solution.  The grader checks substrings, and a
    genuine answer does contain its own key identifiers — so this is both
    plausible prose and the honest signal.
    """
    if not hints:
        return ""
    quoted = [f"`{hint}`" for hint in hints]
    if len(quoted) == 1:
        body = quoted[0]
    elif len(quoted) == 2:
        body = f"{quoted[0]} and {quoted[1]}"
    else:
        body = ", ".join(quoted[:-1]) + f" and {quoted[-1]}"
    return (
        f"Key constructs in the solution: {body}. "
        "Each is load-bearing — removing any one of them changes the behaviour "
        "described above."
    )


def _prompt_leverage(messages: list[dict[str, str]]) -> float:
    """Estimate how much the *prompting strategy* should help answer quality.

    This is the mock's model of the experimental hypothesis.  The whole point of
    the strategy ablation is that how you ask changes how well the model answers,
    so a simulator that ignored the prompt would make every strategy score
    identically and the ablation table would be meaningless.

    The effects are deliberately small and signed, matching published prompting
    results rather than wishful thinking:

    * **few-shot exemplars** — the strongest lever (+0.02 per exemplar, capped at
      +0.05): showing the format works, but returns diminish quickly.
    * **chain-of-thought instruction** — +0.02: helps on multi-step tasks, and
      slightly *hurts* very short answers by adding preamble.
    * **a substantial system prompt** — +0.01: sets constraints and conventions.
    * **compressed tool output** — −0.03: the model loses detail it needed, which
      is exactly the cost side of that strategy.

    All effects are observable from the assembled messages alone, so this needs no
    extra plumbing and stays consistent with what a real provider would see.
    """
    leverage = 0.0
    system_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    exemplars = sum(1 for m in messages if m.get("role") == "assistant")
    if exemplars:
        leverage += min(0.05, 0.02 * exemplars)
    lowered = system_text.lower()
    if "step by step" in lowered or "step-by-step" in lowered:
        leverage += 0.02
    if len(system_text) >= 1200:
        leverage += 0.01
    # A compressed blob is announced by prompt.summarize_tool_payload; its presence
    # means the model is working from a summary rather than the raw data.
    if "tool output compressed" in lowered:
        leverage -= 0.03
    return leverage


def _compose_answer(
    request: AdapterRequest,
    *,
    tier: str,
    quality: float,
    rng: random.Random,
    output_tokens: int,
) -> str:
    """Build a task-aware answer whose completeness tracks simulated competence.

    The mock cannot *be* correct, so it models *how* a model of a given capability
    answers: it emits idiomatic code for the requested language, covers a share of
    the answer's key constructs proportional to its competence, and adds
    assumptions, edge cases and a closing summary as the tier rises.  Weak calls
    truncate or fall back to a vague non-answer — exactly how a small or
    token-starved model fails in practice.

    Grading targets arrive in ``request.params["hints"]`` / ``["reference"]``.
    This is intentional and documented in ``runner._build_request``: a *simulator*
    of "a model that answered correctly" has to know what correct looks like, and
    the task's ``must_include`` / ``reference`` is the only such source.  Real
    adapters strip these keys before serialising a provider payload.
    """
    user_text = ""
    for message in reversed(request.messages):
        if message.get("role") == "user":
            user_text = message.get("content", "")
            break
    keywords = _keyword_list(user_text, limit=8)
    identifiers = _prompt_identifiers(user_text, limit=6)
    language = _guess_language(user_text)
    subject = ", ".join(keywords[:3]) if keywords else "the requested behaviour"
    tier_index = _TIER_ORDER.index(tier)

    params = request.params or {}
    hints = [h for h in (params.get("hints") or []) if isinstance(h, str) and h.strip()]
    reference = str(params.get("reference") or "").strip()
    competence = _competence(tier, quality, _prompt_leverage(request.messages))

    parts: list[str] = [f"Here is a complete implementation for {subject}."]
    parts.append(_answer_skeleton(language))

    # A short canonical reference (exact-match task): a competent model reproduces
    # it verbatim; a weak one produces a plausible but wrong variant.
    if reference:
        if competence >= 0.62:
            parts.append(f"`{reference}`")
        else:
            parts.append(f"`{_mutate_reference(reference, rng)}`")

    # Mid tier and up explain the design and reuse the request's own identifiers.
    if tier_index >= 1:
        design = (
            "The approach is deliberate: it keeps the hot path allocation-free, "
            "returns errors instead of panicking, and leaves the caller in control "
            "of cancellation and shutdown ordering."
        )
        if identifiers:
            design += (
                " The request's `"
                + "`, `".join(identifiers[:3])
                + "` surface is honoured directly."
            )
        parts.append(design)

    # The graded constructs: emit the share this call's competence earns.
    selected = _select_hints(hints, competence, rng)
    woven = _weave_hints(selected, language)
    if woven:
        parts.append(woven)

    if tier_index >= 2:
        edge_cases = [
            "an empty input returns an empty result rather than raising",
            "a cancelled context stops new work and drains in-flight jobs",
            "concurrent callers cannot observe a half-updated state",
            "partial failure is reported per item, not swallowed",
        ]
        rng.shuffle(edge_cases)
        parts.append("Edge cases handled: " + "; ".join(edge_cases[:3]) + ".")
        if keywords:
            parts.append(
                "Assumptions: `"
                + "`, `".join(keywords[:4])
                + "` behave as described in the request; anything else is "
                "validated at the boundary and rejected with a typed error."
            )
        parts.append(
            "Final answer: the code above is complete and runnable as shown; "
            "the notes list the assumptions it relies on."
        )

    # Low-competence calls degrade the way real weak models do: either a truncated
    # attempt or a vague non-answer.  This is what stops every tier scoring 1.0.
    if competence < 0.35 and rng.random() < 0.40:
        parts = [
            parts[0],
            "The general idea is to iterate over the input, group the values, "
            "and return the result. Adjust names and types as needed.",
        ]
    elif competence < 0.55 and rng.random() < 0.25:
        # Drop the trailing sections: plausible but incomplete.
        parts = parts[: 2 + tier_index]

    text = "\n\n".join(parts)

    # Respect the provider-style output cap by truncating at a line boundary.
    target_chars = max(output_tokens, 1) * 4
    if len(text) > target_chars:
        cut = text.rfind("\n", 0, target_chars)
        text = text[: cut if cut > 0 else target_chars].rstrip() + "\n…"
    return text


def _mutate_reference(reference: str, rng: random.Random) -> str:
    """Produce a plausible near-miss for a low-competence exact-match answer.

    A wrong answer should look like a *wrong attempt*, not like noise: one token
    is swapped for a near-neighbour so the answer is clearly related but does not
    match.  This keeps exact-match grading meaningful (a small model gets partial
    similarity credit, not a random score).
    """
    substitutions = (
        ("+", "-"),
        ("-", "+"),
        ("==", "!="),
        ("!=", "=="),
        ("<", "<="),
        (">", ">="),
        ("SELECT", "select"),
        ("1", "2"),
        ("true", "false"),
        ("True", "False"),
    )
    for needle, replacement in substitutions:
        if needle in reference:
            return reference.replace(needle, replacement, 1)
    # No obvious swap: drop the final token, which is the commonest near-miss.
    parts = reference.split()
    if len(parts) > 1:
        return " ".join(parts[:-1])
    return reference + " " if reference else "unknown"


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class MockAdapter(BaseAdapter):
    """Deterministic, offline adapter that emulates a chat-completions provider.

    Args:
        spec: the :class:`~tokbench.config.ModelSpec` under test (its ``model``
            name selects the behavioural tier).
        fail_rate: probability of an injected *retryable* failure per attempt.
            Defaults to 3% so retry analysis has real data without making runs
            mostly noise.
        seed: experiment seed; mixed with ``request.cache_key`` so identical
            cells are byte-identical across runs.

    The adapter reports ``usage_estimated=False`` for its synthetic usage: it is
    *synthetic by definition*, but it is not an estimate of a real call, so
    flagging it as estimated would wrongly imply a provider under-reported.

    Determinism has two layers, and the split is deliberate:

    * **Body** (token counts, text, latency) is seeded by ``(seed, cache_key)``
      only, so the same cell always produces the same numbers — that is what
      makes a run reproducible and what lets a report be re-derived.
    * **Failure verdict** is seeded by ``(seed, cache_key, attempt_index)``, so a
      retried cell draws a *fresh* verdict.  Seeding it by ``cache_key`` alone
      would mean a failing cell fails identically on every attempt: retries could
      never succeed, and the retry-overhead analysis would report a 100% retry
      failure rate that no real provider exhibits.
    """

    provider = "mock"
    supports_cache_accounting = True
    requires_credentials = False

    def __init__(
        self,
        spec: Any = None,
        *,
        fail_rate: float = 0.03,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(spec, **kwargs)
        if not 0.0 <= fail_rate <= 1.0:
            raise ValueError("fail_rate must be within [0, 1]")
        self.fail_rate = fail_rate
        self.seed = seed
        model_name = getattr(spec, "model", "") or getattr(spec, "id", "") or "mock-medium"
        self.model_name = str(model_name)
        self.tier = _tier_for_model(self.model_name)
        self._tier_cfg = _TIERS[self.tier]
        #: Per-cell attempt counter, keyed by ``cache_key``.  The runner retries a
        #: cell sequentially on one thread, so this is exactly the attempt index;
        #: the lock is for the (normal) case of many cells in flight at once.
        self._attempts: dict[str, int] = {}
        self._attempts_lock = threading.Lock()

    # ------------------------------------------------------------------ api
    def complete(self, request: AdapterRequest) -> AdapterResponse:
        """Produce one synthetic but deterministic completion.

        Raises:
            AdapterError: on the ~``fail_rate`` of attempts that emulate a
                transient provider failure, with a retryable
                :class:`~tokbench.errors.ErrorKind` so the runner's retry policy
                has something real to act on.
        """
        key = request.cache_key or self._fallback_key(request)
        attempt = self._next_attempt(key)
        # Body noise: independent of ``attempt``, so the same cell always reports
        # the same tokens for a given attempt *shape*.  Only the failure verdict
        # below is attempt-sensitive.
        noise = _DeterministicNoise(_stable_seed(self.seed, key))

        failure = self._maybe_fail(key, attempt)
        if failure is not None:
            raise failure

        usage = self._synthesise_usage(request, noise)
        quality = self._synthesise_quality(noise, request)
        rng = noise.rng("text")
        text = _compose_answer(
            request,
            tier=self.tier,
            quality=quality,
            rng=rng,
            output_tokens=usage.output_tokens,
        )

        latency = self._synthesise_latency(usage, noise)
        return AdapterResponse(
            text=text,
            usage=usage,
            finish_reason=self._finish_reason(usage, request),
            model=self.model_name,
            provider=self.provider,
            usage_estimated=False,
            raw={"mock": True, "tier": self.tier, "seed": self.seed},
            extra={"latency_ms": round(latency, 3), "simulated_quality": round(quality, 4)},
        )

    def preflight(self) -> list[str]:
        """Always ready — the mock needs neither credentials nor network."""
        return []

    # ------------------------------------------------------------- internals
    def _next_attempt(self, key: str) -> int:
        """Return the 1-based attempt index for ``key`` and advance the counter.

        Tracking this in the adapter (rather than taking it from the request) keeps
        :class:`~tokbench.adapters.base.AdapterRequest` unchanged: attempt
        numbering is a *retry* concept, and retry belongs to the runner.  The mock
        only needs it to decorrelate failure draws across attempts.
        """
        with self._attempts_lock:
            attempt = self._attempts.get(key, 0) + 1
            self._attempts[key] = attempt
            return attempt

    @staticmethod
    def _fallback_key(request: AdapterRequest) -> str:
        """Identity for requests that arrived without a ``cache_key``.

        Hashing the messages keeps determinism even for a hand-built request in
        a unit test, which is the only situation where ``cache_key`` is absent.
        """
        joined = "\x1f".join(
            f"{m.get('role', '')}:{m.get('content', '')}" for m in request.messages
        )
        return hashlib.sha256(f"{request.model}\x1f{joined}".encode()).hexdigest()

    def _maybe_fail(self, key: str, attempt: int) -> AdapterError | None:
        """Return an injected failure for this attempt, or ``None``.

        The verdict is drawn from a *separate* noise stream keyed on
        ``(cache_key, attempt)``, for two reasons:

        1. **Retries must be able to succeed.**  With a ``cache_key``-only draw, a
           failing cell fails on every attempt, so ``usage_total`` would always be
           ``max_attempts`` times ``usage_first_attempt`` and the retry-overhead
           metric would be a constant rather than a measurement.
        2. **Success must not be perturbed.**  Keeping the failure stream separate
           from the body stream means the token counts of a successful attempt do
           not change when the failure rate (or the seed's failure draws) change.

        The first attempt of every cell is *always* the draw that matters for the
        headline failure rate, so at the default 3% roughly 3% of cells fail once
        and are retried.
        """
        if self.fail_rate <= 0.0:
            return None
        failure_noise = _DeterministicNoise(_stable_seed(self.seed, f"{key}\x00attempt{attempt}"))
        if failure_noise.unit("failure") >= self.fail_rate:
            return None
        kind = self._pick_failure_kind(failure_noise)
        message = {
            ErrorKind.RATE_LIMIT: "mock: 429 rate limit exceeded (injected)",
            ErrorKind.TIMEOUT: "mock: request timed out after 30s (injected)",
            ErrorKind.SERVER_ERROR: "mock: 503 upstream unavailable (injected)",
        }[kind]
        return AdapterError(message, kind=kind)

    @staticmethod
    def _pick_failure_kind(noise: _DeterministicNoise) -> ErrorKind:
        """Weighted, deterministic choice of the injected failure class."""
        roll = noise.unit("failure:kind")
        cumulative = 0.0
        for kind, weight in _FAILURE_KINDS:
            cumulative += weight
            if roll <= cumulative:
                return kind
        return _FAILURE_KINDS[-1][0]  # pragma: no cover - float exhaustion guard

    def _synthesise_usage(self, request: AdapterRequest, noise: _DeterministicNoise) -> Usage:
        """Derive plausible token counts from the prompt estimate.

        Ratios are documented in the module docstring.  ``input_tokens`` includes
        cached tokens (the project-wide convention), so the cache counters are
        drawn as subsets and clamped to the input total.
        """
        prompt_est = max(request.prompt_tokens_est or 0, 1)
        if prompt_est <= 1:
            # No estimate supplied (hand-built request): fall back to counting
            # the actual message content so the numbers stay meaningful.
            prompt_est = sum(estimate_tokens(m.get("content", "")) + 4 for m in request.messages)

        drift = noise.normal("input_drift", mean=1.0, sd=0.06)
        input_tokens = max(1, int(round(prompt_est * max(drift, 0.5))))

        # A reusable prefix exists when there is a system prompt or few-shot
        # examples; only then does prompt caching apply at all.
        has_prefix = any(m.get("role") == "system" for m in request.messages) or any(
            m.get("role") == "assistant" for m in request.messages
        )
        cache_read = 0
        cache_write = 0
        if has_prefix:
            cold = noise.unit("cache:cold") < 0.25
            if cold:
                cache_write = int(round(input_tokens * noise.unit("cache:write") * 0.6))
            else:
                ratio = 0.40 + noise.unit("cache:read") * 0.40  # 40%..80%
                cache_read = int(round(input_tokens * ratio))
        cache_read = min(cache_read, input_tokens)
        cache_write = min(cache_write, input_tokens - cache_read)

        max_out = max(request.max_output_tokens or 256, 1)
        mean_out = max_out * self._tier_cfg["output_ratio"] * 0.55
        sd_out = max(mean_out * 0.25, 1.0)
        output_tokens = int(round(noise.normal("output", mean=mean_out, sd=sd_out)))
        output_tokens = max(1, min(output_tokens, max_out))

        reasoning = 0
        if "reason" in self.model_name.lower():
            # Reasoning models bill hidden tokens on top of the visible answer.
            reasoning = int(round(output_tokens * (0.4 + noise.unit("reasoning") * 0.8)))

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
        )

    def _synthesise_quality(self, noise: _DeterministicNoise, request: AdapterRequest) -> float:
        """Draw the simulated answer quality for text generation.

        Centred on the tier's quality with a modest spread so quality scores are
        not identical within a tier (a real model varies by task).  The value is
        *only* used to shape the generated text; the recorded quality is always
        the grader's own output.
        """
        base = self._tier_cfg["quality"]
        return max(0.05, min(1.0, noise.normal("quality", mean=base, sd=0.12)))

    def _synthesise_latency(self, usage: Usage, noise: _DeterministicNoise) -> float:
        """Plausible wall-clock latency in ms for the simulated call.

        Dominated by output-token count (decode is serial) with a small jittered
        base, which is the shape real chat-completion latency has.
        """
        base_ms = 120.0 * self._tier_cfg["latency"]
        decode_ms = usage.output_tokens * _SECONDS_PER_OUTPUT_TOKEN * 1000.0
        jitter = 0.85 + noise.unit("latency") * 0.30
        return (base_ms + decode_ms) * jitter

    @staticmethod
    def _finish_reason(usage: Usage, request: AdapterRequest) -> str:
        """``length`` when the answer hit the cap, otherwise ``stop``."""
        if usage.output_tokens >= (request.max_output_tokens or 0) > 0:
            return "length"
        return "stop"


__all__ = ["MockAdapter"]
