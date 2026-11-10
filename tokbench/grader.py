"""Quality grading — turn a model answer into a normalised ``[0, 1]`` score.

Why this exists as its own module: a token-efficiency benchmark is worthless
without a quality axis.  "Cheapest tokens" is trivially achieved by returning
nothing, so every cost number has to be paired with a quality number produced by
a grader that is **deterministic, offline and inspectable**.  No LLM judge here
— an LLM judge would add cost and non-reproducibility to the very thing we are
measuring.

Grading is deliberately three-tiered, because one metric cannot fit every task:

* ``exact_match`` — a short, unambiguous reference exists (a command, a magic
  string, a single number).  Graded on a normalised comparison; partial credit
  comes from edit similarity so a near-miss is visibly better than a miss.
* ``contains`` — the task declares ``must_include`` substrings.  Score is the
  fraction present, which makes "answered 2 of 3 required points" legible
  instead of collapsing to a pass/fail bit.
* ``heuristic`` — free-form answers with no reference: structure, completeness,
  keyword coverage and refusal detection, combined with documented weights.

``kind="auto"`` picks the tier from the task's own fields so a task set does not
need to carry grading metadata.  Every score carries a ``rationale`` and
``details`` so a report can explain *why* a model scored low.
"""

from __future__ import annotations

import difflib
import re

from .models import GraderKind, QualityScore, Task

# --------------------------------------------------------------------------- #
# Thresholds (documented constants beat magic numbers in a review)
# --------------------------------------------------------------------------- #
#: A reference shorter than this is specific enough for exact-match grading.
#: Longer references are prose, where exact match would be meaningless.
EXACT_MATCH_MAX_REFERENCE_CHARS = 64

#: Weights for the free-form heuristic.  Coverage dominates because *saying the
#: right things* matters more than formatting; structure and length are support
#: signals, not the objective.
HEURISTIC_WEIGHTS: dict[str, float] = {
    "coverage": 0.45,
    "structure": 0.25,
    "completeness": 0.20,
    "focus": 0.10,
}

#: A partial exact-match score is capped below 1.0 so only a true match passes.
EXACT_PARTIAL_CEILING = 0.75

#: Character length window for a "complete enough" free-form answer.
COMPLETENESS_MIN_CHARS = 160
COMPLETENESS_SATURATION_CHARS = 700
COMPLETENESS_MAX_CHARS = 8000

#: Words that carry no grading signal when checking keyword coverage.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "please",
        "write",
        "write",
        "me",
        "give",
        "show",
        "using",
        "use",
        "used",
        "make",
        "sure",
        "need",
        "want",
        "want",
        "it's",
        "don't",
        "doesn't",
        "let's",
        "also",
        "may",
        "might",
        "must",
        "shall",
        "can't",
        "won't",
    ]
)

#: Refusal / hedging openers that indicate the model declined instead of answering.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*i\s+(?:can\s*not|can't|cannot|am\s+unable|won't)\b",
        r"^\s*i'?m\s+sorry\b",
        r"^\s*sorry,?\s+(?:but\s+)?i\b",
        r"^\s*as\s+an\s+ai\b",
        r"^\s*i\s+do\s+not\s+have\s+(?:access|the\s+ability)\b",
        r"\bi\s+(?:can\s*not|can't)\s+(?:help|assist|comply)\b",
    )
)

_FENCE_RE = re.compile(r"```")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase.

    Graders must not reward cosmetic differences: trailing newlines, double
    spaces and casing are formatting, not correctness.
    """
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _compact(text: str) -> str:
    """Normalise and also strip all whitespace.

    Used for short-reference comparison so ``return x + 1`` matches
    ``return x+1`` — a difference no engineer would call wrong.
    """
    return re.sub(r"\s+", "", (text or "").lower())


def _keywords(text: str) -> list[str]:
    """Ordered, de-duplicated content words of ``text`` (length >= 4)."""
    seen: dict[str, None] = {}
    for match in _WORD_RE.finditer(text or ""):
        word = match.group(0).lower()
        if word in STOPWORDS:
            continue
        seen.setdefault(word, None)
    return list(seen)


def _is_refusal(text: str) -> bool:
    head = (text or "").strip()[:400]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- #
# Tier 1 — exact match
# --------------------------------------------------------------------------- #
def grade_exact_match(task: Task, text: str) -> QualityScore:
    """Grade against a short canonical reference.

    Full credit requires an exact (whitespace/case-insensitive) match.  Near
    misses earn edit-similarity credit capped at :data:`EXACT_PARTIAL_CEILING`,
    because in a cost benchmark the interesting signal is "the cheap model got
    it almost right" — a binary bit would throw that away.
    """
    reference = (task.reference or "").strip()
    answer = (text or "").strip()
    if not reference:
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.EXACT_MATCH,
            rationale="task has no reference; exact-match grading is impossible",
        )
    if not answer:
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.EXACT_MATCH,
            rationale="empty answer",
            details={"reference": reference},
        )

    if _compact(answer) == _compact(reference):
        return QualityScore(
            score=1.0,
            passed=True,
            grader=GraderKind.EXACT_MATCH,
            rationale="answer matches the reference exactly after normalisation",
            details={"reference": reference},
        )

    ratio = difflib.SequenceMatcher(None, _compact(answer), _compact(reference)).ratio()
    score = round(_clamp(ratio) * EXACT_PARTIAL_CEILING, 4)
    return QualityScore(
        score=score,
        passed=False,
        grader=GraderKind.EXACT_MATCH,
        rationale=(
            f"answer differs from the reference (similarity {ratio:.2f}); "
            "partial credit is capped so only an exact match passes"
        ),
        details={"reference": reference, "similarity": round(ratio, 4)},
    )


# --------------------------------------------------------------------------- #
# Tier 2 — must_include coverage
# --------------------------------------------------------------------------- #
def grade_contains(task: Task, text: str) -> QualityScore:
    """Grade by the fraction of required substrings present.

    Substring checks are case-insensitive and whitespace-tolerant so that
    ``SELECT`` vs ``select`` and re-wrapped code are not scored as misses.  Each
    missing item is reported, which is what makes a quality delta diagnosable
    ("the small model never emitted the JOIN clause").
    """
    needles = [n for n in task.must_include if n and n.strip()]
    if not needles:
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.CONTAINS,
            rationale="task declares no must_include entries",
        )
    haystack = _compact(text)
    if not haystack:
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.CONTAINS,
            rationale="empty answer",
            details={"missing": needles, "found": []},
        )

    found = [n for n in needles if _compact(n) in haystack]
    missing = [n for n in needles if n not in found]
    coverage = len(found) / len(needles)
    # Small bonus for producing a substantial answer; a model that emits only
    # the required tokens and nothing else should not look equal to one that
    # also explains itself.
    substance = _clamp(len(text) / 400.0) * 0.05
    score = round(_clamp(coverage * 0.95 + substance), 4)
    return QualityScore(
        score=score,
        passed=not missing,
        grader=GraderKind.CONTAINS,
        rationale=(
            f"found {len(found)}/{len(needles)} required substrings"
            + (f"; missing: {', '.join(missing)}" if missing else "")
        ),
        details={"found": found, "missing": missing, "coverage": round(coverage, 4)},
    )


# --------------------------------------------------------------------------- #
# Tier 3 — heuristic free-form grading
# --------------------------------------------------------------------------- #
def _structure_score(text: str) -> tuple[float, dict[str, object]]:
    """Reward the shape of a real engineering answer.

    Signals: fenced code blocks (balanced), ordered steps or bullets, a
    non-trivial paragraph count, and a terminated final sentence.  Each signal
    is worth an equal share; none of them alone can carry the score.
    """
    fences = len(_FENCE_RE.findall(text))
    has_code = fences >= 2
    balanced = fences % 2 == 0
    has_list = bool(re.search(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", text))
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    has_paragraphs = len(paragraphs) >= 2
    terminates = text.rstrip().endswith((".", "```", ":", ")", "]", "}", "!"))

    parts = {
        "code_block": 1.0 if has_code else 0.0,
        "balanced_fences": 1.0 if (not has_code or balanced) else 0.0,
        "list_or_steps": 1.0 if has_list else 0.0,
        "paragraphs": 1.0 if has_paragraphs else 0.0,
        "terminated": 1.0 if terminates else 0.0,
    }
    score = sum(parts.values()) / len(parts)
    return score, {
        "fences": fences,
        "has_code": has_code,
        "has_list": has_list,
        "paragraphs": len(paragraphs),
    }


def _completeness_score(text: str) -> tuple[float, dict[str, object]]:
    """Length adequacy, saturating rather than linear.

    Below :data:`COMPLETENESS_MIN_CHARS` an answer is usually a fragment.  Past
    :data:`COMPLETENESS_SATURATION_CHARS` extra length stops earning credit —
    verbosity is a cost, not a quality signal.  Extremely long answers lose a
    little, because in a token-cost benchmark padding must not look free.
    """
    length = len(text)
    if length <= 0:
        return 0.0, {"chars": 0}
    if length < COMPLETENESS_MIN_CHARS:
        score = length / COMPLETENESS_MIN_CHARS
    elif length <= COMPLETENESS_SATURATION_CHARS:
        score = 1.0
    elif length <= COMPLETENESS_MAX_CHARS:
        score = 1.0 - 0.15 * (length - COMPLETENESS_SATURATION_CHARS) / (
            COMPLETENESS_MAX_CHARS - COMPLETENESS_SATURATION_CHARS
        )
    else:
        score = 0.75
    return _clamp(score), {"chars": length}


def _coverage_score(task: Task, text: str) -> tuple[float, dict[str, object]]:
    """Keyword coverage of the answer against the task's own vocabulary.

    The keywords come from the prompt (what was asked) plus the reference (what
    a good answer mentions).  This is a cheap proxy for "did the answer actually
    address the question", and it is fully deterministic.
    """
    prompt_keywords = _keywords(task.prompt)
    reference_keywords = _keywords(task.reference or "")
    combined = list(dict.fromkeys(prompt_keywords + reference_keywords))
    if not combined:
        return 0.5, {"keywords": 0, "covered": 0, "source": "none"}
    haystack = _compact(text)
    covered = [k for k in combined if k in haystack]
    ratio = len(covered) / len(combined)
    missing = [k for k in combined if k not in covered][:10]
    return ratio, {
        "keywords": len(combined),
        "covered": len(covered),
        "missing_sample": missing,
    }


def _focus_score(task: Task, text: str) -> tuple[float, dict[str, object]]:
    """Penalise refusals, empty filler and answers that ignore the question.

    The refusal check is a real quality signal: a model that declines a benign
    engineering request is wrong, and without this penalty a politely-worded
    refusal would score on structure and length alone.
    """
    if _is_refusal(text):
        return 0.0, {"refusal": True}
    if not text.strip():
        return 0.0, {"refusal": False, "empty": True}
    # Answers that are almost entirely the prompt echoed back are not answers.
    overlap = difflib.SequenceMatcher(None, _compact(text), _compact(task.prompt)).ratio()
    if overlap > 0.85:
        return 0.2, {"refusal": False, "prompt_echo_ratio": round(overlap, 4)}
    return 1.0, {"refusal": False, "prompt_echo_ratio": round(overlap, 4)}


def grade_heuristic(task: Task, text: str) -> QualityScore:
    """Grade a free-form answer with documented, weighted signals.

    See :data:`HEURISTIC_WEIGHTS` for the weighting rationale.  The result is
    never binary, which is exactly what a Pareto cost/quality analysis needs: it
    can distinguish "the cheap model is 80% as good" from "the cheap model is
    useless".
    """
    if not (text or "").strip():
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.HEURISTIC,
            rationale="empty answer",
        )

    coverage, coverage_details = _coverage_score(task, text)
    structure, structure_details = _structure_score(text)
    completeness, completeness_details = _completeness_score(text)
    focus, focus_details = _focus_score(task, text)

    components = {
        "coverage": coverage,
        "structure": structure,
        "completeness": completeness,
        "focus": focus,
    }
    score = sum(components[name] * weight for name, weight in HEURISTIC_WEIGHTS.items())
    score = round(_clamp(score), 4)

    # Difficulty is a documented expectation, not a fudge: a hard task answered
    # briefly is weaker evidence than an easy one answered briefly, so the
    # completeness component is scaled by how much was expected.
    if task.difficulty >= 0.7 and len(text) < 400:
        score = round(_clamp(score * 0.9), 4)
        difficulty_note = "; short answer on a high-difficulty task"
    else:
        difficulty_note = ""

    return QualityScore(
        score=score,
        passed=score >= 0.7,
        grader=GraderKind.HEURISTIC,
        rationale=(
            "weighted heuristic: "
            + ", ".join(
                f"{name}={components[name]:.2f}*{weight:g}"
                for name, weight in HEURISTIC_WEIGHTS.items()
            )
            + f" -> {score:.3f}"
            + difficulty_note
        ),
        details={
            "components": {k: round(v, 4) for k, v in components.items()},
            "coverage": coverage_details,
            "structure": structure_details,
            "completeness": completeness_details,
            "focus": focus_details,
        },
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def resolve_kind(task: Task, kind: str = "auto") -> GraderKind:
    """Pick the grading tier for a task.

    ``auto`` prefers the most objective tier the task supports: a short
    reference means the answer is checkable exactly; ``must_include`` means it is
    checkable by containment; anything else falls back to the heuristic.
    """
    if kind and kind != "auto":
        try:
            return GraderKind(kind)
        except ValueError as exc:
            raise ValueError(
                f"unknown grader kind {kind!r}; "
                f"expected one of {[k.value for k in GraderKind]} or 'auto'"
            ) from exc
    reference = (task.reference or "").strip()
    if reference and len(reference) <= EXACT_MATCH_MAX_REFERENCE_CHARS:
        return GraderKind.EXACT_MATCH
    if any(n and n.strip() for n in task.must_include):
        return GraderKind.CONTAINS
    return GraderKind.HEURISTIC


def grade(task: Task, text: str, *, kind: str = "auto") -> QualityScore:
    """Score ``text`` as an answer to ``task``.

    Args:
        task: the benchmark item (its ``reference`` / ``must_include`` /
            ``prompt`` decide how grading is possible).
        text: the model's final answer for the cell.
        kind: ``"auto"`` (default) or an explicit :class:`GraderKind` value.
            ``"none"`` records that no grading was requested.

    Returns:
        A deterministic :class:`~tokbench.models.QualityScore` in ``[0, 1]``.

    Never raises for a bad *answer* — a broken answer is a score, not an
    exception.  It only raises for a bad ``kind`` argument, which is a
    programming error.
    """
    resolved = resolve_kind(task, kind)
    if resolved is GraderKind.NONE:
        return QualityScore(
            score=0.0,
            passed=False,
            grader=GraderKind.NONE,
            rationale="grading disabled for this cell",
        )
    if resolved is GraderKind.EXACT_MATCH:
        return grade_exact_match(task, text)
    if resolved is GraderKind.CONTAINS:
        return grade_contains(task, text)
    if resolved is GraderKind.HEURISTIC:
        return grade_heuristic(task, text)
    # LLM_JUDGE is part of the shared enum but deliberately not implemented:
    # an online judge would make the benchmark non-reproducible and add cost to
    # the very measurement.  Falling back keeps the run honest and total.
    return grade_heuristic(task, text).model_copy(
        update={"rationale": "llm_judge is not implemented; used heuristic grading instead"}
    )


__all__ = [
    "COMPLETENESS_MIN_CHARS",
    "COMPLETENESS_SATURATION_CHARS",
    "EXACT_MATCH_MAX_REFERENCE_CHARS",
    "EXACT_PARTIAL_CEILING",
    "HEURISTIC_WEIGHTS",
    "STOPWORDS",
    "grade",
    "grade_contains",
    "grade_exact_match",
    "grade_heuristic",
    "resolve_kind",
]
