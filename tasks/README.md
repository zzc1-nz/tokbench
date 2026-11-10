# `coding_bench` — the tokbench task set

Fifty realistic engineering requests used as the fixed workload in every tokbench
experiment. The task set is deliberately **frozen and versioned**: a benchmark
whose inputs drift between runs cannot compare anything, so changes here must be
deliberate, reviewed, and accompanied by a re-run of every experiment whose
numbers you intend to quote.

```
tasks/coding_bench.jsonl     50 lines, one Task object per line (JSONL)
scripts/validate_tasks.py    the gate — exits non-zero on any contract violation
scripts/build_taskset.py     regenerates the JSONL (tool payloads are assembled, not hand-escaped)
```

## Category distribution

The split is the experimental design, not an accident: it decides how much weight
each skill area carries in an aggregate score. `validate_tasks.py` asserts these
counts exactly.

| Category   | Count | Share | What it measures | Example |
|------------|------:|------:|------------------|---------|
| `codegen`  |    10 |  20% | Writing new production code from a spec: retries, worker pools, middleware, streaming I/O | Go worker pool with graceful shutdown |
| `debug`    |     8 |  16% | Diagnosing a described failure from evidence: leaks, races, deadlocks, stale closures | asyncpg pool exhaustion after 20 minutes |
| `refactor` |     6 |  12% | Restructuring working code without changing behaviour: strategy extraction, generics, error propagation | Collapsing 23 `unwrap()` calls into `?` |
| `test-gen` |     6 |  12% | Writing a test suite with the right edge cases and a runnable harness | pytest for `parse_range` with parametrize |
| `sql`      |     5 |  10% | Query design and migration: window functions, recursive CTEs, expression indexes | Rolling weekly retention over 8M events |
| `explain`  |     5 |  10% | Reasoning about trade-offs and explaining them: consistency models, latency signatures, semver | Active-active vs single-primary ledger |
| `tool-use` |     6 |  12% | Acting on a large tool result: CI logs, build output, pod status, API captures, git history, query plans | Root-causing a red CI run from its payload |
| `reasoning`|     4 |   8% | Ambiguous requests where the first correct move is to state assumptions | "checkout feels slow, make it faster" |

**Totals: 50 tasks.** The six `tool-use` tasks carry the large `tool_payload` blobs
that the `compress_tool_output` strategy ablates; the other 44 carry none.

### Languages

Mixed on purpose, because token-per-token cost and quality differ sharply by
language and a single-language benchmark would hide that. `validate_tasks.py`
asserts every one of these appears at least once.

| Language | Tasks | Notes |
|----------|------:|-------|
| Python | 11 | Retries, streaming uploads, asyncio pools, memory leaks |
| SQL | 8 | Postgres window functions, CTEs, jsonb, dedupe migrations |
| TypeScript | 7 | Debounce, typed emitters, React hooks, Express handlers |
| `architecture` | 7 | Design and trade-off tasks (explain/reasoning); no single language |
| Rust | 6 | Borrow checker, error propagation, tokio shutdown, serde config |
| Go | 5 | Worker pools, middleware, generics, race conditions |
| Shell | 3 | Log rotation, cron PATH, bats tests |
| `kubernetes` | 1 | CrashLoopBackOff incident triage (`cb-043`) |
| `markdown` | 1 | Release notes from git history (`cb-045`) |
| `math` | 1 | Cache-economics arithmetic (`cb-049`) |

`language` is a `metadata` convention rather than a schema field: six concrete
programming languages plus four task-shaped labels (`architecture`,
`kubernetes`, `markdown`, `math`) where the deliverable is prose, a diagnosis or
a number rather than code in one language.

## Field reference

Every line is one JSON object validated against
[`tokbench.models.Task`](../tokbench/models.py). The model uses
`extra="forbid"`, so an unknown key is a hard error rather than silently ignored
data — a typo'd field name cannot quietly drop information.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `id` | `str` | yes | Stable identifier, `cb-001`..`cb-050`. Reports, resume logic and the call log all key off this, so it must never change for an existing task. |
| `category` | `str` | no (default `"general"`) | One of the eight categories above. Drives few-shot example selection and per-category breakdowns. |
| `prompt` | `str` | yes | The request as a real ticket would phrase it. **30–120 words** (asserted): shorter reads as a toy question, longer starts rewarding verbosity. |
| `reference` | `str \| null` | no | A short canonical answer, used for exact-match grading. **Only valid at ≤64 characters** — the grader switches to containment above that, so a long reference would be silently ignored. Exactly three tasks carry one. |
| `must_include` | `list[str]` | no (default `[]`) | Substrings any correct answer must contain. This is the primary grading signal. See below. |
| `tool_payload` | `str \| null` | no | A realistic JSON blob of tool output. **Required on all six `tool-use` tasks**, and rejected on any other category. Target **800–1500 estimated tokens** (~3,500–6,000 chars). |
| `max_output_tokens` | `int \| null` | no | Per-task output cap. Overrides the model/strategy default in prompt assembly. Short-answer tasks (exact match) get very tight caps — `cb-049` is 32 tokens — so a verbose model is penalised in cost as well as quality. |
| `difficulty` | `float` | no (default `0.5`) | `0.0`–`1.0`. Currently spread over 0.20–0.90. Used by the heuristic grader (a short answer on a hard task is weaker evidence) and to segment quality results. |
| `metadata` | `dict` | no (default `{}`) | Free-form. Conventions used here: `language` (asserted by the validator), `tags` (topic labels), `expected_output_tokens`, and `ambiguity` on the deliberately underspecified tasks. |

## How grading uses `reference` and `must_include`

Grading is deterministic, offline and inspectable — no LLM judge, because an
online judge would add both cost and non-reproducibility to the very thing being
measured. `grader.resolve_kind()` picks the tier from the task's own fields, so a
task needs no explicit grading metadata.

The selection order is **exact match → containment → heuristic**:

1. **`exact_match`** — chosen when `reference` is non-empty and ≤64 chars. The
   answer is compared after collapsing whitespace, lowercasing, and stripping all
   spaces (`return x + 1` matches `return x+1`). A full match scores `1.0`; a
   near miss earns edit-similarity credit capped at `0.75`, so only a true match
   passes. Used by `cb-016` (a shebang line), `cb-034` (a `CREATE INDEX`), and
   `cb-049` (a decimal answer).

2. **`contains`** — chosen when `must_include` is non-empty and no short reference
   exists. This covers **47 of the 50 tasks**, and is the workhorse tier. The
   score is the fraction of needles found, weighted 0.95, plus up to 0.05 for
   producing a substantial answer. Matching is case-insensitive and
   whitespace-tolerant, which is why needles like `"def "` and `"func "` work as
   intended: they assert the token, not the exact spacing. Each missing needle is
   recorded in the score's `details`, so "the small model never emitted a JOIN
   clause" is a diagnosable finding rather than a number.

   **A needle must be genuinely checkable.** Entries are rejected by the validator
   if they compact to fewer than 2 characters. The test suite goes further and
   proves the loop end-to-end: it builds an answer out of each task's own needles
   and asserts the grader scores it ≥0.9 and passes. If the grader could not find
   its own needles, every containment-graded task would score 0 for correct
   answers — that test would fail first.

3. **`heuristic`** — the fallback for free-form answers with neither field. A
   weighted combination of keyword coverage against the prompt/reference (0.45),
   structural signals like fenced code and lists (0.25), length adequacy (0.20,
   saturating so padding stops earning credit) and focus (0.10, which penalises
   refusals and prompt echo).

   **No task in this set currently resolves to this tier** — every task carries
   either a short reference or checkable needles, which is the stronger signal.
   The tier still matters: it is what a newly added task falls back to, and the
   validator's rule ("needles or a short reference, never neither") is what keeps
   it from silently becoming the *default* for tasks nobody bothered to specify.

Current split: **47 `contains` / 3 `exact_match` / 0 `heuristic`**.

Two consequences worth knowing when reading a report:

- A `contains` score is a **floor on correctness, not a proof of it**. Emitting the
  needles in the right shape is necessary but not sufficient, which is why the
  needles are chosen to be structurally meaningful (`"sync.RWMutex"`, `"UNION ALL"`,
  `"retry-after"`) rather than generic words.
- Because the tier is picked from the data, adding a ≤64-char `reference` to a task
  **changes its grader** and therefore its scores. Re-run before comparing.

## Token budget math

The flagship experiment is sized around a **2,000 prompt-token-per-call** planning
constant:

```
cells           = models × strategies × tasks × repeats
                = 5 × 3 × 50 × 3
                = 2,250 cells

prompt tokens   = 2,250 cells × 2,000 tokens/call
                = 4,500,000 tokens   (4.5M — the number the budget covers)
```

> **Note on the arithmetic.** The design brief calls this "4,500 cells". The
> product above is 2,250; the 4.5M token figure is only consistent with 2,250
> cells at 2,000 tokens/call. `tests/test_taskset.py` asserts the true arithmetic
> and documents the discrepancy so it stays visible rather than being quietly
> baked into a constant.

Where the 2,000-token assumption comes from, and why the real number is lower:

| Component | Tokens (approx.) |
|-----------|-----------------|
| Task prompt | 90–125 (avg ~97) |
| `short` system preset | ~15 |
| Chat-message overhead | ~12 |
| **Subtotal, typical task** | **~125** |
| `tool_payload` on the six tool-use tasks | 1,146–1,499 |
| 3 few-shot exemplars (`fewshot-compress` arm) | ~400 |

Measured against the real task file, a `baseline` call averages **~282 prompt
tokens** (max 1,637, on a tool-use task), `cot` ~323, and `fewshot-compress`
~604. The 2,000-token constant is therefore a conservative planning ceiling, not
an estimate of the mean: it leaves headroom for longer system presets, more
few-shot examples, and provider-side prompt templates, so a budget derived from it
will not abort a run midway. The test suite asserts that **no** assembled prompt
in the flagship matrix exceeds it.

Dataset-level totals:

```
prompt chars (avg)          385
prompt tokens (avg)         96.6     (chars/4 heuristic)
dataset chars              51,105
dataset est. tokens        12,776
tool payloads (6)          avg 1,328 tokens, 5,309 chars
```

Cost follows from the price book, never from the task set: `cost = f(tokens,
price_book)`, so re-pricing a stored run is a file edit and costs nothing.

## How to add a task

1. **Pick the category and honour the distribution.** Adding a 51st task changes
   the design. Either update `EXPECTED_CATEGORIES` in *both*
   `scripts/validate_tasks.py` and `tests/test_taskset.py`, or replace an existing
   task of the same category. Never silently unbalance the split.

2. **Write the prompt like a ticket, not a textbook question.** 30–120 words. Name
   the system, the version, the constraint and the observed symptom. Say what the
   caller already tried. Include the awkward requirement that makes it hard
   (`"must not retry KeyboardInterrupt"`, `"must be safe to run twice
   concurrently"`). Do not include the answer, and do not pad with background the
   engineer would already know.

3. **Choose checkable `must_include` needles.** Ask: *if a model produced a correct
   answer, would this substring certainly appear?* Good: API names
   (`"sync.RWMutex"`), SQL keywords used structurally (`"WITH RECURSIVE"`),
   language constructs (`"def "`), error identifiers (`"UniqueViolation"`). Bad:
   words that appear in the prompt, generic English, or anything that a wrong
   answer might also contain. Prefer 3–5 needles; the score is a fraction, so too
   many needles makes a correct answer look mediocre.

4. **Only add a `reference` if the whole answer is genuinely short** (≤64 chars).
   A command, a magic string, a single number. Otherwise leave it `null` and rely
   on `must_include`. The three tasks that carry one are `cb-016`
   (`#!/usr/bin/env bash`), `cb-034` (a one-line `CREATE INDEX`) and `cb-049`
   (`0.46`).

5. **For a `tool-use` task, build the payload from a realistic structure.** Use
   `scripts/build_taskset.py`, where payloads are assembled from Python dicts and
   pretty-printed — hand-escaping a 5KB JSON blob into a JSONL line is how
   corrupted data gets committed. The payload must:
   - be **valid JSON** (compression summarises it structurally; a malformed blob
     silently takes the fallback path),
   - land in the **800–1500 token band** (asserted),
   - contain the evidence needed to answer the prompt, plus realistic noise. If
     the answer is inferable from one line, the task measures nothing.

6. **Set `difficulty` honestly** and spread it. The heuristic grader uses it, and a
   flat difficulty column makes a quality spread unexplainable.

7. **Regenerate and validate:**

   ```bash
   python scripts/build_taskset.py     # rewrites tasks/coding_bench.jsonl
   python scripts/validate_tasks.py    # must exit 0
   python -m pytest tests/test_taskset.py -q
   ```

8. **Version the change.** If you are replacing rather than appending, note it in
   the commit message; previously published numbers were produced by the old set
   and are not comparable to new ones.

## Deliberately hard / ambiguous tasks

Three tasks are underspecified on purpose (`metadata.ambiguity` is set) so quality
scores spread instead of clustering near 1.0. A good answer states its assumptions
and asks the missing question before proposing a solution; a model that dives
straight into code should be visibly penalised.

| Task | Why it is ambiguous |
|------|---------------------|
| `cb-047` | "Checkout feels slow, make it faster" — no target, no traffic data, no budget. The correct first move is measurement and questions, not code. |
| `cb-048` | Shard 14GB across 8 Postgres instances — the proposal may be right or premature; the answer must argue with numbers and make a defensible call. |
| `cb-050` | Global rate limit across 3 regions with no single point of failure — the spec is genuinely contradictory at the edges, so the answer must name which failure mode it accepts. |

Plus one high-difficulty task that is well-specified but hard (`cb-032`, the
2M-row dedupe migration, difficulty 0.70) and one near-trivial exact-match task
(`cb-016`, the shebang, difficulty 0.20) so the difficulty axis has real range.

## Reproducing the published numbers

```bash
python scripts/validate_tasks.py                       # 1. is the input sound?
tokbench validate -c examples/experiment.yaml          # 2. config + taskset + prices check
tokbench run      -c examples/experiment.yaml          # 3. run it (mock = free, offline)
tokbench show     reports/<run-id>                     # 4. terminal summary
tokbench report   reports/<run-id>                     # 5. full JSON/HTML report
```

The run directory contains `calls.jsonl` (one record per HTTP attempt),
`results.jsonl` (one per cell), `manifest.json` (counters, config hash, totals) and
a config snapshot — enough to reinterpret the run without this file. Because the
price book is applied at analysis time, you can re-derive every dollar figure with
`tokbench report <run-dir> --prices <your-rates>.yaml` without re-running a single
call.
