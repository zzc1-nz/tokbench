# Methodology — how tokbench measures token economics

This document is the "show your work" companion to the README. Every number the
tool emits traces back to one of the definitions below.

---

## 1. The unit of measurement

A benchmark run is a **4-dimensional matrix**:

```
models × strategies × tasks × repeats
```

- **models** — what you're comparing (price/perf tiers)
- **strategies** — the prompt-assembly knob you're ablating
- **tasks** — the fixed workload
- **repeats** — variance control (LLM output is stochastic; 3 repeats lets you
  report a spread instead of a single lucky draw)

Each `(model, strategy, task, repeat)` tuple is a **cell**. A cell may involve
several **attempts** if the first call fails and is retried. Every attempt
produces one `CallRecord` (the atomic JSONL log line); every cell produces one
`TaskResult` (the aggregate).

This separation is the key modelling decision: **retries are not a footnote,
they are a first-class measured cost.** Most token-cost analyses silently drop
failed calls. That's how you get a cost model that's 15% optimistic.

---

## 2. Token accounting

Providers report tokens inconsistently. tokbench normalises everything to one
convention:

| field | meaning |
|---|---|
| `input_tokens` | **total** prompt tokens, *including* cached ones |
| `cache_read_tokens` | subset of `input_tokens` served from prompt cache |
| `cache_write_tokens` | subset of `input_tokens` written into the cache |
| `output_tokens` | generated tokens (reasoning included where the provider does so) |
| `reasoning_tokens` | separately-reported reasoning tokens, when available |

From these we derive three *different* totals, and the distinction matters:

```
total_tokens    = input_tokens + output_tokens
                  → "how much context did I push through the model"
                  → the number to watch for rate limits and context windows

uncached_input  = input_tokens − cache_read − cache_write
                  → "how much did I pay full price for"

billable_tokens = uncached_input + output_tokens
                  → "tokens after the cache discount"
                  → THE ranking denominator
```

**Why `billable_tokens` and not `total_tokens`?** Because a cache hit costs a
fraction of a fresh token (typically 10%). Ranking models on raw token volume
would penalise exactly the behaviour you want to reward. Ranking on billable
tokens makes cache efficiency part of the score, which is the economically
correct thing to measure.

---

## 3. Cost model

Cost is computed **only** in `pricing.py`, from token counts and a price book.
Adapters never touch money — they report counts. This means:

- you can re-price an entire historical run by swapping `prices.yaml`
  (`tokbench report --prices my-negotiated-rates.yaml`), with **zero** API calls;
- the cost arithmetic is auditable in one file;
- mock/offline runs and live runs use identical cost logic.

For one call:

```
cost = (uncached_input  × P_input
      + cache_read      × P_cache_read
      + cache_write     × P_cache_write
      + output          × P_output
      + reasoning       × P_reasoning) / 1e6
```

with `P_*` in USD per million tokens. Defaults when a provider doesn't publish a
separate cache price: `P_cache_read = 0.1 × P_input`,
`P_cache_write = 1.25 × P_input`, `P_reasoning = P_output`.

---

## 4. Token-efficiency leaderboard

For each `(model, strategy)` pair, aggregated over all cells:

```
tokens_per_task  = Σ billable_tokens / n_cells
cost_per_task    = Σ cost_usd       / n_cells
quality          = mean(quality.score)          # in [0, 1]
```

The efficiency score answers *"quality delivered per billable token"*:

```
raw_value        = quality / max(tokens_per_task, 1)
efficiency_score = raw_value / max_over_all_rows(raw_value)
```

Normalising to `[0, 1]` against the best row makes the leaderboard readable
(the best row is 1.000) while preserving the ratio between rows. Ties are broken
by lower `cost_per_task`.

`tokens_per_quality_point = tokens_per_task / max(quality, ε)` — the "price of
quality" in tokens. Useful for spotting a model that's cheap per token but needs
so many tokens to reach the same quality that it loses on total cost.

---

## 5. Pareto frontier (cost vs quality)

The frontier is the answer to *"what's the cheapest way to hit quality Q?"*.

A point **A** dominates **B** iff:

```
A.cost_per_task <= B.cost_per_task  AND  A.quality >= B.quality
and at least one inequality is strict
```

Points not dominated by any other point are **on the frontier**. Sorted by cost
ascending, they form the efficient trade-off curve: every step right costs more,
and buys quality you cannot get cheaper.

**Knee point** — the frontier point furthest (perpendicular distance) from the
straight line joining the cheapest and highest-quality frontier points. This is
the classic "elbow" heuristic: it's the point where extra spend stops buying
proportional quality, i.e. the pragmatic recommendation.

`value_score = quality / max(cost_per_task, ε)` — quality per dollar, normalised.
The best-value point is often the knee, but not always; both are reported.

---

## 6. Retry overhead

This is the analysis most token-cost tools omit, and it's often the largest
avoidable cost line.

For a cell with attempts `a₁ … aₙ`:

```
retry_overhead = Σ_{i=2..n} usage(aᵢ)
```

i.e. everything spent **after** the first attempt — pure waste if the first
attempt had succeeded. Then, per `(model, strategy)`:

```
retry_cost_share = retry_cost_usd / total_cost_usd
retry_token_share = retry_tokens  / total_tokens
retry_rate       = retried_cells  / cells
```

and `wasted_pct_of_budget = total_retry_cost_usd / total_cost_usd` at the run
level, plus an `error_breakdown` histogram of normalised `ErrorKind`s so you can
see *why*: rate limits, timeouts, 5xx, context overflow, auth.

**Reading it:** a 3% retry rate at 2× token cost per retry ≈ 6% of spend
vaporised. If one model/strategy pair owns most of that, the fix is targeted
(concurrency down, longer timeout, smaller context) rather than global.

---

## 7. Cache sensitivity

Cache hit rate is the single biggest cost lever, so we model it explicitly
rather than just reporting the observed value.

Hold each model's **total input tokens** and **output tokens** fixed. Then sweep
a hypothetical cache hit rate `h ∈ [0, 1]`:

```
cache_read(h)   = h × input_tokens
uncached(h)     = (1 − h) × input_tokens
cost(h)         = (uncached(h) × P_input
                 + cache_read(h) × P_cache_read
                 + output × P_output) / 1e6
```

The result is a **straight line** in `h` with slope proportional to
`(P_input − P_cache_read)`. That linearity is the useful insight: each
percentage point of cache hit rate saves a fixed, predictable amount, so
`marginal_usd_per_10pp` is a directly actionable number.

We report `zero_cache_cost` (h=0), `full_cache_cost` (h=1), the observed point,
and `savings_at_observed`.

---

## 8. Strategy ablation

For strategies `X` (baseline) and `Y`:

```
token_delta_pct  = (tokens_per_task(Y) − tokens_per_task(X)) / tokens_per_task(X)
cost_delta_pct   = (cost_per_task(Y)   − cost_per_task(X))   / cost_per_task(X)
quality_delta    = quality(Y) − quality(X)
```

The interesting derived quantity is the **marginal token cost of quality**:

```
tokens_per_quality_delta = (tokens_per_task(Y) − tokens_per_task(X))
                           / (quality_delta × 100)
```

= extra tokens paid per +0.01 quality. This turns "CoT helps" into a number you
can compare against a budget.

Verdict thresholds (documented, deliberately blunt):

| condition | verdict |
|---|---|
| `quality_delta > +0.02` and `cost_delta_pct < +25%` | worth it |
| `quality_delta < +0.005` and `cost_delta_pct > +10%` | wasteful |
| otherwise | neutral |

---

## 9. Reproducibility

- The config is hashed (`config_hash`) and snapshotted into the run directory, so
  a run directory is self-describing.
- Cell expansion order is fixed and documented in `config.py`, so a partial or
  budget-aborted run is a **prefix** of the full matrix, not a random subset.
- Mock mode derives every token count from `sha256(cache_key)` — same input,
  same output, byte for byte. That's what makes the bundled demo reproducible
  and the test suite deterministic.
- The call log is append-only JSONL, fsync'd per line: a crash leaves a valid
  partial dataset, never a corrupt one.

---

## 10. What this methodology does *not* claim

Honesty section, because a benchmark that oversells itself isn't useful:

- **Quality grading is heuristic** (`must_include` substring checks + structural
  heuristics). It is a cheap proxy, not a human eval. Scores are comparable
  *within* a run, not absolute.
- **Mock-mode numbers are synthetic.** They validate the *pipeline and the cost
  model*, not the models. The bundled demo exists so the tool is inspectable
  without an API key.
- **Prices are a snapshot.** They drift, and negotiated rates differ. Always pass
  `--prices` with your real numbers before quoting a figure.
- **Repeat count controls variance, not significance.** 3 repeats gives a spread;
  it is not a statistical test. Use 5+ repeats before claiming a small quality
  delta is real.
