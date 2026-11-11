# Results & Reproducibility

This document records the measured outcome of the bundled flagship experiment and
states exactly how each number can be re-derived. It is the reference page for
anyone quoting a figure from this repository.

Every claim below is asserted mechanically by
[`scripts/verify_results.py`](../scripts/verify_results.py) (31 checks),
which reads the committed artefacts rather than trusting this prose:

```bash
python scripts/verify_results.py
# -> 31/31 claims verified
```

If you change the code or re-run the experiment, run that script before quoting
any number from this page.

---

## 1. The experiment

| | |
|---|---|
| Config | [`examples/experiment.yaml`](../examples/experiment.yaml) |
| Task set | [`tasks/coding_bench.jsonl`](../tasks/coding_bench.jsonl) 鈥?50 tasks, 8 categories |
| Matrix | 5 models 脳 3 prompt strategies 脳 50 tasks 脳 3 repeats |
| Cells | **2,250** |
| Mode | `mock` (deterministic, offline, no credentials) |
| Concurrency | 8 |

The three strategies form a single-variable ablation. Each differs from
`baseline` in exactly one named field, so a token delta between two rows of the
report is attributable to that field rather than to "the prompt":

| Strategy | Differs from baseline by |
|---|---|
| `baseline` | 鈥?(control: short system prompt, no examples, no CoT) |
| `cot` | `cot: true` 鈥?a reasoning instruction |
| `fewshot-compress` | `few_shot: 3` + `compress_tool_output: true` |

## 2. Measured outcome

| Metric | Value |
|---|---|
| Cells completed | 2,250 / 2,250 |
| API attempts | 2,332 |
| Failed attempts | 82 |
| Retried cells | 80 |
| Total tokens | 1,990,790 (1.99M) |
| Billable tokens (after cache discount) | 1,514,472 (1.51M) |
| Total cost | $1.9557 |
| Cost per 1k billable tokens | $0.001291 |
| Wall clock | 79.5 s |
| Cache hit rate | 48.9% |
| Mean graded quality | 0.618 |

## 3. Cost/quality Pareto frontier

15 (model 脳 strategy) candidates, of which **8 are non-dominated**.

| Role | Candidate | Cost/task | Quality |
|---|---|---|---|
| Cheapest | `mock-small / baseline` | $0.000080 | 0.462 |
| Best value | `mock-small / cot` | $0.000081 | 0.479 |
| **Knee** | `deepseek-chat / baseline` | $0.000676 | 0.763 |
| Best quality | `deepseek-reasoner / fewshot-compress` | $0.003285 | 0.907 |

**The marginal-return collapse.** Along the frontier, quality is bought at
~504 quality-points per USD below the knee and ~55 above it 鈥?a **9.1脳 drop**.
Equivalently, the marginal price of a quality point rises from $0.0020 to $0.0181.

This is the practical output of the whole exercise: past the knee, additional
budget buys proportionally less quality, so a model-selection decision above that
point needs a quality justification that unit price cannot provide.

## 4. Retry overhead

Retries are billed. `TaskResult.usage_total` sums every attempt and
`retry_overhead_usage` isolates the spend after attempt 1.

| Metric | Value |
|---|---|
| Retry cost | $0.0734 |
| Share of total spend | **3.75%** |
| Retry tokens | 53,381 (3.52% of billable tokens) |
| Retried cells | 80 / 2,250 |

The waste is concentrated, not uniform:

| Pair | Retry share of its own spend | Retried cells | Dominant error |
|---|---|---|---|
| `gpt-4o-mini / fewshot-compress` | **7.11%** | 11 / 150 | `rate_limit` |
| `deepseek-reasoner / baseline` | 5.74% | 6 / 150 | `rate_limit` |
| `mock-small / baseline` | 1.68% | 2 / 150 | `rate_limit` |

Error-kind distribution across the run: `rate_limit` 46, `timeout` 19,
`server_error` 17. A pair whose retry share is twice the average and whose
dominant error is `rate_limit` is a client-side concurrency problem 鈥?the fix is
to cap that pair's in-flight requests, not to change the prompt.

## 5. Cache sensitivity

Holding input and output token counts fixed, the projected bill is affine in
cache hit rate:

```
cost(h) = ((1鈭抙)路input路P_in + h路input路P_cache_read + output路P_out) / 1e6
```

| Metric | Value |
|---|---|
| Observed hit rate | 48.9% |
| Cost at observed rate | $1.9557 |
| Cost at 0% hit rate | $2.0291 |
| Cost at 100% hit rate | $1.8809 |
| Savings at observed rate | $0.0734 (3.62% of the no-cache bill) |
| Marginal value | **$0.0148 per +10pp** |

The linearity is the useful part: a percentage point of hit rate has a fixed,
predictable dollar value, which makes caching a budget line item rather than an
optimisation hoped for after the fact.

**Definition note.** The hit-rate denominator is `input_tokens 鈭?cache_write_tokens`,
not `input_tokens`. A cache write is a miss that populates the cache, so counting
writes as misses would penalise a workload that aggressively warms its cache. One
definition (`Usage.cache_hit_rate`) is used by the summary KPI, the time-series
running ratio, and this curve.

## 6. Strategy ablation

Run-wide, versus `baseline`:

| Strategy | Token delta | Cost delta | Quality delta | Verdict |
|---|---|---|---|---|
| `cot` | +3.3% | +1.4% | +0.007 | neutral |
| `fewshot-compress` | +25.7% | +9.6% | +0.007 | neutral |

The run-wide means hide the interesting result: the same strategy is worth very
different amounts depending on the model.

| Pair | Token delta | Quality delta | Verdict |
|---|---|---|---|
| `deepseek-reasoner` + few-shot | +19.5% | **+0.033** | worth it |
| `mock-small` + few-shot | +38.0% | +0.026 | worth it |
| `gpt-4o-mini` + few-shot | +30.6% | **鈭?.008** | wasteful |
| `gpt-4.1-nano` + few-shot | +29.4% | +0.007 | wasteful |

A blanket "add few-shot examples" policy would spend roughly a quarter more
tokens on three of five models without buying quality. Strategy effects are
model-dependent, which is the argument for measuring rather than assuming.

## 7. Task-category difficulty

Mean graded quality per category, averaged over all 15 model/strategy pairs:

| Category | Mean quality | Category | Mean quality |
|---|---:|---|---:|
| reasoning | 0.475 | test-gen | 0.594 |
| sql | 0.487 | tool-use | 0.632 |
| debug | 0.510 | refactor | 0.701 |
| explain | 0.519 | codegen | 0.833 |

Category difficulty varies by nearly 2脳, which is directly actionable for routing:
on codegen-shaped work a cheap tier is already near the ceiling
(`deepseek-reasoner / baseline` scores 1.000 there), while reasoning and SQL are
where a premium tier earns its price.

## 8. Reproducibility

**Byte-identical aggregates.** Two independent runs of the flagship config
produce identical totals:

```
input 907,007 路 output 1,083,783 路 cache_read 411,543
cost $1.9557020575 路 calls 2,332 路 failed 82 路 retried 80
```

This holds because the mock adapter derives every token count from
`sha256(cache_key)` plus the attempt index 鈥?no wall-clock or RNG state enters
the measurement.

**Reproduce it:**

```bash
pip install -e ".[panel]"
python scripts/validate_tasks.py                 # 50 tasks, 8 categories
tokbench run -c examples/experiment.yaml         # ~80 s, offline, $0
tokbench report --run latest                     # writes report.json + report.html
python scripts/verify_results.py           # 31/31 claims verified
```

The committed [`example-report.json`](example-report.json) and
[`example-report.html`](example-report.html) are the output of that exact run.

## 9. Scale and coverage

| | |
|---|---|
| Package modules | 30 files, 13,078 lines |
| Tests | 199 (analysis 54 / engine 74 / render 30 / taskset 41) |
| Test code | 4,119 lines |
| Lint | `ruff check` and `ruff format --check` clean |
| CI | 3.10 / 3.12 / 3.13 on Ubuntu + Windows, plus an offline end-to-end job |

## 10. Limitations

Read these before quoting a number from this page.

1. **These are mock-mode results.** The mock adapter is a behavioural simulator:
   it models token accounting, latency, failure injection, and tier-dependent
   answer quality. The figures therefore validate the *pipeline and the cost
   model*, not the underlying models. Conclusions about which real model is more
   economical require `adapter_mode: live` and provider credentials.
2. **Quality grading is heuristic.** `must_include` substring checks plus
   structural rules. It is a cheap proxy for human evaluation; scores are
   comparable *within* one run, not absolute.
3. **3 repeats control variance, not significance.** Detecting a small quality
   delta reliably needs 5+ repeats.
4. **Prices are a 2026 list-price snapshot.** Negotiated rates differ; pass
   `--prices` with real numbers before quoting a dollar figure.
5. **Cache projections assume fixed token counts.** Section 5 redistributes input
   tokens between full-price and cache-read; it does not model how a real
   caching strategy would change prompt construction.

The methodology behind each metric is derived in
[`METHODOLOGY.md`](METHODOLOGY.md).
