# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev,panel]"
```

## Before you push

```bash
python scripts/validate_tasks.py    # task set constraints
python -m pytest -q                 # full suite (199 tests)
ruff check . && ruff format --check .
```

## The five invariants this codebase is built around

If you change something and one of these breaks, the change is wrong — these are
the properties the whole design exists to protect.

1. **Adapters count tokens; they never price them.**
   Cost lives only in `pricing.py`. This is what lets a historical run be
   re-priced by swapping a price book, with zero API calls.

2. **One number, one owner.**
   Every derived quantity is computed once and *published* on the report payload.
   The Pareto knee is `ParetoFrontier.knee_label`; the cache hit rate is
   `Usage.cache_hit_rate`; cost comes from `PriceBook.cost`. Renderers read these
   fields, they never re-derive them. Re-deriving is how the HTML report, the
   panel and the README figures start disagreeing about the same run.

3. **A failed call is billed.**
   Retries are a measured cost, not a footnote. `TaskResult.usage_total` sums
   every attempt; `retry_overhead_usage` isolates the waste.

4. **Run directories are append-only.**
   `calls.jsonl` is written with `fsync` per line. A crash or a budget abort must
   leave a valid partial dataset, never a corrupt one.

5. **The pipeline must work offline.**
   `adapter_mode: mock` is deterministic (byte-identical aggregates across runs),
   needs no credentials, and is what CI exercises. If a change makes the suite
   require network access, it will not be merged.

## Adding a model provider

1. Subclass `BaseAdapter` in `tokbench/adapters/`, implement `complete()`, and
   normalise **every** failure path to an `ErrorKind` — the retry policy and the
   retry-overhead analysis both key off it.
2. Register it in `ADAPTERS` in `adapters/registry.py`.
3. Never invent usage: if the provider omits token counts, estimate them and set
   `AdapterResponse.usage_estimated = True`.

## Adding benchmark tasks

See [`tasks/README.md`](tasks/README.md). `must_include` entries must be genuine
substrings of a correct answer, or grading silently stops discriminating.

## Adding a metric

Add the field to `report_schema.py` first, compute it in `analysis/`, and have
renderers read it. Do not compute a metric inside a renderer — see invariant 2.
