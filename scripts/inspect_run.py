"""Ad-hoc inspection of a run's headline analysis numbers (dev helper)."""

import collections
import sys

from tokbench.analysis.report import build_report
from tokbench.store import RunStore

run = sys.argv[1] if len(sys.argv) > 1 else "reports/latest"
p = build_report(RunStore(run))

print("=== SUMMARY ===")
s = p.summary
print(f"cells {s.completed_cells}/{s.planned_cells} calls {s.total_calls} failed {s.failed_calls}")
print(f"tokens {s.total_usage.total_tokens:,} billable {s.total_billable_tokens:,}")
print(f"cost ${s.total_cost_usd:.6f}  cache hit {s.cache_hit_rate:.2%}")

print("\n=== LEADERBOARD ===")
for r in p.leaderboard:
    print(
        f"{r.rank:2d} {r.model_id:18s} {r.strategy_id:17s} "
        f"tok/task {r.billable_tokens_per_task:7.0f} q {r.mean_quality:.3f} "
        f"$/task {r.cost_per_task_usd:.6f} score {r.efficiency_score:.3f}"
    )

print("\n=== PARETO ===")
for pt in p.pareto.points:
    mark = "ON " if pt.on_frontier else "   "
    print(
        f"{mark}{pt.label:36s} ${pt.cost_per_task_usd:.6f}/task q {pt.quality:.3f} "
        f"value {pt.value_score:.3f}"
    )
print(f"frontier: {p.pareto.frontier_labels}")
print(f"explanation: {p.pareto.explanation}")

print("\n=== RETRY ===")
r = p.retry
print(f"retry cost ${r.total_retry_cost_usd:.6f} share {r.overall_retry_cost_share:.2%}")
print(f"retry tokens {r.total_retry_tokens:,} share {r.retry_token_share:.2%}")
print(f"worst offender: {r.worst_offender}")
print(f"error breakdown: {r.error_breakdown}")
print(f"explanation: {r.explanation}")
for row in r.rows:
    if row.retried_cells:
        print(
            f"  {row.model_id:18s} {row.strategy_id:17s} "
            f"retried {row.retried_cells}/{row.cells} rate {row.retry_rate:.1%} "
            f"cost share {row.retry_cost_share:.2%} dominant {row.dominant_error_kind}"
        )

print("\n=== CACHE ===")
c = p.cache
# NB: fields named ``*_pct`` are already in percent units (0-100); ``*_share``
# and ``cache_hit_rate`` are fractions (0-1). Do not re-scale.
print(f"observed hit {c.observed_hit_rate:.2%} cost ${c.observed_cost_usd:.6f}")
print(f"zero-cache ${c.zero_cache_cost_usd:.6f} full-cache ${c.full_cache_cost_usd:.6f}")
print(
    f"savings ${c.savings_at_observed_usd:.6f} ({c.savings_at_observed_pct:.2f}% of no-cache bill)"
)
print(f"marginal per +10pp: ${c.marginal_usd_per_10pp:.6f}")
print(f"explanation: {c.explanation}")

print("\n=== STRATEGY DELTAS ===")
for d in p.strategy_deltas:
    model = d.model_id or "(all)"
    print(
        f"{model:18s} {d.strategy_id:18s} tok {d.token_delta_pct:+7.1f}% "
        f"cost {d.cost_delta_pct:+7.1f}% q {d.quality_delta:+.3f} "
        f"-> {d.verdict}"
    )

print("\n=== QUALITY DIST ===")
res = RunStore(run).read_results()
qs = [x.quality.score for x in res if x.quality]
print(f"n={len(qs)} mean={sum(qs) / len(qs):.3f} min={min(qs):.3f} max={max(qs):.3f}")
print(f"passed {sum(1 for x in res if x.quality and x.quality.passed)}/{len(res)}")
print(f"graders {dict(collections.Counter(x.quality.grader.value for x in res if x.quality))}")
