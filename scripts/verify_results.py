"""Verify every quantitative claim made in docs/RESULTS.md.

A published number that cannot be reproduced is worse than no number. This script
recomputes each figure from the committed artefacts and prints PASS/FAIL, so the
document stays honest as the code changes.

    python scripts/verify_results.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "example-report.json"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


payload = json.loads(REPORT.read_text(encoding="utf-8"))
summary = payload["summary"]
pareto = payload["pareto"]
retry = payload["retry"]
cache = payload["cache"]

# --- run scale -------------------------------------------------------------
check("2,250 cells", summary["completed_cells"] == 2250, str(summary["completed_cells"]))
check("2,332 calls", summary["total_calls"] == 2332, str(summary["total_calls"]))
check("82 failed calls", summary["failed_calls"] == 82, str(summary["failed_calls"]))
check(
    "1.99M total tokens",
    1_980_000
    < summary["total_usage"]["input_tokens"] + summary["total_usage"]["output_tokens"]
    < 2_000_000,
    f"{summary['total_usage']['input_tokens'] + summary['total_usage']['output_tokens']:,}",
)
check(
    "1.51M billable tokens",
    1_510_000 < summary["total_billable_tokens"] < 1_520_000,
    f"{summary['total_billable_tokens']:,}",
)
check(
    "$1.96 total cost",
    abs(summary["total_cost_usd"] - 1.9557) < 0.01,
    f"{summary['total_cost_usd']:.4f}",
)
check(
    "$0.00129 / 1k billable",
    abs(summary["cost_per_1k_billable_tokens"] - 0.00129) < 0.00005,
    f"{summary['cost_per_1k_billable_tokens']:.6f}",
)
check(
    "79.5s wall clock", abs(summary["wall_clock_s"] - 79.5) < 1.0, f"{summary['wall_clock_s']:.1f}s"
)
check(
    "48.9% cache hit",
    abs(summary["cache_hit_rate"] - 0.4886) < 0.001,
    f"{summary['cache_hit_rate']:.4f}",
)

# --- pareto ----------------------------------------------------------------
n_points = len(pareto["points"])
n_frontier = len(pareto["frontier_labels"])
check("15 candidates", n_points == 15, str(n_points))
check("8 non-dominated", n_frontier == 8, str(n_frontier))
check(
    "knee label published",
    pareto.get("knee_label") == "deepseek-chat / baseline",
    str(pareto.get("knee_label")),
)

# marginal return collapse: recompute from the frontier itself
frontier = sorted(
    (p for p in pareto["points"] if p["on_frontier"]), key=lambda p: p["cost_per_task_usd"]
)
knee_i = next(i for i, p in enumerate(frontier) if p["label"] == pareto["knee_label"])
below = (frontier[knee_i]["quality"] - frontier[0]["quality"]) / (
    frontier[knee_i]["cost_per_task_usd"] - frontier[0]["cost_per_task_usd"]
)
above = (frontier[-1]["quality"] - frontier[knee_i]["quality"]) / (
    frontier[-1]["cost_per_task_usd"] - frontier[knee_i]["cost_per_task_usd"]
)
ratio = below / above
check(
    "9.1x marginal collapse",
    abs(ratio - 9.1) < 0.2,
    f"{ratio:.2f}x (below {below:.0f} / above {above:.0f} q per USD)",
)

knee_point = frontier[knee_i]
check(
    "knee cost $0.000676/task",
    abs(knee_point["cost_per_task_usd"] - 0.000676) < 0.000002,
    f"{knee_point['cost_per_task_usd']:.6f}",
)
check(
    "knee quality 0.763",
    abs(knee_point["quality"] - 0.763) < 0.002,
    f"{knee_point['quality']:.3f}",
)

# --- retry -----------------------------------------------------------------
check(
    "3.75% retry waste",
    abs(retry["overall_retry_cost_share"] - 0.0375) < 0.0005,
    f"{retry['overall_retry_cost_share']:.4f}",
)
check(
    "$0.0734 retry cost",
    abs(retry["total_retry_cost_usd"] - 0.0734) < 0.001,
    f"{retry['total_retry_cost_usd']:.4f}",
)
check("53,381 retry tokens", retry["total_retry_tokens"] == 53381, str(retry["total_retry_tokens"]))
worst = max(retry["rows"], key=lambda r: r["retry_cost_share"])
check(
    "worst pair 7.11%",
    abs(worst["retry_cost_share"] - 0.0711) < 0.001,
    f"{worst['model_id']}/{worst['strategy_id']} = {worst['retry_cost_share']:.4f}",
)
best_pair = min(
    (r for r in retry["rows"] if r["retried_cells"] > 0), key=lambda r: r["retry_cost_share"]
)
check(
    "cheapest retry pair 1.68%",
    abs(best_pair["retry_cost_share"] - 0.0168) < 0.001,
    f"{best_pair['model_id']}/{best_pair['strategy_id']} = {best_pair['retry_cost_share']:.4f}",
)
check(
    "error breakdown 46/19/17",
    retry["error_breakdown"].get("rate_limit") == 46
    and retry["error_breakdown"].get("timeout") == 19
    and retry["error_breakdown"].get("server_error") == 17,
    str(retry["error_breakdown"]),
)

# --- cache -----------------------------------------------------------------
check(
    "$0.0148 per +10pp",
    abs(abs(cache["marginal_usd_per_10pp"]) - 0.0148) < 0.0005,
    f"{abs(cache['marginal_usd_per_10pp']):.5f}",
)

# --- code / test scale -----------------------------------------------------
pkg = [p for p in (ROOT / "tokbench").rglob("*.py") if "__pycache__" not in str(p)]
pkg_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in pkg)
check("30 package files", len(pkg) == 30, str(len(pkg)))
check("~13.1k package lines", 12_900 < pkg_lines < 13_300, f"{pkg_lines:,}")

tests = list((ROOT / "tests").glob("*.py"))
test_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in tests)
check("4 test files", len(tests) == 4, str(len(tests)))
check("~4.1k test lines", 4_000 < test_lines < 4_300, f"{test_lines:,}")

collected = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "-p", "no:cacheprovider"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
# `-q --collect-only` prints a per-file summary like "tests/test_analysis.py: 54".
# Sum those counts rather than counting test ids (the ids are not printed in -q).
n_tests = sum(int(m) for m in re.findall(r"^tests/[\w.]+\.py:\s*(\d+)$", collected.stdout, re.M))
check("199 tests", n_tests == 199, str(n_tests))

# --- report and figures exist ---------------------------------------------
for rel in (
    "docs/example-report.html",
    "docs/pareto-frontier.png",
    "docs/retry-cost-share.png",
    "docs/cache-sensitivity.png",
):
    check(f"{rel} exists", (ROOT / rel).exists(), "")

# ---------------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
print(f"\n{len(checks) - len(failed)}/{len(checks)} claims verified")
if failed:
    print(f"\n{len(failed)} claim(s) DO NOT match the data — fix the document:", file=sys.stderr)
    for name, _, detail in failed:
        print(f"  - {name} (actual: {detail})", file=sys.stderr)
    raise SystemExit(1)
print("all published claims are reproducible from the committed artefacts")
