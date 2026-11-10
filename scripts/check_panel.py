"""Smoke-check the Streamlit panel's data layer against a real run directory.

Launches no server: it exercises exactly the functions ``streamlit run`` would
call, so a broken payload shape or a raising figure builder is caught in CI
rather than by a user staring at a blank tab.

Usage::

    python scripts/check_panel.py reports/<run-id>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/check_panel.py <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])

    from tokbench.panel import (  # imported late so --help stays fast
        build_cache_figure,
        build_pareto_figure,
        build_retry_figure,
        build_series_figure,
        build_stacked_figure,
        leaderboard_frame,
        load_payload,
    )

    payload = load_payload(run_dir)
    print(f"run           {payload.meta.run_id}")
    print(f"experiment    {payload.meta.experiment}")
    print(
        f"shape         {len(payload.leaderboard)} pairs · "
        f"{len(payload.pareto.points)} pareto points · "
        f"{len(payload.series)} series points · "
        f"{len(payload.stacked)} stacked bars"
    )

    failures = 0
    builders = [
        ("pareto", build_pareto_figure),
        ("series", build_series_figure),
        ("stacked", build_stacked_figure),
        ("retry", build_retry_figure),
        ("cache", build_cache_figure),
    ]
    for name, builder in builders:
        try:
            fig = builder(payload)
        except Exception as exc:  # noqa: BLE001 - a smoke check reports, not raises
            failures += 1
            print(f"  FAIL {name:8s} {type(exc).__name__}: {exc}")
            continue
        n_traces = len(getattr(fig, "data", []) or [])
        print(f"  ok   {name:8s} {type(fig).__name__} ({n_traces} traces)")

    try:
        frame = leaderboard_frame(payload)
        print(f"  ok   frame    leaderboard {len(frame)} rows x {len(frame.columns)} cols")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  FAIL frame    {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{failures} panel check(s) failed", file=sys.stderr)
        return 1
    print("\npanel data layer is healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
