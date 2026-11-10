"""Headless render check for the Streamlit panel.

Runs the panel's script body through Streamlit's own ``AppTest`` harness against a
real run directory, so a rendering exception is caught here instead of by a user
looking at a blank page.

Why this exists separately from ``tests/test_render.py``: the test suite checks
the *data layer* (figure builders, payload loading) without a Streamlit runtime.
This script boots the real script runner, which is where things like the
``streamlit run`` relative-import problem surface.

Usage::

    python scripts/check_panel_render.py reports/<run-id>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/check_panel_render.py <run-dir>", file=sys.stderr)
        return 2

    run_dir = Path(argv[1]).resolve()
    os.environ.setdefault("TOKBENCH_RUNS_DIR", str(run_dir.parent))

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "tokbench" / "panel" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=180)
    at.run()

    if at.exception:
        print(f"FAIL: {len(at.exception)} exception(s) while rendering", file=sys.stderr)
        for exc in at.exception:
            print(f"  {exc.value}", file=sys.stderr)
        return 1

    print("panel rendered without exceptions")
    print(f"  tabs       {len(at.tabs)}")
    print(f"  dataframes {len(at.dataframe)}")
    print(f"  selectboxes {[s.label for s in at.selectbox]}")
    print(f"  sliders    {[s.label for s in at.slider]}")
    print(f"  metrics    {len(at.metric)}")
    print("\npanel render is healthy")
    # Streamlit leaves a temp dir whose ACL the sandbox cannot clean; exiting
    # explicitly keeps the teardown traceback out of the output.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
