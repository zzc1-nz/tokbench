"""Prove the bare-script regression test actually catches the relative-import bug.

Temporarily disables the panel's bootstrap branch, runs the regression test, and
asserts it FAILS; then restores the file. Run manually when touching panel/app.py.

    python scripts/verify_panel_import_guard.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tokbench" / "panel" / "app.py"
BACKUP = ROOT / ".app.py.bak"

GUARD_START = 'if __package__ in (None, ""):'
ELSE_MARKER = "else:  # imported as a module"


def main() -> int:
    original = APP.read_text(encoding="utf-8")
    start = original.index(GUARD_START)
    end = original.index(ELSE_MARKER)

    # Break the bootstrap: the script branch no longer imports anything, so the
    # relative imports below run with no package context.
    broken = (
        original[:start]
        + "if False:  # deliberately broken for this check\n    pass\n"
        + original[end:]
    )

    shutil.copy2(APP, BACKUP)
    try:
        APP.write_text(broken, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_render.py::test_panel_runs_as_a_bare_script",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        caught = proc.returncode != 0 and "relative import" in output
        print("with bootstrap disabled:")
        print(f"  pytest exit = {proc.returncode}")
        print(f"  'relative import' in output = {'relative import' in output}")
        print(
            f"  RESULT: {'PASS - test catches the bug' if caught else 'FAIL - test does not catch it'}"
        )
        return 0 if caught else 1
    finally:
        shutil.copy2(BACKUP, APP)
        BACKUP.unlink(missing_ok=True)
        print("restored app.py")


if __name__ == "__main__":
    raise SystemExit(main())
