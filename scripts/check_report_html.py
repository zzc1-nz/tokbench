"""Integrity check for the rendered HTML report.

Verifies the properties the report is *supposed* to have, so a regression in the
renderer is caught as a failing check rather than by eyeballing a 200KB file:

- self-contained: no external http(s) resource references, no <script> tags
- all eight sections present
- at least five inline SVG charts, all well-formed
- no leaked Python literals (None/NaN) in the output
- tables present for the leaderboard / retry / deltas

Usage::

    python scripts/check_report_html.py reports/<run-id>/report.html
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = (
    "section-pareto",
    "section-retry",
    "section-cache",
    "section-series",
    "section-stacked",
    "section-leaderboard",
    "section-strategy",
)

# Matches src=/href= pointing at an absolute web URL. A self-contained report must
# have none: it has to open offline, with no network.
EXTERNAL_REF = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/check_report_html.py <report.html>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    html = path.read_text(encoding="utf-8")
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {message}")
        if not condition:
            failures.append(message)

    print(f"checking {path} ({len(html):,} bytes)")

    n_svg = html.count("<svg")
    check(n_svg >= 5, f"inline SVG charts >= 5 (found {n_svg})")
    check(html.count("</svg>") == n_svg, "SVG tags balanced")
    check(not EXTERNAL_REF.search(html), "no external http(s) resource references")
    check("<script" not in html.lower(), "no <script> tags (CSS-only tooltips)")
    check(html.count("<table") >= 3, f"tables present (found {html.count('<table')})")

    for section in REQUIRED_SECTIONS:
        check(f'id="{section}"' in html, f"section {section}")

    # Leaked Python literals are the classic templating bug: `None` or `nan`
    # stringified into a cell. Look for them as standalone tokens.
    for literal in ("None", "nan", "NaN"):
        found = re.search(rf">\s*{literal}\s*<", html)
        check(found is None, f"no leaked '{literal}' literal in rendered text")

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nreport HTML is self-contained and complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
