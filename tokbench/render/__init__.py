"""Rendering layer: self-contained HTML reports.

The public surface is deliberately tiny — one renderer, one writer — so the CLI
(``tokbench report``) has exactly one way to produce a report and cannot drift
from what the tests exercise.

The HTML is hand-authored inline SVG rather than a plotting library so the
output is a single file that opens offline with no network access and no JS.
Interactivity in the browser is the Streamlit panel's job, not this module's.
"""

from __future__ import annotations

from .html import render_html, write_html

__all__ = ["render_html", "write_html"]
