"""tokbench — a token-consumption benchmark harness for LLM coding agents.

The package is organised as a strict pipeline::

    config.yaml ──▶ runner ──▶ run dir (calls.jsonl + results.jsonl + manifest)
                                   │
                                   ▼
                              analysis ──▶ ReportPayload
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
                 render/html.py           panel/app.py
                 (static report)        (Streamlit dashboard)

Each arrow is a stable data contract (:mod:`tokbench.models`,
:mod:`tokbench.report_schema`), which is what lets the layers be tested and
reasoned about independently.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "tokbench"

__all__ = ["__version__"]
