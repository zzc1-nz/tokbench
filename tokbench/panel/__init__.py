"""Streamlit dashboard package for tokbench.

``app.py`` is a Streamlit *script* (Streamlit executes it top to bottom), so it
cannot be imported without side effects.  This package exposes the pure pieces —
loading and the figure builders — for tests and for embedding in a notebook,
and re-exports ``main`` for the ``tokbench panel`` CLI command.

Keeping the import light matters: ``tokbench.cli`` imports this package to find
``app.py``, and that path must not drag in a Streamlit runtime.
"""

from __future__ import annotations

from .app import (
    build_cache_figure,
    build_delta_figure,
    build_pareto_figure,
    build_retry_figure,
    build_series_figure,
    build_stacked_figure,
    cache_what_if,
    deltas_frame,
    leaderboard_frame,
    load_payload,
    main,
    main_payload,
    raw_frames,
    retry_frame,
    run_choices,
    series_frame,
    stacked_frame,
)

__all__ = [
    "build_cache_figure",
    "build_delta_figure",
    "build_pareto_figure",
    "build_retry_figure",
    "build_series_figure",
    "build_stacked_figure",
    "cache_what_if",
    "deltas_frame",
    "leaderboard_frame",
    "load_payload",
    "main",
    "main_payload",
    "raw_frames",
    "retry_frame",
    "run_choices",
    "series_frame",
    "stacked_frame",
]
