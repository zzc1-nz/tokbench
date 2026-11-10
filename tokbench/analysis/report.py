"""Assemble a run directory into a complete :class:`~tokbench.report_schema.ReportPayload`.

This is the **only** module in :mod:`tokbench.analysis` that touches the
filesystem.  Everything it calls is a pure function over lists, so the same
payload can be reproduced from an in-memory fixture in a test, from a JSONL dump
in CI, or from a notebook — the file layer is a thin shell around
:func:`build_report`.

Report assembly rules
---------------------
* **Meta** comes from the manifest when present, otherwise from the first
  result/call (a crashed run may have no manifest but a usable call log).  The
  price book's name/date/currency are stamped in so a number in the report can
  always be traced to the price list that produced it.
* **Notes** record every assumption the analysis had to make — adapter mode,
  missing manifest, models absent from the price book, an ignored baseline
  request.  A report that hides its assumptions is not auditable, and this is the
  field that keeps it honest.
* **Provider** is filled onto leaderboard rows from the call log, because
  :class:`~tokbench.models.TaskResult` deliberately does not carry it (it is a
  property of the call, not of the graded outcome).
* Every section is computed even for an empty run; the schema's defaults plus the
  modules' explanatory strings make a zero-cell run a *valid* report that says why
  it is empty.

Output formats
--------------
:func:`write_report` writes JSON natively and delegates HTML/Markdown to
:mod:`tokbench.render` via a lazy import, so the analysis layer never depends on
the rendering layer at import time (and the panel keeps working while renderers
are still being built).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..errors import ReportError
from ..models import CallRecord, RunManifest, TaskResult
from ..pricing import DEFAULT_PRICE_BOOK, PriceBook
from ..report_schema import ReportMeta, ReportPayload
from ..store import RunStore
from .aggregate import (
    category_breakdown,
    leaderboard,
    resolve_baseline_strategy,
    stacked_bars,
    summarize,
    time_series,
)
from .cache import cache_sensitivity
from .deltas import strategy_deltas
from .pareto import pareto_frontier
from .retry import retry_analysis

#: Report formats :func:`write_report` knows how to produce.
SUPPORTED_FORMATS: tuple[str, ...] = ("json", "html", "md")


def _resolve_experiment(
    manifest: RunManifest | None,
    results: list[TaskResult],
    calls: list[CallRecord],
) -> str:
    """Experiment name: manifest first, then any record, then a placeholder."""
    if manifest is not None and manifest.experiment_name:
        return manifest.experiment_name
    if results:
        return results[0].experiment
    if calls:
        return calls[0].experiment
    return "unknown"


def _resolve_run_id(
    store: RunStore,
    manifest: RunManifest | None,
    results: list[TaskResult],
    calls: list[CallRecord],
) -> str:
    """Run id: manifest first, else the directory name (always available)."""
    if manifest is not None and manifest.run_id:
        return manifest.run_id
    return store.run_id


def _providers_by_model(calls: list[CallRecord]) -> dict[str, str]:
    """Model id -> provider, chosen deterministically (lowest record id wins).

    A run should never see one model id served by two providers, but if it does,
    picking the smallest record id keeps the report stable across renders.
    """
    chosen: dict[str, str] = {}
    for call in sorted(calls, key=lambda c: (c.model_id, c.record_id)):
        if call.model_id not in chosen and call.provider:
            chosen[call.model_id] = call.provider
    return chosen


def _unpriced_models(results: list[TaskResult], price_book: PriceBook) -> list[str]:
    """Models present in the run but absent from the price book (sorted)."""
    return sorted({r.model_id for r in results if not price_book.has(r.model_id)})


def _quality_histogram(results: list[TaskResult]) -> dict[str, int]:
    """Coarse distribution of graded quality, keyed by decile label.

    Cheap to compute and the one distribution the schema has no home for; it lets
    a consumer spot a bimodal grader (0.0/1.0) without re-reading the results.
    Ungraded cells are excluded, same rule as :func:`mean_quality`.
    """
    buckets = {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": 0 for i in range(10)}
    for result in results:
        if result.quality is None:
            continue
        index = min(int(result.quality.score * 10), 9)
        buckets[f"{index / 10:.1f}-{(index + 1) / 10:.1f}"] += 1
    return buckets


def _build_notes(
    *,
    manifest: RunManifest | None,
    results: list[TaskResult],
    calls: list[CallRecord],
    price_book: PriceBook,
    requested_baseline: str | None,
    resolved_baseline: str | None,
    unpriced: list[str],
) -> list[str]:
    """Documented assumptions behind the numbers (see module docstring)."""
    notes: list[str] = []

    mode = manifest.adapter_mode if manifest is not None else "unknown"
    mode_notes = {
        "mock": "adapter_mode=mock: token counts are synthetic and deterministic; "
        "treat cost figures as a model of spend, not observed spend.",
        "live": "adapter_mode=live: token counts and costs come from real provider responses.",
        "replay": "adapter_mode=replay: results were replayed from a previous run's "
        "recorded calls; no API was called.",
    }
    notes.append(
        mode_notes.get(
            mode, f"adapter_mode={mode}: cost figures inherit the adapter's token accounting."
        )
    )

    if manifest is None:
        notes.append(
            "No manifest.json in this run directory: summary counts fall back to the "
            "result and call logs, and wall-clock time is inferred from the call log."
        )
    if not results:
        notes.append(
            "No completed cells in this run: every section is empty and the "
            "explanations state why rather than reporting a misleading zero."
        )
    if unpriced:
        notes.append(
            f"{len(unpriced)} model(s) have no price-book entry "
            f"({', '.join(unpriced)}); their recorded cost_usd was used verbatim and "
            "cache projections were skipped for them."
        )
    if requested_baseline is not None and resolved_baseline != requested_baseline:
        notes.append(
            f"Requested baseline strategy {requested_baseline!r} is not present in "
            f"the results; auto-picked {resolved_baseline!r} (fewest billable tokens "
            "per task) for the ablation table."
        )
    if resolved_baseline is not None and requested_baseline is None:
        notes.append(
            f"Ablation baseline auto-selected: {resolved_baseline!r} (fewest billable "
            "tokens per task)."
        )
    notes.append(
        "Costs are re-derived from the price book over recorded token counts, so the "
        "report can be restated with a different price book without re-running."
    )
    return notes


def build_report(
    store: RunStore,
    *,
    price_book: PriceBook | None = None,
    baseline_strategy: str | None = None,
    manifest: RunManifest | None = None,
) -> ReportPayload:
    """Load a run directory and assemble the complete report payload.

    Parameters
    ----------
    store:
        Handle on the run directory; its ``calls.jsonl`` / ``results.jsonl`` /
        ``manifest.json`` are read here and nowhere else.
    price_book:
        Price list used to re-derive cost.  Defaults to
        :data:`tokbench.pricing.DEFAULT_PRICE_BOOK`.
    baseline_strategy:
        Strategy id the ablation table measures against.  Unknown ids fall back to
        the leanest strategy and the fallback is recorded in ``meta.notes``.
    manifest:
        Pre-loaded manifest (e.g. from the runner that just finished); otherwise
        read from disk.

    Never raises on an empty or partial run: a run directory with zero results
    yields a valid payload whose notes and section explanations say so.
    """
    book = price_book if price_book is not None else DEFAULT_PRICE_BOOK
    calls = store.read_calls()
    results = store.read_results()
    resolved_manifest = manifest if manifest is not None else store.read_manifest()

    resolved_baseline = resolve_baseline_strategy(results, baseline_strategy)
    unpriced = _unpriced_models(results, book)

    meta = ReportMeta(
        run_id=_resolve_run_id(store, resolved_manifest, results, calls),
        experiment=_resolve_experiment(resolved_manifest, results, calls),
        price_book=book.name,
        price_book_date=book.effective_date,
        currency=book.currency,
        notes=_build_notes(
            manifest=resolved_manifest,
            results=results,
            calls=calls,
            price_book=book,
            requested_baseline=baseline_strategy,
            resolved_baseline=resolved_baseline,
            unpriced=unpriced,
        ),
    )

    board = leaderboard(results, book)
    providers = _providers_by_model(calls)
    for row in board:
        row.provider = providers.get(row.model_id, "")

    return ReportPayload(
        meta=meta,
        summary=summarize(calls, results, resolved_manifest, price_book=book),
        leaderboard=board,
        categories=category_breakdown(results, book),
        pareto=pareto_frontier(results, book, baseline_strategy=baseline_strategy),
        retry=retry_analysis(calls, results, book),
        cache=cache_sensitivity(results, book),
        series=time_series(calls),
        stacked=stacked_bars(results, book),
        strategy_deltas=strategy_deltas(results, baseline=baseline_strategy, price_book=book),
        manifest=resolved_manifest,
        raw={
            "baseline_strategy": resolved_baseline,
            "baseline_source": (
                "none"
                if resolved_baseline is None
                else "explicit"
                if resolved_baseline == baseline_strategy
                else "auto"
            ),
            "unpriced_models": unpriced,
            "quality_histogram": _quality_histogram(results),
            "cells": len(results),
            "calls": len(calls),
            "groups": len({(r.model_id, r.strategy_id) for r in results}),
        },
    )


def dumps(payload: ReportPayload) -> str:
    """Compact, key-sorted JSON for the payload.

    Sorted keys keep the artefact byte-identical for identical runs, which makes
    ``report.json`` diffable in git — useful for spotting a metric that silently
    changed between engine versions.
    """
    return json.dumps(
        payload.to_json_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def load_report(path: str | Path) -> ReportPayload:
    """Read a previously written ``report.json`` back into a payload."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"cannot read report {target}: {exc}") from exc
    try:
        return ReportPayload.model_validate_json(text)
    except ValueError as exc:
        raise ReportError(f"invalid report payload in {target}: {exc}") from exc


def _write_json(payload: ReportPayload, store: RunStore) -> Path:
    return store.write_json("report.json", payload)


def _write_html(payload: ReportPayload, store: RunStore) -> Path:
    """Delegate to the renderer; import lazily so analysis has no render dep."""
    try:
        from ..render.html import write_html
    except ImportError as exc:  # pragma: no cover - renderer built separately
        raise ReportError(
            "HTML output requested but tokbench.render.html is unavailable: "
            f"{exc}. Use formats=('json',) or install the render layer."
        ) from exc
    return write_html(payload, store.report_html_path)


def _write_markdown(payload: ReportPayload, store: RunStore) -> Path:
    """Minimal built-in Markdown renderer for ``formats=('md',)``.

    Deliberately self-contained: the HTML/panel renderers are a separate layer,
    but a Markdown summary is cheap enough to own here, and it is the format
    people paste into a PR description.
    """
    summary = payload.summary
    retry = payload.retry
    cache = payload.cache

    lines: list[str] = [
        f"# tokbench report — {payload.meta.experiment}",
        "",
        f"- run: `{payload.meta.run_id}`",
        f"- price book: `{payload.meta.price_book}` "
        f"({payload.meta.price_book_date or 'undated'}), {payload.meta.currency}",
        f"- cells: {summary.completed_cells} / {summary.planned_cells} planned "
        f"({summary.success_rate * 100:.1f}% success)",
        f"- spend: ${summary.total_cost_usd:.6f} over "
        f"{summary.total_usage.total_tokens:,} tokens "
        f"({summary.total_billable_tokens:,} billable)",
        f"- cache hit rate: {summary.cache_hit_rate * 100:.1f}%",
        f"- retry waste: {retry.overall_retry_cost_share * 100:.2f}% of spend "
        f"(${retry.total_retry_cost_usd:.6f})",
        "",
        "## Token-efficiency leaderboard",
        "",
        "| # | model | strategy | billable tok/task | quality | $/task | score |",
        "|--:|-------|----------|------------------:|--------:|-------:|------:|",
    ]
    for row in payload.leaderboard:
        lines.append(
            f"| {row.rank} | {row.model_id} | {row.strategy_id} | "
            f"{row.billable_tokens_per_task:,.0f} | {row.mean_quality:.3f} | "
            f"${row.cost_per_task_usd:.6f} | {row.efficiency_score:.3f} |"
        )

    lines += ["", "## Pareto frontier", ""]
    if payload.pareto.frontier_labels:
        lines.append("Frontier: " + " → ".join(payload.pareto.frontier_labels))
    lines.append("")
    lines.append(payload.pareto.explanation)
    lines += ["", "## Retry overhead", "", retry.explanation]
    lines += ["", "## Cache sensitivity", "", cache.explanation]

    if payload.strategy_deltas:
        lines += [
            "",
            "## Strategy ablation",
            "",
            "| model | strategy | token Δ | cost Δ | quality Δ | verdict |",
            "|-------|----------|--------:|-------:|----------:|---------|",
        ]
        for delta in payload.strategy_deltas:
            lines.append(
                f"| {delta.model_id or 'all models'} | {delta.strategy_id} | "
                f"{delta.token_delta_pct:+.1f}% | {delta.cost_delta_pct:+.1f}% | "
                f"{delta.quality_delta:+.3f} | {delta.verdict} |"
            )

    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in payload.meta.notes]
    lines.append("")

    return store.write_text("report.md", "\n".join(lines))


def write_report(
    payload: ReportPayload,
    store: RunStore,
    *,
    formats: Iterable[str] = ("json",),
) -> list[Path]:
    """Write ``payload`` into ``store`` in the requested formats.

    Returns the written paths **in the order the formats were requested**, so a
    caller can report "wrote ..." deterministically.  Raises
    :class:`~tokbench.errors.ReportError` for an unknown format instead of
    silently writing nothing.
    """
    requested = [f.strip().lower() for f in formats if f and f.strip()]
    if not requested:
        raise ReportError("no report formats requested")

    unknown = sorted({f for f in requested if f not in SUPPORTED_FORMATS})
    if unknown:
        raise ReportError(
            f"unknown report format(s): {', '.join(unknown)}; "
            f"supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    writers = {
        "json": _write_json,
        "html": _write_html,
        "md": _write_markdown,
    }
    paths: list[Path] = []
    for fmt in requested:
        paths.append(writers[fmt](payload, store))
    return paths


def build_and_write(
    store: RunStore,
    *,
    price_book: PriceBook | None = None,
    baseline_strategy: str | None = None,
    formats: Iterable[str] = ("json",),
) -> tuple[ReportPayload, list[Path]]:
    """Convenience wrapper: build then write.  Kept for CLI/notebook ergonomics."""
    payload = build_report(store, price_book=price_book, baseline_strategy=baseline_strategy)
    return payload, write_report(payload, store, formats=formats)


__all__ = [
    "SUPPORTED_FORMATS",
    "build_and_write",
    "build_report",
    "dumps",
    "load_report",
    "write_report",
]
