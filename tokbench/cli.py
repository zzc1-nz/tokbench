"""Console entry point: ``tokbench``.

Command surface::

    tokbench run      --config experiment.yaml    # execute the matrix
    tokbench report   --run latest                # analyse + render
    tokbench show     --run latest                # terminal summary
    tokbench list     --dir reports               # discover runs
    tokbench validate --config experiment.yaml    # dry-run the plan
    tokbench panel                                # launch Streamlit
    tokbench providers                            # adapter inventory

Design note: the CLI is a thin shell. All real work lives in
``tokbench.runner``, ``tokbench.analysis`` and ``tokbench.render`` so the same
pipeline can be driven from a notebook or CI script without the CLI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .config import ExperimentConfig
from .errors import TokbenchError
from .pricing import DEFAULT_PRICE_BOOK, load_price_book
from .store import RunStore, discover_runs, resolve_run_dir

app = typer.Typer(
    name="tokbench",
    help="Benchmark LLM token consumption and cost across models and prompt strategies.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Formatting helpers (shared by `show` and `report`)
# --------------------------------------------------------------------------- #
def fmt_tokens(n: float | int) -> str:
    """Human token count: 1234 -> '1.2k', 4500000 -> '4.5M'."""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n)}"


def fmt_usd(v: float) -> str:
    if v == 0:
        return "$0"
    if abs(v) < 0.01:
        return f"${v:.5f}"
    if abs(v) < 1:
        return f"${v:.4f}"
    return f"${v:,.2f}"


def fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


def _load_config(path: Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.from_yaml(path)
    except TokbenchError as exc:
        _fail(str(exc))
        raise  # unreachable, keeps type checkers happy


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", help="Experiment YAML.")],
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Explicit run id (default: timestamped).")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Override output directory.")
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Cap the number of tasks (smoke testing)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan and exit without calling models.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress the progress bar.")
    ] = False,
) -> None:
    """Execute a benchmark matrix defined by an experiment config."""
    from .runner import load_taskset, run_experiment  # local import: keep CLI startup fast

    cfg = _load_config(config)
    if output is not None:
        cfg.output.dir = str(output)
    if limit is not None:
        cfg.taskset.limit = limit

    try:
        taskset = load_taskset(cfg.resolve_path(cfg.taskset.path), limit=cfg.taskset.limit)
    except TokbenchError as exc:
        _fail(str(exc))
        raise

    total_cells = cfg.planned_cells(len(taskset))
    est_prompt_tokens = total_cells * 2000  # documented planning assumption

    # ---- dry run ---------------------------------------------------------
    if dry_run:
        _print_plan(cfg, len(taskset), total_cells, est_prompt_tokens)
        return

    _print_banner(cfg, len(taskset), total_cells)

    progress_bar: Progress | None = None
    task_id = None
    if not quiet:
        progress_bar = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=32, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            TimeElapsedColumn(),
            console=console,
        )
        progress_bar.start()
        task_id = progress_bar.add_task("benchmarking", total=total_cells, detail="starting...")

    def on_progress(event) -> None:  # noqa: ANN001 - duck-typed ProgressEvent
        if progress_bar is None or task_id is None:
            return
        detail = (
            f"{event.model_id} · {event.strategy_id} · {fmt_tokens(event.usage.total_tokens)} tok"
        )
        progress_bar.update(task_id, advance=1, detail=detail)

    started = time.perf_counter()
    try:
        outcome = run_experiment(
            cfg,
            taskset=taskset,
            run_id=run_id,
            output_base=cfg.output.dir,
            progress=on_progress,
        )
    except TokbenchError as exc:
        if progress_bar:
            progress_bar.stop()
        _fail(str(exc))
        raise
    finally:
        if progress_bar is not None:
            progress_bar.stop()

    elapsed = time.perf_counter() - started
    _print_outcome(outcome, elapsed)

    if outcome.aborted_reason:
        console.print(
            Panel(
                f"[yellow]{outcome.aborted_reason}[/]\n"
                f"Partial results were still written to [cyan]{outcome.store.root}[/].",
                title="run aborted",
                border_style="yellow",
            )
        )

    console.print(f"\n[dim]next:[/] tokbench report --run [cyan]{outcome.store.root}[/]")


def _print_banner(cfg: ExperimentConfig, n_tasks: int, total_cells: int) -> None:
    lines = [
        f"[bold]experiment[/]  {cfg.experiment.name}",
        f"[bold]models[/]      {', '.join(m.id for m in cfg.models)}",
        f"[bold]strategies[/]  {', '.join(s.id for s in cfg.strategies)}",
        f"[bold]tasks[/]       {n_tasks}  ·  [bold]repeats[/] {cfg.experiment.repeats}",
        f"[bold]cells[/]       {total_cells:,}  ·  [bold]concurrency[/] {cfg.experiment.concurrency}",
        f"[bold]mode[/]        {cfg.adapter_mode}",
    ]
    console.print(Panel("\n".join(lines), title="[cyan]tokbench run[/]", border_style="cyan"))


def _print_plan(cfg: ExperimentConfig, n_tasks: int, total_cells: int, est_tokens: int) -> None:
    table = Table(title="dry run — no API calls will be made", border_style="cyan")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("experiment", cfg.experiment.name)
    table.add_row("models", ", ".join(m.id for m in cfg.models))
    table.add_row("strategies", ", ".join(s.id for s in cfg.strategies))
    table.add_row("tasks", str(n_tasks))
    table.add_row("repeats", str(cfg.experiment.repeats))
    table.add_row("cells", f"{total_cells:,}")
    table.add_row("est. prompt tokens", f"~{fmt_tokens(est_tokens)}")
    table.add_row("max attempts / cell", str(cfg.retry.max_attempts))
    table.add_row("budget", json.dumps(cfg.budget.model_dump(exclude_none=True)))
    table.add_row("adapter mode", cfg.adapter_mode)
    table.add_row("output dir", cfg.output.dir)
    console.print(table)
    console.print(
        "[dim]The estimate assumes 2,000 prompt tokens per call. "
        "Run with --limit N first to calibrate.[/]"
    )


def _print_outcome(outcome, elapsed: float) -> None:  # noqa: ANN001
    m = outcome.manifest
    table = Table(border_style="green", show_header=False, pad_edge=False)
    table.add_column("k", style="dim")
    table.add_column("v", style="bold")
    table.add_row("run id", outcome.run_id)
    table.add_row("cells", f"{outcome.completed_cells:,} / {outcome.planned_cells:,}")
    table.add_row("calls", f"{m.completed_calls:,}")
    table.add_row("failed calls", str(m.failed_calls))
    table.add_row("retried cells", str(m.retried_cells))
    table.add_row("total tokens", fmt_tokens(m.total_usage.total_tokens))
    table.add_row("billable tokens", fmt_tokens(m.total_usage.billable_tokens))
    table.add_row("cost", fmt_usd(m.total_cost_usd))
    table.add_row("wall clock", f"{elapsed:.1f}s")
    console.print(table)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
@app.command()
def report(
    run_dir: Annotated[
        Path | None,
        typer.Option("--run", "-r", help="Run directory (default: latest under --dir)."),
    ] = None,
    dir_: Annotated[
        Path, typer.Option("--dir", "-d", help="Base directory containing runs.")
    ] = Path("reports"),
    prices: Annotated[
        Path | None, typer.Option("--prices", help="Price-book YAML override.")
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="Baseline strategy id for ablation deltas."),
    ] = None,
    formats: Annotated[
        str, typer.Option("--formats", help="Comma-separated: json,html,md.")
    ] = "json,html",
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the HTML report after writing.")
    ] = False,
) -> None:
    """Analyse a finished run and render the structured report."""
    from .analysis.report import build_report, write_report

    store = _resolve_store(run_dir, dir_)
    book = _load_prices(prices)

    with console.status("[cyan]analysing run...", spinner="dots"):
        try:
            payload = build_report(store, price_book=book, baseline_strategy=baseline)
        except TokbenchError as exc:
            _fail(str(exc))
            raise

    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    unknown = set(fmts) - {"json", "html", "md"}
    if unknown:
        _fail(f"unknown report format(s): {', '.join(sorted(unknown))}")

    paths = write_report(payload, store, formats=fmts)
    _print_report_summary(payload)

    console.print()
    for p in paths:
        console.print(f"[green]wrote[/] [cyan]{p}[/]")

    if open_browser:
        html_path = next((p for p in paths if p.suffix == ".html"), None)
        if html_path:
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())


def _resolve_store(run_dir: Path | None, dir_: Path) -> RunStore:
    target = run_dir if run_dir is not None else dir_
    try:
        return resolve_run_dir(target)
    except TokbenchError as exc:
        _fail(str(exc))
        raise


def _load_prices(prices: Path | None):
    if prices is None:
        return DEFAULT_PRICE_BOOK
    try:
        return load_price_book(str(prices))
    except Exception as exc:
        _fail(f"could not load price book {prices}: {exc}")
        raise


def _print_report_summary(payload) -> None:  # noqa: ANN001
    s = payload.summary
    console.print(
        Panel(
            "\n".join(
                [
                    f"[bold]experiment[/]  {payload.meta.experiment}",
                    f"[bold]run[/]         {payload.meta.run_id}",
                    f"[bold]tokens[/]      {fmt_tokens(s.total_usage.total_tokens)} total"
                    f" · {fmt_tokens(s.total_billable_tokens)} billable",
                    f"[bold]cost[/]        {fmt_usd(s.total_cost_usd)}"
                    f"  ([dim]{fmt_usd(s.cost_per_1k_billable_tokens)} / 1k billable[/])",
                    f"[bold]cache[/]       {fmt_pct(s.cache_hit_rate)} hit rate",
                    f"[bold]retry waste[/] {fmt_pct(payload.retry.overall_retry_cost_share)}"
                    f" of spend = {fmt_usd(payload.retry.total_retry_cost_usd)}",
                ]
            ),
            title="[cyan]report[/]",
            border_style="cyan",
        )
    )

    if payload.pareto.frontier_labels:
        console.print(
            "\n[bold]pareto frontier:[/] " + " [dim]→[/] ".join(payload.pareto.frontier_labels)
        )
        if payload.pareto.explanation:
            console.print(f"[dim]{payload.pareto.explanation}[/]")


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
@app.command()
def show(
    run_dir: Annotated[Path | None, typer.Option("--run", "-r", help="Run directory.")] = None,
    dir_: Annotated[Path, typer.Option("--dir", "-d")] = Path("reports"),
    top: Annotated[int, typer.Option("--top", help="Leaderboard rows to show.")] = 10,
) -> None:
    """Print a terminal summary of a run (no files written)."""
    store = _resolve_store(run_dir, dir_)
    manifest = store.read_manifest()
    if manifest is None:
        _fail(f"{store.root} has no manifest.json")

    table = Table(title=f"run {manifest.run_id}", border_style="cyan")
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right", style="bold")
    table.add_row("experiment", manifest.experiment_name)
    table.add_row("mode", manifest.adapter_mode)
    table.add_row("calls", f"{manifest.completed_calls:,} (+{manifest.failed_calls} failed)")
    table.add_row("retried cells", str(manifest.retried_cells))
    table.add_row("total tokens", fmt_tokens(manifest.total_usage.total_tokens))
    table.add_row("billable tokens", fmt_tokens(manifest.total_usage.billable_tokens))
    table.add_row("cache hit rate", fmt_pct(manifest.total_usage.cache_hit_rate))
    table.add_row("cost", fmt_usd(manifest.total_cost_usd))
    table.add_row("wall clock", f"{manifest.wall_clock_s:.1f}s")
    console.print(table)

    try:
        from .analysis.report import build_report

        payload = build_report(store)
    except Exception:
        return

    if payload.leaderboard:
        lb = Table(title=f"top {top} by token efficiency", border_style="green")
        for col in ("#", "model", "strategy", "tok/task", "quality", "$/task", "score"):
            lb.add_column(col, justify="right" if col != "model" and col != "strategy" else "left")
        for row in payload.leaderboard[:top]:
            lb.add_row(
                str(row.rank),
                row.model_id,
                row.strategy_id,
                fmt_tokens(row.billable_tokens_per_task),
                f"{row.mean_quality:.3f}",
                fmt_usd(row.cost_per_task_usd),
                f"{row.efficiency_score:.3f}",
            )
        console.print(lb)


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
@app.command("list")
def list_runs(
    dir_: Annotated[Path, typer.Option("--dir", "-d")] = Path("reports"),
) -> None:
    """List discovered runs under a directory."""
    runs = discover_runs(dir_)
    if not runs:
        console.print(f"[dim]no runs found under {dir_}[/]")
        return

    table = Table(title=f"runs in {dir_}", border_style="cyan")
    for col in ("run id", "experiment", "calls", "tokens", "cost", "report"):
        table.add_column(col)
    for store in runs:
        m = store.read_manifest()
        if m is None:
            continue
        table.add_row(
            m.run_id,
            m.experiment_name,
            f"{m.completed_calls:,}",
            fmt_tokens(m.total_usage.total_tokens),
            fmt_usd(m.total_cost_usd),
            "[green]yes[/]" if store.report_json_path.exists() else "[dim]no[/]",
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
@app.command()
def validate(
    config: Annotated[Path, typer.Option("--config", "-c")],
    prices: Annotated[Path | None, typer.Option("--prices")] = None,
) -> None:
    """Validate an experiment config without running anything."""
    from .runner import load_taskset

    cfg = _load_config(config)
    ok = True

    try:
        taskset = load_taskset(cfg.resolve_path(cfg.taskset.path), limit=cfg.taskset.limit)
        console.print(
            f"[green]ok[/] taskset: {len(taskset)} tasks "
            f"across {len(taskset.categories)} categories"
        )
    except TokbenchError as exc:
        console.print(f"[red]fail[/] taskset: {exc}")
        ok = False
        taskset = None

    book = _load_prices(prices)
    for m in cfg.models:
        if book.has(m.price_key):
            p = book.get(m.price_key)
            console.print(
                f"[green]ok[/] price for [bold]{m.id}[/]: in {p.input}/M out {p.output}/M"
            )
        else:
            console.print(f"[red]fail[/] no price entry for model [bold]{m.id}[/]")
            ok = False

    if taskset is not None:
        cells = cfg.planned_cells(len(taskset))
        est = cells * 2000
        console.print(
            f"[green]ok[/] plan: {cells:,} cells (~{fmt_tokens(est)} prompt tokens at 2k/call)"
        )
        if cfg.budget.max_total_tokens and est > cfg.budget.max_total_tokens:
            console.print(
                f"[yellow]warn[/] estimate {fmt_tokens(est)} exceeds budget "
                f"{fmt_tokens(cfg.budget.max_total_tokens)}; the run will abort early"
            )

    if not ok:
        raise typer.Exit(1)
    console.print("[bold green]config is valid[/]")


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
@app.command()
def panel(
    dir_: Annotated[Path, typer.Option("--dir", "-d")] = Path("reports"),
    port: Annotated[int, typer.Option("--port", "-p")] = 8501,
    headless: Annotated[bool, typer.Option("--headless")] = True,
) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).parent / "panel" / "app.py"
    if not app_path.exists():
        _fail(f"panel not found at {app_path}")

    env = dict(os.environ)
    env.setdefault("TOKBENCH_RUNS_DIR", str(dir_.resolve()))
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
    ]
    console.print(f"[cyan]launching panel[/] → http://localhost:{port}")
    console.print(f"[dim]{' '.join(cmd)}[/]")
    os.execvpe(sys.executable, cmd, env)


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
@app.command()
def providers() -> None:
    """List the available model adapters and their credential requirements."""
    try:
        from .adapters.registry import ADAPTERS, available_providers
    except Exception as exc:  # pragma: no cover - registry always present in-tree
        _fail(f"adapter registry unavailable: {exc}")
        return

    # Inspect the classes directly rather than a bespoke descriptor helper: the
    # registry's contract is the adapter classes themselves, so the CLI cannot
    # drift out of sync with it.
    table = Table(title="model adapters", border_style="cyan")
    for col in ("provider", "cache accounting", "credentials", "notes"):
        table.add_column(col)
    for name in available_providers():
        cls = ADAPTERS[name]
        table.add_row(
            name,
            "yes" if getattr(cls, "supports_cache_accounting", False) else "no",
            "required" if getattr(cls, "requires_credentials", False) else "none",
            "offline / deterministic"
            if name == "mock"
            else (cls.__doc__ or "").strip().splitlines()[0][:60],
        )
    console.print(table)
    console.print(f"[dim]{len(available_providers())} providers registered[/]")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"tokbench {__version__}")


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
