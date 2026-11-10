"""Storage layer: run directories, JSONL call log, and result persistence.

Layout of a run directory::

    <output.dir>/<run_id>/
        manifest.json      RunManifest (written at start, updated at end)
        calls.jsonl        one CallRecord per line, appended as calls complete
        results.jsonl      one TaskResult per line (all attempts aggregated)
        config.yaml        frozen config snapshot
        report.json        ReportPayload (written by `tokbench report`)
        report.html        rendered report

Append-only JSONL is deliberate: a crashed or budget-aborted run still leaves a
valid, partial dataset. Nothing is buffered in memory longer than one line.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import TokbenchError
from .models import CallRecord, RunManifest, TaskResult

T = TypeVar("T", bound=BaseModel)

RUN_ID_FMT = "%Y%m%d-%H%M%S"


def new_run_id(experiment: str) -> str:
    """Timestamped, filesystem-safe run id."""
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in experiment)[:32]
    return f"{datetime.now().strftime(RUN_ID_FMT)}-{safe or 'run'}"


class RunStore:
    """Read/write handle for one run directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.calls_path = self.root / "calls.jsonl"
        self.results_path = self.root / "results.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.config_path = self.root / "config.yaml"
        self.report_json_path = self.root / "report.json"
        self.report_html_path = self.root / "report.html"

    # ------------------------------------------------------------------ create
    @classmethod
    def create(cls, base_dir: str | Path, run_id: str) -> RunStore:
        root = Path(base_dir) / run_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @property
    def exists(self) -> bool:
        return self.root.exists()

    @property
    def run_id(self) -> str:
        return self.root.name

    # ------------------------------------------------------------------ append
    def _append_line(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append_call(self, record: CallRecord) -> None:
        self._append_line(self.calls_path, record.model_dump(mode="json"))

    def append_result(self, result: TaskResult) -> None:
        self._append_line(self.results_path, result.model_dump(mode="json"))

    # ------------------------------------------------------------------ read
    def _read_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TokbenchError(f"corrupt JSONL at {path.name}:{lineno}: {exc}") from exc

    def read_calls(self) -> list[CallRecord]:
        return [CallRecord.model_validate(d) for d in self._read_jsonl(self.calls_path)]

    def read_results(self) -> list[TaskResult]:
        return [TaskResult.model_validate(d) for d in self._read_jsonl(self.results_path)]

    def iter_calls(self) -> Iterator[CallRecord]:
        for d in self._read_jsonl(self.calls_path):
            yield CallRecord.model_validate(d)

    # ------------------------------------------------------------------ manifest
    def write_manifest(self, manifest: RunManifest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def read_manifest(self) -> RunManifest | None:
        if not self.manifest_path.exists():
            return None
        return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ misc
    def write_config_snapshot(self, yaml_text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml_text, encoding="utf-8")

    def write_text(self, name: str, text: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def write_json(self, name: str, payload: dict[str, Any] | BaseModel) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        p.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return p

    def stats(self) -> dict[str, Any]:
        def count(path: Path) -> int:
            if not path.exists():
                return 0
            with open(path, "rb") as fh:
                return sum(1 for line in fh if line.strip())

        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "calls": count(self.calls_path),
            "results": count(self.results_path),
            "has_report": self.report_json_path.exists(),
            "bytes": sum(p.stat().st_size for p in self.root.glob("*") if p.is_file()),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RunStore {self.root}>"


# --------------------------------------------------------------------------- #
# Discovery helpers (used by CLI + panel)
# --------------------------------------------------------------------------- #
def discover_runs(base_dir: str | Path) -> list[RunStore]:
    """All run directories under ``base_dir`` that contain a manifest."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = [
        RunStore(p)
        for p in sorted(base.iterdir(), reverse=True)
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    return runs


def latest_run(base_dir: str | Path) -> RunStore | None:
    runs = discover_runs(base_dir)
    return runs[0] if runs else None


def resolve_run_dir(path: str | Path) -> RunStore:
    """Accept either a run dir or a parent dir and return the right RunStore."""
    p = Path(path)
    if (p / "manifest.json").exists() or (p / "calls.jsonl").exists():
        return RunStore(p)
    latest = latest_run(p)
    if latest is None:
        raise TokbenchError(f"no tokbench runs found under {p}")
    return latest


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = [
    "RunStore",
    "discover_runs",
    "iter_jsonl",
    "latest_run",
    "load_json",
    "new_run_id",
    "resolve_run_dir",
]
