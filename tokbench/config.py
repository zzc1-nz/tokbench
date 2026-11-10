"""Experiment configuration schema (``experiment.yaml``).

The config is the *entire* experiment declaration: what to test (models), how to
prompt (strategies), on what (task set), how many times (repeats), and under what
budget/retry policy.  ``tokbench run`` expands it into a deterministic list of
:class:`Cell` objects so a run is reproducible from the YAML alone.

Expansion order (stable, so a partial run can resume by index):

    for repeat in 0..repeats-1
        for model in models
            for strategy in strategies
                for task in taskset.tasks

The nesting matters: outer loops are cheap to skip, so a budget abort truncates
the tail rather than scattering holes through the matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError

SystemPromptPreset = Literal["none", "short", "medium", "long"]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ModelSpec(BaseModel):
    """One model under test."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable alias used in reports (e.g. 'ds-chat')")
    provider: str = Field(
        default="mock",
        description="Adapter id: mock | openai | anthropic | deepseek | ollama",
    )
    model: str = Field(default="", description="Provider model name; defaults to id")
    base_url: str | None = None
    env_key: str | None = None
    api_key: str | None = None
    max_output_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults(self) -> ModelSpec:
        if not self.model:
            object.__setattr__(self, "model", self.id)
        return self

    @property
    def price_key(self) -> str:
        """Key used to look the model up in the price book."""
        return self.model or self.id


# --------------------------------------------------------------------------- #
# Strategies — the experimental variable
# --------------------------------------------------------------------------- #
class StrategySpec(BaseModel):
    """A prompt strategy: the knob we are ablating.

    Every field maps to a concrete, measurable prompt-assembly decision, so the
    token delta between two strategies is attributable to exactly one change.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable alias used in reports (e.g. 'cot-on')")
    description: str = ""
    system_prompt_preset: SystemPromptPreset = "short"
    system_prompt: str | None = Field(
        default=None, description="Inline override; wins over the preset"
    )
    few_shot: int = Field(default=0, ge=0, le=16)
    cot: bool = Field(default=False, description="Ask for step-by-step reasoning")
    compress_tool_output: bool = Field(
        default=False, description="Collapse tool/context blobs to a summary"
    )
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("few_shot")
    @classmethod
    def _few_shot_sane(cls, v: int) -> int:
        if v > 16:
            raise ValueError("few_shot above 16 is not a useful ablation")
        return v


# --------------------------------------------------------------------------- #
# Retry / budget / output
# --------------------------------------------------------------------------- #
class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_s: float = Field(default=0.5, ge=0.0)
    backoff_max_s: float = Field(default=8.0, ge=0.0)
    jitter: bool = True
    retry_on: list[str] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error", "network"]
    )

    @model_validator(mode="after")
    def _backoff_order(self) -> RetryPolicy:
        if self.backoff_max_s < self.backoff_base_s:
            raise ValueError("backoff_max_s must be >= backoff_base_s")
        return self


class BudgetSpec(BaseModel):
    """Hard ceilings. Hitting one aborts remaining cells, it does not fail the run."""

    model_config = ConfigDict(extra="forbid")

    max_total_tokens: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_calls: int | None = Field(default=None, ge=0)


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "reports"
    formats: list[Literal["json", "html", "md"]] = Field(default_factory=lambda: ["json", "html"])
    keep_outputs: bool = Field(
        default=True, description="Store raw model output text in the call log"
    )


class TasksetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "tasks/coding_bench.jsonl"
    name: str | None = None
    limit: int | None = Field(default=None, ge=1)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "experiment"
    description: str = ""
    seed: int = 42
    repeats: int = Field(default=1, ge=1, le=20)
    concurrency: int = Field(default=4, ge=1, le=64)


class ExperimentConfig(BaseModel):
    """Root of ``experiment.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    experiment: ExperimentSpec = Field(default_factory=ExperimentSpec)
    taskset: TasksetSpec = Field(default_factory=TasksetSpec)
    models: list[ModelSpec] = Field(min_length=1)
    strategies: list[StrategySpec] = Field(min_length=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    prices: str | None = Field(default=None, description="Path to a price-book YAML")
    adapter_mode: Literal["mock", "live", "auto"] = Field(
        default="mock",
        description="mock = synthetic deterministic responses; live = real API calls",
    )
    config_path: str | None = Field(default=None, exclude=True)

    # ------------------------------------------------------------ validation
    @model_validator(mode="after")
    def _unique_ids(self) -> ExperimentConfig:
        m = [x.id for x in self.models]
        if len(set(m)) != len(m):
            raise ValueError(f"duplicate model ids: {sorted({i for i in m if m.count(i) > 1})}")
        s = [x.id for x in self.strategies]
        if len(set(s)) != len(s):
            raise ValueError(f"duplicate strategy ids: {sorted({i for i in s if s.count(i) > 1})}")
        return self

    # ------------------------------------------------------------ helpers
    @property
    def task_count(self) -> int:
        """Number of tasks; requires the task set to be loaded."""
        return getattr(self, "_task_count", 0)

    def planned_cells(self, task_count: int) -> int:
        return len(self.models) * len(self.strategies) * task_count * self.experiment.repeats

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"config_path"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def expanded_cells(self, tasks: list[Any]) -> list[Cell]:
        """Deterministic expansion into experiment cells."""
        cells: list[Cell] = []
        for repeat in range(self.experiment.repeats):
            for model in self.models:
                for strategy in self.strategies:
                    for task in tasks:
                        cells.append(
                            Cell(
                                model=model,
                                strategy=strategy,
                                task=task,
                                repeat=repeat,
                            )
                        )
        return cells

    # ------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        try:
            return cls.model_validate(data)
        except Exception as exc:  # pydantic -> our error type
            raise ConfigError(f"invalid experiment config: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML parse error in {p}: {exc}") from exc
        cfg = cls.from_dict(data)
        cfg.config_path = str(p.resolve())
        return cfg

    def resolve_path(self, value: str) -> Path:
        """Resolve a relative path against the config file's directory."""
        p = Path(value)
        if p.is_absolute():
            return p
        if self.config_path:
            return (Path(self.config_path).parent / p).resolve()
        return p.resolve()

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude={"config_path"}),
            sort_keys=False,
            allow_unicode=True,
        )


# --------------------------------------------------------------------------- #
# Cell — one unit of work
# --------------------------------------------------------------------------- #
class Cell(BaseModel):
    """A single (model, strategy, task, repeat) unit of work."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelSpec
    strategy: StrategySpec
    task: Any
    repeat: int = 0

    @property
    def cell_key(self) -> str:
        return (
            f"{self.model.id}|{self.strategy.id}|"
            f"{getattr(self.task, 'id', 'unknown')}|r{self.repeat}"
        )

    @property
    def task_id(self) -> str:
        return str(getattr(self.task, "id", "unknown"))

    @property
    def category(self) -> str:
        return str(getattr(self.task, "category", "general"))


__all__ = [
    "BudgetSpec",
    "Cell",
    "ExperimentConfig",
    "ExperimentSpec",
    "ModelSpec",
    "OutputSpec",
    "RetryPolicy",
    "StrategySpec",
    "SystemPromptPreset",
    "TasksetSpec",
]
