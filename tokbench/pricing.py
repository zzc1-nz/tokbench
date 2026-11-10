"""Price book + cost model.

Cost is *never* hard-coded inside an adapter.  Every adapter returns token
counts; this module turns token counts into USD.  That separation is what makes
the cost model auditable — you can swap the price book and re-derive every
number in a report without re-running a single API call
(``tokbench report --prices prices.yaml``).

Price units: **USD per 1,000,000 tokens** (the way every vendor publishes it).
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError
from .models import Usage

MILLION = 1_000_000.0


class ModelPrice(BaseModel):
    """Per-model token prices in USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0.0, description="USD / 1M uncached input tokens")
    output: float = Field(ge=0.0, description="USD / 1M output tokens")
    cache_read: float | None = Field(
        default=None,
        ge=0.0,
        description="USD / 1M cached-input read tokens. Defaults to 10% of input.",
    )
    cache_write: float | None = Field(
        default=None,
        ge=0.0,
        description="USD / 1M cache-write tokens. Defaults to 125% of input.",
    )
    reasoning: float | None = Field(
        default=None,
        ge=0.0,
        description="USD / 1M reasoning tokens. Defaults to the output price.",
    )

    @model_validator(mode="after")
    def _fill_defaults(self) -> ModelPrice:
        if self.cache_read is None:
            object.__setattr__(self, "cache_read", self.input * 0.1)
        if self.cache_write is None:
            object.__setattr__(self, "cache_write", self.input * 1.25)
        if self.reasoning is None:
            object.__setattr__(self, "reasoning", self.output)
        return self


class PriceBook(BaseModel):
    """A named collection of :class:`ModelPrice` entries, keyed by model id."""

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    currency: str = "USD"
    effective_date: str = ""
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    # ---------------------------------------------------------------- lookup
    def get(self, model_id: str) -> ModelPrice:
        if model_id in self.models:
            return self.models[model_id]
        # tolerate provider-prefixed ids such as "openai/gpt-4o-mini"
        tail = model_id.split("/")[-1]
        if tail in self.models:
            return self.models[tail]
        raise ConfigError(
            f"no price entry for model {model_id!r}; "
            f"add it to the price book or pass --prices. Known: {sorted(self.models)[:8]}..."
        )

    def has(self, model_id: str) -> bool:
        return model_id in self.models or model_id.split("/")[-1] in self.models

    # ---------------------------------------------------------------- cost
    def cost(self, model_id: str, usage: Usage) -> float:
        """USD cost of one call's token usage."""
        p = self.get(model_id)
        uncached = usage.uncached_input_tokens
        cache_read = usage.cache_read_tokens
        cache_write = usage.cache_write_tokens
        # reasoning tokens are billed on top of visible output when the provider
        # reports them separately; when it does not, output_tokens already
        # includes them and this term is zero.
        billable_output = max(usage.output_tokens, 0)
        total = (
            uncached * p.input
            + cache_read * (p.cache_read or 0.0)
            + cache_write * (p.cache_write or 0.0)
            + billable_output * p.output
            + usage.reasoning_tokens * (p.reasoning or 0.0)
        )
        return total / MILLION

    def blended_input_price(self, model_id: str) -> float:
        p = self.get(model_id)
        return p.input

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriceBook:
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str) -> PriceBook:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        book = cls.from_dict(data)
        book.name = book.name or path
        return book

    def merged(self, override: PriceBook) -> PriceBook:
        models = dict(self.models)
        models.update(override.models)
        return PriceBook(
            name=override.name or self.name,
            currency=self.currency,
            effective_date=override.effective_date or self.effective_date,
            models=models,
        )


# --------------------------------------------------------------------------- #
# Bundled default price book (public list prices, 2026 snapshot).
# These are illustrative defaults for offline/mock runs — always override with
# your own negotiated prices before drawing conclusions about real spend.
# --------------------------------------------------------------------------- #
DEFAULT_PRICE_BOOK = PriceBook(
    name="tokbench-default-2026",
    effective_date="2026-01-01",
    models={
        # --- OpenAI -------------------------------------------------------
        "gpt-4o": ModelPrice(input=2.50, output=10.00),
        "gpt-4o-mini": ModelPrice(input=0.15, output=0.60),
        "gpt-4.1": ModelPrice(input=2.00, output=8.00),
        "gpt-4.1-mini": ModelPrice(input=0.40, output=1.60),
        "gpt-4.1-nano": ModelPrice(input=0.10, output=0.40),
        "o4-mini": ModelPrice(input=1.10, output=4.40),
        # --- Anthropic ----------------------------------------------------
        "claude-sonnet-4-20250514": ModelPrice(
            input=3.00, output=15.00, cache_read=0.30, cache_write=3.75
        ),
        "claude-3-5-haiku-20241022": ModelPrice(
            input=0.80, output=4.00, cache_read=0.08, cache_write=1.00
        ),
        "claude-3-5-sonnet-20241022": ModelPrice(
            input=3.00, output=15.00, cache_read=0.30, cache_write=3.75
        ),
        # --- DeepSeek -----------------------------------------------------
        "deepseek-chat": ModelPrice(input=0.27, output=1.10, cache_read=0.07, cache_write=0.27),
        "deepseek-reasoner": ModelPrice(input=0.55, output=2.19, cache_read=0.14, cache_write=0.55),
        # --- Google -------------------------------------------------------
        "gemini-2.0-flash": ModelPrice(input=0.10, output=0.40),
        "gemini-2.5-pro": ModelPrice(input=1.25, output=10.00),
        # --- Local / free -------------------------------------------------
        "ollama/llama3.1:8b": ModelPrice(input=0.0, output=0.0),
        "ollama/qwen2.5:7b": ModelPrice(input=0.0, output=0.0),
        "mock-small": ModelPrice(input=0.05, output=0.20),
        "mock-medium": ModelPrice(input=0.50, output=2.00),
        "mock-large": ModelPrice(input=3.00, output=12.00),
    },
)


def load_price_book(path: str | None) -> PriceBook:
    """Load a price book, overlaying it on the bundled defaults."""
    if not path:
        return DEFAULT_PRICE_BOOK
    override = PriceBook.from_yaml(path)
    return DEFAULT_PRICE_BOOK.merged(override)


__all__ = [
    "DEFAULT_PRICE_BOOK",
    "MILLION",
    "ModelPrice",
    "PriceBook",
    "load_price_book",
]
