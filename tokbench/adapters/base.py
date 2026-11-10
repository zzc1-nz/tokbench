"""Adapter contract — the ONLY surface the runner talks to.

Design rule: **adapters count tokens, they never price them.**  Every adapter
must return provider-reported token counts (or a clearly-labelled estimate) and
let :mod:`tokbench.pricing` apply the price book.  This keeps the cost model
auditable and re-derivable without re-running any API call.

Adding a provider = subclass :class:`BaseAdapter`, implement :meth:`complete`,
register it in ``registry.py``.  Nothing else in the codebase changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..errors import AdapterError, ErrorKind
from ..models import Usage


@dataclass(slots=True)
class AdapterRequest:
    """One fully-assembled request, ready to send."""

    messages: list[dict[str, str]]
    model: str
    max_output_tokens: int = 512
    temperature: float = 0.0
    stop: list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    #: Stable key used by the mock adapter to derive deterministic pseudo-usage
    #: and by prompt-caching adapters to route cache hints.
    cache_key: str | None = None
    #: Rough prompt-token estimate, computed before the call. Adapters may
    #: override it with a provider-reported value.
    prompt_tokens_est: int = 0


@dataclass(slots=True)
class AdapterResponse:
    """Normalised result of one successful call."""

    text: str
    usage: Usage
    finish_reason: str | None = None
    model: str = ""
    provider: str = ""
    #: True when ``usage`` came from a local estimator rather than the provider.
    usage_estimated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(abc.ABC):
    """Base class for all model adapters.

    Subclasses must implement :meth:`complete`.  They should raise
    :class:`~tokbench.errors.AdapterError` with a normalised
    :class:`~tokbench.errors.ErrorKind` for *every* failure path — the retry
    policy and the retry-overhead analysis both depend on that normalisation.
    """

    #: Registry id, e.g. ``"openai"``. Must match ``ModelSpec.provider``.
    provider: str = "base"
    #: Whether this adapter can emit real cache-token counters.
    supports_cache_accounting: bool = False
    #: Whether this adapter needs network credentials.
    requires_credentials: bool = False

    def __init__(self, spec: Any = None, **kwargs: Any) -> None:
        self.spec = spec
        self.options = kwargs

    # ------------------------------------------------------------------ api
    @abc.abstractmethod
    def complete(self, request: AdapterRequest) -> AdapterResponse:
        """Execute one call and return normalised usage. May raise AdapterError."""
        raise NotImplementedError

    def preflight(self) -> list[str]:
        """Return human-readable warnings about missing credentials/config."""
        return []

    def close(self) -> None:
        """Release any pooled connections. Idempotent.

        Deliberately **not** abstract: most adapters are stateless per call, so
        forcing every subclass to write an empty override would be noise. It is a
        hook, not a requirement.
        """
        return None

    # ------------------------------------------------------------- helpers
    @staticmethod
    def wrap_error(exc: Exception, *, kind: ErrorKind | None = None) -> AdapterError:
        """Convert an arbitrary exception into a normalised AdapterError."""
        if isinstance(exc, AdapterError):
            return exc
        return AdapterError(str(exc) or exc.__class__.__name__, kind=kind or ErrorKind.UNKNOWN)

    def __enter__(self) -> BaseAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} provider={self.provider}>"


__all__ = ["AdapterRequest", "AdapterResponse", "BaseAdapter"]
