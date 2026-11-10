"""Typed error taxonomy for tokbench.

Retry policy and the retry-overhead analysis both key off :class:`ErrorKind`, so
every adapter MUST normalise its provider-specific exceptions into one of these
kinds.  Anything unknown becomes ``ErrorKind.UNKNOWN`` and is *not* retried by
default (fail fast, keep the benchmark honest).
"""

from __future__ import annotations

from enum import Enum


class ErrorKind(str, Enum):
    """Normalised failure classes shared by every adapter."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    CONTEXT_OVERFLOW = "context_overflow"
    CONTENT_FILTER = "content_filter"
    NETWORK = "network"
    BUDGET = "budget"
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        """Whether a plain retry can plausibly succeed without changing the request."""
        return self in _RETRYABLE


_RETRYABLE = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.SERVER_ERROR,
        ErrorKind.NETWORK,
    }
)


class TokbenchError(Exception):
    """Base class for all tokbench failures."""

    kind: ErrorKind = ErrorKind.UNKNOWN

    def __init__(self, message: str, *, kind: ErrorKind | None = None) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.kind.value}] {self.message}"


class ConfigError(TokbenchError):
    """Invalid experiment configuration or task set."""

    kind = ErrorKind.BAD_REQUEST


class AdapterError(TokbenchError):
    """An adapter-level failure (already normalised to an ``ErrorKind``)."""


class BudgetExceeded(TokbenchError):
    """A run-level token or cost ceiling was hit."""

    kind = ErrorKind.BUDGET


class TaskSetError(TokbenchError):
    """Task file missing / malformed / empty."""

    kind = ErrorKind.BAD_REQUEST


class ReportError(TokbenchError):
    """Report rendering failure."""

    kind = ErrorKind.UNKNOWN
