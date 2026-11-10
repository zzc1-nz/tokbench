"""Adapter layer: the only part of tokbench that talks to a model.

Public surface:

* :class:`~tokbench.adapters.base.BaseAdapter` — the contract the runner uses.
* :class:`~tokbench.adapters.base.AdapterRequest` /
  :class:`~tokbench.adapters.base.AdapterResponse` — the request/response DTOs.
* :func:`~tokbench.adapters.registry.get_adapter` — provider id -> adapter.
* :class:`~tokbench.adapters.mock.MockAdapter` — deterministic offline adapter.

Design rule (repeated here because it is the project's central invariant):
**adapters count tokens, they never price them.**  Cost lives in
:mod:`tokbench.pricing`, so any run can be re-priced after the fact without
replaying a single call.
"""

from __future__ import annotations

from .base import AdapterRequest, AdapterResponse, BaseAdapter
from .mock import MockAdapter
from .registry import (
    ADAPTERS,
    available_providers,
    get_adapter,
    has_credentials,
    mode_notes,
    resolve_mode,
)

__all__ = [
    "ADAPTERS",
    "AdapterRequest",
    "AdapterResponse",
    "BaseAdapter",
    "MockAdapter",
    "available_providers",
    "get_adapter",
    "has_credentials",
    "mode_notes",
    "resolve_mode",
]
