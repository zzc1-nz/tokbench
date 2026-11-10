"""Adapter registry — the single place that maps a provider id to an implementation.

Why a registry instead of an ``if/elif`` in the runner: adding a provider should
touch exactly one file, and the runner must stay provider-agnostic.  It also
gives ``adapter_mode`` a clean decision point — the *only* place in the engine
that decides whether a cell is executed live or against the mock.

Decision table for :func:`get_adapter`:

==================  ====================================================
mode                behaviour
==================  ====================================================
``mock``            always :class:`MockAdapter` (fully offline, default)
``live``            the provider's real adapter; a missing credential is a
                    warning recorded on the adapter, not a silent downgrade
``auto``            live when the provider is real *and* credentials exist,
                    otherwise mock with a recorded note
==================  ====================================================

``auto`` is the default because it makes the same config runnable in CI (no
keys -> mock) and on a laptop with keys (live), without editing the YAML.  The
note it records is carried into the run manifest so a report can never
accidentally present synthetic numbers as real API measurements.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from ..errors import ConfigError
from .anthropic import AnthropicAdapter
from .base import BaseAdapter
from .deepseek import DeepSeekAdapter
from .mock import MockAdapter
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter, resolve_api_key

#: Provider id -> adapter class. Keys must match ``ModelSpec.provider``.
ADAPTERS: dict[str, type[BaseAdapter]] = {
    "mock": MockAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "deepseek": DeepSeekAdapter,
    "ollama": OllamaAdapter,
}

#: Providers that need no network at all.
OFFLINE_PROVIDERS: frozenset[str] = frozenset({"mock"})

#: Environment variable consulted per provider when deciding "are credentials
#: present?" for ``adapter_mode: auto``.  Ollama has none by design.
DEFAULT_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ollama": "",
}


def available_providers() -> list[str]:
    """Sorted provider ids that have a registered adapter."""
    return sorted(ADAPTERS)


def has_credentials(spec: ModelSpec) -> bool:
    """Whether the spec can plausibly authenticate against its provider.

    Checks the explicit ``api_key`` first, then the provider's env var (or the
    spec's ``env_key`` override).  "Plausible" is the right standard here: the
    only cost of a false positive is a live attempt that fails with ``AUTH``,
    which is itself a legitimate measurement.
    """
    if spec.provider in OFFLINE_PROVIDERS:
        return True
    if spec.provider == "ollama":
        # A local server needs no credential; reachability is the adapter's
        # problem, and a dead server is a normal, retryable NETWORK failure.
        return True
    default_env = DEFAULT_ENV_KEYS.get(spec.provider, "")
    return resolve_api_key(spec, default_env) is not None


def resolve_mode(spec: ModelSpec, mode: str) -> tuple[str, str | None]:
    """Decide the effective adapter mode for one model.

    Returns ``(effective_provider, note)`` where ``note`` is a human-readable
    explanation when ``auto`` downgraded to the mock.  Split out from
    :func:`get_adapter` so the runner can surface the note in the manifest
    *without* having to introspect the returned adapter.
    """
    normalised = (mode or "auto").strip().lower()
    if normalised == "mock":
        return "mock", None
    if normalised == "live":
        if spec.provider not in ADAPTERS:
            raise ConfigError(
                f"unknown provider {spec.provider!r} for model {spec.id!r}; "
                f"known: {available_providers()}"
            )
        return spec.provider, None
    if normalised != "auto":
        raise ConfigError(f"unknown adapter_mode {mode!r}; expected one of mock, live, auto")

    if spec.provider in OFFLINE_PROVIDERS:
        return "mock", None
    if spec.provider not in ADAPTERS:
        return "mock", (f"model {spec.id!r}: unknown provider {spec.provider!r}; using mock")
    if not has_credentials(spec):
        env_key = spec.env_key or DEFAULT_ENV_KEYS.get(spec.provider, "")
        return "mock", (
            f"model {spec.id!r}: no credentials for provider {spec.provider!r} "
            f"(set {env_key or 'model.api_key'}); using mock"
        )
    return spec.provider, None


def get_adapter(spec: ModelSpec, *, mode: str = "auto", **options: Any) -> BaseAdapter:
    """Instantiate the adapter for one model.

    Args:
        spec: the model under test.
        mode: ``"mock"``, ``"live"`` or ``"auto"`` (see the module docstring).
        **options: forwarded to the adapter constructor, e.g. ``seed=`` and
            ``fail_rate=`` for :class:`MockAdapter` — this is how the runner
            threads the experiment seed into the deterministic mock.

    Returns:
        A ready-to-use :class:`~tokbench.adapters.base.BaseAdapter`.

    Raises:
        ConfigError: for an unknown ``mode`` or, in ``live`` mode, an unknown
            provider.  ``auto`` never raises: it degrades to the mock so a
            key-less CI run still produces a complete dataset.
    """
    provider, _note = resolve_mode(spec, mode)
    adapter_cls = ADAPTERS[provider]
    if adapter_cls is MockAdapter:
        # Only the mock accepts behavioural knobs; forwarding them to a real
        # adapter would raise a TypeError far from the config that caused it.
        mock_options = {k: v for k, v in options.items() if k in ("fail_rate", "seed")}
        return MockAdapter(spec, **mock_options)
    return adapter_cls(spec, **options)


def mode_notes(config_models: list[ModelSpec], mode: str) -> list[str]:
    """Collect the ``auto`` downgrade notes for a whole model list.

    Used by the runner to seed ``RunManifest.notes`` up front, so even a run that
    crashes immediately explains why nothing was called live.
    """
    notes: list[str] = []
    for spec in config_models:
        _provider, note = resolve_mode(spec, mode)
        if note:
            notes.append(note)
    return notes


__all__ = [
    "ADAPTERS",
    "DEFAULT_ENV_KEYS",
    "OFFLINE_PROVIDERS",
    "available_providers",
    "get_adapter",
    "has_credentials",
    "mode_notes",
    "resolve_mode",
]
