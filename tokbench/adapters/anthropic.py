"""Anthropic Messages API adapter.

Anthropic differs from OpenAI in ways that matter to a token benchmark, so it
gets its own adapter rather than a translation layer:

* the system prompt is a **top-level field**, not a message with ``role=system``;
* ``usage`` reports ``input_tokens`` *excluding* cache tokens, so the adapter must
  add them back to satisfy the project convention (``input_tokens`` includes
  cached tokens) — getting this wrong would double-count or under-count the
  prompt and silently corrupt every cost number;
* cache reads and cache writes are billed at *different* prices, which is why
  both counters are preserved separately.

Cost is still computed by :mod:`tokbench.pricing`, never here.
"""

from __future__ import annotations

import contextlib
from typing import Any

import httpx

from ..errors import AdapterError, ErrorKind
from ..models import Usage
from .base import AdapterRequest, AdapterResponse, BaseAdapter
from .openai import (
    DEFAULT_TIMEOUT_S,
    estimate_usage_from_request,
    map_http_error,
    map_transport_error,
    provider_params,
    resolve_api_key,
    resolve_base_url,
)

#: Anthropic requires ``max_tokens``; this is the API's documented ceiling.
MAX_OUTPUT_TOKENS_CEILING = 64_000

#: Bumped when the request shape changes; sent as the API version header.
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter(BaseAdapter):
    """Adapter for ``POST /v1/messages`` on the Anthropic API."""

    provider = "anthropic"
    requires_credentials = True
    supports_cache_accounting = True
    default_base_url = "https://api.anthropic.com/v1"
    default_env_key = "ANTHROPIC_API_KEY"

    def __init__(self, spec: Any = None, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self.base_url = resolve_base_url(spec, self.default_base_url)
        self.api_key = resolve_api_key(spec, self.default_env_key)
        params: dict[str, Any] = dict(getattr(spec, "params", {}) or {})
        self.timeout_s = float(params.pop("timeout_s", DEFAULT_TIMEOUT_S))
        self.api_version = str(params.pop("anthropic_version", ANTHROPIC_VERSION))
        #: Opt-in prompt caching. Off by default: caching changes the cost model,
        #: so it must be an explicit experimental decision, not a surprise.
        self.enable_cache_control = bool(params.pop("prompt_caching", False))
        self.extra_body = params
        self.model_name = str(getattr(spec, "model", "") or getattr(spec, "id", "") or "")
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_s),
            headers=self._headers(),
            follow_redirects=True,
        )

    # ------------------------------------------------------------------ setup
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.api_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _endpoint(self) -> str:
        return f"{self.base_url}/messages"

    def preflight(self) -> list[str]:
        """Warn about a missing key (see the OpenAI adapter for the rationale)."""
        warnings: list[str] = []
        if not self.api_key:
            env_key = getattr(self.spec, "env_key", None) or self.default_env_key
            warnings.append(
                f"{self.provider}: no API key; set {env_key} or model.api_key "
                "(run will fail with AUTH errors)"
            )
        return warnings

    def split_system(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Move ``role=system`` messages into the top-level ``system`` field.

        Multiple system messages are joined in order; the remaining messages keep
        their order so few-shot exemplars stay where assembly put them.
        """
        system_parts: list[str] = []
        rest: list[dict[str, str]] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(message.get("content", ""))
            else:
                rest.append(
                    {"role": message.get("role", "user"), "content": message.get("content", "")}
                )
        return ("\n\n".join(p for p in system_parts if p) or None), rest

    def build_payload(self, request: AdapterRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into a Messages API body."""
        system_text, messages = self.split_system(request.messages)
        if self.enable_cache_control and system_text:
            # Cache the stable prefix (system + few-shot) only: caching volatile
            # task text would never produce a hit and would add write cost.
            system_field: Any = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_field = system_text

        payload: dict[str, Any] = {
            "model": request.model or self.model_name,
            "messages": messages,
            "max_tokens": max(1, min(request.max_output_tokens, MAX_OUTPUT_TOKENS_CEILING)),
            "temperature": request.temperature,
        }
        if system_field:
            payload["system"] = system_field
        if request.stop:
            payload["stop_sequences"] = request.stop
        payload.update(self.extra_body)
        payload.update(provider_params(request.params))
        return payload

    # ------------------------------------------------------------------ call
    def complete(self, request: AdapterRequest) -> AdapterResponse:
        """Execute one Messages API call."""
        payload = self.build_payload(request)
        try:
            response = self._client.post(self._endpoint(), json=payload)
        except Exception as exc:
            raise map_transport_error(exc, provider=self.provider, model=request.model) from exc

        if response.status_code >= 400:
            raise map_http_error(
                response.status_code,
                response.text,
                provider=self.provider,
                model=request.model,
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise AdapterError(
                f"{self.provider}: response was not JSON",
                kind=ErrorKind.UNKNOWN,
            ) from exc
        return self.parse_response(data, request)

    def parse_response(self, data: dict[str, Any], request: AdapterRequest) -> AdapterResponse:
        """Extract text, stop reason and real usage from a Messages response.

        Anthropic returns ``content`` as a list of blocks; all ``text`` blocks are
        concatenated in order (thinking blocks are not answer text).
        """
        blocks = data.get("content") or []
        text_parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
        text = "".join(text_parts)

        usage_block = data.get("usage")
        usage_estimated = False
        if isinstance(usage_block, dict) and (
            usage_block.get("input_tokens") is not None
            or usage_block.get("output_tokens") is not None
        ):
            usage = self.parse_usage(usage_block)
        else:
            usage = estimate_usage_from_request(request, text)
            usage_estimated = True

        return AdapterResponse(
            text=text,
            usage=usage,
            finish_reason=data.get("stop_reason"),
            model=str(data.get("model") or request.model or self.model_name),
            provider=self.provider,
            usage_estimated=usage_estimated,
            raw={"id": data.get("id", ""), "usage": usage_block or {}},
        )

    def parse_usage(self, usage_block: dict[str, Any]) -> Usage:
        """Map a Messages ``usage`` object onto :class:`Usage`.

        The critical correction: Anthropic's ``input_tokens`` **excludes** cache
        reads and cache writes, while this project's convention is that
        ``input_tokens`` includes them.  The adapter therefore reports
        ``input_tokens = reported + cache_read + cache_write`` so that
        ``uncached_input_tokens`` (and hence the price calculation) is correct.
        """
        reported_input = int(usage_block.get("input_tokens") or 0)
        output_tokens = int(usage_block.get("output_tokens") or 0)
        cache_read = int(usage_block.get("cache_read_input_tokens") or 0)
        cache_write = int(usage_block.get("cache_creation_input_tokens") or 0)
        return Usage(
            input_tokens=reported_input + cache_read + cache_write,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=0,
        )

    def close(self) -> None:
        """Release pooled connections (idempotent)."""
        with contextlib.suppress(Exception):  # pragma: no cover
            self._client.close()


__all__ = ["ANTHROPIC_VERSION", "MAX_OUTPUT_TOKENS_CEILING", "AnthropicAdapter"]
