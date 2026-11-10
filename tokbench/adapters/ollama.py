"""Ollama adapter — local models via ``POST /api/chat``.

Ollama matters to this benchmark as the **zero-marginal-cost control**: local
inference has no per-token bill, so it anchors the cheap end of the cost/quality
Pareto chart.  Two consequences shape this adapter:

* **Usage is always estimated.** Ollama reports prompt/eval *counts*, but for
  older models and some endpoints they are absent or reported as durations only.
  When they are missing, the shared estimator fills in and
  ``usage_estimated=True`` is set — the report must never present an estimate as
  a provider measurement.
* **No API key.** ``requires_credentials`` is ``False`` and ``preflight`` warns
  only when the local server is not configured at all, since "no key" is the
  normal, expected state.

Error mapping is shared with the other HTTP adapters so a dead local server looks
like ``NETWORK`` (retryable) rather than an unknown failure.
"""

from __future__ import annotations

import contextlib
from typing import Any

import httpx

from ..errors import AdapterError, ErrorKind
from ..models import Usage
from ..prompt import estimate_tokens
from .base import AdapterRequest, AdapterResponse, BaseAdapter
from .openai import (
    DEFAULT_TIMEOUT_S,
    estimate_usage_from_request,
    map_http_error,
    map_transport_error,
    provider_params,
    resolve_base_url,
)

#: Default local endpoint of the Ollama daemon.
OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaAdapter(BaseAdapter):
    """Adapter for a locally running Ollama server."""

    provider = "ollama"
    requires_credentials = False
    supports_cache_accounting = False
    default_base_url = OLLAMA_BASE_URL

    def __init__(self, spec: Any = None, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self.base_url = resolve_base_url(spec, self.default_base_url)
        params: dict[str, Any] = dict(getattr(spec, "params", {}) or {})
        # Local models on CPU can be very slow; the timeout is generous and
        # configurable because a timeout here is a *measurement*, not a bug.
        self.timeout_s = float(params.pop("timeout_s", DEFAULT_TIMEOUT_S * 2))
        self.extra_body = params
        self.model_name = str(getattr(spec, "model", "") or getattr(spec, "id", "") or "")
        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s), follow_redirects=True)

    # ------------------------------------------------------------------ setup
    def _endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def preflight(self) -> list[str]:
        """Warn that a local server must be running; credentials are irrelevant."""
        warnings: list[str] = []
        if not getattr(self.spec, "base_url", None):
            warnings.append(
                f"{self.provider}: using default local endpoint {self.base_url}; "
                "start `ollama serve` or set model.base_url"
            )
        return warnings

    def build_payload(self, request: AdapterRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into an Ollama chat body."""
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_output_tokens,
        }
        if request.stop:
            options["stop"] = request.stop
        options.update(self.extra_body)
        options.update(provider_params(request.params))
        return {
            "model": request.model or self.model_name,
            "messages": request.messages,
            "stream": False,
            "options": options,
        }

    # ------------------------------------------------------------------ call
    def complete(self, request: AdapterRequest) -> AdapterResponse:
        """Execute one local chat call."""
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
                f"{self.provider}: response was not JSON", kind=ErrorKind.UNKNOWN
            ) from exc
        return self.parse_response(data, request)

    def parse_response(self, data: dict[str, Any], request: AdapterRequest) -> AdapterResponse:
        """Extract the answer and usage from an Ollama chat response.

        Ollama returns ``prompt_eval_count`` / ``eval_count`` (real token counts
        produced by its own tokenizer).  Those are used when present; the local
        estimator only fills the gap, always flagged as estimated.
        """
        message = data.get("message") or {}
        text = str(message.get("content") or "")

        prompt_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")
        usage_estimated = prompt_tokens is None or output_tokens is None
        if usage_estimated:
            usage = estimate_usage_from_request(request, text)
            # Ollama's own counts are the authoritative source when present;
            # mixing one real and one estimated counter would be misleading.
            usage = Usage(
                input_tokens=int(prompt_tokens)
                if prompt_tokens is not None
                else usage.input_tokens,
                output_tokens=int(output_tokens)
                if output_tokens is not None
                else estimate_tokens(text),
            )
        else:
            usage = Usage(
                input_tokens=int(prompt_tokens),
                output_tokens=int(output_tokens),
            )

        return AdapterResponse(
            text=text,
            usage=usage,
            finish_reason=data.get("done_reason") or ("stop" if data.get("done") else None),
            model=str(data.get("model") or request.model or self.model_name),
            provider=self.provider,
            usage_estimated=usage_estimated,
            raw={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
            },
            extra={"latency_ms": round(float(data.get("total_duration") or 0) / 1_000_000.0, 3)},
        )

    def close(self) -> None:
        """Release pooled connections (idempotent)."""
        with contextlib.suppress(Exception):  # pragma: no cover
            self._client.close()


__all__ = ["OLLAMA_BASE_URL", "OllamaAdapter"]
