"""OpenAI chat-completions adapter, and the shared HTTP/error-mapping layer.

Why hand-rolled ``httpx`` instead of the official SDK: the project must run with
only the packages that are actually installed (``httpx`` yes, ``openai`` /
``anthropic`` no), and a benchmark engine should not take a transitive dependency
on an SDK's retry and timeout policy — the runner owns retries so the
retry-overhead analysis stays honest.  One ``httpx.Client`` per adapter keeps
connection pooling across the thread pool.

This module also exports the provider-agnostic helpers (:func:`map_http_error`,
:func:`estimate_usage_from_request`) that the Anthropic, DeepSeek and Ollama
adapters reuse, so error classification is defined exactly once.  Every adapter
in the package therefore turns failures into the same
:class:`~tokbench.errors.ErrorKind` taxonomy, which is what the retry policy and
the retry-cost report both key off.

Cost is never computed here — adapters count tokens, :mod:`tokbench.pricing`
prices them.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import httpx

from ..errors import AdapterError, ErrorKind
from ..models import Usage
from ..prompt import estimate_messages_tokens, estimate_tokens
from .base import AdapterRequest, AdapterResponse, BaseAdapter

#: Substrings that mean "the prompt did not fit the context window" rather than
#: "the request was malformed".  Providers word this differently, so the list is
#: deliberately broad — misclassifying an overflow as a plain 400 would hide the
#: single most actionable failure in a token benchmark.
CONTEXT_OVERFLOW_MARKERS: tuple[str, ...] = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "token limit",
    "exceeds the maximum",
    "reduce the length",
)

#: Substrings that indicate a safety filter rather than a technical failure.
CONTENT_FILTER_MARKERS: tuple[str, ...] = (
    "content_filter",
    "content filter",
    "content policy",
    "safety",
)

DEFAULT_TIMEOUT_S = 60.0

#: Request-param keys that are *engine-internal* and must never be serialised into
#: a provider payload.  The runner passes grading targets (``hints``/``reference``)
#: through ``AdapterRequest.params`` so the mock adapter can simulate a competent
#: answer; a real provider would reject or ignore unknown fields, and leaking a
#: benchmark's expected answers to a vendor API would be a correctness and privacy
#: bug.  Stripping happens here, in the shared layer, so every real adapter is
#: covered by one rule.
INTERNAL_PARAM_KEYS: frozenset[str] = frozenset({"hints", "reference"})


def provider_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the params safe to send to a provider.

    See :data:`INTERNAL_PARAM_KEYS` for why this filter exists.
    """
    if not params:
        return {}
    return {k: v for k, v in params.items() if k not in INTERNAL_PARAM_KEYS}


def _truncate(text: str, limit: int = 500) -> str:
    """Keep provider error bodies readable in the call log."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def map_http_error(
    status: int,
    body: str,
    *,
    provider: str,
    model: str = "",
) -> AdapterError:
    """Normalise an HTTP failure into an :class:`~tokbench.errors.AdapterError`.

    Mapping table (documented because the retry policy is derived from it):

    ============  ==========================  ==========================
    Status        Kind                        Retried by default?
    ============  ==========================  ==========================
    429           ``RATE_LIMIT``              yes
    401 / 403     ``AUTH``                    no (a retry cannot fix a key)
    400           ``CONTEXT_OVERFLOW`` when    no
                  the body mentions a limit,
                  else ``BAD_REQUEST``
    413 / 422     ``CONTEXT_OVERFLOW`` /      no
                  ``BAD_REQUEST``
    5xx           ``SERVER_ERROR``            yes
    other         ``UNKNOWN``                 no
    ============  ==========================  ==========================

    The "no" rows are deliberate: retrying a bad request burns money without any
    chance of success, which would pollute the retry-overhead metric.
    """
    lowered = (body or "").lower()
    detail = _truncate(body)
    prefix = f"{provider} {model} -> HTTP {status}" if model else f"{provider} -> HTTP {status}"

    if status == 429:
        kind = ErrorKind.RATE_LIMIT
    elif status in (401, 403):
        kind = ErrorKind.AUTH
    elif status == 413:
        kind = ErrorKind.CONTEXT_OVERFLOW
    elif status == 400:
        if any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS):
            kind = ErrorKind.CONTEXT_OVERFLOW
        elif any(marker in lowered for marker in CONTENT_FILTER_MARKERS):
            kind = ErrorKind.CONTENT_FILTER
        else:
            kind = ErrorKind.BAD_REQUEST
    elif status == 422:
        if any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS):
            kind = ErrorKind.CONTEXT_OVERFLOW
        else:
            kind = ErrorKind.BAD_REQUEST
    elif 500 <= status < 600:
        kind = ErrorKind.SERVER_ERROR
    else:
        kind = ErrorKind.UNKNOWN
    return AdapterError(f"{prefix}: {detail}", kind=kind)


def map_transport_error(exc: Exception, *, provider: str, model: str = "") -> AdapterError:
    """Normalise a transport-layer exception into an ``AdapterError``.

    ``TimeoutException`` is separated from other transport failures because the
    two have different retry economics: a timeout may have already been billed by
    the provider, while a connection error definitely was not.
    """
    prefix = f"{provider} {model}" if model else provider
    if isinstance(exc, httpx.TimeoutException):
        return AdapterError(f"{prefix}: timeout: {exc}", kind=ErrorKind.TIMEOUT)
    if isinstance(exc, httpx.TransportError):
        return AdapterError(f"{prefix}: transport error: {exc}", kind=ErrorKind.NETWORK)
    return BaseAdapter.wrap_error(exc, kind=ErrorKind.UNKNOWN)


def estimate_usage_from_request(request: AdapterRequest, text: str) -> Usage:
    """Best-effort usage when the provider reported none.

    Used only as a last resort (and always paired with
    ``usage_estimated=True`` on the response) so a run never silently records
    zero tokens — a zero would look like a free call in the cost report.
    """
    prompt_tokens = request.prompt_tokens_est or estimate_messages_tokens(request.messages)
    return Usage(input_tokens=prompt_tokens, output_tokens=estimate_tokens(text))


def resolve_api_key(spec: Any, default_env_key: str) -> str | None:
    """Credential lookup order: explicit config, then env, then nothing.

    The explicit ``api_key`` field exists for local/proxy endpoints; the env
    variable is the normal path and keeps secrets out of the YAML.
    """
    explicit = getattr(spec, "api_key", None)
    if explicit:
        return str(explicit)
    env_key = getattr(spec, "env_key", None) or default_env_key
    value = os.environ.get(env_key, "").strip()
    return value or None


def resolve_base_url(spec: Any, default: str) -> str:
    """Base URL from the spec, else the provider default (trailing slash trimmed)."""
    value = getattr(spec, "base_url", None) or default
    return str(value).rstrip("/")


class OpenAIAdapter(BaseAdapter):
    """Chat-completions adapter for OpenAI and any OpenAI-compatible endpoint.

    Subclasses (e.g. :class:`tokbench.adapters.deepseek.DeepSeekAdapter`) only
    override ``provider``/``default_base_url``/``default_env_key`` and, when the
    provider's usage JSON differs, :meth:`parse_usage`.
    """

    provider = "openai"
    requires_credentials = True
    supports_cache_accounting = True
    default_base_url = "https://api.openai.com/v1"
    default_env_key = "OPENAI_API_KEY"

    def __init__(self, spec: Any = None, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self.base_url = resolve_base_url(spec, self.default_base_url)
        self.api_key = resolve_api_key(spec, self.default_env_key)
        params: dict[str, Any] = dict(getattr(spec, "params", {}) or {})
        self.timeout_s = float(params.pop("timeout_s", DEFAULT_TIMEOUT_S))
        #: Some models (o-series, newer GPT) reject ``max_tokens``; the knob lets
        #: a config opt into ``max_completion_tokens`` without a code change.
        self.token_param = str(params.pop("token_param", "max_tokens"))
        self.extra_body = params
        self.model_name = str(getattr(spec, "model", "") or getattr(spec, "id", "") or "")
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_s),
            headers=self._headers(),
            follow_redirects=True,
        )

    # ------------------------------------------------------------------ setup
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def preflight(self) -> list[str]:
        """Warn (do not fail) when the run cannot possibly authenticate.

        A warning instead of an exception because ``adapter_mode: auto`` is
        allowed to degrade to the mock adapter, and a live run should still
        produce a manifest explaining why nothing was called.
        """
        warnings: list[str] = []
        if not self.api_key:
            env_key = getattr(self.spec, "env_key", None) or self.default_env_key
            warnings.append(
                f"{self.provider}: no API key; set {env_key} or model.api_key "
                "(run will fail with AUTH errors)"
            )
        return warnings

    def build_payload(self, request: AdapterRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into this provider's JSON body."""
        payload: dict[str, Any] = {
            "model": request.model or self.model_name,
            "messages": request.messages,
            self.token_param: request.max_output_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = request.stop
        payload.update(self.extra_body)
        payload.update(provider_params(request.params))
        return payload

    # ------------------------------------------------------------------ call
    def complete(self, request: AdapterRequest) -> AdapterResponse:
        """Execute one chat-completion call.

        Raises:
            AdapterError: for every failure path, normalised to an
                :class:`~tokbench.errors.ErrorKind`.
        """
        payload = self.build_payload(request)
        try:
            response = self._client.post(self._endpoint(), json=payload)
        except Exception as exc:  # httpx hierarchy + anything unexpected
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
                f"{self.provider}: response was not JSON: {_truncate(response.text)}",
                kind=ErrorKind.UNKNOWN,
            ) from exc

        return self.parse_response(data, request)

    def parse_response(self, data: dict[str, Any], request: AdapterRequest) -> AdapterResponse:
        """Extract text, finish reason and *real* usage from a chat completion.

        Never invents usage: if the provider omitted the ``usage`` block, the
        local estimate is used and ``usage_estimated`` is set so the report can
        flag it.
        """
        choices = data.get("choices") or []
        if not choices:
            raise AdapterError(
                f"{self.provider}: response contained no choices", kind=ErrorKind.UNKNOWN
            )
        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = message.get("content") or ""
        finish_reason = choice.get("finish_reason")

        usage_block = data.get("usage")
        usage_estimated = False
        if isinstance(usage_block, dict) and (
            usage_block.get("prompt_tokens") is not None
            or usage_block.get("completion_tokens") is not None
        ):
            usage = self.parse_usage(usage_block)
        else:
            usage = estimate_usage_from_request(request, text)
            usage_estimated = True

        return AdapterResponse(
            text=text,
            usage=usage,
            finish_reason=finish_reason,
            model=str(data.get("model") or request.model or self.model_name),
            provider=self.provider,
            usage_estimated=usage_estimated,
            raw={"id": data.get("id", ""), "usage": usage_block or {}},
        )

    def parse_usage(self, usage_block: dict[str, Any]) -> Usage:
        """Map an OpenAI-shaped ``usage`` object onto :class:`Usage`.

        ``prompt_tokens`` already includes cached tokens, which matches the
        project convention exactly.  ``prompt_tokens_details.cached_tokens`` is
        the cache *read* subset; OpenAI does not bill cache writes separately.
        """
        prompt_details = usage_block.get("prompt_tokens_details") or {}
        completion_details = usage_block.get("completion_tokens_details") or {}
        input_tokens = int(usage_block.get("prompt_tokens") or 0)
        output_tokens = int(usage_block.get("completion_tokens") or 0)
        cached = int(prompt_details.get("cached_tokens") or 0)
        reasoning = int(completion_details.get("reasoning_tokens") or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=min(cached, input_tokens),
            cache_write_tokens=0,
            # Reasoning tokens are already part of completion_tokens in the
            # OpenAI API; keeping them as a separate counter is reporting only.
            reasoning_tokens=min(reasoning, output_tokens),
        )

    def close(self) -> None:
        """Release pooled connections (idempotent)."""
        with contextlib.suppress(Exception):  # pragma: no cover - closing must never raise
            self._client.close()


__all__ = [
    "CONTENT_FILTER_MARKERS",
    "CONTEXT_OVERFLOW_MARKERS",
    "DEFAULT_TIMEOUT_S",
    "OpenAIAdapter",
    "estimate_usage_from_request",
    "map_http_error",
    "map_transport_error",
    "resolve_api_key",
    "resolve_base_url",
]
