"""DeepSeek adapter — OpenAI-compatible chat completions with its own usage shape.

DeepSeek exposes the OpenAI chat-completions route at ``https://api.deepseek.com``,
so :class:`~tokbench.adapters.openai.OpenAIAdapter` already handles transport,
retries-by-classification and payload shape.  Only two things genuinely differ,
and both matter for a *token cost* benchmark:

1. **Usage fields.** DeepSeek reports ``prompt_cache_hit_tokens`` and
   ``prompt_cache_miss_tokens`` instead of OpenAI's
   ``prompt_tokens_details.cached_tokens``.  Cache hits are billed far cheaper
   than misses, so mis-parsing this would overstate cost by a large factor on
   warm prefixes.
2. **Reasoning models.** ``deepseek-reasoner`` bills reasoning tokens as output;
   the separate counter is kept for reporting but is already included in
   ``completion_tokens``.

Everything else (error mapping, timeout handling, "never invent usage") is
inherited, which is the point of the shared base: one taxonomy, one behaviour.
"""

from __future__ import annotations

from typing import Any

from ..errors import AdapterError, ErrorKind
from ..models import Usage
from .base import AdapterRequest
from .openai import OpenAIAdapter

#: Public DeepSeek endpoint. It serves the OpenAI-compatible ``/chat/completions``
#: route, hence the ``/v1`` suffix (DeepSeek accepts both with and without it).
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekAdapter(OpenAIAdapter):
    """Chat-completions adapter for the DeepSeek platform API."""

    provider = "deepseek"
    requires_credentials = True
    supports_cache_accounting = True
    default_base_url = DEEPSEEK_BASE_URL
    default_env_key = "DEEPSEEK_API_KEY"

    def parse_usage(self, usage_block: dict[str, Any]) -> Usage:
        """Map a DeepSeek ``usage`` object onto :class:`Usage`.

        DeepSeek's ``prompt_tokens`` already includes cached tokens (same
        convention as this project), so no correction is needed — unlike the
        Anthropic adapter.  ``prompt_cache_hit_tokens`` is the cache-read subset;
        cache writes are not reported or billed separately, so that counter stays
        zero rather than being guessed.
        """
        input_tokens = int(usage_block.get("prompt_tokens") or 0)
        output_tokens = int(usage_block.get("completion_tokens") or 0)
        cache_read = int(usage_block.get("prompt_cache_hit_tokens") or 0)
        completion_details = usage_block.get("completion_tokens_details") or {}
        reasoning = int(completion_details.get("reasoning_tokens") or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=min(cache_read, input_tokens),
            cache_write_tokens=0,
            reasoning_tokens=min(reasoning, output_tokens),
        )

    def parse_response(self, data: dict[str, Any], request: AdapterRequest):
        """Delegate to the OpenAI parser, then verify the answer is not empty.

        ``deepseek-reasoner`` puts its chain of thought in
        ``message.reasoning_content`` and the answer in ``message.content``; an
        empty ``content`` therefore means the model spent its whole budget
        thinking.  That is reported as an explicit ``CONTENT_FILTER``-adjacent
        failure rather than recorded as a successful zero-token answer, which
        would otherwise look like a suspiciously cheap cell.
        """
        response = super().parse_response(data, request)
        if not response.text.strip():
            choice = (data.get("choices") or [{}])[0] or {}
            message = choice.get("message") or {}
            if message.get("reasoning_content"):
                raise AdapterError(
                    f"{self.provider}: model returned reasoning but no final "
                    "answer (output budget exhausted)",
                    kind=ErrorKind.BAD_REQUEST,
                )
        return response


__all__ = ["DEEPSEEK_BASE_URL", "DeepSeekAdapter"]
