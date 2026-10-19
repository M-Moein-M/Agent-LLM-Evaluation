"""LiteLLM proxy callback: keep reasoning_content attached to assistant
tool-call turns for reasoning models that require it (Kimi K2.x's kimi_k25
parser opens every turn already inside a thinking block, reasoning_at_start
=True, and loses coherence if the previous turn's real reasoning is absent).

Two failure paths this fixes, both converging on the same symptom
(model loops emitting only reasoning, never closes into content/tool_calls,
client sees APIEmptyResponseError):

1. Claude Code / Anthropic bridge (:30000/v1/messages): litellm translates
   Claude's echoed `thinking` blocks into an internal `thinking_blocks` key,
   but the generic openai/ provider transform (trtllm, sglang targets) never
   renames that back to `reasoning_content` before forwarding downstream ->
   the field is silently dropped even though the client echoed it correctly.

2. Native OpenAI clients (kimi code agent, etc.): never echo
   reasoning_content in replayed history at all -> nothing to preserve, the
   field is simply absent at the source.

Fix: on every response, cache the real reasoning_content the backend
produced, keyed by tool_call_id. On every inbound request, for assistant
messages without reasoning_content, recover it from thinking_blocks
(case 1). For tool-call turns without thinking_blocks, recover it from the
cache (case 2), or fall back to a non-empty placeholder so the field is at
least present.

The restore step runs in two hooks. async_pre_call_hook sees the request as
the client sent it, which is enough for /v1/chat/completions. A /v1/messages
request is still in Anthropic shape there (thinking and tool_use are content
blocks, the message has no tool_calls or thinking_blocks key), so the step
found nothing and Claude Code's echoed thinking never reached the backend.
async_pre_call_deployment_hook runs inside litellm.acompletion, after the
Anthropic->OpenAI translation, and repeats the step on the translated
messages.
"""

import time
from collections import OrderedDict
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger

_MAX_ENTRIES = 5000
_TTL_SECONDS = 6 * 3600
_PLACEHOLDER = " "


class ReasoningContentBridge(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def _remember(self, tool_call_id: str, reasoning: str) -> None:
        if not tool_call_id or not reasoning:
            return
        self._cache[tool_call_id] = (time.time(), reasoning)
        self._cache.move_to_end(tool_call_id)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)

    def _recall(self, tool_call_id: str) -> Optional[str]:
        entry = self._cache.get(tool_call_id)
        if entry is None:
            return None
        ts, reasoning = entry
        if time.time() - ts > _TTL_SECONDS:
            del self._cache[tool_call_id]
            return None
        return reasoning

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            choices = getattr(response_obj, "choices", None) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message is None:
                    continue
                reasoning = getattr(message, "reasoning_content", None)
                tool_calls = getattr(message, "tool_calls", None)
                if not reasoning or not tool_calls:
                    continue
                for tc in tool_calls:
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        self._remember(tc_id, reasoning)
        except Exception:
            # Logging hooks must never break the response path.
            pass

    def _restore(self, messages: Any) -> None:
        if not messages:
            return
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("reasoning_content"):
                continue

            # Case 1: litellm's Anthropic bridge left thinking text under
            # thinking_blocks instead of renaming it to reasoning_content.
            # Applies to turns with and without tool calls.
            thinking_blocks = message.get("thinking_blocks")
            if thinking_blocks:
                text = "".join(
                    b.get("thinking", "") for b in thinking_blocks if isinstance(b, dict)
                )
                if text:
                    message["reasoning_content"] = text
                    continue

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue

            # Case 2: recover from what we cached when this tool call
            # was originally produced.
            recovered = None
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    recovered = self._recall(tc_id)
                    if recovered:
                        break
            message["reasoning_content"] = recovered or _PLACEHOLDER

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        try:
            self._restore(data.get("messages"))
        except Exception:
            pass
        return data

    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: Any) -> Optional[dict]:
        # Called inside litellm.acompletion: a /v1/messages request has been
        # translated to OpenAI shape by now (see module docstring).
        try:
            self._restore(kwargs.get("messages"))
        except Exception:
            pass
        return kwargs


proxy_handler_instance = ReasoningContentBridge()
