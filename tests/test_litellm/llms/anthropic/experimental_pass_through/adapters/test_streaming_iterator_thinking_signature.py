"""
Regression tests for Claude extended thinking proxied through `/v1/messages`
onto an OpenAI-format upstream that is itself Anthropic-backed (for example
another LiteLLM proxy).

Such an upstream streams `thinking_blocks` deltas and then closes the thought
with one chunk carrying the *accumulated* thinking text together with the
signature. The adapter previously raised
"Both `thinking` and `signature` in a single streaming chunk isn't supported"
on that chunk, killing every streamed thinking response mid-stream, and the
synthesized `content_block_start` repeated the first thinking token.

Replaying such a response also failed: the adapter attached an empty
`cache_control` to every replayed thinking block, which Anthropic rejects
("thinking.cache_control: Extra inputs are not permitted").
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import Delta, StreamingChoices, Usage


def _chunk(delta: Delta, finish_reason: Optional[str] = None, usage: Optional[Usage] = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [StreamingChoices(finish_reason=finish_reason, index=0, delta=delta, logprobs=None)]
    chunk.usage = usage
    return chunk


def _thinking_delta(text: str, signature: Optional[str] = None) -> Delta:
    block: Dict[str, Any] = {"type": "thinking", "thinking": text}
    if signature is not None:
        block["signature"] = signature
    return Delta(
        content=None,
        role="assistant",
        reasoning_content=text or None,
        thinking_blocks=[block],
        tool_calls=None,
    )


class _AsyncList:
    def __init__(self, items: List[Any]) -> None:
        self._it = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _upstream_thinking_stream() -> List[MagicMock]:
    """What an Anthropic-backed LiteLLM upstream emits for a short thought."""
    return [
        _chunk(_thinking_delta("17")),
        _chunk(_thinking_delta(" * 23 = 391")),
        # Closing chunk: accumulated text plus signature.
        _chunk(_thinking_delta("17 * 23 = 391", signature="sig-abc")),
        _chunk(Delta(content="391", role="assistant", tool_calls=None)),
        _chunk(Delta(content=None, role="assistant", tool_calls=None), finish_reason="stop"),
        _chunk(
            Delta(content=None, role="assistant", tool_calls=None),
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        ),
    ]


@pytest.mark.asyncio
async def test_thinking_stream_with_closing_signature_chunk_survives_and_does_not_duplicate():
    wrapper = AnthropicStreamWrapper(completion_stream=_AsyncList(_upstream_thinking_stream()), model="anthropic/claude")

    events: List[Dict[str, Any]] = []
    async for raw in wrapper.async_anthropic_sse_wrapper():
        text = raw.decode() if isinstance(raw, bytes) else raw
        for line in text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

    assert not any(event.get("type") == "error" for event in events), events
    assert events[-1]["type"] == "message_stop"

    thinking_starts = [
        event
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "thinking"
    ]
    assert len(thinking_starts) == 1
    assert thinking_starts[0]["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}

    thinking_index = thinking_starts[0]["index"]
    deltas = [
        event["delta"]
        for event in events
        if event["type"] == "content_block_delta" and event["index"] == thinking_index
    ]
    thinking_text = "".join(delta["thinking"] for delta in deltas if delta["type"] == "thinking_delta")
    signatures = [delta["signature"] for delta in deltas if delta["type"] == "signature_delta"]

    # Exactly the streamed text, no repeated first token and no re-emitted accumulation.
    assert thinking_text == "17 * 23 = 391"
    assert signatures == ["sig-abc"]

    answer = "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
    )
    assert answer == "391"


def test_replayed_thinking_blocks_carry_no_cache_control_unless_sent():
    adapter = LiteLLMAnthropicMessagesAdapter()
    messages: List[Any] = [
        {"role": "user", "content": "What is 17*23?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "17 * 23 = 391", "signature": "sig-abc"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "391"},
            ],
        },
    ]

    result = adapter.translate_anthropic_messages_to_openai(messages=messages, model="anthropic/claude")

    thinking_blocks = result[1]["thinking_blocks"]
    assert thinking_blocks == [
        {"type": "thinking", "thinking": "17 * 23 = 391", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    assert all("cache_control" not in block for block in thinking_blocks)


def test_replayed_thinking_block_keeps_explicit_cache_control():
    adapter = LiteLLMAnthropicMessagesAdapter()
    messages: List[Any] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "t",
                    "signature": "s",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    result = adapter.translate_anthropic_messages_to_openai(messages=messages, model="anthropic/claude")

    assert result[0]["thinking_blocks"][0]["cache_control"] == {"type": "ephemeral"}
