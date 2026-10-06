"""
Regression: in-stream error events (type="error", type="response.failed") must
raise instead of being returned as benign chunks, mirroring chat streaming
semantics (_handle_stream_fallback_error): non-retriable 4xx (except 429)
raise litellm.APIError directly; 429 and 5xx are wrapped in
MidStreamFallbackError so the Router's mid-stream fallback machinery fires.

Status mapping must consider both the OpenAI error `type` (e.g.
"invalid_request_error") and `code` (e.g. "invalid_prompt",
"rate_limit_exceeded") fields — previously only `code` was read, so
type-classified client errors fell through to 500.

Also covers: ErrorEventError.param must accept dict payloads without raising a
Pydantic ValidationError (previously typed as Optional[str]).
"""

import json
from collections.abc import Mapping
from importlib import import_module
from unittest.mock import Mock, patch

import pytest


import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    _ERROR_CODE_HTTP_STATUS,
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
    _status_code_for_error_fields,
)
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    ResponseAPIUsage,
    ResponsesAPIStreamEvents,
)


def _make_iterator() -> BaseResponsesAPIStreamingIterator:
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_response = Mock()
    mock_response.headers = {}
    return BaseResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )


def _make_error_chunk(error_type: str, code: str, message: str = "err") -> ErrorEvent:
    error_obj = ErrorEventError(type=error_type, code=code, message=message)
    return ErrorEvent(type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj)


def test_maybe_raise_for_error_event_wraps_unknown_error_in_mid_stream_fallback():
    iterator = _make_iterator()
    chunk = _make_error_chunk("server_error", "internal_error", "something went wrong")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.original_exception, litellm.APIError)
    assert exc_info.value.original_exception.status_code == 500


def test_maybe_raise_for_error_event_maps_rate_limit_code_to_429_mid_stream_fallback():
    """429 is retriable: it must be wrapped so the Router can fall back, carrying the mapped APIError."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("tokens", "rate_limit_exceeded", "Too many requests")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429
    assert exc_info.value.generated_content == ""
    assert exc_info.value.is_pre_first_chunk is True
    assert isinstance(exc_info.value.original_exception, litellm.APIError)
    assert exc_info.value.original_exception.status_code == 429


def test_maybe_raise_for_error_event_maps_invalid_request_type_to_400():
    """Client errors classified via the `type` field must raise APIError directly (no fallback)."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("invalid_request_error", "invalid_prompt", "bad request")
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_maps_context_length_code_to_400():
    """Client errors classified via the `code` field alone must still map to 400."""
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = "error"
    chunk.error = {"code": "context_length_exceeded", "message": "too long"}
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_maps_insufficient_quota_to_429():
    """OpenAI returns HTTP 429 for insufficient_quota; it must not map to 400 even though its type
    is invalid_request_error-adjacent, and it must be wrapped for fallback."""
    iterator = _make_iterator()
    chunk = _make_error_chunk("invalid_request_error", "insufficient_quota", "You exceeded your current quota")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429


def test_maybe_raise_for_error_event_passes_through_normal_chunk():
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    iterator._maybe_raise_for_error_event(chunk)  # must not raise


def test_error_event_error_param_accepts_dict():
    error_obj = ErrorEventError(
        type="invalid_request_error",
        code="context_length_exceeded",
        message="too long",
        param={"field": "messages", "index": 0},
    )
    assert isinstance(error_obj.param, dict)


def _make_async_iterator_with_events(events: list) -> ResponsesAPIStreamingIterator:
    sse_payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def mock_aiter_bytes():
        yield sse_payload

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    def transform(model, parsed_chunk, logging_obj):
        if parsed_chunk.get("type") == "error":
            return ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=0,
                error=ErrorEventError(**parsed_chunk["error"]),
            )
        delta_event = Mock()
        delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
        delta_event.delta = parsed_chunk.get("delta", "")
        return delta_event

    mock_config.transform_streaming_response.side_effect = transform

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_async_iterator_raises_mid_stream_fallback_on_rate_limit_error_event():
    iterator = _make_async_iterator_with_events(
        [
            {
                "type": "error",
                "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "rate limited"},
            }
        ]
    )

    with pytest.raises(MidStreamFallbackError) as exc_info:
        async for _ in iterator:
            pass
    assert exc_info.value.status_code == 429
    assert exc_info.value.is_pre_first_chunk is True
    assert exc_info.value.generated_content == ""
    assert isinstance(exc_info.value.original_exception, litellm.APIError)
    assert exc_info.value.original_exception.status_code == 429


@pytest.mark.asyncio
async def test_async_iterator_error_after_first_chunk_carries_generated_content():
    """An error after streamed output must expose the accumulated text so the router's
    fallback can build a continuation input instead of restarting from scratch."""
    iterator = _make_async_iterator_with_events(
        [
            {"type": "response.output_text.delta", "delta": "hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {
                "type": "error",
                "error": {"type": "server_error", "code": "internal_error", "message": "boom"},
            },
        ]
    )

    chunks = []
    async def _drain():
        async for chunk in iterator:
            chunks.append(chunk)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        await _drain()
    assert len(chunks) == 2
    assert exc_info.value.status_code == 500
    assert exc_info.value.is_pre_first_chunk is False
    assert exc_info.value.generated_content == "hello world"


def test_maybe_raise_for_response_failed_event_with_dict_error():
    """response.failed chunks carry a dict error on .response.error; covers dict branch."""
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 429


def test_maybe_raise_for_error_event_null_error_obj():
    """error chunk with no error field: message and code default; wrapped as 500."""
    iterator = _make_iterator()
    chunk = Mock()
    chunk.type = "error"
    chunk.error = None
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 500
    assert "Response API in-stream error" in str(exc_info.value)


def _make_failed_chunk(error: dict, usage: ResponseAPIUsage | None = None) -> Mock:
    mock_response_obj = Mock()
    mock_response_obj.error = error
    mock_response_obj.usage = usage
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    return chunk


def test_handle_logging_failed_response_maps_rate_limit_to_429():
    """The exception logged to failure handlers must carry the mapped status, not a hardcoded 500."""
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function") as mock_run_async,
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    logged_exception = mock_run_async.call_args.kwargs["exception"]
    assert isinstance(logged_exception, litellm.APIError)
    assert logged_exception.status_code == 429
    assert "throttled" in str(logged_exception)


def test_handle_logging_failed_response_maps_type_field_to_400():
    """Status derivation for failed-response logging must also read the error `type` field."""
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "invalid_request_error", "code": "invalid_prompt", "message": "bad prompt"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function") as mock_run_async,
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    logged_exception = mock_run_async.call_args.kwargs["exception"]
    assert isinstance(logged_exception, litellm.APIError)
    assert logged_exception.status_code == 400


def test_handle_logging_failed_response_records_usage_and_cost():
    """Usage on a response.failed event must reach failure spend accounting via combined_usage_object."""
    iterator = _make_iterator()
    usage = ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    chunk = _make_failed_chunk(
        {"type": "server_error", "code": "server_error", "message": "boom"},
        usage=usage,
    )
    iterator.completed_response = chunk
    iterator.logging_obj._response_cost_calculator.return_value = 0.0042
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function"),
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    combined_usage = iterator.logging_obj.model_call_details["combined_usage_object"]
    assert isinstance(combined_usage, litellm.Usage)
    assert combined_usage.prompt_tokens == 10
    assert combined_usage.completion_tokens == 5
    assert combined_usage.total_tokens == 15
    assert iterator.logging_obj.model_call_details["response_cost"] == 0.0042
    iterator.logging_obj._response_cost_calculator.assert_called_once_with(result=chunk.response)


def test_handle_logging_failed_response_without_usage_skips_recording():
    iterator = _make_iterator()
    iterator.completed_response = _make_failed_chunk(
        {"type": "server_error", "code": "server_error", "message": "boom"}
    )
    with (
        patch.object(import_module("litellm.responses.streaming_iterator"), "run_async_function"),
        patch.object(import_module("litellm.responses.streaming_iterator"), "executor"),
    ):
        iterator._handle_logging_failed_response()
    assert "combined_usage_object" not in iterator.logging_obj.model_call_details
    iterator.logging_obj._response_cost_calculator.assert_not_called()


def test_sync_iterator_raises_mid_stream_fallback_on_rate_limit_error_event():
    """SyncResponsesAPIStreamingIterator must wrap retriable error events for fallback."""
    error_payload = {
        "type": "error",
        "error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "throttled"},
    }
    sse_bytes = f"data: {json.dumps(error_payload)}\n\n".encode()

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = iter([sse_bytes])
    mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    mock_logging_obj.completion_start_time = None
    mock_config = Mock(spec=BaseResponsesAPIConfig)

    error_obj = ErrorEventError(type="tokens", code="rate_limit_exceeded", message="throttled")
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR, sequence_number=0, error=error_obj
    )

    iterator = SyncResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(MidStreamFallbackError) as exc_info:
        for _ in iterator:
            pass
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_exception, litellm.APIError)


def test_every_openai_sdk_response_error_code_has_explicit_status_mapping():
    from typing import get_args

    from openai.types.responses.response_error import ResponseError

    sdk_codes = set(get_args(ResponseError.model_fields["code"].annotation))
    unmapped = sdk_codes - set(_ERROR_CODE_HTTP_STATUS)
    assert unmapped == set(), (
        f"OpenAI SDK ResponseError codes missing from _ERROR_CODE_HTTP_STATUS: {sorted(unmapped)}; "
        "classify each new code with an explicit HTTP status instead of letting it default to 500"
    )


@pytest.mark.parametrize(
    "code,expected_status",
    [
        ("server_error", 500),
        ("rate_limit_exceeded", 429),
        ("insufficient_quota", 429),
        ("vector_store_timeout", 504),
        ("invalid_prompt", 400),
        ("invalid_image", 400),
        ("invalid_image_format", 400),
        ("invalid_base64_image", 400),
        ("invalid_image_url", 400),
        ("image_too_large", 400),
        ("image_too_small", 400),
        ("image_parse_error", 400),
        ("image_content_policy_violation", 400),
        ("invalid_image_mode", 400),
        ("image_file_too_large", 400),
        ("unsupported_image_media_type", 400),
        ("empty_image_file", 400),
        ("failed_to_download_image", 400),
        ("image_file_not_found", 400),
        ("totally_unknown_future_code", 500),
    ],
)
def test_status_code_for_documented_response_error_codes(code: str, expected_status: int):
    assert _status_code_for_error_fields(None, code) == expected_status


def test_specific_error_code_wins_over_generic_error_type():
    assert _status_code_for_error_fields("server_error", "invalid_image") == 400


def test_maybe_raise_for_response_failed_event_maps_image_code_to_400():
    iterator = _make_iterator()
    mock_response_obj = Mock()
    mock_response_obj.error = {"code": "image_content_policy_violation", "message": "image rejected"}
    chunk = Mock()
    chunk.type = "response.failed"
    chunk.response = mock_response_obj
    with pytest.raises(litellm.APIError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 400
    assert not isinstance(exc_info.value, MidStreamFallbackError)


def test_maybe_raise_for_error_event_maps_vector_store_timeout_to_retriable_504():
    iterator = _make_iterator()
    chunk = _make_error_chunk("server_error", "vector_store_timeout", "vector store timed out")
    with pytest.raises(MidStreamFallbackError) as exc_info:
        iterator._maybe_raise_for_error_event(chunk)
    assert exc_info.value.status_code == 504


def _typeless_upstream_proxy_error_frame(code: int | str) -> Mapping[str, object]:
    return {"error": {"message": "Upstream response body was incomplete", "type": "None", "param": None, "code": code}}


@pytest.mark.parametrize(
    "error_event,expected_status,expected_message",
    [
        (_typeless_upstream_proxy_error_frame(429), 429, "Upstream response body was incomplete"),
        (_typeless_upstream_proxy_error_frame("503"), 503, "Upstream response body was incomplete"),
        (_typeless_upstream_proxy_error_frame("rate_limit_exceeded"), 429, "Upstream response body was incomplete"),
        (
            {
                "type": "error",
                "sequence_number": 1,
                "code": "rate_limit_exceeded",
                "message": "Slow down",
                "param": None,
            },
            429,
            "Slow down",
        ),
    ],
    ids=["integer-code", "numeric-string-code", "named-code", "openai-top-level-error-event"],
)
def test_native_stream_raises_on_upstream_error_event(
    error_event: Mapping[str, object], expected_status: int, expected_message: str
):
    """A native /v1/responses stream must raise an upstream in-stream error with the upstream status and message.
    Covers the typeless {"error": {...}} frame of an upstream LiteLLM proxy and OpenAI's typed error event, which
    carries code and message at the top level instead of in a nested error object"""
    import httpx

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    events = (
        {"type": "response.created", "sequence_number": 0, "response": {"id": "resp_1", "output": []}},
        error_event,
    )
    sse_body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"}, request=request)

    stream = litellm.responses(
        model="openai/gpt-5.5",
        input="hi",
        stream=True,
        api_key="fake-api-key",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
    )
    with pytest.raises(MidStreamFallbackError) as exc_info:
        for _ in stream:
            pass
    assert exc_info.value.status_code == expected_status
    assert expected_message in str(exc_info.value)
