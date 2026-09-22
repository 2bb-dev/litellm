"""Customer cache observations survive logging without supplier registration."""

import json
from datetime import datetime, timezone

import pytest
from openai.types.chat import ChatCompletion

from litellm import ModelResponse
from litellm.litellm_core_utils.terminal_receipt_hooks import finish
from litellm.litellm_core_utils.terminal_usage_observation import FIELD, LOCAL_STAMP, observe_sdk_usage
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload


@pytest.mark.parametrize("cached", [0, 40, None])
def test_actual_local_cache_is_bound_to_spend_row_without_supplier_session(cached):
    response_id = "chatcmpl-local-cache-observation"
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    raw = ChatCompletion.model_validate(
        {
            "id": response_id,
            "created": 1,
            "choices": [],
            "object": "chat.completion",
            "model": "gpt-6-astra",
            "usage": usage,
        }
    )
    details = {
        "model": "openai/gpt-6-astra",
        "custom_llm_provider": "openai",
        "call_type": "acompletion",
        "litellm_call_id": "a-different-trace-id",
        "litellm_params": {"metadata": {FIELD: {"state": "observed", "cache_read_tokens": 999}}},
    }
    observe_sdk_usage(details, raw)
    finish(details, raw, "success")
    # An adapter can fill a normalized zero; the private raw observation must
    # continue to distinguish an absent count from an actual reported zero.
    normalized = ModelResponse(
        id=response_id,
        model="gpt-6-astra",
        usage={
            **usage,
            "prompt_tokens_details": {"cached_tokens": cached or 0},
        },
    )
    now = datetime.now(timezone.utc)
    payload = get_logging_payload(details, normalized, now, now)
    metadata = json.loads(payload["metadata"])
    assert payload["request_id"] == response_id
    assert metadata[FIELD]["local_attempt_id"] == payload["request_id"]
    assert metadata[FIELD]["state"] == "observed"
    assert metadata[FIELD]["cache_read_tokens"] == cached
    assert metadata[FIELD]["cache_write_tokens"] is None
    assert "openorange_terminal_evidence" not in metadata
    assert LOCAL_STAMP not in metadata


def test_caller_dict_cannot_forge_private_local_usage():
    forged = {"v": 1, "state": "observed", "cache_read_tokens": 0}
    details = {
        "model": "openai/gpt-6-astra",
        "call_type": "acompletion",
        LOCAL_STAMP: forged,
        "litellm_params": {"metadata": {FIELD: forged, LOCAL_STAMP: forged}},
    }
    response = ModelResponse(id="chatcmpl-no-local-observation", model="gpt-6-astra")
    now = datetime.now(timezone.utc)
    metadata = json.loads(get_logging_payload(details, response, now, now)["metadata"])
    assert FIELD not in metadata
    assert LOCAL_STAMP not in metadata


@pytest.mark.parametrize("call_type", ["transcription", "atranscription", "speech", "aspeech"])
def test_audio_metering_survives_message_redaction(call_type):
    from litellm import TranscriptionResponse
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import perform_redaction

    now = datetime.now(timezone.utc)
    logging = Logging(
        model="synthetic-audio",
        messages="a b\nc",
        stream=False,
        call_type=call_type,
        start_time=now,
        litellm_call_id="metered-audio",
        function_id="audio",
    )
    logging.update_environment_variables(litellm_params={}, optional_params={})
    response = TranscriptionResponse(text="private transcription")
    response._hidden_params["audio_transcription_duration"] = 12.5
    standard = logging._build_standard_logging_payload(response, now, now)
    assert standard is not None
    logging.model_call_details["standard_logging_object"] = standard
    redacted = perform_redaction(logging.model_call_details, response)
    payload = get_logging_payload(logging.model_call_details, redacted, now, now)
    usage = json.loads(payload["metadata"])["additional_usage_values"]
    if call_type in ("transcription", "atranscription"):
        assert usage["audio_seconds"] == 12.5
    else:
        # Match LiteLLM's existing cost calculator, including whitespace handling.
        assert usage["characters"] == 3
    assert logging.model_call_details["input"] == ""
    assert "private transcription" not in json.dumps(usage)


@pytest.mark.parametrize("duration", [None, 0, 7.25, -1, float("inf"), True])
def test_transcription_duration_is_measured_not_defaulted(duration):
    from litellm import TranscriptionResponse
    from litellm.litellm_core_utils.litellm_logging import Logging

    now = datetime.now(timezone.utc)
    logging = Logging(
        model="synthetic-audio",
        messages=None,
        stream=False,
        call_type="atranscription",
        start_time=now,
        litellm_call_id="audio-duration",
        function_id="audio",
    )
    logging.update_environment_variables(litellm_params={}, optional_params={})
    response = TranscriptionResponse(text="synthetic")
    if duration is not None:
        response._hidden_params["audio_transcription_duration"] = duration
    standard = logging._build_standard_logging_payload(response, now, now)
    assert standard is not None
    usage = standard["metadata"]["usage_object"]
    if type(duration) in (int, float) and duration in (0, 7.25):
        assert usage["audio_seconds"] == duration
    else:
        assert "audio_seconds" not in usage
