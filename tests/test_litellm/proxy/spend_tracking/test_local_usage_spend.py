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
