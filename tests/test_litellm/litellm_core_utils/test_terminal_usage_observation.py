"""No default or estimated counts may become private provider observation."""

from uuid import uuid4

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from litellm.litellm_core_utils.credential_ownership import strip_ownership
from litellm.litellm_core_utils.terminal_receipt_client import Authority
from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP, Terminal
from litellm.litellm_core_utils.terminal_usage_observation import (
    FIELD,
    LOCAL_STAMP,
    UsageSnapshot,
    observe_native_usage,
    observe_sdk_usage,
    safe_usage,
    snapshot,
    usage_for_spend,
)
from tests.test_litellm.litellm_core_utils.test_terminal_receipt_evidence import signed_session


def native_session(provider):
    session, _, _, _ = signed_session()
    session.root.authority = Authority(session.root.authority.settings.model_copy(update={"role": "producer"}))
    session.terminal = Terminal(deployment_id="terminal-model", model="model", provider=provider)
    return session


def test_snapshot_exact_schema_missing_is_not_zero_and_cache_aliases_are_checked():
    fields = snapshot({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    assert fields.prompt_tokens == 0
    assert fields.cache_read_tokens is None
    assert fields.cache_write_tokens is None
    value = {"v": 1, "local_attempt_id": str(uuid4()), "state": "observed", **fields.model_dump()}
    assert len(value) == 10
    assert safe_usage(value) == value
    for name in tuple(value):
        missing = {key: item for key, item in value.items() if key != name}
        assert safe_usage(missing) == {"v": 1, "state": "invalid"}
    assert safe_usage({**value, "unexpected": "private"}) == {"v": 1, "state": "invalid"}
    assert safe_usage({**value, "state": "unobserved"}) == {"v": 1, "state": "invalid"}
    assert safe_usage({**value, "v": True}) == {"v": 1, "state": "invalid"}
    with pytest.raises(ValueError):
        snapshot({"cache_read_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 2}})
    assert strip_ownership({"nested": [{FIELD: value}]}) == {"nested": [{}]}


@pytest.mark.parametrize("invalid", [True, False, -1, 1.2, "0", 9007199254740992, float("nan"), float("inf")])
def test_invalid_native_quantity_is_not_coerced_or_estimated(invalid):
    with pytest.raises(ValueError):
        snapshot({"prompt_tokens": invalid})


def test_sdk_defaults_and_central_serialized_usage_are_not_terminal_authority():
    session, _, _, _ = signed_session()
    response = ChatCompletion.model_construct(
        id="synthetic", choices=[], created=1, model="model", object="chat.completion"
    )
    observe_sdk_usage({STAMP: session}, response)
    assert session.local_usage_snapshot is None
    response = ChatCompletion.model_validate(
        {
            "id": "synthetic",
            "choices": [],
            "created": 1,
            "model": "model",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
    observe_sdk_usage({STAMP: session}, response)
    assert session.local_usage_snapshot.prompt_tokens == 0
    session.usage_final = True
    fact = usage_for_spend(session)[FIELD]
    assert fact["state"] == "unobserved"
    assert all(fact[name] is None for name in UsageSnapshot.model_fields)


@pytest.mark.parametrize("cached", [None, 0, 25])
@pytest.mark.parametrize("streamed", [False, True])
def test_local_sdk_usage_needs_no_terminal_authority_and_preserves_cache_presence(cached, streamed):
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish, metadata_for_spend

    usage = {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    cls = ChatCompletionChunk if streamed else ChatCompletion
    response = cls.model_validate(
        {
            "id": "chatcmpl-local",
            "choices": [],
            "created": 1,
            "model": "gpt-6-astra",
            "object": "chat.completion.chunk" if streamed else "chat.completion",
            "usage": usage,
        }
    )
    details = {}
    observe_sdk_usage(details, response)
    finish(details, response, "success")
    result = metadata_for_spend(None, local_observation=details[LOCAL_STAMP], request_id=response.id)
    assert set(result) == {FIELD}
    fact = result[FIELD]
    assert fact["local_attempt_id"] == response.id
    assert fact["state"] == "observed"
    assert fact["prompt_tokens"] == 40
    assert fact["cache_read_tokens"] == cached
    assert fact["cache_write_tokens"] is None
    assert safe_usage(fact) == fact


@pytest.mark.parametrize("streamed", [False, True])
def test_local_native_astra_usage_does_not_require_supplier_session(streamed):
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish

    details = {"custom_llm_provider": "chatgpt"}
    body = {
        "object": "response",
        "usage": {
            "input_tokens": 40,
            "output_tokens": 3,
            "total_tokens": 43,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    observe_native_usage(
        details, {"type": "response.completed", "response": body} if streamed else body, streamed=streamed
    )
    finish(details, None, "success")
    fact = usage_for_spend(None, local_observation=details[LOCAL_STAMP], request_id="resp-local")[FIELD]
    assert fact["state"] == "observed"
    assert fact["cache_read_tokens"] == 0
    assert fact["cache_write_tokens"] is None


def test_local_observation_is_private_and_failure_does_not_become_final_zero():
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish

    forged = {"usage_snapshot": {"cache_read_tokens": 0}, "usage_final": True, "native_usage_final": True}
    assert usage_for_spend(None, local_observation=forged, request_id="chatcmpl-local") == {}
    assert strip_ownership({"nested": {LOCAL_STAMP: forged}}) == {"nested": {}}
    details = {"custom_llm_provider": "openai"}
    observe_native_usage(details, {"usage": {"prompt_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}}})
    finish(details, None, "failure")
    fact = usage_for_spend(None, local_observation=details[LOCAL_STAMP], request_id="chatcmpl-local")[FIELD]
    assert fact["state"] == "partial"
    assert fact["cache_read_tokens"] == 0
    assert fact["completion_tokens"] is None


def test_local_sdk_default_usage_is_absent_and_cannot_borrow_supplier_measurements():
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish

    details = {}
    response = ChatCompletion.model_construct(
        id="chatcmpl-local", choices=[], created=1, model="model", object="chat.completion"
    )
    observe_sdk_usage(details, response)
    finish(details, response, "success")
    assert LOCAL_STAMP not in details
    assert usage_for_spend(None, local_observation=details.get(LOCAL_STAMP), request_id=response.id) == {}

    session, _, _, _ = signed_session()
    session.usage_snapshot = UsageSnapshot(cache_read_tokens=99)
    session.usage_final = True
    observe_native_usage(details, {"usage": {"prompt_tokens": 3}})
    finish(details, None, "success")
    fact = usage_for_spend(session, local_observation=details[LOCAL_STAMP], request_id="chatcmpl-local")[FIELD]
    assert fact["state"] == "observed"
    assert fact["cache_read_tokens"] is None
    assert session.usage_snapshot.cache_read_tokens == 99


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retry_preparation_discards_previous_local_usage_without_terminal_session(asynchronous):
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish, prepare, prepare_async

    details = {}
    observe_native_usage(details, {"usage": {"prompt_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}}})
    finish(details, None, "success")
    assert (
        usage_for_spend(None, local_observation=details[LOCAL_STAMP], request_id="first")[FIELD]["state"] == "observed"
    )
    if asynchronous:
        await prepare_async({}, details)
    else:
        prepare({}, details)
    finish(details, None, "success")
    assert usage_for_spend(None, local_observation=details.get(LOCAL_STAMP), request_id="second") == {}


def test_anthropic_actual_partial_counters_merge_without_defaulting_cache_terms():
    session = native_session("anthropic")
    observe_native_usage(
        {STAMP: session},
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 1},
                }
            },
        },
        streamed=True,
    )
    assert session.usage_snapshot.prompt_tokens == 16
    assert session.usage_snapshot.completion_tokens == 0
    assert not session.native_usage_final
    observe_native_usage({STAMP: session}, {"type": "message_delta", "usage": {"output_tokens": 3}}, streamed=True)
    assert session.native_usage_final
    assert session.usage_snapshot.prompt_tokens == 16
    assert session.usage_snapshot.completion_tokens == 3
    assert session.usage_snapshot.total_tokens is None
    assert session.usage_snapshot.cache_write_5m_tokens == 1
    missing = native_session("anthropic")
    observe_native_usage({STAMP: missing}, {"usage": {"input_tokens": 9, "output_tokens": 3}})
    assert missing.usage_snapshot.prompt_tokens is None
    assert missing.usage_snapshot.cache_read_tokens is None
    assert missing.usage_snapshot.cache_write_tokens is None


@pytest.mark.parametrize("last_usage", [{}, {"input_tokens": 9}])
def test_anthropic_start_cursor_cannot_replace_absent_final_completion(last_usage):
    from litellm.litellm_core_utils.terminal_usage_evidence import terminal_snapshot

    session = native_session("anthropic")
    observe_native_usage(
        {STAMP: session},
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            },
        },
        streamed=True,
    )
    observe_native_usage({STAMP: session}, {"type": "message_delta", "usage": last_usage}, streamed=True)
    assert terminal_snapshot(session).state == "partial"
    assert terminal_snapshot(session).completion_tokens == 1
    observe_native_usage({STAMP: session}, {"type": "message_delta", "usage": {"output_tokens": 0}}, streamed=True)
    assert terminal_snapshot(session).state == "observed"
    assert terminal_snapshot(session).completion_tokens == 0


@pytest.mark.parametrize("choices", [[{"finish_reason": None}], None])
def test_deepseek_interim_usage_is_partial_until_terminal_usage_event(choices):
    from litellm.litellm_core_utils.terminal_usage_evidence import terminal_snapshot

    session = native_session("deepseek")
    body = {"usage": {"prompt_tokens": 9, "completion_tokens": 1}, "choices": choices}
    observe_native_usage({STAMP: session}, body, streamed=True)
    assert terminal_snapshot(session).state == "partial"
    observe_native_usage(
        {STAMP: session}, {**body, "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 0}}, streamed=True
    )
    assert terminal_snapshot(session).state == "observed"
    assert terminal_snapshot(session).completion_tokens == 0


@pytest.mark.parametrize("provider", ["deepseek", "chatgpt"])
def test_terminal_native_counts_keep_absent_cache_writes_null(provider):
    session = native_session(provider)
    body = {"usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12, "prompt_cache_hit_tokens": 0}}
    if provider == "chatgpt":
        body = {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 3,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                }
            },
        }
    observe_native_usage({STAMP: session}, body, streamed=provider == "chatgpt")
    assert session.usage_snapshot.prompt_tokens == 9
    assert session.usage_snapshot.cache_read_tokens == 0
    assert session.usage_snapshot.cache_write_tokens is None
    assert session.native_usage_final


@pytest.mark.parametrize(
    "extra",
    [
        {"cache_read_input_tokens": 2},
        {"prompt_tokens_details": {"cached_tokens": 2}},
        {"prompt_cache_miss_tokens": 7},
    ],
)
def test_deepseek_conflicting_native_cache_meters_cannot_authorize_observation(extra):
    from litellm.litellm_core_utils.terminal_usage_evidence import terminal_snapshot

    session = native_session("deepseek")
    observe_native_usage(
        {STAMP: session},
        {
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
                "prompt_cache_hit_tokens": 1,
                **extra,
            }
        },
    )
    assert session.usage_invalid
    assert terminal_snapshot(session).state == "unobserved"


@pytest.mark.parametrize("reason", ["aborted", "insufficient_system_resource"])
def test_deepseek_interrupted_stream_keeps_partial_measurement(reason):
    from litellm.litellm_core_utils.terminal_usage_evidence import terminal_snapshot

    session = native_session("deepseek")
    observe_native_usage(
        {STAMP: session},
        {
            "choices": [{"finish_reason": reason}],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
                "prompt_cache_hit_tokens": 1,
                "prompt_cache_miss_tokens": 8,
            },
        },
        streamed=True,
    )
    fact = terminal_snapshot(session)
    assert session.native_usage_final is False
    assert fact.state == "partial"
    assert fact.prompt_tokens == 9
    assert fact.completion_tokens == 3
    assert fact.cache_read_tokens == 1
    assert fact.cache_write_tokens is None


@pytest.mark.parametrize("provider", ["deepseek", "chatgpt"])
def test_native_mapping_does_not_promote_unestablished_generic_write_aliases(provider):
    session = native_session(provider)
    body = {
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 0,
            "input_tokens": 9,
            "output_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 7},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 7},
            "cache_creation_input_tokens": 7,
            "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 2},
        }
    }
    observe_native_usage({STAMP: session}, body)
    assert session.native_usage_final
    assert session.usage_snapshot.prompt_tokens == 9
    assert session.usage_snapshot.completion_tokens == 3
    assert session.usage_snapshot.cache_read_tokens == 0
    assert session.usage_snapshot.cache_write_tokens is None
    assert session.usage_snapshot.cache_write_5m_tokens is None
    assert session.usage_snapshot.cache_write_1h_tokens is None


@pytest.mark.parametrize("write", [0, 7])
def test_native_responses_canonical_write_measurement_preserves_actual_value(write):
    session = native_session("chatgpt")
    observe_native_usage(
        {STAMP: session},
        {
            "usage": {
                "input_tokens": 9,
                "output_tokens": 3,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": write},
                "cache_creation_input_tokens": 99,
            }
        },
    )
    assert session.native_usage_final
    assert session.usage_snapshot.cache_write_tokens == write
    assert session.usage_snapshot.cache_write_5m_tokens is None
    assert session.usage_snapshot.cache_write_1h_tokens is None


@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize(
    "reason,expected", [("stop", "observed"), ("length", "observed"), ("tool_calls", "observed"), (None, "partial")]
)
def test_proxy_final_usage_may_arrive_with_the_last_choice(sdk, reason, expected):
    from litellm.litellm_core_utils.terminal_receipt_hooks import finish

    body = {
        "id": "chatcmpl-final",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "provider/model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 3,
            "total_tokens": 103,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    details = {"custom_llm_provider": "litellm_proxy"}
    if sdk:
        observe_sdk_usage(details, ChatCompletionChunk.model_validate(body))
    else:
        observe_native_usage(details, body, streamed=True)
    finish(details, None, "success")
    fact = usage_for_spend(None, local_observation=details[LOCAL_STAMP], request_id="chatcmpl-final")[FIELD]
    assert fact["state"] == expected
    assert fact["cache_read_tokens"] == 0
    assert fact["cache_write_tokens"] is None
    finish(details, None, "failure")
    assert (
        usage_for_spend(None, local_observation=details[LOCAL_STAMP], request_id="chatcmpl-final")[FIELD]["state"]
        == "partial"
    )
