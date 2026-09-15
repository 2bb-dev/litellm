"""The deliberately small plaintext analytics contract for encrypted spend rows."""

import json
import math
import re
from collections.abc import Mapping

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import SpendLogsPayload
from litellm.litellm_core_utils.credential_ownership import FIELD, safe_ownership
from litellm.proxy.spend_tracking.request_content_encryption import (
    CaptureFailure,
    ContentEnvelope,
    configured_encryptor,
    record_transform_failure,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$")
_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_SPEND_FIELDS: TypeAdapter[Mapping[str, object]] = TypeAdapter(Mapping[str, object])
_IDENTITY_FIELDS = frozenset(
    "user_api_key_team_id user_api_key_project_id user_api_key_org_id "
    "user_api_key_user_id litellm_call_id openclaw_user_id openclaw_parent_user_id "
    "openclaw_actor_id openclaw_agent_id openclaw_execution_id openclaw_bot_id openclaw_sub_agent_id "
    "openclaw_session_id openclaw_session_id_raw openclaw_sender_id "
    "openclaw_conversation_id openclaw_parent_session_id openclaw_cron_id openclaw_cron_run_id "
    "openclaw_source_session_id openclaw_target_session_id openclaw_child_session_id "
    "openclaw_subagent_session_id openclaw_conversation_topic_id openclaw_conversation_message_id "
    "openclaw_mattermost_user_id openorange_code_session_id openorange_code_project_id".split()
)
_ENUM_FIELDS = {
    "status": {"success", "failure"},
    "openclaw_actor_type": {"human", "bot", "agent", "subagent", "system", "user"},
    "openclaw_execution_type": {"direct", "subagent", "sub_agent", "heartbeat", "cron", "bot", "human"},
    "openorange_request_kind": {
        "chat",
        "human",
        "interactive",
        "direct",
        "heartbeat",
        "cron",
        "subagent",
        "bot",
        "system",
        "operator_chat",
        "webchat",
        "code",
        "embedding",
        "audio_transcription",
        "audio_speech",
    },
    "openorange_key_kind": {"agent_runtime", "external_client"},
    "openclaw_channel": {
        "webchat",
        "telegram",
        "discord",
        "slack",
        "signal",
        "whatsapp",
        "imessage",
        "matrix",
        "mattermost",
        "code",
        "cli",
        "api",
        "operator",
        "unknown",
        "cron",
        "heartbeat",
    },
    "openclaw_source_tool": {
        "sessions_send",
        "sessions_spawn",
        "subagent_announce",
        "cron",
        "heartbeat",
        "webchat",
        "code",
    },
}
_USAGE_NUMBERS = frozenset(
    "prompt_tokens completion_tokens total_tokens input_tokens output_tokens "
    "cache_read_input_tokens cache_creation_input_tokens cached_tokens cache_write_tokens "
    "cache_creation_tokens reasoning_tokens audio_tokens text_tokens image_tokens "
    "accepted_prediction_tokens rejected_prediction_tokens ephemeral_5m_input_tokens "
    "ephemeral_1h_input_tokens web_search_requests server_tool_use_tokens "
    "audio_length_seconds seconds duration_seconds audio_duration_seconds audio_seconds".split()
)
_USAGE_COUNTS = frozenset("characters character_count image_count".split())
_USAGE_CONTAINERS = frozenset(
    "prompt_tokens_details completion_tokens_details input_tokens_details output_tokens_details "
    "cache_creation cache_creation_token_details server_tool_use".split()
)
_COST_NUMBERS = frozenset(
    "input_cost cache_read_cost cache_creation_cost output_cost total_cost tool_usage_cost "
    "original_cost discount_percent discount_amount margin_percent margin_fixed_amount margin_total_amount".split()
)


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _object(value: object) -> dict[str, JsonValue]:
    try:
        parsed = _JSON_VALUE.validate_json(value) if isinstance(value, str) else _JSON_VALUE.validate_python(value)
    except ValidationError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _usage(value: object, depth: int = 0) -> dict[str, JsonValue]:
    if depth > 3:
        return {}
    result: dict[str, JsonValue] = {}
    for key, item in _object(value).items():
        if key in _USAGE_NUMBERS and isinstance(item, (float, int)) and _number(item) and item >= 0:
            result[key] = item
        elif key in _USAGE_COUNTS and isinstance(item, (float, int)) and _number(item) and item >= 0 and item % 1 == 0:
            result[key] = item
        elif key == "type" and item == "duration":
            result[key] = item
        elif key in _USAGE_CONTAINERS and isinstance(item, dict):
            result[key] = _usage(item, depth + 1)
    if "cache_write_tokens" in result and "cache_creation_tokens" not in result:
        result["cache_creation_tokens"] = result["cache_write_tokens"]
    return result


def safe_metadata(value: object) -> dict[str, JsonValue]:
    source = _object(value)
    result: dict[str, JsonValue] = {}
    if FIELD in source:
        result[FIELD] = safe_ownership(source[FIELD], _object(source[FIELD]).get("deployment_id"))
    for key, item in source.items():
        if key == "user_api_key" and isinstance(item, str) and re.fullmatch(r"[a-f0-9]{64}", item):
            result[key] = item
        elif key in _IDENTITY_FIELDS and _identifier(item) is not None:
            result[key] = item
        elif key in _ENUM_FIELDS and isinstance(item, str) and item in _ENUM_FIELDS[key]:
            result[key] = item
        elif (
            key
            in {
                "openclaw_heartbeat",
                "prompt_cache_eligible",
                "openorange_prompt_cache_eligible",
                "openclaw_actor_is_bot",
                "openclaw_mattermost_is_bot",
            }
            and type(item) is bool
        ):
            result[key] = item
        elif (
            key
            in {
                "openclaw_heartbeat",
                "prompt_cache_eligible",
                "openorange_prompt_cache_eligible",
                "openclaw_actor_is_bot",
                "openclaw_mattermost_is_bot",
            }
            and isinstance(item, str)
            and item in {"true", "false"}
        ):
            result[key] = item == "true"
        elif key in {"usage_object", "additional_usage_values"}:
            result[key] = _usage(item)
        elif key in {"attempted_retries", "max_retries", "litellm_overhead_time_ms"} and _number(item):
            result[key] = item
        elif key == "cost_breakdown":
            result[key] = {
                name: number for name, number in _object(item).items() if name in _COST_NUMBERS and _number(number)
            }
        elif key == "model_map_information":
            model_map = _object(item)
            prices = _object(model_map.get("model_map_value"))
            allowed_prices = {
                "input_cost_per_token",
                "output_cost_per_token",
                "cache_read_input_token_cost",
                "cache_creation_input_token_cost",
                "input_cost_per_second",
                "output_cost_per_second",
                "input_cost_per_image",
                "output_cost_per_image",
            }
            result[key] = {
                "model_map_value": {
                    name: number for name, number in prices.items() if name in allowed_prices and _number(number)
                }
            }
        elif key in {"user_api_key_metadata", "user_api_key_auth_metadata"}:
            result[key] = {
                name: ident
                for name, ident in _object(item).items()
                if name in {"openorange_key_kind", "managed_by", "agent_id"} and _identifier(ident) is not None
            }
    nested = source.get("spend_logs_metadata")
    if isinstance(nested, dict):
        # Do not recursively retain arbitrary metadata trees.
        result["spend_logs_metadata"] = safe_metadata(
            {key: item for key, item in nested.items() if key not in {"spend_logs_metadata", FIELD}}
        )
    return result


def _content_value(value: object) -> JsonValue:
    if isinstance(value, str):
        try:
            return _JSON_VALUE.validate_json(value)
        except ValidationError:
            return value
    return _JSON_VALUE.validate_python(value)


def protect_spend_payload(payload: SpendLogsPayload) -> SpendLogsPayload:
    """Return a new row. Neither inference objects nor the input row are mutated."""
    source = _SPEND_FIELDS.validate_python(payload)
    metadata = safe_metadata(source.get("metadata"))
    metadata[FIELD] = safe_ownership(metadata.get(FIELD), source.get("model_id"))
    content: dict[str, JsonValue] = {
        "v": 1,
        "request": _content_value(source.get("proxy_server_request")),
        "messages": _content_value(source.get("messages")),
        "response": _content_value(source.get("response")),
        "metadata": _content_value(source.get("metadata")),
        "request_tags": _content_value(source.get("request_tags")),
    }
    encryptor = configured_encryptor()
    envelope = encryptor if isinstance(encryptor, CaptureFailure) else encryptor.encrypt(payload["request_id"], content)
    facts: dict[str, JsonValue] = {}
    nested = _object(metadata.get("spend_logs_metadata"))
    for target, name in {
        "conversation_id": "openclaw_conversation_id",
        "source_session_id": "openclaw_session_id",
        "parent_session_id": "openclaw_parent_session_id",
        "cron_id": "openclaw_cron_id",
        "cron_run_id": "openclaw_cron_run_id",
    }.items():
        value = metadata.get(name) or nested.get(name)
        if value is not None:
            facts[target] = value
    request = _object(content["request"])
    facts["prompt_cache_key_present"] = bool(request.get("prompt_cache_key"))
    marker: dict[str, JsonValue] = {
        "v": 1,
        "content_status": "capture_failed" if isinstance(envelope, CaptureFailure) else "encrypted",
        "facts": facts,
    }
    if isinstance(envelope, CaptureFailure):
        marker["failure_code"] = envelope.code
    metadata["openorange_request_log"] = marker
    return _clear_content_columns(payload, metadata, envelope)


def failed_spend_payload(payload: SpendLogsPayload) -> SpendLogsPayload:
    """Emergency row that cannot retain request data, even if metadata parsing failed."""
    source = _SPEND_FIELDS.validate_python(payload)
    metadata: dict[str, JsonValue] = {"status": "failure" if source.get("status") == "failure" else "success"}
    try:
        original = _object(source.get("metadata"))
        metadata[FIELD] = safe_ownership(original.get(FIELD), source.get("model_id"))
        metadata["usage_object"] = _usage(original.get("usage_object"))
        metadata["additional_usage_values"] = _usage(original.get("additional_usage_values"))
    except Exception:
        # Top-level token and spend columns are independent of the metadata parser.
        pass
    failure = record_transform_failure()
    metadata["openorange_request_log"] = {
        "v": 1,
        "content_status": "capture_failed",
        "failure_code": failure.code,
        "facts": {},
    }
    return _clear_content_columns(payload, metadata, failure)


def _clear_content_columns(
    source: SpendLogsPayload, metadata: Mapping[str, JsonValue], envelope: ContentEnvelope | CaptureFailure
) -> SpendLogsPayload:
    return SpendLogsPayload(
        request_id=source["request_id"],
        call_type=source["call_type"],
        api_key=source["api_key"],
        spend=source["spend"],
        total_tokens=source["total_tokens"],
        prompt_tokens=source["prompt_tokens"],
        completion_tokens=source["completion_tokens"],
        startTime=source["startTime"],
        endTime=source["endTime"],
        completionStartTime=source["completionStartTime"],
        model=_identifier(source["model"]) or "",
        model_id=_identifier(source["model_id"]),
        model_group=_identifier(source["model_group"]),
        custom_llm_provider=_identifier(source["custom_llm_provider"]),
        agent_id=_identifier(source["agent_id"]),
        session_id=_identifier(source["session_id"]),
        user=_identifier(source["user"]) or "",
        team_id=_identifier(source["team_id"]),
        organization_id=_identifier(source["organization_id"]),
        end_user=_identifier(source["end_user"]),
        request_duration_ms=source["request_duration_ms"],
        cache_hit=source["cache_hit"],
        status=source["status"],
        metadata=json.dumps(metadata, separators=(",", ":")),
        messages="{}",
        response="{}",
        proxy_server_request="{}"
        if isinstance(envelope, CaptureFailure)
        else json.dumps(envelope, separators=(",", ":")),
        request_tags="[]",
        mcp_namespaced_tool_name=None,
        requester_ip_address=None,
        api_base="",
        cache_key="",
    )
