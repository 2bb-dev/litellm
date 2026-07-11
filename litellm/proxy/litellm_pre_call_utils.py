import asyncio
import copy
import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import Headers

import litellm
from litellm._uuid import uuid
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.constants import PRE_CALL_EXECUTED_GUARDRAILS_KEY
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host
from litellm.proxy._types import (
    AddTeamCallback,
    CommonProxyErrors,
    LitellmDataForBackendLLMCall,
    LitellmUserRoles,
    SpecialHeaders,
    TeamCallbackMetadata,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.callback_utils import (
    decrypt_callback_vars,
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers

# Cache special headers as a frozenset for O(1) lookup performance
_SPECIAL_HEADERS_CACHE = frozenset(v.value.lower() for v in SpecialHeaders._member_map_.values())

# Matches any header of the form x-<something>-session-id (case-insensitive).
# Excludes the two explicit litellm headers which are handled with higher priority.
_GENERIC_SESSION_ID_HEADER_RE = re.compile(r"^x-.+-session-id$", re.IGNORECASE)
_EXPLICIT_SESSION_HEADERS = frozenset({"x-litellm-trace-id", "x-litellm-session-id"})
# Session-id values must be non-empty strings of alphanumerics, hyphens, or underscores
# (covers UUIDs and most common session-id formats).
_SESSION_ID_VALUE_RE = re.compile(r"^[a-zA-Z0-9_\-]{8,}$")


def _sanitize_for_log(value: Any) -> str:
    """
    Basic log sanitization helper to reduce log-injection risk.

    Removes newline and carriage-return characters so user-controlled
    values cannot forge additional log lines when written to text logs.
    """
    try:
        text = str(value)
    except Exception:
        # Fallback to repr if str() fails for any reason
        text = repr(value)
    # Strip CR/LF characters commonly used for log injection
    return text.replace("\r", "").replace("\n", "")


from litellm.router import Router
from litellm.secret_managers.main import get_secret_bool
from litellm.types.llms.anthropic import ANTHROPIC_API_HEADERS
from litellm.types.services import ServiceTypes
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    LlmProviders,
    ProviderSpecificHeader,
    StandardLoggingUserAPIKeyMetadata,
    SupportedCacheControls,
)

service_logger_obj = ServiceLogging()  # used for tracking latency on OTEL
# Bounded dedup for stale-alias warnings (FIFO eviction when over cap).
_MAX_STALE_ALIAS_WARNING_KEYS = 10_000
_STALE_TEAM_ALIAS_WARNING_KEYS: OrderedDict[str, None] = OrderedDict()
# Cache the stale alias bypass flag at module load to avoid hot-path secret lookups
_ENABLE_TEAM_STALE_ALIAS_BYPASS: Optional[bool] = None


if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any
    PolicyMatchContext = Any


def parse_cache_control(cache_control):
    cache_dict = {}
    directives = cache_control.split(", ")

    for directive in directives:
        if "=" in directive:
            key, value = directive.split("=")
            cache_dict[key] = value
        else:
            cache_dict[directive] = True

    return cache_dict


LITELLM_METADATA_ROUTES = (
    "batches",
    "bedrock",
    "/v1/messages",
    "responses",
    "files",
)
OPENCLAW_CONVERSATION_INFO_LABEL = "Conversation info (untrusted metadata):"
OPENCLAW_SENDER_INFO_LABEL = "Sender (untrusted metadata):"
OPENCLAW_INBOUND_CONTEXT_LABEL = "## Inbound Context (trusted metadata)"
OPENCLAW_INBOUND_SCHEMA = "openclaw.inbound_meta.v2"
OPENCLAW_SESSION_PREFIX = "openclaw"
OPENCLAW_HEARTBEAT_SESSION_ID = f"{OPENCLAW_SESSION_PREFIX}:heartbeat"
OPENCLAW_HEARTBEAT_USER_ID = f"{OPENCLAW_SESSION_PREFIX}:heartbeat"
OPENCLAW_HEARTBEAT_TAG = "heartbeat"
OPENCLAW_HUMAN_SESSION_ID = f"{OPENCLAW_SESSION_PREFIX}:human"
OPENCLAW_SUB_AGENT_TAG = "subagent"
OPENCLAW_SUBAGENT_CONTEXT_MARKER = "[Subagent Context]"
OPENCLAW_SUBAGENT_TASK_MARKER = "[Subagent Task]"
OPENCLAW_SUBAGENT_SYSTEM_HEADING = "# Subagent Context"
OPENCLAW_SUBAGENT_SPAWNED_MARKER = "You are a **subagent** spawned"
OPENCLAW_INTER_SESSION_PROMPT_PREFIX = "[Inter-session message]"
_OPENCLAW_SUBAGENT_REQUESTER_SESSION_RE = re.compile(
    r"^\s*-\s*Requester session:\s*([^\n.]+)\.?\s*$",
    re.MULTILINE,
)
_OPENCLAW_SUBAGENT_CHILD_SESSION_RE = re.compile(
    r"^\s*-\s*Your session:\s*([^\n.]+)\.?\s*$",
    re.MULTILINE,
)
_OPENCLAW_INTER_SESSION_PREFIX_RE = re.compile(
    r"\[Inter-session message\](?P<header>[^\n]*)",
)
_OPENCLAW_INTER_SESSION_FIELD_RE = re.compile(r"(?P<key>sourceSession|sourceChannel|sourceTool|isUser)=(?P<value>\S+)")
_OPENCLAW_DIRECT_SESSION_IN_MESSAGE_ID_RE = re.compile(
    r"(?P<session>agent:[^:\s]+:"
    r"(?P<channel>mattermost|telegram):direct:"
    r"(?P<peer>[^:\s]+))(?::agentTurn(?::|$)|$)",
    re.IGNORECASE,
)

_UNTRUSTED_ROOT_CONTROL_FIELDS = (
    "proxy_server_request",
    "standard_logging_object",
    "secret_fields",
    "mock_response",
    "mock_tool_calls",
    "disable_global_guardrails",
    "disable_global_guardrail",
    "opted_out_global_guardrails",
    "applied_guardrails",
    "applied_policies",
    "policy_sources",
    "pillar_response_headers",
    "_guardrail_pipelines",
    "_pipeline_managed_guardrails",
    # Callback-registration fields. ``callbacks``, ``service_callback``,
    # and ``logger_fn`` are read by ``litellm.utils.function_setup`` and
    # appended to process-wide ``litellm.{input,success,failure,_async_*,
    # service}_callback`` lists / ``litellm.user_logger_fn`` — one request
    # poisons the worker for every subsequent caller.
    # ``litellm_disabled_callbacks`` is the inverse primitive: the
    # legitimate path reads it from key/team metadata, the request-body
    # version silently turns off admin-configured audit/observability
    # for the caller's request.
    "callbacks",
    "service_callback",
    "logger_fn",
    "litellm_disabled_callbacks",
    # Agentic-loop control fields. These bound or drive an interceptor's agentic
    # loop (web search, compression, code interpreter) and are server-controlled.
    # A client-supplied value would forge loop depth/cycle state, mark an
    # interception as active (triggering sandbox code execution without the
    # native tool ever being present), force the completed response to be
    # re-wrapped as a synthetic stream the caller never asked for, or raise the
    # loop ceiling to drive many upstream model calls and sandbox executions
    # from a single request.
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_converted_stream",
    "_code_interpreter_interception_sandbox_key",
    "max_agentic_loops",
)

_UNTRUSTED_METADATA_CONTROL_FIELDS = (
    "disable_global_guardrails",
    "disable_global_guardrail",
    "opted_out_global_guardrails",
    "pillar_response_headers",
    "_pillar_response_headers_trusted",
    "pillar_flagged",
    "pillar_scanners",
    "pillar_evidence",
    "pillar_evidence_truncated",
    "pillar_session_id_response",
    "applied_guardrails",
    "applied_policies",
    "policy_sources",
    "standard_logging_object",
    "proxy_server_request",
    "secret_fields",
    "_guardrail_pipelines",
    "_pipeline_managed_guardrails",
    "client_disconnected",
    "error_information",
    PRE_CALL_EXECUTED_GUARDRAILS_KEY,
)

_UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS = frozenset(
    {
        "litellm-disable-message-redaction",
    }
)
_CLIENT_MOCK_CONTROL_FIELDS = frozenset({"mock_response", "mock_tool_calls"})
_ALLOW_CLIENT_MOCK_RESPONSE_METADATA_KEY = "allow_client_mock_response"
_ALLOW_CLIENT_MESSAGE_REDACTION_OPT_OUT_METADATA_KEY = "allow_client_message_redaction_opt_out"

# Per-request pricing parameters mutate cost-tracking output and (via
# ``litellm.completion`` → ``register_model``) the process-wide
# ``litellm.model_cost`` map. Both effects belong to deployment configuration,
# not to user-supplied request bodies, so the proxy strips them before they
# reach the call path. Built from the Pydantic model so newly-added pricing
# fields are covered automatically.
_CLIENT_PRICING_CONTROL_FIELDS = frozenset(CustomPricingLiteLLMParams.model_fields.keys())
# ``model_info`` carries the same pricing fields when read by
# ``use_custom_pricing_for_model``; strip from metadata for the same reason.
_CLIENT_PRICING_METADATA_FIELDS = frozenset({"model_info"})
_ALLOW_CLIENT_PRICING_OVERRIDE_METADATA_KEY = "allow_client_pricing_override"

# Request fields whose value, when URL-valued, becomes the outbound destination
# for a provider call. Letting a proxy caller pin the destination is an SSRF
# primitive (HuggingFace/Oobabooga `model`, Gemini files `file_id`); guard
# them centrally so SDK users keep working but proxy users default-deny.
_URL_DESTINATION_REQUEST_FIELDS = ("model", "file_id")


def _reject_url_valued_destinations(data: Dict[str, Any]) -> None:
    """Reject URL-valued ``model``/``file_id`` unless admin-allowlisted.

    Some providers (HuggingFace, Oobabooga, Gemini files) accept a URL in the
    identifier field and use it as the outbound destination. On the proxy that
    is an SSRF primitive — a low-privilege caller can point traffic at any
    host the proxy can reach, including internal services. Reject here at the
    proxy boundary so SDK users (who legitimately pass URL-valued identifiers)
    are unaffected, while admins can opt specific hosts back in via
    ``litellm.provider_url_destination_allowed_hosts``.
    """
    allowed_hosts = getattr(litellm, "provider_url_destination_allowed_hosts", []) or []
    for field in _URL_DESTINATION_REQUEST_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        if is_url_destination_allowed_by_host(value, allowed_hosts):
            continue
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "param": field,
                "message": (
                    f"URL-valued '{field}' is not allowed. Configure custom "
                    "endpoints with api_base instead, or add the destination "
                    "host to `provider_url_destination_allowed_hosts` in "
                    "litellm_settings."
                ),
            },
        )


def _strip_untrusted_request_header_controls(
    headers: Any,
    *,
    allow_client_message_redaction_opt_out: bool = False,
) -> None:
    if not isinstance(headers, dict):
        return

    for header_name in list(headers.keys()):
        if isinstance(header_name, str) and header_name.lower() in _UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS:
            if allow_client_message_redaction_opt_out:
                continue
            headers.pop(header_name, None)


def _is_false_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no", "off"}
    return False


def _key_or_team_metadata_flag_is_true(
    user_api_key_dict: UserAPIKeyAuth,
    metadata_key: str,
) -> bool:
    for admin_metadata in (user_api_key_dict.metadata, user_api_key_dict.team_metadata):
        if isinstance(admin_metadata, dict) and admin_metadata.get(metadata_key) is True:
            return True
    return False


def _key_or_team_allows_client_mock_response(
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    return _key_or_team_metadata_flag_is_true(
        user_api_key_dict=user_api_key_dict,
        metadata_key=_ALLOW_CLIENT_MOCK_RESPONSE_METADATA_KEY,
    )


def _key_or_team_allows_client_message_redaction_opt_out(
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    return _key_or_team_metadata_flag_is_true(
        user_api_key_dict=user_api_key_dict,
        metadata_key=_ALLOW_CLIENT_MESSAGE_REDACTION_OPT_OUT_METADATA_KEY,
    )


def _key_or_team_allows_client_pricing_override(
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    return _key_or_team_metadata_flag_is_true(
        user_api_key_dict=user_api_key_dict,
        metadata_key=_ALLOW_CLIENT_PRICING_OVERRIDE_METADATA_KEY,
    )


def _strip_client_pricing_overrides(data: Dict[str, Any]) -> None:
    """Drop pricing overrides from the request body and any metadata variant.

    Skipped only when the calling key/team carries
    ``allow_client_pricing_override: True`` in its metadata. Emits a
    ``debug``-level log line naming the dropped fields so operators can
    trace why a client-supplied pricing override stopped being applied
    (otherwise the strip is invisible from the caller's perspective).
    """
    stripped: List[str] = []
    for field in _CLIENT_PRICING_CONTROL_FIELDS:
        if field in data:
            stripped.append(field)
            data.pop(field, None)
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = data.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        for field in _CLIENT_PRICING_METADATA_FIELDS:
            if field in metadata:
                stripped.append(f"{metadata_key}.{field}")
                metadata.pop(field, None)
    if stripped:
        verbose_proxy_logger.debug(
            "Stripped client-supplied pricing fields from request body: %s. "
            "Set `allow_client_pricing_override: true` on the key or team "
            "metadata to keep these values.",
            ", ".join(stripped),
        )


def _get_metadata_variable_name(request: Request) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata"
    """
    # Inline imports — auth_utils/route_checks participate in a proxy import cycle.
    from litellm.proxy.auth.auth_utils import get_request_route  # noqa: PLC0415

    path = get_request_route(request)
    if not isinstance(path, str) or not path.startswith("/"):
        url_path = getattr(request.url, "path", None)
        path = url_path if isinstance(url_path, str) else str(request.url)
    if "thread" in path or "assistant" in path:
        return "litellm_metadata"

    if any(route in path for route in LITELLM_METADATA_ROUTES):
        return "litellm_metadata"

    return "metadata"


def _extract_generic_session_id_from_headers(
    normalized: Dict[str, str],
) -> Optional[str]:
    """
    Scan a normalised (lower-cased keys) header dict for any header that looks
    like ``x-<vendor>-session-id`` and whose value is a plausible session/trace
    identifier (alphanumeric + hyphens/underscores, at least 8 chars).

    The two explicit LiteLLM headers (``x-litellm-trace-id`` /
    ``x-litellm-session-id``) are excluded here because they are handled with
    higher priority by the caller.

    Example: ``x-claude-code-session-id: e96634a3-fa28-4083-b354-55542e2dca01``
    """
    for key, value in normalized.items():
        if (
            key not in _EXPLICIT_SESSION_HEADERS
            and _GENERIC_SESSION_ID_HEADER_RE.match(key)
            and isinstance(value, str)
            and _SESSION_ID_VALUE_RE.match(value)
        ):
            return value
    return None


def _normalize_openclaw_observability_string(value: Any, max_length: Optional[int] = None) -> Optional[str]:
    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    if max_length is not None and max_length > 0 and len(normalized) > max_length:
        return normalized[:max_length]

    return normalized


def _is_openclaw_request_like_session_id(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip()
    if not normalized:
        return True
    if normalized.startswith(("resp_", "resp-")):
        return True
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        normalized,
        re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^openclaw:(?:human|heartbeat):"
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}$",
        normalized,
        re.IGNORECASE,
    ):
        return True
    return False


def _build_openclaw_bounded_identifier(
    raw_value: str,
    prefix: str,
    max_length: int = 200,
) -> str:
    if len(raw_value) <= max_length:
        return raw_value

    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:40]
    return f"{prefix}:{digest}"


def _extract_openclaw_json_block(text: str, label: str) -> Optional[dict]:
    search_start = 0
    while True:
        label_index = text.find(label, search_start)
        if label_index == -1:
            return None

        remaining = text[label_index + len(label) :].lstrip()
        if not remaining.startswith("```"):
            search_start = label_index + len(label)
            continue

        remaining = remaining[3:]
        if remaining[:4].lower() == "json":
            remaining = remaining[4:]
        remaining = remaining.lstrip()

        block_end = remaining.find("```")
        if block_end == -1:
            return None

        parsed = safe_json_loads(remaining[:block_end].strip())
        if isinstance(parsed, dict):
            return parsed

        search_start = label_index + len(label)


def _extract_openclaw_trusted_inbound_metadata(text: str) -> Optional[dict]:
    label_index = text.find(OPENCLAW_INBOUND_CONTEXT_LABEL)
    if label_index == -1:
        return None

    section = text[label_index + len(OPENCLAW_INBOUND_CONTEXT_LABEL) :]
    next_heading = section.find("\n## ")
    if next_heading != -1:
        section = section[:next_heading]

    fence_start = section.find("```")
    if fence_start == -1:
        return None

    fenced = section[fence_start + 3 :]
    if fenced[:4].lower() == "json":
        fenced = fenced[4:]
    fenced = fenced.lstrip()

    fence_end = fenced.find("```")
    if fence_end == -1:
        return None

    parsed = safe_json_loads(fenced[:fence_end].strip())
    return parsed if isinstance(parsed, dict) else None


def _iter_openclaw_candidate_texts(
    value: Any,
    seen: Optional[set[int]] = None,
    depth: int = 0,
):
    if depth > 8:
        return

    if seen is None:
        seen = set()

    if isinstance(value, str):
        if OPENCLAW_CONVERSATION_INFO_LABEL in value or OPENCLAW_SENDER_INFO_LABEL in value:
            yield value
        return

    if not isinstance(value, (dict, list)):
        return

    object_id = id(value)
    if object_id in seen:
        return
    seen.add(object_id)

    if isinstance(value, list):
        for item in value:
            yield from _iter_openclaw_candidate_texts(item, seen=seen, depth=depth + 1)
        return

    for item in value.values():
        yield from _iter_openclaw_candidate_texts(item, seen=seen, depth=depth + 1)


def _iter_openclaw_input_content_texts(data: dict, allowed_roles: Optional[set] = None):
    for payload_key in ("input", "messages"):
        payload_value = data.get(payload_key)
        if not isinstance(payload_value, list):
            continue
        for item in payload_value:
            yield from _iter_openclaw_content_item_texts(item, allowed_roles)


def _iter_openclaw_content_item_texts(item: Any, allowed_roles: Optional[set] = None):
    if not isinstance(item, dict):
        return
    role = item.get("role")
    if allowed_roles is not None and role not in allowed_roles:
        return
    content = item.get("content")
    if isinstance(content, str):
        yield content
        return
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            yield text
        return
    if not isinstance(content, list):
        return
    for content_item in content:
        if isinstance(content_item, str):
            yield content_item
        elif isinstance(content_item, dict):
            text = content_item.get("text")
            if isinstance(text, str):
                yield text


def _infer_openclaw_channel(group_channel: Optional[str]) -> Optional[str]:
    normalized_group_channel = _normalize_openclaw_observability_string(group_channel, max_length=160)
    if normalized_group_channel is None:
        return None

    if ":" in normalized_group_channel:
        channel = normalized_group_channel.split(":", 1)[0].strip().lower()
        if not channel:
            return None

        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")
        if any(char not in allowed for char in channel):
            return None

        return channel

    # Mattermost shapes its DM/group channel ids as "#<hash>__<hash>"
    # (channel id starts with '#', two user ids joined by '__').
    if normalized_group_channel.startswith("#") and "__" in normalized_group_channel:
        head, _sep, tail = normalized_group_channel[1:].partition("__")
        if head and tail:
            return "mattermost"

    return None


def _normalize_openclaw_channel_name(value: Optional[str]) -> Optional[str]:
    normalized_value = _normalize_openclaw_observability_string(value, max_length=120)
    if normalized_value is None:
        return None

    channel = normalized_value.strip().lower()
    if not channel:
        return None

    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")
    if any(char not in allowed for char in channel):
        return None

    return channel


def _resolve_openclaw_channel(sender_info: Optional[dict], conversation_info: Optional[dict]) -> Optional[str]:
    candidate_values = [
        (conversation_info or {}).get("group_channel"),
        (conversation_info or {}).get("chat_id"),
        (sender_info or {}).get("group_channel"),
        (sender_info or {}).get("chat_id"),
        (conversation_info or {}).get("channel"),
        (sender_info or {}).get("channel"),
        (conversation_info or {}).get("provider"),
        (sender_info or {}).get("provider"),
        (conversation_info or {}).get("platform"),
        (sender_info or {}).get("platform"),
    ]

    for candidate in candidate_values:
        inferred_channel = _infer_openclaw_channel(candidate)
        if inferred_channel is not None:
            return inferred_channel

        normalized_channel = _normalize_openclaw_channel_name(candidate)
        if normalized_channel is not None:
            return normalized_channel

    return None


def _resolve_openclaw_chat_peer(
    conversation_info: Optional[dict],
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(conversation_info, dict):
        return None, None

    chat_id = _normalize_openclaw_observability_string(conversation_info.get("chat_id"), max_length=160)
    if chat_id is None or ":" not in chat_id:
        return None, None

    raw_channel, _, raw_peer_id = chat_id.partition(":")
    channel = _normalize_openclaw_channel_name(raw_channel)
    peer_id = _normalize_openclaw_observability_string(raw_peer_id, max_length=120)
    if channel is None or peer_id is None:
        return None, None

    if channel == "telegram" and peer_id.isdigit():
        return channel, peer_id
    if channel == "mattermost":
        return channel, peer_id
    return None, None


def _resolve_openclaw_direct_session_from_message_id(
    conversation_info: Optional[dict],
    inferred_channel: Optional[str],
    sender_id: Optional[str],
) -> Optional[str]:
    if not isinstance(conversation_info, dict):
        return None

    message_id = _normalize_openclaw_observability_string(conversation_info.get("message_id"), max_length=240)
    if message_id is None:
        return None

    match = _OPENCLAW_DIRECT_SESSION_IN_MESSAGE_ID_RE.search(message_id)
    if not match:
        return None

    channel = match.group("channel").lower()
    peer_id = match.group("peer")
    if inferred_channel is not None and channel != inferred_channel:
        return None
    if sender_id is not None and peer_id != sender_id:
        return None
    return match.group("session")


def _is_openclaw_heartbeat(
    data: dict,
) -> bool:
    """Detect OpenClaw heartbeats from trusted inbound metadata only."""

    channel = _resolve_openclaw_trusted_inbound_channel(data)
    return channel == "heartbeat"


def _resolve_openclaw_trusted_inbound_channel(data: dict) -> Optional[str]:
    """Resolve the OpenClaw channel from trusted inbound metadata."""

    for text in _iter_openclaw_input_content_texts(data, allowed_roles={"developer", "system"}):
        inbound_metadata = _extract_openclaw_trusted_inbound_metadata(text)
        if not isinstance(inbound_metadata, dict):
            continue
        schema = inbound_metadata.get("schema")
        channel = inbound_metadata.get("channel")
        if schema == OPENCLAW_INBOUND_SCHEMA and isinstance(channel, str):
            return _normalize_openclaw_channel_name(channel)
    return None


def _is_openclaw_subagent_request(data: dict) -> bool:
    """Detect OpenClaw subagent requests from current and legacy templates."""

    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = data.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        tags = metadata.get("tags")
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag.lower() == OPENCLAW_SUB_AGENT_TAG:
                return True

    has_context_marker = False
    has_task_marker = False
    for text in _iter_openclaw_input_content_texts(data):
        if OPENCLAW_SUBAGENT_SYSTEM_HEADING in text and OPENCLAW_SUBAGENT_SPAWNED_MARKER in text:
            return True
        if OPENCLAW_SUBAGENT_CONTEXT_MARKER in text and "running as a subagent" in text.lower():
            return True
        if OPENCLAW_SUBAGENT_CONTEXT_MARKER in text:
            has_context_marker = True
        if OPENCLAW_SUBAGENT_TASK_MARKER in text:
            has_task_marker = True
        if has_context_marker and has_task_marker:
            return True
    return False


def _extract_openclaw_inter_session_context(data: dict) -> dict:
    """Extract OpenClaw inter-session provenance from metadata or prompt prefix."""

    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = data.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        provenance = metadata.get("inputProvenance") or metadata.get("input_provenance")
        if isinstance(provenance, str):
            try:
                parsed = safe_json_loads(provenance)
            except Exception:
                parsed = None
            provenance = parsed
        if not isinstance(provenance, dict):
            continue
        if provenance.get("kind") != "inter_session":
            continue
        context: Dict[str, str] = {"kind": "inter_session"}
        source_session = _normalize_openclaw_observability_string(
            provenance.get("sourceSessionKey") or provenance.get("source_session_key"),
            max_length=200,
        )
        source_channel = _normalize_openclaw_observability_string(
            provenance.get("sourceChannel") or provenance.get("source_channel"),
            max_length=120,
        )
        source_tool = _normalize_openclaw_observability_string(
            provenance.get("sourceTool") or provenance.get("source_tool"),
            max_length=120,
        )
        if source_session is not None:
            context["source_session_id"] = source_session
        if source_channel is not None:
            context["source_channel"] = source_channel
        if source_tool is not None:
            context["source_tool"] = source_tool
        return context

    for text in _iter_openclaw_input_content_texts(data):
        prefix_match = _OPENCLAW_INTER_SESSION_PREFIX_RE.search(text)
        if not prefix_match:
            continue
        context = {"kind": "inter_session"}
        for field_match in _OPENCLAW_INTER_SESSION_FIELD_RE.finditer(prefix_match.group("header")):
            key = field_match.group("key")
            value = _normalize_openclaw_observability_string(field_match.group("value"), max_length=200)
            if value is None:
                continue
            if key == "sourceSession":
                context["source_session_id"] = value
            elif key == "sourceChannel":
                context["source_channel"] = value
            elif key == "sourceTool":
                context["source_tool"] = value
        return context

    return {}


def _is_openclaw_subagent_session_value(value: Optional[str]) -> bool:
    normalized = _normalize_openclaw_observability_string(value, max_length=240)
    if normalized is None:
        return False
    return ":subagent:" in normalized.lower()


def _resolve_openclaw_inter_session_execution_type(
    inter_session_context: dict,
    fallback_execution_type: str,
    *,
    source_session_id: Optional[str],
    target_session_id: Optional[str],
    is_subagent_request: bool,
) -> str:
    if not inter_session_context:
        return fallback_execution_type

    source_tool = inter_session_context.get("source_tool", "").strip().lower().replace("-", "_")
    if (
        is_subagent_request
        or source_tool.startswith("subagent")
        or _is_openclaw_subagent_session_value(source_session_id)
        or _is_openclaw_subagent_session_value(target_session_id)
    ):
        return "subagent"

    if fallback_execution_type in {"heartbeat", "cron", "system"}:
        return fallback_execution_type
    return "agent_to_agent"


def _extract_openclaw_subagent_session_context(data: dict) -> dict:
    """Extract parent/child session keys from OpenClaw subagent prompt text."""

    context: Dict[str, str] = {}
    for text in _iter_openclaw_input_content_texts(data):
        requester_match = _OPENCLAW_SUBAGENT_REQUESTER_SESSION_RE.search(text)
        if requester_match and "parent_session_id" not in context:
            parent_session_id = _normalize_openclaw_observability_string(requester_match.group(1), max_length=200)
            if parent_session_id is not None:
                context["parent_session_id"] = parent_session_id

        child_match = _OPENCLAW_SUBAGENT_CHILD_SESSION_RE.search(text)
        if child_match and "subagent_session_id" not in context:
            subagent_session_id = _normalize_openclaw_observability_string(child_match.group(1), max_length=200)
            if subagent_session_id is not None:
                context["subagent_session_id"] = subagent_session_id

        if "parent_session_id" in context and "subagent_session_id" in context:
            break

    return context


def _resolve_openclaw_sub_agent_suffix(sender_info: Optional[dict]) -> Optional[str]:
    """
    Return a normalized sub-agent suffix if the sender is a sub-agent.

    Checks explicit sender_info fields: `sub_agent_id`, `agent_id`, or a
    `role`/`agent_role` value of "sub_agent"/"subagent" (with an
    accompanying id). Returns None for regular senders.
    """
    if not isinstance(sender_info, dict):
        return None

    for key in ("sub_agent_id", "subagent_id"):
        suffix = _normalize_openclaw_observability_string(sender_info.get(key), max_length=120)
        if suffix is not None:
            return suffix

    role_value = sender_info.get("agent_role") or sender_info.get("role")
    if isinstance(role_value, str) and role_value.strip().lower().replace("-", "_") in {
        "sub_agent",
        "subagent",
    }:
        agent_id = _normalize_openclaw_observability_string(sender_info.get("agent_id"), max_length=120)
        if agent_id is not None:
            return agent_id

    return None


def _resolve_openclaw_user_name(sender_info: Optional[dict]) -> Optional[str]:
    """
    Pick a human-readable display name for the Langfuse user profile.

    Prefers sender_info.name, then username, then label. Mirrors the
    precedence the existing Langfuse callback expects in `metadata.user_name`.
    """
    if not isinstance(sender_info, dict):
        return None
    for key in ("name", "username", "label"):
        value = _normalize_openclaw_observability_string(sender_info.get(key), max_length=160)
        if value is not None:
            return value
    return None


def _resolve_openclaw_actor_type(
    sender_info: Optional[dict],
    sub_agent_suffix: Optional[str],
    is_heartbeat: bool,
) -> Optional[str]:
    if is_heartbeat:
        return "system"
    if sub_agent_suffix is not None:
        return "subagent"
    if not isinstance(sender_info, dict):
        return None

    for key in ("actor_type", "sender_type", "type", "role", "agent_role"):
        value = sender_info.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"human", "user"}:
            return "human"
        if normalized in {"bot", "agent", "assistant"}:
            return "bot"
        if normalized in {"subagent", "sub_agent"}:
            return "subagent"
        if normalized == "system":
            return "system"

    if sender_info.get("is_bot") is True or sender_info.get("bot") is True:
        return "bot"
    return "human"


def _resolve_openclaw_parent_user_id(
    sender_info: Optional[dict],
    conversation_info: Optional[dict],
    fallback_channel: Optional[str],
) -> Optional[str]:
    for info in (sender_info, conversation_info):
        if not isinstance(info, dict):
            continue
        for key in (
            "openclaw_parent_user_id",
            "parent_user_id",
            "parent_sender_id",
            "human_user_id",
            "root_user_id",
        ):
            value = _normalize_openclaw_observability_string(info.get(key), max_length=160)
            if value is None:
                continue
            if ":" in value:
                return _build_openclaw_bounded_identifier(value, prefix="openclaw-parent-user")
            channel = _resolve_openclaw_channel(info, conversation_info) or fallback_channel
            if channel is not None:
                return _build_openclaw_bounded_identifier(f"{channel}:{value}", prefix=f"{channel}-user")
    return None


def _resolve_openclaw_actor_id(
    sender_info: Optional[dict],
    conversation_info: Optional[dict],
    actor_type: Optional[str],
    fallback_channel: Optional[str],
    trace_user_id: Optional[str],
    bot_id: Optional[str],
) -> Optional[str]:
    for info in (sender_info, conversation_info):
        if not isinstance(info, dict):
            continue
        for key in ("openclaw_actor_id", "actor_id"):
            value = _normalize_openclaw_observability_string(info.get(key), max_length=160)
            if value is not None:
                if ":" in value:
                    return _build_openclaw_bounded_identifier(value, prefix="openclaw-actor")
                channel = _resolve_openclaw_channel(info, conversation_info) or fallback_channel
                if channel is not None:
                    return _build_openclaw_bounded_identifier(f"{channel}:{value}", prefix=f"{channel}-actor")
                return _build_openclaw_bounded_identifier(f"actor:{value}", prefix="openclaw-actor")

    actor_keys: Tuple[str, ...]
    if actor_type == "subagent":
        actor_keys = ("sub_agent_id", "subagent_id", "agent_id", "bot_id")
    elif actor_type == "bot":
        actor_keys = ("bot_id", "agent_id", "sub_agent_id", "subagent_id")
    else:
        actor_keys = ()

    for info in (sender_info, conversation_info):
        if not isinstance(info, dict):
            continue
        for key in actor_keys:
            value = _normalize_openclaw_observability_string(info.get(key), max_length=160)
            if value is not None:
                if ":" in value:
                    return _build_openclaw_bounded_identifier(value, prefix="openclaw-actor")
                prefix = "subagent" if actor_type == "subagent" else "bot"
                return _build_openclaw_bounded_identifier(f"{prefix}:{value}", prefix=f"{prefix}-actor")

    if actor_type == "bot" and bot_id is not None:
        if ":" in bot_id:
            return _build_openclaw_bounded_identifier(bot_id, prefix="openclaw-actor")
        return _build_openclaw_bounded_identifier(f"bot:{bot_id}", prefix="bot-actor")
    return trace_user_id


def _resolve_openclaw_execution_type(
    sender_info: Optional[dict],
    conversation_info: Optional[dict],
    actor_type: Optional[str],
    parent_user_id: Optional[str],
    is_heartbeat: bool,
    cron_id: Optional[str],
) -> str:
    if is_heartbeat:
        return "heartbeat"
    if cron_id is not None:
        return "cron"

    for info in (sender_info, conversation_info):
        if not isinstance(info, dict):
            continue
        value = info.get("openclaw_execution_type") or info.get("execution_type")
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            if normalized in {
                "human",
                "direct",
                "subagent",
                "agent_to_agent",
                "heartbeat",
                "cron",
                "bot_only",
                "system",
                "unknown",
            }:
                return "direct" if normalized == "human" else normalized

    if actor_type == "subagent":
        return "subagent"
    if parent_user_id is not None and actor_type in {"bot", "system"}:
        return "agent_to_agent"
    if actor_type == "human":
        return "direct"
    if actor_type == "bot":
        return "bot_only"
    return "unknown"


def _merge_openclaw_tags(existing_tags: Any, new_tags: List[str]) -> List[str]:
    """Append tags to the existing list (if any) without duplicates, preserving order."""
    merged: List[str] = []
    if isinstance(existing_tags, list):
        for tag in existing_tags:
            if isinstance(tag, str) and tag not in merged:
                merged.append(tag)
    for tag in new_tags:
        if tag and tag not in merged:
            merged.append(tag)
    return merged


def _generate_openclaw_heartbeat_session_id() -> str:
    return OPENCLAW_HEARTBEAT_SESSION_ID


def _generate_openclaw_human_session_id() -> str:
    return f"{OPENCLAW_HUMAN_SESSION_ID}:{uuid.uuid4()}"


def _first_openclaw_metadata_value(
    keys: Tuple[str, ...],
    *sources: Optional[dict],
    max_length: int = 160,
) -> Optional[str]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = _normalize_openclaw_observability_string(source.get(key), max_length=max_length)
            if value is not None:
                return value
    return None


def _has_openclaw_metadata_signal(source: Optional[dict]) -> bool:
    signal_keys = {
        "openclaw_user_id",
        "openclaw_actor_id",
        "openclaw_actor_type",
        "openclaw_channel",
        "openclaw_conversation_id",
        "openclaw_session_id",
        "openclaw_parent_session_id",
        "openclaw_sub_agent_id",
        "openclaw_subagent_session_id",
        "openclaw_heartbeat",
        "openclaw_input_provenance_kind",
        "openclaw_source_tool",
        "openclaw_source_session_id",
        "openclaw_target_session_id",
    }

    def has_signal_key(candidate: dict) -> bool:
        return any(key in candidate for key in signal_keys)

    if not isinstance(source, dict):
        return False
    if has_signal_key(source):
        return True
    spend_logs_metadata = source.get("spend_logs_metadata")
    return isinstance(spend_logs_metadata, dict) and has_signal_key(spend_logs_metadata)


def _build_openclaw_cron_metadata(
    existing_metadata: dict,
    cron_id: str,
    cron_name: Optional[str],
    cron_run_id: Optional[str],
) -> dict:
    raw_session_id = (
        f"{OPENCLAW_SESSION_PREFIX}:cron:{cron_id}:run:{cron_run_id}"
        if cron_run_id
        else f"{OPENCLAW_SESSION_PREFIX}:cron:{cron_id}"
    )
    session_id = _build_openclaw_bounded_identifier(
        raw_session_id,
        prefix=f"{OPENCLAW_SESSION_PREFIX}-session",
    )
    synthetic: Dict[str, Any] = {
        "session_id": session_id,
        "openclaw_session_id": session_id,
        "openclaw_conversation_id": session_id,
        "openclaw_session_id_raw": raw_session_id,
        "openclaw_cron_id": cron_id,
        "openclaw_cron_name": cron_name,
        "openclaw_cron_run_id": cron_run_id,
        "openclaw_execution_type": "cron",
        "tags": _merge_openclaw_tags(existing_metadata.get("tags"), ["cron"]),
    }
    merged_spend_logs_metadata = copy.deepcopy(
        existing_metadata.get("spend_logs_metadata")
        if isinstance(existing_metadata.get("spend_logs_metadata"), dict)
        else {}
    )
    for key, value in synthetic.items():
        if key.startswith("openclaw_") and value is not None:
            merged_spend_logs_metadata[key] = value
    synthetic["spend_logs_metadata"] = merged_spend_logs_metadata
    return {key: value for key, value in synthetic.items() if value is not None}


def _build_openclaw_subagent_template_metadata(data: dict, existing_metadata: dict) -> dict:
    """
    Build metadata for legacy OpenClaw subagent requests that only include the
    prompt template markers, not the newer sender/conversation JSON blocks.

    Preserve the child session as the LiteLLM/OpenClaw session when callers
    provide one. The parent conversation remains available through
    openclaw_parent_session_id so dashboards can show a combined timeline
    without erasing the per-run subagent boundary.
    """

    prompt_context = _extract_openclaw_subagent_session_context(data)
    prompt_parent_session_id = prompt_context.get("parent_session_id")
    prompt_subagent_session_id = prompt_context.get("subagent_session_id")

    metadata_conversation_id_raw = _first_openclaw_metadata_value(
        ("openclaw_conversation_id",),
        existing_metadata,
        max_length=200,
    )
    metadata_conversation_id = (
        metadata_conversation_id_raw
        if metadata_conversation_id_raw is not None
        and not _is_openclaw_request_like_session_id(metadata_conversation_id_raw)
        else None
    )
    metadata_parent_session_id_raw = _first_openclaw_metadata_value(
        ("openclaw_parent_session_id", "parent_session_id"),
        existing_metadata,
        max_length=200,
    )
    metadata_parent_session_id = (
        metadata_parent_session_id_raw
        if metadata_parent_session_id_raw is not None
        and not _is_openclaw_request_like_session_id(metadata_parent_session_id_raw)
        else None
    )
    metadata_session_id = _first_openclaw_metadata_value(
        (
            "openclaw_session_id",
            "session_id",
            "litellm_session_id",
        ),
        existing_metadata,
        max_length=200,
    )
    metadata_session_is_subagent = _is_openclaw_subagent_session_value(metadata_session_id)
    stable_metadata_session_id = (
        metadata_session_id
        if metadata_session_id is not None
        and not _is_openclaw_request_like_session_id(metadata_session_id)
        and not metadata_session_is_subagent
        else None
    )
    subagent_session_id = (
        _first_openclaw_metadata_value(
            (
                "openclaw_subagent_session_id",
                "openclaw_child_session_id",
                "child_session_id",
            ),
            existing_metadata,
            max_length=200,
        )
        or prompt_subagent_session_id
        or (metadata_session_id if metadata_session_is_subagent else None)
    )
    sub_agent_id = (
        _first_openclaw_metadata_value(
            (
                "openclaw_sub_agent_id",
                "openclaw_subagent_id",
                "sub_agent_id",
                "subagent_id",
            ),
            existing_metadata,
            max_length=120,
        )
        or subagent_session_id
    )
    subagent_session_id = subagent_session_id or (
        sub_agent_id if _is_openclaw_subagent_session_value(sub_agent_id) else None
    )
    raw_session_id = (
        subagent_session_id
        or (metadata_session_id if metadata_session_is_subagent else None)
        or stable_metadata_session_id
        or metadata_conversation_id
        or metadata_parent_session_id
        or prompt_parent_session_id
    )
    session_id = (
        _build_openclaw_bounded_identifier(
            raw_session_id,
            prefix=f"{OPENCLAW_SESSION_PREFIX}-session",
        )
        if raw_session_id is not None
        else None
    )
    parent_session_id = (
        metadata_parent_session_id
        or prompt_parent_session_id
        or (
            metadata_conversation_id
            if metadata_conversation_id is not None and metadata_conversation_id != raw_session_id
            else None
        )
        or (
            stable_metadata_session_id
            if subagent_session_id is not None
            and stable_metadata_session_id is not None
            and stable_metadata_session_id != raw_session_id
            else None
        )
        or stable_metadata_session_id
    )
    execution_id = _first_openclaw_metadata_value(
        ("openclaw_execution_id", "execution_id", "response_id", "request_id"),
        existing_metadata,
        max_length=180,
    )
    synthetic: Dict[str, Any] = {
        "session_id": session_id,
        "openclaw_session_id": session_id,
        "openclaw_conversation_id": session_id,
        "openclaw_session_id_raw": raw_session_id,
        "openclaw_parent_session_id": parent_session_id,
        "openclaw_execution_id": execution_id,
        "openclaw_execution_type": "subagent",
        "openclaw_actor_type": "subagent",
        "openclaw_sub_agent_id": sub_agent_id,
        "openclaw_subagent_session_id": subagent_session_id,
        "tags": _merge_openclaw_tags(existing_metadata.get("tags"), [OPENCLAW_SUB_AGENT_TAG]),
    }
    merged_spend_logs_metadata = copy.deepcopy(
        existing_metadata.get("spend_logs_metadata")
        if isinstance(existing_metadata.get("spend_logs_metadata"), dict)
        else {}
    )
    child_session_spend_keys = {
        "openclaw_session_id",
        "openclaw_conversation_id",
        "openclaw_session_id_raw",
        "openclaw_subagent_session_id",
        "openclaw_execution_type",
        "openclaw_actor_type",
    }
    for key, value in synthetic.items():
        if key.startswith("openclaw_") and value is not None:
            if key in child_session_spend_keys:
                merged_spend_logs_metadata[key] = value
            else:
                merged_spend_logs_metadata.setdefault(key, value)
    synthetic["spend_logs_metadata"] = merged_spend_logs_metadata
    return {key: value for key, value in synthetic.items() if value is not None}


def _build_openclaw_observability_metadata(data: dict, existing_metadata: Optional[dict]) -> Optional[dict]:
    texts_to_scan = []
    for key in ("messages", "input", "prompt", "message"):
        if key in data:
            texts_to_scan.append(data[key])

    sender_info: Optional[dict] = None
    conversation_info: Optional[dict] = None
    for candidate in texts_to_scan:
        for text in _iter_openclaw_candidate_texts(candidate):
            if sender_info is None:
                sender_info = _extract_openclaw_json_block(text, OPENCLAW_SENDER_INFO_LABEL)
            if conversation_info is None:
                conversation_info = _extract_openclaw_json_block(text, OPENCLAW_CONVERSATION_INFO_LABEL)
            if sender_info is not None and conversation_info is not None:
                break
        if sender_info is not None and conversation_info is not None:
            break

    trusted_inbound_channel = _resolve_openclaw_trusted_inbound_channel(data=data)
    is_heartbeat = trusted_inbound_channel == OPENCLAW_HEARTBEAT_TAG
    is_subagent_request = _is_openclaw_subagent_request(data=data)
    inter_session_context = _extract_openclaw_inter_session_context(data=data)
    heartbeat_session_id = _generate_openclaw_heartbeat_session_id() if is_heartbeat else None

    existing_metadata_dict: dict = existing_metadata if isinstance(existing_metadata, dict) else {}
    has_openclaw_signal = (
        sender_info is not None
        or conversation_info is not None
        or trusted_inbound_channel is not None
        or is_heartbeat
        or is_subagent_request
        or bool(inter_session_context)
        or _has_openclaw_metadata_signal(existing_metadata_dict)
    )
    generic_metadata_cron_id = _first_openclaw_metadata_value(
        ("cron_id", "schedule_id"),
        existing_metadata_dict,
        max_length=120,
    )
    generic_metadata_cron_name = _first_openclaw_metadata_value(
        ("cron_name", "schedule_name"),
        existing_metadata_dict,
        max_length=160,
    )
    generic_metadata_cron_run_id = _first_openclaw_metadata_value(
        ("cron_run_id", "schedule_run_id"),
        existing_metadata_dict,
        max_length=160,
    )
    metadata_cron_id = _first_openclaw_metadata_value(
        ("openclaw_cron_id",),
        existing_metadata_dict,
        max_length=120,
    ) or (generic_metadata_cron_id if has_openclaw_signal else None)
    metadata_cron_name = _first_openclaw_metadata_value(
        ("openclaw_cron_name",),
        existing_metadata_dict,
        max_length=160,
    ) or (generic_metadata_cron_name if has_openclaw_signal else None)
    metadata_cron_run_id = _first_openclaw_metadata_value(
        ("openclaw_cron_run_id",),
        existing_metadata_dict,
        max_length=160,
    ) or (generic_metadata_cron_run_id if has_openclaw_signal else None)

    if sender_info is None and conversation_info is None:
        if metadata_cron_id is not None:
            return _build_openclaw_cron_metadata(
                existing_metadata=existing_metadata_dict,
                cron_id=metadata_cron_id,
                cron_name=metadata_cron_name,
                cron_run_id=metadata_cron_run_id,
            )
        if (
            not is_heartbeat
            and not is_subagent_request
            and not inter_session_context
            and trusted_inbound_channel is None
        ):
            return None
        if trusted_inbound_channel is not None and not is_heartbeat:
            if is_subagent_request:
                trusted_metadata = _build_openclaw_subagent_template_metadata(
                    data,
                    existing_metadata_dict,
                )
                trusted_metadata["tags"] = _merge_openclaw_tags(
                    trusted_metadata.get("tags"),
                    [f"channel:{trusted_inbound_channel}"],
                )
                return trusted_metadata
            return {
                "tags": _merge_openclaw_tags(
                    existing_metadata_dict.get("tags"),
                    [f"channel:{trusted_inbound_channel}"],
                )
            }
        if is_subagent_request and not is_heartbeat:
            return _build_openclaw_subagent_template_metadata(data, existing_metadata_dict)
        if inter_session_context and not is_heartbeat:
            raw_source_session_id = inter_session_context.get("source_session_id")
            raw_sub_agent_id = _first_openclaw_metadata_value(
                ("openclaw_sub_agent_id",),
                existing_metadata_dict,
                max_length=200,
            )
            sub_agent_session_id = raw_sub_agent_id if _is_openclaw_subagent_session_value(raw_sub_agent_id) else None
            raw_target_session_id = (
                _first_openclaw_metadata_value(
                    (
                        "openclaw_target_session_id",
                        "openclaw_subagent_session_id",
                        "openclaw_child_session_id",
                    ),
                    existing_metadata_dict,
                    max_length=200,
                )
                or sub_agent_session_id
                or _first_openclaw_metadata_value(
                    (
                        "openclaw_session_id",
                        "session_id",
                        "litellm_session_id",
                        "openclaw_conversation_id",
                    ),
                    existing_metadata_dict,
                    max_length=200,
                )
            )
            execution_type = _resolve_openclaw_inter_session_execution_type(
                inter_session_context,
                "agent_to_agent",
                source_session_id=raw_source_session_id,
                target_session_id=raw_target_session_id,
                is_subagent_request=is_subagent_request,
            )
            raw_session_id = (
                raw_target_session_id
                if execution_type == "subagent" and raw_target_session_id is not None
                else raw_source_session_id or raw_target_session_id
            )
            session_id = (
                _build_openclaw_bounded_identifier(
                    raw_session_id,
                    prefix=f"{OPENCLAW_SESSION_PREFIX}-session",
                )
                if raw_session_id is not None
                else None
            )
            synthetic: Dict[str, Any] = {
                "session_id": session_id,
                "openclaw_session_id": session_id,
                "openclaw_conversation_id": session_id,
                "openclaw_session_id_raw": raw_session_id,
                "openclaw_parent_session_id": raw_source_session_id,
                "openclaw_source_session_id": raw_source_session_id,
                "openclaw_source_channel": inter_session_context.get("source_channel"),
                "openclaw_source_tool": inter_session_context.get("source_tool"),
                "openclaw_target_session_id": raw_target_session_id,
                "openclaw_input_provenance_kind": "inter_session",
                "openclaw_execution_type": execution_type,
                "openclaw_actor_type": ("subagent" if execution_type == "subagent" else "bot"),
                "tags": _merge_openclaw_tags(
                    existing_metadata_dict.get("tags"),
                    [
                        "inter_session",
                        (OPENCLAW_SUB_AGENT_TAG if execution_type == "subagent" else "agent_to_agent"),
                    ],
                ),
            }
            merged_spend_logs_metadata = copy.deepcopy(
                existing_metadata_dict.get("spend_logs_metadata")
                if isinstance(existing_metadata_dict.get("spend_logs_metadata"), dict)
                else {}
            )
            child_session_spend_keys = (
                {
                    "openclaw_session_id",
                    "openclaw_conversation_id",
                    "openclaw_session_id_raw",
                    "openclaw_target_session_id",
                    "openclaw_execution_type",
                    "openclaw_actor_type",
                }
                if execution_type == "subagent"
                else set()
            )
            for key, value in synthetic.items():
                if key.startswith("openclaw_") and value is not None:
                    if key in child_session_spend_keys:
                        merged_spend_logs_metadata[key] = value
                    else:
                        merged_spend_logs_metadata.setdefault(key, value)
            synthetic["spend_logs_metadata"] = merged_spend_logs_metadata
            return {key: value for key, value in synthetic.items() if value is not None}
        synthetic: Dict[str, Any] = {}
        if not existing_metadata_dict.get("trace_user_id"):
            synthetic["trace_user_id"] = OPENCLAW_HEARTBEAT_USER_ID
        synthetic["session_id"] = heartbeat_session_id
        synthetic["user_name"] = OPENCLAW_HEARTBEAT_USER_ID
        synthetic["channel"] = OPENCLAW_HEARTBEAT_TAG
        synthetic["openclaw_user_id"] = OPENCLAW_HEARTBEAT_USER_ID
        synthetic["openclaw_actor_id"] = OPENCLAW_HEARTBEAT_USER_ID
        synthetic["openclaw_actor_type"] = "system"
        synthetic["openclaw_execution_type"] = OPENCLAW_HEARTBEAT_TAG
        synthetic["openclaw_session_id"] = heartbeat_session_id
        synthetic["openclaw_channel"] = OPENCLAW_HEARTBEAT_TAG
        synthetic["openclaw_session_id_raw"] = heartbeat_session_id
        synthetic["openclaw_heartbeat"] = True
        synthetic["tags"] = _merge_openclaw_tags(
            existing_metadata_dict.get("tags"),
            [f"channel:{OPENCLAW_HEARTBEAT_TAG}", OPENCLAW_HEARTBEAT_TAG],
        )
        spend_logs_metadata = existing_metadata_dict.get("spend_logs_metadata")
        merged_spend_logs_metadata = copy.deepcopy(spend_logs_metadata) if isinstance(spend_logs_metadata, dict) else {}
        for key, value in synthetic.items():
            if key.startswith("openclaw_"):
                merged_spend_logs_metadata[key] = value
        synthetic["spend_logs_metadata"] = merged_spend_logs_metadata
        return synthetic or None

    group_channel = _normalize_openclaw_observability_string(
        (conversation_info or {}).get("group_channel"),
        max_length=160,
    )
    topic_id = _normalize_openclaw_observability_string(
        (conversation_info or {}).get("topic_id"),
        max_length=120,
    )
    conversation_label = _normalize_openclaw_observability_string(
        (conversation_info or {}).get("conversation_label"),
        max_length=160,
    )
    inferred_channel = _resolve_openclaw_channel(sender_info=sender_info, conversation_info=conversation_info)
    chat_channel, chat_peer_id = _resolve_openclaw_chat_peer(conversation_info)
    sender_id = _normalize_openclaw_observability_string(
        (sender_info or {}).get("id") or (conversation_info or {}).get("sender_id"),
        max_length=120,
    )
    if sender_id is None and chat_channel is not None:
        if inferred_channel is None or inferred_channel == chat_channel:
            sender_id = chat_peer_id
            inferred_channel = inferred_channel or chat_channel
    if sender_id is None:
        return None

    direct_session_id = _resolve_openclaw_direct_session_from_message_id(
        conversation_info=conversation_info,
        inferred_channel=inferred_channel,
        sender_id=sender_id,
    )
    sub_agent_suffix = _resolve_openclaw_sub_agent_suffix(sender_info)
    actor_type = _resolve_openclaw_actor_type(sender_info, sub_agent_suffix, is_heartbeat)
    cron_id = (
        _first_openclaw_metadata_value(
            ("openclaw_cron_id", "cron_id", "schedule_id"),
            sender_info,
            conversation_info,
            existing_metadata_dict,
            max_length=120,
        )
        or metadata_cron_id
    )
    cron_name = (
        _first_openclaw_metadata_value(
            ("openclaw_cron_name", "cron_name", "schedule_name"),
            sender_info,
            conversation_info,
            existing_metadata_dict,
            max_length=160,
        )
        or metadata_cron_name
    )
    cron_run_id = (
        _first_openclaw_metadata_value(
            ("openclaw_cron_run_id", "cron_run_id", "schedule_run_id"),
            sender_info,
            conversation_info,
            existing_metadata_dict,
            max_length=160,
        )
        or metadata_cron_run_id
    )
    explicit_conversation_id = _first_openclaw_metadata_value(
        ("openclaw_conversation_id", "conversation_id", "thread_id"),
        sender_info,
        conversation_info,
        existing_metadata_dict,
        max_length=200,
    )

    trace_user_id: Optional[str] = None
    raw_session_id: Optional[str] = None
    target_raw_session_id: Optional[str] = None
    parent_raw_session_id: Optional[str] = None
    openclaw_session_id: Optional[str] = None
    if inferred_channel is not None:
        raw_user_id = f"{inferred_channel}:{sender_id}"
        trace_user_id = _build_openclaw_bounded_identifier(
            raw_user_id,
            prefix=f"{inferred_channel}-user",
        )

        if cron_id and cron_run_id:
            raw_session_id = f"{OPENCLAW_SESSION_PREFIX}:cron:{cron_id}:run:{cron_run_id}"
        elif cron_id:
            raw_session_id = f"{OPENCLAW_SESSION_PREFIX}:cron:{cron_id}"
        elif is_heartbeat:
            raw_session_id = heartbeat_session_id
        elif explicit_conversation_id is not None:
            raw_session_id = explicit_conversation_id
        elif group_channel and topic_id:
            raw_session_id = f"{OPENCLAW_SESSION_PREFIX}:{group_channel}:topic:{topic_id}"
        elif group_channel:
            raw_session_id = f"{OPENCLAW_SESSION_PREFIX}:{group_channel}"
        elif direct_session_id is not None:
            raw_session_id = direct_session_id
        else:
            raw_session_id = f"{OPENCLAW_SESSION_PREFIX}:{inferred_channel}:dm:{sender_id}"

        if sub_agent_suffix is not None and not is_heartbeat:
            parent_raw_session_id = raw_session_id
        if inter_session_context and not is_heartbeat:
            target_raw_session_id = raw_session_id
            source_session_id = inter_session_context.get("source_session_id")
            if source_session_id is not None:
                parent_raw_session_id = source_session_id
                raw_session_id = source_session_id

        openclaw_session_id = _build_openclaw_bounded_identifier(
            raw_session_id,
            prefix=f"{OPENCLAW_SESSION_PREFIX}-session",
        )

    user_name = _resolve_openclaw_user_name(sender_info)
    parent_user_id = _resolve_openclaw_parent_user_id(sender_info, conversation_info, inferred_channel)
    if trace_user_id is None and actor_type == "subagent":
        trace_user_id = parent_user_id
    execution_type = _resolve_openclaw_execution_type(
        sender_info,
        conversation_info,
        actor_type,
        parent_user_id,
        is_heartbeat,
        cron_id,
    )
    execution_type = _resolve_openclaw_inter_session_execution_type(
        inter_session_context,
        execution_type,
        source_session_id=parent_raw_session_id,
        target_session_id=target_raw_session_id or raw_session_id,
        is_subagent_request=is_subagent_request,
    )
    if inter_session_context and execution_type == "subagent":
        actor_type = "subagent"
    elif inter_session_context and actor_type == "human":
        actor_type = "bot"
    execution_id = _normalize_openclaw_observability_string(
        (sender_info or {}).get("openclaw_execution_id")
        or (sender_info or {}).get("execution_id")
        or (conversation_info or {}).get("openclaw_execution_id")
        or (conversation_info or {}).get("execution_id"),
        max_length=180,
    )
    bot_id = _normalize_openclaw_observability_string(
        (sender_info or {}).get("bot_id")
        or (sender_info or {}).get("agent_id")
        or (conversation_info or {}).get("bot_id"),
        max_length=160,
    )
    actor_id = _resolve_openclaw_actor_id(
        sender_info,
        conversation_info,
        actor_type,
        inferred_channel,
        trace_user_id,
        bot_id,
    )
    parent_session_id = (
        _build_openclaw_bounded_identifier(
            parent_raw_session_id,
            prefix=f"{OPENCLAW_SESSION_PREFIX}-session",
        )
        if parent_raw_session_id is not None
        else None
    )

    metadata_updates = {
        "trace_user_id": trace_user_id,
        "session_id": openclaw_session_id,
        # `user_name` / `channel` are the keys the Langfuse Users page
        # surfaces as the "Username" / "Channel" columns (issue 2BB-289,
        # fix #1). They mirror `openclaw_sender_username` / `openclaw_channel`
        # but use Langfuse-standard naming so the existing UI columns
        # populate without per-tenant config.
        "user_name": user_name,
        "channel": inferred_channel,
        "openclaw_user_id": trace_user_id,
        "openclaw_parent_user_id": parent_user_id,
        "openclaw_actor_id": actor_id,
        "openclaw_actor_type": actor_type,
        "openclaw_execution_id": execution_id,
        "openclaw_execution_type": execution_type,
        "openclaw_bot_id": bot_id,
        "openclaw_session_id": openclaw_session_id,
        "openclaw_conversation_id": openclaw_session_id,
        "openclaw_channel": inferred_channel,
        "openclaw_session_id_raw": raw_session_id,
        "openclaw_parent_session_id": parent_session_id,
        "openclaw_sub_agent_id": sub_agent_suffix,
        "openclaw_source_session_id": inter_session_context.get("source_session_id"),
        "openclaw_source_channel": inter_session_context.get("source_channel"),
        "openclaw_source_tool": inter_session_context.get("source_tool"),
        "openclaw_target_session_id": target_raw_session_id,
        "openclaw_input_provenance_kind": ("inter_session" if inter_session_context else None),
        "openclaw_heartbeat": True if is_heartbeat else None,
        "openclaw_cron_id": cron_id,
        "openclaw_cron_name": cron_name,
        "openclaw_cron_run_id": cron_run_id,
        "openclaw_sender_id": sender_id,
        "openclaw_sender_name": _normalize_openclaw_observability_string(
            (sender_info or {}).get("name"),
            max_length=160,
        ),
        "openclaw_sender_username": _normalize_openclaw_observability_string(
            (sender_info or {}).get("username"),
            max_length=160,
        ),
        "openclaw_sender_label": _normalize_openclaw_observability_string(
            (sender_info or {}).get("label"),
            max_length=200,
        ),
        "openclaw_conversation_label": conversation_label,
        "openclaw_conversation_message_id": _normalize_openclaw_observability_string(
            (conversation_info or {}).get("message_id"),
            max_length=160,
        ),
        "openclaw_conversation_reply_to_id": _normalize_openclaw_observability_string(
            (conversation_info or {}).get("reply_to_id"),
            max_length=160,
        ),
        "openclaw_conversation_group_channel": group_channel,
        "openclaw_conversation_topic_id": topic_id,
    }

    resolved_metadata: dict = {}
    existing_metadata = existing_metadata_dict
    for key, value in metadata_updates.items():
        if value is None:
            continue
        if (
            key in {"trace_user_id", "session_id"}
            and existing_metadata.get(key)
            and not (key == "session_id" and (is_heartbeat or actor_type == "human"))
        ):
            continue
        resolved_metadata[key] = value

    # Langfuse tags — channel surfaces in the Users page filter bar, and
    # heartbeat/sub-agent tags let operators slice those buckets quickly.
    new_tags: List[str] = []
    if inferred_channel is not None:
        new_tags.append(f"channel:{inferred_channel}")
    elif trusted_inbound_channel is not None:
        new_tags.append(f"channel:{trusted_inbound_channel}")
    if is_heartbeat:
        new_tags.append(OPENCLAW_HEARTBEAT_TAG)
    if execution_type == "cron":
        new_tags.append("cron")
    if (sub_agent_suffix is not None or is_subagent_request) and not is_heartbeat:
        new_tags.append(OPENCLAW_SUB_AGENT_TAG)
    if inter_session_context and not is_heartbeat:
        new_tags.append("inter_session")
        new_tags.append(OPENCLAW_SUB_AGENT_TAG if execution_type == "subagent" else "agent_to_agent")
    if new_tags:
        resolved_metadata["tags"] = _merge_openclaw_tags(existing_metadata.get("tags"), new_tags)

    spend_logs_metadata = existing_metadata.get("spend_logs_metadata")
    merged_spend_logs_metadata = copy.deepcopy(spend_logs_metadata) if isinstance(spend_logs_metadata, dict) else {}
    for key in (
        "openclaw_user_id",
        "openclaw_parent_user_id",
        "openclaw_actor_id",
        "openclaw_actor_type",
        "openclaw_execution_id",
        "openclaw_execution_type",
        "openclaw_bot_id",
        "openclaw_session_id",
        "openclaw_conversation_id",
        "openclaw_channel",
        "openclaw_session_id_raw",
        "openclaw_parent_session_id",
        "openclaw_sub_agent_id",
        "openclaw_source_session_id",
        "openclaw_source_channel",
        "openclaw_source_tool",
        "openclaw_target_session_id",
        "openclaw_input_provenance_kind",
        "openclaw_heartbeat",
        "openclaw_cron_id",
        "openclaw_cron_name",
        "openclaw_cron_run_id",
        "openclaw_sender_id",
        "openclaw_sender_username",
        "openclaw_sender_label",
        "openclaw_conversation_group_channel",
        "openclaw_conversation_topic_id",
    ):
        value = resolved_metadata.get(key) or metadata_updates.get(key)
        if value is None or key in merged_spend_logs_metadata:
            continue
        merged_spend_logs_metadata[key] = value

    if merged_spend_logs_metadata:
        resolved_metadata["spend_logs_metadata"] = merged_spend_logs_metadata

    return resolved_metadata or None


def get_chain_id_from_headers(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Extract chain id for call chaining from request headers.

    Priority order:
    1. ``x-litellm-trace-id`` (explicit, highest priority)
    2. ``x-litellm-session-id`` (explicit)
    3. Any ``x-<vendor>-session-id`` header whose value looks like a session id
       (alphanumeric / UUID, at least 8 chars).  E.g. ``x-claude-code-session-id``.

    Header keys are matched case-insensitively so this works with raw header
    dicts from any transport.

    Used by MCP (and other paths that have raw_headers but no Request) to set
    litellm_trace_id/litellm_session_id for spend logs and logging consistency.
    """
    if not headers:
        return None
    normalized = {k.lower(): v for k, v in headers.items() if isinstance(k, str)}
    return (
        normalized.get("x-litellm-trace-id")
        or normalized.get("x-litellm-session-id")
        or _extract_generic_session_id_from_headers(normalized)
    )


def is_claude_code_user_agent(user_agent: str) -> bool:
    """Claude Code identifies itself as ``claude-cli/<version> ...``; the IDE
    extensions and the Agent SDK run through the same CLI and share that prefix."""
    return user_agent.startswith("claude-cli/")


def should_auto_drop_params_for_claude_code(user_agent: str, data: dict, proxy_config: ProxyConfig) -> bool:
    """drop_params defaults to on for Claude Code so its Anthropic-specific
    params (e.g. thinking) don't fail requests routed to non-Anthropic
    providers. An explicit drop_params from the caller or in the operator's
    ``litellm_settings`` always wins over this default."""
    if not is_claude_code_user_agent(user_agent):
        return False
    if "drop_params" in data:
        return False
    config = getattr(proxy_config, "config", None)
    litellm_settings = config.get("litellm_settings") if isinstance(config, dict) else None
    return not (isinstance(litellm_settings, dict) and "drop_params" in litellm_settings)


def safe_add_api_version_from_query_params(data: dict, request: Request):
    try:
        if hasattr(request, "query_params"):
            query_params = dict(request.query_params)
            if "api-version" in query_params:
                data["api_version"] = query_params["api-version"]
    except KeyError:
        pass
    except Exception as e:
        verbose_logger.exception("error checking api version in query params: %s", str(e))


def convert_key_logging_metadata_to_callback(
    data: AddTeamCallback, team_callback_settings_obj: Optional[TeamCallbackMetadata]
) -> TeamCallbackMetadata:
    if team_callback_settings_obj is None:
        team_callback_settings_obj = TeamCallbackMetadata()
    if data.callback_type == "success":
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []

        if data.callback_name not in team_callback_settings_obj.success_callback:
            team_callback_settings_obj.success_callback.append(data.callback_name)
    elif data.callback_type == "failure":
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []

        if data.callback_name not in team_callback_settings_obj.failure_callback:
            team_callback_settings_obj.failure_callback.append(data.callback_name)
    elif (
        not data.callback_type or data.callback_type == "success_and_failure"
    ):  # assume 'success_and_failure' = litellm.callbacks
        if team_callback_settings_obj.success_callback is None:
            team_callback_settings_obj.success_callback = []
        if team_callback_settings_obj.failure_callback is None:
            team_callback_settings_obj.failure_callback = []
        if team_callback_settings_obj.callbacks is None:
            team_callback_settings_obj.callbacks = []

        if data.callback_name not in team_callback_settings_obj.success_callback:
            team_callback_settings_obj.success_callback.append(data.callback_name)

        if data.callback_name not in team_callback_settings_obj.failure_callback:
            team_callback_settings_obj.failure_callback.append(data.callback_name)

        if data.callback_name not in team_callback_settings_obj.callbacks:
            team_callback_settings_obj.callbacks.append(data.callback_name)

    for var, value in data.callback_vars.items():
        if team_callback_settings_obj.callback_vars is None:
            team_callback_settings_obj.callback_vars = {}
        team_callback_settings_obj.callback_vars[var] = str(value)

    return team_callback_settings_obj


def _get_validated_callback_metadata(item: dict, *, source: str) -> Optional[AddTeamCallback]:
    try:
        return AddTeamCallback(**item)
    except (PydanticValidationError, ValueError) as e:
        verbose_proxy_logger.warning(
            "Ignoring invalid %s callback metadata: %s",
            source,
            _sanitize_for_log(str(e)),
        )
        return None


class KeyAndTeamLoggingSettings:
    """
    Helper class to get the dynamic logging settings for the key and team
    """

    @staticmethod
    def get_key_dynamic_logging_settings(user_api_key_dict: UserAPIKeyAuth):
        if user_api_key_dict.metadata is not None and "logging" in user_api_key_dict.metadata:
            return decrypt_callback_vars(user_api_key_dict.metadata).get("logging")
        return None

    @staticmethod
    def get_team_dynamic_logging_settings(user_api_key_dict: UserAPIKeyAuth):
        if user_api_key_dict.team_metadata is not None and "logging" in user_api_key_dict.team_metadata:
            return decrypt_callback_vars(user_api_key_dict.team_metadata).get("logging")
        return None


def _get_dynamic_logging_metadata(
    user_api_key_dict: UserAPIKeyAuth, proxy_config: ProxyConfig
) -> Optional[TeamCallbackMetadata]:
    callback_settings_obj: Optional[TeamCallbackMetadata] = None
    key_dynamic_logging_settings: Optional[dict] = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(
        user_api_key_dict
    )
    team_dynamic_logging_settings: Optional[dict] = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        user_api_key_dict
    )
    #########################################################################################
    # Key-based callbacks
    #########################################################################################
    if key_dynamic_logging_settings is not None:
        for item in key_dynamic_logging_settings:
            callback = _get_validated_callback_metadata(item=item, source="key-level")
            if callback is None:
                continue
            callback_settings_obj = convert_key_logging_metadata_to_callback(
                data=callback,
                team_callback_settings_obj=callback_settings_obj,
            )
    #########################################################################################
    # Team-based callbacks
    #########################################################################################
    elif team_dynamic_logging_settings is not None:
        for item in team_dynamic_logging_settings:
            callback = _get_validated_callback_metadata(item=item, source="team-level")
            if callback is None:
                continue
            callback_settings_obj = convert_key_logging_metadata_to_callback(
                data=callback,
                team_callback_settings_obj=callback_settings_obj,
            )
    #########################################################################################
    # Deprecated format - maintained for backwards compatibility
    #########################################################################################
    elif user_api_key_dict.team_metadata is not None and "callback_settings" in user_api_key_dict.team_metadata:
        """
        callback_settings = {
            {
            'callback_vars': {'langfuse_public_key': 'pk', 'langfuse_secret_key': 'sk_'},
            'failure_callback': [],
            'success_callback': ['langfuse', 'langfuse']
        }
        }
        """
        team_metadata = decrypt_callback_vars(user_api_key_dict.team_metadata)
        callback_settings = team_metadata.get("callback_settings", None) or {}
        callback_settings_obj = TeamCallbackMetadata(**callback_settings)
        verbose_proxy_logger.debug("Team callback settings activated: %s", callback_settings_obj)
    #########################################################################################
    # Enter here when configured on the config.yaml file.
    #########################################################################################
    elif user_api_key_dict.team_id is not None:
        callback_settings_obj = LiteLLMProxyRequestSetup.add_team_based_callbacks_from_config(
            team_id=user_api_key_dict.team_id, proxy_config=proxy_config
        )
    return callback_settings_obj


def clean_headers(
    headers: Headers,
    litellm_key_header_name: Optional[str] = None,
    forward_llm_provider_auth_headers: bool = False,
    authenticated_with_header: Optional[str] = None,
) -> dict:
    """
    Removes litellm api key from headers

    Args:
        headers: Request headers
        litellm_key_header_name: Custom header name for LiteLLM API key
        forward_llm_provider_auth_headers: Whether to forward provider auth headers
        authenticated_with_header: Which header was used for LiteLLM authentication
            (e.g., "x-litellm-api-key", "authorization", "x-api-key")

    Returns:
        Cleaned headers dict
    """
    from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

    clean_headers = {}
    litellm_key_lower = litellm_key_header_name.lower() if litellm_key_header_name is not None else None
    for header, value in headers.items():
        header_lower = header.lower()

        if header_lower == "authorization" and is_anthropic_oauth_key(value):
            if authenticated_with_header is None or authenticated_with_header.lower() != "authorization":
                clean_headers[header] = value
            continue
        # Special handling for x-api-key: forward it based on authenticated_with_header
        elif header_lower == "x-api-key":
            if forward_llm_provider_auth_headers and (
                authenticated_with_header is None or authenticated_with_header.lower() != "x-api-key"
            ):
                clean_headers[header] = value
        elif forward_llm_provider_auth_headers and header_lower in _SPECIAL_HEADERS_CACHE:
            if litellm_key_lower and header_lower == litellm_key_lower:
                continue
            if header_lower == "authorization":
                continue
            # Never forward x-litellm-api-key (it's for proxy auth only)
            if header_lower == "x-litellm-api-key":
                continue
            clean_headers[header] = value
        # Check if header should be excluded: either in special headers cache or matches custom litellm key
        elif header_lower not in _SPECIAL_HEADERS_CACHE and (
            litellm_key_lower is None or header_lower != litellm_key_lower
        ):
            clean_headers[header] = value
    return clean_headers


class LiteLLMProxyRequestSetup:
    @staticmethod
    def _get_timeout_from_request(headers: dict) -> Optional[float]:
        """
        Workaround for client request from Vercel's AI SDK.

        Allow's user to set a timeout in the request headers.

        Example:

        ```js
        const openaiProvider = createOpenAI({
            baseURL: liteLLM.baseURL,
            apiKey: liteLLM.apiKey,
            compatibility: "compatible",
            headers: {
                "x-litellm-timeout": "90"
            },
        });
        ```
        """
        timeout_header = headers.get("x-litellm-timeout", None)
        if timeout_header is not None:
            return float(timeout_header)
        return None

    @staticmethod
    def _get_stream_timeout_from_request(headers: dict) -> Optional[float]:
        """
        Get the `stream_timeout` from the request headers.
        """
        stream_timeout_header = headers.get("x-litellm-stream-timeout", None)
        if stream_timeout_header is not None:
            return float(stream_timeout_header)
        return None

    @staticmethod
    def _get_num_retries_from_request(headers: dict) -> Optional[int]:
        """
        Workaround for client request from Vercel's AI SDK.
        """
        num_retries_header = headers.get("x-litellm-num-retries", None)
        if num_retries_header is not None:
            return int(num_retries_header)
        return None

    @staticmethod
    def _get_spend_logs_metadata_from_request_headers(headers: dict) -> Optional[dict]:
        """
        Get the `spend_logs_metadata` from the request headers.
        """
        from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

        spend_logs_metadata_header = headers.get("x-litellm-spend-logs-metadata", None)
        if spend_logs_metadata_header is not None:
            return safe_json_loads(spend_logs_metadata_header)
        return None

    @staticmethod
    def _get_forwardable_headers(
        headers: Union[Headers, dict],
    ):
        """
        Get the headers that should be forwarded to the LLM Provider.

        Looks for any `x-` headers and sends them to the LLM Provider.

        [07/09/2025] - Support 'anthropic-beta' header as well.
        """
        forwarded_headers = {}
        for header, value in headers.items():
            if header.lower().startswith("x-") and not header.lower().startswith(
                "x-stainless"
            ):  # causes openai sdk to fail
                forwarded_headers[header] = value
            elif header.lower().startswith("anthropic-beta"):
                forwarded_headers[header] = value

        return forwarded_headers

    @staticmethod
    def _get_case_insensitive_header(headers: dict, key: str) -> Optional[str]:
        """
        Get a case-insensitive header from the headers dictionary.
        """
        for header, value in headers.items():
            if header.lower() == key.lower():
                return value
        return None

    @staticmethod
    def add_internal_user_from_user_mapping(
        general_settings: Optional[Dict],
        user_api_key_dict: UserAPIKeyAuth,
        headers: dict,
    ) -> UserAPIKeyAuth:
        if general_settings is None:
            return user_api_key_dict
        user_header_mapping = general_settings.get("user_header_mappings")
        if not user_header_mapping:
            return user_api_key_dict
        header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(user_header_mapping)
        if not header_name:
            return user_api_key_dict
        header_value = LiteLLMProxyRequestSetup._get_case_insensitive_header(headers, header_name)
        if header_value:
            user_api_key_dict.user_id = header_value
            return user_api_key_dict
        return user_api_key_dict

    @staticmethod
    def get_user_from_headers(headers: dict, general_settings: Optional[Dict] = None) -> Optional[str]:
        """
        Get the user from the specified header if `general_settings.user_header_name` is set.
        """
        if general_settings is None:
            return None

        header_name = general_settings.get("user_header_name")
        if header_name is None or header_name == "":
            return None

        if not isinstance(header_name, str):
            raise TypeError(f"Expected user_header_name to be a str but got {type(header_name)}")

        user = LiteLLMProxyRequestSetup._get_case_insensitive_header(headers, header_name)
        if user is not None:
            verbose_logger.info(f'found user "{user}" in header "{header_name}"')

        return user

    @staticmethod
    def get_openai_org_id_from_headers(headers: dict, general_settings: Optional[Dict] = None) -> Optional[str]:
        """
        Get the OpenAI Org ID from the headers.
        """
        if general_settings is not None and general_settings.get("forward_openai_org_id") is not True:
            return None
        for header, value in headers.items():
            if header.lower() == "openai-organization":
                verbose_logger.info(f"found openai org id: {value}, sending to llm")
                return value
        return None

    @staticmethod
    def add_headers_to_llm_call(headers: dict, user_api_key_dict: UserAPIKeyAuth) -> dict:
        """
        Add headers to the LLM call

        - Checks request headers for forwardable headers
        - Checks if user information should be added to the headers
        """

        returned_headers = LiteLLMProxyRequestSetup._get_forwardable_headers(headers)

        if litellm.add_user_information_to_llm_headers is True:
            litellm_logging_metadata_headers = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
                user_api_key_dict=user_api_key_dict
            )
            for k, v in litellm_logging_metadata_headers.items():
                if v is None:
                    continue
                # httpx requires header values to be str or bytes; coerce numbers/bools
                # to str and JSON-encode dict/list (e.g. user_api_key_spend is float,
                # user_api_key_auth_metadata is dict). See #27458.
                if isinstance(v, (dict, list)):
                    returned_headers["x-litellm-{}".format(k)] = json.dumps(v)
                elif isinstance(v, (str, bytes)):
                    returned_headers["x-litellm-{}".format(k)] = v
                else:
                    returned_headers["x-litellm-{}".format(k)] = str(v)

        return returned_headers

    @staticmethod
    def add_headers_to_llm_call_by_model_group(data: dict, headers: dict, user_api_key_dict: UserAPIKeyAuth) -> dict:
        """
        Add headers to the LLM call by model group
        """
        from litellm.proxy.auth.auth_checks import _check_model_access_helper
        from litellm.proxy.proxy_server import llm_router

        data_model = data.get("model")

        if (
            data_model is not None
            and litellm.model_group_settings is not None
            and litellm.model_group_settings.forward_client_headers_to_llm_api is not None
            and _check_model_access_helper(
                model=data_model,
                llm_router=llm_router,
                models=litellm.model_group_settings.forward_client_headers_to_llm_api,
                team_model_aliases=user_api_key_dict.team_model_aliases,
                team_id=user_api_key_dict.team_id,
            )  # handles aliases, wildcards, etc.
        ):
            _headers = LiteLLMProxyRequestSetup.add_headers_to_llm_call(headers, user_api_key_dict)
            if _headers != {}:
                data["headers"] = _headers
        return data

    @staticmethod
    def get_internal_user_header_from_mapping(user_header_mapping) -> Optional[str]:
        if not user_header_mapping:
            return None
        items = user_header_mapping if isinstance(user_header_mapping, list) else [user_header_mapping]
        for item in items:
            if not isinstance(item, dict):
                continue
            role = item.get("litellm_user_role")
            header_name = item.get("header_name")
            if role is None or not header_name:
                continue
            if str(role).lower() == str(LitellmUserRoles.INTERNAL_USER).lower():
                return header_name
        return None

    @staticmethod
    def add_litellm_data_for_backend_llm_call(
        *,
        headers: dict,
        user_api_key_dict: UserAPIKeyAuth,
        general_settings: Optional[Dict[str, Any]] = None,
    ) -> LitellmDataForBackendLLMCall:
        """
        - Adds user from headers
        - Adds forwardable headers
        - Adds org id
        """
        data = LitellmDataForBackendLLMCall()

        if general_settings and general_settings.get("forward_client_headers_to_llm_api") is True:
            _headers = LiteLLMProxyRequestSetup.add_headers_to_llm_call(headers, user_api_key_dict)
            if _headers != {}:
                data["headers"] = _headers
        _organization = LiteLLMProxyRequestSetup.get_openai_org_id_from_headers(headers, general_settings)
        if _organization is not None:
            data["organization"] = _organization

        timeout = LiteLLMProxyRequestSetup._get_timeout_from_request(headers)
        if timeout is not None:
            data["timeout"] = timeout

        stream_timeout = LiteLLMProxyRequestSetup._get_stream_timeout_from_request(headers)
        if stream_timeout is not None:
            data["stream_timeout"] = stream_timeout

        num_retries = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers)
        if num_retries is not None:
            data["num_retries"] = num_retries

        return data

    @staticmethod
    def add_litellm_metadata_from_request_headers(
        headers: dict,
        data: dict,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Add litellm metadata from request headers

        Relevant issue: https://github.com/BerriAI/litellm/issues/14008
        """
        from litellm.proxy._types import LitellmMetadataFromRequestHeaders

        metadata_from_headers = LitellmMetadataFromRequestHeaders()
        spend_logs_metadata = LiteLLMProxyRequestSetup._get_spend_logs_metadata_from_request_headers(headers)
        if spend_logs_metadata is not None:
            metadata_from_headers["spend_logs_metadata"] = spend_logs_metadata

        #########################################################################################
        # Finally update the requests metadata with the `metadata_from_headers`
        #########################################################################################

        agent_id_from_header = headers.get("x-litellm-agent-id")
        # Explicit litellm headers take precedence; fall back to any x-*-session-id header.
        chain_id = get_chain_id_from_headers(dict(headers))

        if agent_id_from_header:
            metadata_from_headers["agent_id"] = agent_id_from_header
            verbose_proxy_logger.debug(f"Extracted agent_id from header: {agent_id_from_header}")

        if chain_id:
            metadata_from_headers["trace_id"] = chain_id
            metadata_from_headers["session_id"] = chain_id
            data["litellm_session_id"] = chain_id
            data["litellm_trace_id"] = chain_id
            verbose_proxy_logger.debug(f"Extracted chain_id from header (trace-id/session-id): {chain_id}")

        if isinstance(data[_metadata_variable_name], dict):
            data[_metadata_variable_name].update(metadata_from_headers)
        return data

    @staticmethod
    def get_sanitized_user_information_from_key(
        user_api_key_dict: UserAPIKeyAuth,
    ) -> StandardLoggingUserAPIKeyMetadata:
        user_api_key_logged_metadata = StandardLoggingUserAPIKeyMetadata(
            user_api_key_hash=user_api_key_dict.api_key,  # just the hashed token
            user_api_key_alias=user_api_key_dict.key_alias,
            user_api_key_spend=user_api_key_dict.spend,
            user_api_key_max_budget=user_api_key_dict.max_budget,
            user_api_key_team_id=user_api_key_dict.team_id,
            user_api_key_project_id=user_api_key_dict.project_id,
            user_api_key_project_alias=user_api_key_dict.project_alias,
            user_api_key_user_id=user_api_key_dict.user_id,
            user_api_key_org_id=user_api_key_dict.org_id,
            user_api_key_org_alias=user_api_key_dict.organization_alias,
            user_api_key_team_alias=user_api_key_dict.team_alias,
            user_api_key_end_user_id=user_api_key_dict.end_user_id,
            user_api_key_user_email=user_api_key_dict.user_email,
            user_api_key_request_route=user_api_key_dict.request_route,
            user_api_key_budget_reset_at=(
                user_api_key_dict.budget_reset_at.isoformat() if user_api_key_dict.budget_reset_at else None
            ),
            user_api_key_auth_metadata=user_api_key_dict.metadata,
        )
        return user_api_key_logged_metadata

    @staticmethod
    def add_user_api_key_auth_to_request_metadata(
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Adds the `UserAPIKeyAuth` object to the request metadata.
        """
        user_api_key_logged_metadata = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=user_api_key_dict
        )
        data[_metadata_variable_name].update(user_api_key_logged_metadata)
        data[_metadata_variable_name]["user_api_key"] = user_api_key_dict.api_key  # this is just the hashed token

        # Key-owned agent_id for spend attribution; keep existing (e.g. from header) if key has none
        _key_agent_id = getattr(user_api_key_dict, "agent_id", None)
        _existing_agent_id = data[_metadata_variable_name].get("agent_id")
        _resolved_agent_id = _key_agent_id or _existing_agent_id
        data[_metadata_variable_name]["agent_id"] = _resolved_agent_id

        data[_metadata_variable_name]["user_api_end_user_max_budget"] = getattr(
            user_api_key_dict, "end_user_max_budget", None
        )
        if user_api_key_dict.budget_reservation is not None:
            data[_metadata_variable_name]["user_api_key_budget_reservation"] = user_api_key_dict.budget_reservation
        # Add the full UserAPIKeyAuth object for MCP server access control
        data[_metadata_variable_name]["user_api_key_auth"] = user_api_key_dict
        return data

    @staticmethod
    def add_management_endpoint_metadata_to_request_metadata(
        data: dict,
        management_endpoint_metadata: dict,
        _metadata_variable_name: str,
    ) -> dict:
        """
        Adds the `UserAPIKeyAuth` metadata to the request metadata.

        ignore any sensitive fields like logging, api_key, etc.
        """
        if _metadata_variable_name not in data:
            return data
        from litellm.proxy._types import (
            LiteLLM_ManagementEndpoint_MetadataFields,
            LiteLLM_ManagementEndpoint_MetadataFields_Premium,
        )

        # ignore any special fields
        added_metadata = {}
        for k, v in management_endpoint_metadata.items():
            if k not in (LiteLLM_ManagementEndpoint_MetadataFields_Premium + LiteLLM_ManagementEndpoint_MetadataFields):
                added_metadata[k] = v
        if data[_metadata_variable_name].get("user_api_key_auth_metadata") is None:
            data[_metadata_variable_name]["user_api_key_auth_metadata"] = {}
        data[_metadata_variable_name]["user_api_key_auth_metadata"].update(added_metadata)
        return data

    @staticmethod
    def add_key_level_controls(key_metadata: Optional[dict], data: dict, _metadata_variable_name: str):
        if key_metadata is None:
            return data
        if "cache" in key_metadata:
            data["cache"] = {}
            if isinstance(key_metadata["cache"], dict):
                for k, v in key_metadata["cache"].items():
                    if k in SupportedCacheControls:
                        data["cache"][k] = v

        ## KEY-LEVEL SPEND LOGS / TAGS
        if "tags" in key_metadata and key_metadata["tags"] is not None:
            data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(
                request_tags=data[_metadata_variable_name].get("tags"),
                tags_to_add=key_metadata["tags"],
            )
        if "disable_global_guardrails" in key_metadata and isinstance(key_metadata["disable_global_guardrails"], bool):
            data[_metadata_variable_name]["disable_global_guardrails"] = key_metadata["disable_global_guardrails"]
        if "spend_logs_metadata" in key_metadata and isinstance(key_metadata["spend_logs_metadata"], dict):
            if "spend_logs_metadata" in data[_metadata_variable_name] and isinstance(
                data[_metadata_variable_name]["spend_logs_metadata"], dict
            ):
                for key, value in key_metadata["spend_logs_metadata"].items():
                    if (
                        key not in data[_metadata_variable_name]["spend_logs_metadata"]
                    ):  # don't override k-v pair sent by request (user request)
                        data[_metadata_variable_name]["spend_logs_metadata"][key] = value
            else:
                data[_metadata_variable_name]["spend_logs_metadata"] = key_metadata["spend_logs_metadata"]

        ## KEY-LEVEL DISABLE FALLBACKS
        if "disable_fallbacks" in key_metadata and isinstance(key_metadata["disable_fallbacks"], bool):
            data["disable_fallbacks"] = key_metadata["disable_fallbacks"]

        ## KEY-LEVEL METADATA
        data = LiteLLMProxyRequestSetup.add_management_endpoint_metadata_to_request_metadata(
            data=data,
            management_endpoint_metadata=key_metadata,
            _metadata_variable_name=_metadata_variable_name,
        )
        return data

    @staticmethod
    def _merge_tags(request_tags: Optional[list], tags_to_add: Optional[list]) -> list:
        """
        Helper function to merge two lists of tags, ensuring no duplicates.

        Args:
            request_tags (Optional[list]): List of tags from the original request
            tags_to_add (Optional[list]): List of tags to add

        Returns:
            list: Combined list of unique tags
        """
        final_tags = []

        if request_tags and isinstance(request_tags, list):
            final_tags.extend(request_tags)

        if tags_to_add and isinstance(tags_to_add, list):
            for tag in tags_to_add:
                if tag not in final_tags:
                    final_tags.append(tag)

        return final_tags

    @staticmethod
    def add_team_based_callbacks_from_config(
        team_id: str,
        proxy_config: ProxyConfig,
    ) -> Optional[TeamCallbackMetadata]:
        """
        Add team-based callbacks from the config
        """
        team_config = proxy_config.load_team_config(team_id=team_id)
        if not isinstance(team_config, dict) or len(team_config) == 0:
            return None

        callback_vars_dict = {**team_config.get("callback_vars", team_config)}
        callback_vars_dict.pop("team_id", None)
        callback_vars_dict.pop("success_callback", None)
        callback_vars_dict.pop("failure_callback", None)
        callback_vars_dict = {
            key: (litellm.utils.get_secret(value, default_value=value) or value if isinstance(value, str) else value)
            for key, value in callback_vars_dict.items()
        }

        return TeamCallbackMetadata(
            success_callback=team_config.get("success_callback", None),
            failure_callback=team_config.get("failure_callback", None),
            callback_vars=callback_vars_dict,
        )

    @staticmethod
    def add_request_tag_to_metadata(
        llm_router: Optional[Router],
        headers: dict,
        data: dict,
    ) -> Optional[List[str]]:
        tags = None

        # Check request headers for tags
        if "x-litellm-tags" in headers:
            if isinstance(headers["x-litellm-tags"], str):
                _tags = headers["x-litellm-tags"].split(",")
                tags = [tag.strip() for tag in _tags]
            elif isinstance(headers["x-litellm-tags"], list):
                tags = headers["x-litellm-tags"]
        # Check request body for tags
        if "tags" in data and isinstance(data["tags"], list):
            tags = data["tags"]

        return tags

    @staticmethod
    def pre_seed_litellm_metadata_for_route(
        request_data: dict,
        route: str,
    ) -> None:
        """Pre-seed ``litellm_metadata`` for routes that track tags there.

        Routes in ``LITELLM_METADATA_ROUTES`` (e.g. Bedrock, ``/v1/messages``,
        responses, batches, files) store request-scoped tag metadata in
        ``litellm_metadata`` rather than the provider-facing ``metadata``
        field. ``get_metadata_variable_name_from_kwargs`` picks the target
        based on whether ``litellm_metadata`` is present, so it must be
        seeded BEFORE any tag merge runs; otherwise header tags from
        ``apply_client_tag_policy_pre_auth`` land in ``metadata`` while
        key tags from ``apply_key_tags_pre_auth`` and the read in
        ``_tag_max_budget_check`` resolve to ``litellm_metadata``, leaving
        header tags invisible to per-tag budget enforcement.
        """
        if any(metadata_route in route for metadata_route in LITELLM_METADATA_ROUTES):
            request_data.setdefault("litellm_metadata", {})

    @staticmethod
    def apply_key_tags_pre_auth(
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """Merge key metadata tags into request_data before _tag_max_budget_check."""
        key_metadata = user_api_key_dict.metadata
        if not key_metadata:
            return

        key_tags = key_metadata.get("tags")
        if not key_tags or not isinstance(key_tags, list):
            return

        _metadata_variable_name = get_metadata_variable_name_from_kwargs(request_data)
        metadata = request_data.get(_metadata_variable_name)
        if isinstance(metadata, str):
            parsed = safe_json_loads(metadata)
            metadata = parsed if isinstance(parsed, dict) else {}
            request_data[_metadata_variable_name] = metadata
        elif not isinstance(metadata, dict):
            metadata = {}
            request_data[_metadata_variable_name] = metadata

        existing_tags = metadata.get("tags")
        metadata["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=existing_tags if isinstance(existing_tags, list) else None,
            tags_to_add=key_tags,
        )

    @staticmethod
    def apply_client_tag_policy_pre_auth(
        request: Request,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Merge ``x-litellm-tags`` header tags into ``request_data`` BEFORE
        auth budget gates run, so ``_tag_max_budget_check`` (which only
        inspects ``request_data``) sees them. Without this, header-tagged
        requests silently bypass per-tag budget enforcement.

        Why: ``add_litellm_data_to_request`` runs the equivalent merge
        post-auth, after ``_tag_max_budget_check`` has already executed.
        Header-supplied tags merged there are invisible to that check.
        Running the merge here closes that gap; the post-auth merge in
        ``add_litellm_data_to_request`` remains as defense-in-depth.

        How to apply: invoked from the auth chain just before
        ``common_checks``. Mutates ``request_data`` in place; idempotent
        when followed by ``add_litellm_data_to_request``.
        """
        # No allow_client_tags opt-in: caller-supplied tags always flow
        # into metadata.tags (see add_litellm_data_to_request). The pre-auth
        # merge mirrors that so _tag_max_budget_check sees the same tags.
        headers = _safe_get_request_headers(request=request)
        raw_header_tags = headers.get("x-litellm-tags")
        if not raw_header_tags:
            return

        if isinstance(raw_header_tags, str):
            header_tags: List[str] = [t.strip() for t in raw_header_tags.split(",") if t.strip()]
        elif isinstance(raw_header_tags, list):
            header_tags = [t for t in raw_header_tags if isinstance(t, str) and t]
        else:
            return

        if not header_tags:
            return

        # Match the metadata key that get_tags_from_request_body will read
        # from (litellm_metadata vs metadata) so the merged tags are visible
        # to _tag_max_budget_check.
        _metadata_variable_name = get_metadata_variable_name_from_kwargs(request_data)
        metadata = request_data.get(_metadata_variable_name)
        # metadata can arrive as a JSON string (multipart/form-data, extra_body).
        # Parse it so existing tags survive the merge — overwriting the string
        # with {} would let a caller bypass _tag_max_budget_check on an
        # over-budget body tag by also sending a within-budget header tag.
        if isinstance(metadata, str):
            parsed = safe_json_loads(metadata)
            metadata = parsed if isinstance(parsed, dict) else {}
            request_data[_metadata_variable_name] = metadata
        elif not isinstance(metadata, dict):
            metadata = {}
            request_data[_metadata_variable_name] = metadata

        existing_tags = metadata.get("tags")
        metadata["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=existing_tags if isinstance(existing_tags, list) else None,
            tags_to_add=header_tags,
        )


async def add_litellm_data_to_request(
    data: dict,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    proxy_config: ProxyConfig,
    general_settings: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
):
    """
    Adds LiteLLM-specific data to the request.

    Args:
        data (dict): The data dictionary to be modified.
        request (Request): The incoming request.
        user_api_key_dict (UserAPIKeyAuth): The user API key dictionary.
        general_settings (Optional[Dict[str, Any]], optional): General settings. Defaults to None.
        version (Optional[str], optional): Version. Defaults to None.

    Returns:
        dict: The modified data dictionary.

    """

    from litellm.proxy.proxy_server import llm_router, premium_user
    from litellm.types.proxy.litellm_pre_call_utils import RedactedDict, SecretFields

    # Strip internal-only keys from user input before the proxy sets its own.
    # These keys are injected by the proxy itself below — user-supplied values
    # must not be trusted.
    _allow_client_mock_response = _key_or_team_allows_client_mock_response(user_api_key_dict)
    _allow_client_message_redaction_opt_out = _key_or_team_allows_client_message_redaction_opt_out(user_api_key_dict)
    for _internal_key in _UNTRUSTED_ROOT_CONTROL_FIELDS:
        if _allow_client_mock_response and _internal_key in _CLIENT_MOCK_CONTROL_FIELDS:
            continue
        data.pop(_internal_key, None)
    _reject_url_valued_destinations(data)
    # Strip spoofable auth metadata from user-supplied metadata dict
    _user_metadata = data.get("metadata")
    if isinstance(_user_metadata, dict):
        for _mk in list(_user_metadata.keys()):
            if _mk.startswith("user_api_key_"):
                del _user_metadata[_mk]

    _raw_headers: Dict[str, str] = RedactedDict(_safe_get_request_headers(request))

    forward_llm_auth = False
    if general_settings:
        forward_llm_auth = general_settings.get("forward_llm_provider_auth_headers", False)
    if not forward_llm_auth:
        forward_llm_auth = getattr(litellm, "forward_llm_provider_auth_headers", False)
    # Determine which header was used for authentication
    # This enables forwarding provider keys (e.g., x-api-key) when they weren't used for LiteLLM auth
    authenticated_with_header = None
    if "x-litellm-api-key" in request.headers:
        # If x-litellm-api-key is present, it was used for auth
        authenticated_with_header = "x-litellm-api-key"
    elif "authorization" in request.headers:
        # Authorization header was used for auth
        authenticated_with_header = "authorization"
    else:
        # x-api-key or another header was used for auth
        authenticated_with_header = "x-api-key"

    _headers: Dict[str, str] = clean_headers(
        request.headers,
        litellm_key_header_name=(
            general_settings.get("litellm_key_header_name") if general_settings is not None else None
        ),
        forward_llm_provider_auth_headers=forward_llm_auth,
        authenticated_with_header=authenticated_with_header,
    )
    _strip_untrusted_request_header_controls(
        _headers,
        allow_client_message_redaction_opt_out=_allow_client_message_redaction_opt_out,
    )
    if (
        not _allow_client_message_redaction_opt_out
        and litellm.turn_off_message_logging is True
        and "turn_off_message_logging" in data
        and _is_false_like(data["turn_off_message_logging"])
    ):
        data.pop("turn_off_message_logging", None)
    verbose_proxy_logger.debug(f"Request Headers: {_headers}")
    verbose_proxy_logger.debug(f"Raw Headers: {_raw_headers}")

    if forward_llm_auth and "x-api-key" in _headers:
        data["api_key"] = _headers["x-api-key"]
        verbose_proxy_logger.debug(
            "Setting client-provided x-api-key as api_key parameter (will override deployment key)"
        )

    ##########################################################
    # Init - Proxy Server Request
    # we do this as soon as entering so we track the original request
    ##########################################################
    # Track arrival time for queue time metric. The body snapshot is filled
    # in after the admin-injection strip below so the audit / spend-tracking
    # consumers of proxy_server_request["body"] see the cleaned metadata
    # rather than attacker-forged user_api_key_* fields.
    arrival_time = time.time()
    data["proxy_server_request"] = {
        "url": str(request.url),
        "method": request.method,
        "headers": _headers,
        "body": None,  # filled in post-strip; see below
        "arrival_time": arrival_time,  # Track when request arrived at proxy
    }

    safe_add_api_version_from_query_params(data, request)
    _metadata_variable_name = _get_metadata_variable_name(request)
    if data.get(_metadata_variable_name, None) is None:
        data[_metadata_variable_name] = {}

    data.update(
        LiteLLMProxyRequestSetup.add_litellm_data_for_backend_llm_call(
            headers=_headers,
            user_api_key_dict=user_api_key_dict,
            general_settings=general_settings,
        )
    )

    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=_headers,
        data=data,
        _metadata_variable_name=_metadata_variable_name,
    )

    # Expose request headers under the metadata field for guardrails (fixes #17477)
    if _metadata_variable_name in data and isinstance(data[_metadata_variable_name], dict):
        data[_metadata_variable_name]["headers"] = _headers

    # check for forwardable headers
    data = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
        data=data, headers=_headers, user_api_key_dict=user_api_key_dict
    )

    user_api_key_dict = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, _headers
    )

    # Parse user info from headers (fallback to general_settings.user_header_name)
    user = LiteLLMProxyRequestSetup.get_user_from_headers(_headers, general_settings)
    if user is not None:
        if user_api_key_dict.end_user_id is None:
            user_api_key_dict.end_user_id = user
        if "user" not in data:
            data["user"] = user

    data["secret_fields"] = SecretFields(raw_headers=_raw_headers)

    ## Dynamic api version (Azure OpenAI endpoints) ##
    try:
        query_params = request.query_params
        # Convert query parameters to a dictionary (optional)
        query_dict = dict(query_params)
    except KeyError:
        query_dict = {}

    ## check for api version in query params
    dynamic_api_version: Optional[str] = query_dict.get("api-version")

    if dynamic_api_version is not None:  # only pass, if set
        data["api_version"] = dynamic_api_version

    ## Forward any LLM API Provider specific headers in extra_headers
    add_provider_specific_headers_to_request(data=data, headers=_headers)

    ## Cache Controls
    cache_control_header = _headers.get("Cache-Control", None)
    if cache_control_header:
        cache_dict = parse_cache_control(cache_control_header)
        data["ttl"] = cache_dict.get("s-maxage")

    verbose_proxy_logger.debug("receiving data: %s", data)

    # Parse metadata if it's a string (e.g., from multipart/form-data)
    if "metadata" in data and data["metadata"] is not None:
        if isinstance(data["metadata"], str):
            data["metadata"] = safe_json_loads(data["metadata"])
            if not isinstance(data["metadata"], dict):
                verbose_proxy_logger.warning(
                    f"Failed to parse 'metadata' as JSON dict. Received value: {data['metadata']}"
                )
        # requester_metadata is snapshotted AFTER the strip below so
        # downstream consumers (e.g. PANW guardrail reading user_ip /
        # profile_id) don't see attacker-injected admin slots preserved in
        # the deepcopy.

    # Parse litellm_metadata if it's a string (e.g., from multipart/form-data or extra_body)
    if "litellm_metadata" in data and data["litellm_metadata"] is not None:
        if isinstance(data["litellm_metadata"], str):
            parsed_litellm_metadata = safe_json_loads(data["litellm_metadata"])
            if not isinstance(parsed_litellm_metadata, dict):
                verbose_proxy_logger.warning(
                    f"Failed to parse 'litellm_metadata' as JSON dict. Received value: {data['litellm_metadata']}"
                )
            else:
                data["litellm_metadata"] = parsed_litellm_metadata

    # Strip internal pipeline state and admin-injection slots from user input.
    # Runs AFTER the string-to-dict parse above so JSON-string metadata (sent
    # via multipart/form-data or extra_body) cannot smuggle admin fields past
    # the isinstance(dict) guard.
    #
    # The proxy populates a family of ``user_api_key_*`` fields below
    # (user_api_key_metadata, user_api_key_user_id, user_api_key_alias,
    # user_api_key_spend, user_api_key_team_metadata, …) into
    # data[_metadata_variable_name]. Because the proxy only writes to ONE of
    # the two metadata dicts, a caller pre-populating any of these keys on
    # the OTHER metadata dict would have their forged values surface in
    # guardrails, spend tracking, audit logs, and identity resolution. Strip
    # by prefix so new ``user_api_key_*`` fields added in the future are
    # covered without per-key maintenance.
    for _meta_key in ("metadata", "litellm_metadata"):
        _user_meta = data.get(_meta_key)
        if isinstance(_user_meta, dict):
            _strip_untrusted_request_header_controls(
                _user_meta.get("headers"),
                allow_client_message_redaction_opt_out=(_allow_client_message_redaction_opt_out),
            )
            for _k in [
                k for k in _user_meta if k.startswith("user_api_key_") or k in _UNTRUSTED_METADATA_CONTROL_FIELDS
            ]:
                _user_meta.pop(_k, None)

    # Strip pricing overrides AFTER the litellm_metadata string-to-dict parse
    # above, for the same reason as the user_api_key_* strip — JSON-string
    # metadata (sent via multipart/form-data or extra_body) wouldn't be a
    # dict yet at the earlier strip point and the isinstance(dict) guard
    # would silently skip the field.
    if not _key_or_team_allows_client_pricing_override(user_api_key_dict):
        _strip_client_pricing_overrides(data)

    # Fill in the proxy_server_request body snapshot now that metadata has
    # been parsed. Consumers (standard_logging_payload, lago,
    # spend_tracking_utils, streaming_iterator) read `body` to audit the
    # request; taking the snapshot here ensures they see cleaned metadata.
    #
    # Exclude:
    #   - secret_fields: contains raw_headers with Authorization tokens; must
    #     never be persisted in spend logs or any other audit trail.
    #   - proxy_server_request: already a key on `data` at this point (set
    #     earlier in this function); including it would make the snapshot
    #     self-reference — body.proxy_server_request.body would be the same
    #     dict as body, producing an infinite traversal loop for any consumer
    #     that walks the structure.
    _body_snapshot_exclude = {"secret_fields", "proxy_server_request"}
    _body_snapshot = {k: v for k, v in data.items() if k not in _body_snapshot_exclude}
    data["proxy_server_request"]["body"] = _body_snapshot

    # Snapshot the requester-supplied metadata for downstream consumers.
    # Taking the deepcopy after the user_api_key_* / _pipeline_managed_guardrails
    # strip above prevents those proxy-internal slots — if a caller forged
    # them — from leaking into requester_metadata where guardrails and audit
    # paths may read from it.
    if "metadata" in data and isinstance(data["metadata"], dict):
        data[_metadata_variable_name]["requester_metadata"] = copy.deepcopy(data["metadata"])

    # Merge litellm_metadata into the metadata variable (preserving existing
    # values). Runs after the user_api_key_* / _pipeline_managed_guardrails
    # strip above so those proxy-internal slots — if a caller forged them
    # into litellm_metadata — cannot cross-contaminate the admin-authoritative
    # metadata dict.
    if "litellm_metadata" in data and isinstance(data["litellm_metadata"], dict):
        for key, value in data["litellm_metadata"].items():
            if key not in data[_metadata_variable_name]:
                data[_metadata_variable_name][key] = value

    openclaw_observability_metadata = _build_openclaw_observability_metadata(
        data=data,
        existing_metadata=data.get(_metadata_variable_name),
    )
    if openclaw_observability_metadata is not None:
        data[_metadata_variable_name].update(openclaw_observability_metadata)
        resolved_session_id = (
            data[_metadata_variable_name].get("session_id") if isinstance(data[_metadata_variable_name], dict) else None
        )
        should_override_litellm_session_id = bool(openclaw_observability_metadata.get("openclaw_conversation_id"))
        if resolved_session_id is not None and (
            data.get("litellm_session_id") is None or should_override_litellm_session_id
        ):
            data["litellm_session_id"] = resolved_session_id

    data = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data,
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name=_metadata_variable_name,
    )
    data[_metadata_variable_name]["litellm_api_version"] = version

    if general_settings is not None:
        data[_metadata_variable_name]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )

    ### KEY-LEVEL Controls
    key_metadata = user_api_key_dict.metadata
    data = LiteLLMProxyRequestSetup.add_key_level_controls(
        key_metadata=key_metadata,
        data=data,
        _metadata_variable_name=_metadata_variable_name,
    )
    ## TEAM-LEVEL SPEND LOGS/TAGS
    team_metadata = user_api_key_dict.team_metadata or {}
    if "tags" in team_metadata and team_metadata["tags"] is not None:
        data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=data[_metadata_variable_name].get("tags"),
            tags_to_add=team_metadata["tags"],
        )
    if "disable_global_guardrails" in team_metadata and isinstance(team_metadata["disable_global_guardrails"], bool):
        data[_metadata_variable_name]["disable_global_guardrails"] = team_metadata["disable_global_guardrails"]
    if "opted_out_global_guardrails" in team_metadata and isinstance(
        team_metadata["opted_out_global_guardrails"], list
    ):
        data[_metadata_variable_name]["opted_out_global_guardrails"] = team_metadata["opted_out_global_guardrails"]
    if "spend_logs_metadata" in team_metadata and isinstance(team_metadata["spend_logs_metadata"], dict):
        if "spend_logs_metadata" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["spend_logs_metadata"], dict
        ):
            for key, value in team_metadata["spend_logs_metadata"].items():
                if (
                    key not in data[_metadata_variable_name]["spend_logs_metadata"]
                ):  # don't override k-v pair sent by request (user request)
                    data[_metadata_variable_name]["spend_logs_metadata"][key] = value
        else:
            data[_metadata_variable_name]["spend_logs_metadata"] = team_metadata["spend_logs_metadata"]

    ## PROJECT-LEVEL TAGS
    project_metadata = user_api_key_dict.project_metadata or {}
    if "tags" in project_metadata and project_metadata["tags"] is not None:
        data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=data[_metadata_variable_name].get("tags"),
            tags_to_add=project_metadata["tags"],
        )

    ## TEAM-LEVEL METADATA
    data = LiteLLMProxyRequestSetup.add_management_endpoint_metadata_to_request_metadata(
        data=data,
        management_endpoint_metadata=team_metadata,
        _metadata_variable_name=_metadata_variable_name,
    )

    # Team spend, budget - used by prometheus.py
    data[_metadata_variable_name]["user_api_key_team_max_budget"] = user_api_key_dict.team_max_budget
    data[_metadata_variable_name]["user_api_key_team_spend"] = user_api_key_dict.team_spend
    data[_metadata_variable_name]["user_api_key_request_route"] = user_api_key_dict.request_route

    # API Key spend, budget - used by prometheus.py
    data[_metadata_variable_name]["user_api_key_spend"] = user_api_key_dict.spend
    data[_metadata_variable_name]["user_api_key_max_budget"] = user_api_key_dict.max_budget
    data[_metadata_variable_name]["user_api_key_model_max_budget"] = user_api_key_dict.model_max_budget
    data[_metadata_variable_name]["user_api_key_end_user_model_max_budget"] = (
        user_api_key_dict.end_user_model_max_budget
    )

    # User spend, budget - used by prometheus.py
    # Follow same pattern as team and API key budgets
    data[_metadata_variable_name]["user_api_key_user_spend"] = user_api_key_dict.user_spend
    data[_metadata_variable_name]["user_api_key_user_max_budget"] = user_api_key_dict.user_max_budget

    data[_metadata_variable_name]["user_api_key_metadata"] = user_api_key_dict.metadata
    data[_metadata_variable_name]["user_api_key_team_metadata"] = user_api_key_dict.team_metadata
    data[_metadata_variable_name]["user_api_key_object_permission_id"] = getattr(
        user_api_key_dict, "object_permission_id", None
    )
    data[_metadata_variable_name]["user_api_key_team_object_permission_id"] = getattr(
        user_api_key_dict, "team_object_permission_id", None
    )
    data[_metadata_variable_name]["headers"] = _headers
    data[_metadata_variable_name]["endpoint"] = str(request.url)
    # Carry the proxy-receive instant via metadata (like `endpoint`) so the
    # OTel layer can compute pre-request latency, including on the failure
    # path after the logging object is popped.
    data[_metadata_variable_name]["litellm_received_at"] = getattr(request.state, "litellm_received_at", None)

    # OTEL Controls / Tracing
    # Add the OTEL Parent Trace before sending it LiteLLM
    data[_metadata_variable_name]["litellm_parent_otel_span"] = user_api_key_dict.parent_otel_span
    _add_otel_traceparent_to_data(data, request=request)

    ### END-USER SPECIFIC PARAMS ###
    if user_api_key_dict.allowed_model_region is not None:
        data["allowed_model_region"] = user_api_key_dict.allowed_model_region
    start_time = time.time()
    ## [Enterprise Only]
    # Add User-IP Address
    requester_ip_address = ""
    if True:  # Always set the IP Address if available
        # logic for tracking IP Address

        # logic for tracking IP Address
        if (
            general_settings is not None
            and general_settings.get("use_x_forwarded_for") is True
            and request is not None
            and hasattr(request, "headers")
            and "x-forwarded-for" in request.headers
        ):
            requester_ip_address = request.headers["x-forwarded-for"]
        elif (
            request is not None
            and hasattr(request, "client")
            and hasattr(request.client, "host")
            and request.client is not None
        ):
            requester_ip_address = request.client.host
    data[_metadata_variable_name]["requester_ip_address"] = requester_ip_address

    # Add User-Agent
    user_agent = ""
    if request is not None and hasattr(request, "headers") and "user-agent" in request.headers:
        user_agent = request.headers["user-agent"]
    data[_metadata_variable_name]["user_agent"] = user_agent

    if should_auto_drop_params_for_claude_code(user_agent, data, proxy_config):
        data["drop_params"] = True

    # Merge caller-supplied tags (x-litellm-tags header, data["tags"] root-level)
    # into request metadata for tag-based routing and spend attribution.
    tags = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
        llm_router=llm_router,
        headers=_headers,
        data=data,
    )

    if tags is not None:
        data[_metadata_variable_name]["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=data[_metadata_variable_name].get("tags"),
            tags_to_add=tags,
        )

    # Team Callbacks controls
    callback_settings_obj = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=proxy_config
    )
    if callback_settings_obj is not None:
        data["success_callback"] = callback_settings_obj.success_callback
        data["failure_callback"] = callback_settings_obj.failure_callback

        if callback_settings_obj.callback_vars is not None:
            # unpack callback_vars in data
            for k, v in callback_settings_obj.callback_vars.items():
                data[k] = v

    # Add disabled callbacks from key metadata
    if user_api_key_dict.metadata and "litellm_disabled_callbacks" in user_api_key_dict.metadata:
        disabled_callbacks = user_api_key_dict.metadata["litellm_disabled_callbacks"]
        if disabled_callbacks and isinstance(disabled_callbacks, list):
            data["litellm_disabled_callbacks"] = disabled_callbacks

    # Guardrails from key/team metadata and policy engine
    await move_guardrails_to_metadata(
        data=data,
        _metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    # Save pre-alias model name for credential override lookup
    _pre_alias_model = data.get("model")

    # Team Model Aliases
    _update_model_if_team_alias_exists(
        data=data,
        user_api_key_dict=user_api_key_dict,
    )

    # Key Model Aliases
    _update_model_if_key_alias_exists(
        data=data,
        user_api_key_dict=user_api_key_dict,
    )

    verbose_proxy_logger.debug("[PROXY] returned data from litellm_pre_call_utils: %s", data)

    # Team/Project credential overrides from model_config
    # Placed after the debug log to avoid leaking credential secrets in logs
    _apply_credential_overrides_from_model_config(
        data=data,
        user_api_key_dict=user_api_key_dict,
        pre_alias_model_name=_pre_alias_model,
        llm_router=llm_router,
    )

    ## ENFORCED PARAMS CHECK
    # loop through each enforced param
    # example enforced_params ['user', 'metadata', 'metadata.generation_name']
    _enforced_params_check(
        request_body=data,
        general_settings=general_settings,
        user_api_key_dict=user_api_key_dict,
        premium_user=premium_user,
    )

    end_time = time.time()
    asyncio.create_task(
        service_logger_obj.async_service_success_hook(
            service=ServiceTypes.PROXY_PRE_CALL,
            duration=end_time - start_time,
            call_type="add_litellm_data_to_request",
            start_time=start_time,
            end_time=end_time,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
    )

    return data


def _update_model_if_team_alias_exists(
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Update the model if the team alias exists

    If a alias map has been set on a team, then we want to make the request with the model the team alias is pointing to

    eg.
        - user calls `gpt-4o`
        - team.model_alias_map = {
            "gpt-4o": "gpt-4o-team-1"
        }
        - requested_model = "gpt-4o-team-1"

    Note: model_aliases for team models are deprecated. This function only applies
    to legacy non-team-scoped aliases. Team-scoped deployments use team_public_model_name
    and are resolved via map_team_model in route_llm_request.
    """
    _model = data.get("model")
    if _model and user_api_key_dict.team_model_aliases and _model in user_api_key_dict.team_model_aliases:
        from litellm.proxy.proxy_server import llm_router

        # Skip alias rewrite if this model resolves to team-specific deployments
        # (team models use team_public_model_name, not model_aliases)
        aliased_target = user_api_key_dict.team_model_aliases[_model]

        # Optional bypass for stale aliases from pre-PR deployments:
        # only enabled via feature flag to preserve backwards compatibility.
        # Cached at module level to avoid hot-path secret lookups on every request.
        global _ENABLE_TEAM_STALE_ALIAS_BYPASS
        if _ENABLE_TEAM_STALE_ALIAS_BYPASS is None:
            _ENABLE_TEAM_STALE_ALIAS_BYPASS = get_secret_bool("LITELLM_ENABLE_TEAM_STALE_ALIAS_BYPASS", False)
        enable_stale_alias_bypass = _ENABLE_TEAM_STALE_ALIAS_BYPASS
        # Check if the alias points to a team-scoped UUID name
        # (format: "model_name_{team_id}_{uuid}")
        is_stale_team_alias = aliased_target.startswith(f"model_name_{user_api_key_dict.team_id}_")
        if is_stale_team_alias and llm_router:
            # This is a stale alias from pre-PR deployments.
            # Check if current team deployments exist for the public name.
            key = (user_api_key_dict.team_id, _model)
            if key in llm_router.team_model_to_deployment_indices:
                if enable_stale_alias_bypass:
                    # Team deployments exist; skip stale alias
                    return
                warning_key = f"{user_api_key_dict.team_id}:{_model}:{aliased_target}"
                if warning_key not in _STALE_TEAM_ALIAS_WARNING_KEYS:
                    _STALE_TEAM_ALIAS_WARNING_KEYS[warning_key] = None
                    while len(_STALE_TEAM_ALIAS_WARNING_KEYS) > _MAX_STALE_ALIAS_WARNING_KEYS:
                        _STALE_TEAM_ALIAS_WARNING_KEYS.popitem(last=False)
                    verbose_proxy_logger.warning(
                        "Stale team model alias detected for model='%s', team_id='%s'. "
                        "New sibling deployments may be unreachable. "
                        "Set LITELLM_ENABLE_TEAM_STALE_ALIAS_BYPASS=true to enable "
                        "team-scoped sibling routing.",
                        _sanitize_for_log(_model),
                        user_api_key_dict.team_id,
                    )

        data["model"] = aliased_target
    return


def _update_model_if_key_alias_exists(
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Update the model if the key alias exists

    If an alias map has been set on a key, then we want to make the request with the model the key alias is pointing to

    eg.
        - user calls `modelAlias`
        - key.aliases = {
            "modelAlias": "xai/grok-4-fast-non-reasoning"
        }
        - requested_model = "xai/grok-4-fast-non-reasoning"
    """
    _model = data.get("model")
    if (
        _model
        and user_api_key_dict.aliases
        and isinstance(user_api_key_dict.aliases, dict)
        and _model in user_api_key_dict.aliases
    ):
        data["model"] = user_api_key_dict.aliases[_model]
    return


def _apply_credential_overrides_from_model_config(
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
    pre_alias_model_name: Optional[str] = None,
    llm_router: Optional[Router] = None,
) -> None:
    """
    Walk the model_config precedence chain in team/project metadata.
    If a matching credential is found, set api_base/api_key/api_version on data
    so they override deployment defaults in the router.

    Precedence (highest to lowest):
    1. Clientside credentials (already in data — skip if present)
    2. Project model-specific override
    3. Project default override (defaultconfig)
    4. Team model-specific override
    5. Team default override (defaultconfig)
    6. Deployment default (no action needed)
    """
    # Feature flag gate — disabled by default, opt in with litellm.enable_model_config_credential_overrides = True
    if not litellm.enable_model_config_credential_overrides:
        return

    # Respect clientside credentials — highest precedence
    if data.get("api_base") is not None or data.get("api_key") is not None:
        return

    model_name = data.get("model")
    if not model_name:
        return

    project_metadata = user_api_key_dict.project_metadata or {}
    team_metadata = user_api_key_dict.team_metadata or {}

    project_model_config = project_metadata.get("model_config")
    team_model_config = team_metadata.get("model_config")

    if not project_model_config and not team_model_config:
        return

    # Extract provider hint from model name (e.g. "azure/gpt-4" -> "azure").
    # When the user-facing name has no provider prefix, fall back to the
    # deployment's litellm_params so multi-provider defaultconfig entries
    # don't silently match the first dict key (#27516).
    provider: Optional[str] = None
    if "/" in model_name:
        provider = model_name.split("/", 1)[0]
    elif llm_router is not None:
        provider = _resolve_provider_from_deployment(
            llm_router=llm_router,
            model_name=model_name,
            pre_alias_model_name=pre_alias_model_name,
        )

    credential_name = _resolve_credential_from_model_config(
        model_name=model_name,
        project_model_config=project_model_config,
        team_model_config=team_model_config,
        pre_alias_model_name=pre_alias_model_name,
        provider=provider,
    )

    if not credential_name:
        return

    credential_values = CredentialAccessor.get_credential_values(credential_name)
    if not credential_values:
        _safe_cred = str(credential_name).replace("\n", "").replace("\r", "")
        verbose_proxy_logger.warning(
            "model_config references credential '%s' but it was not found or has no values",
            _safe_cred,
        )
        return

    # Apply credential overrides only for keys not already in the request
    for key in ("api_base", "api_key", "api_version"):
        if key in credential_values and key not in data:
            data[key] = credential_values[key]

    _safe_model = str(model_name).replace("\n", "").replace("\r", "")
    _safe_cred = str(credential_name).replace("\n", "").replace("\r", "")
    verbose_proxy_logger.debug(
        "Applied credential override '%s' for model '%s'",
        _safe_cred,
        _safe_model,
    )


def _resolve_provider_from_deployment(
    llm_router: Router,
    model_name: str,
    pre_alias_model_name: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a provider hint from the deployment's litellm_params when the
    user-facing model name has no provider prefix.

    Tries the post-alias name first (the resolved model group), then the
    pre-alias name. Returns None if no deployment is found or the deployment
    has no usable provider info.
    """
    candidates = [model_name]
    if pre_alias_model_name and pre_alias_model_name != model_name:
        candidates.append(pre_alias_model_name)

    for name in candidates:
        try:
            deployment = llm_router.get_deployment_by_model_group_name(model_group_name=name)
        except Exception:
            deployment = None
        if deployment is None:
            continue

        litellm_params = getattr(deployment, "litellm_params", None)
        if litellm_params is None:
            continue

        custom_provider = getattr(litellm_params, "custom_llm_provider", None)
        if custom_provider:
            return custom_provider

        deployment_model = getattr(litellm_params, "model", "") or ""
        if "/" in deployment_model:
            return deployment_model.split("/", 1)[0]

    return None


def _resolve_credential_from_model_config(
    model_name: str,
    project_model_config: Optional[dict],
    team_model_config: Optional[dict],
    pre_alias_model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[str]:
    """
    Walk the precedence chain and return the first matching credential name.

    Checks (in order):
    1. project_model_config[model_name][provider] — project model-specific
    2. project_model_config[pre_alias_model_name][provider] — project pre-alias
    3. project_model_config["defaultconfig"][provider] — project default
    4. team_model_config[model_name][provider] — team model-specific
    5. team_model_config[pre_alias_model_name][provider] — team pre-alias
    6. team_model_config["defaultconfig"][provider] — team default

    When a model-specific entry exists but contains no litellm_credentials,
    the function falls through to defaultconfig. This is intentional —
    an entry without litellm_credentials is treated as incomplete config,
    not as an explicit "no override" signal.
    """
    # Build the list of model names to try (post-alias first, then pre-alias)
    model_names_to_try = [model_name]
    if pre_alias_model_name and pre_alias_model_name != model_name:
        model_names_to_try.append(pre_alias_model_name)

    for model_config in (project_model_config, team_model_config):
        if not model_config or not isinstance(model_config, dict):
            continue

        # Model-specific check (try resolved name, then pre-alias name)
        for name in model_names_to_try:
            model_entry = model_config.get(name)
            if model_entry:
                credential_name = _extract_credential_from_entry(model_entry, provider=provider)
                if credential_name:
                    return credential_name
                _safe_name = str(name).replace("\n", "").replace("\r", "")
                verbose_proxy_logger.debug(
                    "model_config entry '%s' found but has no litellm_credentials, trying next candidate",
                    _safe_name,
                )

        # Default check
        default_entry = model_config.get("defaultconfig")
        if default_entry:
            credential_name = _extract_credential_from_entry(default_entry, provider=provider)
            if credential_name:
                return credential_name

    return None


def _extract_credential_from_entry(entry: dict, provider: Optional[str] = None) -> Optional[str]:
    """
    Extract litellm_credentials from a model_config entry.

    Entry structure: {"azure": {"litellm_credentials": "name"}, ...}

    When provider is given (e.g. "azure"), tries an exact provider match first.
    Falls back to the first credential found across all provider keys.
    """
    if not isinstance(entry, dict):
        return None

    # Prefer exact provider match when provider hint is available
    if provider and provider in entry:
        provider_config = entry[provider]
        if isinstance(provider_config, dict):
            credential_name = provider_config.get("litellm_credentials")
            if credential_name:
                return credential_name

    # Fall back to first available provider
    for provider_config in entry.values():
        if isinstance(provider_config, dict):
            credential_name = provider_config.get("litellm_credentials")
            if credential_name:
                return credential_name
    return None


def _get_enforced_params(general_settings: Optional[dict], user_api_key_dict: UserAPIKeyAuth) -> Optional[list]:
    enforced_params: Optional[list] = None
    if general_settings is not None:
        enforced_params = general_settings.get("enforced_params")
        if (
            "service_account_settings" in general_settings
            and check_if_token_is_service_account(user_api_key_dict) is True
        ):
            service_account_settings = general_settings["service_account_settings"]
            if "enforced_params" in service_account_settings:
                if enforced_params is None:
                    enforced_params = []
                enforced_params.extend(service_account_settings["enforced_params"])
    if user_api_key_dict.metadata.get("enforced_params", None) is not None:
        if enforced_params is None:
            enforced_params = []
        enforced_params.extend(user_api_key_dict.metadata["enforced_params"])
    return enforced_params


def check_if_token_is_service_account(valid_token: UserAPIKeyAuth) -> bool:
    """
    Checks if the token is a service account

    Returns:
        bool: True if token is a service account

    """
    if valid_token.metadata:
        if "service_account_id" in valid_token.metadata:
            return True
    return False


def _enforced_params_check(
    request_body: dict,
    general_settings: Optional[dict],
    user_api_key_dict: UserAPIKeyAuth,
    premium_user: bool,
) -> bool:
    """
    If enforced params are set, check if the request body contains the enforced params.
    """
    enforced_params: Optional[list] = _get_enforced_params(
        general_settings=general_settings, user_api_key_dict=user_api_key_dict
    )
    if enforced_params is None:
        return True
    if enforced_params and premium_user is not True:
        raise ValueError(
            f"Enforced Params is an Enterprise feature. Enforced Params: {enforced_params}. {CommonProxyErrors.not_premium_user.value}"
        )

    for enforced_param in enforced_params:
        _enforced_params = enforced_param.split(".")
        if len(_enforced_params) == 1:
            if _enforced_params[0] not in request_body:
                raise ValueError(
                    f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                )
        elif len(_enforced_params) == 2:
            # this is a scenario where user requires request['metadata']['generation_name'] to exist
            if _enforced_params[0] not in request_body:
                raise ValueError(
                    f"BadRequest please pass param={_enforced_params[0]} in request body. This is a required param"
                )
            if _enforced_params[1] not in request_body[_enforced_params[0]]:
                raise ValueError(
                    f"BadRequest please pass param=[{_enforced_params[0]}][{_enforced_params[1]}] in request body. This is a required param"
                )
    return True


def _add_guardrails_from_key_or_team_metadata(
    key_metadata: Optional[dict],
    team_metadata: Optional[dict],
    data: dict,
    metadata_variable_name: str,
    project_metadata: Optional[dict] = None,
) -> None:
    """
    Helper add guardrails from key, team, or project metadata to request data

    Key guardrails are set first, then team and project guardrails are appended (without duplicates).

    Args:
        key_metadata: The key metadata dictionary to check for guardrails
        team_metadata: The team metadata dictionary to check for guardrails
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data
        project_metadata: The project metadata dictionary to check for guardrails

    """
    from litellm.proxy.utils import _premium_user_check

    # Initialize guardrails set (avoiding duplicates)
    combined_guardrails = set()

    # Add key-level guardrails first
    if key_metadata and "guardrails" in key_metadata:
        if isinstance(key_metadata["guardrails"], list) and len(key_metadata["guardrails"]) > 0:
            _premium_user_check()
            combined_guardrails.update(key_metadata["guardrails"])

    # Add team-level guardrails (set automatically handles duplicates)
    if team_metadata and "guardrails" in team_metadata:
        if isinstance(team_metadata["guardrails"], list) and len(team_metadata["guardrails"]) > 0:
            _premium_user_check()
            combined_guardrails.update(team_metadata["guardrails"])

    # Add project-level guardrails (set automatically handles duplicates)
    if project_metadata and "guardrails" in project_metadata:
        if isinstance(project_metadata["guardrails"], list) and len(project_metadata["guardrails"]) > 0:
            _premium_user_check()
            combined_guardrails.update(project_metadata["guardrails"])

    # Set combined guardrails in metadata as list
    if combined_guardrails:
        data[metadata_variable_name]["guardrails"] = list(combined_guardrails)


def _add_guardrails_from_policies_in_metadata(
    key_metadata: Optional[dict],
    team_metadata: Optional[dict],
    data: dict,
    metadata_variable_name: str,
    project_metadata: Optional[dict] = None,
) -> None:
    """
    Helper to resolve guardrails from policies attached to key/team/project metadata.

    This function:
    1. Gets policy names from key, team, and project metadata
    2. Resolves guardrails from those policies (including inheritance)
    3. Adds resolved guardrails to request metadata

    Args:
        key_metadata: The key metadata dictionary to check for policies
        team_metadata: The team metadata dictionary to check for policies
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data
        project_metadata: The project metadata dictionary to check for policies
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
    from litellm.proxy.utils import _premium_user_check
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    # Collect policy names from key and team metadata
    policy_names: set = set()

    # Add key-level policies first
    if key_metadata and "policies" in key_metadata:
        if isinstance(key_metadata["policies"], list) and len(key_metadata["policies"]) > 0:
            _premium_user_check()
            policy_names.update(key_metadata["policies"])

    # Add team-level policies
    if team_metadata and "policies" in team_metadata:
        if isinstance(team_metadata["policies"], list) and len(team_metadata["policies"]) > 0:
            _premium_user_check()
            policy_names.update(team_metadata["policies"])

    # Add project-level policies
    if project_metadata and "policies" in project_metadata:
        if isinstance(project_metadata["policies"], list) and len(project_metadata["policies"]) > 0:
            _premium_user_check()
            policy_names.update(project_metadata["policies"])

    if not policy_names:
        return

    verbose_proxy_logger.debug(f"Policy engine: resolving guardrails from key/team policies: {policy_names}")

    # Check if policy registry is initialized
    registry = get_policy_registry()
    if not registry.is_initialized():
        verbose_proxy_logger.debug("Policy engine not initialized, skipping policy resolution from metadata")
        return

    # Build context for policy resolution (model from request data)
    context = PolicyMatchContext(model=data.get("model"))

    # Get all policies from registry
    all_policies = registry.get_all_policies()

    # Resolve guardrails from the specified policies
    resolved_guardrails: set = set()
    for policy_name in policy_names:
        if registry.has_policy(policy_name):
            resolved_policy = PolicyResolver.resolve_policy_guardrails(
                policy_name=policy_name,
                policies=all_policies,
                context=context,
            )
            resolved_guardrails.update(resolved_policy.guardrails)
            verbose_proxy_logger.debug(
                f"Policy engine: resolved guardrails from policy '{policy_name}': {resolved_policy.guardrails}"
            )
        else:
            verbose_proxy_logger.warning(f"Policy engine: policy '{policy_name}' not found in registry")

    if not resolved_guardrails:
        return

    # Add resolved guardrails to request metadata
    if metadata_variable_name not in data:
        data[metadata_variable_name] = {}

    existing_guardrails = data[metadata_variable_name].get("guardrails", [])
    if not isinstance(existing_guardrails, list):
        existing_guardrails = []

    # Combine existing guardrails with policy-resolved guardrails (no duplicates)
    combined = set(existing_guardrails)
    combined.update(resolved_guardrails)
    data[metadata_variable_name]["guardrails"] = list(combined)

    # Store applied policies in metadata for tracking
    if "applied_policies" not in data[metadata_variable_name]:
        data[metadata_variable_name]["applied_policies"] = []
    data[metadata_variable_name]["applied_policies"].extend(list(policy_names))

    verbose_proxy_logger.debug(
        f"Policy engine: added guardrails from key/team policies to request metadata: {list(resolved_guardrails)}"
    )


async def move_guardrails_to_metadata(
    data: dict,
    _metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Helper to add guardrails from request to metadata

    - If guardrails set on API Key metadata then sets guardrails on request metadata
    - If guardrails not set on API key, then checks request metadata
    - Adds guardrails from policies attached to key/team metadata
    - Adds guardrails from policy engine based on team/key/model context
    """
    # Early-out: skip all guardrails processing when nothing is configured
    key_metadata = user_api_key_dict.metadata
    team_metadata = user_api_key_dict.team_metadata
    project_metadata = user_api_key_dict.project_metadata or {}

    has_key_config = key_metadata and ("guardrails" in key_metadata or "policies" in key_metadata)
    has_team_config = team_metadata and ("guardrails" in team_metadata or "policies" in team_metadata)
    has_project_config = project_metadata and ("guardrails" in project_metadata or "policies" in project_metadata)
    has_request_config = "guardrails" in data or "guardrail_config" in data or "policies" in data

    # Only check policy engine if no local config (avoid import + registry lookup)
    if not (has_key_config or has_team_config or has_project_config or has_request_config):
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        if not get_policy_registry().is_initialized():
            # Nothing configured anywhere - clean up request body fields and return
            data.pop("policies", None)
            return

    # Check key/team/project-level guardrails
    _add_guardrails_from_key_or_team_metadata(
        key_metadata=user_api_key_dict.metadata,
        team_metadata=user_api_key_dict.team_metadata,
        project_metadata=project_metadata,
        data=data,
        metadata_variable_name=_metadata_variable_name,
    )

    #########################################################################################
    # Add guardrails from policies attached to key/team/project metadata
    #########################################################################################
    _add_guardrails_from_policies_in_metadata(
        key_metadata=user_api_key_dict.metadata,
        team_metadata=user_api_key_dict.team_metadata,
        project_metadata=project_metadata,
        data=data,
        metadata_variable_name=_metadata_variable_name,
    )

    #########################################################################################
    # Add guardrails from policy engine based on team/key/model context
    #########################################################################################
    await add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name=_metadata_variable_name,
        user_api_key_dict=user_api_key_dict,
    )

    #########################################################################################
    # User's might send "guardrails" in the request body, we need to add them to the request metadata.
    # Since downstream logic requires "guardrails" to be in the request metadata
    #########################################################################################
    if "guardrails" in data:
        request_body_guardrails = data.pop("guardrails")
        if "guardrails" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["guardrails"], list
        ):
            data[_metadata_variable_name]["guardrails"].extend(request_body_guardrails)
        else:
            data[_metadata_variable_name]["guardrails"] = request_body_guardrails

    #########################################################################################
    if "guardrail_config" in data:
        request_body_guardrail_config = data.pop("guardrail_config")
        if "guardrail_config" in data[_metadata_variable_name] and isinstance(
            data[_metadata_variable_name]["guardrail_config"], dict
        ):
            data[_metadata_variable_name]["guardrail_config"].update(request_body_guardrail_config)
        else:
            data[_metadata_variable_name]["guardrail_config"] = request_body_guardrail_config


def _is_policy_version_id(s: str) -> bool:
    """Return True if string is a policy version ID (starts with policy_<uuid> prefix)."""
    from litellm.proxy.policy_engine.policy_registry import POLICY_VERSION_ID_PREFIX

    return isinstance(s, str) and s.startswith(POLICY_VERSION_ID_PREFIX)


def _extract_policy_id(s: str) -> Optional[str]:
    """Extract raw UUID from policy_<uuid> string, or None if not a valid version ID."""
    from litellm.proxy.policy_engine.policy_registry import POLICY_VERSION_ID_PREFIX

    if not _is_policy_version_id(s):
        return None
    return s[len(POLICY_VERSION_ID_PREFIX) :].strip() or None


def _match_and_track_policies(
    data: dict,
    context: "PolicyMatchContext",
    request_body_policies: Any,
    policies_override: Optional[Dict[str, Any]] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Match policies via attachments and request body, track them in metadata.

    Returns:
        Tuple of (applied_policy_names, policy_reasons)
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.common_utils.callback_utils import (
        add_policy_sources_to_metadata,
        add_policy_to_applied_policies_header,
    )
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher

    # Get matching policies via attachments (with match reasons for attribution)
    attachment_registry = get_attachment_registry()
    matches_with_reasons = attachment_registry.get_attached_policies_with_reasons(context)
    matching_policy_names = [m["policy_name"] for m in matches_with_reasons]
    policy_reasons = {m["policy_name"]: m["matched_via"] for m in matches_with_reasons}

    verbose_proxy_logger.debug(f"Policy engine: matched policies via attachments: {matching_policy_names}")

    # Combine attachment-based policies with dynamic request body policies
    all_policy_names = set(matching_policy_names)
    if request_body_policies and isinstance(request_body_policies, list):
        all_policy_names.update(request_body_policies)
        verbose_proxy_logger.debug(f"Policy engine: added dynamic policies from request body: {request_body_policies}")

    if not all_policy_names:
        return [], {}

    # Filter to only policies whose conditions match the context
    applied_policy_names = PolicyMatcher.get_policies_with_matching_conditions(
        policy_names=list(all_policy_names),
        context=context,
        policies=policies_override,
    )

    verbose_proxy_logger.debug(f"Policy engine: applied policies (conditions matched): {applied_policy_names}")

    # Track applied policies in metadata for response headers
    for policy_name in applied_policy_names:
        add_policy_to_applied_policies_header(request_data=data, policy_name=policy_name)

    # Track policy attribution sources for x-litellm-policy-sources header
    applied_reasons = {name: policy_reasons[name] for name in applied_policy_names if name in policy_reasons}
    add_policy_sources_to_metadata(request_data=data, policy_sources=applied_reasons)

    return applied_policy_names, policy_reasons


def _apply_resolved_guardrails_to_metadata(
    data: dict,
    metadata_variable_name: str,
    context: "PolicyMatchContext",
    policy_names: Optional[List[str]] = None,
    policies: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply resolved guardrails and pipelines to request metadata."""
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    # Resolve guardrails from matching policies
    resolved_guardrails = PolicyResolver.resolve_guardrails_for_context(
        context=context,
        policies=policies,
        policy_names=policy_names,
    )

    verbose_proxy_logger.debug(f"Policy engine: resolved guardrails: {resolved_guardrails}")

    # Resolve pipelines from matching policies
    pipelines = PolicyResolver.resolve_pipelines_for_context(
        context=context,
        policies=policies,
        policy_names=policy_names,
    )

    # Add resolved guardrails to request metadata
    if metadata_variable_name not in data:
        data[metadata_variable_name] = {}

    # Track pipeline-managed guardrails to exclude from independent execution
    pipeline_managed_guardrails: set = set()
    if pipelines:
        pipeline_managed_guardrails = PolicyResolver.get_pipeline_managed_guardrails(pipelines)
        data[metadata_variable_name]["_guardrail_pipelines"] = pipelines
        data[metadata_variable_name]["_pipeline_managed_guardrails"] = pipeline_managed_guardrails
        verbose_proxy_logger.debug(
            f"Policy engine: resolved {len(pipelines)} pipeline(s), managed guardrails: {pipeline_managed_guardrails}"
        )

    if not resolved_guardrails and not pipelines:
        return

    existing_guardrails = data[metadata_variable_name].get("guardrails", [])
    if not isinstance(existing_guardrails, list):
        existing_guardrails = []

    # Combine existing guardrails with policy-resolved guardrails (no duplicates)
    # Exclude pipeline-managed guardrails from the flat list
    combined = set(existing_guardrails)
    combined.update(resolved_guardrails)
    combined -= pipeline_managed_guardrails
    data[metadata_variable_name]["guardrails"] = list(combined)

    verbose_proxy_logger.debug(f"Policy engine: added guardrails to request metadata: {list(combined)}")


async def add_guardrails_from_policy_engine(
    data: dict,
    metadata_variable_name: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Add guardrails from the policy engine based on request context.

    This function:
    1. Extracts "policies" from request body (if present) for dynamic policy application
    2. Supports policy_<uuid> in policies to execute a specific version (e.g. published)
    3. Gets matching policies based on team_alias, key_alias, and model (via attachments)
    4. Combines dynamic policies with attachment-based policies
    5. Resolves guardrails from all policies (including inheritance)
    6. Adds guardrails to request metadata
    7. Tracks applied policies in metadata for response headers
    8. Removes "policies" from request body so it's not forwarded to LLM provider

    Args:
        data: The request data to update
        metadata_variable_name: The name of the metadata field in data
        user_api_key_dict: The user's API key authentication info
    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.types.proxy.policy_engine import PolicyMatchContext

    # Extract dynamic policies from request body (if present)
    request_body_policies_raw = data.pop("policies", None)

    registry = get_policy_registry()
    verbose_proxy_logger.debug(
        f"Policy engine: registry initialized={registry.is_initialized()}, "
        f"policy_count={len(registry.get_all_policies())}"
    )
    if not registry.is_initialized():
        verbose_proxy_logger.debug("Policy engine not initialized, skipping policy matching")
        return

    # Extract tags and build context
    all_tags = get_tags_from_request_body(data) or None
    _team_alias = user_api_key_dict.team_alias
    _key_alias = user_api_key_dict.key_alias
    context = PolicyMatchContext(
        team_alias=_team_alias if isinstance(_team_alias, str) else None,
        key_alias=_key_alias if isinstance(_key_alias, str) else None,
        model=data.get("model"),
        tags=all_tags,
    )

    verbose_proxy_logger.debug(
        f"Policy engine: matching policies for context team_alias={context.team_alias}, "
        f"key_alias={context.key_alias}, model={context.model}, tags={context.tags}"
    )

    # Separate policy names from policy version IDs (policy_<uuid>)
    request_body_names: List[str] = []
    request_body_version_ids: List[str] = []
    if request_body_policies_raw and isinstance(request_body_policies_raw, list):
        for item in request_body_policies_raw:
            if not isinstance(item, str):
                continue
            if _is_policy_version_id(item):
                policy_id = _extract_policy_id(item)
                if policy_id:
                    request_body_version_ids.append(policy_id)
            else:
                request_body_names.append(item)

    # Resolve policy versions by ID from in-memory cache (populated by sync job; no DB in hot path)
    merged_policies: Dict[str, Any] = dict(registry.get_all_policies())
    fetched_policy_names: List[str] = []
    for policy_id in request_body_version_ids:
        result = registry.get_policy_by_id_for_request(policy_id=policy_id)
        if result is not None:
            pname, policy = result
            merged_policies[pname] = policy
            fetched_policy_names.append(pname)
            verbose_proxy_logger.debug(f"Policy engine: loaded version by ID policy_{policy_id} -> {pname}")
        else:
            verbose_proxy_logger.debug(f"Policy engine: policy version {policy_id} not found in cache, skipping")

    # Build request body list: names + policy names from fetched versions
    request_body_policies = request_body_names + fetched_policy_names

    # Match and track policies (with merged_policies when we have version overrides)
    applied_policy_names, _ = _match_and_track_policies(
        data,
        context,
        request_body_policies,
        policies_override=merged_policies if request_body_version_ids else None,
    )

    # Resolve and apply guardrails. Use applied_policy_names so request-body policies
    # (names + version IDs) are included. Use merged_policies when we have version overrides.
    _apply_resolved_guardrails_to_metadata(
        data,
        metadata_variable_name,
        context,
        policy_names=applied_policy_names if applied_policy_names else None,
        policies=merged_policies if request_body_version_ids else None,
    )


def add_provider_specific_headers_to_request(
    data: dict,
    headers: dict,
):
    from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key

    anthropic_headers = {}
    # boolean to indicate if a header was added
    added_header = False
    for header in ANTHROPIC_API_HEADERS:
        if header in headers:
            header_value = headers[header]
            anthropic_headers[header] = header_value
            added_header = True

    # Check for Authorization header with Anthropic OAuth token (sk-ant-oat*)
    # This needs to be handled via provider-specific headers to ensure it only
    # goes to Anthropic-compatible providers, not all providers in the router
    for header, value in headers.items():
        if header.lower() == "authorization" and is_anthropic_oauth_key(value):
            anthropic_headers[header] = value
            added_header = True
            break
    if added_header is True:
        # Anthropic headers work across multiple providers
        # Store as comma-separated list so retrieval can match any of them
        data["provider_specific_header"] = ProviderSpecificHeader(
            custom_llm_provider=f"{LlmProviders.ANTHROPIC.value},{LlmProviders.BEDROCK.value},{LlmProviders.VERTEX_AI.value}",
            extra_headers=anthropic_headers,
        )

    return


def _add_otel_traceparent_to_data(data: dict, request: Request):
    from litellm.proxy.proxy_server import open_telemetry_logger

    if data is None:
        return
    if open_telemetry_logger is None:
        # if user is not use OTEL don't send extra_headers
        # relevant issue: https://github.com/BerriAI/litellm/issues/4448
        return

    if litellm.forward_traceparent_to_llm_provider is True:
        if request.headers:
            if "traceparent" in request.headers:
                # we want to forward this to the LLM Provider
                # Relevant issue: https://github.com/BerriAI/litellm/issues/4419
                # pass this in extra_headers
                if "extra_headers" not in data:
                    data["extra_headers"] = {}
                _exra_headers = data["extra_headers"]
                if "traceparent" not in _exra_headers:
                    _exra_headers["traceparent"] = request.headers["traceparent"]
