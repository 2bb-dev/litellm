"""The opt-in protected profile admits only content-safe persistence paths."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MethodType
from typing import Literal

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.request_content_mode import encryption_enabled
from litellm.proxy.spend_tracking.request_content_encryption import encryption_readiness

_FIELDS: TypeAdapter[Mapping[str, object]] = TypeAdapter(Mapping[str, object])
_ROWS: TypeAdapter[tuple[Mapping[str, object], ...]] = TypeAdapter(tuple[Mapping[str, object], ...])
_CALLBACKS: TypeAdapter[tuple[object, ...]] = TypeAdapter(tuple[object, ...])
_BUDGET_QUERY: TypeAdapter[Callable[[str], Awaitable[object]]] = TypeAdapter(Callable[[str], Awaitable[object]])
_CALLBACK_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
)
_COLLECTOR = "litellm.proxy.hooks.proxy_track_cost_callback._ProxyDBLogger"
_APPROVED_CALLBACKS = frozenset(
    {
        _COLLECTOR,
        "callbacks.request_context.OpenOrangeRequestContextCallback",
        "litellm._service_logger.ServiceLogging",
        "litellm.proxy.hooks.model_max_budget_limiter._PROXY_VirtualKeyModelMaxBudgetLimiter",
        "litellm.proxy.hooks.max_budget_limiter._PROXY_MaxBudgetLimiter",
        "litellm.proxy.hooks.parallel_request_limiter._PROXY_MaxParallelRequestsHandler",
        "litellm.proxy.hooks.parallel_request_limiter_v3._PROXY_MaxParallelRequestsHandler_v3",
        "litellm.proxy.hooks.cache_control_check._PROXY_CacheControlCheck",
        "litellm.proxy.hooks.responses_id_security.ResponsesIDSecurity",
        "litellm.proxy.hooks.litellm_skills.main.SkillsInjectionHook",
        "litellm.proxy.hooks.max_iterations_limiter._PROXY_MaxIterationsHandler",
        "litellm.proxy.hooks.max_budget_per_session_limiter._PROXY_MaxBudgetPerSessionHandler",
        "litellm.proxy.hooks.sensitive_data_routing._PROXY_SensitiveDataRoutingHandler",
        "litellm.router_strategy.lowest_tpm_rpm.LowestTPMLoggingHandler",
        "litellm.router_strategy.lowest_tpm_rpm_v2.LowestTPMLoggingHandler_v2",
        "litellm.router_strategy.lowest_latency.LowestLatencyLoggingHandler",
        "litellm.router_strategy.lowest_cost.LowestCostLoggingHandler",
        "litellm.router_strategy.least_busy.LeastBusyLoggingHandler",
        "litellm.router_utils.pre_call_checks.encrypted_content_affinity_check.EncryptedContentAffinityCheck",
    }
)
_ROUTER_METHODS = frozenset(
    {
        "deployment_callback_on_success",
        "sync_deployment_callback_on_success",
        "deployment_callback_on_failure",
        "async_deployment_callback_on_failure",
    }
)
_DYNAMIC_OPTIONS = frozenset(
    {
        *_CALLBACK_LISTS,
        "logger_fn",
        "logging_callback",
        "additional_success_callbacks",
        "additional_failure_callbacks",
        "additional_input_callbacks",
        "guardrails",
        "prompt_id",
        "turn_off_message_logging",
        "standard_logging_payload_excluded_fields",
        "redact_messages",
        "log_raw_request_response",
        "disable_streaming_logging",
        "cache",
        "caching",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileFailure:
    code: Literal[
        "encryption_unavailable",
        "collector_unavailable",
        "retention_conflict",
        "unsupported_callback",
        "unsupported_cache",
        "unsupported_diagnostics",
        "request_override",
        "tag_budget_unsupported",
    ]


class ProtectedProfileUnavailable(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=503, detail="Encrypted request logging is unavailable")


def _class_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def approved_callback(callback: object) -> bool:
    if isinstance(callback, MethodType):
        return _class_name(callback.__self__) == "litellm.router.Router" and callback.__name__ in _ROUTER_METHODS
    return _class_name(callback) in _APPROVED_CALLBACKS


def _fields(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return {}


def request_policy_failure(data: object) -> ProfileFailure | None:
    fields = _fields(data)
    for source in (fields, _fields(fields.get("litellm_params")), _fields(fields.get("router_settings_override"))):
        if any(source.get(name) not in (None, False, [], {}) for name in _DYNAMIC_OPTIONS):
            return ProfileFailure("request_override")
    return None


def protected_profile_failure() -> ProfileFailure | None:
    if not encryption_enabled():
        return None
    if encryption_readiness() is not None:
        return ProfileFailure("encryption_unavailable")
    from litellm.litellm_core_utils import litellm_logging
    from litellm.proxy import proxy_server

    runtime = _FIELDS.validate_python(vars(litellm))
    server = _FIELDS.validate_python(vars(proxy_server))
    settings = _fields(server.get("general_settings"))
    try:
        registries = {name: _CALLBACKS.validate_python(runtime.get(name, ())) for name in _CALLBACK_LISTS}
    except ValidationError:
        return ProfileFailure("unsupported_callback")
    collectors = tuple(
        callback for callback in registries["_async_success_callback"] if _class_name(callback) == _COLLECTOR
    )
    if (
        proxy_server.prisma_client is None
        or getattr(proxy_server.prisma_client, "_spend_log_spool", None) is None
        or not collectors
        or not any(_class_name(callback) == _COLLECTOR for callback in registries["callbacks"])
        or server.get("disable_spend_logs")
        or any(
            settings.get(name) is True for name in ("disable_spend_logs", "disable_spend_updates", "disable_error_logs")
        )
    ):
        return ProfileFailure("collector_unavailable")
    if (
        settings.get("store_prompts_in_spend_logs") is not True
        or litellm.turn_off_message_logging
        or litellm.standard_logging_payload_excluded_fields
        or litellm.disable_streaming_logging
        or not litellm.logging
        or any(getattr(callback, "message_logging", True) is not True for callback in collectors)
        or any(getattr(callback, "turn_off_message_logging", False) is not False for callback in collectors)
    ):
        return ProfileFailure("retention_conflict")
    if any(not approved_callback(callback) for callbacks in registries.values() for callback in callbacks):
        return ProfileFailure("unsupported_callback")
    if runtime.get("pre_call_rules") or runtime.get("post_call_rules"):
        return ProfileFailure("unsupported_callback")
    if litellm.cache is not None or getattr(proxy_server.llm_router, "cache_responses", False):
        return ProfileFailure("unsupported_cache")
    if (
        litellm.set_verbose
        or litellm.log_raw_request_response
        or litellm_logging.user_logger_fn is not None
        or getattr(litellm_logging, "capture_exception", None) is not None
        or getattr(proxy_server.proxy_logging_obj, "alerting", None)
        or proxy_server.open_telemetry_logger is not None
        or any(
            os.getenv(name, "").lower() in {"true", "1"}
            for name in ("DD_TRACE_ENABLED", "USE_DDTRACE", "USE_DDPROFILER")
        )
        or any(
            logging.getLogger(name).isEnabledFor(logging.DEBUG)
            for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
        )
    ):
        return ProfileFailure("unsupported_diagnostics")
    return None


def enforce_protected_profile(data: object = None) -> None:
    if not encryption_enabled():
        return
    if protected_profile_failure() is not None or request_policy_failure(data) is not None:
        raise ProtectedProfileUnavailable()


async def protected_tag_budget_failure() -> ProfileFailure | None:
    if not encryption_enabled():
        return None
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return ProfileFailure("collector_unavailable")
    try:
        query_raw = _BUDGET_QUERY.validate_python(getattr(prisma_client.writer_db, "query_raw", None))
        rows = _ROWS.validate_python(
            await asyncio.wait_for(
                query_raw(
                    'SELECT EXISTS (SELECT 1 FROM "LiteLLM_TagTable" t '
                    'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = t.budget_id '
                    "WHERE b.max_budget IS NOT NULL) AS configured"
                ),
                timeout=1.0,
            )
        )
    except Exception:
        return ProfileFailure("collector_unavailable")
    if len(rows) != 1 or type(rows[0].get("configured")) is not bool:
        return ProfileFailure("collector_unavailable")
    return ProfileFailure("tag_budget_unsupported") if rows[0]["configured"] else None


async def enforce_protected_tag_budget_profile() -> None:
    if await protected_tag_budget_failure() is not None:
        raise ProtectedProfileUnavailable()
