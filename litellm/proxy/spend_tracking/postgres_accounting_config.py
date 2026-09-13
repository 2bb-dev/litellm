from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

_OBJECT = TypeAdapter(dict[str, JsonValue])


def _only(data: dict[str, JsonValue], allowed: set[str], area: str) -> None:
    if data.keys() - allowed:
        raise ValueError(
            f"PostgreSQL accounting first vertical: {area} integration pending: {sorted(data.keys() - allowed)}"
        )


def _reject_features(data: dict[str, JsonValue], unsupported: set[str], area: str) -> None:
    enabled = sorted(name for name in unsupported if data.get(name))
    if enabled:
        raise ValueError(f"PostgreSQL accounting: {area} integration pending: {enabled}")


def validate_config(value: object) -> None:
    config = _OBJECT.validate_python(value)
    _only(config, {"model_list", "general_settings", "router_settings", "litellm_settings"}, "config")
    general = _OBJECT.validate_python(config.get("general_settings", {}))
    _reject_features(
        general,
        {
            "custom_auth",
            "max_parallel_requests",
            "global_max_parallel_requests",
            "pass_through_endpoints",
            "background_health_checks",
            "disable_spend_logs",
            "disable_spend_updates",
            "disable_budget_reservation",
            "use_redis_transaction_buffer",
            "database_type",
            "user_header_name",
            "user_header_mappings",
        },
        "general settings",
    )
    sdk = _OBJECT.validate_python(config.get("litellm_settings", {}))
    _reject_features(
        sdk,
        {
            "max_budget",
            "success_callback",
            "failure_callback",
            "post_call_rules",
            "pre_call_rules",
            "fallbacks",
            "context_window_fallbacks",
            "aclient_session",
        },
        "SDK execution/accounting",
    )
    if sdk.get("callbacks") not in (None, ["callbacks.request_context.handler"]):
        raise ValueError("PostgreSQL accounting: callback integration pending")
    if sdk.get("cache_params") not in (None, {"type": "local"}):
        raise ValueError("PostgreSQL accounting first vertical: external cache integration pending")
    router = _OBJECT.validate_python(config.get("router_settings", {}))
    _reject_features(
        router,
        {
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "model_group_retry_policy",
            "default_litellm_params",
            "redis_url",
            "redis_host",
            "routing_groups",
            "model_group_alias",
            "model_group_affinity_config",
        },
        "router state/execution",
    )
    checks = router.get("optional_pre_call_checks") or []
    if not isinstance(checks, list) or any(check != "encrypted_content_affinity" for check in checks):
        raise ValueError("PostgreSQL accounting: optional router state integration pending")
    if router.get("routing_strategy", "simple-shuffle") not in {"simple-shuffle", "usage-based-routing"}:
        raise ValueError("PostgreSQL accounting: retry/routing policy not qualified")
    models = config.get("model_list") or []
    if not isinstance(models, list):
        raise ValueError("PostgreSQL accounting requires native model deployments")
    for model in models:
        validate_deployment(model)


def validate_deployment(value: object, *, partial: bool = False) -> None:
    model = _OBJECT.validate_python(value)
    params = _OBJECT.validate_python(model.get("litellm_params") or {})
    info = _OBJECT.validate_python(model.get("model_info") or {})
    if not partial or "model" in params:
        if not str(params.get("model", "")).startswith(("openai/", "litellm_proxy/")):
            raise ValueError("PostgreSQL accounting: provider dispatch integration pending")
    unsupported = {
        "tpm",
        "rpm",
        "max_parallel_requests",
        "max_budget",
        "budget_duration",
        "budget_reset_at",
        "team_id",
        "litellm_credential_name",
        "mock_response",
        "client",
        "fallbacks",
        "context_window_fallbacks",
    }
    if any(settings.get(name) is not None for settings in (params, info) for name in unsupported):
        raise ValueError("PostgreSQL accounting: deployment scope/limiter/client integration pending")


def validate_request(value: object) -> None:
    request = _OBJECT.validate_python(value)
    unsupported = {
        "api_key",
        "api_base",
        "base_url",
        "client",
        "custom_llm_provider",
        "litellm_credential_name",
        "mock_response",
        "mock_tool_calls",
        "fallbacks",
        "context_window_fallbacks",
        "routing_strategy",
        "router_settings",
        "litellm_params",
        "litellm_metadata",
        "extra_body",
        "extra_headers",
        "user",
        "tags",
        "budget_id",
        "guardrails",
        "background",
        "previous_response_id",
        "conversation",
        "complete_response",
        "num_retries",
        "caching",
        "cache",
    }
    if any(request.get(name) is not None and request.get(name) is not False for name in unsupported):
        raise HTTPException(503, "PostgreSQL accounting: execution override/scope integration pending")
    metadata = _OBJECT.validate_python(request.get("metadata") or {})
    if any(metadata.get(name) for name in ("tags", "budget_id", "guardrails", "model_group", "model_info")):
        raise HTTPException(503, "PostgreSQL accounting: request metadata scope/routing integration pending")
    tools = request.get("tools")
    if isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("type") not in {"function", "custom"} for tool in tools
    ):
        raise HTTPException(503, "PostgreSQL accounting: server-side tool execution ownership pending")
