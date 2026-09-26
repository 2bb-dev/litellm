"""Protected admission, readiness and content-sink regressions using synthetic data."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Response

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.request_content_mode import (
    CONFIG_ENV,
    encryption_enabled,
    protection_marker_path,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.spend_log_queue import SQLiteSpendLogSpool
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger, _get_request_tags_for_cost_tracking
from litellm.proxy.spend_tracking.request_content_encryption import CaptureFailure, INSTANCE_ENV, configured_encryptor
from litellm.proxy.spend_tracking.request_content_metadata import safe_metadata
from litellm.proxy.spend_tracking.request_content_policy import (
    ProfileFailure,
    ProtectedProfileUnavailable,
    approved_callback,
    enforce_protected_profile,
    protected_profile_failure,
    protected_tag_budget_failure,
    request_policy_failure,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/browser_jwe_test_only.json").read_text())
CANARY = "BILLING_PRIVATE_CANARY_1202 with spaces"
CALLBACK_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
)


@pytest.fixture
def protected_runtime(monkeypatch, tmp_path):
    from litellm.litellm_core_utils import litellm_logging
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import request_content_encryption

    config = {"version": 1, "instanceUid": FIXTURE["instanceUid"], "publicKey": FIXTURE["publicKey"]}
    config_path = tmp_path / "public.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv(CONFIG_ENV, str(config_path))
    monkeypatch.setenv(INSTANCE_ENV, FIXTURE["instanceUid"])
    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "spend.sqlite"))
    for name in ("DD_TRACE_ENABLED", "USE_DDTRACE", "USE_DDPROFILER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(request_content_encryption, "_runtime_failure", None)
    spool = SQLiteSpendLogSpool(str(tmp_path / "spend.sqlite"))
    client = SimpleNamespace(
        _spend_log_spool=spool,
        _spend_log_transactions_lock=asyncio.Lock(),
        spend_log_transactions=[],
        db=SimpleNamespace(query_raw=AsyncMock(return_value=[{"configured": False}])),
    )
    client.writer_db = PrismaWrapper(client.db, iam_token_db_auth=False)
    collector = _ProxyDBLogger()
    for name in CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [collector])
    monkeypatch.setattr(litellm, "_async_success_callback", [collector])
    for name in ("turn_off_message_logging", "disable_streaming_logging", "set_verbose", "log_raw_request_response"):
        monkeypatch.setattr(litellm, name, False)
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", None)
    monkeypatch.setattr(litellm, "logging", True)
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "pre_call_rules", [])
    monkeypatch.setattr(litellm, "post_call_rules", [])
    monkeypatch.setattr(litellm_logging, "user_logger_fn", None)
    monkeypatch.setattr(litellm_logging, "capture_exception", None)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "alerting", None)
    previous_levels = tuple(
        (logging.getLogger(name), logging.getLogger(name).level)
        for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
    )
    for logger, _ in previous_levels:
        logger.setLevel(logging.WARNING)
    yield SimpleNamespace(client=client, collector=collector, config=config, config_path=config_path, spool=spool)
    for logger, level in previous_levels:
        logger.setLevel(level)


@pytest.mark.asyncio
async def test_ready_profile_preserves_request_and_installs_identity_marker(protected_runtime):
    from litellm.proxy import proxy_server

    data = {"model": "test-model", "messages": [{"role": "user", "content": CANARY}], "no-log": True}
    assert protected_profile_failure() is None
    assert await protected_tag_budget_failure() is None
    assert await proxy_server.proxy_logging_obj.pre_call_hook(UserAPIKeyAuth(), data, "acompletion") is data
    marker = protection_marker_path()
    assert marker is not None and marker.is_file()
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.read_text().splitlines() == [
        "openorange.request-log.v1",
        FIXTURE["instanceUid"],
        FIXTURE["publicKey"]["kid"],
    ]
    assert data["messages"][0]["content"] == CANARY


@pytest.mark.parametrize("protected", [True, False])
def test_actual_proxy_and_usage_router_callbacks_fit_the_protected_profile(protected_runtime, monkeypatch, protected):
    from litellm.integrations.shadow_eval_logger import ShadowEvalLogger
    from litellm.proxy import proxy_server

    if not protected:
        monkeypatch.delenv(CONFIG_ENV)
    assert encryption_enabled() is protected
    monkeypatch.setenv("LITELLM_COLLECTOR_ENABLED", "false")
    monkeypatch.setattr(proxy_server, "spend_event_producer", None)
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    proxy_server.cost_tracking()
    router = litellm.Router(
        model_list=[{"model_name": "test", "litellm_params": {"model": "openai/test-model", "api_key": "synthetic"}}],
        routing_strategy="usage-based-routing",
        enable_pre_call_checks=True,
        optional_pre_call_checks=["encrypted_content_affinity"],
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    proxy_server.proxy_logging_obj._init_litellm_callbacks(llm_router=router)
    assert sum(isinstance(callback, _ProxyDBLogger) for callback in litellm.callbacks) == 1
    assert sum(isinstance(callback, _ProxyDBLogger) for callback in litellm._async_success_callback) == 1
    assert sum(isinstance(callback, ShadowEvalLogger) for callback in litellm.callbacks) == int(not protected)
    assert protected_profile_failure() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("request_context", "OpenOrangeRequestContextCallback"),
        ("central_call_id", "CentralCallIdCallback"),
    ],
)
async def test_config_refresh_preserves_initialized_callback_identity(
    protected_runtime, monkeypatch, tmp_path, module_name, class_name
):
    from litellm.proxy import proxy_server

    callbacks = tmp_path / "callbacks"
    callbacks.mkdir()
    (callbacks / f"{module_name}.py").write_text(
        "from litellm.integrations.custom_logger import CustomLogger\n"
        f"class {class_name}(CustomLogger):\n"
        "    pass\n"
        f"handler = {class_name}()\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  store_prompts_in_spend_logs: true\n"
        "litellm_settings:\n"
        f"  callbacks: [callbacks.{module_name}.handler]\n"
    )
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    pc = proxy_server.ProxyConfig()
    with monkeypatch.context() as startup:
        startup.setattr(proxy_server, "prisma_client", None)
        await pc.load_config(router=None, config_file_path=str(config))
    proxy_server.proxy_logging_obj._init_litellm_callbacks()
    initialized = next(
        callback for callback in litellm.callbacks if type(callback).__module__ == f"callbacks.{module_name}"
    )
    assert protected_profile_failure() is None

    (callbacks / f"{module_name}.py").unlink()
    for _ in range(2):
        pc._add_callbacks_from_db_config({"litellm_settings": {"callbacks": [f"callbacks.{module_name}.handler"]}})
        assert protected_profile_failure() is None
        assert sum(callback is initialized for callback in litellm.callbacks) == 1
        assert not any(isinstance(callback, str) for callback in litellm.callbacks)

    pc._add_callbacks_from_db_config({"litellm_settings": {"callbacks": [f"callbacks.{module_name}.other_handler"]}})
    assert protected_profile_failure() == ProfileFailure("unsupported_callback")
    litellm.callbacks.remove(f"callbacks.{module_name}.other_handler")
    litellm.callbacks.remove(initialized)
    pc._add_callbacks_from_db_config({"litellm_settings": {"callbacks": [f"callbacks.{module_name}.handler"]}})
    assert f"callbacks.{module_name}.handler" in litellm.callbacks
    assert protected_profile_failure() == ProfileFailure("unsupported_callback")


@pytest.mark.parametrize(
    "field,value",
    [
        ("disable_spend_logs", True),
        ("disable_spend_updates", True),
        ("disable_error_logs", True),
        ("store_prompts_in_spend_logs", False),
    ],
)
def test_general_settings_cannot_silently_disable_collection(protected_runtime, monkeypatch, field, value):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True, field: value})
    with pytest.raises(ProtectedProfileUnavailable) as rejected:
        enforce_protected_profile({"messages": [CANARY]})
    assert rejected.value.status_code == 503
    assert CANARY not in rejected.value.detail


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("turn_off_message_logging", True, "retention_conflict"),
        ("disable_streaming_logging", True, "retention_conflict"),
        ("standard_logging_payload_excluded_fields", ["response"], "retention_conflict"),
        ("logging", False, "retention_conflict"),
        ("cache", object(), "unsupported_cache"),
        ("set_verbose", True, "unsupported_diagnostics"),
        ("log_raw_request_response", True, "unsupported_diagnostics"),
        ("pre_call_rules", [lambda data: data], "unsupported_callback"),
    ],
)
def test_sdk_retention_and_sink_conflicts_fail_closed(protected_runtime, monkeypatch, field, value, code):
    monkeypatch.setattr(litellm, field, value)
    assert protected_profile_failure() == ProfileFailure(code)


@pytest.mark.parametrize("registry", CALLBACK_LISTS)
def test_unknown_sinks_rejected_in_every_callback_registry(protected_runtime, monkeypatch, registry):
    monkeypatch.setattr(litellm, registry, [*getattr(litellm, registry), "langfuse"])
    assert protected_profile_failure() == ProfileFailure("unsupported_callback")
    assert not approved_callback(CustomLogger())


@pytest.mark.parametrize("field,value", [("message_logging", False), ("turn_off_message_logging", True)])
def test_collector_redaction_conflict_is_rejected_without_disabling_redaction(
    protected_runtime, monkeypatch, field, value
):
    monkeypatch.setattr(protected_runtime.collector, field, value)
    assert protected_profile_failure() == ProfileFailure("retention_conflict")
    assert getattr(protected_runtime.collector, field) == value


@pytest.mark.parametrize("mutation", ["missing_collector", "missing_spool", "missing_db"])
def test_missing_mandatory_collector_is_not_ready(protected_runtime, monkeypatch, mutation):
    from litellm.proxy import proxy_server

    if mutation == "missing_collector":
        monkeypatch.setattr(litellm, "_async_success_callback", [])
    elif mutation == "missing_spool":
        protected_runtime.client._spend_log_spool = None
    else:
        monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert protected_profile_failure() == ProfileFailure("collector_unavailable")


@pytest.mark.parametrize(
    "data",
    [
        {"success_callback": ["langfuse"]},
        {"litellm_params": {"logger_fn": "export"}},
        {"turn_off_message_logging": True},
        {"cache": {"ttl": 300}},
        {"caching": True},
        {"router_settings_override": {"callbacks": ["otel"]}},
        {"guardrails": ["external"]},
    ],
)
def test_dynamic_content_options_are_rejected(protected_runtime, data):
    assert request_policy_failure(data) == ProfileFailure("request_override")
    with pytest.raises(ProtectedProfileUnavailable):
        enforce_protected_profile(data)


def test_identity_or_marker_tampering_cannot_reenable_plaintext(protected_runtime, monkeypatch):
    assert protected_profile_failure() is None
    replacement_uid = "11111111-1111-4111-8111-111111111111"
    protected_runtime.config_path.write_text(json.dumps(protected_runtime.config | {"instanceUid": replacement_uid}))
    monkeypatch.setenv(INSTANCE_ENV, replacement_uid)
    assert protected_profile_failure() == ProfileFailure("encryption_unavailable")
    monkeypatch.delenv(CONFIG_ENV)
    assert encryption_enabled()
    assert protected_profile_failure() == ProfileFailure("encryption_unavailable")


@pytest.mark.asyncio
async def test_crypto_failure_rejects_next_request_then_readiness_probe_recovers(protected_runtime):
    from litellm.proxy import proxy_server

    encryptor = configured_encryptor()
    with patch(
        "litellm.proxy.spend_tracking.request_content_encryption.jwe.encrypt_compact", side_effect=RuntimeError(CANARY)
    ):
        encryptor.encrypt("after-billing", {"v": 1, "request": CANARY})
        with pytest.raises(ProtectedProfileUnavailable):
            await proxy_server.proxy_logging_obj.pre_call_hook(UserAPIKeyAuth(), {"messages": [CANARY]}, "acompletion")
    assert protected_profile_failure() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_tag_budget_profile_query_never_sends_request_tags_to_caches(protected_runtime, configured):
    from litellm.proxy.auth.auth_checks import _tag_max_budget_check

    protected_runtime.client.db.query_raw.return_value = [{"configured": configured}]
    cache = SimpleNamespace(async_get_cache=AsyncMock(), async_set_cache=AsyncMock())
    call = _tag_max_budget_check(
        {"metadata": {"tags": [CANARY]}}, protected_runtime.client, cache, Mock(), UserAPIKeyAuth()
    )
    if configured:
        with pytest.raises(ProtectedProfileUnavailable):
            await call
    else:
        await call
    cache.async_get_cache.assert_not_awaited()
    cache.async_set_cache.assert_not_awaited()
    protected_runtime.client.db.query_raw.assert_awaited_once()
    assert CANARY not in str(protected_runtime.client.db.query_raw.call_args)
    assert _get_request_tags_for_cost_tracking({"request_tags": [CANARY]}, {"tags": [CANARY]}) is None


@pytest.mark.asyncio
async def test_unreadable_tag_budget_configuration_fails_closed(protected_runtime):
    protected_runtime.client.db.query_raw.side_effect = RuntimeError(CANARY)
    assert await protected_tag_budget_failure() == ProfileFailure("collector_unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[], [{"configured": "false"}], [{"configured": False}, {"configured": False}]])
async def test_malformed_delegated_tag_budget_result_fails_closed(protected_runtime, result):
    protected_runtime.client.db.query_raw.return_value = result
    assert await protected_tag_budget_failure() == ProfileFailure("collector_unavailable")
    protected_runtime.client.db.query_raw.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "database", [SimpleNamespace(), SimpleNamespace(query_raw=None), SimpleNamespace(query_raw=Mock(return_value=[]))]
)
async def test_unavailable_delegated_tag_budget_query_fails_closed(protected_runtime, database):
    protected_runtime.client.writer_db = PrismaWrapper(database, iam_token_db_auth=False)
    assert await protected_tag_budget_failure() == ProfileFailure("collector_unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["admission", "readiness"])
async def test_hung_tag_budget_query_is_cancelled_and_fails_closed(protected_runtime, entrypoint):
    from litellm.proxy import proxy_server
    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    cancelled = asyncio.Event()

    async def stalled_query(query):
        assert CANARY not in query
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    protected_runtime.client.writer_db.query_raw.side_effect = stalled_query
    if entrypoint == "admission":
        with pytest.raises(ProtectedProfileUnavailable):
            await asyncio.wait_for(
                proxy_server.proxy_logging_obj.pre_call_hook(UserAPIKeyAuth(), {"messages": [CANARY]}, "acompletion"),
                timeout=2.0,
            )
    else:
        response = Response()
        result = await asyncio.wait_for(health_readiness(response), timeout=2.0)
        assert response.status_code == 503
        assert result == {"status": "unhealthy", "request_logging": "unavailable"}
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_tag_budget_policy_uses_authoritative_writer_not_lagging_replica(protected_runtime):
    protected_runtime.client.writer_db = PrismaWrapper(
        SimpleNamespace(query_raw=AsyncMock(return_value=[{"configured": True}])), iam_token_db_auth=False
    )
    assert await protected_tag_budget_failure() == ProfileFailure("tag_budget_unsupported")
    protected_runtime.client.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_readiness_withholds_internal_failure_details(protected_runtime):
    from litellm.proxy.health_endpoints._health_endpoints import health_readiness, health_readiness_details

    protected_runtime.config_path.unlink()
    public_response, detail_response = Response(), Response()
    assert await health_readiness(public_response) == {"status": "unhealthy", "request_logging": "unavailable"}
    assert public_response.status_code == 503
    assert await health_readiness_details(detail_response) == {
        "status": "unhealthy",
        "request_logging": "encryption_unavailable",
    }
    assert detail_response.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("authentication_failure", [False, True])
async def test_rejected_requests_do_not_reach_unsupported_failure_sinks(
    protected_runtime, monkeypatch, authentication_failure
):
    from litellm.proxy import proxy_server

    external = CustomLogger()
    external.async_post_call_failure_hook = AsyncMock()
    external.async_log_failure_event = AsyncMock()
    monkeypatch.setattr(litellm, "callbacks", [protected_runtime.collector, external])
    monkeypatch.setattr(litellm, "_async_failure_callback", [external])
    request = {"model": "test-model", "messages": [{"role": "user", "content": CANARY}]}
    error = (
        HTTPException(status_code=401, detail="Unauthorized")
        if authentication_failure
        else ProtectedProfileUnavailable()
    )
    result = await proxy_server.proxy_logging_obj.post_call_failure_hook(
        request, error, UserAPIKeyAuth(request_route="/v1/chat/completions")
    )
    assert result is error
    external.async_post_call_failure_hook.assert_not_awaited()
    external.async_log_failure_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_unhealthy_profile_keeps_recovered_partial_stream_billing(protected_runtime, monkeypatch):
    from litellm.proxy import proxy_server

    configured_encryptor().record_failure(CaptureFailure("transformation_failed"))
    writer = SimpleNamespace(update_database=AsyncMock())
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "db_spend_update_writer", writer)
    usage = litellm.Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": CANARY}],
        "litellm_logging_obj": SimpleNamespace(
            model_call_details={"combined_usage_object": usage, "response_cost": 0.125}
        ),
    }
    await proxy_server.proxy_logging_obj.post_call_failure_hook(
        request, RuntimeError(CANARY), UserAPIKeyAuth(user_id="user-1", request_route="/v1/chat/completions")
    )
    writer.update_database.assert_awaited_once()
    assert writer.update_database.call_args.kwargs["response_cost"] == 0.125
    assert writer.update_database.call_args.kwargs["kwargs"]["combined_usage_object"] == usage


def test_legacy_profile_and_tag_counter_inputs_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "never-encrypted.sqlite"))
    with patch(
        "litellm.proxy.spend_tracking.request_content_policy.encryption_readiness", side_effect=AssertionError("legacy")
    ):
        assert protected_profile_failure() is None
        enforce_protected_profile({"success_callback": ["langfuse"]})
    assert _get_request_tags_for_cost_tracking({"request_tags": [CANARY]}, {}) == [CANARY]


def test_subagent_completion_classification_survives_projection():
    assert safe_metadata(
        {"spend_logs_metadata": {"openclaw_source_tool": "subagent_announce", "tool_name": CANARY}}
    ) == {"spend_logs_metadata": {"openclaw_source_tool": "subagent_announce"}}


@pytest.mark.parametrize("variable", ["USE_DDTRACE", "USE_DDPROFILER"])
def test_tracing_cannot_export_rejected_requests_before_admission(protected_runtime, monkeypatch, variable):
    from litellm.litellm_core_utils.dd_tracing import _should_use_dd_profiler, _should_use_dd_tracer

    monkeypatch.setenv(variable, "true")
    assert protected_profile_failure() == ProfileFailure("unsupported_diagnostics")
    assert not _should_use_dd_tracer()
    assert not _should_use_dd_profiler()


@pytest.mark.asyncio
async def test_rejected_request_headers_do_not_reach_an_unsupported_sink(protected_runtime, monkeypatch):
    from litellm.proxy import proxy_server

    observed = AsyncMock(return_value={})

    class HeaderSink(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):
            return await observed(**kwargs)

    monkeypatch.setattr(litellm, "callbacks", [protected_runtime.collector, HeaderSink()])
    result = await proxy_server.proxy_logging_obj.post_call_response_headers_hook(
        {"messages": [CANARY]}, UserAPIKeyAuth(), None
    )
    assert result == {}
    observed.assert_not_awaited()
