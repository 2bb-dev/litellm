"""Synthetic credential evidence: caller JSON is never a producer stamp."""

import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import litellm
from litellm.litellm_core_utils.credential_ownership import (
    CONTEXT,
    FIELD,
    STAMP,
    ownership_for_spend,
    resolve_ownership,
    safe_ownership,
    select_credential,
)
from litellm.types.utils import CredentialItem
from litellm.utils import load_credentials_from_list

REGISTRATION = {
    "v": 1,
    "source": "platform",
    "registration_id": "3e15c0c2-feca-4104-a648-8d579315ef51",
    "registration_revision": "e7ca1c3e-b6ea-4ce0-8538-b8f149448038",
}


def deployment(source="platform", deployment_id="selected-a", named=False):
    return {
        "model_name": "requested-group",
        "model_info": {"id": deployment_id, **({} if named else {FIELD: {**REGISTRATION, "source": source}})},
        "litellm_params": {
            "model": "openai/test-model",
            "api_base": "https://example.invalid/v1",
            **({"litellm_credential_name": "registered-name"} if named else {"api_key": "synthetic-upstream-key"}),
        },
    }


def selected_request(route, overrides=None):
    request = overrides or {}
    selection = select_credential(route, request, route["model_info"]["id"])
    return {
        **route["litellm_params"],
        **request,
        "metadata": {"model_info": route["model_info"], CONTEXT: selection},
    }


@pytest.mark.parametrize("provider", ["openai", "litellm_proxy"])
@pytest.mark.parametrize("source", ["platform", "byok"])
def test_explicit_route_registration_and_immutable_stamp(source, provider):
    route = deployment(source)
    route["litellm_params"]["model"] = f"{provider}/test-model"
    request = selected_request(route)
    stamp = resolve_ownership(request, {}, {})
    route["model_info"][FIELD]["source"] = "byok" if source == "platform" else "platform"
    fact = ownership_for_spend(copy.deepcopy(stamp), "selected-a")
    assert fact == {**REGISTRATION, "source": source, "provenance": "route_registration", "deployment_id": "selected-a"}
    assert "synthetic" not in json.dumps(fact)
    assert ownership_for_spend(stamp, "different-deployment")["source"] == "unknown"


@pytest.mark.parametrize(
    "override",
    [
        {"api_key": "synthetic-other"},
        {"api_base": "https://other.invalid"},
        {"base_url": "https://other.invalid"},
        {"extra_headers": {"Authorization": "synthetic-other"}},
        {"client": object()},
        {"litellm_credential_name": "other"},
        {"azure_ad_token": "synthetic"},
    ],
)
@pytest.mark.parametrize("provider", ["openai", "veniceai", "litellm_proxy"])
def test_request_overrides_never_inherit_ownership(override, provider):
    route = deployment()
    route["litellm_params"]["model"] = f"{provider}/test-model"
    stamp = resolve_ownership(selected_request(route, override), {}, {})
    assert ownership_for_spend(stamp, "selected-a")["provenance"] == "credential_override"


@pytest.mark.parametrize("provider", ["openai", "veniceai", "litellm_proxy"])
def test_late_hook_override_and_unregistered_environment_key_are_unknown(provider):
    route = deployment()
    route["litellm_params"]["model"] = f"{provider}/test-model"
    request = selected_request(route)
    request["api_key"] = "changed-after-router"
    assert ownership_for_spend(resolve_ownership(request, {}, {}), "selected-a")["source"] == "unknown"
    route["model_info"].pop(FIELD)
    route["litellm_params"]["api_key"] = "os.environ/PLATFORM_KEY"
    assert ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")["source"] == "unknown"


@pytest.mark.parametrize("provider", ["openai", "litellm_proxy"])
def test_registered_route_with_unresolved_environment_key_is_credential_override(provider):
    route = deployment()
    route["litellm_params"]["model"] = f"{provider}/test-model"
    route["litellm_params"]["api_key"] = "os.environ/PLATFORM_KEY"
    fact = ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")
    assert (fact["source"], fact["provenance"]) == ("unknown", "credential_override")


def test_explicit_custom_llm_provider_param_is_ambiguous():
    route = deployment()
    route["litellm_params"]["model"] = "litellm_proxy/test-model"
    route["litellm_params"]["custom_llm_provider"] = "litellm_proxy"
    fact = ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")
    assert (fact["source"], fact["provenance"]) == ("unknown", "ambiguous")


@pytest.mark.parametrize("provider", ["openai", "litellm_proxy"])
def test_router_resolves_environment_key_before_selection(monkeypatch, provider):
    # Static server configuration binds `api_key: os.environ/NAME`; the Router
    # resolves it while loading the deployment, so selection digests the real key.
    monkeypatch.setenv("OPENORANGE_TEST_UPSTREAM_KEY", "synthetic-upstream-key")
    route = deployment()
    route["litellm_params"]["model"] = f"{provider}/test-model"
    route["litellm_params"]["api_key"] = "os.environ/OPENORANGE_TEST_UPSTREAM_KEY"
    router = litellm.Router(model_list=[route], num_retries=0)
    loaded = router.get_deployment(model_id="selected-a")
    assert loaded is not None
    params = loaded.litellm_params.model_dump(exclude_none=True)
    assert params["api_key"] == "synthetic-upstream-key"
    stored = {
        "model_name": loaded.model_name,
        "model_info": loaded.model_info.model_dump(exclude_none=True),
        "litellm_params": params,
    }
    fact = ownership_for_spend(resolve_ownership(selected_request(stored), {}, {}), "selected-a")
    assert (fact["source"], fact["provenance"], fact["deployment_id"]) == (
        "platform",
        "route_registration",
        "selected-a",
    )


@pytest.mark.parametrize("model", ["openrouter/test-model", "chatgpt/test-model", "litellm_proxyx/test-model"])
def test_unlisted_provider_prefix_stays_ambiguous(model):
    route = deployment()
    route["litellm_params"]["model"] = model
    fact = ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")
    assert (fact["source"], fact["provenance"]) == ("unknown", "ambiguous")


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_caller_json_cannot_construct_selection_or_stamp(metadata_key):
    forged = {**REGISTRATION, "deployment_id": "selected-a", "provenance": "route_registration"}
    request = {"model": "openai/test-model", metadata_key: {FIELD: forged, CONTEXT: forged}, STAMP: forged}
    assert ownership_for_spend(resolve_ownership(request, {}, {}), "selected-a")["source"] == "unknown"
    assert ownership_for_spend(forged, "selected-a")["source"] == "unknown"


@pytest.mark.parametrize(
    "invalid",
    [
        {"v": True},
        {"registration_id": "not-a-uuid"},
        {"registration_revision": ""},
        {"source": "unknown"},
        {"extra": "content"},
    ],
)
def test_malformed_registration_is_unknown(invalid):
    route = deployment()
    route["model_info"][FIELD].update(invalid)
    assert ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")["source"] == "unknown"


@pytest.mark.parametrize("provider", ["openai", "veniceai"])
def test_named_resolution_uses_one_snapshot_and_survives_rotation(monkeypatch, provider):
    route = deployment(named=True)
    route["litellm_params"]["model"] = f"{provider}/test-model"
    credential = CredentialItem(
        credential_name="registered-name",
        credential_values={"api_key": "synthetic-old"},
        credential_info={FIELD: {**REGISTRATION, "source": "byok"}},
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])
    logging = SimpleNamespace(model_call_details={})
    request = {**selected_request(route), "litellm_logging_obj": logging}
    load_credentials_from_list(request)
    first = copy.deepcopy(logging.model_call_details[STAMP])
    assert request["api_key"] == "synthetic-old"
    assert ownership_for_spend(first, "selected-a")["source"] == "byok"
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="registered-name",
                credential_values={"api_key": "synthetic-new"},
                credential_info={FIELD: REGISTRATION},
            )
        ],
    )
    assert ownership_for_spend(first, "selected-a")["source"] == "byok"
    load_credentials_from_list(request)
    assert ownership_for_spend(logging.model_call_details[STAMP], "selected-a")["source"] == "unknown"


def test_duplicate_or_mixed_named_registration_is_unknown():
    request = selected_request(deployment(named=True))
    values = {"api_key": "synthetic-key"}
    info = {FIELD: REGISTRATION}
    assert (
        ownership_for_spend(resolve_ownership(request, values, info, credential_ambiguous=True), "selected-a")["source"]
        == "unknown"
    )
    assert (
        ownership_for_spend(resolve_ownership(request, {**values, "azure_ad_token": "synthetic"}, info), "selected-a")[
            "source"
        ]
        == "unknown"
    )


def test_fallback_router_replaces_source_and_deployment():
    router = litellm.Router(model_list=[deployment(), deployment("byok", "selected-b")])
    request = {"metadata": {FIELD: {**REGISTRATION, "source": "byok"}}}
    for route, expected in [(deployment(), "platform"), (deployment("byok", "selected-b"), "byok")]:
        router._update_kwargs_with_deployment(route, request)
        effective = {**route["litellm_params"], **request}
        fact = ownership_for_spend(resolve_ownership(effective, {}, {}), route["model_info"]["id"])
        assert fact["source"] == expected
        assert fact["deployment_id"] == route["model_info"]["id"]
        assert FIELD not in request["metadata"]


def test_spend_builder_removes_nested_spoofs_without_changing_native_spend(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(proxy_server, "general_settings", {})
    forged = {**REGISTRATION, "provenance": "route_registration", "deployment_id": "selected-a"}
    kwargs = {
        "model": "test-model",
        "call_type": "acompletion",
        "response_cost": 0.25,
        STAMP: forged,
        "litellm_params": {
            "metadata": {
                "model_info": {"id": "selected-a"},
                FIELD: forged,
                "spend_logs_metadata": {FIELD: forged, "nested": {FIELD: forged}},
                "user_api_key_metadata": {FIELD: forged},
            }
        },
    }
    row = get_logging_payload(
        kwargs,
        {"id": "test", "usage": {"prompt_tokens": 9, "completion_tokens": 3}},
        datetime.now(timezone.utc),
        datetime.now(timezone.utc),
    )
    metadata = json.loads(row["metadata"])
    assert metadata[FIELD]["source"] == "unknown"
    assert row["prompt_tokens"] == 9
    assert json.dumps(metadata).count(FIELD) == 1


def test_projection_rejects_mismatched_selected_deployment():
    fact = {**REGISTRATION, "provenance": "route_registration", "deployment_id": "selected-a"}
    assert safe_ownership(fact, "selected-b")["source"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "veniceai", "litellm_proxy"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_real_router_sdk_callbacks_stamp_actual_completed_credential(
    monkeypatch, provider, fallback, synchronous, upstream_server
):
    import asyncio
    import httpx
    from openai import AsyncOpenAI, OpenAI
    from litellm.integrations.custom_logger import CustomLogger

    class Capture(CustomLogger):
        def __init__(self):
            self.success = asyncio.Event()
            self.rows = []
            self.failures = []

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            if synchronous:
                self.rows.append(copy.deepcopy(kwargs))
                loop.call_soon_threadsafe(self.success.set)

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            if synchronous:
                self.failures.append(
                    {
                        STAMP: copy.deepcopy(kwargs.get(STAMP)),
                        "litellm_params": {"metadata": copy.deepcopy(kwargs["litellm_params"]["metadata"])},
                    }
                )

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            if not synchronous:
                self.failures.append(
                    {
                        STAMP: copy.deepcopy(kwargs.get(STAMP)),
                        "litellm_params": {"metadata": copy.deepcopy(kwargs["litellm_params"]["metadata"])},
                    }
                )

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            if not synchronous:
                self.rows.append(copy.deepcopy(kwargs))
                self.success.set()

    loop = asyncio.get_running_loop()
    capture = Capture()
    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "input_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [capture])
    primary_source = "byok" if provider == "veniceai" and not fallback else "platform"
    routes = [deployment(primary_source), deployment("byok", "selected-b")]
    base_url, calls, replies = upstream_server
    for route in routes:
        route["litellm_params"]["model"] = f"{provider}/test-model"
        route["litellm_params"]["api_base"] = base_url
    if fallback:
        replies.append((429, {"error": {"message": "synthetic retry", "type": "rate_limit_error"}}))
    routes[0]["model_name"] = "primary"
    routes[1]["model_name"] = "backup"
    routes[1]["litellm_params"]["api_key"] = "synthetic-byok-key"
    router = litellm.Router(model_list=routes, num_retries=0, fallbacks=[{"primary": ["backup"]}])
    clients = [
        OpenAI(
            api_key=route["litellm_params"]["api_key"],
            base_url=base_url,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        if synchronous
        else AsyncOpenAI(
            api_key=route["litellm_params"]["api_key"],
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False),
        )
        for route in routes
    ]
    try:
        for route, client in zip(routes, clients):
            suffix = "client" if synchronous else "async_client"
            router.cache.set_cache(key=f"{route['model_info']['id']}_{suffix}", value=client, local_only=True)
        call = {
            "model": "primary",
            "messages": [{"role": "user", "content": "synthetic"}],
            "metadata": {FIELD: {**REGISTRATION, "source": "forged"}},
        }
        response = (
            await asyncio.to_thread(router.completion, **call) if synchronous else await router.acompletion(**call)
        )
        await asyncio.wait_for(capture.success.wait(), 5)
        kwargs = capture.rows[-1]
        expected_id = "selected-b" if fallback else "selected-a"
        fact = ownership_for_spend(kwargs.get(STAMP), expected_id)
        assert fact["source"] == ("byok" if fallback else primary_source), fact
        assert kwargs["litellm_params"]["metadata"]["model_info"]["id"] == expected_id
        assert response.usage.total_tokens == 12
        if provider in ("veniceai", "litellm_proxy"):
            from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

            row = get_logging_payload(kwargs, response, datetime.now(timezone.utc), datetime.now(timezone.utc))
            metadata = json.loads(row["metadata"])
            assert metadata[FIELD] == fact
            assert row["model_id"] == expected_id
            assert row["custom_llm_provider"] == provider
            assert row["request_id"] == response.id
            assert (row["prompt_tokens"], row["completion_tokens"]) == (9, 3)
            assert "openorange_terminal_evidence" not in metadata
            assert "openorange_terminal_usage_evidence" not in metadata
        if fallback:
            assert capture.failures
            assert ownership_for_spend(capture.failures[0].get(STAMP), "selected-a")["source"] == "platform"
            assert capture.failures[0]["litellm_params"]["metadata"]["model_info"]["id"] == "selected-a"
        assert calls[-1][1] == ("Bearer synthetic-byok-key" if fallback else "Bearer synthetic-upstream-key")
    finally:
        for client in clients:
            if synchronous:
                client.close()
            else:
                await client.close()


def test_db_model_registration_is_not_reviewed_static_authority():
    route = deployment()
    route["model_info"]["db_model"] = True
    assert ownership_for_spend(resolve_ownership(selected_request(route), {}, {}), "selected-a")["source"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["none", "caller", "swap", "closed", "auth", "headers", "cookies", "trace", "method", "json"]
)
async def test_only_unchanged_plain_proxy_shared_session_retains_named_proof(monkeypatch, change):
    from aiohttp import BasicAuth, ClientSession, TraceConfig

    from litellm.proxy import proxy_server

    async with ClientSession() as session, ClientSession() as other:
        monkeypatch.setattr(proxy_server, "shared_aiohttp_session", session)
        route = deployment(named=True)
        route["model_info"]["db_model"] = True
        candidate = other if change == "caller" else {"trusted": True} if change == "json" else session
        request = selected_request(route, {"shared_session": candidate})
        request["metadata"] = copy.deepcopy(request["metadata"])
        if change == "swap":
            request["shared_session"] = other
        elif change == "closed":
            await session.close()
        elif change == "auth":
            session._default_auth = BasicAuth("synthetic", "synthetic")
        elif change == "headers":
            session.headers["Authorization"] = "Bearer synthetic-other"
        elif change == "cookies":
            session.cookie_jar.update_cookies({"synthetic": "synthetic"})
        elif change == "trace":
            session.trace_configs.append(TraceConfig())
        elif change == "method":
            session.__dict__["_request"] = object()
        stamp = resolve_ownership(request, {"api_key": "synthetic-key"}, {FIELD: {**REGISTRATION, "source": "byok"}})
        fact = ownership_for_spend(stamp, "selected-a")
        assert fact["source"] == ("byok" if change == "none" else "unknown")
        assert fact["provenance"] == ("credential_registration" if change == "none" else "credential_override")


@pytest.mark.asyncio
async def test_authenticated_proxy_startup_pool_named_venice_spend(monkeypatch, upstream_server):
    import asyncio

    import httpx

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    ready = asyncio.Event()
    rows = []

    class Capture(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            rows.append(get_logging_payload(kwargs, response_obj, start_time, end_time))
            ready.set()

    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "input_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [Capture()])
    registration = {**REGISTRATION, "source": "byok"}
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="registered-name",
                credential_values={"api_key": "synthetic-byok-key"},
                credential_info={FIELD: registration},
            )
        ],
    )
    base, calls, _ = upstream_server
    route = deployment(named=True)
    route["model_info"]["db_model"] = True
    route["litellm_params"].update(model="veniceai/test-model", api_base=base)
    router = litellm.Router(model_list=[route], num_retries=0, fallbacks=[])
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", "sk-synthetic-proxy")
    monkeypatch.setattr(proxy_server, "general_settings", {"master_key": "sk-synthetic-proxy"})
    session = await proxy_server._initialize_shared_aiohttp_session()
    assert session is not None
    monkeypatch.setattr(proxy_server, "shared_aiohttp_session", session)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://localhost:4000"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-synthetic-proxy"},
                json={
                    "model": "requested-group",
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "max_tokens": 16,
                    "num_retries": 0,
                    "model_group_retry_policy": {},
                    "disable_fallbacks": True,
                    "max_fallbacks": 0,
                    "fallbacks": [],
                    "context_window_fallbacks": [],
                    "content_policy_fallbacks": [],
                },
            )
        assert response.status_code == 200
        await asyncio.wait_for(ready.wait(), 5)
        row = rows[-1]
        assert json.loads(row["metadata"])[FIELD] == {
            **registration,
            "provenance": "credential_registration",
            "deployment_id": "selected-a",
        }
        assert row["request_id"] == response.json()["id"]
        assert row["model_id"] == "selected-a"
        assert row["custom_llm_provider"] == "veniceai"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (9, 3)
        assert len(calls) == 1 and calls[0][1] == "Bearer synthetic-byok-key"
    finally:
        await session.close()


@pytest.mark.parametrize(
    "patch_registration,expected",
    [
        (None, False),
        (REGISTRATION, False),
        ({**REGISTRATION, "registration_revision": "9ef18b91-2d17-487a-93ac-4a856b046212"}, True),
    ],
)
def test_credential_patch_invalidates_old_revision_in_persisted_record(patch_registration, expected):
    from litellm.proxy.credential_endpoints.endpoints import update_db_credential

    old = CredentialItem(
        credential_name="registered",
        credential_values={"api_key": "synthetic-encrypted-old"},
        credential_info={FIELD: REGISTRATION, "description": "preserved"},
    )
    patch = CredentialItem(
        credential_name="registered",
        credential_values={"api_key": "synthetic-new"},
        credential_info={FIELD: patch_registration} if patch_registration else {},
    )
    updated = update_db_credential(old, patch, new_encryption_key="synthetic-test-encryption-key")
    assert (FIELD in updated.credential_info) is expected
    assert updated.credential_info["description"] == "preserved"
    if expected:
        assert updated.credential_info[FIELD] == patch_registration


@pytest.mark.parametrize("role", ["internal_user", "proxy_admin_viewer"])
def test_registration_write_requires_actual_proxy_administrator(role):
    from fastapi import HTTPException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.credential_endpoints.endpoints import _require_ownership_registration_admin

    credential = CredentialItem(
        credential_name="registered", credential_values={}, credential_info={FIELD: REGISTRATION}
    )
    with pytest.raises(HTTPException) as error:
        _require_ownership_registration_admin(credential, UserAPIKeyAuth(user_role=role))
    assert error.value.status_code == 403
    _require_ownership_registration_admin(credential, UserAPIKeyAuth(user_role="proxy_admin"))


@pytest.fixture
def upstream_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.path, self.headers.get("Authorization")))
            code, body = (
                replies.pop(0)
                if replies
                else (
                    200,
                    {
                        "id": "synthetic-completed",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "synthetic"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
                    },
                )
            )
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, replies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("customization", ["auth", "request_hook", "response_hook", "transport"])
def test_custom_http_credential_mutation_is_unknown(customization):
    import httpx
    from openai import OpenAI

    class ReplaceAuth(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer synthetic-other"
            yield request

    options = {
        "auth": {"auth": ReplaceAuth()},
        "request_hook": {
            "event_hooks": {
                "request": [lambda request: request.headers.update({"Authorization": "Bearer synthetic-other"})]
            }
        },
        "response_hook": {"event_hooks": {"response": [lambda response: None]}},
        "transport": {"transport": httpx.MockTransport(lambda request: httpx.Response(200))},
    }[customization]
    with OpenAI(
        api_key="synthetic-upstream-key", base_url="https://example.invalid/v1", http_client=httpx.Client(**options)
    ) as client:
        request = {**selected_request(deployment()), "client": client}
        assert ownership_for_spend(resolve_ownership(request, {}, {}), "selected-a")["source"] == "unknown"


@pytest.mark.parametrize(
    "name,value",
    [
        ("headers", {"Authorization": "synthetic"}),
        ("client_session", object()),
        ("aclient_session", object()),
        ("network_mock", True),
    ],
)
def test_global_transport_overrides_are_unknown(monkeypatch, name, value):
    monkeypatch.setattr(litellm, name, value, raising=False)
    assert (
        ownership_for_spend(resolve_ownership(selected_request(deployment()), {}, {}), "selected-a")["source"]
        == "unknown"
    )


def test_reserved_metadata_sanitization_preserves_cycle_safety():
    from litellm.litellm_core_utils.credential_ownership import strip_ownership

    cyclic = {FIELD: {"source": "byok"}}
    cyclic["cycle"] = cyclic
    assert FIELD not in json.dumps(strip_ownership(cyclic))


@pytest.mark.parametrize("change", [{"mock_response": "synthetic"}, {"shared_session": object()}, {"api_base": None}])
def test_unproven_dispatch_or_endpoint_is_unknown(change):
    request = {**selected_request(deployment()), **change}
    assert ownership_for_spend(resolve_ownership(request, {}, {}), "selected-a")["source"] == "unknown"


def test_database_route_can_use_separate_credential_registration():
    route = deployment(named=True)
    route["model_info"]["db_model"] = True
    stamp = resolve_ownership(selected_request(route), {"api_key": "synthetic-key"}, {FIELD: REGISTRATION})
    assert ownership_for_spend(stamp, "selected-a")["provenance"] == "credential_registration"
