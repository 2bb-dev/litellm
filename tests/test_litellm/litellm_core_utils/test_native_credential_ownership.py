"""Actual native adapter dispatch against loopback-only synthetic providers."""

import asyncio
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.credential_ownership import FIELD, STAMP, ownership_for_spend

REGISTRATION = {
    "v": 1,
    "source": "platform",
    "registration_id": "3e15c0c2-feca-4104-a648-8d579315ef51",
    "registration_revision": "e7ca1c3e-b6ea-4ce0-8538-b8f149448038",
}
MODELS = {"anthropic": "anthropic/claude-fable-5-1", "deepseek": "deepseek/deepseek-flash"}


def response_body(provider):
    if provider == "anthropic":
        return {
            "id": "synthetic-anthropic",
            "type": "message",
            "role": "assistant",
            "model": "claude-fable-5-1",
            "content": [{"type": "text", "text": "synthetic"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }
    return {
        "id": "synthetic-deepseek",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "synthetic"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }


@pytest.fixture
def native_server():
    calls, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            calls.append({"path": self.path, "headers": dict(self.headers), "body": body})
            code, response = replies.pop(0)
            if code == 0:
                self.close_connection = True
                return
            streamed = bool(body.get("stream")) and code == 200
            data = native_stream(response).encode() if streamed else json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/event-stream" if streamed else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, replies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Capture(CustomLogger):
    def __init__(self, synchronous):
        self.synchronous = synchronous
        self.loop = asyncio.get_running_loop()
        self.done = asyncio.Event()
        self.rows = []
        self.failures = []

    def record(self, kwargs, failed):
        row = {
            STAMP: copy.deepcopy(kwargs.get(STAMP)),
            "metadata": copy.deepcopy(kwargs.get("litellm_params", {}).get("metadata", {})),
        }
        (self.failures if failed else self.rows).append(row)
        if not failed:
            self.loop.call_soon_threadsafe(self.done.set)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self.synchronous:
            self.record(kwargs, False)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if self.synchronous:
            self.record(kwargs, True)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if not self.synchronous:
            self.record(kwargs, False)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if not self.synchronous:
            self.record(kwargs, True)


def route(provider, source, base, number):
    return {
        "model_name": f"group-{number}",
        "model_info": {"id": f"deployment-{number}", FIELD: {**REGISTRATION, "source": source}},
        "litellm_params": {
            "model": MODELS[provider],
            "api_key": f"sk-ant-synthetic-{number}" if provider == "anthropic" else f"synthetic-deepseek-{number}",
            "api_base": base,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("mode", ["sync", "async_httpx", "async_aiohttp"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_adapter_completed_and_failed_fallback_ownership(
    monkeypatch, native_server, provider, mode, fallback, stream
):
    base, calls, replies = native_server
    if fallback:
        replies.append((429, {"error": {"type": "rate_limit_error", "message": "synthetic retry"}}))
    replies.append((200, response_body(provider)))
    synchronous = mode == "sync"
    capture = Capture(synchronous)
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
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", mode == "async_httpx")
    routes = [route(provider, "platform", base, 1), route(provider, "byok", base, 2)]
    router = litellm.Router(model_list=routes, num_retries=0, fallbacks=[{"group-1": ["group-2"]}])
    args = {
        "model": "group-1",
        "messages": [{"role": "user", "content": "synthetic"}],
        "metadata": {FIELD: {"source": "forged"}},
        "stream": stream,
    }
    response = await asyncio.to_thread(router.completion, **args) if synchronous else await router.acompletion(**args)
    if stream:
        chunks = await asyncio.to_thread(list, response) if synchronous else [chunk async for chunk in response]
        response = litellm.stream_chunk_builder(chunks)
    await asyncio.wait_for(capture.done.wait(), 5)
    expected = "deployment-2" if fallback else "deployment-1"
    fact = ownership_for_spend(capture.rows[-1][STAMP], expected)
    assert fact["source"] == ("byok" if fallback else "platform"), fact
    assert capture.rows[-1]["metadata"]["model_info"]["id"] == expected
    assert response.usage.total_tokens == 12
    if fallback:
        assert capture.failures
        assert ownership_for_spend(capture.failures[0][STAMP], "deployment-1")["source"] == "platform"
    number = 2 if fallback else 1
    headers = {key.lower(): value for key, value in calls[-1]["headers"].items()}
    assert calls[-1]["path"] == ("/v1/messages" if provider == "anthropic" else "/chat/completions")
    assert headers.get("x-api-key") == (f"sk-ant-synthetic-{number}" if provider == "anthropic" else None)
    assert headers.get("authorization") == (f"Bearer synthetic-deepseek-{number}" if provider == "deepseek" else None)


def native_stream(response):
    if response.get("type") == "message":
        events = [
            {
                "type": "message_start",
                "message": {
                    **response,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "synthetic"}},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        ]
        return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    chunk = {
        **response,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "synthetic"}, "finish_reason": None}],
    }
    final = {
        **response,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("mutation", ["httpx_auth", "httpx_hook", "aiohttp_connector"])
async def test_actual_late_auth_mutation_does_not_inherit_registration(monkeypatch, native_server, provider, mutation):
    import httpx
    from aiohttp import ClientSession, TCPConnector

    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    base, calls, replies = native_server
    replies.append((200, response_body(provider)))
    capture = Capture(False)
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
    router = litellm.Router(model_list=[route(provider, "platform", base, 1)], num_retries=0)

    def replace_auth(request):
        request.headers.pop("x-api-key", None)
        request.headers.pop("Authorization", None)
        request.headers["Authorization"] = "Bearer synthetic-other-credential"

    class ReplaceAuth(httpx.Auth):
        def auth_flow(self, request):
            replace_auth(request)
            yield request

    class MutatingConnector(TCPConnector):
        async def _create_connection(self, req, traces, timeout):
            replace_auth(req)
            return await super()._create_connection(req, traces, timeout)

    async def replace_hook(request):
        replace_auth(request)

    session = ClientSession(connector=MutatingConnector()) if mutation == "aiohttp_connector" else None
    http_client = httpx.AsyncClient(
        trust_env=False,
        **(
            {"auth": ReplaceAuth()}
            if mutation == "httpx_auth"
            else {"event_hooks": {"request": [replace_hook]}}
            if mutation == "httpx_hook"
            else {"transport": LiteLLMAiohttpTransport(client=session)}
        ),
    )
    handler = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = http_client
    try:
        router.cache.set_cache(key="deployment-1_async_client", value=handler, local_only=True)
        await router.acompletion(model="group-1", messages=[{"role": "user", "content": "synthetic"}])
        await asyncio.wait_for(capture.done.wait(), 5)
        assert ownership_for_spend(capture.rows[-1][STAMP], "deployment-1")["source"] == "unknown"
        sent = {key.lower(): value for key, value in calls[-1]["headers"].items()}
        assert sent["authorization"] == "Bearer synthetic-other-credential"
        assert "x-api-key" not in sent
    finally:
        await http_client.aclose()
        if session is not None:
            await session.close()


@pytest.mark.parametrize(
    "provider,endpoint",
    [
        ("anthropic", "https://api.anthropic.com/v1/messages"),
        ("deepseek", "https://api.deepseek.com/beta/chat/completions"),
    ],
)
def test_native_default_endpoint_and_terminal_response_binding(provider, endpoint):
    import httpx

    from litellm.litellm_core_utils.credential_ownership import (
        CONTEXT,
        ownership_after_http_response,
        resolve_ownership,
        select_credential,
    )

    deployment = route(provider, "byok", None, 1)
    deployment["litellm_params"].pop("api_base")
    pending = resolve_ownership(
        {**deployment["litellm_params"], "metadata": {CONTEXT: select_credential(deployment, {}, "deployment-1")}},
        {},
        {},
    )
    assert ownership_for_spend(pending, "deployment-1")["source"] == "unknown"
    key = deployment["litellm_params"]["api_key"]
    headers = {"x-api-key": key} if provider == "anthropic" else {"Authorization": f"Bearer {key}"}
    with httpx.Client(trust_env=False) as client:
        request = client.build_request("POST", endpoint, headers=headers)
        response = httpx.Response(500, request=request)
        fact = ownership_after_http_response(pending, request, client, response)
        assert ownership_for_spend(fact, "deployment-1")["source"] == "byok"
        wrong = client.build_request("POST", "https://other.invalid/path", headers=headers)
        assert (
            ownership_for_spend(
                ownership_after_http_response(pending, wrong, client, httpx.Response(200, request=wrong)),
                "deployment-1",
            )["source"]
            == "unknown"
        )
        response.history = [httpx.Response(307, request=request)]
        assert (
            ownership_for_spend(ownership_after_http_response(pending, request, client, response), "deployment-1")[
                "source"
            ]
            == "unknown"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("mode", ["sync", "async_httpx", "async_aiohttp"])
async def test_disconnect_cannot_reuse_prior_dispatch_proof(monkeypatch, native_server, provider, mode):
    from datetime import datetime

    import httpx

    from litellm.litellm_core_utils.credential_ownership import CONTEXT, DISPATCH, resolve_ownership, select_credential
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    base, calls, replies = native_server
    replies.extend([(200, response_body(provider)), (0, {}), (0, {}), (0, {})])
    deployment = route(provider, "platform", base, 1)
    pending = resolve_ownership(
        {**deployment["litellm_params"], "metadata": {CONTEXT: select_credential(deployment, {}, "deployment-1")}},
        {},
        {},
    )
    logging_obj = Logging(
        MODELS[provider], [], False, "completion", datetime.now(), "synthetic-call", "synthetic-function"
    )
    logging_obj.model_call_details.update({STAMP: pending, DISPATCH: pending})
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", mode == "async_httpx")
    synchronous = mode == "sync"
    handler = HTTPHandler() if synchronous else AsyncHTTPHandler()
    key = deployment["litellm_params"]["api_key"]
    args = {
        "url": base + ("/v1/messages" if provider == "anthropic" else "/chat/completions"),
        "json": {},
        "headers": {"x-api-key": key} if provider == "anthropic" else {"Authorization": f"Bearer {key}"},
        "logging_obj": logging_obj,
    }
    try:
        if synchronous:
            await asyncio.to_thread(handler.post, **args)
        else:
            await handler.post(**args)
        assert ownership_for_spend(logging_obj.model_call_details[STAMP], "deployment-1")["source"] == "platform"
        with pytest.raises(httpx.TransportError):
            if synchronous:
                await asyncio.to_thread(handler.post, **args)
            else:
                await handler.post(**args)
        assert len(calls) >= 2
        assert ownership_for_spend(logging_obj.model_call_details[STAMP], "deployment-1")["source"] == "unknown"
    finally:
        if synchronous:
            handler.close()
        else:
            await handler.close()
