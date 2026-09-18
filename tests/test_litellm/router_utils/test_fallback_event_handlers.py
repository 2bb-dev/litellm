import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import litellm
from litellm.router_utils.fallback_event_handlers import (
    get_fallback_model_group,
    run_async_fallback,
)
from litellm.types.router import RetryPolicy


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_failure", ["encrypted", "quota", "server", "retry_quota"])
@pytest.mark.parametrize("error_encoding", ["code", "proxy_message"])
async def test_invalid_encrypted_content_stops_nested_fallbacks(initial_failure: str, error_encoding: str) -> None:
    from collections import Counter

    attempts: Counter[str] = Counter()
    encrypted_error = litellm.BadRequestError(
        message=(
            "Encrypted content could not be verified"
            if error_encoding == "code"
            else 'Proxy error: {"code":"invalid_encrypted_content"}'
        ),
        model="openai/test",
        llm_provider="openai",
        body=({"code": "invalid_encrypted_content"} if error_encoding == "code" else None),
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "openai/test", "api_key": "test"},
            }
            for name in ("primary", "fallback", "unused")
        ],
        fallbacks=[
            {"primary": ["fallback", "unused"]},
            {"fallback": ["primary", "unused"]},
        ],
        retry_policy=RetryPolicy(
            BadRequestErrorRetries=2,
            RateLimitErrorRetries=1 if initial_failure == "retry_quota" else 0,
        ),
        num_retries=0,
        max_fallbacks=2,
    )

    async def fail(model: str, **kwargs: object) -> None:
        attempts[model] += 1
        if (
            model != "primary"
            or initial_failure == "encrypted"
            or (initial_failure == "retry_quota" and attempts[model] > 1)
        ):
            raise encrypted_error
        if initial_failure in ("quota", "retry_quota"):
            raise litellm.RateLimitError("Quota exhausted", "openai", model)
        raise litellm.InternalServerError("Temporary failure", "openai", model)

    try:
        with pytest.raises(litellm.BadRequestError) as caught:
            await router.async_function_with_fallbacks(original_function=fail, model="primary", metadata={})
        assert caught.value is encrypted_error
        if initial_failure == "encrypted":
            assert attempts == {"primary": 1}
        elif initial_failure == "retry_quota":
            assert attempts == {"primary": 2}
        else:
            assert attempts == {"primary": 1, "fallback": 1}
    finally:
        router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["quota", "server", "context", "policy", "bad_request"])
async def test_other_errors_keep_configured_fallbacks(error_kind: str) -> None:
    errors = {
        "quota": litellm.RateLimitError("Quota exhausted", "openai", "primary"),
        "server": litellm.InternalServerError("Temporary failure", "openai", "primary"),
        "context": litellm.ContextWindowExceededError("Too long", "primary", "openai"),
        "policy": litellm.ContentPolicyViolationError("Policy rejected", "primary", "openai"),
        "bad_request": litellm.BadRequestError("Unsupported option", "primary", "openai"),
    }
    router = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "openai/test", "api_key": "test"},
            }
            for name in ("primary", "fallback")
        ],
        fallbacks=[{"primary": ["fallback"]}],
        context_window_fallbacks=[{"primary": ["fallback"]}],
        content_policy_fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )

    async def recover(model: str, **kwargs: object) -> litellm.ModelResponse:
        if model == "primary":
            raise errors[error_kind]
        return litellm.ModelResponse(model=model)

    try:
        result = await router.async_function_with_fallbacks(original_function=recover, model="primary", metadata={})
        assert result.model == "fallback"
    finally:
        router.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_responses_http_encrypted_error_does_not_cycle_slots(stream: bool) -> None:
    attempts: list[int] = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            slot = int(self.path.split("/")[1])
            attempts.append(slot)
            status = 429 if slot in (1, 2, 4, 5) else 400
            code = "usage_limit_reached" if status == 429 else "invalid_encrypted_content"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"code": code, "message": code}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    slots = (5, 3, 1, 2, 4, 6)
    router = litellm.Router(
        model_list=[
            {
                "model_name": f"model/{slot}",
                "litellm_params": {
                    "model": "litellm_proxy/test",
                    "api_key": "synthetic",
                    "api_base": f"http://127.0.0.1:{server.server_port}/{slot}",
                },
                "model_info": {"id": f"test-slot{slot}"},
            }
            for slot in slots
        ],
        fallbacks=[{f"model/{slot}": [f"model/{other}" for other in slots if other != slot]} for slot in slots],
        retry_policy=RetryPolicy(RateLimitErrorRetries=0),
        allowed_fails=0,
        cooldown_time=300,
        num_retries=2,
    )
    try:
        with pytest.raises(litellm.BadRequestError, match="invalid_encrypted_content") as caught:
            await router.aresponses(model="model/5", input="Synthetic test", stream=stream)
        assert caught.value.status_code == 400
        assert attempts == [5, 3]
    finally:
        router.reset()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
