import asyncio
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

import litellm
from litellm import RateLimitError, Router
from litellm.exceptions import MidStreamFallbackError
from litellm.router_utils.chatgpt_rate_limit import (
    chatgpt_quota_reset_seconds,
    is_chatgpt_quota_error,
    is_chatgpt_rate_limit,
    is_chatgpt_transient_error,
)
from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment

SIDECAR_MODEL = "chatgpt/gpt-6.1-sol"
OVERLOADED = "Our servers are currently overloaded. Please try again later."


class ChatGPTRateLimitTest(unittest.TestCase):
    def error(
        self, payload: object, provider: str = "litellm_proxy", model: str = "litellm_proxy/chatgpt/gpt-5.6-sol"
    ) -> RateLimitError:
        return RateLimitError(message=json.dumps(payload), llm_provider=provider, model=model)

    def test_exact_quota_codes_survive_proxy_wrapping(self):
        for code in ("usage_limit_reached", "insufficient_quota"):
            for key in ("type", "code"):
                payload = {"error": {key: code, "message": "Synthetic quota exhausted"}}
                wrapped = {
                    "error": {
                        "message": "litellm.RateLimitError: ChatgptException - " + json.dumps(payload),
                        "code": "429",
                    }
                }
                for body in (payload, wrapped):
                    self.assertTrue(is_chatgpt_quota_error(self.error(body)))
                    self.assertTrue(is_chatgpt_quota_error(self.error(body, "chatgpt", "gpt-5.6-sol")))

    def test_transient_unknown_and_text_only_errors_are_not_quota(self):
        for payload in (
            {"detail": "Rate limit exceeded"},
            {"error": {"type": "rate_limit_exceeded"}},
            {"error": {"message": "usage_limit_reached"}},
            {"error": {"code": ["usage_limit_reached"]}},
            {"input": {"error": {"code": "usage_limit_reached"}}},
            "malformed {usage_limit_reached}",
        ):
            self.assertFalse(is_chatgpt_quota_error(self.error(payload)))
        self.assertFalse(is_chatgpt_quota_error(ValueError("usage_limit_reached")))

    def test_native_provider_behavior_is_not_reclassified(self):
        payload = {"error": {"code": "insufficient_quota"}}
        for provider, model in (("openai", "gpt-5.6-sol"), ("litellm_proxy", "openai/gpt-5.6-sol")):
            self.assertFalse(is_chatgpt_rate_limit(self.error(payload, provider, model)))
            self.assertFalse(is_chatgpt_quota_error(self.error(payload, provider, model)))

    def test_cooldown_exempts_only_transient_chatgpt_429(self):
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "model_info": {"id": "test-slot"},
                    "litellm_params": {
                        "model": "litellm_proxy/chatgpt/gpt-5.6-sol",
                        "api_key": "synthetic",
                        "api_base": "http://127.0.0.1:1",
                    },
                }
            ],
            allowed_fails=0,
        )
        try:
            self.assertFalse(
                _should_cooldown_deployment(router, "test-slot", 429, self.error({"detail": "Rate limit exceeded"}))
            )
            self.assertTrue(
                _should_cooldown_deployment(
                    router, "test-slot", 429, self.error({"error": {"type": "usage_limit_reached"}})
                )
            )
            self.assertTrue(
                _should_cooldown_deployment(
                    router, "test-slot", 429, self.error({"detail": "Rate limit exceeded"}, "openai", "gpt-5.6-sol")
                )
            )
        finally:
            router.reset()


class ChatGPTTransientErrorTest(unittest.TestCase):
    def test_openai_hiccups_on_a_chatgpt_account_are_transient(self):
        for provider, model in (("litellm_proxy", SIDECAR_MODEL), ("chatgpt", "gpt-6.1-sol")):
            overloaded = litellm.InternalServerError(message=OVERLOADED, llm_provider=provider, model=model)
            for error in (
                overloaded,
                litellm.ServiceUnavailableError(message=OVERLOADED, llm_provider=provider, model=model),
                litellm.BadGatewayError(message="bad gateway", llm_provider=provider, model=model),
                litellm.Timeout(message="timed out", llm_provider=provider, model=model),
                MidStreamFallbackError(
                    message=OVERLOADED, model=model, llm_provider=provider, original_exception=overloaded
                ),
            ):
                self.assertTrue(is_chatgpt_transient_error(error), type(error).__name__)

    def test_rate_limits_unreachable_sidecars_client_errors_and_native_routes_are_not(self):
        throttled = RateLimitError(message="Rate limit exceeded", llm_provider="litellm_proxy", model=SIDECAR_MODEL)
        for error in (
            throttled,
            MidStreamFallbackError(
                message="throttled", model=SIDECAR_MODEL, llm_provider="litellm_proxy", original_exception=throttled
            ),
            litellm.APIConnectionError(message="refused", llm_provider="litellm_proxy", model=SIDECAR_MODEL),
            litellm.BadRequestError(message="bad", llm_provider="litellm_proxy", model=SIDECAR_MODEL),
            litellm.InternalServerError(message=OVERLOADED, llm_provider="openai", model="gpt-6.1-sol"),
            litellm.InternalServerError(message=OVERLOADED, llm_provider="litellm_proxy", model="openai/gpt-6.1-sol"),
            ValueError(OVERLOADED),
        ):
            self.assertFalse(is_chatgpt_transient_error(error), repr(error))


class ChatGPTQuotaResetTest(unittest.IsolatedAsyncioTestCase):
    def quota(self, upstream: object, headers: tuple[tuple[str, str], ...] = ()) -> RateLimitError:
        """A quota 429 as a ChatGPT sidecar answers it: OpenAI's body inside its own error message."""
        wrapped = {"error": {"message": "litellm.RateLimitError: ChatgptException - " + json.dumps(upstream)}}
        return RateLimitError(
            message=json.dumps(wrapped),
            llm_provider="litellm_proxy",
            model=SIDECAR_MODEL,
            response=httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://127.0.0.1:4101")),
        )

    def test_retry_after_wins_also_when_the_sidecar_renamed_it(self):
        body = {"error": {"type": "usage_limit_reached", "resets_in_seconds": 40}}
        self.assertEqual(chatgpt_quota_reset_seconds(self.quota(body, (("llm_provider-retry-after", "20"),))), 20)
        self.assertEqual(chatgpt_quota_reset_seconds(self.quota(body, (("retry-after", "15"),))), 15)

    def test_reset_time_in_the_body_without_retry_after(self):
        self.assertEqual(
            chatgpt_quota_reset_seconds(
                self.quota({"error": {"type": "usage_limit_reached", "resets_in_seconds": 40}})
            ),
            40,
        )
        resets_at = time.time() + 30
        seconds = chatgpt_quota_reset_seconds(
            self.quota({"error": {"code": "usage_limit_reached", "resets_at": resets_at}})
        )
        self.assertIsNotNone(seconds)
        self.assertAlmostEqual(seconds or 0, 30, delta=2)

    def test_no_wait_named_or_not_a_quota_error(self):
        self.assertIsNone(chatgpt_quota_reset_seconds(self.quota({"error": {"type": "usage_limit_reached"}})))
        self.assertIsNone(
            chatgpt_quota_reset_seconds(
                self.quota({"error": {"type": "usage_limit_reached", "resets_at": time.time() - 5}})
            )
        )
        throttled = self.quota({"detail": "Rate limit exceeded"}, (("llm_provider-retry-after", "2"),))
        self.assertIsNone(chatgpt_quota_reset_seconds(throttled))
        self.assertIsNone(
            chatgpt_quota_reset_seconds(litellm.InternalServerError(OVERLOADED, "litellm_proxy", SIDECAR_MODEL))
        )

    async def test_the_bench_lasts_as_long_as_openai_says(self):
        router = Router(
            model_list=[
                {
                    "model_name": "sol",
                    "model_info": {"id": "sol-slot1"},
                    "litellm_params": {
                        "model": "litellm_proxy/" + SIDECAR_MODEL,
                        "api_key": "synthetic",
                        "api_base": "http://127.0.0.1:1",
                    },
                }
            ],
            allowed_fails=0,
            cooldown_time=300,
        )
        try:
            error = self.quota({"error": {"type": "usage_limit_reached"}}, (("llm_provider-retry-after", "20"),))
            kwargs = {
                "exception": error,
                "litellm_params": {"model_info": {"id": "sol-slot1"}, "metadata": {"model_group": "sol"}},
            }
            router.deployment_callback_on_failure(kwargs, None, None, None)
            key = router.cooldown_cache.get_cooldown_cache_key("sol-slot1")
            self.assertEqual(router.cooldown_cache.cooldown_store.get_cache(key=key)["cooldown_time"], 20)
        finally:
            router.reset()


class _Sidecar(BaseHTTPRequestHandler):
    """A ChatGPT sidecar: each call takes the next scripted answer, then answers normally."""

    def log_message(self, *_: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        server = self.server
        step = server.script.pop(0) if server.script else "ok"
        server.calls.append(step)
        if step == "503":
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": OVERLOADED, "code": "503"}}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        response = {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-6.1-sol", "output": []}
        frames = [("response.created", {"type": "response.created", "response": {**response, "status": "in_progress"}})]
        if step == "partial":
            frames.append(
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "half an ans",
                    },
                )
            )
        if step in ("overloaded", "partial"):
            failed = {**response, "status": "failed", "error": {"code": "server_is_overloaded", "message": OVERLOADED}}
            frames.append(("response.failed", {"type": "response.failed", "response": failed}))
        else:
            usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            completed = {**response, "status": "completed", "usage": usage}
            frames.append(("response.completed", {"type": "response.completed", "response": completed}))
        for number, (name, data) in enumerate(frames):
            self.wfile.write(f"event: {name}\ndata: {json.dumps({**data, 'sequence_number': number})}\n\n".encode())


class ChatGPTSameAccountRetryTest(unittest.IsolatedAsyncioTestCase):
    """A ChatGPT account that hiccups is retried, never benched and never swapped for another one."""

    def setUp(self) -> None:
        self.sidecars = {}
        for slot in (1, 2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Sidecar)
            server.script = []
            server.calls = []
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.sidecars[slot] = server
        self.router = Router(
            model_list=[
                {
                    "model_name": f"sol/{slot}",
                    "model_info": {"id": f"sol-slot{slot}"},
                    "litellm_params": {
                        "model": "litellm_proxy/" + SIDECAR_MODEL,
                        "api_key": "synthetic",
                        "api_base": f"http://127.0.0.1:{server.server_port}",
                    },
                }
                for slot, server in self.sidecars.items()
            ],
            fallbacks=[{"sol/1": ["sol/2"]}],
            allowed_fails=0,
            cooldown_time=300,
            num_retries=2,
            allowed_fails_policy={"InternalServerErrorAllowedFails": 1000, "ServiceUnavailableErrorAllowedFails": 1000},
        )

    def tearDown(self) -> None:
        self.router.reset()
        for server in self.sidecars.values():
            server.shutdown()
            server.server_close()

    async def stream(self) -> str | None:
        response = await self.router.aresponses(model="sol/1", input="Hi", stream=True)
        statuses = [event.response.status async for event in response if event.type == "response.completed"]
        return statuses[-1] if statuses else None

    def calls(self, slot: int) -> list[str]:
        return self.sidecars[slot].calls

    async def test_a_stream_cut_off_by_an_overload_restarts_on_the_same_account(self):
        self.sidecars[1].script = ["overloaded"]
        self.assertEqual(await asyncio.wait_for(self.stream(), 20), "completed")
        self.assertEqual(self.calls(1), ["overloaded", "ok"])
        self.assertEqual(self.calls(2), [])

    async def test_hiccups_end_with_an_error_on_the_same_account_not_another_one(self):
        for step in ("503", "overloaded"):
            with self.subTest(step=step):
                self.sidecars[1].script = [step] * 3
                self.sidecars[1].calls.clear()
                with self.assertRaises((litellm.InternalServerError, litellm.ServiceUnavailableError)):
                    await asyncio.wait_for(self.stream(), 30)
                self.assertEqual(self.calls(1), [step] * 3)
                self.assertEqual(self.calls(2), [])

    async def test_a_partly_streamed_answer_is_not_repeated(self):
        self.sidecars[1].script = ["partial"]
        with self.assertRaises(litellm.InternalServerError):
            await asyncio.wait_for(self.stream(), 20)
        self.assertEqual(self.calls(1), ["partial"])
        self.assertEqual(self.calls(2), [])


if __name__ == "__main__":
    unittest.main()
