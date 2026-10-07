import json
import unittest

from litellm import RateLimitError, Router
from litellm.router_utils.chatgpt_rate_limit import is_chatgpt_quota_error, is_chatgpt_rate_limit
from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment


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


if __name__ == "__main__":
    unittest.main()
