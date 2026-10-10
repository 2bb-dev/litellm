from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.router_utils.subscription_exhaustion import SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER


@pytest.mark.parametrize(
    "slot_headers,expected",
    [
        pytest.param({SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: "1800003600"}, "1800003600", id="out-of-quota"),
        pytest.param({"retry-after": "1"}, None, id="load"),
    ],
)
@pytest.mark.asyncio
async def test_an_out_of_quota_subscription_answer_keeps_its_signal_for_the_caller(
    slot_headers: Mapping[str, str], expected: str | None
) -> None:
    proxy_logging: Final = MagicMock()
    proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
    proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})
    error: Final = litellm.RateLimitError(message="limit", llm_provider="anthropic", model="anthropic/claude-opus-5-5")
    error.litellm_response_headers = httpx.Headers(slot_headers)
    with pytest.raises(ProxyException) as raised:
        await ProxyBaseLLMRequestProcessing(data={})._handle_llm_api_exception(
            e=error, user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"), proxy_logging_obj=proxy_logging
        )
    assert raised.value.code == "429"
    assert raised.value.headers.get(SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER) == expected
