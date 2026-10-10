import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.proxy.anthropic_endpoints import endpoints
from litellm.router_utils.subscription_exhaustion import SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER


@pytest.mark.parametrize(
    "slot_headers,expected",
    [
        pytest.param({SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: "1800003600"}, "1800003600", id="out-of-quota"),
        pytest.param({"retry-after": "1"}, None, id="load"),
    ],
)
@pytest.mark.asyncio
async def test_messages_callers_get_the_out_of_quota_signal_only_for_exhaustion(
    slot_headers: Mapping[str, str], expected: str | None
) -> None:
    request: Final = MagicMock()
    request.headers = {}
    error: Final = litellm.RateLimitError(message="limit", llm_provider="anthropic", model="anthropic/claude-opus-5-5")
    error.litellm_response_headers = httpx.Headers(slot_headers)
    with (
        patch.object(
            endpoints, "_read_request_body", new=AsyncMock(return_value={"model": "anthropic/claude-opus-5-5/pi"})
        ),
        patch.object(
            endpoints.ProxyBaseLLMRequestProcessing, "base_process_llm_request", new=AsyncMock(side_effect=error)
        ),
        patch.object(proxy_server, "proxy_logging_obj") as proxy_logging,
    ):
        proxy_logging.post_call_failure_hook = AsyncMock()
        response: Final = await endpoints.anthropic_response(
            fastapi_response=MagicMock(), request=request, user_api_key_dict=MagicMock()
        )
    assert response.status_code == 429
    assert json.loads(response.body)["error"]["type"] == "rate_limit_error"
    assert response.headers.get(SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER) == expected
