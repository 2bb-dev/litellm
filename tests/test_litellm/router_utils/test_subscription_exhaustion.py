from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.router_utils.subscription_exhaustion import (
    SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER,
    subscription_exhausted_until,
)

MODEL: Final = "anthropic/claude-opus-5-5"
UNTIL: Final = 1_800_003_600


def rate_limit(headers: Mapping[str, str] | None = None) -> litellm.RateLimitError:
    return litellm.RateLimitError(message="limit", llm_provider="anthropic", model=MODEL, headers=dict(headers or {}))


def test_the_signal_is_a_429_with_the_header_wherever_the_error_keeps_it() -> None:
    from_response: Final = litellm.RateLimitError(
        message="out",
        llm_provider="anthropic",
        model=MODEL,
        response=httpx.Response(429, headers={SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: f" {UNTIL} "}),
    )
    assert subscription_exhausted_until(from_response) == UNTIL
    from_provider: Final = rate_limit()
    from_provider.litellm_response_headers = httpx.Headers({"X-OpenOrange-Subscription-Exhausted-Until": str(UNTIL)})
    assert subscription_exhausted_until(from_provider) == UNTIL
    assert subscription_exhausted_until(rate_limit({"X-OpenOrange-Subscription-Exhausted-Until": str(UNTIL)})) == UNTIL
    assert subscription_exhausted_until(rate_limit()) is None


@pytest.mark.parametrize("value", ["", "soon", "-1", "1.5", "1" * 13])
def test_a_malformed_reset_is_no_signal(value: str) -> None:
    assert subscription_exhausted_until(rate_limit({SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: value})) is None


def test_only_a_429_carries_the_signal() -> None:
    overloaded: Final = litellm.InternalServerError(
        message="overloaded",
        llm_provider="anthropic",
        model=MODEL,
        response=httpx.Response(529, headers={SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: str(UNTIL)}),
    )
    assert subscription_exhausted_until(overloaded) is None
