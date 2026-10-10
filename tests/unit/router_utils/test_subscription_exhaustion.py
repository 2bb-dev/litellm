from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_utils.subscription_exhaustion import (
    PI_SLOT_AT_CAPACITY_HEADER,
    SUBSCRIPTION_ACCOUNT_FLAG,
    SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER,
    is_subscription_account_group,
    is_subscription_load,
    pi_slot_at_capacity,
    subscription_exhausted_until,
    subscription_exhaustion_headers,
    subscription_walk_verdict,
)

MODEL: Final = "anthropic/claude-opus-5-5"
UNTIL: Final = 1_800_003_600


def rate_limit(headers: Mapping[str, str] | None = None) -> litellm.RateLimitError:
    return litellm.RateLimitError(message="limit", llm_provider="anthropic", model=MODEL, headers=dict(headers or {}))


def exhausted(until: int = UNTIL) -> litellm.RateLimitError:
    return rate_limit({SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: str(until)})


def broken() -> litellm.InternalServerError:
    return litellm.InternalServerError(message="Provider request failed", llm_provider="anthropic", model=MODEL)


def test_the_signal_is_a_429_with_the_header_wherever_the_error_keeps_it() -> None:
    from_response: Final = litellm.RateLimitError(
        message="out",
        llm_provider="anthropic",
        model=MODEL,
        response=httpx.Response(429, headers={SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: f" {UNTIL} "}),
    )
    assert subscription_exhausted_until(from_response) == UNTIL
    assert dict(subscription_exhaustion_headers(from_response)) == {SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: str(UNTIL)}
    from_provider: Final = rate_limit()
    from_provider.litellm_response_headers = httpx.Headers({"X-OpenOrange-Subscription-Exhausted-Until": str(UNTIL)})
    assert subscription_exhausted_until(from_provider) == UNTIL
    assert subscription_exhausted_until(rate_limit({"X-OpenOrange-Subscription-Exhausted-Until": str(UNTIL)})) == UNTIL
    assert subscription_exhausted_until(rate_limit()) is None
    assert dict(subscription_exhaustion_headers(rate_limit())) == {}


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


def test_load_is_a_429_or_an_overload_without_the_signal() -> None:
    assert is_subscription_load(rate_limit())
    assert not is_subscription_load(exhausted())
    full: Final = rate_limit({PI_SLOT_AT_CAPACITY_HEADER: "1"})
    assert is_subscription_load(full) and pi_slot_at_capacity(full)
    assert not pi_slot_at_capacity(rate_limit())
    assert is_subscription_load(
        litellm.InternalServerError(
            message='AnthropicException - {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
            llm_provider="anthropic",
            model=MODEL,
        )
    )
    assert is_subscription_load(
        MidStreamFallbackError(
            message="rate limited", model=MODEL, llm_provider="anthropic", original_exception=rate_limit()
        )
    )
    for other in (
        broken(),
        litellm.AuthenticationError(message="revoked", llm_provider="anthropic", model=MODEL),
        litellm.Timeout(message="timed out", model=MODEL, llm_provider="anthropic"),
    ):
        assert not is_subscription_load(other)


@pytest.mark.parametrize(
    "answers,verdict",
    [
        pytest.param(("busy", "out1", "broken"), "busy", id="load-wins"),
        pytest.param(("broken", "out1", "full"), "full", id="a-full-slot-is-load"),
        pytest.param(("broken", "out1", "out2"), "out1", id="first-out-of-quota-otherwise"),
        pytest.param(("out2", "broken", "out1"), "out2", id="first-in-walk-order"),
        pytest.param(("broken", "busy"), None, id="no-verdict-without-out-of-quota"),
    ],
)
def test_a_walk_answers_load_first_then_the_first_out_of_quota_answer(
    answers: tuple[str, ...], verdict: str | None
) -> None:
    errors: Final = {
        "busy": rate_limit(),
        "full": rate_limit({PI_SLOT_AT_CAPACITY_HEADER: "1"}),
        "broken": broken(),
        "out1": exhausted(UNTIL + 1),
        "out2": exhausted(UNTIL + 2),
    }
    result: Final = subscription_walk_verdict(tuple(errors[answer] for answer in answers))
    assert result is (None if verdict is None else errors[verdict])


def test_only_flagged_deployments_are_subscription_accounts() -> None:
    def deployment(model_info: Mapping[str, object]) -> dict[str, object]:
        return {"model_name": f"{MODEL}/pi", "litellm_params": {"model": MODEL}, "model_info": dict(model_info)}

    assert is_subscription_account_group([deployment({"id": "slot1", SUBSCRIPTION_ACCOUNT_FLAG: True})])
    assert not is_subscription_account_group([deployment({"id": "slot1"})])
    assert not is_subscription_account_group([deployment({"id": "slot1", SUBSCRIPTION_ACCOUNT_FLAG: "yes"})])
    assert not is_subscription_account_group(None)
