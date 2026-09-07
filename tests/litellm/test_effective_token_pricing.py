"""Offline deployment-level accounting, including delayed logging at expiry."""

import copy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage

EXPIRY = datetime(2026, 9, 9, 16, tzinfo=timezone.utc)
GLM_PRICING = {
    "input_cost_per_token": 0.15e-6,
    "output_cost_per_token": 0.5e-6,
    "cache_read_input_token_cost": 0.03e-6,
    "pricing_periods": [
        {
            "effective_until": "2026-09-09T16:00:00Z",
            "input_cost_per_token": 0.075e-6,
            "output_cost_per_token": 0.25e-6,
            "cache_read_input_token_cost": 0.015e-6,
        }
    ],
}
GROK_PRICING = {
    "input_cost_per_token": 2e-6,
    "output_cost_per_token": 6e-6,
    "cache_read_input_token_cost": 0.5e-6,
    "input_cost_per_token_above_200k_tokens": 4e-6,
    "output_cost_per_token_above_200k_tokens": 12e-6,
    "cache_read_input_token_cost_above_200k_tokens": 1e-6,
    "pricing_tier_threshold_inclusive": True,
}


@pytest.fixture(autouse=True)
def isolated_registry():
    with patch.dict(litellm.model_cost, copy.deepcopy(litellm.model_cost)):
        yield
    litellm.utils._invalidate_model_cost_lowercase_map()
    litellm.utils._cached_get_model_info_helper.cache_clear()


def deployment(provider, pricing, location="model_info"):
    model = f"{provider}/effective-pricing-test"
    params = {"model": model, "api_key": "fake-key"}
    info = {"id": f"{provider}-effective-deployment", "litellm_provider": provider, "mode": "chat"}
    (params if location == "litellm_params" else info).update(copy.deepcopy(pricing))
    router = litellm.Router(
        model_list=[
            {
                "model_name": "effective-alias",
                "litellm_params": params,
                "model_info": info,
            },
            {
                "model_name": "sibling-alias",
                "litellm_params": {"model": model, "api_key": "fake-key"},
                "model_info": {
                    "id": f"{provider}-sibling",
                    "input_cost_per_token": 9e-6,
                    "output_cost_per_token": 9e-6,
                    "cache_read_input_token_cost": 9e-6,
                },
            },
        ]
    )
    return router, model, info["id"]


def response(model, prompt_tokens=1000):
    return litellm.ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=100,
            total_tokens=prompt_tokens + 100,
            prompt_tokens_details={"cached_tokens": 400},
        ),
    )


def logging_obj(model, provider, instant):
    obj = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=instant,
        litellm_call_id="period-test",
        function_id="period-test",
    )
    obj.model_call_details["custom_llm_provider"] = provider
    obj.litellm_params = {"metadata": {"model_info": litellm.model_cost[f"{provider}-effective-deployment"]}}
    obj.optional_params = {}
    return obj


@pytest.mark.parametrize("provider", ["openai", "zai"])
@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("offset,multiplier", [(-1, 0.5), (0, 1), (1, 1)])
def test_deployment_request_time_and_cache_cost(provider, location, offset, multiplier):
    _, model, uid = deployment(provider, GLM_PRICING, location)
    obj = logging_obj(model, provider, EXPIRY + timedelta(microseconds=offset))
    result = response(model)
    result._hidden_params["model_id"] = uid
    before = copy.deepcopy(litellm.model_cost)
    expected = (600 * 0.15e-6 + 400 * 0.03e-6 + 100 * 0.5e-6) * multiplier
    assert obj._response_cost_calculator(result) == pytest.approx(expected)
    assert litellm.model_cost == before
    assert (
        litellm.get_model_info(uid, custom_llm_provider=provider)["pricing_periods"] == GLM_PRICING["pricing_periods"]
    )
    assert not litellm.get_model_info(f"{provider}-sibling", custom_llm_provider=provider).get("pricing_periods")


@pytest.mark.parametrize("provider", ["openai", "xai"])
@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("tokens,multiplier", [(199999, 1), (200000, 2), (200001, 2)])
def test_deployment_inclusive_tier(provider, location, tokens, multiplier):
    _, model, uid = deployment(provider, GROK_PRICING, location)
    obj = logging_obj(model, provider, EXPIRY)
    result = response(model, tokens)
    result._hidden_params["model_id"] = uid
    expected = ((tokens - 400) * 2e-6 + 400 * 0.5e-6 + 100 * 6e-6) * multiplier
    assert obj._response_cost_calculator(result) == pytest.approx(expected)


def test_existing_exclusive_threshold_unchanged():
    pricing = dict(GROK_PRICING, pricing_tier_threshold_inclusive=False)
    _, _, uid = deployment("openai", pricing)
    costs = generic_cost_per_token(uid, response(uid, 200000).usage, "openai")
    assert sum(costs) == pytest.approx(199600 * 2e-6 + 400 * 0.5e-6 + 100 * 6e-6)


def test_effective_start_and_explicit_time_override():
    pricing = copy.deepcopy(GLM_PRICING)
    pricing["pricing_periods"][0]["effective_from"] = "2026-09-09T15:00:00Z"
    _, model, uid = deployment("openai", pricing)
    base = 600 * 0.15e-6 + 400 * 0.03e-6 + 100 * 0.5e-6
    obj = logging_obj(model, "openai", EXPIRY + timedelta(days=1))
    for instant, multiplier in [
        (EXPIRY - timedelta(hours=1, microseconds=1), 1),
        (EXPIRY - timedelta(hours=1), 0.5),
        (EXPIRY, 1),
    ]:
        assert litellm.completion_cost(
            response(model),
            model=model,
            custom_llm_provider="openai",
            router_model_id=uid,
            litellm_logging_obj=obj,
            custom_pricing=True,
            request_time=instant,
        ) == pytest.approx(base * multiplier)


def test_passthrough_request_start_not_logging_time():
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
        OpenAIPassthroughLoggingHandler,
    )

    _, _, uid = deployment("openai", GLM_PRICING)
    obj = logging_obj(uid, "openai", EXPIRY - timedelta(seconds=1))
    result = response(uid)
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler.get_standard_logging_object_payload",
        return_value={},
    ):
        OpenAIPassthroughLoggingHandler()._create_response_logging_payload(
            result, uid, {}, obj.start_time, EXPIRY + timedelta(seconds=1), obj
        )
    assert obj.model_call_details["response_cost"] == pytest.approx(600 * 0.075e-6 + 400 * 0.015e-6 + 100 * 0.25e-6)


@pytest.mark.parametrize(
    "periods,match",
    [
        ([{"effective_until": "2026-09-09T16:00:00"}], "timezone"),
        ([{"effective_from": "2026-09-10T00:00:00Z", "effective_until": "2026-09-09T00:00:00Z"}], "precede"),
        ([{}, {}], "Overlapping"),
    ],
)
def test_invalid_periods_fail_explicitly(periods, match):
    _, _, uid = deployment("openai", dict(GLM_PRICING, pricing_periods=periods))
    with pytest.raises(ValueError, match=match):
        generic_cost_per_token(uid, response(uid).usage, "openai", request_time=EXPIRY)


def test_out_of_order_requests_preserve_registry_and_zero_override():
    pricing = copy.deepcopy(GLM_PRICING)
    pricing["pricing_periods"][0]["cache_read_input_token_cost"] = 0.0
    _, _, uid = deployment("openai", pricing)
    before = copy.deepcopy(litellm.model_cost)
    for delta, expected in [(1, 0.000152), (-1, 0.000070), (0, 0.000152), (-1, 0.000070)]:
        costs = generic_cost_per_token(
            uid, response(uid).usage, "openai", request_time=EXPIRY + timedelta(seconds=delta)
        )
        assert sum(costs) == pytest.approx(expected)
    assert litellm.model_cost == before


@pytest.mark.parametrize("offset,multiplier", [(-1, 0.5), (0, 1), (1, 1)])
@pytest.mark.parametrize("timestamp", [False, True])
def test_native_responses_uses_logging_start_not_response_created(offset, multiplier, timestamp):
    from litellm.types.llms.openai import ResponsesAPIResponse

    _, model, uid = deployment("openai", GLM_PRICING)
    instant = EXPIRY + timedelta(seconds=offset)
    obj = logging_obj(model, "openai", instant.timestamp() if timestamp else instant)
    obj.call_type = "aresponses"
    obj.stream = True
    obj.litellm_params = {"litellm_metadata": {"model_info": litellm.model_cost[uid]}}
    result = ResponsesAPIResponse(
        id="resp-period-test",
        created_at=(EXPIRY + timedelta(days=1)).timestamp(),
        model=model,
        output=[],
        usage={
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_tokens_details": {"cached_tokens": 400},
        },
    )
    assert obj._response_cost_calculator(result) == pytest.approx(0.000152 * multiplier)
