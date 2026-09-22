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
    with (
        patch.dict(litellm.model_cost, copy.deepcopy(litellm.model_cost)),
        patch("litellm.llms.chatgpt.authenticator.Authenticator.get_access_token", return_value="offline"),
    ):
        yield
    litellm.utils._invalidate_model_cost_lowercase_map()
    litellm.utils._cached_get_model_info_helper.cache_clear()


def deployment(provider, pricing, location="model_info", mode="chat"):
    model = f"{provider}/effective-pricing-test"
    params = {"model": model, "api_key": "fake-key"}
    info = {"id": f"{provider}-effective-deployment", "litellm_provider": provider, "mode": mode}
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


@pytest.mark.parametrize("provider", ["openai", "zai", "anthropic", "deepseek", "gemini", "xai", "litellm_proxy"])
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


@pytest.mark.parametrize("provider", ["openai", "xai", "litellm_proxy"])
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
@pytest.mark.parametrize("provider", ["openai", "litellm_proxy", "chatgpt"])
def test_native_responses_uses_logging_start_not_response_created(offset, multiplier, timestamp, provider):
    from litellm.types.llms.openai import ResponsesAPIResponse

    _, model, uid = deployment(provider, GLM_PRICING, mode="responses")
    instant = EXPIRY + timedelta(seconds=offset)
    obj = logging_obj(model, provider, instant.timestamp() if timestamp else instant)
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
    assert litellm.completion_cost(
        completion_response=result,
        model=model,
        custom_pricing=True,
        router_model_id=uid,
        call_type="aresponses",
        request_time=instant.timestamp() if timestamp else instant,
    ) == pytest.approx(0.000152 * multiplier)


@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("rate", [0.0, 0.0001, 0.00018])
def test_custom_character_pricing_uses_deployment_not_backend(location, rate):
    _, model, uid = deployment("elevenlabs", {"input_cost_per_character": rate}, location)
    assert litellm.completion_cost(
        model=model,
        custom_llm_provider="elevenlabs",
        custom_pricing=True,
        router_model_id=uid,
        call_type="speech",
        prompt="x" * 1000,
    ) == pytest.approx(1000 * rate)


@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("duration,expected_seconds", [(0, 0), (1, 10), (10, 10), (60, 60)])
def test_custom_transcription_minimum_duration(location, duration, expected_seconds):
    _, _, uid = deployment(
        "groq",
        {
            "input_cost_per_second": 0.04 / 3600,
            "output_cost_per_second": 0.0,
            "minimum_billable_duration_seconds": 10,
        },
        location,
        mode="audio_transcription",
    )
    costs = litellm.cost_per_token(
        model=uid,
        custom_llm_provider="groq",
        call_type="transcription",
        audio_transcription_file_duration=duration,
    )
    assert sum(costs) == pytest.approx(expected_seconds * 0.04 / 3600)


@pytest.mark.parametrize("provider", ["xai", "litellm_proxy"])
@pytest.mark.parametrize("image_rate", [None, 0.0, 8e-6])
def test_image_tokens_fall_back_to_selected_context_tier(provider, image_rate):
    pricing = dict(GROK_PRICING)
    if image_rate is not None:
        pricing["input_cost_per_image_token"] = image_rate
    _, _, uid = deployment(provider, pricing)
    usage = Usage(
        prompt_tokens=200000, completion_tokens=100, prompt_tokens_details={"cached_tokens": 400, "image_tokens": 1000}
    )
    costs = litellm.cost_per_token(model=uid, custom_llm_provider=provider, usage_object=usage)
    assert sum(costs) == pytest.approx(
        198600 * 4e-6 + 400 * 1e-6 + 1000 * (4e-6 if image_rate is None else image_rate) + 100 * 12e-6
    )


@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("tokens,multiplier", [(272000, 1), (272001, 2)])
def test_cache_write_tier_survives_registration(location, tokens, multiplier):
    pricing = {
        "input_cost_per_token": 10e-6,
        "output_cost_per_token": 50e-6,
        "input_cost_per_token_above_272k_tokens": 20e-6,
        "cache_creation_input_token_cost": 12.5e-6,
        "cache_creation_input_token_cost_above_272k_tokens": 25e-6,
    }
    _, _, uid = deployment("litellm_proxy", pricing, location)
    usage = Usage(prompt_tokens=tokens, completion_tokens=0, prompt_tokens_details={"cache_creation_tokens": 1000})
    assert sum(generic_cost_per_token(uid, usage, "litellm_proxy")) == pytest.approx(
        ((tokens - 1000) * 10e-6 + 1000 * 12.5e-6) * multiplier
    )
    assert not litellm.model_cost["litellm_proxy/effective-pricing-test"].get(
        "cache_creation_input_token_cost_above_272k_tokens"
    )


@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
def test_recurring_pricing_tier_and_finite_period_precedence(location):
    pricing = dict(
        GROK_PRICING,
        cache_creation_input_token_cost=2e-6,
        cache_creation_input_token_cost_above_1hr=3e-6,
        off_peak_pricing={
            "windows": [{"hours_utc": ["01:00-04:00", "06:00-10:00"], "weekdays": [1, 2, 3, 4, 5]}],
            "weekday_timezone": "UTC",
            "input_cost_per_token": 8e-6,
            "output_cost_per_token": 16e-6,
            "cache_read_input_token_cost": 0.0,
            "cache_creation_input_token_cost": 4e-6,
        },
        pricing_periods=[
            {
                "effective_until": "2026-08-16T16:00:00Z",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 0.1e-6,
            }
        ],
    )
    _, _, uid = deployment("deepseek", pricing, location)
    usage = Usage(
        prompt_tokens=200000,
        completion_tokens=100,
        prompt_tokens_details={
            "cached_tokens": 400,
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
    )
    before = copy.deepcopy(litellm.model_cost)
    for instant, rates in [
        (datetime(2026, 9, 7, 2, tzinfo=timezone.utc), (8e-6, 16e-6, 0, 4e-6)),
        (datetime(2026, 9, 7, 4, tzinfo=timezone.utc), (4e-6, 12e-6, 1e-6, 2e-6)),
        (datetime(2026, 9, 6, 2, tzinfo=timezone.utc), (4e-6, 12e-6, 1e-6, 2e-6)),
    ]:
        expected = 198600 * rates[0] + 100 * rates[1] + 400 * rates[2] + 600 * rates[3] + 400 * 3e-6
        assert sum(
            litellm.cost_per_token(model=uid, custom_llm_provider="deepseek", usage_object=usage, request_time=instant)
        ) == pytest.approx(expected)
    small = Usage(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details={"cached_tokens": 400})
    assert sum(
        litellm.cost_per_token(
            model=uid,
            custom_llm_provider="deepseek",
            usage_object=small,
            request_time=datetime(2026, 8, 14, 2, tzinfo=timezone.utc),
        )
    ) == pytest.approx(600e-6 + 200e-6 + 40e-6)
    assert litellm.model_cost == before


def test_dated_one_hour_cache_write_rate():
    _, _, uid = deployment(
        "anthropic",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_creation_input_token_cost": 3.75e-6,
            "cache_creation_input_token_cost_above_1hr": 6e-6,
            "pricing_periods": [
                {
                    "effective_until": "2026-09-09T16:00:00Z",
                    "cache_creation_input_token_cost": 2.5e-6,
                    "cache_creation_input_token_cost_above_1hr": 4e-6,
                }
            ],
        },
    )
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=0,
        prompt_tokens_details={
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
    )
    assert sum(
        litellm.cost_per_token(
            model=uid, custom_llm_provider="anthropic", usage_object=usage, request_time=EXPIRY - timedelta(seconds=1)
        )
    ) == pytest.approx(600 * 2.5e-6 + 400 * 4e-6)


@pytest.mark.parametrize("provider", ["deepseek", "litellm_proxy", "xai"])
@pytest.mark.parametrize("explicit_provider", [False, True])
def test_deployment_id_collision_does_not_select_backend_pricing(provider, explicit_provider):
    uid = "effective-collision-model"
    backend = provider + "/" + uid
    litellm.register_model(
        {
            backend: {
                "litellm_provider": provider,
                "mode": "chat",
                "input_cost_per_token": 9e-6,
                "output_cost_per_token": 9e-6,
            }
        }
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "collision-alias",
                "litellm_params": {"model": backend, "api_key": "fake"},
                "model_info": {
                    "id": uid,
                    "custom_pricing": True,
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 2e-6,
                },
            }
        ]
    )
    assert router.model_list[0]["model_info"]["id"] == uid
    before = copy.deepcopy(litellm.model_cost)
    assert litellm.completion_cost(
        response(backend),
        model=backend,
        custom_llm_provider=provider if explicit_provider else None,
        custom_pricing=True,
        router_model_id=uid,
    ) == pytest.approx(600e-6 + 200e-6)
    assert litellm.get_model_info(backend, custom_llm_provider=provider)["input_cost_per_token"] == 9e-6
    assert litellm.model_cost == before


@pytest.mark.parametrize("finite_tier", [None, 3e-6])
def test_finite_period_preserves_context_tiers_unless_explicitly_overridden(finite_tier):
    period = {"effective_until": "2026-09-09T16:00:00Z", "input_cost_per_token": 1e-6}
    if finite_tier is not None:
        period["input_cost_per_token_above_272k_tokens"] = finite_tier
    pricing = {
        "input_cost_per_token": 2e-6,
        "input_cost_per_token_above_272k_tokens": 4e-6,
        "output_cost_per_token": 0,
        "pricing_periods": [period],
        "off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 8e-6},
    }
    _, _, uid = deployment("litellm_proxy", pricing)
    usage = Usage(prompt_tokens=272001, completion_tokens=0)
    assert sum(
        generic_cost_per_token(uid, usage, "litellm_proxy", request_time=EXPIRY - timedelta(seconds=1))
    ) == pytest.approx(272001 * (4e-6 if finite_tier is None else finite_tier))


def test_dated_cache_breakdown_matches_mixed_ttl_total_and_fast_multiplier():
    _, model, uid = deployment(
        "anthropic",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 0.3e-6,
            "cache_creation_input_token_cost": 3.75e-6,
            "cache_creation_input_token_cost_above_1hr": 6e-6,
            "provider_specific_entry": {"fast": 2},
            "pricing_periods": [
                {
                    "effective_until": "2026-09-09T16:00:00Z",
                    "cache_read_input_token_cost": 0.2e-6,
                    "cache_creation_input_token_cost": 2.5e-6,
                    "cache_creation_input_token_cost_above_1hr": 4e-6,
                }
            ],
        },
    )
    obj = logging_obj(model, "anthropic", EXPIRY - timedelta(seconds=1))
    usage = Usage(
        prompt_tokens=2000,
        completion_tokens=100,
        speed="fast",
        prompt_tokens_details={
            "cached_tokens": 400,
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
    )
    result = litellm.ModelResponse(model=model, usage=usage)
    result._hidden_params["model_id"] = uid
    expected_cache = 400 * 0.2e-6 + 600 * 2.5e-6 + 400 * 4e-6
    assert obj._response_cost_calculator(result) == pytest.approx((600 * 3e-6 + 100 * 15e-6) * 2 + expected_cache)
    assert obj.cost_breakdown["cache_read_cost"] == pytest.approx(400 * 0.2e-6)
    assert obj.cost_breakdown["cache_creation_cost"] == pytest.approx(600 * 2.5e-6 + 400 * 4e-6)


@pytest.mark.parametrize("disable_schedule", [False, True])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_finite_period_null_disables_schedule_but_omission_inherits(disable_schedule, offset):
    period = {"effective_until": "2026-08-16T16:00:00Z", "input_cost_per_token": 1e-6}
    if disable_schedule:
        period["off_peak_pricing"] = None
    _, _, uid = deployment(
        "deepseek",
        {
            "input_cost_per_token": 4e-6,
            "output_cost_per_token": 8e-6,
            "cache_read_input_token_cost": 2e-6,
            "pricing_periods": [period],
            "off_peak_pricing": {
                "hours_utc": "00:00-00:00",
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 4e-6,
                "cache_read_input_token_cost": 0,
            },
        },
    )
    instant = datetime(2026, 8, 16, 16, tzinfo=timezone.utc) + timedelta(seconds=offset)
    expected = (600e-6 + (800e-6 + 800e-6 if disable_schedule else 400e-6)) if offset < 0 else 1200e-6 + 400e-6
    assert sum(
        litellm.cost_per_token(
            model=uid,
            custom_llm_provider="deepseek",
            usage_object=response(uid).usage,
            request_time=instant.timestamp(),
        )
    ) == pytest.approx(expected)


@pytest.mark.parametrize("location", ["model_info", "litellm_params"])
@pytest.mark.parametrize("tokens", [199999, 200000, 200001])
def test_dated_generic_thresholds_include_one_hour_cache_writes(location, tokens):
    period = {
        "effective_until": "2026-09-09T16:00:00Z",
        "input_cost_per_token_above_200k_tokens": 3e-6,
        "output_cost_per_token_above_200k_tokens": 7e-6,
        "cache_read_input_token_cost_above_200k_tokens": 0.25e-6,
        "cache_creation_input_token_cost_above_200k_tokens": 4e-6,
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens": 5e-6,
        "input_cost_per_character": 900,
    }
    pricing = dict(
        GROK_PRICING,
        cache_creation_input_token_cost=2e-6,
        cache_creation_input_token_cost_above_1hr=3e-6,
        pricing_periods=[period],
    )
    _, _, uid = deployment("litellm_proxy", pricing, location)
    usage = Usage(
        prompt_tokens=tokens,
        completion_tokens=100,
        prompt_tokens_details={
            "cached_tokens": 400,
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
    )
    rates = (3e-6, 7e-6, 0.25e-6, 4e-6, 5e-6) if tokens >= 200000 else (2e-6, 6e-6, 0.5e-6, 2e-6, 3e-6)
    expected = (tokens - 1400) * rates[0] + 100 * rates[1] + 400 * rates[2] + 600 * rates[3] + 400 * rates[4]
    assert sum(
        generic_cost_per_token(uid, usage, "litellm_proxy", request_time=EXPIRY - timedelta(seconds=1))
    ) == pytest.approx(expected)


def test_free_speech_output_does_not_fall_through_to_paid_input():
    from litellm.llms.openai.cost_calculation import cost_per_second

    _, _, uid = deployment(
        "openai",
        {
            "input_cost_per_second": 0.1,
            "output_cost_per_second": 0.0,
        },
        mode="audio_speech",
    )
    assert cost_per_second(uid, "openai", 10) == (0.0, 0.0)


@pytest.mark.parametrize("custom_deployment", [False, True])
def test_xai_reported_supplier_cost_preserves_custom_tariff(custom_deployment):
    _, model, uid = deployment("xai", {"input_cost_per_token": 2e-6, "output_cost_per_token": 6e-6})
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cost=0.9)
    costs = litellm.cost_per_token(
        model=uid if custom_deployment else model, custom_llm_provider="xai", usage_object=usage
    )
    assert sum(costs) == pytest.approx(0.0026 if custom_deployment else 0.9)


def test_native_off_peak_prices_one_hour_cache_and_ignores_invalid_rate():
    _, _, uid = deployment(
        "deepseek",
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 6e-6,
            "cache_creation_input_token_cost": 3e-6,
            "cache_creation_input_token_cost_above_1hr": 4e-6,
            "off_peak_pricing": {
                "hours_utc": "23:00-02:00",
                "input_cost_per_token": float("inf"),
                "cache_creation_input_token_cost": 1e-6,
                "cache_creation_input_token_cost_above_1hr": 2e-6,
            },
        },
    )
    usage = Usage(
        prompt_tokens=2000,
        completion_tokens=0,
        prompt_tokens_details={
            "cache_creation_tokens": 1000,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
        },
    )
    costs = generic_cost_per_token(uid, usage, "deepseek", request_time=datetime(2026, 9, 7, 1, tzinfo=timezone.utc))
    assert sum(costs) == pytest.approx(1000 * 2e-6 + 600 * 1e-6 + 400 * 2e-6)
