import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.xai.chat.transformation import XAIChatConfig


MODEL = "xai/grok-4.5"


def load_model_maps():
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as file:
        main_cost = json.load(file)
    with open(repo_root / "litellm" / "model_prices_and_context_window_backup.json") as file:
        backup_cost = json.load(file)
    return main_cost, backup_cost


def test_xai_grok_4_5_model_info():
    main_cost, backup_cost = load_model_maps()
    info = main_cost[MODEL]

    assert backup_cost[MODEL] == info
    assert info["litellm_provider"] == "xai"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 500000
    assert info["max_output_tokens"] == 500000
    assert info["max_tokens"] == 500000
    assert info["input_cost_per_token"] == 2e-06
    assert info["cache_read_input_token_cost"] == 3e-07
    assert info["output_cost_per_token"] == 6e-06
    assert info["input_cost_per_token_above_200k_tokens"] == 4e-06
    assert info["cache_read_input_token_cost_above_200k_tokens"] == 6e-07
    assert info["output_cost_per_token_above_200k_tokens"] == 12e-06

    for capability in (
        "supports_function_calling",
        "supports_prompt_caching",
        "supports_reasoning",
        "supports_response_schema",
        "supports_tool_choice",
        "supports_vision",
        "supports_web_search",
    ):
        assert info[capability] is True


def test_xai_grok_4_5_provider_routing_and_params():
    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert routed_model == "grok-4.5"
    assert provider == "xai"

    supported_params = XAIChatConfig().get_supported_openai_params("grok-4.5")
    assert "reasoning_effort" in supported_params
    assert "stop" not in supported_params
    assert "frequency_penalty" not in supported_params


def test_xai_grok_4_5_uses_base_and_long_context_pricing(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())

    base_prompt_cost, base_completion_cost = cost_per_token(
        model=MODEL,
        prompt_tokens=200000,
        completion_tokens=100000,
    )
    long_prompt_cost, long_completion_cost = cost_per_token(
        model=MODEL,
        prompt_tokens=200001,
        completion_tokens=100000,
    )

    assert base_prompt_cost == pytest.approx(0.4)
    assert base_completion_cost == pytest.approx(0.6)
    assert long_prompt_cost == pytest.approx(0.800004)
    assert long_completion_cost == pytest.approx(1.2)
