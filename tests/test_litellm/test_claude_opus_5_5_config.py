import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap


def test_claude_opus_5_5_cost_and_capabilities_match_backup():
    root_map = json.loads(
        (Path(__file__).resolve().parents[2] / "model_prices_and_context_window.json").read_text()
    )
    backup_map = GetModelCostMap.load_local_model_cost_map()

    for cost_map in (root_map, backup_map):
        info = cost_map["claude-opus-5-5"]
        assert info["litellm_provider"] == "anthropic"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == info["max_tokens"] == 128000
        assert info["input_cost_per_token"] == pytest.approx(4e-06)
        assert info["output_cost_per_token"] == pytest.approx(2e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(2e-07)
        assert info["cache_creation_input_token_cost"] == pytest.approx(5e-06)
        assert info["cache_creation_input_token_cost_above_1hr"] == pytest.approx(8e-06)
        assert info["supports_adaptive_thinking"] is True
        assert info["thinking_always_on"] is True
        assert info["supports_forced_tool_use"] is False
        assert info["supports_sampling_params"] is False

    assert backup_map["claude-opus-5-5"] == root_map["claude-opus-5-5"]
