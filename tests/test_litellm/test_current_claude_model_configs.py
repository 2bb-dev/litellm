import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

# Current Anthropic lineup: context, output, input, output, cache read,
# 5m cache write, 1h cache write, and whether the model always thinks.
CURRENT_CLAUDE_MODELS = {
    "claude-fable-5-1": (1000000, 128000, 1e-05, 5e-05, 2.5e-07, 1.25e-05, 2e-05, True),
    "claude-opus-5-5": (1000000, 128000, 4e-06, 2e-05, 2e-07, 5e-06, 8e-06, True),
    "claude-sonnet-5-5": (1000000, 128000, 2e-06, 1e-05, 2e-07, 2.5e-06, 4e-06, True),
    "claude-haiku-4-5-20251001": (200000, 64000, 1e-06, 5e-06, 1e-07, 1.25e-06, 2e-06, False),
}


@pytest.mark.parametrize("model", sorted(CURRENT_CLAUDE_MODELS))
def test_current_claude_cost_and_capabilities_match_backup(model):
    root_map = json.loads(
        (Path(__file__).resolve().parents[2] / "model_prices_and_context_window.json").read_text()
    )
    backup_map = GetModelCostMap.load_local_model_cost_map()
    context, output, input_cost, output_cost, cache_read, cache_write, cache_write_1h, always_thinks = (
        CURRENT_CLAUDE_MODELS[model]
    )

    for cost_map in (root_map, backup_map):
        info = cost_map[model]
        assert info["litellm_provider"] == "anthropic"
        assert info["max_input_tokens"] == context
        assert info["max_output_tokens"] == info["max_tokens"] == output
        assert info["input_cost_per_token"] == pytest.approx(input_cost)
        assert info["output_cost_per_token"] == pytest.approx(output_cost)
        assert info["cache_read_input_token_cost"] == pytest.approx(cache_read)
        assert info["cache_creation_input_token_cost"] == pytest.approx(cache_write)
        assert info["cache_creation_input_token_cost_above_1hr"] == pytest.approx(cache_write_1h)
        if always_thinks:
            assert info["supports_adaptive_thinking"] is True
            assert info["thinking_always_on"] is True
            assert info["supports_forced_tool_use"] is False
            assert info["supports_sampling_params"] is False
        else:
            assert "thinking_always_on" not in info
            assert "supports_forced_tool_use" not in info

    assert backup_map[model] == root_map[model]
