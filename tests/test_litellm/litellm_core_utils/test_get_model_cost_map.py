"""Tests for loading the model cost map.

The remote map is upstream's copy, so it cannot describe models served by
providers that only exist in this fork.
"""

from litellm.litellm_core_utils.get_model_cost_map import (
    FORK_OWNED_MODEL_PREFIXES,
    GetModelCostMap,
    get_model_cost_map,
    merge_fork_owned_entries,
)

REMOTE_URL = "https://example.invalid/model_prices_and_context_window.json"

FORK_ENTRY = {
    "litellm_provider": "chatgpt",
    "max_input_tokens": 372000,
    "mode": "responses",
    "supports_reasoning": True,
}


class TestMergeForkOwnedEntries:
    def test_model_missing_from_the_fetched_map_is_added(self):
        merged = merge_fork_owned_entries({}, {"chatgpt/gpt-5.6-luna": FORK_ENTRY})

        assert merged["chatgpt/gpt-5.6-luna"]["supports_reasoning"] is True

    def test_fields_missing_from_a_fetched_entry_are_filled(self):
        fetched = {"chatgpt/gpt-5.6-luna": {"litellm_provider": "chatgpt", "mode": "responses"}}

        merged = merge_fork_owned_entries(fetched, {"chatgpt/gpt-5.6-luna": FORK_ENTRY})

        assert merged["chatgpt/gpt-5.6-luna"]["supports_reasoning"] is True

    def test_fetched_values_are_never_overridden(self):
        """A custom catalog keeps its own pricing and limits."""
        fetched = {
            "chatgpt/gpt-5.6-luna": {
                "max_input_tokens": 999,
                "input_cost_per_token": 0.000123,
            }
        }

        merged = merge_fork_owned_entries(fetched, {"chatgpt/gpt-5.6-luna": FORK_ENTRY})

        assert merged["chatgpt/gpt-5.6-luna"]["max_input_tokens"] == 999
        assert merged["chatgpt/gpt-5.6-luna"]["input_cost_per_token"] == 0.000123
        assert merged["chatgpt/gpt-5.6-luna"]["supports_reasoning"] is True

    def test_other_providers_are_untouched(self):
        fetched = {"gpt-5.4": {"litellm_provider": "openai", "input_cost_per_token": 123}}

        merged = merge_fork_owned_entries(fetched, {"chatgpt/gpt-5.6-luna": FORK_ENTRY})

        assert merged["gpt-5.4"] == fetched["gpt-5.4"]

    def test_inputs_are_not_mutated(self):
        fetched: dict = {}

        merge_fork_owned_entries(fetched, {"chatgpt/gpt-5.6-luna": FORK_ENTRY})

        assert fetched == {}


class TestForkOwnedEntries:
    def test_only_fork_owned_models_are_retained_from_the_backup(self):
        entries = GetModelCostMap.get_fork_owned_entries()

        assert entries
        assert all(model.startswith(FORK_OWNED_MODEL_PREFIXES) for model in entries)

    def test_subscription_models_survive_a_remote_map_without_them(self, monkeypatch):
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        backup = GetModelCostMap.load_local_model_cost_map()
        remote = {
            model: info
            for model, info in backup.items()
            if not model.startswith(FORK_OWNED_MODEL_PREFIXES)
        }
        remote["chatgpt/gpt-5.4"] = {"litellm_provider": "chatgpt", "mode": "responses"}

        loaded = get_model_cost_map(url=REMOTE_URL, fetch_remote=lambda _url: remote)

        assert loaded["chatgpt/gpt-5.6-luna"]["supports_reasoning"] is True
        assert loaded["chatgpt/gpt-5.4"]["supports_reasoning"] is True
