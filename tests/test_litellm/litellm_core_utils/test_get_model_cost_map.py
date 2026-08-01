"""Tests for loading the model cost map.

The remote map is upstream's copy, so it cannot describe models served by
providers that only exist in this fork.
"""

from unittest.mock import patch

from litellm.litellm_core_utils.get_model_cost_map import (
    FORK_OWNED_MODEL_PREFIXES,
    GetModelCostMap,
    get_model_cost_map,
)

REMOTE_URL = "https://example.invalid/model_prices_and_context_window.json"


def _remote_map_without_fork_models() -> dict:
    backup = GetModelCostMap.load_local_model_cost_map()
    remote = {
        model: info
        for model, info in backup.items()
        if not model.startswith(FORK_OWNED_MODEL_PREFIXES)
    }
    remote["chatgpt/gpt-5.4"] = {
        "litellm_provider": "chatgpt",
        "max_input_tokens": 1050000,
        "mode": "responses",
    }
    return remote


class TestForkOwnedEntries:
    def test_remote_map_keeps_fork_owned_models(self, monkeypatch):
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        GetModelCostMap._fork_owned_entries = None
        remote = _remote_map_without_fork_models()

        with patch.object(
            GetModelCostMap, "fetch_remote_model_cost_map", return_value=remote
        ):
            loaded = get_model_cost_map(url=REMOTE_URL)

        assert "chatgpt/gpt-5.6-luna" in loaded
        assert loaded["chatgpt/gpt-5.6-luna"]["supports_reasoning"] is True

    def test_remote_map_does_not_downgrade_fork_owned_models(self, monkeypatch):
        """A staler remote copy of a fork-owned model must not win."""
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        GetModelCostMap._fork_owned_entries = None
        remote = _remote_map_without_fork_models()

        with patch.object(
            GetModelCostMap, "fetch_remote_model_cost_map", return_value=remote
        ):
            loaded = get_model_cost_map(url=REMOTE_URL)

        assert loaded["chatgpt/gpt-5.4"]["supports_reasoning"] is True

    def test_upstream_models_are_untouched(self, monkeypatch):
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        GetModelCostMap._fork_owned_entries = None
        remote = _remote_map_without_fork_models()
        remote["gpt-5.4"] = {"litellm_provider": "openai", "input_cost_per_token": 123}

        with patch.object(
            GetModelCostMap, "fetch_remote_model_cost_map", return_value=remote
        ):
            loaded = get_model_cost_map(url=REMOTE_URL)

        assert loaded["gpt-5.4"]["input_cost_per_token"] == 123

    def test_only_fork_owned_entries_are_retained_from_the_backup(self):
        GetModelCostMap._fork_owned_entries = None
        entries = GetModelCostMap.get_fork_owned_entries()

        assert entries
        assert all(model.startswith(FORK_OWNED_MODEL_PREFIXES) for model in entries)
