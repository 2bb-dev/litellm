"""Catalog coverage for the ChatGPT subscription models."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT_CATALOG = REPO_ROOT / "model_prices_and_context_window.json"
PACKAGED_CATALOG = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# Subscription models served through the ChatGPT slot backends.
SUBSCRIPTION_MODELS = (
    "chatgpt/gpt-5.4-mini",
    "chatgpt/gpt-5.4",
    "chatgpt/gpt-5.5",
    "chatgpt/gpt-5.6-sol",
    "chatgpt/gpt-5.6-terra",
    "chatgpt/gpt-5.6-luna",
)


def _catalog(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _chatgpt_entries(path: Path) -> dict:
    return {name: info for name, info in _catalog(path).items() if name.startswith("chatgpt/")}


@pytest.mark.parametrize("path", (ROOT_CATALOG, PACKAGED_CATALOG), ids=lambda p: p.name)
@pytest.mark.parametrize("model", SUBSCRIPTION_MODELS)
def test_subscription_model_is_in_catalog(path, model):
    assert model in _catalog(path), f"{model} missing from {path.name}"


@pytest.mark.parametrize("path", (ROOT_CATALOG, PACKAGED_CATALOG), ids=lambda p: p.name)
def test_every_chatgpt_entry_declares_reasoning_support(path):
    missing = [
        name
        for name, info in _chatgpt_entries(path).items()
        if info.get("supports_reasoning") is not True
    ]
    assert missing == [], f"chatgpt entries without supports_reasoning: {missing}"


def test_chatgpt_entries_agree_across_catalog_files():
    """The packaged catalog is the shipped copy of the root catalog."""
    assert _chatgpt_entries(ROOT_CATALOG) == _chatgpt_entries(PACKAGED_CATALOG)


@pytest.fixture(scope="module")
def local_model_cost():
    """Load the packaged catalog the way a deployment without network does."""
    previous = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        yield get_model_cost_map(url="")
    finally:
        if previous is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous


@pytest.mark.parametrize("model", SUBSCRIPTION_MODELS)
def test_loaded_model_cost_reports_reasoning_and_responses_mode(local_model_cost, model):
    """`/v1/model/info` reads the loaded map, so assert the loaded view too."""
    info = local_model_cost[model]
    assert info.get("supports_reasoning") is True
    assert info.get("mode") == "responses"
    assert info.get("max_input_tokens")
