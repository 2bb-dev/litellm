"""Catalog coverage for the ChatGPT subscription models."""

import json
from pathlib import Path

import pytest

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
