"""Tests for ChatGPT subscription Chat Completions transformation."""

from unittest.mock import MagicMock

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.openai.openai import OpenAIConfig


def test_chatgpt_normalizes_system_messages_without_reordering_content():
    messages = [
        {"role": "system", "content": "application instructions", "name": "app"},
        {"role": "developer", "content": "existing developer instructions"},
        {"role": "user", "content": "hello"},
    ]

    request = ChatGPTConfig().transform_request(
        model="gpt-5.6",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["messages"] == [
        {"role": "developer", "content": "application instructions", "name": "app"},
        {"role": "developer", "content": "existing developer instructions"},
        {"role": "user", "content": "hello"},
    ]
    assert messages[0]["role"] == "system"


def test_openai_chat_preserves_system_messages():
    request = OpenAIConfig().transform_request(
        model="gpt-5.6",
        messages=[{"role": "system", "content": "application instructions"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["messages"][0]["role"] == "system"


def test_prompt_cache_key_sets_session_id_header():
    config = ChatGPTConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_account_id.return_value = "acct-123"
    prompt_cache_key = "conversation-123"

    for call_id in ("call-1", "call-2"):
        litellm_params = {"litellm_call_id": call_id}
        optional_params = {"prompt_cache_key": prompt_cache_key}
        headers = config.validate_environment(
            headers={"session_id": f"explicit-{call_id}"},
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key="access-123",
        )
        request = config.transform_request(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert request["prompt_cache_key"] == prompt_cache_key
        assert headers["session_id"] == prompt_cache_key


def test_without_prompt_cache_key_preserves_session_id_header():
    config = ChatGPTConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_account_id.return_value = "acct-123"
    litellm_params = {"litellm_session_id": "fallback-session"}

    headers = config.validate_environment(
        headers={"session_id": "explicit-session"},
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params=litellm_params,
        api_key="access-123",
    )

    assert headers["session_id"] == "explicit-session"
