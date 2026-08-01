"""
Regression tests for prompt-cache params dropped by `completion()`.

`prompt_cache_key` and `prompt_cache_retention` are OpenAI Chat Completions
params: they are listed in `OPENAI_CHAT_COMPLETION_PARAMS`, carry defaults in
`DEFAULT_CHAT_COMPLETION_PARAM_VALUES`, and every OpenAI-compatible provider
config reports them as supported. They were still never sent, because
`completion()` had no named parameter for them:

- `get_non_default_completion_params()` excludes anything already known as an
  OpenAI param, so they were not treated as passthrough model params;
- `optional_param_args` is assembled from `completion()`'s named parameters, so
  a caller-supplied value stayed in `**kwargs` and never reached
  `get_optional_params()`.

The visible damage was on ChatGPT subscription routes: with no
`prompt_cache_key` reaching the provider, `ChatGPTConfig` fell back to a
per-request session id, so every request landed on a different upstream prompt
cache partition and prompt caching effectively never hit.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.constants import (
    DEFAULT_CHAT_COMPLETION_PARAM_VALUES,
    OPENAI_CHAT_COMPLETION_PARAMS,
)
from litellm.utils import get_non_default_completion_params

PROMPT_CACHE_PARAMS = ("prompt_cache_key", "prompt_cache_retention")


@pytest.mark.parametrize("param", PROMPT_CACHE_PARAMS)
def test_prompt_cache_params_are_known_openai_params(param):
    assert param in OPENAI_CHAT_COMPLETION_PARAMS
    assert param in DEFAULT_CHAT_COMPLETION_PARAM_VALUES


@pytest.mark.parametrize("param", PROMPT_CACHE_PARAMS)
def test_prompt_cache_params_are_not_passthrough_model_params(param):
    """Known OpenAI params are mapped explicitly, never forwarded as extras."""
    assert param not in get_non_default_completion_params({param: "conversation-123"})


def _mock_openai_client():
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5.4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_raw_response = MagicMock()
    mock_raw_response.headers = {}
    mock_raw_response.parse.return_value = mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response
    return mock_client


def test_completion_forwards_prompt_cache_key_to_provider_request():
    mock_client = _mock_openai_client()

    litellm.completion(
        model="openai/gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        prompt_cache_key="conversation-123",
        prompt_cache_retention="24h",
        api_key="sk-test",
        client=mock_client,
    )

    create_kwargs = mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    assert create_kwargs.get("prompt_cache_key") == "conversation-123"
    assert create_kwargs.get("prompt_cache_retention") == "24h"


def test_completion_omits_prompt_cache_params_when_not_requested():
    mock_client = _mock_openai_client()

    litellm.completion(
        model="openai/gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        client=mock_client,
    )

    create_kwargs = mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    assert "prompt_cache_key" not in create_kwargs
    assert "prompt_cache_retention" not in create_kwargs


def test_responses_bridge_receives_prompt_cache_key():
    """The ChatGPT subscription route reaches the provider through the
    /chat/completions -> /responses bridge, so the mapped optional params must
    still carry the cache key when the bridge builds its request."""
    optional_params = litellm.utils.get_optional_params(
        model="chatgpt/gpt-5.4",
        custom_llm_provider="litellm_proxy",
        prompt_cache_key="conversation-123",
        drop_params=True,
    )
    assert optional_params.get("prompt_cache_key") == "conversation-123"

    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    request = LiteLLMResponsesTransformationHandler().transform_request(
        model="chatgpt/gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
    )
    assert request.get("prompt_cache_key") == "conversation-123"
