"""
OpenOrange: the OpenAI SDK never retries an image generation unless the caller
asks for it. A retry is a second paid generation, and the router's own retry
policy decides everything else (OPENORANGE_PATCHES.md, "Image generation is
never retried by the SDK").
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.openai.openai import OpenAIChatCompletion


def _client() -> MagicMock:
    image = MagicMock()
    image.model_dump.return_value = {"created": 1700000000, "data": [{"b64_json": "aW1hZ2U="}]}
    raw = MagicMock()
    raw.parse.return_value = image
    raw.headers = {}
    client = MagicMock()
    client.images.with_raw_response.generate.return_value = raw
    client.api_key = "test-key"
    client._base_url._uri_reference = "https://api.openai.com"
    return client


@pytest.mark.parametrize(("optional_params", "expected"), [({}, 0), ({"max_retries": 3}, 3)])
def test_image_generation_retries_only_when_the_caller_asks(optional_params, expected):
    handler = OpenAIChatCompletion()
    with patch.object(handler, "_get_openai_client", return_value=_client()) as get_client:
        handler.image_generation(model="gpt-image-2", prompt="An orange cat", timeout=60.0,
                                 optional_params=dict(optional_params), logging_obj=MagicMock(), api_key="test-key")
    assert get_client.call_args.kwargs["max_retries"] == expected


def test_async_image_generation_gets_the_same_retry_default():
    handler = OpenAIChatCompletion()
    # The sync entry point returns the coroutine; a plain mock stands in for it.
    with patch.object(handler, "aimage_generation", new=MagicMock()) as generate:
        handler.image_generation(model="gpt-image-2", prompt="An orange cat", timeout=60.0, optional_params={},
                                 logging_obj=MagicMock(), api_key="test-key", aimg_generation=True)
    assert generate.call_args.kwargs["max_retries"] == 0
