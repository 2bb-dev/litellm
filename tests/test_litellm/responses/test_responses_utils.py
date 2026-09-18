import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponseAPILoggingUtils, ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.utils import Usage


@pytest.fixture
def cache_write_deployment() -> Iterator[str]:
    deployment_id = "openai/cache-write-regression"
    with patch.dict(litellm.model_cost):
        litellm.register_model(
            {
                deployment_id: {
                    "litellm_provider": "openai",
                    "mode": "responses",
                    "input_cost_per_token": 2e-6,
                    "output_cost_per_token": 8e-6,
                    "cache_read_input_token_cost": 0.5e-6,
                    "cache_creation_input_token_cost": 3e-6,
                }
            }
        )
        yield deployment_id
    litellm.utils._invalidate_model_cost_lowercase_map()
    litellm.utils._cached_get_model_info_helper.cache_clear()


class TestResponsesAPIRequestUtils:
    def test_get_optional_params_responses_api(self):
        """Test that optional parameters are correctly processed for responses API"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {
                "temperature": 0.7,
                "max_output_tokens": 100,
                "prompt": {"id": "pmpt_123"},
            }
        )

        # Execute
        result = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model=model,
            responses_api_provider_config=config,
            response_api_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert "temperature" in result
        assert result["temperature"] == 0.7
        assert "max_output_tokens" in result
        assert result["max_output_tokens"] == 100
        assert "prompt" in result
        assert result["prompt"] == {"id": "pmpt_123"}

    def test_get_optional_params_responses_api_unsupported_param(self):
        """Test that unsupported parameters raise an error"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {"temperature": 0.7, "unsupported_param": "value"}
        )

        # Execute and Assert
        with pytest.raises(litellm.UnsupportedParamsError) as excinfo:
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model=model,
                responses_api_provider_config=config,
                response_api_optional_params=optional_params,
            )

        assert "unsupported_param" in str(excinfo.value)
        assert model in str(excinfo.value)

    def test_get_requested_response_api_optional_param(self):
        """Test filtering parameters to only include those in ResponsesAPIOptionalRequestParams"""
        # Setup
        params = {
            "temperature": 0.7,
            "max_output_tokens": 100,
            "prompt": {"id": "pmpt_456"},
            "invalid_param": "value",
            "model": "gpt-4o",  # This is not in ResponsesAPIOptionalRequestParams
        }

        # Execute
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            params
        )

        # Assert
        assert "temperature" in result
        assert "max_output_tokens" in result
        assert "invalid_param" not in result
        assert "model" not in result
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100
        assert result["prompt"] == {"id": "pmpt_456"}

    def test_decode_previous_response_id_to_original_previous_response_id(self):
        """Test decoding a LiteLLM encoded previous_response_id to the original previous_response_id"""
        # Setup
        test_provider = "openai"
        test_model_id = "gpt-4o"
        original_response_id = "resp_abc123"

        # Use the helper method to build an encoded response ID
        encoded_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider=test_provider,
            model_id=test_model_id,
            response_id=original_response_id,
        )

        # Execute
        result = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            encoded_id
        )

        # Assert
        assert result == original_response_id

        # Test with a non-encoded ID
        plain_id = "resp_xyz789"
        result_plain = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            plain_id
        )
        assert result_plain == plain_id

    def test_update_responses_api_response_id_with_model_id_handles_dict(self):
        """Ensure _update_responses_api_response_id_with_model_id works with dict input"""
        responses_api_response = {"id": "resp_abc123"}
        litellm_metadata = {"model_info": {"id": "gpt-4o"}}
        updated = (
            ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=responses_api_response,
                custom_llm_provider="openai",
                litellm_metadata=litellm_metadata,
            )
        )
        assert updated["id"] != "resp_abc123"
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(
            updated["id"]
        )
        assert decoded.get("response_id") == "resp_abc123"
        assert decoded.get("model_id") == "gpt-4o"
        assert decoded.get("custom_llm_provider") == "openai"

    def test_build_decode_container_id_omits_none_model_id(self):
        """model_id=None must not round-trip as the truthy string 'None'."""
        encoded = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id=None,
            container_id="cntr_upstream_abc",
        )
        assert "None" not in base64.b64decode(
            encoded.replace("cntr_", "").encode("utf-8")
        ).decode("utf-8")
        decoded = ResponsesAPIRequestUtils._decode_container_id(encoded)
        assert decoded.get("custom_llm_provider") == "azure"
        assert decoded.get("model_id") is None
        assert decoded.get("response_id") == "cntr_upstream_abc"

    def test_decode_container_id_legacy_literal_none_model_id(self):
        """IDs encoded before the None fix should decode without a bogus model_id."""
        legacy_inner = (
            "litellm:custom_llm_provider:azure;model_id:None;container_id:cntr_x"
        )
        legacy_id = "cntr_" + base64.b64encode(legacy_inner.encode("utf-8")).decode(
            "utf-8"
        )
        decoded = ResponsesAPIRequestUtils._decode_container_id(legacy_id)
        assert decoded.get("model_id") is None
        assert decoded.get("custom_llm_provider") == "azure"
        assert decoded.get("response_id") == "cntr_x"


class TestResponseAPILoggingUtils:
    def test_is_response_api_usage_true(self):
        """Test identification of Response API usage format"""
        # Setup
        usage = {"input_tokens": 10, "output_tokens": 20}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is True

    def test_is_response_api_usage_false(self):
        """Test identification of non-Response API usage format"""
        # Setup
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is False

    def test_transform_response_api_usage_to_chat_usage(self):
        """Test transformation from Response API usage to Chat usage format"""
        # Setup
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert
        assert isinstance(result, Usage)
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.prompt_tokens_details and result.prompt_tokens_details.cached_tokens == 2

    @pytest.mark.parametrize("typed", (False, True))
    @pytest.mark.parametrize("cache_write_tokens", (None, 0, 300))
    def test_transform_response_api_usage_preserves_cache_writes(
        self, typed: bool, cache_write_tokens: int | None
    ) -> None:
        raw_usage = {
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_tokens_details": {
                "cached_tokens": 200,
                **({"cache_write_tokens": cache_write_tokens} if cache_write_tokens is not None else {}),
            },
            "output_tokens_details": {"reasoning_tokens": 20},
        }
        original = json.dumps(raw_usage, sort_keys=True)
        usage = ResponseAPIUsage(**raw_usage) if typed else raw_usage

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 50
        assert result.total_tokens == 1050
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cached_tokens == 200
        assert getattr(result.prompt_tokens_details, "cache_write_tokens", None) == cache_write_tokens
        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.reasoning_tokens == 20
        assert json.dumps(raw_usage, sort_keys=True) == original
        assert result.model_dump()["prompt_tokens_details"].get("cache_write_tokens") == cache_write_tokens

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", (False, True))
    @pytest.mark.parametrize("is_async", (False, True))
    @pytest.mark.parametrize("cache_write_tokens", (None, 0, 300))
    @pytest.mark.parametrize("redact", (False, True))
    async def test_responses_cache_writes_reach_logging_and_cost(
        self, stream: bool, is_async: bool, cache_write_tokens: int | None, cache_write_deployment: str, redact: bool
    ) -> None:
        started = datetime(2026, 9, 7, tzinfo=timezone.utc)
        logger = Logging(
            model="gpt-5.5",
            messages=[],
            stream=stream,
            call_type="aresponses" if is_async else "responses",
            start_time=started,
            litellm_call_id="cache-write-regression",
            function_id="cache-write-regression",
        )
        logger.update_environment_variables(
            litellm_params={
                "metadata": {
                    "model_info": {"id": cache_write_deployment, **litellm.model_cost[cache_write_deployment]}
                },
                "aresponses": is_async,
            },
            optional_params={},
            custom_llm_provider="openai",
        )
        raw_response = {
            "id": "resp_cache_write_regression",
            "object": "response",
            "created_at": int(started.timestamp()),
            "model": "gpt-5.5",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "total_tokens": 1050,
                "input_tokens_details": {
                    "cached_tokens": 200,
                    **({"cache_write_tokens": cache_write_tokens} if cache_write_tokens is not None else {}),
                },
                "output_tokens_details": {"reasoning_tokens": 20},
            },
        }
        logger.model_call_details["standard_callback_dynamic_params"] = {"turn_off_message_logging": redact}
        config = OpenAIResponsesAPIConfig()
        if stream:
            iterator = BaseResponsesAPIStreamingIterator(
                response=httpx.Response(200),
                model="gpt-5.5",
                responses_api_provider_config=config,
                logging_obj=logger,
                custom_llm_provider="openai",
            )
            result = iterator._process_chunk(
                json.dumps({"type": "response.completed", "sequence_number": 1, "response": raw_response})
            )
            assert isinstance(result, ResponseCompletedEvent)
        else:
            result = config.transform_response_api_response(
                model="gpt-5.5", raw_response=httpx.Response(200, json=raw_response), logging_obj=logger
            )

        if is_async:
            await logger.async_success_handler(
                result=result, start_time=started, end_time=started + timedelta(seconds=1)
            )
        else:
            logger.success_handler(result=result, start_time=started, end_time=started + timedelta(seconds=1))

        payload = logger.model_call_details["standard_logging_object"]
        assert payload is not None
        assert payload["prompt_tokens"] == 1000
        assert payload["completion_tokens"] == 50
        assert payload["total_tokens"] == 1050
        usage = payload["response"]["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == 200
        assert usage["prompt_tokens_details"].get("cache_write_tokens") == cache_write_tokens
        assert usage["completion_tokens_details"]["reasoning_tokens"] == 20
        writes = cache_write_tokens or 0
        expected_cost = (800 - writes) * 2e-6 + 200 * 0.5e-6 + writes * 3e-6 + 50 * 8e-6
        assert payload["response_cost"] == pytest.approx(expected_cost)

    def test_transform_response_api_usage_with_none_values(self):
        """Test transformation handles None values properly"""
        # Setup
        usage = {
            "input_tokens": 0,  # Changed from None to 0
            "output_tokens": 20,
            "total_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 20
        assert result.total_tokens == 20

    def test_transform_response_api_usage_calculates_total_from_input_and_output_tokens_if_available(
        self,
    ):
        """Test transformation calculates total_tokens when it's None and input / output tokens are present"""
        # Setup
        usage = {
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": None,
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40  # 15 + 25

    def test_transform_response_api_usage_with_image_tokens(self):
        """Test transformation handles image_tokens from image generation responses.

        Note: _transform_response_api_usage_to_chat_usage() is used by multiple
        endpoints including /images/generations and Response API (/responses),
        both of which use the input_tokens/output_tokens format.

        This tests the fix for image generation responses that include image_tokens
        in both input_tokens_details and output_tokens_details.

        Example from gpt-image-1.5:
        - input: text prompt with 13 tokens
        - output: generated image with 272 image tokens + 100 text tokens
        """
        # Setup - simulating image generation usage from OpenAI
        usage = {
            "input_tokens": 13,
            "output_tokens": 372,
            "total_tokens": 385,
            "input_tokens_details": {
                "image_tokens": 0,
                "text_tokens": 13,
            },
            "output_tokens_details": {
                "image_tokens": 272,
                "text_tokens": 100,
            },
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert - verify basic token counts
        assert isinstance(result, Usage)
        assert result.prompt_tokens == 13
        assert result.completion_tokens == 372
        assert result.total_tokens == 385

        # Assert - verify prompt_tokens_details includes image_tokens and text_tokens
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.image_tokens == 0
        assert result.prompt_tokens_details.text_tokens == 13

        # Assert - verify completion_tokens_details includes image_tokens and text_tokens
        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.image_tokens == 272
        assert result.completion_tokens_details.text_tokens == 100

    def test_transform_response_api_usage_mixed_details(self):
        """Test transformation handles mixed token details (cached + image + audio)."""
        # Setup - hypothetical usage with mixed token types
        usage = {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "input_tokens_details": {
                "cached_tokens": 50,
                "audio_tokens": 10,
                "image_tokens": 20,
                "text_tokens": 20,
            },
            "output_tokens_details": {
                "reasoning_tokens": 30,
                "image_tokens": 100,
                "text_tokens": 50,
                "audio_tokens": 20,
            },
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        # Assert - all token detail types should be preserved
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cached_tokens == 50
        assert result.prompt_tokens_details.audio_tokens == 10
        assert result.prompt_tokens_details.image_tokens == 20
        assert result.prompt_tokens_details.text_tokens == 20

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.reasoning_tokens == 30
        assert result.completion_tokens_details.image_tokens == 100
        assert result.completion_tokens_details.text_tokens == 50
        assert result.completion_tokens_details.audio_tokens == 20

    def test_transform_response_api_usage_with_realtime_keys(self):
        """Realtime input_token_details / output_token_details normalize for Usage."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_token_details": {
                "text_tokens": 8,
                "audio_tokens": 2,
                "cached_tokens": 0,
            },
            "output_token_details": {
                "text_tokens": 12,
                "audio_tokens": 8,
            },
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.text_tokens == 8
        assert result.prompt_tokens_details.audio_tokens == 2

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.text_tokens == 12
        assert result.completion_tokens_details.audio_tokens == 8

    def test_transform_response_api_usage_tokens_details_keep_values(self):
        """Keeps input_tokens_details / output_tokens_details when singular keys are also present."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"text_tokens": 10},
            "output_tokens_details": {"text_tokens": 20},
            "input_token_details": {"text_tokens": 1, "audio_tokens": 99},
            "output_token_details": {"text_tokens": 2, "audio_tokens": 98},
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
            usage
        )

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.text_tokens == 10
        assert result.prompt_tokens_details.audio_tokens is None

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.text_tokens == 20
        assert result.completion_tokens_details.audio_tokens is None


class TestResponsesAPIProviderSpecificParams:
    """
    Tests for fix #19782: provider-specific params (aws_*, vertex_*) should work
    without explicitly passing custom_llm_provider.
    """

    def test_provider_specific_params_no_crash_with_bedrock(self):
        """Test that processing aws_* params with bedrock provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "bedrock",
            "kwargs": {"aws_region_name": "eu-central-1"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            params
        )
        assert "temperature" in result

    def test_provider_specific_params_no_crash_with_openai(self):
        """Test that processing aws_* params with openai provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "openai",
            "kwargs": {"aws_region_name": "eu-central-1"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            params
        )
        assert "temperature" in result

    def test_provider_specific_params_no_crash_with_vertex_ai(self):
        """Test that processing vertex_* params with vertex_ai provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "vertex_ai",
            "kwargs": {"vertex_project": "my-project"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
            params
        )
        assert "temperature" in result


def test_responses_extra_body_forwarded_to_completion_transformation_handler():
    """
    Regression test: extra_body must be forwarded to response_api_handler
    when responses_api_provider_config is None (completion transformation path).

    Before the fix, extra_body was a named parameter of responses() but was
    not passed to litellm_completion_transformation_handler.response_api_handler(),
    so it was silently dropped.
    """
    with (
        patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ),
        patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
        ) as mock_handler,
    ):
        mock_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            extra_body={"custom_key": "custom_value"},
        )

        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        # extra_body can be a positional or keyword arg; check both
        assert call_kwargs.kwargs.get("extra_body") == {"custom_key": "custom_value"}


def test_responses_maps_reasoning_effort_from_litellm_params_to_reasoning():
    """
    Test that when reasoning_effort is passed in kwargs (e.g. from proxy litellm_params)
    and reasoning is None, it is mapped to reasoning before the request.

    Supports per-model reasoning_effort/summary config in proxy for clients like Open WebUI
    that cannot set extra_body.
    """
    with (
        patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ),
        patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
        ) as mock_handler,
    ):
        mock_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            reasoning_effort={"effort": "high", "summary": "detailed"},
        )

        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        responses_api_request = call_kwargs.kwargs.get("responses_api_request", {})
        assert "reasoning" in responses_api_request
        assert responses_api_request["reasoning"] == {
            "effort": "high",
            "summary": "detailed",
        }
