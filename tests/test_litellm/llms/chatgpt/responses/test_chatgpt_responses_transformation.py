"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import hashlib
import json
import os
import sys
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.common_utils import get_chatgpt_default_instructions
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.main import responses_api_bridge_check
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize("is_async,prompt_swap", [(False, False), (True, False), (False, True)])
    @pytest.mark.asyncio
    async def test_public_responses_ignores_chat_bridge_for_chatgpt(
        self, tmp_path, monkeypatch, is_async, prompt_swap, respx_mock
    ):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        monkeypatch.setenv("CHATGPT_API_BASE", "https://chatgpt.fixture")
        (tmp_path / "auth.json").write_text(
            json.dumps({"access_token": "fixture-token", "account_id": "fixture-account", "expires_at": 4102444800})
        )
        captured: Final = []
        payload: Final = {
            "id": "resp_fixture",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [],
        }

        def backend(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.path.endswith("/chat/completions"):
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl_fixture",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-6-sol",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps({'type': 'response.completed', 'response': payload})}\n\n",
            )

        kwargs: Final = dict(
            model="chatgpt/gpt-6-sol",
            input="hello",
            use_chat_completions_api=True,
            extra_headers={"Authorization": "caller", "ChatGPT-Account-Id": "caller"},
            extra_body={"model": "caller-model", "store": True, "service_tier": "caller"},
            num_retries=0,
        )
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj: Final = MagicMock(spec=Logging)
        logging_obj.model_call_details = {}
        logging_obj.caching_details = None
        logging_obj.completion_start_time = None
        logging_obj.dynamic_success_callbacks = []
        logging_obj.should_run_prompt_management_hooks.return_value = True
        logging_obj.get_chat_completion_prompt.return_value = (
            "chatgpt/gpt-6-sol",
            [{"role": "user", "content": "hello"}],
            {},
        )
        request_kwargs: Final = {
            **kwargs,
            **(
                {"model": "openai/gpt-6-sol", "prompt_id": "fixture-prompt", "litellm_logging_obj": logging_obj}
                if prompt_swap
                else {}
            ),
        }
        respx_mock.route(host="chatgpt.fixture").mock(side_effect=backend)
        if is_async:
            await litellm.aresponses(**request_kwargs)
        else:
            litellm.responses(**request_kwargs)
        assert len(captured) == 1
        request: Final = captured[0]
        assert request.method == "POST"
        assert request.url.path == "/responses"
        assert request.headers["authorization"] == "Bearer fixture-token"
        assert request.headers["chatgpt-account-id"] == "fixture-account"
        body: Final = json.loads(request.content)
        assert body["model"] == "gpt-6-sol"
        assert body["store"] is False
        assert body["stream"] is True
        assert "service_tier" not in body

    @pytest.mark.parametrize(
        "is_async,stream,api_base,response_format",
        [
            (is_async, stream, api_base, "sse")
            for is_async in (False, True)
            for stream in (False, True)
            for api_base in ("https://chatgpt.fixture", "https://chatgpt.fixture?route=create")
        ]
        + [(False, False, "https://chatgpt.fixture", "json")],
    )
    @pytest.mark.asyncio
    async def test_final_http_boundary_preserves_chatgpt_provider_policy(
        self, tmp_path, monkeypatch, is_async, stream, api_base, response_format
    ):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        monkeypatch.setenv("CHATGPT_API_BASE", "https://chatgpt.fixture")
        (tmp_path / "auth.json").write_text(
            json.dumps(
                {
                    "access_token": "fixture-provider-token",
                    "account_id": "fixture-provider-account",
                    "expires_at": 4102444800,
                }
            )
        )
        captured = []
        payload = {
            "id": "resp_fixture",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [],
        }

        def backend(request):
            captured.append(request)
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json" if response_format == "json" else "text/event-stream",
                    "request-id": "fixture-other-request",
                    "x-ratelimit-remaining-requests": "101",
                    "x-ratelimit-remaining-tokens": "102",
                    "x-ratelimit-limit-requests": "103",
                    "x-ratelimit-limit-tokens": "104",
                    "x-ratelimit-reset-requests": "105",
                    "x-ratelimit-reset-tokens": "106",
                    "x-request-id": "fixture-request",
                    "retry-after": "7",
                    "set-cookie": "fixture-cookie=private",
                    "x-codex-turn-state": "fixture-turn-state",
                    "llm_provider-set-cookie": "fixture-prefixed-cookie=private",
                    "x-private-backend-header": "private",
                },
                text=(
                    json.dumps(payload)
                    if response_format == "json"
                    else f"data: {json.dumps({'type': 'response.completed', 'response': payload})}\n\n"
                ),
            )

        from litellm.litellm_core_utils.credential_ownership import CONTEXT as CREDENTIAL_CONTEXT
        from litellm.litellm_core_utils.credential_ownership import DISPATCH, select_credential
        from litellm.litellm_core_utils.terminal_receipt_evidence import Terminal
        from litellm.litellm_core_utils.terminal_receipt_hooks import CONTEXT, STAMP, Session

        registration_path: Final = tmp_path / "registration.json"
        registration_path.write_text(
            json.dumps(
                {
                    "v": 1,
                    "source": "byok",
                    "registration_id": str(uuid4()),
                    "registration_revision": str(uuid4()),
                    "account_sha256": hashlib.sha256(b"fixture-provider-account").hexdigest(),
                    "api_base": "https://chatgpt.fixture",
                }
            )
        )
        registration_path.chmod(0o600)
        monkeypatch.setenv("OPENORANGE_CHATGPT_CREDENTIAL_REGISTRATION_FILE", str(registration_path))
        session: Final = Session(
            root=MagicMock(),
            attempt_id=str(uuid4()),
            terminal=Terminal(deployment_id="fixture", model="chatgpt/gpt-6-sol", provider="chatgpt"),
            base=None,
            begun=True,
        )
        metadata: Final = {
            CONTEXT: session,
            CREDENTIAL_CONTEXT: select_credential({"litellm_params": {"model": "chatgpt/gpt-6-sol"}}, {}, "fixture"),
        }
        config = ChatGPTResponsesAPIConfig()
        from litellm.litellm_core_utils.terminal_receipt_oauth import AccountSnapshot
        from litellm.llms.chatgpt.authenticator import Authenticator

        class RotatingAuthenticator(Authenticator):
            def get_account_snapshot(self, access_token: str | None) -> AccountSnapshot | None:
                snapshot: Final = super().get_account_snapshot(access_token)
                (tmp_path / "auth.json").write_text(
                    json.dumps(
                        {
                            "access_token": "fixture-refreshed-token",
                            "account_id": "fixture-provider-account",
                            "expires_at": 4102444800,
                        }
                    )
                )
                return snapshot

        config.authenticator = RotatingAuthenticator()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {STAMP: session}
        logging_obj.dynamic_success_callbacks = []
        protected_headers: Final = (
            "authorization",
            "proxy-authorization",
            "cookie",
            "chatgpt-account-id",
            "openai-organization",
            "openai-project",
            "originator",
            "user-agent",
            "host",
            "content-length",
            "transfer-encoding",
            "content-type",
            "accept",
        )
        caller_headers: Final = {
            "aUtHoRiZaTiOn": "Bearer fixture-caller-token",
            "chatgpt-account-id": "fixture-caller-account",
            "originator": "caller",
            "User-Agent": "caller",
            "Cookie": "fixture-cookie=caller",
            "session-id": "fixture-session",
            "thread-id": "fixture-thread",
            "x-codex-turn-state": "fixture-provider-replay",
            **{key: "caller-protected" for key in protected_headers if key != "authorization"},
            **{key.replace("-", "_"): "caller-underscore" for key in protected_headers},
            "session_id": "fixture-underscore-session",
        }
        kwargs = dict(
            model="gpt-6-sol",
            input="hello",
            responses_api_provider_config=config,
            response_api_optional_request_params={"stream": stream, "extra_headers": caller_headers},
            custom_llm_provider="chatgpt",
            litellm_params=GenericLiteLLMParams(metadata=metadata, api_base=api_base),
            logging_obj=logging_obj,
            extra_headers=caller_headers,
            extra_body={
                "model": "gpt-other-expensive",
                "unknown_backend_option": "caller",
                "extra_headers": {"Authorization": "nested"},
                "store": True,
                "stream": False,
                "include": [],
                "prompt_cache_key": "fixture-cache" if response_format == "sse" else "",
            },
        )
        transport = httpx.MockTransport(backend)
        handler = BaseLLMHTTPHandler()
        if is_async:
            async with httpx.AsyncClient(transport=transport) as http_client:
                client = AsyncHTTPHandler()
                await client.client.aclose()
                client.client = http_client
                result = await handler.async_response_api_handler(**kwargs, client=client)
        else:
            with httpx.Client(transport=transport) as http_client:
                result = handler.response_api_handler(**kwargs, client=HTTPHandler(client=http_client))
        assert len(captured) == 1
        request = captured[0]
        assert not any(value in ("caller-protected", "caller-underscore") for value in request.headers.values())
        assert request.headers["authorization"] == "Bearer fixture-refreshed-token"
        sent_digest: Final = hashlib.sha256(request.headers["authorization"].removeprefix("Bearer ").encode()).digest()
        assert session.oauth.key_digest == sent_digest
        assert logging_obj.model_call_details[DISPATCH].key_digest == sent_digest
        assert request.headers["chatgpt-account-id"] == "fixture-provider-account"
        assert request.headers["originator"] != "caller"
        assert request.headers["user-agent"] != "caller"
        assert "cookie" not in request.headers
        assert request.headers["session-id"] == "fixture-session"
        assert request.headers["thread-id"] == "fixture-thread"
        assert request.headers["x-codex-turn-state"] == "fixture-provider-replay"
        assert request.headers["session_id"] == (
            "fixture-cache" if response_format == "sse" else "fixture-underscore-session"
        )
        body = json.loads(request.content)
        assert logging_obj.pre_call.call_args.kwargs["additional_args"]["complete_input_dict"] == body
        assert body["model"] == "gpt-6-sol"
        assert "unknown_backend_option" not in body
        assert "extra_headers" not in body
        assert body["store"] is False
        assert body["stream"] is True
        assert "reasoning.encrypted_content" in body["include"]
        assert body["prompt_cache_key"] == ("fixture-cache" if response_format == "sse" else "")
        headers = result._hidden_params["additional_headers"]
        assert headers["llm_provider-x-request-id"] == "fixture-request"
        assert headers["llm_provider-retry-after"] == "7"
        assert headers["llm_provider-request-id"] == "fixture-other-request"
        for key, value in (
            ("x-ratelimit-remaining-requests", "101"),
            ("x-ratelimit-remaining-tokens", "102"),
            ("x-ratelimit-limit-requests", "103"),
            ("x-ratelimit-limit-tokens", "104"),
            ("x-ratelimit-reset-requests", "105"),
            ("x-ratelimit-reset-tokens", "106"),
        ):
            assert headers[key] == value
            assert headers[f"llm_provider-{key}"] == value
        assert "llm_provider-set-cookie" not in headers
        assert "llm_provider-x-codex-turn-state" not in headers
        assert "llm_provider-x-private-backend-header" not in headers

    def test_compact_http_request_preserves_its_body_contract(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
        (tmp_path / "auth.json").write_text(
            json.dumps({"access_token": "fixture-token", "account_id": "fixture-account", "expires_at": 4102444800})
        )
        captured = []

        def backend(request):
            captured.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-6-sol",
                    "output": [],
                },
            )

        with httpx.Client(transport=httpx.MockTransport(backend)) as http_client:
            BaseLLMHTTPHandler().compact_response_api_handler(
                model="gpt-6-sol",
                input="hello",
                responses_api_provider_config=ChatGPTResponsesAPIConfig(),
                response_api_optional_request_params={},
                litellm_params=GenericLiteLLMParams(api_base="https://chatgpt.fixture"),
                logging_obj=MagicMock(),
                custom_llm_provider="chatgpt",
                client=HTTPHandler(client=http_client),
            )
        assert captured[0].url.path == "/responses/compact"
        assert json.loads(captured[0].content) == {"model": "gpt-6-sol", "input": "hello"}

    def test_system_input_is_normalized_without_reordering_content(self):
        input_items = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "application instructions"}],
            },
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "existing developer instructions"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]

        request = ChatGPTResponsesAPIConfig().transform_responses_api_request(
            model="gpt-5.6",
            input=input_items,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "application instructions"}],
            },
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "existing developer instructions"}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]
        assert input_items[0]["role"] == "system"

    def test_openai_responses_preserves_system_input(self):
        request = OpenAIResponsesAPIConfig().transform_responses_api_request(
            model="gpt-5.6",
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "application instructions"}],
                }
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_non_stream_caller_buffers_provider_forced_sse(self):
        config = ChatGPTResponsesAPIConfig()
        config.validate_environment = MagicMock(return_value={})
        config.get_complete_url = MagicMock(
            return_value="https://chatgpt.example.com/responses"
        )
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.5",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
        }
        sse_body = (
            "\n\n".join(
                [
                    f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                    "data: [DONE]",
                ]
            )
            + "\n\n"
        )
        client = AsyncHTTPHandler()
        client.post = AsyncMock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse_body,
                request=httpx.Request(
                    "POST", "https://chatgpt.example.com/responses"
                ),
            )
        )
        logging_obj = MagicMock()
        logging_obj.dynamic_success_callbacks = []
        # Native usage observation reads the call context from a real mapping.
        logging_obj.model_call_details = {}
        handler = BaseLLMHTTPHandler()
        handler._call_agentic_completion_hooks = AsyncMock(return_value=None)

        result = await handler.async_response_api_handler(
            model="gpt-5.5",
            input="Reply with ok.",
            responses_api_provider_config=config,
            response_api_optional_request_params={},
            custom_llm_provider="chatgpt",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=logging_obj,
            client=client,
        )

        request_kwargs = client.post.call_args.kwargs
        assert request_kwargs["stream"] is True
        assert json.loads(request_kwargs["data"])["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with ok."}],
            }
        ]
        assert result.output_text == "ok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("caller_instructions", [None, "Follow the agent rules."])
    async def test_responses_echo_caller_instructions_not_codex_prompt(
        self, stream, caller_instructions
    ):
        config = ChatGPTResponsesAPIConfig()
        config.validate_environment = MagicMock(return_value={})
        config.get_complete_url = MagicMock(
            return_value="https://chatgpt.example.com/responses"
        )

        async def post(**kwargs):
            # The backend echoes the instructions it received in every response body.
            response_payload = {
                "id": "resp_test",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": "gpt-5.5",
                "instructions": json.loads(kwargs["data"])["instructions"],
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
            events = [
                {"type": "response.created", "response": {**response_payload, "status": "in_progress", "output": []}},
                {"type": "response.completed", "response": response_payload},
            ]
            sse_body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse_body,
                request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
            )

        client = AsyncHTTPHandler()
        client.post = AsyncMock(side_effect=post)
        logging_obj = MagicMock()
        logging_obj.dynamic_success_callbacks = []
        logging_obj.model_call_details = {}
        logging_obj.completion_start_time = None
        handler = BaseLLMHTTPHandler()
        handler._call_agentic_completion_hooks = AsyncMock(return_value=None)
        optional_params = {"stream": stream}
        if caller_instructions is not None:
            optional_params["instructions"] = caller_instructions

        result = await handler.async_response_api_handler(
            model="gpt-5.5",
            input="Reply with ok.",
            responses_api_provider_config=config,
            response_api_optional_request_params=optional_params,
            custom_llm_provider="chatgpt",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=logging_obj,
            client=client,
        )
        responses = (
            [event.response async for event in result if getattr(event, "response", None) is not None]
            if stream
            else [result]
        )

        assert json.loads(client.post.call_args.kwargs["data"])["instructions"].startswith(
            "You are Codex, based on GPT-5."
        )
        assert len(responses) == (2 if stream else 1)
        assert [response.instructions for response in responses] == [caller_instructions] * len(responses)

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.5",
            "chatgpt/gpt-5.6-luna",
            "chatgpt/gpt-5.6-sol",
            "chatgpt/gpt-5.6-terra",
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @pytest.mark.parametrize(
        ("model_name", "max_input_tokens"),
        [
            ("chatgpt/gpt-5.5", 1050000),
            ("chatgpt/gpt-5.6-luna", 372000),
            ("chatgpt/gpt-5.6-sol", 372000),
            ("chatgpt/gpt-5.6-terra", 372000),
        ],
    )
    def test_chatgpt_responses_model_metadata(
        self, model_name: str, max_input_tokens: int, local_model_cost_map: None
    ) -> None:
        model_info = litellm.get_model_info(model_name)

        assert model_info["litellm_provider"] == "chatgpt"
        assert model_info["mode"] == "responses"
        assert model_info["supported_endpoints"] == [
            "/v1/chat/completions",
            "/v1/responses",
        ]
        assert model_info["max_input_tokens"] == max_input_tokens
        assert model_info["max_output_tokens"] == 128000

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
    )
    def test_chatgpt_models_bridge_chat_completions_to_responses(
        self, model_name: str, local_model_cost_map: None
    ) -> None:
        """A chat completions request for these models must take the Responses bridge.

        `gpt-5.6-*` also exists as an openai chat model, so an unregistered
        chatgpt model resolves to mode "chat" here and never reaches the bridge.
        """
        model_info, resolved_model = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider="chatgpt",
        )

        assert model_info["mode"] == "responses"
        assert resolved_model == model_name

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_headers(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="session-123")
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] != "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_prompt_cache_key_sets_session_id_header(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        prompt_cache_key = "conversation-123"

        for call_id in ("call-1", "call-2"):
            litellm_params = GenericLiteLLMParams(litellm_call_id=call_id)
            headers = config.validate_environment(
                headers={"session_id": f"explicit-{call_id}"},
                model="gpt-5.4",
                litellm_params=litellm_params,
            )
            request = config.transform_responses_api_request(
                model="gpt-5.4",
                input="hi",
                response_api_optional_request_params={
                    "prompt_cache_key": prompt_cache_key
                },
                litellm_params=litellm_params,
                headers=headers,
            )

            assert request["prompt_cache_key"] == prompt_cache_key
            assert headers["session_id"] == prompt_cache_key

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_without_prompt_cache_key_preserves_session_id_header(
        self, mock_authenticator_class
    ):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="fallback-session")
        headers = config.validate_environment(
            headers={"session_id": "explicit-session"},
            model="gpt-5.4",
            litellm_params=litellm_params,
        )
        request = config.transform_responses_api_request(
            model="gpt-5.4",
            input="hi",
            response_api_optional_request_params={},
            litellm_params=litellm_params,
            headers=headers,
        )

        assert "prompt_cache_key" not in request
        assert headers["session_id"] == "explicit-session"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_stateless_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert request["store"] is False
        assert request["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ]
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 200000}
                ],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
                "prompt_cache_key": "agent:operator:thread-1",
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                # unsupported by ChatGPT Codex
                "prompt_cache_retention": "24h",
                "store": False,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }
        assert request["prompt_cache_key"] == "agent:operator:thread-1"
        assert "prompt_cache_options" not in request
        assert "extra_body" not in request
        assert "prompt_cache_retention" not in request
        assert request["store"] is False

    @pytest.mark.parametrize(
        ("requested", "sent"),
        [
            ("priority", "priority"),
            ("fast", "priority"),
            ("FAST", "priority"),
            ("default", "default"),
            ("auto", None),
            ("flex", None),
            ("ultrafast", None),
            (None, None),
        ],
    )
    def test_chatgpt_forwards_fast_mode_service_tier(self, requested, sent):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-6.1-sol",
            input="hi",
            response_api_optional_request_params=({} if requested is None else {"service_tier": requested}),
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request.get("service_tier") == sent

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "instructions": f"{get_chatgpt_default_instructions()}\n\nBe brief.",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"
        assert parsed.instructions == "Be brief."

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_recovers_output_items(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from stream!"}],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello from stream!"

    def test_chatgpt_non_stream_sse_recovers_whitespace_padded_chunks(self):
        """Chunks with leading whitespace before `data:` must still parse.

        `_strip_sse_data_from_chunk` only matches the prefix at position 0,
        so without an outer `.strip()` such chunks would fail JSON parsing
        and silently drop the contained event.
        """
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Recovered from padded"}],
        }
        sse_body = "\n".join(
            [
                f"   data:  {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}   ",
                f"\tdata: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Recovered from padded"

    @pytest.mark.parametrize(
        "error_chunk",
        [
            {
                "type": "response.failed",
                "response": {"error": {"message": "ChatGPT upstream failed"}},
            },
            {
                "type": "error",
                "error": {"message": "ChatGPT upstream failed"},
            },
        ],
    )
    def test_chatgpt_non_stream_sse_response_raises_openai_error(self, error_chunk):
        config = ChatGPTResponsesAPIConfig()
        sse_body = "\n".join(
            [
                f"data: {json.dumps(error_chunk)}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            502, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert "ChatGPT upstream failed" in str(exc_info.value)
        assert exc_info.value.status_code == 502
