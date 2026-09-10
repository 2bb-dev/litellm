import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek/deepseek-flash"])
@pytest.mark.parametrize("thinking", [{"type": "enabled"}, {"type": "disabled"}])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_flash_preserves_thinking_controls(model: str, thinking: dict[str, str], effort: str) -> None:
    params = {"thinking": thinking, "reasoning_effort": effort}
    assert DeepSeekChatConfig().map_openai_params(params, {}, model, False) == params


def test_flash_keeps_provider_defaults_and_legacy_mapping() -> None:
    config = DeepSeekChatConfig()
    assert config.map_openai_params({}, {}, "deepseek-flash", False) == {}
    assert config._thinking_mode_active("deepseek-flash", {})
    assert not config._thinking_mode_active("deepseek-flash", {"thinking": {"type": "disabled"}})
    assert config.map_openai_params({"reasoning_effort": "high"}, {}, "deepseek-reasoner", False) == {
        "thinking": {"type": "enabled"}
    }


@pytest.mark.parametrize("is_async", [False, True])
async def test_flash_preserves_images_and_reasoning_history(is_async: bool) -> None:
    config = DeepSeekChatConfig()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Help the user"}]},
        {"role": "user", "content": [{"type": "text", "text": "Describe"}, image]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Checking"}],
            "provider_specific_fields": {"reasoning_content": "I see red"},
        },
        {"role": "user", "content": "Continue"},
    ]
    kwargs = {
        "model": "deepseek-flash",
        "messages": messages,
        "optional_params": {},
        "litellm_params": {},
        "headers": {},
    }
    body = await config.async_transform_request(**kwargs) if is_async else config.transform_request(**kwargs)
    assert body["messages"][0]["content"] == "Help the user"
    assert body["messages"][1]["content"] == messages[1]["content"]
    assert body["messages"][2]["content"] == "Checking"
    assert body["messages"][2]["reasoning_content"] == "I see red"
    assert messages[0]["content"] == [{"type": "text", "text": "Help the user"}]
    assert "thinking" not in body


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"
