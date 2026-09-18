"""Private upstream usage presence, before adapter defaults or token estimates."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from typing import TYPE_CHECKING, Annotated, Literal

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from litellm.litellm_core_utils.terminal_receipt_evidence import Closed, UUIDText

if TYPE_CHECKING:
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

FIELD = "openorange_usage_observation"
Quantity = Annotated[int, Field(strict=True, ge=0, le=9007199254740991)]
_VALUES = TypeAdapter(dict[str, object])
_OPAQUE = TypeAdapter(object)
_QUANTITY: TypeAdapter[int] = TypeAdapter(Quantity)
_CHOICES = TypeAdapter(list[object])


class UsageSnapshot(Closed):
    prompt_tokens: Quantity | None = None
    completion_tokens: Quantity | None = None
    total_tokens: Quantity | None = None
    cache_read_tokens: Quantity | None = None
    cache_write_tokens: Quantity | None = None
    cache_write_5m_tokens: Quantity | None = None
    cache_write_1h_tokens: Quantity | None = None


class UsageObservation(UsageSnapshot):
    v: Annotated[int, Field(strict=True, ge=1, le=1)]
    local_attempt_id: UUIDText
    state: Literal["observed", "partial", "unobserved"]

    @model_validator(mode="after")
    def empty_unobserved(self) -> "UsageObservation":
        if self.state == "unobserved" and any(getattr(self, name) is not None for name in UsageSnapshot.model_fields):
            raise ValueError("invalid_usage_observation")
        return self


def _value(usage: dict[str, object], paths: tuple[tuple[str, ...], ...]) -> int | None:
    present: tuple[int, ...] = ()
    for path in paths:
        value: object = usage
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = _VALUES.validate_python(value).get(key)
        if value is None:
            continue
        number = _QUANTITY.validate_python(value)
        present = (*present, number)
    if not present:
        return None
    if len(set(present)) != 1:
        raise ValueError("inconsistent_usage_observation")
    return present[0]


def snapshot(value: object) -> UsageSnapshot:
    usage = _VALUES.validate_python(value)
    return UsageSnapshot(
        prompt_tokens=_value(usage, (("prompt_tokens",),)),
        completion_tokens=_value(usage, (("completion_tokens",),)),
        total_tokens=_value(usage, (("total_tokens",),)),
        cache_read_tokens=_value(usage, (("cache_read_input_tokens",), ("prompt_tokens_details", "cached_tokens"))),
        cache_write_tokens=_value(
            usage,
            (
                ("cache_creation_input_tokens",),
                ("prompt_tokens_details", "cache_write_tokens"),
                ("prompt_tokens_details", "cache_creation_tokens"),
            ),
        ),
        cache_write_5m_tokens=_value(
            usage,
            (
                ("cache_creation", "ephemeral_5m_input_tokens"),
                ("prompt_tokens_details", "cache_creation_token_details", "ephemeral_5m_input_tokens"),
            ),
        ),
        cache_write_1h_tokens=_value(
            usage,
            (
                ("cache_creation", "ephemeral_1h_input_tokens"),
                ("prompt_tokens_details", "cache_creation_token_details", "ephemeral_1h_input_tokens"),
            ),
        ),
    )


def observe_sdk_usage(details: dict[str, object], response: object) -> None:
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    session = details.get(STAMP)
    if type(session) is not Session or session.root.authority.settings.role != "audience":
        return
    if not isinstance(response, (ChatCompletion, ChatCompletionChunk)):
        return
    if "usage" not in response.model_fields_set or response.usage is None:
        return
    with session.lock:
        try:
            session.local_usage_snapshot = snapshot(response.usage.model_dump(exclude_unset=True))
        except (ValidationError, ValueError):
            session.usage_invalid = True


def usage_for_spend(value: object) -> dict[str, JsonValue]:
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(value) is not Session or value.root.authority.settings.role != "audience":
        return {}
    observed = value.usage_snapshot
    state = (
        "unobserved" if observed is None else "observed" if value.usage_final and not value.usage_invalid else "partial"
    )
    fact = UsageObservation.model_validate(
        {
            "v": 1,
            "local_attempt_id": value.attempt_id,
            "state": state,
            **(observed or UsageSnapshot()).model_dump(),
        }
    )
    return {FIELD: fact.model_dump(mode="json")}


def safe_usage(value: object) -> dict[str, JsonValue]:
    try:
        data = _VALUES.validate_python(value)
        if (
            set(data) != set(UsageObservation.model_fields)
            or len(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()) > 2048
        ):
            raise ValueError("invalid_usage_observation")
        return UsageObservation.model_validate(data).model_dump(mode="json")
    except (ValidationError, ValueError, TypeError):
        return {"v": 1, "state": "invalid"}


def _anthropic_native(
    session: "Session", body: dict[str, object], streamed: bool
) -> tuple[dict[str, object], bool] | None:
    if streamed and body.get("type") == "message_start":
        source = _VALUES.validate_python(body.get("message") or {}).get("usage")
    elif not streamed or body.get("type") == "message_delta":
        source = body.get("usage")
    else:
        return None
    if source is None:
        return None
    current = _VALUES.validate_python(source)
    usage = {**session.native_usage_fields, **current}
    session.native_usage_fields = usage
    read = _value(usage, (("cache_read_input_tokens",),))
    write = _value(usage, (("cache_creation_input_tokens",),))
    uncached = _value(usage, (("input_tokens",),))
    inclusive = uncached + read + write if uncached is not None and read is not None and write is not None else None
    normalized = {**usage, "prompt_tokens": inclusive, "completion_tokens": usage.get("output_tokens")}
    final = not streamed or (body.get("type") == "message_delta" and _value(current, (("output_tokens",),)) is not None)
    return normalized, final


def _chatgpt_native(body: dict[str, object], streamed: bool) -> tuple[dict[str, object], bool] | None:
    if streamed:
        if body.get("type") not in {"response.completed", "response.incomplete"}:
            return None
        usage = _VALUES.validate_python(body.get("response") or {}).get("usage")
    else:
        usage = body.get("usage")
    if usage is None:
        return None
    fields = _VALUES.validate_python(usage)
    normalized = {
        "prompt_tokens": fields.get("input_tokens"),
        "completion_tokens": fields.get("output_tokens"),
        "total_tokens": fields.get("total_tokens"),
        "cache_read_input_tokens": _value(fields, (("input_tokens_details", "cached_tokens"),)),
        "cache_creation_input_tokens": _value(fields, (("input_tokens_details", "cache_write_tokens"),)),
    }
    return normalized, not streamed or body.get("type") == "response.completed"


def _deepseek_native(body: dict[str, object], streamed: bool) -> tuple[dict[str, object], bool] | None:
    usage = body.get("usage")
    if usage is None:
        return None
    fields = _VALUES.validate_python(usage)
    read = _value(
        fields, (("prompt_cache_hit_tokens",), ("cache_read_input_tokens",), ("prompt_tokens_details", "cached_tokens"))
    )
    missed = _value(fields, (("prompt_cache_miss_tokens",),))
    prompt = _value(fields, (("prompt_tokens",),))
    if read is not None and missed is not None and prompt is not None and read + missed != prompt:
        raise ValueError("inconsistent_native_cache_measurement")
    normalized = {
        "prompt_tokens": fields.get("prompt_tokens"),
        "completion_tokens": fields.get("completion_tokens"),
        "total_tokens": fields.get("total_tokens"),
        "cache_read_input_tokens": read,
    }
    choices = body.get("choices")
    final = not streamed or (
        isinstance(choices, list)
        and all(
            isinstance(choice, dict)
            and _VALUES.validate_python(choice).get("finish_reason")
            in {"stop", "length", "tool_calls", "content_filter", "function_call"}
            for choice in _CHOICES.validate_python(choices)
        )
    )
    return normalized, final


def observe_native_usage(details: dict[str, object], raw: object, *, streamed: bool = False) -> None:
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    session = details.get(STAMP)
    if type(session) is not Session or session.root.authority.settings.role != "producer" or session.relay:
        return
    with session.lock:
        try:
            body = _VALUES.validate_python(raw)
            match session.terminal.provider:
                case "anthropic":
                    captured = _anthropic_native(session, body, streamed)
                case "chatgpt":
                    captured = _chatgpt_native(body, streamed)
                case "deepseek":
                    captured = _deepseek_native(body, streamed)
                case _:
                    return
            if captured is None:
                return
            normalized, final = captured
            session.usage_snapshot = snapshot(normalized)
            session.native_usage_final = final
        except (ValidationError, ValueError, TypeError):
            session.usage_invalid = True


def observe_native_line(details: dict[str, object], line: str) -> None:
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    session = details.get(STAMP)
    if type(session) is not Session or session.root.authority.settings.role != "producer" or session.relay:
        return
    if not line.startswith("data:"):
        return
    if len(line.encode()) > 65536:
        session.usage_invalid = True
        return
    raw = line[5:].lstrip()
    if raw == "[DONE]":
        return
    try:
        body = _OPAQUE.validate_json(raw)
    except ValueError:
        session.usage_invalid = True
        return
    observe_native_usage(details, body, streamed=True)


def observe_lines(lines: Iterator[str], details: dict[str, object]) -> Iterator[str]:
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(details.get(STAMP)) is not Session:
        return lines

    def observed() -> Iterator[str]:
        try:
            for line in lines:
                observe_native_line(details, line)
                yield line
        finally:
            if isinstance(lines, Generator):
                lines.close()

    return observed()


def observe_lines_async(lines: AsyncIterator[str], details: dict[str, object]) -> AsyncIterator[str]:
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(details.get(STAMP)) is not Session:
        return lines

    async def observed() -> AsyncIterator[str]:
        try:
            async for line in lines:
                observe_native_line(details, line)
                yield line
        finally:
            if isinstance(lines, AsyncGenerator):
                await lines.aclose()

    return observed()
