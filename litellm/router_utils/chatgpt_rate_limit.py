import json
import math
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Final

import openai

from litellm.exceptions import MidStreamFallbackError, RateLimitError, Timeout
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_response_headers,  # pyright: ignore[reportPrivateUsage]  # the router reads error headers through it too
)

QUOTA_CODES = frozenset({"usage_limit_reached", "insufficient_quota"})
# OpenAI's Retry-After as it reaches the router: direct, or renamed by the ChatGPT sidecar's proxy.
_RETRY_AFTER_HEADERS: Final = ("retry-after", "llm_provider-retry-after")


def _is_chatgpt_route(error: object) -> bool:
    provider: Final = getattr(error, "llm_provider", None)
    model: Final = getattr(error, "model", None)
    return provider == "chatgpt" or (
        provider == "litellm_proxy"
        and isinstance(model, str)
        and model.startswith(("chatgpt/", "litellm_proxy/chatgpt/"))
    )


def is_chatgpt_rate_limit(error: object) -> bool:
    return isinstance(error, RateLimitError) and _is_chatgpt_route(error)


def _has_quota_code(value: object, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(value, dict):
        return any(value.get(key) in QUOTA_CODES for key in ("code", "type") if isinstance(value.get(key), str)) or any(
            _has_quota_code(value.get(key), depth + 1) for key in ("error", "message")
        )
    if not isinstance(value, str) or len(value) > 65536:
        return False
    start = value.find("{")
    if start < 0:
        return False
    try:
        payload, _ = json.JSONDecoder().raw_decode(value[start:])
    except (ValueError, RecursionError):
        return False
    return _has_quota_code(payload, depth + 1)


def is_chatgpt_quota_error(error: object) -> bool:
    if not is_chatgpt_rate_limit(error) or not isinstance(error, RateLimitError):
        return False
    return error.code in QUOTA_CODES or _has_quota_code(error.body) or _has_quota_code(error.message)


def is_chatgpt_transient_error(error: object) -> bool:
    """
    An OpenAI hiccup on a ChatGPT account: a 5xx or an overload, a timeout, or a stream that
    failed with one. The account is fine, so the call waits and tries the same account again.
    Rate limits keep their own rules, and a sidecar that can't be reached is not the account.
    """
    source: Final = (
        error.original_exception
        if isinstance(error, MidStreamFallbackError) and error.original_exception is not None
        else error
    )
    if not _is_chatgpt_route(error) and not _is_chatgpt_route(source):
        return False
    if isinstance(source, Timeout):
        return True
    if isinstance(source, (RateLimitError, openai.APIConnectionError)):
        return False
    status_code: Final = getattr(source, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


def _positive_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if value > 0 else None


def _retry_after_seconds(value: str) -> float | None:
    """Seconds, or an HTTP date, like the router's own Retry-After handling."""
    if value.strip().isdecimal():
        return _positive_seconds(int(value))
    try:
        return _positive_seconds(parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _reset_from_fields(fields: Mapping[object, object]) -> float | None:
    in_seconds: Final = _positive_seconds(fields.get("resets_in_seconds"))
    if in_seconds is not None:
        return in_seconds
    resets_at: Final = _positive_seconds(fields.get("resets_at"))
    return _positive_seconds(resets_at - time.time()) if resets_at is not None else None


def _quota_reset_seconds(value: object, depth: int = 0) -> float | None:
    """The reset time OpenAI puts in a quota 429 body, read through the same wrapping as the code."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        own: Final = _reset_from_fields(value)
        if own is not None:
            return own
        nested: Final = (_quota_reset_seconds(value.get(key), depth + 1) for key in ("error", "message"))
        return next((seconds for seconds in nested if seconds is not None), None)
    if not isinstance(value, str) or len(value) > 65536:
        return None
    start: Final = value.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(value[start:])
    except (ValueError, RecursionError):
        return None
    return _quota_reset_seconds(payload, depth + 1)


def chatgpt_quota_reset_seconds(error: object) -> float | None:
    """
    How long OpenAI says a ChatGPT account is out of quota: its Retry-After, else the reset time
    in the 429 body (resets_in_seconds or resets_at). None when the 429 names neither, so the
    router's cooldown_time applies, and for every other error.
    """
    if not is_chatgpt_quota_error(error) or not isinstance(error, RateLimitError):
        return None
    headers: Final = _get_response_headers(original_exception=error)
    header_values: Final = (headers.get(name) for name in _RETRY_AFTER_HEADERS) if headers else ()
    retry_after: Final = next(
        (
            seconds
            for seconds in (_retry_after_seconds(value) for value in header_values if isinstance(value, str))
            if seconds is not None
        ),
        None,
    )
    if retry_after is not None:
        return retry_after
    from_body: Final = _quota_reset_seconds(error.body)
    return from_body if from_body is not None else _quota_reset_seconds(error.message)
