import json

from litellm.exceptions import RateLimitError

QUOTA_CODES = frozenset({"usage_limit_reached", "insufficient_quota"})


def is_chatgpt_rate_limit(error: object) -> bool:
    if not isinstance(error, RateLimitError):
        return False
    return error.llm_provider == "chatgpt" or (
        error.llm_provider == "litellm_proxy"
        and isinstance(error.model, str)
        and error.model.startswith(("chatgpt/", "litellm_proxy/chatgpt/"))
    )


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
