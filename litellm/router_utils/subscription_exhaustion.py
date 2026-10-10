"""
A Claude subscription account out of quota: the central proxy answers 429 with
``x-openorange-subscription-exhausted-until`` (unix seconds) only when every account it walked is out of quota.
That answer is the only one a workspace's paid API-key hop may follow, and it is never retried: any other 429,
an overload or a rate limit inside a stream is load.
"""

import re
from collections.abc import Mapping
from typing import Final

SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: Final = "x-openorange-subscription-exhausted-until"
_UNIX_SECONDS: Final = re.compile(r"[0-9]{1,12}")


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    if headers is None:
        return None
    return next((value for key, value in headers.items() if key.lower() == name), None)


def _rate_limit_header(error: BaseException, name: str) -> str | None:
    if getattr(error, "status_code", None) != 429:
        return None
    sources: Final[tuple[Mapping[str, str] | None, ...]] = (
        getattr(error, "headers", None),
        getattr(error, "litellm_response_headers", None),
        getattr(getattr(error, "response", None), "headers", None),
    )
    values: Final = (_header(source, name) for source in sources)
    return next((value.strip() for value in values if value is not None), None)


def subscription_exhausted_until(error: BaseException) -> int | None:
    value: Final = _rate_limit_header(error, SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER) or ""
    return int(value) if _UNIX_SECONDS.fullmatch(value) else None
