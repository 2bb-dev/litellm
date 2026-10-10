"""
Claude subscription accounts move a call on only when one is out of quota.

The Pi slot answers an account Anthropic marked out of quota with a 429 carrying
``x-openorange-subscription-exhausted-until`` (unix seconds), and its own concurrency refusal
with ``x-openorange-pi-slot-at-capacity``. Any other 429, and an overload, is load: it is retried
on its account, never starts a walk to the next one and is never paid for. A walk over the accounts
that met one out of quota answers the same in any order: any load wins, otherwise the first
out-of-quota answer, which a proxy repeats to its caller.
"""

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.exceptions import MidStreamFallbackError
from litellm.types.router import DeploymentTypedDict

SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: Final = "x-openorange-subscription-exhausted-until"
PI_SLOT_AT_CAPACITY_HEADER: Final = "x-openorange-pi-slot-at-capacity"
SUBSCRIPTION_ACCOUNT_FLAG: Final = "rate_limit_fallback_requires_exhaustion"
_UNIX_SECONDS: Final = re.compile(r"[0-9]{1,12}")
_LOAD_MARKERS: Final = ("overloaded_error", "Overloaded", "rate_limit_error")
_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
_DEPLOYMENTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_MODEL_INFO: Final = TypeAdapter(Mapping[str, object])


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


def pi_slot_at_capacity(error: BaseException) -> bool:
    return _rate_limit_header(error, PI_SLOT_AT_CAPACITY_HEADER) is not None


def is_subscription_load(error: BaseException) -> bool:
    source: Final = error.original_exception if isinstance(error, MidStreamFallbackError) else error
    if source is None or subscription_exhausted_until(source) is not None:
        return False
    if getattr(source, "status_code", None) in (429, 529):
        return True
    return any(marker in str(source) for marker in _LOAD_MARKERS)


def is_subscription_account_group(deployments: Sequence[DeploymentTypedDict] | None) -> bool:
    rows: Final = _DEPLOYMENTS.validate_python(deployments or ())
    return any(
        _MODEL_INFO.validate_python(row.get("model_info") or _NO_HEADERS).get(SUBSCRIPTION_ACCOUNT_FLAG) is True
        for row in rows
    )


def subscription_walk_verdict(errors: Sequence[BaseException]) -> BaseException | None:
    if all(subscription_exhausted_until(error) is None for error in errors):
        return None
    load: Final = next((error for error in errors if is_subscription_load(error)), None)
    if load is not None:
        return load
    return next(error for error in errors if subscription_exhausted_until(error) is not None)


def subscription_exhaustion_headers(error: BaseException) -> Mapping[str, str]:
    until: Final = subscription_exhausted_until(error)
    if until is None:
        return _NO_HEADERS
    return MappingProxyType({SUBSCRIPTION_EXHAUSTED_UNTIL_HEADER: str(until)})
