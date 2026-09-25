"""Server-owned entitlement for subscription model discovery."""

import os
import re
from collections.abc import Mapping
from typing import Final

PROTECTED_NAMES: Final = frozenset(
    (
        "anthropic/claude-opus-5-5/pi",
        "pi/claude-opus-5-5-chat",
        "claude-opus-5-5-pi-native",
        "claude-opus-5-5-pi-chat",
    )
)
_HASH: Final = re.compile(r"[0-9a-f]{64}\Z")


def subscription_model_visible(token: str | None, name: object) -> bool:
    if not isinstance(name, str) or name not in PROTECTED_NAMES:
        return True
    configured = os.environ.get("OPENORANGE_CLAUDE_SUBSCRIPTION_KEY_HASHES", "")
    return bool(
        isinstance(token, str)
        and _HASH.fullmatch(token)
        and any(token == candidate.strip() for candidate in configured.split(",") if _HASH.fullmatch(candidate.strip()))
    )


def subscription_deployment_visible(token: str | None, row: Mapping[str, object]) -> bool:
    info = row.get("model_info")
    names = (
        (row.get("model_name"), info.get("id"), info.get("team_public_model_name"))
        if isinstance(info, Mapping) else (row.get("model_name"),)
    )
    return all(subscription_model_visible(token, name) for name in names)
