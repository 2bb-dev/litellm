from collections.abc import Mapping, Sequence
from datetime import datetime, timezone, tzinfo
from math import isfinite
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


_RATE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "output_cost_per_reasoning_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
)
_WEEKDAY_NAMES = (
    ("mon", "monday"),
    ("tue", "tues", "tuesday"),
    ("wed", "wednesday"),
    ("thu", "thur", "thurs", "thursday"),
    ("fri", "friday"),
    ("sat", "saturday"),
    ("sun", "sunday"),
)


def _weekday(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 7 else None
    if isinstance(value, str):
        return next(
            (day for day, names in enumerate(_WEEKDAY_NAMES, 1) if value.strip().lower() in names),
            None,
        )
    return None


def _calendar(value: object) -> tzinfo:
    if isinstance(value, str) and value.strip():
        try:
            return ZoneInfo(value.strip())
        except (ValueError, ZoneInfoNotFoundError):
            pass
    return timezone.utc


def _matches_hours(value: object, instant: datetime) -> bool:
    windows = (value,) if isinstance(value, str) else value
    if not isinstance(windows, Sequence):
        return False
    for window in windows:
        if not isinstance(window, str):
            continue
        try:
            start_text, end_text = window.split("-")
            start = datetime.strptime(start_text.strip(), "%H:%M").time()
            end = datetime.strptime(end_text.strip(), "%H:%M").time()
        except ValueError:
            continue
        now = instant.time()
        matches = start <= now < end if start < end else now >= start or now < end
        if matches:
            return True
    return False


def _matches_rule(rule: object, instant: datetime, calendar: tzinfo) -> bool:
    if not isinstance(rule, Mapping):
        return False
    weekdays = rule.get("weekdays")
    if weekdays is not None:
        if isinstance(weekdays, str) or not isinstance(weekdays, Sequence):
            return False
        if instant.astimezone(calendar).isoweekday() not in tuple(map(_weekday, weekdays)):
            return False
    return _matches_hours(rule.get("hours_utc"), instant)


def _is_off_peak(schedule: Mapping[str, object], instant: datetime) -> bool:
    if _matches_hours(schedule.get("hours_utc"), instant):
        return True
    windows = schedule.get("windows")
    if isinstance(windows, str) or not isinstance(windows, Sequence):
        return False
    calendar = _calendar(schedule.get("weekday_timezone"))
    return any(_matches_rule(rule, instant, calendar) for rule in windows)


def _rate(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) and parsed >= 0 else None


def get_off_peak_pricing_overrides(model_info: "ModelInfo", request_time: datetime) -> dict[str, float]:
    """Adapt upstream #31725/#39635 schedules without changing shared model metadata."""
    schedule = model_info.get("off_peak_pricing")
    if not isinstance(schedule, Mapping):
        return {}
    instant = request_time.astimezone(timezone.utc)
    if not _is_off_peak(schedule, instant):
        return {}
    return {field: rate for field in _RATE_FIELDS if (rate := _rate(schedule.get(field))) is not None}
