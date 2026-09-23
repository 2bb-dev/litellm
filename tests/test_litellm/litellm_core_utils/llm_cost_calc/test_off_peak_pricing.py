from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import _open_off_peak_block, _parse_off_peak_rate


def get_off_peak_pricing_overrides(model_info: "ModelInfo", request_time: datetime) -> dict[str, float]:
    schedule = _open_off_peak_block(model_info, request_time)
    if schedule is None:
        return {}
    return {key: rate for key, value in schedule.items() if (rate := _parse_off_peak_rate(value)) is not None}


from litellm.types.utils import ModelInfo


def resolve_off_peak_pricing(entry: ModelInfo, instant: datetime | float) -> ModelInfo:
    timestamp = datetime.fromtimestamp(instant, timezone.utc) if isinstance(instant, float) else instant
    overrides = get_off_peak_pricing_overrides(entry, timestamp)
    return cast(ModelInfo, {**entry, **overrides}) if overrides else entry


def model(schedule: object) -> ModelInfo:
    return cast(
        ModelInfo,
        {
            "input_cost_per_token": 0.44,
            "output_cost_per_token": 1.32,
            "cache_read_input_token_cost": 0.014,
            "cache_creation_input_token_cost": 0.55,
            "cache_creation_input_token_cost_above_1hr": 0.88,
            "output_cost_per_reasoning_token": 1.5,
            "off_peak_pricing": schedule,
        },
    )


def test_only_supplied_rates_returned_without_changing_finite_or_tier_metadata() -> None:
    entry = cast(
        ModelInfo,
        {
            **model(
                {
                    "hours_utc": "00:00-00:00",
                    "input_cost_per_token": 0,
                    "cache_creation_input_token_cost_above_1hr": 0.4,
                }
            ),
            "pricing_periods": [{"effective_until": "2026-09-09T16:00:00Z", "input_cost_per_token": 0.075}],
            "input_cost_per_token_above_200k_tokens": 4,
        },
    )
    original = deepcopy(entry)
    assert get_off_peak_pricing_overrides(entry, datetime(2026, 9, 7, tzinfo=timezone.utc)) == {
        "input_cost_per_token": 0,
        "cache_creation_input_token_cost_above_1hr": 0.4,
    }
    assert entry == original


def test_invalid_weekday_timezone_falls_back_to_utc() -> None:
    entry = model(
        {
            "windows": [{"hours_utc": "00:00-00:00", "weekdays": ["Monday"]}],
            "weekday_timezone": "not-a-timezone",
            "input_cost_per_token": 0,
        }
    )
    assert get_off_peak_pricing_overrides(entry, datetime(2026, 9, 7, 23, tzinfo=timezone.utc)) == {
        "input_cost_per_token": 0
    }


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        ("2026-09-07T00:59:59.999999+00:00", 0.22),
        ("2026-09-07T01:00:00+00:00", 0.44),
        ("2026-09-07T03:59:59.999999+00:00", 0.44),
        ("2026-09-07T04:00:00+00:00", 0.22),
        ("2026-09-07T05:59:59.999999+00:00", 0.22),
        ("2026-09-07T06:00:00+00:00", 0.44),
        ("2026-09-07T09:59:59.999999+00:00", 0.44),
        ("2026-09-07T10:00:00+00:00", 0.22),
        ("2026-09-11T01:00:00+00:00", 0.44),
        ("2026-09-12T01:00:00+00:00", 0.22),
        ("2026-09-13T06:00:00+00:00", 0.22),
        ("2026-09-13T23:59:59.999999+00:00", 0.22),
        ("2026-09-14T00:00:00+00:00", 0.22),
        ("2026-09-14T01:00:00+00:00", 0.44),
    ],
)
def test_deepseek_weekly_boundaries(timestamp: str, expected: float) -> None:
    entry = model(
        {
            "windows": [
                {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
                {"hours_utc": "00:00-00:00", "weekdays": ["Saturday", "sun"]},
            ],
            "input_cost_per_token": 0.22,
            "output_cost_per_token": 0.66,
            "cache_read_input_token_cost": 0.007,
        }
    )
    result = resolve_off_peak_pricing(entry, datetime.fromisoformat(timestamp))
    assert result["input_cost_per_token"] == expected
    assert result["output_cost_per_token"] == pytest.approx(expected * 3)
    assert result["cache_read_input_token_cost"] == (0.007 if expected == 0.22 else 0.014)


@pytest.mark.parametrize("hour,active", [(22, False), (23, True), (0, True), (1, True), (2, False)])
def test_wrapping_midnight(hour: int, active: bool) -> None:
    result = resolve_off_peak_pricing(
        model({"hours_utc": "23:00-02:00", "input_cost_per_token": 0}),
        datetime(2026, 9, 7, hour, tzinfo=timezone.utc),
    )
    assert result["input_cost_per_token"] == (0 if active else 0.44)


def test_request_instant_timezone_offset() -> None:
    entry = model({"hours_utc": "04:00-06:00", "input_cost_per_token": 0.22})
    instant = datetime(2026, 9, 7, 4, tzinfo=timezone.utc)
    representations = (instant, instant.astimezone(timezone(timedelta(hours=8))))
    assert all(resolve_off_peak_pricing(entry, value)["input_cost_per_token"] == 0.22 for value in representations)


def test_weekday_timezone_uses_current_calendar_not_window_start_day() -> None:
    entry = model(
        {
            "windows": [{"hours_utc": "16:00-02:00", "weekdays": ["sat"]}],
            "weekday_timezone": "Asia/Shanghai",
            "input_cost_per_token": 0,
        }
    )
    assert resolve_off_peak_pricing(entry, datetime(2026, 9, 11, 16, tzinfo=timezone.utc))["input_cost_per_token"] == 0
    assert (
        resolve_off_peak_pricing(entry, datetime(2026, 9, 12, 16, tzinfo=timezone.utc))["input_cost_per_token"] == 0.44
    )


def test_rates_are_independent_and_source_is_unchanged() -> None:
    entry = model(
        {
            "hours_utc": "00:00-00:00",
            "output_cost_per_token": 0.66,
            "output_cost_per_reasoning_token": 0.75,
            "cache_creation_input_token_cost": 0,
            "cache_creation_input_token_cost_above_1hr": 0.4,
        }
    )
    original = deepcopy(entry)
    result = resolve_off_peak_pricing(entry, 1788758400.0)
    assert result is not entry
    assert entry == original
    assert result["input_cost_per_token"] == 0.44
    assert result["output_cost_per_token"] == 0.66
    assert result["output_cost_per_reasoning_token"] == 0.75
    assert result["cache_read_input_token_cost"] == 0.014
    assert result["cache_creation_input_token_cost"] == 0
    assert result["cache_creation_input_token_cost_above_1hr"] == 0.4


def test_missing_write_and_reasoning_rates_inherit_without_inference() -> None:
    entry = model({"hours_utc": "00:00-00:00", "output_cost_per_token": 0.66, "cache_creation_input_token_cost": 0.1})
    result = resolve_off_peak_pricing(entry, 1788758400.0)
    assert result["output_cost_per_reasoning_token"] == 1.5
    assert result["cache_creation_input_token_cost_above_1hr"] == 0.88


@pytest.mark.parametrize("invalid", [None, True, -1, "bad", "nan", float("inf"), float("-inf"), {}, []])
def test_invalid_rates_inherit(invalid: object) -> None:
    result = resolve_off_peak_pricing(
        model({"hours_utc": "00:00-00:00", "input_cost_per_token": invalid}), 1788758400.0
    )
    assert result["input_cost_per_token"] == 0.44


@pytest.mark.parametrize(
    "schedule",
    [
        None,
        [],
        "bad",
        {"hours_utc": "24:00-25:00"},
        {"hours_utc": [None, "broken"]},
        {"windows": "bad"},
        {"windows": [None, {"hours_utc": "00:00-00:00", "weekdays": []}]},
        {"windows": [{"hours_utc": "00:00-00:00", "weekdays": [True, 0, 8, "bad"]}]},
    ],
)
def test_malformed_schedule_does_not_activate(schedule: object) -> None:
    entry = model(schedule)
    assert resolve_off_peak_pricing(entry, 1788758400.0) is entry


def test_flat_windows_and_weekday_rules_are_additive() -> None:
    entry = model(
        {"hours_utc": "04:00-06:00", "windows": [{"hours_utc": "01:00-02:00"}], "input_cost_per_token": "0.22"}
    )
    assert resolve_off_peak_pricing(entry, datetime(2026, 9, 7, 1, tzinfo=timezone.utc))["input_cost_per_token"] == 0.22
    assert resolve_off_peak_pricing(entry, datetime(2026, 9, 7, 4, tzinfo=timezone.utc))["input_cost_per_token"] == 0.22
