# Deployment pricing contracts

Custom pricing belongs to the deployment ID, including IDs that match a
provider's bare model name. The provider is inferred from the original route
before selecting the ID. Public `completion_cost` callers may still omit
`custom_llm_provider`. Character rates, including explicit zero, participate in
custom deployment selection. Shared backend pricing is not overwritten by an
individual deployment's overrides.

## Time and context

`request_time` accepts an aware datetime or Unix timestamp. Naive datetimes
retain the fork's local-time interpretation. Logging supplies request start
time; delayed logging must not select rates using completion or flush time.
Pricing resolution never mutates the shared model registry.

`pricing_tier_threshold_inclusive: true` makes configured prompt thresholds
inclusive; omission retains strict `>` behavior. Thresholds apply to the
whole request's input, output and cache rates. Image tokens without a separate
`input_cost_per_image_token` rate use the selected input tier; an explicit
image-token rate, including zero, still wins.

`pricing_periods` use timezone-qualified, start-inclusive/end-exclusive
`effective_from` and `effective_until`. Either bound may be omitted. Existing
overlap and invalid-boundary errors remain explicit. A period can override
`input_cost_per_token`, `output_cost_per_token`, `cache_read_input_token_cost`,
`cache_creation_input_token_cost`, `cache_creation_input_token_cost_above_1hr`,
and `output_cost_per_reasoning_token`. The first five also accept generic
`_above_<integer>k_tokens` or `_above_<integer>_tokens` suffixes, including
`cache_creation_input_token_cost_above_1hr_above_200k_tokens`. Only these
allowlisted numeric rate fields are applied; rates must be finite and
nonnegative. Missing fields inherit; zero is an override.

## Recurring windows

Set `off_peak_pricing` in `model_info` or `litellm_params`. Base rates describe
peak usage; this block supplies discounted rates during matching windows:

```yaml
off_peak_pricing:
  weekday_timezone: UTC
  windows:
    - weekdays: [1, 2, 3, 4, 5]
      hours_utc: ["00:00-01:00", "04:00-06:00", "10:00-00:00"]
    - weekdays: [6, 7]
      hours_utc: "00:00-00:00"
  input_cost_per_token: 0.00000022
  output_cost_per_token: 0.00000066
  cache_read_input_token_cost: 0.000000007
```

Hours are UTC, start-inclusive/end-exclusive, and may wrap midnight. Equal
endpoints mean the whole day. Weekdays are ISO 1-7 or case-insensitive weekday
names/abbreviations, evaluated at the current instant in `weekday_timezone`
(UTC by default). Windows are ORed. Top-level `hours_utc` supplies an everyday
window. Omitted weekdays mean every day; an empty list means no days.

Recurring overrides support the six unsuffixed rate fields listed above,
including the separate one-hour write rate. Missing rates inherit the selected
context tier; explicit zero is honored. Invalid schedules/rates are ignored,
not allowed to contaminate billed cost. No nested finite periods or schedules
inside rate overrides are supported.

An active finite period's explicitly supplied fields take precedence over
recurring overrides. Existing finite-period context semantics are preserved:
changing a period's base rate does not remove a separately configured long
context tier. Supply that period's threshold rates to change the upper tier.
This matters for historical promotions that have both base and long-context
rates.

The only schedule field supported inside a finite period is the explicit
disable marker `off_peak_pricing: null`; omission inherits the deployment's
recurring schedule. Nested replacement schedule objects are not supported.
For example, historical rates can disable a schedule introduced later:

```yaml
pricing_periods:
  - effective_until: "2026-08-16T16:00:00Z"
    off_peak_pricing: null
    input_cost_per_token: 0.00000014
    output_cost_per_token: 0.00000028
    cache_read_input_token_cost: 0.0000000028
```

## Audio and cache breakdowns

`minimum_billable_duration_seconds` floors positive measured transcription
duration. Missing or zero duration does not produce phantom usage. Groq's
bundled Whisper entries specify 10 seconds; custom deployments should carry
the same field with their explicit per-second rate. A transcription entry
with zero output rate uses its input rate. Other modes preserve explicitly
free output and do not fall through to paid input.

Cache breakdowns use the same resolved date/context/recurring rates and mixed
five-minute/one-hour counts as total costs. Anthropic speed/geo multipliers
continue to exclude cache charges. This change does not alter spend queue
durability, sidecar routing, or Responses input limits.

## Upstream scope

Recurring window semantics are selectively adapted from BerriAI/litellm
#31725 (merge `c913b09e66`, dependency series `9fc77f1222`, `f2c663515c`,
`4f174ffdd1`, `27aefade5f`, `cc3ea1fb08`, `7abed91523`, `1ba13fcc25`,
`4875872fe5`) and #39635 (`e297968826`, reasoning/cache-write overrides).
Reference source was inspected at `eeb7732fc11fd47762ca84cc3fb7cc74235d7097`.
This is a focused standard-library adaptation, not an upstream upgrade or a
wholesale cherry-pick. Fork-specific differences include explicit request
time, separate one-hour cache writes, and existing finite-period semantics.
