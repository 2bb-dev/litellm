# OpenOrange LiteLLM Fork

`main` mirrors upstream LiteLLM. `openorange` is the production integration
branch consumed by OpenOrange through an exact submodule commit.

Layer and Platform pin only commits reachable from `openorange`. Never pin an
open pull-request head: stacked topic branches that ship through a pin drift
away from `openorange`, and the next pin taken from the integration branch then
silently drops their behavior. Land the stack first, then pin the merge commit.

## Fork Invariants

Upstream syncs must preserve these behaviors:

- **OSS-only packaging:** proxy builds must not install, copy, or require
  `litellm-enterprise` or the `enterprise` workspace.
- **ChatGPT subscription routing:** Responses state remains persistent where
  required, upstream storage stays disabled, prompt-cache parameters survive
  transformation, the ChatGPT session header follows `prompt_cache_key`,
  string inputs are normalized for the subscription backend, client `system`
  messages are represented as `developer` messages without changing their
  content or order, and
  provider-forced SSE is accumulated into one complete response for
  non-streaming callers without duplicate streaming hooks or spend logs.
- **OpenClaw attribution:** trusted runtime context and supported OpenClaw
  payload markers continue to populate actor, parent, session, channel,
  execution, and Langfuse metadata without persisting raw credentials.
- **Responses logging:** streamed terminal responses retain reconstructed
  output, annotations, refusals, and ordering for request-detail views.
- **Spend-log resilience:** database writes use byte-bounded adaptive batches,
  one process-local writer at a time, deterministic lock ordering, prompt-safe
  error logging, and resilient cleanup. Deployments that set
  `SPEND_LOG_DURABLE_QUEUE_PATH` persist pending rows in a local SQLite WAL and
  acknowledge them only after the idempotent PostgreSQL insert succeeds.
  Permanently invalid rows are isolated and preserved in local dead-letter
  storage so one poison payload cannot block later telemetry.

## Upstream Sync Procedure

1. Branch from `openorange` as `sync/upstream-vX.Y.Z`.
2. Merge an official stable upstream tag with a merge commit. Do not squash or
   replay the upstream history.
3. Prefer upstream implementations when they satisfy the invariant and retain
   the focused OpenOrange regression test.
4. Take upstream LiteLLM dashboard changes unless OpenOrange actively depends
   on a forked UI behavior.
5. Regenerate `uv.lock` after removing Enterprise dependencies.
6. Run the focused suites below, build the OSS image, and test it on an
   OpenOrange canary before advancing the parent repository's submodule SHA.

## Effective Token Pricing

For OpenAI-compatible and Z.ai chat/Responses accounting, `model_info` (or
`litellm_params`) can declare `pricing_periods`. Keep the normal rates at the
top level; each period overrides only its supplied input, output, cache-read,
or cache-creation per-token rates. `effective_from` is inclusive and
`effective_until` is exclusive; both accept timezone-qualified ISO 8601
strings and can be omitted for an unbounded interval. Overlapping active
periods and timezone-less boundaries are rejected by the generic calculator.

```yaml
input_cost_per_token: 0.00000015
output_cost_per_token: 0.0000005
cache_read_input_token_cost: 0.00000003
pricing_periods:
  - effective_until: "2026-09-09T16:00:00Z"
    input_cost_per_token: 0.000000075
    output_cost_per_token: 0.00000025
    cache_read_input_token_cost: 0.000000015
```

Native accounting uses the logging object's request start, including when
completion or streaming logging happens after expiry. Direct `completion_cost`,
`cost_per_token`, and `generic_cost_per_token` calls accept a `request_time`
datetime or Unix timestamp for historical calculations; without a logging
object or explicit time they use the current time. Resolution never mutates
the model registry.
Other provider-specific calculators and non-token billing are not extended.

`pricing_tier_threshold_inclusive: true` opts a deployment into inclusive
generic token thresholds. For example, the existing `*_above_200k_tokens`
input, output and cache-read fields then apply at 200000 tokens, not 200001.
Without the flag, existing exclusive thresholds are unchanged.

## Focused Regression Suites

- `tests/litellm/test_effective_token_pricing.py`
- `tests/test_litellm/llms/chatgpt/chat/test_chatgpt_transformation.py`
- `tests/test_litellm/llms/chatgpt/responses/test_chatgpt_responses_transformation.py`
- `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`
- `tests/llm_responses_api_testing/test_base_responses_api_streaming_iterator.py`
- `tests/test_litellm/proxy/test_litellm_pre_call_utils.py`
- `tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py`
- `tests/test_litellm/litellm_core_utils/test_credential_ownership.py`
- `tests/test_litellm/litellm_core_utils/test_native_credential_ownership.py`
- `tests/test_litellm/integrations/test_langfuse.py`
- `tests/test_litellm/proxy/db/test_db_spend_update_writer.py`
- `tests/proxy_unit_tests/test_update_spend.py`
- `tests/test_litellm/proxy/test_spend_log_cleanup.py`
