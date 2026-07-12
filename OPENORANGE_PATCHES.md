# OpenOrange LiteLLM Fork

`main` mirrors upstream LiteLLM. `openorange` is the production integration
branch consumed by OpenOrange through an exact submodule commit.

## Fork Invariants

Upstream syncs must preserve these behaviors:

- **OSS-only packaging:** proxy builds must not install, copy, or require
  `litellm-enterprise` or the `enterprise` workspace.
- **ChatGPT subscription routing:** Responses state remains persistent where
  required, upstream storage stays disabled, prompt-cache parameters survive
  transformation, the ChatGPT session header follows `prompt_cache_key`,
  string inputs are normalized for the subscription backend, and
  provider-forced SSE is accumulated into one complete response for
  non-streaming callers without duplicate streaming hooks or spend logs.
- **OpenClaw attribution:** trusted runtime context and supported OpenClaw
  payload markers continue to populate actor, parent, session, channel,
  execution, and Langfuse metadata without persisting raw credentials.
- **Responses logging:** streamed terminal responses retain reconstructed
  output, annotations, refusals, and ordering for request-detail views.
- **Spend-log resilience:** database writes use bounded configurable batches,
  one process-local writer at a time, deterministic lock ordering, prompt-safe
  error logging, and resilient cleanup.

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

## Focused Regression Suites

- `tests/test_litellm/llms/chatgpt/chat/test_chatgpt_transformation.py`
- `tests/test_litellm/llms/chatgpt/responses/test_chatgpt_responses_transformation.py`
- `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`
- `tests/llm_responses_api_testing/test_base_responses_api_streaming_iterator.py`
- `tests/test_litellm/proxy/test_litellm_pre_call_utils.py`
- `tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py`
- `tests/test_litellm/integrations/test_langfuse.py`
- `tests/test_litellm/proxy/db/test_db_spend_update_writer.py`
- `tests/proxy_unit_tests/test_update_spend.py`
- `tests/test_litellm/proxy/test_spend_log_cleanup.py`
