# OpenOrange LiteLLM Fork

`main` mirrors upstream LiteLLM. `openorange` is the production integration
branch consumed by OpenOrange through an exact submodule commit.

Layer and Platform land pins only to commits reachable from `openorange`. A
Layer PR may carry an open fork PR head for review, but must wait for that head
to land on `openorange` before merging. Preserve fork merge history so the
reviewed pin stays reachable and later integration pins retain its behavior.

## Fork Invariants

Upstream syncs must preserve these behaviors:

- **OSS-only packaging:** proxy builds must not install, copy, or require
  `litellm-enterprise` or the `enterprise` workspace. Preserve the upstream
  MIT notice and the native dependency/compiler notices in built wheels and
  runtime images; source omission does not permit removing required notices.
- **Private Pi inference:** `services/pi-inference` remains an independently
  packaged, OSS-only inference sidecar, not an agent runtime. Only explicitly
  enabled aliases are callable. Its internal key and optional per-slot credential
  volume stay separate from client authentication; LiteLLM remains the sole
  public model gateway and spend source. The sidecar serves native Messages only;
  LiteLLM translates Chat clients on the same route. Preserve terminal usage and
  credential-safe correlated stdout traces.
  See `services/pi-inference/README.md` for the API, packaging, and opt-in smoke
  contract; fork CI builds and tests this service without paid inference.
- **Forced tool choice:** on a model whose map entry sets
  `supports_forced_tool_use: false`, a forced Chat `tool_choice` (`required` or
  a named function) is a client-side 400 on the Anthropic, Vertex and Bedrock
  Converse paths even with `drop_params`. Upstream downgrades it to `auto`,
  which silently drops the caller's forced-call contract. Unsupported sampling
  parameters keep the normal `drop_params` behavior. The
  `test_forced_tool_choice_raises_clean_error_*` tests cover both paths.
- **ChatGPT subscription routing:** Responses state remains persistent where
  required, upstream storage stays disabled, prompt-cache parameters survive
  transformation, the ChatGPT session header follows `prompt_cache_key`,
  string inputs are normalized for the subscription backend, client `system`
  messages are represented as `developer` messages without changing their
  content or order, and
  provider-forced SSE is accumulated into one complete response for
  non-streaming callers without duplicate streaming hooks or spend logs.
  Requests still prepend the Codex CLI prompt for the backend, but response
  bodies and their request logs return only the caller's own `instructions`.
- **OpenClaw attribution:** trusted runtime context and supported OpenClaw
  payload markers continue to populate actor, parent, session, channel,
  execution, and Langfuse metadata without persisting raw credentials.
- **Context overflow recovery:** router pre-call context errors retain the
  `context_length_exceeded` marker so clients recognize overflow and can
  compact and retry. Preserve configured limits and unrelated error types.
- **ChatGPT rate limits:** transient subscription 429 responses remain eligible
  for bounded backoff retries. Only structured `usage_limit_reached` or
  `insufficient_quota` codes trigger quota cooldown and skip retries, including
  through a `litellm_proxy/chatgpt/` sidecar. Native-provider and non-429 error
  policies are unchanged.
- **Claude subscription quota:** only a Claude subscription account that is out
  of quota moves a call on. The Pi slot answers it with a 429 carrying
  `x-openorange-subscription-exhausted-until` (unix seconds) when Anthropic's
  429 says `anthropic-ratelimit-unified-status: rejected`, and refuses that
  model without calling Anthropic until the reset. LiteLLM never retries that
  answer on the same deployment. On groups with
  `model_info.rate_limit_fallback_requires_exhaustion`, any other 429 and an
  overload are retried with backoff and never fall back, except the slot's own
  concurrency refusal (`x-openorange-pi-slot-at-capacity`); other errors fall
  back as before. The proxy repeats the header to its caller unless an account
  it reached was only busy or full, and `order_fallback_on_rate_limit_only`
  (the workspace's paid hop) needs that header. The subscription tests in
  `tests/unit/test_router/test_router.py` cover both proxies and both API
  surfaces; `tests/unit/router_utils/test_subscription_exhaustion.py` covers
  the signal itself.
- **Video jobs through a second proxy:** a workspace LiteLLM forwards video
  jobs to the central LiteLLM through an OpenAI-compatible deployment. A video
  ID the upstream proxy encoded for a different provider is wrapped once more
  with the forwarding deployment, so status, content and remix calls reach that
  deployment and pass the upstream ID on unchanged. An ID already encoded for
  the same provider is never wrapped twice. The
  `test_encode_video_id_wraps_an_id_another_proxy_encoded_for_a_different_provider`
  test covers both.
- **xAI video jobs:** `xai/grok-imagine-video` serves the OpenAI video API
  (create, status, content) from xAI's `/v1/videos/generations` and
  `/v1/videos/{request_id}`. Every create names its seconds and resolution, so
  the per-second price charged at create (`output_cost_per_second_<resolution>`)
  is for what xAI renders. Seconds outside 1 to 15, a resolution the model can't
  render, an unknown aspect ratio, or no prompt and no image are refused before
  xAI accepts a job that would fail. Only fields the per-second price covers are
  sent: stored outputs, upload URLs, reference media and keyframes stay out.
  Polls and downloads carry no usage and are never priced; the download fetches
  xAI's temporary file URL without the API key. A deployment forwarding video to
  another LiteLLM proxy keeps the `video_resolution` that proxy priced, so its
  own cost uses the same tier. Covered by
  `tests/unit/llms/xai/videos/test_xai_video_transformation.py`.
- **Images through a second proxy:** a workspace LiteLLM reaches the central
  LiteLLM's image routes as `litellm_proxy/<provider>/<model>`. Their cost is
  computed by that provider's calculator with the deployment's own prices, so a
  token-priced model (GPT Image, Gemini) is charged by the tokens its answer
  reports, as on the central proxy. Without it the per-image default found no
  price and logged $0, so budgets never counted those images. An alias without a
  known provider keeps the default. Covered by
  `test_route_image_generation_cost_prices_a_forwarded_route_as_its_provider`.
- **Image generation is never retried by the SDK:** the OpenAI image handler
  builds its client with `max_retries` 0 unless the caller passes one. Each
  retry is a second paid generation, and a timed-out first attempt can still
  finish and bill. The router's retry policy decides everything else. Covered by
  `test_openai_image_generation_retries.py`.
- **Transcriptions are never free:** a transcription priced by the second is
  billed for its audio's length: the longest of the upload's length and any
  length the provider reports (`duration`, `usage.seconds`). Upstream read only
  what soundfile opens, so m4a, mp4 and webm uploads were priced at $0. soundfile
  still reads what it recognizes, except MP3, which is measured by its frames (a
  Xing header can state far fewer). `audio_utils/container_duration.py` reads the
  rest: MP4 (m4a, mov, fragmented recordings), Matroska and WebM (live
  recordings without sizes or a duration), ADTS AAC and a WAV whose sizes were
  never written. The length is what a decoder plays: the samples and timestamps
  of the audio tracks, each track on its own and measured from its first sample
  (a segment cut from a longer recording keeps its timestamps); an MP4's sample
  table and the fragments after it add up; a duration a header declares counts
  only when there are no samples or blocks. An MP3 soundfile opens is measured
  by its MPEG frames when frames that follow one another make up at least half
  its bytes; otherwise (a free-format MP3, whose frames this reader doesn't
  parse) the longer of its frames and soundfile's reading counts, so neither a
  Xing header that understates nor soundfile's estimate for a VBR file without
  one decides alone. A file soundfile can't open reads as a stream of ADTS or
  MPEG frames only when four matching frames come in a row, and then as the
  longer of the two kinds, so a file holding both bills the longer. Like a
  decoder, the readers skip stray bytes: a frame stream
  resyncs only on a valid frame header, so any number of stray stretches costs
  one search each, and an ID3 tag after the first is stray bytes whose frames
  still count. EBML integers longer than 8 bytes read as 0, and a file with more
  Matroska elements (skipped stretches and BlockGroup fields included) or MP4
  fragments than any recording reads as unmeasured, as does a length soundfile
  can't tell (2^63 - 1 frames, as for a FLAC written to a pipe). Header lies
  soundfile trusts (FLAC STREAMINFO, an Ogg granule) and Matroska or MP4 timing
  lies still bill what they state, unless the provider reports more. A length is
  never more than the bytes carry at 100 bits a second, below even the silence
  of FLAC or of Opus with DTX. The length is read
  once, before the provider is called, and a route priced by the second (not by
  tokens) refuses audio whose length can't be read with a 400, so it never
  reaches a provider that would charge for it. Covered by
  `test_container_duration.py`, `test_audio_utils.py` and the
  `test_atranscription_*` / `test_transcription_*` cases in
  `tests/unit/test_main.py`.
- **Retry privacy and limits:** history is request-local, contains at most four
  flat allowlisted records, and excludes prompts, credentials and exception
  text. A private request counter survives ordinary and streaming fallbacks
  independently of history truncation. Preserve proxy metadata identity for
  post-call callback writes while isolating caller-shared SDK dictionaries.
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

- `tests/test_litellm/router_utils/test_chatgpt_rate_limit.py`
- `tests/unit/router_utils/test_subscription_exhaustion.py`
- `tests/test_litellm/test_effective_token_pricing.py`
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
- `tests/unit/llms/xai/videos/test_xai_video_transformation.py`
- `tests/unit/litellm_core_utils/audio_utils/test_container_duration.py`
- `tests/unit/litellm_core_utils/test_audio_utils.py`

## Stable v1.101.0 integration

The sync retains the published stable tag18243cd7af4c3325165ba68b21379e2719e051c7
as a merge parent. Native off-peak windows, public catalog updates, Responses
prompt-cache options, deadlock classification and bounded fallback traversal
come from upstream. OpenOrange extends those owners for dated deployment
tariffs, subscription transport, authenticated ownership/terminal evidence,
protected content, and local durable spend buffering.

The separate off-peak implementation, duplicated Anthropic cache-cost helper,
custom catalog-fetch injection and obsolete public catalog overrides were
removed after focused parity checks. Keep regression coverage when deleting
an override. Avoid a second pricing or retry implementation in Layer callbacks.

The actual native build uses root rust-toolchain.toml. The old nested toolchain
pin was removed; both fork and Layer images compile with the same pinned Rust
version and carry its standard-library notice. Wheel and runtime-image guards
check the shipped code and notice hashes, not just the source dependency list.
