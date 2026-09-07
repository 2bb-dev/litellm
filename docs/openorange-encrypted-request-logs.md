# Encrypted request-log persistence draft

Production activation is blocked. This branch implements and tests the persistence boundary and the opt-in protected request profile, not a complete privacy deployment. Do not set the encryption environment variable in a production proxy. A consuming draft can pin this revision for isolated integration tests; that is not rollout approval

Refs: 2bb-dev/openorange#1330, 2bb-dev/openorange#1202, 2bb-dev/openorange#1325, 2bb-dev/openorange-platform#312

## Implemented boundary

When a public configuration path is present, `DBSpendUpdateWriter.update_database` encrypts the collected request-content row before its daily-accounting copies and before enqueueing to memory or the SQLite durable spool. The inference request and response are not modified by this transformation

The only retained content column is `proxy_server_request`, containing `{"format":"openorange.request-log.v1","jwe":"<compact JWE>"}`. `messages` and `response` contain empty JSON objects. The decrypted JSON has exactly `v`, `request`, `messages`, `response`, `metadata` and `request_tags`; `v` is integer 1

The protected JWE header contains exactly `alg=RSA-OAEP-256`, `enc=A256GCM`, `typ=openorange-request-content+jwe`, `kid`, integer `v=1`, `instanceUid`, `purpose=request-log` and `recordId`. The record identifier is the actual physical spend-row identifier, not a browser display identifier. Encryption uses the maintained BSD-3-Clause `joserfc` library and its `cryptography` backend; only the public RSA key is loaded by the writer

Plaintext metadata is a fixed projection of bounded attribution IDs, enumerated request classifications, booleans, numeric token/cache usage and numeric pricing/cost fields. Arbitrary metadata, provider error text, aliases, names, prompt hashes, request tags and tool names are inside the envelope. The writer does not populate tool-discovery catalogs in this mode. API-base URLs and content-derived cache keys are omitted from retained rows

The marker `metadata.openorange_request_log` reports `content_status=encrypted` or `capture_failed`, version 1, and available normalized conversation/session/cron facts. These facts are deliberately not reconstructed from ciphertext. Application-side attribution normalization must run before the writer

If encryption fails after a billed request, the writer queues a metadata-only `capture_failed` row and continues the existing key/user/team/organization/end-user and daily spend updates. An unexpected row-transform exception uses an independent content-free fallback and records `transformation_failed`. There is no plaintext fallback. Content that could not be encrypted is not retained; the failure is explicit rather than reported as an encrypted log

Standard LiteLLM diagnostic handlers suppress arbitrary arguments, exception text and extras in this mode, and the standard-payload stdout emitter is disabled. Router cooldown records retain status and timing but omit exception text. This does not cover every possible stdout writer or application-owned logger

## Public configuration contract

The intended settings are `OPENORANGE_REQUEST_LOG_ENCRYPTION_CONFIG` pointing to a regular public JSON file and `OPENORANGE_INSTANCE_UID` containing the immutable canonical UUID for the instance. A UID alone does not enable encryption

The public JSON shape is `{"version":1,"instanceUid":"<canonical UUID>","publicKey":<normalized public RSA JWK>}`. The JWK must contain only `kty`, `n`, `e`, `alg`, `use`, `key_ops` and `kid`; `kty` is RSA, `e` is AQAB, `alg` is RSA-OAEP-256, `use` is enc and `key_ops` is `["encrypt"]`. RSA modulus size is 3072 through 4096 bits. `kid` must equal the RFC 7638 SHA-256 thumbprint. Private JWK parameters, mismatched instances, symlinks and nonregular config files are rejected

The file is bounded to 16 KiB; the serialized content plaintext is bounded to 16 MiB. JSON/JWE serialization expands storage, roughly by the normal base64 factor plus the RSA wrapped key and protected header. The spend-log collector preserves complete strings in protected mode instead of applying the legacy per-field truncation limit, while still stripping credential-bearing `secret_fields`. Oversized complete content produces an explicit metadata-only capture failure; it is never silently reported as a complete encrypted log. Legacy plaintext truncation behavior is unchanged

Collection remains dependent on the existing `store_prompts_in_spend_logs: true` and `turn_off_message_logging: false` configuration. This draft does not override global or per-callback redaction. The database collector alone ignores a per-request `no-log` flag; third-party callback redaction is unchanged

The HTTP readiness routes and inference admission now reject an unavailable protected profile with 503. Public readiness exposes only `request_logging=unavailable`; authenticated details expose a bounded reason code. A transient crypto failure can clear after a successful probe. An unexpected transformation failure remains unhealthy until process restart

Loading a valid public configuration automatically creates and fsyncs a mode-0600 `<SPEND_LOG_DURABLE_QUEUE_PATH>.request-log-encryption-required` marker. Its contents bind the instance UUID and public-key thumbprint. Changing either identity, corrupting the marker or removing the configuration makes the profile unavailable. The dependency-light mode switch still recognizes the marker after the configuration environment variable is removed. Keep the durable queue path fixed across restarts; removing both that path and the configuration is outside this guard

## Protected request profile

The profile requires a database-backed durable spool, the mandatory database collector in both the proxy and asynchronous success registries, successful/error spend logging and the normal full-retention settings. Redaction conflicts are rejected; the profile never disables global or third-party redaction to obtain content. The collector alone remains exempt from per-request `no-log` and callback-disable controls

An exact callback allowlist covers the application metadata-normalization callback and the tested internal accounting, quota and routing hooks. Unknown callbacks in input/success/failure/service registries are rejected. Per-request callback, logger, cache, guardrail, prompt-management and retention overrides are rejected before the common request path initializes logging, and rechecked after key/team options are applied. Unsupported callbacks are also excluded from SDK success/failure dispatch and rejected-request failure/header processing

Response/semantic caches, detailed debug, raw request/response logging, external exception exporters, alerting and tracing conflict with this profile. Datadog tracing/profiling does not initialize in protected mode, and its enable settings are still reported as profile conflicts. Numeric routing and quota caches remain enabled

Arbitrary request tags are encrypted in the envelope and excluded from spend, daily and Redis counter/cache updates. Before admitting requests, a fixed boolean query against the authoritative database writer rejects any configured tag budget; query failure also rejects admission. The query does not include request tag text and does not populate tag caches. Existing unprotected tag-budget enforcement and counters are unchanged

Each policy query has a one-second timeout and fails closed on timeout. The standard authenticated path currently performs up to two writer round trips, at authentication and pre-call admission; readiness performs its own check. This adds pre-inference database latency and availability dependence. No zero-overhead or production latency claim is made

## Activation blockers and exclusions

Historical spool activation is blocked. `PrismaClient` constructs `SQLiteSpendLogSpool` during initialization, and background spend replay can run independently of HTTP readiness or inference admission. Existing queued/dead-letter rows are not inspected, migrated or encrypted by this draft. Adding a public configuration or seeing a healthy profile must not be interpreted as proof that an old spool is protected; an old plaintext queued row can still replay unchanged

This draft does not add startup/replay refusal for historical plaintext: that can interrupt service and billing replay and requires separate approval and implementation. Do not rely on automatic historical-spool rejection

An eventual activation procedure must first stop new traffic and reconcile/drain the legacy queue to the legacy destination, including pending in-memory and dead-letter accounting. It must then either explicitly migrate retained historical content or retain the old spool and sidecars under a separately approved historical-data policy, and provision a verified fresh empty spool for the protected process. No historical spool is to be deleted, silently abandoned or pointed at a protected replay worker. Start the protected worker only after that accounting/migration checkpoint; verify the public configuration, immutable marker, full sink profile and application SQL/readers with synthetic requests before any rollout. These are prerequisites, not operations performed by this draft

The consuming application's encrypted SQL projection/billing compatibility migration has not been applied or integration-tested. Audited ciphertext release, browser key handling and request-viewer integration are separate application changes. This branch has no claim that a complete database/backup dump is free of plaintext content

Historical plaintext rows, existing spool entries, backups, runtime transcripts, provider-side storage, other application databases and transit plaintext are outside this persistence-only draft. A public-key logger also cannot protect prompts from a malicious running inference process that receives the plaintext request

Do not roll a protected deployment back to a writer that does not understand this envelope. Application-owned loggers, historical plaintext and backup retention require their own validation even when LiteLLM readiness is healthy

## Validation and remaining proof

The focused tests use synthetic-only content and keys. They exercise real writer calls, the real accounting batch and entity queues, SQLite enqueue/dump/reopen, successful and failed provider rows, injected crypto and metadata-transform failures, configuration validation, diagnostic filtering and both Python/browser JWE directions. Mock-provider SDK streaming and nonstreaming tests verify unchanged caller output and full encrypted capture with normal retention settings and `no-log: true`. Policy regressions cover admission/readiness, configuration identity binding, supported/unsupported callbacks, redaction conflicts, content-free tag-budget checks, rejected-request sinks, provider cache usage and legacy behavior

Run the focused regressions with:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -p asyncio -p pytest_mock tests/test_litellm/proxy/spend_tracking/test_request_content_encryption.py tests/test_litellm/proxy/spend_tracking/test_request_content_policy.py tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py tests/test_litellm/proxy/db/test_db_spend_update_writer.py tests/litellm/test_effective_token_pricing.py tests/test_litellm/router_utils/test_cooldown_cache.py -q
```

The cross-language fixture is explicitly test-only and includes a synthetic private key. It must never be used for a deployment

Paid-provider curl proof, a complete activated-proxy soak, actual shared SQL billing reconciliation, a complete content-dump inventory and browser reveal audit integration remain unperformed. Test commands are local regression evidence, not end-to-end proof of production privacy
