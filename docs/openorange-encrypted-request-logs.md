# Encrypted request-log persistence draft

Production activation is blocked. This branch implements and tests the persistence boundary, not a complete privacy deployment. Do not set the encryption environment variable in a production proxy or advance the consuming application's LiteLLM pin yet

Refs: 2bb-dev/openorange#1330, 2bb-dev/openorange#1202, 2bb-dev/openorange#1325, 2bb-dev/openorange-platform#312

## Implemented boundary

When a public configuration path is present, `DBSpendUpdateWriter.update_database` encrypts the collected request-content row before its daily-accounting copies and before enqueueing to memory or the SQLite durable spool. The inference request and response are not modified by this transformation

The only retained content column is `proxy_server_request`, containing `{"format":"openorange.request-log.v1","jwe":"<compact JWE>"}`. `messages` and `response` contain empty JSON objects. The decrypted JSON has exactly `v`, `request`, `messages`, `response`, `metadata` and `request_tags`; `v` is integer 1

The protected JWE header contains exactly `alg=RSA-OAEP-256`, `enc=A256GCM`, `typ=openorange-request-content+jwe`, `kid`, integer `v=1`, `instanceUid`, `purpose=request-log` and `recordId`. The record identifier is the actual physical spend-row identifier, not a browser display identifier. Encryption uses the maintained BSD-3-Clause `joserfc` library and its `cryptography` backend; only the public RSA key is loaded by the writer

Plaintext metadata is a fixed projection of bounded attribution IDs, enumerated request classifications, booleans, numeric token/cache usage and numeric pricing/cost fields. Arbitrary metadata, provider error text, aliases, names, prompt hashes, request tags and tool names are inside the envelope. The writer does not populate tool-discovery catalogs in this mode. API-base URLs and content-derived cache keys are omitted from retained rows

The marker `metadata.openorange_request_log` reports `content_status=encrypted` or `capture_failed`, version 1, and available normalized conversation/session/cron facts. These facts are deliberately not reconstructed from ciphertext. Application-side attribution normalization must run before the writer

If encryption fails after a billed request, the writer queues a metadata-only `capture_failed` row and continues the existing key/user/team/organization/end-user and daily spend updates. An unexpected row-transform exception uses an independent content-free fallback and records `transformation_failed`. There is no plaintext fallback. Content that could not be encrypted is not retained; the failure is explicit rather than reported as an encrypted log

Standard LiteLLM diagnostic handlers suppress arbitrary arguments, exception text and extras in this mode, and the standard-payload stdout emitter is disabled. This does not cover every possible stdout writer or external integration

## Public configuration contract

The intended settings are `OPENORANGE_REQUEST_LOG_ENCRYPTION_CONFIG` pointing to a regular public JSON file and `OPENORANGE_INSTANCE_UID` containing the immutable canonical UUID for the instance. A UID alone does not enable encryption

The public JSON shape is `{"version":1,"instanceUid":"<canonical UUID>","publicKey":<normalized public RSA JWK>}`. The JWK must contain only `kty`, `n`, `e`, `alg`, `use`, `key_ops` and `kid`; `kty` is RSA, `e` is AQAB, `alg` is RSA-OAEP-256, `use` is enc and `key_ops` is `["encrypt"]`. RSA modulus size is 3072 through 4096 bits. `kid` must equal the RFC 7638 SHA-256 thumbprint. Private JWK parameters, mismatched instances, symlinks and nonregular config files are rejected

The file is bounded to 16 KiB; the serialized content plaintext is bounded to 16 MiB. JSON/JWE serialization expands storage, roughly by the normal base64 factor plus the RSA wrapped key and protected header. Existing request-content truncation limits still apply before this writer

Collection remains dependent on the existing `store_prompts_in_spend_logs: true` and `turn_off_message_logging: false` configuration. This draft does not override global or per-callback redaction. The database collector alone ignores a per-request `no-log` flag; third-party callback redaction is unchanged

`encryption_readiness()` exposes configuration/crypto failures for future admission integration. A transient crypto failure can clear after a successful probe. An unexpected transformation failure remains unhealthy until process restart. This helper is not connected to the HTTP readiness route or inference admission in this draft

The dependency-light mode helper recognizes an existing `<SPEND_LOG_DURABLE_QUEUE_PATH>.request-log-encryption-required` marker even after the configuration environment variable is removed. `require_protection_marker()` can durably create that marker, but production activation does not call it yet. The marker cannot currently be relied on as an automatically installed accidental-disable guard

## Activation blockers and exclusions

The protected-profile readiness/admission gate, approved callback allowlist, dynamic logging-override checks, debug/export/cache exclusion and mandatory collection checks are not implemented. A configured third-party sink may still receive content independently of this writer. Global or per-callback redaction may still remove content before it reaches the writer

Tag-based budgets are not supported by the proposed protected profile. The writer removes arbitrary tags from its persisted rows and daily queues, but the separate cost-callback path still supplies raw tags to Redis spend counters/cache updates. That path must be addressed together with an explicit rejection of configured tag-budget policies; silently omitting tag-budget accounting is not acceptable

The consuming application's encrypted SQL projection/billing compatibility migration has not been applied or integration-tested. Audited ciphertext release, browser key handling and request-viewer integration are separate application changes. This branch has no claim that a complete database/backup dump is free of plaintext content

Historical plaintext rows, existing spool entries, backups, runtime transcripts, provider-side storage, other application databases and transit plaintext are outside this persistence-only draft. A public-key logger also cannot protect prompts from a malicious running inference process that receives the plaintext request

Do not roll a protected deployment back to a writer that does not understand this envelope. A later activation procedure must verify all readers and migrations, drain or migrate old queue entries, install the durable marker, validate the entire sink profile, and separately account for historical plaintext and backup retention

## Validation and remaining proof

The focused tests use synthetic-only content and keys. They exercise real writer calls, the real accounting batch and entity queues, SQLite enqueue/dump/reopen, successful and failed provider rows, injected crypto and metadata-transform failures, configuration validation, diagnostic filtering and both Python/browser JWE directions. Mock-provider SDK streaming and nonstreaming tests verify unchanged caller output and full encrypted capture with normal retention settings and `no-log: true`

Run the focused regressions with:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -p asyncio -p pytest_mock tests/test_litellm/proxy/spend_tracking/test_request_content_encryption.py tests/test_litellm/proxy/spend_tracking/test_spend_tracking_utils.py tests/test_litellm/proxy/db/test_db_spend_update_writer.py -q
```

The cross-language fixture is explicitly test-only and includes a synthetic private key. It must never be used for a deployment

Paid-provider curl proof, a complete activated-proxy soak, actual shared SQL billing reconciliation, a complete content-dump inventory and browser reveal audit integration remain unperformed. Test commands are local regression evidence, not end-to-end proof of production privacy
