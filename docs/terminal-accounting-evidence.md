# Terminal accounting evidence

This opt-in producer requires Platform's terminal receipt and separate measurement authority (Platform PR458, source `2c6712bc04e9065e9d1e5bfcb4111b63f16a0541`) and an independently verifying billing consumer. It does not activate registrations, change a deployment pin, or certify imported historical SpendLogs

The local Router assigns each receipt-enabled attempt a private UUID. That UUID becomes its SpendLogs `request_id`; `model_id` remains the actual local deployment. The private root request ID groups attempts. Every failed local fallback has a distinct row. Each signed terminal FINISH describes the actual selecting terminal deployment, with its own registration proof. A proxy connection credential cannot establish terminal ownership

## Explicit configuration

`OPENORANGE_TERMINAL_RECEIPT_CLIENT_CONFIG_FILE` must name an absolute private regular 0600 file. Its closed version1 configuration contains `role` (audience or producer), the principal `identity`, configured `issuer`, authority `base_url`, private `bearer`, and versioned `public_keys` mapping key UUIDs to Ed25519 public PEM. Audience `upstreams` explicitly allow the central proxy endpoints; `delegates` maps a producer's local relay deployment to the authenticated downstream producer UUID. Timeout defaults to two seconds, with a ten-second maximum. HTTPS is required except for loopback fixtures

Platform must separately configure the principals, allowed delegation edges, signing custody, and immutable route/credential registrations. Native credential registrations use the reviewed ownership producer contract. A ChatGPT selecting sidecar additionally uses `OPENORANGE_CHATGPT_CREDENTIAL_REGISTRATION_FILE`: a private 0600 registration binding the explicit source and immutable registration UUID/revision to the selected account SHA256 and reviewed endpoint. The account snapshot and token come from the same authenticator read and must match the actual post-override HTTP dispatch. Account hashes, tokens, and private registration configuration are not retained in readable evidence

Missing configuration or registration is not platform ownership. Unknown, mismatched, overridden, unavailable, malformed, or incomplete facts remain unknown/pending. No automatic registration or existing-row classification is provided

## Independent facts

`metadata.openorange_terminal_evidence` retains the unchanged eight-field local wrapper and exact original ownership JWS. `metadata.openorange_terminal_usage_evidence` separately retains the matching eight-field measurement wrapper and original `oo-terminal-usage+jwt` proofs. The latter binds every identity and the exact ASCII SHA256, ID, and sequence of its FINISH. It does not extend the ownership JWS header type

The receiver verifies the configured issuer/public key, audience, local and terminal bindings, original FINISH hash, balanced ownership sequence/seal, and exactly one original measurement for every FINISH. The billing API must independently repeat this verification on capture and replay against explicit trusted public keys. Readable projection first validates closed canonical content-free claims; it retains no extensible opaque-token escape hatch

Limits remain 16 terminal attempts, 33 ownership events, 16 usage originals, 3072 ASCII bytes per original, and 65536 compact UTF-8 bytes per evidence wrapper. Whole invalid/overflow payloads become fixed safe states; no clipping asserts completeness. Null correlation is allowed only for an empty unavailable/invalid admission wrapper, preserving private local IDs before network admission. Combined consumer ledger limits remain independently enforced

The private derivative `openorange_usage_observation` has exactly ten fields: version, local attempt, state, and seven nullable counters. Local SDK/default/estimated values cannot populate its authoritative snapshot. Exactly one terminal FINISH with authenticated observed usage and successful local completion yields observed scalar values. Local failure or authenticated partial terminal usage preserves partial values. Multiple terminal FINISH records retain every original but produce an unobserved all-null scalar. Multi-attempt aggregation remains unresolved implementation scope, never last-wins or a sum priced at the successful tail model

## Native measurement mappings

Anthropic raw input is normalized to inclusive ledger prompt tokens only when uncached input, cache read, and cache creation are all explicitly present. Final streaming completion requires the current message_delta's actual output counter; an earlier message_start cursor cannot substitute. Total is null when not explicitly supplied. Cache creation duration fields are retained only when supplied

DeepSeek uses actual prompt/completion/total and maps `prompt_cache_hit_tokens` to cache read. Interim stream usage remains partial until a terminal usage frame. ChatGPT Responses uses actual input/output/total and input detail cached tokens from the terminal Responses event. Missing cache writes remain null in both adapters. Unsupported native measurement adapters remain unobserved

The consumer requires prompt/completion/cache-read/cache-write before any settled disposition, with additional duration quantities where the owned tariff requires them. Consequently normal DeepSeek/ChatGPT responses lacking writes remain pending. No absent quantity is manufactured as zero. Supplier execution, supplier measurements, local delivery, credential ownership, and customer tariff authority remain independent; HTTP errors cannot release unknown supplier liability

## Validation and rollout boundary

The checked-in loopback harness exercises actual Router/native HTTP adapters through Platform's signer and SQLite authority, then the actual SpendLogs projection. Its supplier HTTP responses and signing principals are test-only. Set `OPENORANGE_TEST_PLATFORM_CHECKOUT` to the reviewed Platform usage checkout and use Node24 plus the installed Python test environment. `OPENORANGE_TEST_TERMINAL_ARTIFACT_DIR` exports immutable content-free original-byte bundles, paired public trust, every local row, cases, and module SHA256 mappings. No external provider calls are made by these tests

Focused tests cover source spoofing, fallback identity, native late overrides, ChatGPT account changes, final versus interim measurements, explicit zero versus absence, early close/cancellation, failure recovery and SQLite acknowledgement, encryption/transform failure, durable spool/replay, and original proof mismatch. Independent billing SQL acceptance and final source mappings remain required. Fork hosted checks are absent; local tests are not represented as hosted CI or consuming-image acceptance

This source is not a dedicated-capacity rollout or a production-ready billing claim. Explicit registrations, isolated fresh producer source data, compatible reviewed consumer/Platform sources, remaining measurement/aggregation implementation, combined artifact acceptance, and final human architecture acceptance remain distinct gates
