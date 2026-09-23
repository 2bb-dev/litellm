# Private Pi inference

An inference-only sidecar for this LiteLLM fork, using locked
`@earendil-works/pi-ai@0.87.1` and `zod@4.4.3`. LiteLLM owns client keys,
routing, and spend accounting. This service has no agent loop, executable tools,
workspace access, shell endpoint, public login flow, or separate spend ledger

## Build and run

Use Node **22.23.2**. From this fork's root, build the OSS LiteLLM image separately
(or select an already-built image from this fork):

```bash
docker build -t litellm-openorange:local .
cd services/pi-inference
npm ci --ignore-scripts
npm run typecheck
npm run build
npm test
npm run lint
node dist/main.js catalog
docker build -t pi-inference:local .
```

The catalog command needs neither credentials nor configuration and makes no
inference calls. It lists all built-in provider/model metadata, not enabled
routes or an authoritative price list. Only aliases in `config.example.json`
are callable; having a provider credential does not enable its models. The
example enables alias `claude-haiku-4-5` as Anthropic's built-in model of the same
name

Load `ANTHROPIC_API_KEY` from your secret manager into the host environment.
Use an Anthropic **API key**, not a Claude subscription/session token, for this
example. Generate separate internal and front-door keys, then start the example:

```bash
export LITELLM_IMAGE=litellm-openorange:local
export PI_INFERENCE_API_KEY="$(openssl rand -hex 32)"
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"
docker compose -f compose.example.yml config --quiet
docker compose -f compose.example.yml up -d --build --wait
```

The front proxy alone publishes `127.0.0.1:4000`. The sidecar has **no published
port**; its authenticated API is reachable only on the dedicated Compose bridge
by attached containers. This bridge allows outbound provider HTTPS, so it is not
an air gap or an egress firewall. Do not attach untrusted containers or publish
port 4001. Use your normal LiteLLM deployment for production client-key, database,
spend persistence, and ingress controls; this small example does not configure them

The sidecar runs as UID/GID 1000 with a read-only root, a bounded `/tmp` tmpfs,
dropped capabilities, and no host shell, Docker socket, home, or workspace mounts.
Only its model configuration is bind-mounted read-only. Docker builds use this
service directory as their entire context, an allowlist `.dockerignore`, pinned
Node image digest, `npm ci --ignore-scripts`, and production-only dependencies
in the final stage. No Enterprise or Pi agent package is installed

## Configuration and credentials

`PI_INFERENCE_API_KEY` must contain at least 32 characters and is private to
LiteLLM and this service. Both example LiteLLM routes read that same key via
`os.environ/PI_INFERENCE_API_KEY`;
it is **not** the front LiteLLM master/virtual key. Runtime settings are
`PI_INFERENCE_CONFIG=/config/models.json`, `PI_INFERENCE_HOST=0.0.0.0`,
`PI_INFERENCE_PORT=4001` (valid range 1–65535),
`PI_INFERENCE_SLOT_ID=slot1`, and `PI_INFERENCE_TIMEOUT_MS=600000` (the whole
request deadline, 1000–3600000; streams get a `ping` or keep-alive comment every
15 seconds of silence). The image entrypoint is
`node dist/main.js`. Restart after changing the enabled model configuration

All Pi built-in providers are available, but the explicit `models` allowlist
controls inference. Each entry has `alias`, `provider`, and `model`. Advanced
entries may override `baseUrl`; unknown models require complete `metadata`
(`api`, `contextWindow`, `maxTokens`, `reasoning`, `input`, and `cost` with
`input`, `output`, `cacheRead`, `cacheWrite`). Unknown providers additionally
require `baseUrl`; their API key environment variable is the uppercase provider
ID with non-alphanumeric characters replaced by `_`, followed by `_API_KEY`.
Treat configuration and endpoint overrides as trusted operator input

Provider environment keys are the simplest credential path. Optionally set
`PI_INFERENCE_AUTH_FILE=/data/auth.json` before starting Compose and provision a
Pi-compatible credential document using trusted operator tooling. The isolated
named volume `pi-inference-auth-slot1` at `/data` is reserved for this service;
never mount an agent's live credential store. The document is keyed by provider:
API keys use `{ "type": "api_key", "key": "..." }`, and existing OAuth credentials
use provider-compatible `{ "type": "oauth", "access": "...", "refresh": "...",
"expires": 0 }` fields with a real expiry in milliseconds. Keep directory mode
0700, file mode 0600, and ownership 1000:1000. The service creates an empty store
if absent, serializes refreshes, and persists refreshed credentials atomically
in its own store. Run one process per credential volume; use distinct volumes,
keys, and slot IDs for additional instances. `proper-lockfile@4.1.2` maintains a
heartbeat lock: live owners fail closed, and an unclean exit requires at least
10 seconds before a new process can recover the stale lock. Never delete a live
lock. Back up credentials securely; `docker compose down -v` deletes the volume

The inference service exposes no login or public OAuth/callback endpoint. Import
only credentials you are authorized to use and only where the provider permits this
usage. A hosted Claude subscription is not an API entitlement: subscription
limits, enabled extra-usage billing, and permission for hosted/third-party access
are separate questions. Refresh capability does not establish that permission

## Claude OAuth compatibility

For Anthropic OAuth, the backend includes an inference-only adaptation of
[`@benvargas/pi-claude-code-use@2.2.0`](https://github.com/ben-vargas/pi-packages/tree/4eaa1e26e44151a01c6977354e7c539322f048be/packages/pi-claude-code-use).
It does **not** install or load a Pi CodingAgent extension. There is no `pi install`
or extension reload step; rebuild the image and restart the sidecar to update it,
retaining its isolated credential volume

The adapter runs only for provider `anthropic`, API `anthropic-messages`, and a
resolved OAuth token matching Pi's `sk-ant-oat` check. API-key requests, other
providers, and explicit SDK client overrides are unchanged. Credential lookup
and refresh remain owned by Pi and the existing credential store

Custom flat tool names become bounded, collision-safe `mcp__pi__…` aliases for
upstream requests. Claude Code core names and existing `mcp__` names retain Pi's
normal handling. Definitions, forced native tool choice, and history use the
same per-request map; responses, including streaming, restore client names.
There is no shared tool registry or tool execution. IDs, arguments, schemas,
cache metadata, and signed/redacted thinking are not rewritten

After native payload overlays, only system text receives these exact substitutions:
`pi itself` → `the cli itself`, `pi .md files` → `cli .md files`, and
`pi packages` → `cli packages`. User messages, tool results, and response prose
are unchanged. The upstream extension's UI hooks and raw-payload debug logging
are not included

This compatibility behavior does not grant API access or guarantee that requests
use subscription credits rather than paid extra usage. Provider permissions,
quota, and billing must be checked separately; no new runtime package is installed

## API and routing

Unauthenticated `GET /health/liveliness` and `/health/readiness` report local
process/configuration readiness, not provider credential validity or quota.
Authenticated routes are `GET /v1/models`, `GET /v1/providers`,
`POST /v1/chat/completions`, and `POST /v1/messages`. Only enabled aliases appear
in model discovery; provider discovery describes supported integrations, not
authorization to call every model. `POST /v1/responses` is explicitly **501**

In `litellm.example.yaml`, client alias `anthropic/claude-haiku-4-5/pi` uses
`model: anthropic/claude-haiku-4-5` with root `api_base: http://pi-inference:4001`
for native Messages. Chat alias `pi/claude-chat` uses
`model: litellm_proxy/claude-haiku-4-5` with
`api_base: http://pi-inference:4001/v1`. Keep these prefixes distinct. The sidecar
alias matches the built-in model ID so native LiteLLM logging can resolve the
model instead of reporting an unmapped generic alias

`model_info.pi_api` labels the wire protocol. Per-token rates appear in both
`litellm_params` and `model_info`, alongside `custom_pricing: true`, `mode: chat`,
and explicit context/output limits. Keep the duplicated rates aligned.
These rates and `pricing_label` are operator-owned **examples**, not authoritative
catalog or subscription prices; verify your contract before accounting against
them

Both APIs return terminal usage and support text streaming. Native Messages
retains supported thinking/tool blocks and refusal stop reasons; tool definitions
are data, never locally executed. `stop_sequences` is explicitly unsupported.
The generic Chat bridge does not preserve opaque/signed reasoning round-trips;
use native Messages for Claude. Chat tool choice supports only `auto` and `none`,
and strict tool schemas are unsupported on that bridge. Other unsupported request
fields/features are explicitly rejected. This is not a complete drop-in
implementation of every upstream API

Structured stdout traces correlate request/trace IDs with slot, route/provider,
latency, status, and usage, without logging prompts, responses, or secrets.
LiteLLM remains the spend source. No OTLP exporter or external trace delivery is
configured by this service; collect stdout with your normal log infrastructure

## Opt-in smoke test

Run this client **only against the front LiteLLM proxy**, never the sidecar.
Use a front LiteLLM master/virtual key that can discover and call the selected
alias. Without `INFERENCE_CONFIRM_PAID=1`, it checks authenticated model discovery
only. It rejects redirects and the example sidecar address/port, and never prints
keys or response bodies

```bash
export INFERENCE_API_BASE=http://127.0.0.1:4000
export INFERENCE_API_KEY="$LITELLM_MASTER_KEY"
export INFERENCE_PROTOCOL=chat
export INFERENCE_MODEL=pi/claude-chat
node tests/smoke.mjs
INFERENCE_CONFIRM_PAID=1 node tests/smoke.mjs
INFERENCE_PROTOCOL=messages \
  INFERENCE_MODEL=anthropic/claude-haiku-4-5/pi \
  INFERENCE_CONFIRM_PAID=1 node tests/smoke.mjs
```

Each confirmed run makes **two billable requests** (non-streaming and streaming)
and verifies text, protocol completion, and terminal usage with bounded SSE
parsing. CI runs offline tests, formatting/type checks, production license guards,
and a Docker build; it neither calls live providers nor pushes images

The fork regression job also exercises real Python LiteLLM transport through a
local sidecar and provider stub for Chat and native Messages, each streaming and
non-streaming. It also verifies Anthropic OAuth tool aliases and system rewriting
with fake credentials, including client-name restoration through real LiteLLM
transport. With the fork's uv test dependencies already synced and this service
built, reproduce that non-billable check from the fork root:

```bash
npm run test:litellm --prefix services/pi-inference
```

When the lock changes, re-check every production package's installed metadata
and licenses, and escalate unclear/restricted licenses under OpenOrange's
`docs/compliance/third-party-licenses.md`. The parent repository's current
scanner does not automatically discover this nested npm root
