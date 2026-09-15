# Supplier credential ownership evidence

The spend producer emits `metadata.openorange_credential_ownership` for each logged attempt. It describes the registered ownership of the selected upstream credential; it does not assert supplier execution, settle supplier liability, or set customer prices

```json
{
  "v": 1,
  "source": "platform",
  "provenance": "route_registration",
  "registration_id": "3e15c0c2-feca-4104-a648-8d579315ef51",
  "registration_revision": "e7ca1c3e-b6ea-4ce0-8538-b8f149448038",
  "deployment_id": "selected-deployment"
}
```

Sources are `platform`, `byok`, and `unknown`. Positive facts use `credential_registration` or `route_registration`. Unknown facts use `unregistered`, `credential_override`, or `ambiguous`, with null registration fields. Deployment ID is the actual selected deployment, where established. All fields are content-free identifiers or enums; credential names, keys, hashes, endpoints, and tenant identifiers are excluded

## Explicit registration is required

Nothing in this change registers or classifies existing credentials. A personal subscription, provider name, static environment variable, caller key, model alias, missing DB row, or old credential-source label does not establish ownership

A proxy administrator may explicitly store this declaration in `credential_info.openorange_credential_ownership` through the existing credential management endpoint:

```json
{
  "v": 1,
  "source": "byok",
  "registration_id": "3e15c0c2-feca-4104-a648-8d579315ef51",
  "registration_revision": "e7ca1c3e-b6ea-4ce0-8538-b8f149448038"
}
```

The declaration is a control-plane policy assertion, not a provider ownership lookup. Registration writers must establish ownership independently and allocate fresh UUID revisions. Credential edits invalidate a retained declaration unless a valid new revision accompanies the edit. The persisted and in-memory registry receive the same declaration; completion logging never looks up a newer current row

Reviewed static server configuration may use the same declaration under deployment `model_info.openorange_credential_ownership`. Its explicit API key and API base must be bound to that reviewed configuration revision. Configuration changes to credentials or endpoints must carry a fresh reviewed registration revision. Arbitrary DB model-level declarations are not accepted as static registration authority; DB routes can instead select a separately registered named credential

## Bounded transport support

This first slice can establish positive evidence for `openai/` and `venice/` routes with a simple explicit API key and API base. The provider prefix limits inspected transport support; it never determines who owns the credential. Named credentials must contain only that simple binding. Mixed registrations, unresolved environment references, custom authentication, unsupported providers, mocks, request credential/endpoint overrides, and unproven clients remain unknown

Concrete OpenAI SDK clients are checked against the selected key and endpoint. Custom headers, HTTP authentication, request/response hooks, custom transports, and global client/header overrides cannot inherit positive ownership. Unsupported paths still retain usage, completed model/provider, native spend, and unknown ownership. Extending positive support requires proving the actual credential binding for the additional transport

Router selection creates a private typed context for each attempt, including each fallback. Credential loading creates a private stamp on the logging object from the same registry snapshot that supplies the key. Caller JSON cannot construct these Python types. The spend builder strips reserved caller fields, including nested copies, and serializes only that stamp. Unstamped paths produce unknown

The encrypted request-log projection and its crypto/transform failure paths retain the same validated content-free fact. Durable replay uses the recorded event fact and does not reclassify it after rotation or model deletion. A native HTTP failure is not evidence of supplier non-execution

## Consumer and release boundary

Consumer migration 0117 follows 0116 and leaves 0105 unchanged. Established BYOK is OO-nonbillable while retaining usage/model/provider visibility; unknown ownership remains pending. Owned customer tariffs and independent native supplier evidence remain separate

The consumer must fence accepted producer versions/activation. A legacy writer could retain forged JSON with this field name; shape validation alone is insufficient. Do not accept historical objects, auto-register current rows, or backfill current classifications. Positive acceptance requires this reviewed stamping/sanitization source (including encrypted persistence) and explicit trusted registrations

This source change does not update a Layer/Platform pin, publish an image, activate registrations, provision capacity, or deploy an instance. Consuming-image qualification belongs to the sole candidate owner. Fork hosted Actions are absent, so local evidence is not a hosted-CI green claim

## Validation

Focused tests cover spoofed and nested metadata, unregistered routes, invalid registrations, rotation, explicit administrator registration, DB route rejection, transport overrides, and sync/async Router plus real SDK calls against a local synthetic HTTP server. Fallback checks compare both attempted and completed deployment ownership. Actual spend-writer tests cover platform/BYOK/unknown through successful and failed provider calls, normal encryption, crypto failure, transform failure, durable SQLite persistence, and unchanged native usage/spend
