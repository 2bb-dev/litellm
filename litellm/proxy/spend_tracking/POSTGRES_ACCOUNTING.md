# PostgreSQL admission/accounting: first vertical

This opt-in implementation is an unfinished production-path vertical, not a
complete shared-limit implementation or a deployment/retirement certificate.
It defaults off. Disabled mode retains the existing admission and spend paths.
Do not enable it alongside an unmodified legacy process: importing live legacy
reservations and proving its accounting barrier are not implemented. An explicit
schema migration is not proof that a live legacy process is safe to replace.

## Qualified Shape

Set `general_settings.postgres_admission_accounting: true` only on a fresh,
isolated/new-protocol installation. Apply the additive
`20260912120000_shared_admission_accounting`,
`20260913090000_accounting_components`,
`20260913160000_accounting_native_user` and
`20260914090000_accounting_native_team` migrations explicitly first; serving
candidates require `disable_prisma_schema_update: true`. The protocol is
`team-v4` (generation version 4), with a new generation UUID. Startup refuses a
database containing older-version generations, including closed generations.
Mixed key-v2/user-v3/team-v4 execution AND management are unsupported; an older process's
`accepting=false` does not fence its management endpoints. The unpublished key-v2
artifacts, UUIDs, receipts and outboxes are not rewritten or retroactively assigned
user membership. This refusal is not historical adoption or a rolling upgrade proof. Startup checks the
protocol marker and required columns, and requires writer-only PostgreSQL reads
(no `DATABASE_URL_READ_REPLICA`) plus a distinct, single-owner
`SPEND_LOG_DURABLE_QUEUE_PATH` per process.

The current subset admits database virtual keys, including authenticated installer,
operator, bot, Code and external-client attribution with primary recurring caps,
through `POST /chat/completions`, `/v1/chat/completions`, `/responses` and
`/v1/responses`. Foreground Responses supports native events, encrypted reasoning
continuation and `max_output_tokens` with the same soft-budget semantics as Chat.
The Responses endpoint validates the inference envelope (input shape, model and
output-token limit) before execution. Nested tools/input items keep the native
transformation's optional/default and pass-through semantics, including omitted
function `strict` and image `detail`; SDK TypedDict `Required` annotations are not
an HTTP-default contract. Accounting checks execution overrides and unfinished
scopes, not a second full Responses parameter catalogue. Token-priced `openai/` and `litellm_proxy/` deployments
support the normal router/SDK retry policy, including retries=2, simple-shuffle or
usage-based routing, and DB-managed model creation, update, deletion and visibility.
Local response caching and the actual OpenOrange request-context callback remain
supported. The usage selector receives a shared committed-receipt usage snapshot
using PostgreSQL minute boundaries, not each process's independent cache. This
snapshot follows settlement, not receipt-free estimates of provider token use.
Usage-v1 requests use the existing async router filter/selector path, including
native encrypted-content affinity. The native `litellm_enc` format embeds the
originating deployment in reasoning content and preserves streaming `rs_*` IDs;
no process-local or PostgreSQL affinity map is introduced. Continuation restores
provider content only after routing to the native binding. Configured
rate/concurrency limits remain guarded pending their own adapters, including native
key creation fields and metadata, generation defaults, per-model budget policies,
deployment default API-key RPM/TPM and router default concurrency. Nonempty limit
maps containing zero are policies, not disabled values. Native null/empty key maps
and a router default concurrency of zero (no native semaphore) remain no-limit
settings. Configuration and effective runtime initialization both enforce these
guards; existing keys and prepared/updated key policy are rechecked before dispatch.
This managed-key slice is not full enabled-product or deployment acceptance.
Configuration/request guards remain for unqualified effective contracts; production
defaults are not weakened to make the supported slice pass.

Apart from foreground inference, the unfinished mode allows key generation, scoped
`/key/update` (cap, duration, models, managed metadata, blocked state, alias and
native-authorized user/Team reassignment), native `/user/new`, `/user/update`, `/user/info`,
`/team/new`, `/team/update` and `/team/info`,
model CRUD (`/model/new`, `/model/update`, `/model/{id}/update`, `/model/delete`),
and selected read-only health/model/key endpoints. It rejects other endpoints,
WebSockets, arbitrary live configuration mutation, external spend writers, disabled accounting,
Redis transaction buffering and background model health checks. Per-key scopes
are rechecked from PostgreSQL on every admission, rather than inferred solely
from cached authentication objects. Key policy/auth cache reads are bypassed in
this mode; supported key mutations and admission share the stable key-scope lock.
Locked admission reuses native Key and Team model authorization with fresh locked
rows, not cached Team permissions. Native sentinel, wildcard, literal and global
alias semantics remain subject to both Key and Team restrictions. Team-specific
model aliases and access-group policies remain unqualified and refused.

Authenticated `user_id` is persisted on the admitted request and remains the
raw/daily dimension even when no native user row exists. Personal requests retain
user scope membership even when uncapped or the optional native row is missing;
adding a cap cannot ignore outstanding admitted work. Only explicit native user
management creates a UserTable row, never metadata or settlement. Settlement uses
the captured user ID, not the key's later assignment. A later key reassignment
changes future requests only. Missing historical spend is not invented or imported.
Primary native user caps and recurring durations share admission/settlement locks.
Native user rate/window/model/organization policies remain guarded. Ordinary Team
membership is supported; Team requests omit personal User holds/cap checks while
still locking and booking the captured User.
OpenOrange
person caps live in the separate OpenOrange policy database, not UserTable. They
retain the existing read-time enforcement against shared `admin_spend_events`,
including positive-cap 429s, static 403s, cache TTLs and logged fail-open behavior.
This mode does not invent shared person reservations or change that policy. Cache
TTLs do not bound the underlying asynchronous projection lag; projection and
attribution parity still require integration evidence.

## Accounting Contract

- A server-only ContextVar carries fresh logical request and component identities.
  Caller/provider/trace IDs are not accounting authority. Original reported IDs
  remain in raw metadata as `reported_request_id`. A repeated HTTP request
  is another request, not inference deduplication. Native OpenAI SDK and
  AsyncHTTPHandler request/response hooks record each actual POST dispatch,
  including internal retries hidden below Router.make_call. Per-call context follows
  logging and stream iteration, so a wrapper total is not another execution or
  another charge. Native router/SDK retry policies are retained; accounting recovery
  never submits an inference POST.
- Generation closure serializes with durable request creation. Admission locks a
  stable key scope/native row, captured User, captured Team and optional membership,
  in that order. The native counter selector determines positive-cap checks using
  booked spend plus outstanding holds. Key/User zero caps deny; null is uncapped.
  Team retains its native finite `spend > cap` check and counter eligibility only
  for `cap > 0`: zero permits unspent Team work, but denies after positive spend.
  Positive-cap equality has no reservation headroom and is denied. Existing estimation and positive-headroom shrink are reused. Provider
  `max_tokens` is unchanged; this remains a soft budget, not a hard final-cost cap.
- The request stays unsealed while the HTTP/stream producer or a tracked logging
  descendant can produce accounting. Tracking starts before detached task creation
  or LoggingWorker enqueue and follows its existing context propagation and retry.
  Queue emptiness and LoggingWorker.flush are not completion certificates.
- The writer reuses raw and daily mapping and awaits the existing SQLite outbox.
  Failed local cost enqueue retains the immutable event through the existing
  memory fallback; that is not a durable completion acknowledgement. A seal
  whose manifest awaits receipts moves transactionally to the SQLite FIFO tail,
  preserving payload and original creation time. Even a one-row writer can then
  settle unrelated requests. This is not quarantine, refund or inference replay;
  the pending owner's seal and liabilities remain. Owner death before memory
  fallback recovery can still lose the cost payload and leaves an unresolved
  manifest, never a manufactured receipt or empty-queue completion certificate.
  The consumer commits raw rows, receipt identity/hash, key/native-user spend, daily
  totals and applicable hold disposition in one Prisma transaction. Native writer
  key/user and daily mapping leaves execute under that transaction, including key
  last_active; there is no private daily SQL or parallel key/user booking path.
  Authenticated key metadata is captured at admission in the raw namespace expected
  by the unchanged SQL projection, rather than accepting request-body authority. Legacy aggregate and
  spend-counter increments are skipped for protocol-owned events. Raw-log retention
  does not delete receipts.
- Monetary liability and retirement are separate. Actual settlement replaces only
  its component's liability with booked actual spend, even while producers remain
  open. For a single execution, committed 0.03 leaves 0.07 of a 0.1 budget available.
  Each membership holds the same component liability; memberships are not summed
  into an extra monetary charge. Replacement recomputes remaining component
  liabilities and rejects inconsistent membership totals instead of clamping a
  subtraction with GREATEST. Each additional permitted execution carries the original soft reservation amount;
  legitimate retries do not re-admit the logical request or clamp max_tokens. An
  earlier accepted/drop attempt retains its liability even when a later retry succeeds.
  This can exceed headroom, as can native soft-budget actual cost; it is not a hard cap.
  A status code alone is not nonexecution evidence: a generic post-admission 400
  or output-validation failure retains liability. Only recognized structured
  provider preexecution errors (for example rate_limit_exceeded or
  context_length_exceeded at the corresponding status) release that component,
  without inventing an actual-zero receipt. The HTTP adapter observes at most
  64 KiB of the error bytes the native client already consumes, before its native
  retry; it does not pre-read or buffer a success stream. Oversized, encoded,
  malformed and unrecognized bodies remain unknown. A lost rejection
  acknowledgement stays unresolved and never triggers inference replay. Other
  transport/status failures remain unknown unless native actual usage settles them.
  Missing cost, failed producer/enqueue, conflicting receipt or unknown execution
  stays pending. There is no refund timer, lease expiry or provider replay.
  Database errors never fall back to local spend for authoritative reads.
- Repeated identical component totals and replay of an immutable receipt apply
  once. A different payload under an existing receipt is an error. Differing actual
  total revisions are not yet integrated and remain pending, rather than silently
  adding another total. The supported revision ordering is estimate -> actual:
  cancellation after dispatch intent can retain the input estimate as an explicit
  admission liability (not fabricated actual raw spend); actual settlement
  supersedes it, and a late cancellation cannot overwrite actual usage. An
  estimate without actual usage cannot certify retirement. The supported
  mid-stream disconnect path retains the full hold when usage is unknown.
  Crashes are not cancellations.
- Aborted bodies, malformed and pre-auth requests with causally known nonexecution
  need no paid receipt: a durable nonexecution seal releases any admission hold.
  Existing zero-cost diagnostic logs remain allowed. The local `dispatched` flag
  is set before awaiting the persistent dispatch intent; neither flag proves
  provider execution. Lost/cancelled acknowledgements cannot become a claim of
  nonexecution. Producer failure diagnostics remain sticky even when no monetary
  liability remains.
- Producers enqueue one immutable seal into the same SQLite outbox after their
  count reaches zero. It includes sorted expected receipt IDs and payload hashes,
  failure state and nonexecution evidence. The consumer validates the component,
  deterministic seal identity and committed receipt manifest before atomically
  recording the seal and terminal state. Identical replay is a no-op; conflicting
  replay fails closed. PostgreSQL write failure or commit-ack loss leaves the
  outbox marker available to a fresh accounting consumer, without inference replay.
  Pre-start task cancellation and dropped logging work are failed producers, not
  successful completion. Empty outbox or zero local producer count alone is never
  proof of a durable seal.
- The authenticated drain endpoint closes this generation's admission and returns
  202/accounting_pending, 503/accounting_unknown, or 200/accounting_complete. This
  says only that the generation's recorded accounting/execution-producer obligations
  are complete; external transport/deployment continuity remains separate.

## Primary Key Epochs

Primary recurring key caps book at settlement time. Admission, settlement,
management changes and the existing reset job serialize on the stable key scope.
They sample PostgreSQL UTC time after locking and use the existing standardized
duration calculator. A stale resetter re-reads the current boundary instead of
applying a previously computed unconditional zero. Holds retain their original
normalized epoch and count across all epochs; reset never clears them. Existing
scope rows survive. Setting or clearing a duration does not zero spend; missing
initial boundaries are initialized without erasing booked spend. This is distinct
from the still-unimplemented request-time window contract.

## Native User Locking And Reset

Request-wide operations lock the request first, then key scope/native row, then
captured user scope/native row. Admission, each physical dispatch, known rejection,
actual/estimate settlement and nonexecution seal release use that order. User
management and reset only lock the user scope/native row; they do not acquire a
key/request in reverse order. Native user upsert and native spend writer leaves
remain in use, under those locks. Endpoint authentication, self-update restrictions,
finite-value validation and password redaction remain native; recurrence mutation
also requires admin authority for self-updates. Bulk/delete/organization
management stays unavailable in this unfinished mode.

Fresh user auth bypasses local cached spend/roles/models and retains native
organization membership context. Admission rechecks native model permissions from
the locked user. The later max-budget hook honors only a server-owned PG admission,
so it cannot reject the request against its own already-filled reservation.
User info reports native booked spend, not a fabricated spend including holds.

Recurring user budgets book at settlement time. The existing primary reset job,
read/admission, user changes and settlement sample PostgreSQL UTC time after locking.
Two resetters re-read the current boundary rather than applying stale zero updates.
Setting/clearing duration preserves booked spend; missing initial boundaries are
initialized without zeroing it. Outstanding user holds remain on their stable
scope through all resets; no history is deleted. Raw/daily/key/user/receipt/hold
settlement cannot partially commit while the native user row is locked, and producer
retirement still requires the independent seal. None of this implements request-time
windows, hard final-cost caps, or historical bootstrap.

## Native Primary Team

Nullable request.team_id is captured with user_id, never reconstructed from a
mutable key. Team requests hold Key/Team even when uncapped, and never add a
personal User hold. Settlement reuses native Team/member increment leaves and
DailyTeam grouping inside the same raw/receipt/Key/User/DailyUser transaction.
Existing membership spend and total_spend increment together; absent optional
membership/User rows are not created by settlement. Parallel projections are not
additional financial charges. A missing Team fails closed.

Native Team creation preserves creator membership and User.teams updates, with
sorted User locks before Team insertion. Key creation/reassignment and Team update
recheck native actor, source/destination Team authority and cap permissions under
locks. Team admins can lower caps; only proxy admins can increase/remove them.
Member budgets, defaults, rates, organizations, linked BudgetTable policy and
sibling member/delete/block/model/permission routes remain explicitly guarded.
BudgetTable is configuration, not a pooled spend balance. The direct member-route
guard does not remove the inherited `/user/new` transitive membership helper;
concurrent membership mutation through that helper is not newly qualified here.

Key insertion shares the native preparation and idempotent upsert leaves with
ordinary mode, using the already-locked transaction without a nested transaction.
A duplicate custom token locks its native row before Users/Teams, checks fresh
actor and original/destination authority, and retains the original row (including
spend, owner and metadata) via `update={}`. It does not reset that duplicate's
spend or reassign ownership. Policy validation decodes prepared JSON using the
native Prisma field types, not a separate field catalogue; arrays and nullable
write semantics remain native. Native management response shapes are unchanged.

Team recurrence uses PostgreSQL time after locks and fresh due-state rechecks in
all native reset entry points. It resets only Team spend, preserving holds and
cumulative User/member/raw/daily totals. Duration changes preserve spend; late
actual books in the current primary period. Stale reset updates cannot erase it.
This UTC slice refuses non-UTC budget timezones in resolved configuration and
again from the native effective timezone at initialization, before generation
registration. Absent/default and explicit UTC remain supported. This does not
qualify arbitrary timezone policies or rolling migration.
The additive Team column uses explicit SQL and does not require regenerating the
frozen native client. This implementation is not a full Team/deployment certificate;
consult exact-source runtime evidence for qualified scenarios and remaining gaps.
SQL-seeded due-state tests alone do not qualify actual scheduled Team recurrence.

## Remaining Integration

Remaining native parent budgets, entity mappings, request-time windows,
model/router/provider/tag budgets, TPM/RPM/concurrency, retry/fallback composition
outside foreground Chat/Responses, BYOK credential management, other API families,
WS/realtime, batch polling and arbitrary actual-total corrections still need
integration and qualification. Request-time windows and other native entity/daily
scope wiring are not waived by the key/native-user mapping leaves. Human and managed
key attribution are exercised through actual callback, raw/daily rows and the native
SQL projection transaction. Separate projector scheduling/retirement and the broader
identity families remain outside that proof. A prior attempt with an
unknown provider outcome cannot be given a fabricated zero receipt, and one
successful retry must not clear its liability. These are remaining contracts,
not a reason to disable the normal production retry or routing configuration.

Set installer-issued `LITELLM_ACCOUNTING_GENERATION_ID` to the immutable generation
UUID. Startup registers it idempotently and checks protocol compatibility without
reopening closed admission. Each runtime gets a separate `incarnation_id`. Without
the environment variable, the random generation is explicitly unbound and cannot
be used as deployment proof. All old/new incarnation requests remain visible under
the same durable generation, including unknown outcomes after actual process death.

`GET /health/accounting` reads that state without mutation. `GET /health/drain`
closes admission; `POST /health/resume` explicitly reopens the retained identity.
All three use `enable_drain_endpoint` and the existing `X-Drain-Token` mechanism.
Configure a token for private deployment control. Responses include
`protocol_version` (`team-v4`), `generation_id`, `incarnation_id`, `generation_bound`,
`accepting`, `pending_requests` and `status`. Status reads and resume return 200 on
success; drain returns 202 while accounting is pending, otherwise 200. Unknown DB
state returns 503 with null accepting/pending values, never invented zero. Managed
drain changes readiness/admission, not liveness or SIGTERM. Resume cannot undo an
actual process shutdown. A complete status while accepting is not a stop certificate.
The installer must still hold its operation lock and verify retained assets, exact
container/config identity and transport/external-owner completion; retired UUIDs
must never be reused.

This groups obligations, not exclusive execution takeover: a fresh incarnation's
zero local producers cannot close old work. Preserve the original generation and
outbox. Loss/failure of the outbox itself, or owner death before a terminal marker
is durably enqueued, cannot be repaired from queue emptiness. The component migration adds explicit per-component liability, dispatch and
nonexecution fields. The user migration adds only nullable request.user_id and a
new protocol marker. Those accounting columns use explicit SQL; this slice does
not regenerate or mutate the frozen native Prisma client. The schema copies include
the additive field for later packaging/client qualification by the integration
owner. Old frozen clients/environments must not be regenerated in place. This is
not a live migration/adoption path.

## Tests

`tests/test_litellm/proxy/spend_tracking/test_postgres_accounting.py` contains the
configuration checks and real-PostgreSQL transaction/producer negatives. Its DB
cases skip unless `LITELLM_ACCOUNTING_TEST_DATABASE_URL` explicitly identifies a
fresh isolated database with the base schema and this migration applied. They
create isolated keys/generations and require a disposable database; never point
them at shared/live data. Tests use the repository's pinned pytest tooling.

The `OpenOrange Fork CI` regression job provisions an isolated digest-pinned
PostgreSQL service, generates the native client in the job's own environment,
pushes the native schema without the additive accounting models, then applies
all four accounting migrations explicitly. It runs the entire accounting test
file and fails if any accounting test skips. The internal HTTP retry cases use
an in-process loopback server that accepts/drops the first POST; they require no
provider key or external inference. Existing fork regressions, OSS import/license
guards and image/wheel builds remain enabled. This tests the fresh migration and
accounting path, not historical adoption or packaged parent compatibility.

The implementation task's scratch evidence additionally exercises separate real
CLI proxies, HTTP/SSE, a controlled local provider, blocked entity UPDATE,
caller-ID forgery/repetition, cache hits and PostgreSQL failure. The task result
records the exact source manifests and outcomes; passing this vertical does not
close whole-deployment acceptance or authorize publication/rollout.
