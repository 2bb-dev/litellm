import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.errors import TransactionError

from litellm.caching import DualCache
from litellm.litellm_core_utils.accounting_context import accounting_request, spawn_accounting
from litellm.litellm_core_utils.logging_worker import LoggingWorker
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.spend_tracking.postgres_accounting import OUTBOX_EVENT, PostgresAccounting, ProducerSeal
from litellm.proxy.spend_tracking.postgres_accounting_config import validate_config, validate_request
from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest_asyncio.fixture(loop_scope="function")
async def protocol(tmp_path, monkeypatch):
    url = os.environ.get("LITELLM_ACCOUNTING_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires an explicitly isolated migrated PostgreSQL database")
    url += ("&" if "?" in url else "?") + "connection_limit=2"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "spool.sqlite"))
    client = PrismaClient(database_url=url, proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()))
    context_token = None
    try:
        await client.connect()
        instance = PostgresAccounting(client)
        await instance.initialize()
        key = str(uuid4())
        await client.db.execute_raw(
            'INSERT INTO "LiteLLM_VerificationToken" (token, models, max_budget) VALUES ($1, ARRAY[]::text[], 0.1)', key
        )
        request = await instance.begin()
        context_token = accounting_request.set(request)
        await instance.admit(key, 1.0, 0.01)
        await instance.dispatch()
        yield instance, request, key
    finally:
        if context_token is not None:
            accounting_request.reset(context_token)
        await client.db.disconnect(timeout=timedelta(seconds=2))


async def observation(instance, request, cost=0.03):
    now = datetime.now(timezone.utc)
    raw = {
        "request_id": "caller-or-provider-id",
        "call_type": "acompletion",
        "api_key": request.token,
        "spend": cost,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "startTime": now.isoformat(),
        "endTime": now.isoformat(),
        "completionStartTime": now,
        "user": "",
        "model": "test-model",
        "model_group": "test-group",
        "custom_llm_provider": "openai",
        "metadata": json.dumps({"status": "success"}),
        "cache_hit": "False",
    }
    daily = await DBSpendUpdateWriter()._common_add_spend_log_transaction_to_daily_transaction(raw, instance.client)
    return raw, daily


async def envelopes(instance):
    batch = await instance.client._spend_log_spool.peek_batch(100, 1000000)
    return batch, [OUTBOX_EVENT.validate_python(row["postgres_accounting"]) for row in batch.logs]


async def apply_all(instance):
    batch, pending = await envelopes(instance)
    for envelope in pending:
        await instance.apply(envelope)
    await instance.client._spend_log_spool.acknowledge(batch.durable_row_ids)
    return pending


async def totals(instance, key):
    return (
        await instance.client.db.query_raw(
            """SELECT
        (SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=$1) AS entity,
        (SELECT COALESCE(SUM(spend),0) FROM "LiteLLM_SpendLogs" WHERE api_key=$1) AS raw,
        (SELECT COALESCE(SUM(spend),0) FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1) AS daily,
        (SELECT COALESCE(SUM(api_requests),0)::int FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1) AS records,
        (SELECT COALESCE(SUM(h.amount),0) FROM "LiteLLM_AccountingHold" h JOIN "LiteLLM_AccountingScope" s ON s.id=h.scope_id WHERE s.entity_id=$1) AS held""",
            key,
        )
    )[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [1, 100])
async def test_missing_receipt_seal_does_not_stall_real_writer(protocol, monkeypatch, batch_size):
    import sqlite3

    from litellm.proxy.db.spend_log_queue import SQLiteSpendLogSpool
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.proxy.utils import update_spend_logs_job

    instance, request, key = protocol
    captured = []

    class OneShotIOFailure(SQLiteSpendLogSpool):
        def _enqueue_many_sync(self, rows):
            if not captured:
                captured.extend(rows)
                raise sqlite3.OperationalError("bounded one-shot disk I/O error")
            return super()._enqueue_many_sync(rows)

    spool = OneShotIOFailure(str(instance.client._spend_log_spool.path))
    instance.client._spend_log_spool = spool
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    raw, daily = await observation(instance, request)
    try:
        await instance.enqueue(raw, daily, 0.03)
    except sqlite3.OperationalError:
        pass
    await request.finish()
    # Reconstruct the durable consumer before the missing observation is recovered.
    retained_memory = instance.client.spend_log_transactions
    assert retained_memory == [json.loads(captured[0][0])]
    instance.client.spend_log_transactions = []
    healthy_key = str(uuid4())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token, models, max_budget) VALUES ($1, ARRAY[]::text[], 0.1)',
        healthy_key,
    )
    healthy = await instance.begin()
    token = accounting_request.set(healthy)
    try:
        await instance.admit(healthy_key, 0, 0)
        await instance.dispatch()
        raw, daily = await observation(instance, healthy, 0)
        await instance.enqueue(raw, daily, 0)
        await healthy.finish()
    finally:
        accounting_request.reset(token)
    errors = []
    for _ in range(6):
        instance.client._spend_log_batch_max_count = batch_size
        try:
            await update_spend_logs_job(instance.client, None, ProxyLogging(user_api_key_cache=DualCache()))
        except RuntimeError as error:
            errors.append(str(error))
    rows = await instance.client.db.query_raw(
        'SELECT id, closed FROM "LiteLLM_AccountingRequest" WHERE id IN ($1,$2)',
        request.request_id,
        healthy.request_id,
    )
    assert {r["id"]: r["closed"] for r in rows} == {request.request_id: False, healthy.request_id: True}, errors
    assert (await totals(instance, key))["held"] == 0.1
    assert await instance.drain() == 1
    batch = await spool.peek_batch(100, 1000000)
    assert len(batch.logs) == 1
    seal_payload = batch.logs[0]
    assert seal_payload["postgres_accounting"]["expected"] == [list(item) for item in sorted(request.expected.items())]
    assert (await spool.stats()).quarantined_count == 0
    # Accounting replay only: original bytes/receipt, including a duplicate after recovery.
    original_payload = json.loads(captured[0][0])
    instance.client.spend_log_transactions = retained_memory
    await spool.enqueue(original_payload)
    for _ in range(6):
        instance.client._spend_log_batch_max_count = 1
        await update_spend_logs_job(instance.client, None, ProxyLogging(user_api_key_cache=DualCache()))
    await spool.enqueue(original_payload)
    await update_spend_logs_job(instance.client, None, ProxyLogging(user_api_key_cache=DualCache()))
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}
    assert await instance.drain() == 0
    assert (await spool.stats()).count == 0


@pytest.mark.asyncio
async def test_usage_selector_uses_shared_receipt_clock_not_local_cache(protocol, monkeypatch):
    from litellm.router_strategy import lowest_tpm_rpm

    instance, request, _ = protocol
    raw, daily = await observation(instance, request)
    group = str(uuid4())
    raw["model_group"] = group
    raw["model_id"] = "used-on-other-proxy"
    daily = await DBSpendUpdateWriter()._common_add_spend_log_transaction_to_daily_transaction(raw, instance.client)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)

    class SkewedClock(datetime):
        @classmethod
        def now(cls):
            return datetime.now() + timedelta(days=73000, minutes=17)

    monkeypatch.setattr(lowest_tpm_rpm, "datetime", SkewedClock)
    cache = DualCache()
    minute = SkewedClock.now().strftime("%H-%M")
    cache.set_cache(key=f"{group}:tpm:{minute}", value={"unused-on-db": 999})
    selector = lowest_tpm_rpm.LowestTPMLoggingHandler(router_cache=cache)
    deployments = [
        {"model_info": {"id": name}, "litellm_params": {}} for name in ("used-on-other-proxy", "unused-on-db")
    ]
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_AccountingReceipt" SET created_at=clock_timestamp() WHERE component_id=$1',
        request.component_id,
    )
    snapshot = await instance.routing_usage(group)
    assert snapshot == ({"used-on-other-proxy": raw["total_tokens"]}, {"used-on-other-proxy": 1})
    assert selector.get_available_deployments(group, deployments)["model_info"]["id"] == "used-on-other-proxy"
    assert (
        selector.get_available_deployments(group, deployments, usage_snapshot=snapshot)["model_info"]["id"]
        == "unused-on-db"
    )
    import litellm
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.responses.utils import ResponsesAPIRequestUtils

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    router = litellm.Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-local-only-" + name},
                "model_info": {"id": name},
            }
            for name in ("used-on-other-proxy", "unused-on-db")
        ],
        routing_strategy="usage-based-routing",
        optional_pre_call_checks=["encrypted_content_affinity"],
    )
    router.cache.set_cache(key=f"{group}:tpm:{minute}", value={"unused-on-db": 999})
    kwargs = {"litellm_metadata": {}, "input": "first"}
    selected = await router.async_get_available_deployment(model=group, request_kwargs=kwargs)
    assert selected["model_info"]["id"] == "unused-on-db"
    assert kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"] is True
    pinned = {
        "litellm_metadata": {},
        "input": [
            {
                "type": "reasoning",
                "id": "rs_fixture",
                "encrypted_content": ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                    "opaque", "used-on-other-proxy"
                ),
            }
        ],
    }
    selected = await router.async_get_available_deployment(model=group, request_kwargs=pinned)
    assert selected["model_info"]["id"] == "used-on-other-proxy"
    assert pinned["_encrypted_content_affinity_pinned"] is True
    await instance.client.db.execute_raw(
        "UPDATE \"LiteLLM_AccountingReceipt\" SET created_at=clock_timestamp()-interval '2 minutes' WHERE component_id=$1",
        request.component_id,
    )
    assert await instance.routing_usage(group) == ({}, {})


@pytest.fixture
def accepted_drop_provider():
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    admissions = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            admissions.append(json.loads(body))
            if len(admissions) == 1:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", admissions
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_transport", [False, True])
async def test_async_http_handler_internal_retry_keeps_each_dispatch(
    protocol, monkeypatch, httpx_transport, accepted_drop_provider
):
    from litellm.litellm_core_utils.accounting_context import AccountingCall, accounting_call
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy.spend_tracking import postgres_accounting

    url, admissions = accepted_drop_provider
    instance, _, _ = protocol
    key = str(uuid4())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token, models, max_budget) VALUES ($1, ARRAY[]::text[], 0.1)', key
    )
    request = await instance.begin()
    request_token = accounting_request.set(request)
    call_token = accounting_call.set(AccountingCall())
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", str(httpx_transport))
    handler = AsyncHTTPHandler(timeout=5)
    try:
        await instance.admit(key, 1.0, 0.01)
        marker = "handler-accepted-drop-httpx" if httpx_transport else "handler-accepted-drop-aiohttp"
        call = handler.post(
            url,
            json={
                "model": "anthropic/claude-sonnet-5",
                "messages": [{"role": "user", "content": marker}],
                "max_tokens": 1000,
            },
            headers={"Authorization": "Bearer sk-probe-provider-local-only"},
        )
        if httpx_transport:
            response = await call
            assert response.status_code == 200
        else:
            from httpx import ReadError

            with pytest.raises(ReadError):
                await call
        rows = await instance.client.db.query_raw(
            'SELECT * FROM "LiteLLM_AccountingComponent" WHERE request_id=$1', request.request_id
        )
        assert len(rows) == len(admissions) == (2 if httpx_transport else 1)
        assert all(body == admissions[0] for body in admissions)
        assert all(r["dispatched"] and not r["actual"] for r in rows)
        assert (await totals(instance, key))["held"] == (0.2 if httpx_transport else 0.1)
        await request.finish()
        await apply_all(instance)
        assert (await totals(instance, key))["raw"] == 0
        assert (await instance.status()).pending_requests >= 1
    finally:
        await handler.close()
        accounting_call.reset(call_token)
        accounting_request.reset(request_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [False, True])
async def test_component_actual_does_not_release_prior_unknown_attempt(protocol, rejected):
    instance, request, key = protocol
    first = request.component_id
    if rejected:
        await instance.reject_component(first)
        await instance.reject_component(first)
    await instance.dispatch()
    second = instance.component_id()
    assert second != first
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)
    state = await totals(instance, key)
    assert state == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0 if rejected else 0.1}
    components = await instance.client.db.query_raw(
        'SELECT id, actual, nonexecution, liability FROM "LiteLLM_AccountingComponent" WHERE request_id=$1',
        request.request_id,
    )
    assert len(components) == 2
    assert {c["id"]: c["actual"] for c in components} == {first: False, second: True}
    assert await instance.drain() == (0 if rejected else 1)


@pytest.mark.asyncio
async def test_native_writer_mapping_updates_key_activity_in_receipt_transaction(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    before = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": key})
    assert before.last_active is None
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)
    after = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": key})
    assert after.last_active is not None
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}


@pytest.mark.asyncio
async def test_generation_registration_preserves_closed_gate_and_old_receipts(protocol):
    from fastapi import HTTPException

    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    _, original = await envelopes(instance)
    assert await instance.drain() == 1
    restarted = PostgresAccounting(instance.client, generation_id=instance.generation_id)
    await restarted.initialize()
    await restarted.initialize()
    state = await restarted.status()
    assert state.generation_bound and state.generation_id == instance.generation_id
    assert state.incarnation_id != instance.incarnation_id
    assert state.accepting is False and state.pending_requests == 1
    with pytest.raises(HTTPException) as error:
        await restarted.begin()
    assert error.value.status_code == 503
    for event in original:
        await restarted.apply(event)
        await restarted.apply(event)
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}
    state = await restarted.status()
    assert state.pending_requests == 0 and state.accepting is False
    other = PostgresAccounting(instance.client)
    await other.initialize()
    assert (await other.status()).accepting is True
    assert (await restarted.set_accepting(True)).accepting is True
    assert (await instance.status()).accepting is True
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_AccountingGeneration" SET version=2 WHERE id=$1', instance.generation_id
    )
    try:
        with pytest.raises(RuntimeError, match="Mixed accounting generations"):
            await restarted.initialize()
        with pytest.raises(RuntimeError, match="incompatible"):
            await restarted.set_accepting(False)
    finally:
        await instance.client.db.execute_raw(
            'UPDATE "LiteLLM_AccountingGeneration" SET version=4 WHERE id=$1', instance.generation_id
        )
    with pytest.raises(ValueError):
        PostgresAccounting(instance.client, generation_id="not-a-uuid")


@pytest.mark.asyncio
async def test_commit_ack_loss_replays_only_accounting_from_retained_outbox(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    batch, pending = await envelopes(instance)

    @asynccontextmanager
    async def lose_commit_ack():
        async with instance.client.db.tx() as tx:
            yield tx
        raise TimeoutError("commit acknowledgement lost after durable PostgreSQL commit")

    consumer = PostgresAccounting(instance.client, settlement_transaction=lose_commit_ack)
    with pytest.raises(TimeoutError):
        await consumer.apply(pending[0])
    assert (await instance.client._spend_log_spool.stats()).count == 2
    restarted_consumer = PostgresAccounting(instance.client)
    await apply_all(restarted_consumer)
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}
    assert await instance.drain() == 0
    assert (await instance.client._spend_log_spool.stats()).count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("actual_first", [False, True])
async def test_actual_replaces_estimate_and_late_cancel_cannot_overwrite_actual(protocol, actual_first):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    if actual_first:
        await instance.enqueue(raw, daily, 0.03)
        await apply_all(instance)
        await instance.cancel()
    else:
        await instance.cancel()
        await apply_all(instance)
        assert await totals(instance, key) == {"entity": 0.0, "raw": 0.0, "daily": 0.0, "records": 0, "held": 0.01}
        await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}
    assert await instance.drain() == 0


@pytest.mark.asyncio
async def test_repeated_total_callback_is_not_an_increment(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    later = {**raw, "request_id": "another-provider-id", "endTime": datetime.now(timezone.utc).isoformat()}
    await instance.enqueue(later, daily, 0.03)
    assert (await instance.client._spend_log_spool.stats()).count == 1
    await request.finish()
    receipts = await apply_all(instance)
    await instance.apply(receipts[0])
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}


@pytest.mark.asyncio
async def test_forged_receipt_and_conflicting_replay_stay_pending(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, pending = await envelopes(instance)
    with pytest.raises(ValueError, match="Unknown accounting"):
        await instance.apply(pending[0].model_copy(update={"receipt_id": str(uuid4())}))
    await instance.apply(pending[0])
    with pytest.raises(ValueError, match="payload conflict"):
        await instance.apply(pending[0].model_copy(update={"total": 0.08}))
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == 1
    assert (await totals(instance, key))["entity"] == 0.03
    assert (await totals(instance, key))["held"] == 0.0


@pytest.mark.asyncio
async def test_actual_releases_money_for_next_admission_but_not_retirement(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await apply_all(instance)
    assert await instance.current_spend(f"spend:key:{key}") == pytest.approx(0.03)
    assert (await totals(instance, key))["held"] == 0
    next_request = await instance.begin()
    token = accounting_request.set(next_request)
    try:
        await instance.admit(key, 1.0, 0.01)
    finally:
        accounting_request.reset(token)
    assert (await totals(instance, key))["held"] == pytest.approx(0.07)
    assert await instance.current_spend(f"spend:key:{key}") == pytest.approx(0.1)
    assert await instance.drain() == 2
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == 1


@pytest.mark.asyncio
async def test_pre_enqueue_helper_and_queued_descendant_prevent_sealing(protocol):
    instance, request, key = protocol
    gate = asyncio.Event()
    callback_gate = asyncio.Event()
    worker = LoggingWorker(timeout=2, concurrency=1)
    raw, daily = await observation(instance, request)

    async def callback():
        await callback_gate.wait()
        await instance.enqueue(raw, daily, 0.03)

    async def helper():
        await gate.wait()
        worker.ensure_initialized_and_enqueue(callback())

    task = spawn_accounting(helper())
    try:
        await request.finish()
        assert (await instance.client._spend_log_spool.stats()).count == 0
        assert await instance.drain() == 1
        gate.set()
        await task
        assert await instance.drain() == 1
        callback_gate.set()
        await worker.flush()
        assert await instance.drain() == 1
        await apply_all(instance)
        assert await instance.drain() == 0
    finally:
        gate.set()
        callback_gate.set()
        await task
        await worker.stop()


@pytest.mark.asyncio
async def test_queue_full_retry_retains_producer_before_callback_starts(protocol):
    instance, request, key = protocol
    worker = LoggingWorker(timeout=3, max_queue_size=2, concurrency=1)
    blocker_started = asyncio.Event()
    blocker_release = asyncio.Event()
    aggressive_started = asyncio.Event()
    aggressive_release = asyncio.Event()
    final_started = asyncio.Event()
    final_release = asyncio.Event()
    raw, daily = await observation(instance, request)

    async def blocker():
        blocker_started.set()
        await blocker_release.wait()

    async def aggressive():
        aggressive_started.set()
        await aggressive_release.wait()

    async def final():
        final_started.set()
        await final_release.wait()
        await instance.enqueue(raw, daily, 0.03)

    worker.ensure_initialized_and_enqueue(blocker())
    await blocker_started.wait()
    worker.enqueue(aggressive_release.wait())
    worker.enqueue(aggressive_release.wait())
    worker.enqueue(aggressive())
    await aggressive_started.wait()
    worker.enqueue(aggressive_release.wait())
    worker.enqueue(final())
    try:
        await request.finish()
        assert not final_started.is_set()
        assert await instance.drain() == 1
        blocker_release.set()
        aggressive_release.set()
        await worker.flush()
        assert await instance.drain() == 1
        await asyncio.wait_for(final_started.wait(), 2)
        assert await instance.drain() == 1
        final_release.set()
        await worker.flush()
        await apply_all(instance)
        assert await instance.drain() == 0
    finally:
        blocker_release.set()
        aggressive_release.set()
        final_release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_callback_timeout_preserves_hold_and_unresolved_request(protocol):
    instance, request, key = protocol
    worker = LoggingWorker(timeout=0.02)
    gate = asyncio.Event()
    worker.ensure_initialized_and_enqueue(gate.wait())
    await request.finish()
    await worker.flush()
    await worker.stop()
    await apply_all(instance)
    assert await instance.drain() == 1
    assert (await totals(instance, key))["held"] == 0.1


@pytest.mark.asyncio
async def test_transaction_timeout_cannot_commit_partial_settlement(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    _, pending = await envelopes(instance)

    @asynccontextmanager
    async def expired_transaction():
        async with instance.client.db.tx(timeout=timedelta(milliseconds=30)) as tx:
            await tx.query_raw("SELECT 1 AS ready FROM pg_sleep(0.1)")
            yield tx

    consumer = PostgresAccounting(instance.client, settlement_transaction=expired_transaction)
    with pytest.raises(TransactionError):
        await consumer.apply(pending[0])
    assert await totals(instance, key) == {"entity": 0.0, "raw": 0.0, "daily": 0.0, "records": 0, "held": 0.1}
    assert await instance.drain() == 1
    await apply_all(instance)
    assert await instance.drain() == 0


@pytest.mark.asyncio
async def test_missing_cost_never_refunds_or_closes(protocol):
    instance, request, key = protocol
    with pytest.raises(ValueError, match="finite nonnegative"):
        await instance.enqueue({}, {}, None)
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == 1
    assert (await totals(instance, key))["held"] == 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_user", [False, True])
async def test_attributed_key_preserves_native_user_without_creating_one(protocol, existing_user):
    instance, original, key = protocol
    user_id = str(uuid4())
    if existing_user:
        await instance.client.db.execute_raw(
            'INSERT INTO "LiteLLM_UserTable" (user_id, models, teams, spend) VALUES ($1, ARRAY[]::text[], ARRAY[]::text[], 0.02)',
            user_id,
        )
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_VerificationToken" SET user_id=$2 WHERE token=$1', key, user_id
    )
    raw, daily = await observation(instance, original, 0)
    await instance.enqueue(raw, daily, 0)
    await original.finish()
    await apply_all(instance)
    request = await instance.begin()
    token = accounting_request.set(request)
    try:
        await instance.admit(key, 1, 0.01)
        await instance.dispatch()
        raw, daily = await observation(instance, request)
        raw["user"] = "forged"
        daily["user_id"] = "forged"
        await instance.enqueue(raw, daily, 0.03)
        await request.finish()
        await apply_all(instance)
    finally:
        accounting_request.reset(token)
    assert await instance.client.db.query_raw(
        'SELECT "user" FROM "LiteLLM_SpendLogs" WHERE request_id=$1', request.component_id
    ) == [{"user": user_id}]
    assert await instance.client.db.query_raw(
        'SELECT user_id, spend FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1 AND user_id=$2', key, user_id
    ) == [{"user_id": user_id, "spend": 0.03}]
    users = await instance.client.db.query_raw('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=$1', user_id)
    assert users == ([{"spend": 0.05}] if existing_user else [])


@pytest.mark.asyncio
async def test_primary_reset_keeps_liability_and_stale_resetters_cannot_erase_settlement(protocol, monkeypatch):
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, held, key = protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    await instance.update_key(key, {"budget_duration": "30d"})
    await instance.client.db.execute_raw(
        "UPDATE \"LiteLLM_VerificationToken\" SET spend=0.04, budget_reset_at=clock_timestamp()-interval '1 second' WHERE token=$1",
        key,
    )
    stale = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": key})
    stale.budget_reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    job = ResetBudgetJob(ProxyLogging(user_api_key_cache=DualCache()), instance.client)
    await asyncio.gather(job.reset_budget(), job.reset_budget())
    assert (await totals(instance, key))["entity"] == 0
    assert (await totals(instance, key))["held"] == 0.1
    competing = await instance.begin()
    token = accounting_request.set(competing)
    try:
        import litellm

        with pytest.raises(litellm.BudgetExceededError):
            await instance.admit(key, 1, 0.01)
        await competing.finish()
    finally:
        accounting_request.reset(token)
    raw, daily = await observation(instance, held)
    await instance.enqueue(raw, daily, 0.03)
    await held.finish()
    await apply_all(instance)
    await asyncio.gather(job._write_key_reset_updates([stale]), job._write_key_reset_updates([stale]))
    assert (await totals(instance, key))["entity"] == 0.03
    assert (await totals(instance, key))["held"] == 0
    rows = await instance.client.db.query_raw(
        "SELECT budget_reset_at > clock_timestamp() AS future, budget_reset_at < clock_timestamp()+interval '32 days' AS bounded FROM \"LiteLLM_VerificationToken\" WHERE token=$1",
        key,
    )
    assert rows == [{"future": True, "bounded": True}]
    for _ in range(2):
        next_request = await instance.begin()
        token = accounting_request.set(next_request)
        try:
            await instance.admit(key, 0, 0)
            await next_request.finish()
        finally:
            accounting_request.reset(token)
    await apply_all(instance)
    assert await instance.client.db.query_raw(
        "SELECT COUNT(*)::int AS count FROM \"LiteLLM_AccountingScope\" WHERE entity_id=$1 AND epoch<>'initial'", key
    ) == [{"count": 1}]
    await instance.update_key(key, {"budget_duration": None})
    assert (await totals(instance, key))["entity"] == 0.03
    assert await instance.drain() == 0


@pytest.mark.asyncio
async def test_key_update_serializes_with_actual_and_merges_trusted_metadata(protocol):
    instance, request, key = protocol
    metadata = {
        "managed_by": "openorange",
        "openorange_key_kind": "external_client",
        "openorange_registry_id": "registry",
        "openorange_prompt_caching_enabled": True,
    }
    await instance.update_key(key, {"metadata": metadata})
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, pending = await envelopes(instance)
    async with instance.client.db.tx() as gate:
        await gate.query_raw('SELECT id FROM "LiteLLM_AccountingScope" WHERE id=$1 FOR UPDATE', f"key:{key}:initial")
        update = asyncio.create_task(
            instance.update_key(key, {"max_budget": 0.2, "metadata": {"openorange_prompt_caching_enabled": False}})
        )
        settlement = asyncio.create_task(instance.apply(pending[0]))
        await asyncio.sleep(0.05)
        assert not update.done() and not settlement.done()
    await asyncio.gather(update, settlement)
    row = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": key})
    assert row.spend == 0.03 and row.max_budget == 0.2
    assert row.metadata == {**metadata, "openorange_prompt_caching_enabled": False}
    assert (await totals(instance, key))["held"] == 0


@pytest.mark.asyncio
async def test_zero_key_cap_does_not_mean_unlimited(protocol):
    import litellm

    instance, original, key = protocol
    await instance.update_key(key, {"max_budget": 0})
    request = await instance.begin()
    token = accounting_request.set(request)
    try:
        with pytest.raises(litellm.BudgetExceededError):
            await instance.admit(key, 0, 0)
        await request.finish()
    finally:
        accounting_request.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, None])
async def test_native_user_concurrency_and_model_caps_remain_guarded(protocol, limit):
    from fastapi import HTTPException

    instance, original, key = protocol
    user = str(uuid4())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_UserTable" (user_id, models, max_parallel_requests, model_max_budget) VALUES ($1, ARRAY[]::text[], $2, $3::jsonb)',
        user,
        limit,
        json.dumps({"model": 1} if limit is None else {}),
    )
    await instance.client.db.execute_raw('UPDATE "LiteLLM_VerificationToken" SET user_id=$2 WHERE token=$1', key, user)
    request = await instance.begin()
    token = accounting_request.set(request)
    try:
        with pytest.raises(HTTPException, match="native user rate/window/organization limits pending") as failure:
            await instance.admit(key, 0, 0)
        assert failure.value.status_code == 503
        assert not request.admitted
        await request.finish()
    finally:
        accounting_request.reset(token)


def configuration():
    return {
        "general_settings": {"postgres_admission_accounting": True},
        "router_settings": {"num_retries": 0},
        "litellm_settings": {"num_retries": 0},
        "model_list": [{"model_name": "model", "litellm_params": {"model": "openai/model"}, "model_info": {}}],
    }


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("router_settings", "routing_strategy", "usage-based-routing-v2"),
        ("general_settings", "max_parallel_requests", 1),
        ("litellm_settings", "callbacks", ["custom"]),
        ("router_settings", "routing_strategy", "cost-based-routing"),
    ],
)
def test_unfinished_enabled_combinations_fail_closed(section, field, value):
    config = configuration()
    validate_config(config)
    config[section][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rpm_limit", 1),
        ("rpm_limit", 0),
        ("model_max_budget", {"model": 1}),
        ("budget_duration", "0d"),
        ("budget_limits", [{"max_budget": 1}]),
        ("config", {"num_retries": 1}),
    ],
)
def test_key_scopes_remain_explicitly_pending(field, value):
    with pytest.raises(Exception):
        PostgresAccounting.validate_key({field: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {"accounting_request_id": "forged"}, '{"accounting_request_id":"forged"}'])
async def test_outbox_keeps_reported_id_without_trusting_it(tmp_path, monkeypatch, metadata):
    from litellm.litellm_core_utils.accounting_context import AccountingRequest

    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "spool.sqlite"))
    client = PrismaClient(
        database_url="postgresql://unused@127.0.0.1:1/probe",
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )
    instance = PostgresAccounting(client)
    request = AccountingRequest(store=instance)
    token = accounting_request.set(request)
    try:
        await instance.enqueue(
            {
                "request_id": "reported-provider-or-caller-id",
                "metadata": metadata,
                "completionStartTime": datetime.now(timezone.utc),
            },
            {},
            0.03,
        )
        _, pending = await envelopes(instance)
        assert pending[0].raw["request_id"] == request.component_id
        assert pending[0].raw["metadata"]["reported_request_id"] == "reported-provider-or-caller-id"
        assert pending[0].raw["metadata"]["accounting_request_id"] == request.request_id
        assert isinstance(pending[0].raw["completionStartTime"], str)
    finally:
        accounting_request.reset(token)
        await client.db.disconnect(timeout=timedelta(seconds=2))


@pytest.mark.asyncio
async def test_read_replica_cannot_supply_authoritative_protocol_reads(monkeypatch):
    from litellm.proxy.spend_tracking.postgres_accounting import initialize

    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://not-used.invalid/replica")
    with pytest.raises(RuntimeError, match="writer-only reads"):
        await initialize(None, {"postgres_admission_accounting": True}, None)
    await initialize(None, {"postgres_admission_accounting": False}, None)


@pytest.mark.parametrize(
    "field",
    ["background", "previous_response_id", "conversation", "extra_body", "complete_response", "litellm_metadata"],
)
def test_responses_unfinished_ownership_is_rejected(field):
    with pytest.raises(Exception):
        validate_request({"model": "openai/gpt-6-astra", "input": "hello", field: {"enabled": True}})


def test_native_foreground_responses_and_installed_settings_are_not_a_second_schema():
    config = configuration()
    config["general_settings"].update(
        always_include_stream_usage=True,
        database_connection_pool_timeout=10,
        store_prompts_in_spend_logs=True,
        maximum_spend_logs_retention_period="90d",
        maximum_spend_logs_retention_interval="1d",
        ui_access_mode="all",
        store_model_in_db=True,
    )
    config["litellm_settings"].update(
        drop_params=True,
        turn_off_message_logging=False,
        redact_messages_in_exceptions=False,
        redact_user_api_key_info=False,
        stream_timeout=30,
        request_timeout=90,
        callbacks=["callbacks.request_context.handler"],
    )
    config["router_settings"].update(
        num_retries=2,
        routing_strategy="usage-based-routing",
        enable_pre_call_checks=True,
        optional_pre_call_checks=["encrypted_content_affinity"],
    )
    validate_config(config)
    validate_request(
        {
            "model": "openai/gpt-6-astra",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 1000,
            "stream": True,
            "store": False,
            "background": False,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "session",
            "tools": [{"type": "function", "name": "read", "parameters": {}}],
        }
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "managed_by": "openorange-installer",
            "openorange_role": "operator",
            "model_access": "all_instance_models",
            "openorange_instance_id": "fixture",
            "openorange_operator_agent_id": "operator",
        },
        {
            "managed_by": "openorange-installer",
            "openorange_key_kind": "agent_server",
            "model_access": "all_instance_models",
            "openorange_instance_id": "fixture",
        },
        {
            "managed_by": "openorange-operator",
            "agent_id": "assistant",
            "channel": "mattermost",
            "username": "assistant",
        },
        {
            "openclaw_channel": "code",
            "openorange_code_session_id": "session",
            "openorange_code_project_id": "project",
            "openorange_code_project_slug": "slug",
        },
    ],
)
def test_authenticated_key_metadata_is_attribution_not_a_new_budget(metadata):
    PostgresAccounting.validate_key({"user_id": "native-user", "metadata": metadata})
    with pytest.raises(Exception):
        PostgresAccounting.validate_key({"metadata": {**metadata, "tags": ["budget-scope"]}})


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (400, {"error": {"type": "invalid_request_error", "code": "output_validation_error"}}, False),
        (400, {"error": {"type": "invalid_request_error", "message": "output invalid"}}, False),
        (429, {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded"}}, True),
        (400, {"error": {"type": "invalid_request_error", "code": "context_length_exceeded"}}, True),
        (401, {"error": {"type": "invalid_request_error", "code": "invalid_api_key"}}, True),
        (404, {"error": {"type": "invalid_request_error", "code": "model_not_found"}}, True),
        (403, {"error": {"message": "policy failed after output"}}, False),
        (422, {"error": {"message": "output validation failed"}}, False),
        (429, {"error": {"type": [], "code": []}}, False),
        (400, [], False),
    ],
)
def test_only_documented_provider_preexecution_errors_release_liability(status, body, expected):
    from litellm.llms.custom_httpx.http_handler import _accounting_preexecution_error

    assert _accounting_preexecution_error(status, json.dumps(body).encode()) is expected
    assert not _accounting_preexecution_error(status, b"not-json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,size,rejected",
    [
        (429, "rate_limit_exceeded", 100, True),
        (429, "rate_limit_exceeded", 70000, False),
        (400, "output_validation_error", 100, False),
    ],
)
async def test_native_error_body_observation_is_bounded_and_does_not_consume_ahead(
    protocol, monkeypatch, status, code, size, rejected
):
    import httpx
    from litellm.llms.custom_httpx.http_handler import _accounting_response_hook
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, request, key = protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    wire = json.dumps(
        {
            "error": {
                "type": "rate_limit_error" if status == 429 else "invalid_request_error",
                "code": code,
                "message": "x" * size,
            }
        }
    ).encode()
    consumed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(wire), 1000):
                consumed.append(offset)
                yield wire[offset : offset + 1000]

    response = httpx.Response(
        status,
        stream=Body(),
        request=httpx.Request(
            "POST", "http://fixture/responses", extensions={"litellm_accounting_component": request.component_id}
        ),
    )
    await _accounting_response_hook(response)
    assert not consumed
    assert await response.aread() == wire
    assert (await totals(instance, key))["held"] == (0 if rejected else 0.1)
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == (0 if rejected else 1)


def test_request_authority_is_not_taken_from_metadata_and_overrides_fail_closed():
    validate_request(
        {
            "model": "model",
            "messages": [],
            "metadata": {
                "request_id": "forged",
                "component_id": "forged",
                "receipt_id": "forged",
                "user_api_key_budget_reservation": {"finalized": True},
            },
        }
    )
    for field in ("user", "tags", "api_base", "num_retries", "mock_response"):
        with pytest.raises(Exception):
            validate_request({"model": "model", "messages": [], field: "override"})


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_loss", [False, True])
async def test_durable_seal_recovery_reconstructs_only_consumer(protocol, ack_loss):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await apply_all(instance)
    assert (await instance.client._spend_log_spool.stats()).count == 0
    assert await instance.drain() == 1
    await request.finish()
    _, pending = await envelopes(instance)
    assert len(pending) == 1 and isinstance(pending[0], ProducerSeal)

    @asynccontextmanager
    async def failing_transaction():
        async with instance.client.db.tx() as tx:
            yield tx
            if not ack_loss:
                await tx.execute_raw("SELECT 1/0")
        raise TimeoutError("seal commit acknowledgement lost")

    consumer = PostgresAccounting(instance.client, settlement_transaction=failing_transaction)
    with pytest.raises(Exception):
        await consumer.apply(pending[0])
    assert (await instance.client._spend_log_spool.stats()).count == 1
    assert await instance.drain() == (0 if ack_loss else 1)
    reopened_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"],
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )
    try:
        await reopened_client.connect()
        restarted = PostgresAccounting(reopened_client)
        await apply_all(restarted)
        await restarted.apply(pending[0])
    finally:
        await reopened_client.db.disconnect(timeout=timedelta(seconds=2))
    assert await instance.drain() == 0
    assert await totals(instance, key) == {"entity": 0.03, "raw": 0.03, "daily": 0.03, "records": 1, "held": 0.0}
    assert await instance.client.db.query_raw(
        "SELECT COUNT(*)::int AS count FROM \"LiteLLM_AccountingReceipt\" WHERE component_id=$1 AND basis='seal'",
        request.component_id,
    ) == [{"count": 1}]


@pytest.mark.asyncio
async def test_seal_manifest_validates_receipts_and_preserves_failure(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    _, pending = await envelopes(instance)
    actual, seal = pending
    with pytest.raises(RuntimeError, match="awaiting accounting receipts"):
        await instance.apply(seal)
    await instance.apply(actual)
    forged = seal.model_copy(update={"expected": ((actual.receipt_id, "wrong-hash"),)})
    with pytest.raises(ValueError, match="manifest conflict"):
        await instance.apply(forged)
    await apply_all(PostgresAccounting(instance.client))
    await instance.apply(seal)
    assert await instance.drain() == 1
    assert await instance.client.db.query_raw(
        'SELECT sealed, failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1', request.request_id
    ) == [{"sealed": True, "failed": True, "closed": False}]
    with pytest.raises(ValueError, match="payload conflict"):
        await instance.apply(seal.model_copy(update={"failed": True}))


@pytest.mark.asyncio
async def test_predispatch_cancel_closes_without_input_or_raw_spend(protocol):
    instance, request, key = protocol
    await instance.cancel()
    await request.finish()
    await apply_all(instance)
    nonexecuted = await instance.begin()
    token = accounting_request.set(nonexecuted)
    try:
        await instance.admit(key, 0.09, 0.01)
        await instance.cancel()
        await nonexecuted.finish()
        await apply_all(instance)
    finally:
        accounting_request.reset(token)
    assert await totals(instance, key) == {"entity": 0.0, "raw": 0.0, "daily": 0.0, "records": 0, "held": 0.01}
    assert await instance.client.db.query_raw(
        'SELECT sealed, failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1', nonexecuted.request_id
    ) == [{"sealed": True, "failed": False, "closed": True}]
    assert await instance.drain() == 1  # Dispatched unknown remains, only nonexecution retires.


@pytest.mark.asyncio
async def test_cancel_during_dispatch_intent_ack_does_not_claim_nonexecution(protocol):
    instance, original, key = protocol
    request = await instance.begin()
    token = accounting_request.set(request)
    try:
        # This isolates the dispatch intent acknowledgement, not a provider execution.
        request.admitted = True
        async with instance.client.db.tx() as gate:
            await gate.query_raw(
                'SELECT id FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', request.request_id
            )
            task = asyncio.create_task(instance.dispatch())
            async with asyncio.timeout(5):
                while True:
                    await gate.query_raw("SELECT pg_stat_clear_snapshot()::text")
                    blocked = await gate.query_raw(
                        "SELECT pid FROM pg_stat_activity WHERE pg_backend_pid() = ANY(pg_blocking_pids(pid))"
                    )
                    if blocked:
                        break
                    await asyncio.sleep(0.01)
            assert request.dispatched and not task.done()
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await request.finish()
        await apply_all(instance)
        assert await instance.client.db.query_raw(
            'SELECT failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1', request.request_id
        ) == [{"failed": True, "closed": False}]
    finally:
        accounting_request.reset(token)


@pytest.mark.asyncio
async def test_blocked_actual_commit_keeps_liability_and_no_partial_spend(protocol):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, pending = await envelopes(instance)
    task = None
    try:
        async with instance.client.db.tx() as gate:
            await gate.query_raw('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=$1 FOR UPDATE', key)
            task = asyncio.create_task(instance.apply(pending[0]))
            for _ in range(100):
                await gate.query_raw("SELECT pg_stat_clear_snapshot()::text")
                blocked = await gate.query_raw(
                    "SELECT pid FROM pg_stat_activity WHERE pg_backend_pid() = ANY(pg_blocking_pids(pid))"
                )
                if blocked:
                    break
                await asyncio.sleep(0.02)
            assert blocked and not task.done()
            state = await gate.query_raw(
                'SELECT (SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=$1) AS spend, '
                '(SELECT amount FROM "LiteLLM_AccountingHold" WHERE request_id=$2) AS held, '
                '(SELECT COUNT(*)::int FROM "LiteLLM_AccountingReceipt" WHERE component_id=$3) AS receipts, '
                '(SELECT COUNT(*)::int FROM "LiteLLM_SpendLogs" WHERE api_key=$1) AS raw',
                key,
                request.request_id,
                request.component_id,
            )
            assert state == [{"spend": 0.0, "held": 0.1, "receipts": 0, "raw": 0}]
    finally:
        if task is not None:
            await asyncio.wait_for(task, 5)
    assert await instance.current_spend(f"spend:key:{key}") == pytest.approx(0.03)
    assert await instance.drain() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("drop", ["prestart_cancel", "uninitialized_queue", "queued_stop"])
async def test_unscheduled_producer_failure_is_durable_even_after_actual(protocol, drop):
    instance, request, key = protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await apply_all(instance)
    worker = LoggingWorker(timeout=0.01, concurrency=1)
    ran = False

    async def callback():
        nonlocal ran
        ran = True

    if drop == "prestart_cancel":
        task = spawn_accounting(callback())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif drop == "uninitialized_queue":
        worker.enqueue(callback())
    else:
        worker.ensure_initialized_and_enqueue(asyncio.Event().wait())
        await asyncio.sleep(0)
        worker.enqueue(callback())
        await worker.stop()
    await request.finish()
    for _ in range(100):
        if (await instance.client._spend_log_spool.stats()).count:
            break
        await asyncio.sleep(0.01)
    await apply_all(PostgresAccounting(instance.client))
    assert request.producers == 0
    assert request.failed
    assert not ran or drop == "queued_stop"
    assert await instance.client.db.query_raw(
        'SELECT sealed, failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1', request.request_id
    ) == [{"sealed": True, "failed": True, "closed": False}]
    assert await instance.drain() == 1
    assert (await totals(instance, key))["held"] == 0


@pytest.mark.asyncio
async def test_stopping_queue_full_retry_cannot_certify_producer_success(protocol):
    instance, request, key = protocol
    worker = LoggingWorker(timeout=0.05, max_queue_size=1, concurrency=1)
    started = asyncio.Event()
    aggressive_started = asyncio.Event()
    release = asyncio.Event()
    replayed = False

    async def blocker(event):
        event.set()
        await release.wait()

    async def retry():
        nonlocal replayed
        replayed = True

    worker.ensure_initialized_and_enqueue(blocker(started))
    await started.wait()
    worker.enqueue(release.wait())
    worker.enqueue(blocker(aggressive_started))
    await aggressive_started.wait()
    worker.enqueue(retry())
    await worker.stop()
    await request.finish()
    for _ in range(100):
        if (await instance.client._spend_log_spool.stats()).count:
            break
        await asyncio.sleep(0.01)
    await apply_all(PostgresAccounting(instance.client))
    assert not replayed
    assert request.producers == 0
    assert await instance.client.db.query_raw(
        'SELECT sealed, failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1', request.request_id
    ) == [{"sealed": True, "failed": True, "closed": False}]
    assert await instance.drain() == 1
    assert (await totals(instance, key))["held"] == 0.1


@pytest_asyncio.fixture(loop_scope="function")
async def user_protocol(protocol):
    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    user = str(uuid4())
    await instance.write_user(user, {"user_id": user, "models": [], "max_budget": 0.1}, {})
    await instance.update_key(key, {"user_id": user})
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        await instance.admit(key, 1, 0.01)
        await instance.dispatch()
        yield instance, request, key
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["unknown", "rejected", "ack_loss", "cancel_before", "cancel_after", "producer", "seal_recovery"]
)
async def test_native_user_component_lifecycle(user_protocol, case):
    instance, request, key = user_protocol
    if case in {"unknown", "rejected"}:
        await test_component_actual_does_not_release_prior_unknown_attempt(user_protocol, case == "rejected")
    elif case == "ack_loss":
        await test_commit_ack_loss_replays_only_accounting_from_retained_outbox(user_protocol)
    elif case.startswith("cancel_"):
        await test_actual_replaces_estimate_and_late_cancel_cannot_overwrite_actual(
            user_protocol, case == "cancel_after"
        )
    elif case == "producer":
        await test_actual_releases_money_for_next_admission_but_not_retirement(user_protocol)
    else:
        await test_durable_seal_recovery_reconstructs_only_consumer(user_protocol, True)
    row = await instance.refresh_user(request.user_id, include_holds=True)
    assert row["spend"] == pytest.approx(0.13 if case == "unknown" else (0.1 if case == "producer" else 0.03))
    user = await instance.refresh_user(request.user_id)
    assert user["spend"] == pytest.approx(0.03)
    memberships = await instance.client.db.query_raw(
        'SELECT amount FROM "LiteLLM_AccountingHold" WHERE request_id=$1 ORDER BY scope_id', request.request_id
    )
    assert len(memberships) == 2 and memberships[0] == memberships[1]


@pytest.mark.asyncio
async def test_native_user_uncapped_membership_cap_changes_reassignment_and_auth(protocol, monkeypatch):
    import litellm
    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    user = str(uuid4())
    other = str(uuid4())
    await instance.write_user(user, {"user_id": user, "models": []}, {})
    await instance.write_user(other, {"user_id": other, "models": []}, {})
    await instance.update_key(key, {"user_id": user, "max_budget": 10})
    request = await instance.begin()
    context = accounting_request.set(request)
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    try:
        await instance.admit(key, 1, 0.01)
        await instance.dispatch()
        cache = DualCache()
        await cache.async_set_cache(key=user, value={"user_id": user, "spend": 100, "max_budget": 0})
        for cap in (0.1, 0.05, 0):
            await instance.client.update_data(user_id=user, data={"max_budget": cap})
            authoritative = await get_user_object(user, instance.client, cache, False)
            assert authoritative.max_budget == cap and authoritative.spend == 0
            competing = await instance.begin()
            inner = accounting_request.set(competing)
            try:
                with pytest.raises(litellm.BudgetExceededError):
                    await instance.admit(key, 1, 0.01)
                await competing.finish()
            finally:
                accounting_request.reset(inner)
        await instance.client.update_data(user_id=user, data={"max_budget": None})
        assert (await instance.refresh_user(user, include_holds=True))["spend"] == 1
        await instance.update_key(key, {"user_id": other})
        from fastapi import HTTPException
        from litellm.proxy._types import UserAPIKeyAuth

        with pytest.raises(HTTPException, match="actor key authority changed"):
            await instance.update_key(
                key,
                {"key_alias": "stale-owner"},
                auth=UserAPIKeyAuth(token=key, user_id=user, user_role="internal_user"),
            )
        raw, daily = await observation(instance, request)
        await instance.enqueue(raw, daily, 0.03)
        await instance.dispatch()
        await instance.reject_component(instance.component_id())
        await request.finish()
        await apply_all(instance)
        assert (await instance.refresh_user(user))["spend"] == 0.03
        assert (await instance.refresh_user(other))["spend"] == 0
        assert (await instance.refresh_user(user, include_holds=True))["spend"] == 0.03
        following = await instance.begin()
        following_context = accounting_request.set(following)
        try:
            await instance.admit(key, 0.1, 0.01)
            assert following.user_id == other
            await following.finish()
            await apply_all(instance)
        finally:
            accounting_request.reset(following_context)
        assert await instance.drain() == 0
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
async def test_native_user_reset_uses_pg_clock_and_does_not_erase_old_liability(user_protocol, monkeypatch):
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, request, key = user_protocol
    user = request.user_id
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    await instance.client.update_data(user_id=user, data={"budget_duration": "30d"})
    await instance.client.db.execute_raw(
        "UPDATE \"LiteLLM_UserTable\" SET spend=0.04, budget_reset_at=clock_timestamp()-interval '1 second' WHERE user_id=$1",
        user,
    )
    stale = await instance.client.db.litellm_usertable.find_unique(where={"user_id": user})
    stale.budget_reset_at = datetime(2099, 1, 1, tzinfo=timezone.utc)
    job = ResetBudgetJob(instance.client.proxy_logging_obj, instance.client)
    await asyncio.gather(job.reset_budget(), job.reset_budget_for_litellm_users())
    assert (await instance.refresh_user(user))["spend"] == 0
    assert (await instance.refresh_user(user, include_holds=True))["spend"] == 0.1
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)
    await asyncio.gather(job._write_user_reset_updates([stale]), job._write_user_reset_updates([stale]))
    assert (await instance.refresh_user(user))["spend"] == 0.03
    assert await instance.client.db.query_raw(
        "SELECT budget_reset_at > clock_timestamp() AND budget_reset_at < clock_timestamp()+interval '32 days' AS bounded FROM \"LiteLLM_UserTable\" WHERE user_id=$1",
        user,
    ) == [{"bounded": True}]
    await instance.client.update_data(user_id=user, data={"budget_duration": None})
    assert (await instance.refresh_user(user))["budget_reset_at"] is None
    assert (await instance.refresh_user(user))["spend"] == 0.03


@pytest.mark.asyncio
async def test_native_user_mutation_and_all_settlement_leaves_wait_for_native_row(user_protocol):
    instance, request, key = user_protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, pending = await envelopes(instance)
    async with instance.client.db.tx() as gate:
        await gate.query_raw('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=$1 FOR UPDATE', request.user_id)
        settlement = asyncio.create_task(instance.apply(pending[0]))
        await asyncio.sleep(0.1)
        assert not settlement.done()
        # Use the already-held connection; the fixture deliberately has only two.
        assert await gate.query_raw(
            'SELECT (SELECT COUNT(*)::int FROM "LiteLLM_SpendLogs" WHERE request_id=$1) AS raw, '
            '(SELECT COUNT(*)::int FROM "LiteLLM_AccountingReceipt" WHERE component_id=$1) AS receipts',
            request.component_id,
        ) == [{"raw": 0, "receipts": 0}]
        update = asyncio.create_task(instance.write_user(request.user_id, {}, {"max_budget": 0.02}))
        await asyncio.sleep(0.05)
        assert not update.done()
    await asyncio.gather(settlement, update)
    assert (await instance.refresh_user(request.user_id))["spend"] == 0.03
    assert (await instance.refresh_user(request.user_id))["max_budget"] == 0.02
    assert (await totals(instance, key))["held"] == 0
    assert await instance.drain() == 1
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == 0


@pytest.mark.asyncio
async def test_native_user_membership_mismatch_rolls_back_instead_of_clamping(user_protocol):
    instance, request, key = user_protocol
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_AccountingHold" SET amount=0.01 WHERE request_id=$1 AND scope_id=$2',
        request.request_id,
        f"user:{request.user_id}:initial",
    )
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, events = await envelopes(instance)
    with pytest.raises(ValueError, match="membership liability mismatch"):
        await instance.apply(events[0])
    assert (await totals(instance, key))["raw"] == 0
    assert (await instance.refresh_user(request.user_id))["spend"] == 0
    assert await instance.drain() == 1


@pytest.mark.asyncio
async def test_native_user_pre_call_hook_honors_server_owned_admission(user_protocol, monkeypatch):
    from fastapi import HTTPException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, request, key = user_protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    hook = _PROXY_MaxBudgetLimiter()
    auth = UserAPIKeyAuth(token=key, user_id=request.user_id, user_max_budget=0.1)
    await hook.async_pre_call_hook(auth, DualCache(), {"model": "test-model"}, "acompletion")
    assert (await instance.refresh_user(request.user_id, include_holds=True))["spend"] == 0.1
    token = accounting_request.set(None)
    try:
        with pytest.raises(HTTPException, match="budget admission missing"):
            await hook.async_pre_call_hook(auth, DualCache(), {}, "acompletion")
    finally:
        accounting_request.reset(token)


@pytest.mark.asyncio
async def test_native_user_static_models_and_team_key_exemption_remain_guarded(protocol):
    from fastapi import HTTPException

    instance, original, key = protocol
    user = str(uuid4())
    await instance.write_user(user, {"user_id": user, "models": ["allowed-only"], "max_budget": 0.1}, {})
    await instance.update_key(key, {"user_id": user})
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        with pytest.raises(Exception, match="model"):
            await instance.admit(key, 1, 0.01, model="not-allowed")
        assert not request.admitted
        await instance.client.db.execute_raw(
            'UPDATE "LiteLLM_VerificationToken" SET team_id=$2 WHERE token=$1', key, "unqualified-team"
        )
        with pytest.raises(HTTPException, match="native Team missing"):
            await instance.admit(key, 1, 0.01, model="allowed-only")
        assert not request.admitted
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
async def test_native_user_created_after_admission_inherits_only_outstanding_membership(protocol):
    import litellm

    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    user = str(uuid4())
    await instance.update_key(key, {"user_id": user})
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        await instance.admit(key, 1, 0.01)
        assert await instance.refresh_user(user) is None
        await instance.write_user(user, {"user_id": user, "models": [], "max_budget": 0.05}, {})
        assert (await instance.refresh_user(user, include_holds=True))["spend"] == 0.1
        other_key = str(uuid4())
        await instance.client.db.execute_raw(
            'INSERT INTO "LiteLLM_VerificationToken" (token, models, user_id) VALUES ($1, ARRAY[]::text[], $2)',
            other_key,
            user,
        )
        competing = await instance.begin()
        inner = accounting_request.set(competing)
        try:
            with pytest.raises(litellm.BudgetExceededError):
                await instance.admit(other_key, 1, 0.01)
            await competing.finish()
        finally:
            accounting_request.reset(inner)
        await request.finish()
        await apply_all(instance)
        assert (await instance.refresh_user(user, include_holds=True))["spend"] == 0
        assert await instance.drain() == 0
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
async def test_native_user_endpoint_recurrence_removal_and_self_update_authz(user_protocol, monkeypatch):
    from fastapi import HTTPException
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UpdateUserRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import _update_single_user_helper
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, request, key = user_protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    monkeypatch.setattr(proxy_server, "prisma_client", instance.client)
    admin = UserAPIKeyAuth(user_role="proxy_admin")
    owner = UserAPIKeyAuth(user_id=request.user_id, user_role="internal_user")
    for duration in ("30d", None):
        data = UpdateUserRequest(user_id=request.user_id, budget_duration=duration)
        with pytest.raises(HTTPException) as error:
            await _update_single_user_helper(data, owner)
        assert error.value.status_code == 403
        await _update_single_user_helper(data, admin)
        user = await instance.refresh_user(request.user_id)
        assert user["budget_duration"] == duration and user["spend"] == 0
        assert (user["budget_reset_at"] is None) == (duration is None)
        assert (await instance.refresh_user(request.user_id, include_holds=True))["spend"] == 0.1


@pytest.mark.asyncio
async def test_native_user_cancel_after_known_rejection_does_not_recreate_liability(user_protocol):
    instance, request, key = user_protocol
    await instance.reject_component(request.component_id)
    await instance.cancel()
    await request.finish()
    await apply_all(instance)
    assert (await instance.refresh_user(request.user_id, include_holds=True))["spend"] == 0
    assert (await totals(instance, key))["held"] == 0
    assert await instance.drain() == 0


def team_admin():
    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS, user_role="proxy_admin")


@pytest_asyncio.fixture(loop_scope="function")
async def team_protocol(protocol):
    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    user, team = str(uuid4()), str(uuid4())
    await instance.write_user(user, {"user_id": user, "models": [], "max_budget": 0, "user_role": "internal_user"}, {})
    await instance.create_team(
        {"team_id": team, "max_budget": 0.1, "admins": [], "members": [], "models": []},
        [{"user_id": user, "role": "admin"}],
        team_admin(),
    )
    await instance.update_key(key, {"user_id": user, "team_id": team}, auth=team_admin())
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        await instance.admit(key, 1, 0.01)
        await instance.dispatch()
        yield instance, request, key
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["unknown", "rejected", "ack_loss", "cancel_before", "cancel_after", "producer", "seal_recovery", "known_cancel"],
)
async def test_native_team_component_lifecycle(team_protocol, case):
    instance, request, key = team_protocol
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_TeamMembership" (team_id,user_id) VALUES ($1,$2)', request.team_id, request.user_id
    )
    if case in {"unknown", "rejected"}:
        await test_component_actual_does_not_release_prior_unknown_attempt(team_protocol, case == "rejected")
    elif case == "ack_loss":
        await test_commit_ack_loss_replays_only_accounting_from_retained_outbox(team_protocol)
    elif case.startswith("cancel_"):
        await test_actual_replaces_estimate_and_late_cancel_cannot_overwrite_actual(
            team_protocol, case == "cancel_after"
        )
    elif case == "producer":
        await test_actual_releases_money_for_next_admission_but_not_retirement(team_protocol)
    elif case == "known_cancel":
        await test_native_user_cancel_after_known_rejection_does_not_recreate_liability(team_protocol)
    else:
        await test_durable_seal_recovery_reconstructs_only_consumer(team_protocol, True)
    booked = 0 if case == "known_cancel" else 0.03
    assert (await instance.refresh_user(request.user_id))["spend"] == pytest.approx(booked)
    assert (await instance.refresh_team(request.team_id))["spend"] == pytest.approx(booked)
    assert (await instance.refresh_team(request.team_id, include_holds=True))["spend"] == pytest.approx(
        0.13 if case == "unknown" else 0.1 if case == "producer" else booked
    )
    member = await instance.client.db.query_raw(
        'SELECT spend,total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=$1 AND user_id=$2',
        request.team_id,
        request.user_id,
    )
    assert member == [{"spend": booked, "total_spend": booked}]
    scopes = await instance.client.db.query_raw(
        'SELECT s.kind,h.amount FROM "LiteLLM_AccountingHold" h JOIN "LiteLLM_AccountingScope" s ON s.id=h.scope_id WHERE h.request_id=$1 ORDER BY s.kind',
        request.request_id,
    )
    assert [s["kind"] for s in scopes] == ["key", "team"] and scopes[0]["amount"] == scopes[1]["amount"]


@pytest.mark.asyncio
async def test_native_team_bookkeeping_user_lock_and_absent_member(team_protocol):
    instance, request, key = team_protocol
    await test_native_user_mutation_and_all_settlement_leaves_wait_for_native_row(team_protocol)
    assert (await instance.refresh_team(request.team_id))["spend"] == 0.03
    assert (
        await instance.client.db.query_raw('SELECT * FROM "LiteLLM_TeamMembership" WHERE team_id=$1', request.team_id)
        == []
    )
    assert await instance.client.db.query_raw(
        'SELECT SUM(spend) AS spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=$1', request.team_id
    ) == [{"spend": 0.03}]


@pytest.mark.asyncio
async def test_native_team_recurrence_and_stale_resetter(team_protocol, monkeypatch):
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, request, key = team_protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    await instance.update_team(request.team_id, {"budget_duration": "30d"}, team_admin())
    await instance.client.db.execute_raw(
        "UPDATE \"LiteLLM_TeamTable\" SET spend=.04,budget_reset_at=clock_timestamp()-interval '1 second' WHERE team_id=$1",
        request.team_id,
    )
    stale = await instance.client.db.litellm_teamtable.find_unique(where={"team_id": request.team_id})
    job = ResetBudgetJob(instance.client.proxy_logging_obj, instance.client)
    await asyncio.gather(job.reset_budget(), job.reset_budget_for_litellm_teams())
    assert (await instance.refresh_team(request.team_id))["spend"] == 0
    assert (await instance.refresh_team(request.team_id, include_holds=True))["spend"] == 0.1
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    await request.finish()
    await apply_all(instance)
    await asyncio.gather(job._write_team_reset_updates([stale]), job._write_team_reset_updates([stale]))
    await instance.update_team(request.team_id, {"budget_duration": None}, team_admin())
    assert (await instance.refresh_team(request.team_id))["spend"] == 0.03
    assert (await instance.refresh_user(request.user_id))["spend"] == 0.03


@pytest.mark.asyncio
async def test_native_team_fresh_authority_and_uncapped_holds(team_protocol):
    import litellm
    from fastapi import HTTPException
    from litellm.proxy._types import UserAPIKeyAuth

    instance, request, key = team_protocol
    actor_key = str(uuid4())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token,user_id,models) VALUES ($1,$2,ARRAY[]::text[])',
        actor_key,
        request.user_id,
    )
    actor = UserAPIKeyAuth(token=actor_key, user_id=request.user_id, user_role="internal_user")
    await instance.update_team(request.team_id, {"max_budget": 0.05}, actor)
    for cap in (0.1, None):
        with pytest.raises(HTTPException):
            await instance.update_team(request.team_id, {"max_budget": cap}, actor)
    await instance.update_team(request.team_id, {"max_budget": None}, team_admin())
    assert (await instance.refresh_team(request.team_id))["spend"] == 0
    await instance.update_team(request.team_id, {"max_budget": 0.1}, actor)
    other_key = str(uuid4())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token,user_id,team_id,models) VALUES ($1,$2,$3,ARRAY[]::text[])',
        other_key,
        request.user_id,
        request.team_id,
    )
    competing = await instance.begin()
    context = accounting_request.set(competing)
    try:
        with pytest.raises(litellm.BudgetExceededError):
            await instance.admit(other_key, 0.1, 0.01)
        await competing.finish()
    finally:
        accounting_request.reset(context)
    other_team = str(uuid4())
    await instance.create_team(
        {"team_id": other_team, "admins": [], "members": [], "models": []},
        [{"user_id": request.user_id, "role": "user"}],
        team_admin(),
    )
    with pytest.raises(HTTPException):
        await instance.update_key(key, {"team_id": other_team}, auth=actor)
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_TeamTable" SET members_with_roles=$2::jsonb WHERE team_id=$1',
        request.team_id,
        json.dumps([{"user_id": request.user_id, "role": "user"}]),
    )
    with pytest.raises(HTTPException):
        await instance.update_team(request.team_id, {"max_budget": 0.05}, actor)
    stale_admin = actor.model_copy(update={"user_role": "proxy_admin"})
    with pytest.raises(HTTPException):
        await instance.update_team(request.team_id, {"max_budget": 10}, stale_admin)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cap,spend,allowed", [(None, 0.2, True), (0, 0, True), (0, 0.03, False), (-1, 0, False), (0.1, 0.1, False)]
)
async def test_native_team_zero_null_negative_and_equality(protocol, cap, spend, allowed):
    import litellm
    from litellm.proxy.auth.auth_checks import team_budget_exceeded

    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    team = str(uuid4())
    await instance.create_team(
        {"team_id": team, "max_budget": cap, "spend": spend, "admins": [], "members": [], "models": []},
        [],
        team_admin(),
    )
    await instance.update_key(key, {"team_id": team}, auth=team_admin())
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        assert team_budget_exceeded(spend, cap) == (cap is not None and spend > cap)
        if allowed:
            await instance.admit(key, 0.1, 0.01)
            assert request.team_id == team
        else:
            with pytest.raises(litellm.BudgetExceededError):
                await instance.admit(key, 0.1, 0.01)
        await request.finish()
        await apply_all(instance)
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
async def test_native_team_public_preparation_and_duplicate_key(protocol, monkeypatch):
    from starlette.requests import Request
    from litellm.proxy import proxy_server
    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_helper_fn
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, _, _ = protocol
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    monkeypatch.setattr(proxy_server, "prisma_client", instance.client)
    team = str(uuid4())
    auth = team_admin().model_copy(update={"user_id": str(uuid4())})
    result = await new_team(
        NewTeamRequest(team_id=team, max_budget=0.1),
        Request({"type": "http", "method": "POST", "path": "/team/new", "headers": []}),
        auth,
    )
    assert result["team_id"] == team
    assert team in (await instance.refresh_user(auth.user_id))["teams"]
    raw_key = "sk-" + str(uuid4())
    data = {
        "token": raw_key,
        "team_id": team,
        "user_id": auth.user_id,
        "models": [],
        "key_max_budget": 10,
        "accounting_auth": auth,
        "table_name": "key",
    }
    created = await generate_key_helper_fn(request_type="key", **data)
    assert isinstance(created, dict)
    token = created["token_id"]
    before = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": token})
    duplicate = await generate_key_helper_fn(
        request_type="key", **{**data, "spend": 0.7, "metadata": {"duplicate": True}}
    )
    after = await instance.client.db.litellm_verificationtoken.find_unique(where={"token": token})
    assert duplicate["token_id"] == token
    assert after.model_dump() == before.model_dump()
    assert (await instance.refresh_team(team))["spend"] == 0


@pytest.mark.asyncio
async def test_native_team_row_barrier_blocks_every_projection(team_protocol):
    instance, request, key = team_protocol
    raw, daily = await observation(instance, request)
    await instance.enqueue(raw, daily, 0.03)
    _, pending = await envelopes(instance)
    async with instance.client.db.tx() as gate:
        await gate.query_raw('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id=$1 FOR UPDATE', request.team_id)
        settlement = asyncio.create_task(instance.apply(pending[0]))
        await asyncio.sleep(0.1)
        assert not settlement.done()
        assert await gate.query_raw(
            'SELECT (SELECT COUNT(*)::int FROM "LiteLLM_SpendLogs" WHERE request_id=$1) AS raw, '
            '(SELECT COUNT(*)::int FROM "LiteLLM_AccountingReceipt" WHERE component_id=$1) AS receipts, '
            '(SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=$2) AS user_spend',
            request.component_id,
            request.user_id,
        ) == [{"raw": 0, "receipts": 0, "user_spend": 0}]
    await settlement
    assert (await instance.refresh_user(request.user_id))["spend"] == 0.03
    assert (await instance.refresh_team(request.team_id))["spend"] == 0.03
    await request.finish()
    await apply_all(instance)
    assert await instance.drain() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_native_responses_http_optional_fields_and_predispatch_denials(protocol, monkeypatch, enabled):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import httpx
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.proxy.utils import hash_token

    instance, original, _ = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    received = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "id": "resp_native_test",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o",
                    "output": [
                        {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    token, team = "sk-native-http-" + str(uuid4()), str(uuid4())
    await instance.create_team({"team_id": team, "models": ["gpt-4o"], "admins": [], "members": []}, [], team_admin())
    await instance.client.db.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token, models, team_id, max_budget) VALUES ($1,$2::text[],$3,1)',
        hash_token(token),
        ["all-team-models"],
        team,
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "local-provider-only",
                    "api_base": f"http://127.0.0.1:{provider.server_port}/v1",
                },
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "prisma_client", instance.client)
    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "general_settings", {"postgres_admission_accounting": enabled})
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", instance.client.proxy_logging_obj)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(postgres_accounting, "runtime", instance if enabled else None)
    context = accounting_request.set(None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ) as client:
            for strict, detail in [(None, None), (False, "auto"), (True, "low")]:
                body = {
                    "model": "gpt-4o",
                    "max_output_tokens": 10,
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                                "required": [],
                            },
                        }
                    ],
                }
                if strict is not None:
                    body["tools"][0]["strict"] = strict
                if detail is not None:
                    body["input"][0]["content"][0]["detail"] = detail
                response = await client.post("/v1/responses", headers={"Authorization": "Bearer " + token}, json=body)
                assert response.status_code == 200, response.text
                assert received[-1]["input"] == body["input"]
                assert received[-1]["tools"] == body["tools"]
            before = len(received)
            for body in [
                {"model": "gpt-4o", "input": 123},
                {"model": "gpt-4o", "input": "hello", "max_output_tokens": "invalid"},
                {"model": "other", "input": "hello"},
            ]:
                response = await client.post("/v1/responses", headers={"Authorization": "Bearer " + token}, json=body)
                assert 400 <= response.status_code < 500, response.text
                assert len(received) == before
            if enabled:
                await instance.update_team(team, {"models": ["changed"]}, team_admin())
                response = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer " + token},
                    json={"model": "gpt-4o", "input": "hello"},
                )
                assert response.status_code == 403, response.text
                assert len(received) == before
                assert await instance.client.db.query_raw(
                    'SELECT COUNT(*)::int AS n FROM "LiteLLM_AccountingRequest" WHERE key_token=$1 AND dispatched',
                    hash_token(token),
                ) == [{"n": 3}]
                for field in ("model_rpm_limit", "model_tpm_limit"):
                    for policy in ({field: {"gpt-4o": 0}}, {"metadata": {field: {"gpt-4o": 1}}}):
                        response = await client.post(
                            "/key/generate", headers={"Authorization": "Bearer sk-test-master"}, json=policy
                        )
                        assert response.status_code == 503, response.text
                    response = await client.post(
                        "/key/update",
                        headers={"Authorization": "Bearer sk-test-master"},
                        json={"key": token, "metadata": {field: {"gpt-4o": 0}}},
                    )
                    assert response.status_code == 503, response.text
                assert len(received) == before
                from litellm.proxy.utils import update_spend_logs_job

                for path, body in (
                    ("/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}),
                    ("/v1/responses", {"model": "gpt-4o", "input": "hello"}),
                ):
                    denied = await client.post(
                        path, headers={"Authorization": "Bearer sk-not-issued-" + str(uuid4())}, json=body
                    )
                    assert denied.status_code == 401, denied.text
                    assert len(received) == before
                    rejected = await instance.client.db.query_raw(
                        'SELECT id FROM "LiteLLM_AccountingRequest" WHERE generation_id=$1 ORDER BY created_at DESC LIMIT 1',
                        instance.generation_id,
                    )
                    request_id = rejected[0]["id"]
                    async with asyncio.timeout(5):
                        while True:
                            await update_spend_logs_job(instance.client, None, instance.client.proxy_logging_obj)
                            state = await instance.client.db.query_raw(
                                'SELECT key_token, dispatched, sealed, failed, closed FROM "LiteLLM_AccountingRequest" WHERE id=$1',
                                request_id,
                            )
                            if state[0]["closed"]:
                                break
                            await asyncio.sleep(0.01)
                    assert state == [
                        {"key_token": None, "dispatched": False, "sealed": True, "failed": False, "closed": True}
                    ]
                    assert await instance.client.db.query_raw(
                        'SELECT nonexecution, liability FROM "LiteLLM_AccountingComponent" WHERE request_id=$1',
                        request_id,
                    ) == [{"nonexecution": True, "liability": 0}]
                    diagnostics = await instance.client.db.query_raw(
                        'SELECT l.api_key, l.spend FROM "LiteLLM_AccountingComponent" c '
                        'JOIN "LiteLLM_SpendLogs" l ON l.request_id=c.id WHERE c.request_id=$1',
                        request_id,
                    )
                    assert all(row == {"api_key": "", "spend": 0} for row in diagnostics)
                    assert (
                        await instance.client.db.query_raw(
                            'SELECT amount FROM "LiteLLM_AccountingHold" WHERE request_id=$1', request_id
                        )
                        == []
                    )
    finally:
        accounting_request.reset(context)
        await asyncio.to_thread(provider.shutdown)
        provider.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("zone", [None, "UTC", "America/Los_Angeles"])
async def test_native_timezone_config_and_startup_refusal(tmp_path, monkeypatch, zone):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import postgres_accounting

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setenv("ACCOUNTING_TEST_ZONE", zone or "UTC")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "general_settings:\n  postgres_admission_accounting: true\n"
        + ("litellm_settings:\n  timezone: os.environ/ACCOUNTING_TEST_ZONE\n" if zone else "")
    )
    config = await proxy_server.ProxyConfig().get_config(str(config_path))
    client = SimpleNamespace(db=SimpleNamespace(query_raw=AsyncMock(), execute_raw=AsyncMock()))
    monkeypatch.setattr(litellm, "timezone", zone, raising=False)
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    if zone == "America/Los_Angeles":
        with pytest.raises(ValueError, match="only UTC"):
            validate_config(config)
        with pytest.raises(ValueError, match="only UTC"):
            await PostgresAccounting(client).initialize()
        with pytest.raises(ValueError, match="only UTC"):
            await postgres_accounting.initialize(client, {"postgres_admission_accounting": True}, None)
        assert postgres_accounting.runtime is None
        client.db.query_raw.assert_not_awaited()
        client.db.execute_raw.assert_not_awaited()
    else:
        validate_config(config)
        from litellm.proxy.common_utils.timezone_utils import get_budget_reset_timezone
        from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time

        now = datetime(2026, 9, 13, 12, tzinfo=timezone.utc)
        assert (
            PostgresAccounting.next_reset("1d", now)
            == get_next_standardized_reset_time("1d", now, get_budget_reset_timezone())
            == datetime(2026, 9, 14, tzinfo=timezone.utc)
        )
    await postgres_accounting.initialize(None, {"postgres_admission_accounting": False}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("zone", [None, "UTC"])
async def test_native_timezone_runtime_registration_has_no_rejected_side_effects(protocol, monkeypatch, zone):
    import litellm
    from litellm.proxy.spend_tracking import postgres_accounting

    instance, _, _ = protocol
    generation = str(uuid4())
    monkeypatch.setenv("LITELLM_ACCOUNTING_GENERATION_ID", generation)
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    monkeypatch.setattr(litellm, "timezone", "America/Los_Angeles", raising=False)
    settings = {"postgres_admission_accounting": True, "disable_prisma_schema_update": True}
    with pytest.raises(ValueError, match="only UTC"):
        await postgres_accounting.initialize(instance.client, settings, None)
    assert postgres_accounting.runtime is None
    assert (
        await instance.client.db.query_raw('SELECT id FROM "LiteLLM_AccountingGeneration" WHERE id=$1', generation)
        == []
    )
    assert (
        await instance.client.db.query_raw(
            'SELECT id FROM "LiteLLM_AccountingRequest" WHERE generation_id=$1', generation
        )
        == []
    )
    monkeypatch.setattr(litellm, "timezone", zone)
    await postgres_accounting.initialize(instance.client, settings, None)
    runtime = postgres_accounting.runtime
    assert runtime is not None and runtime.generation_id == generation
    assert (await runtime.status()).pending_requests == 0
    request = await runtime.begin()
    await request.finish()
    await apply_all(runtime)
    assert await runtime.drain() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 0, 1])
async def test_native_router_default_concurrency_and_effective_startup(monkeypatch, limit):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.proxy.spend_tracking import postgres_accounting

    deployment = {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "local-test-key"}}
    config = {"model_list": [deployment], "router_settings": {"default_max_parallel_requests": limit}}
    router = litellm.Router(model_list=[deployment], default_max_parallel_requests=limit)
    semaphore = router._get_client(deployment=router.model_list[0], client_type="max_parallel_requests", kwargs={})
    assert (semaphore is not None) == bool(limit)
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    if limit:
        client = SimpleNamespace(db=SimpleNamespace(query_raw=AsyncMock()))
        with pytest.raises(ValueError, match="concurrency"):
            validate_config(config)
        with pytest.raises(ValueError, match="concurrency"):
            await postgres_accounting.initialize(client, {"postgres_admission_accounting": True}, router)
        client.db.query_raw.assert_not_awaited()
        assert postgres_accounting.runtime is None
    else:
        validate_config(config)
    await postgres_accounting.initialize(None, {"postgres_admission_accounting": False}, router)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,placement",
    [
        (field, "litellm_params")
        for field in ("default_api_key_rpm_limit", "default_api_key_tpm_limit", "rpm", "tpm", "max_parallel_requests")
    ]
    + [(field, placement) for field in ("rpm", "tpm") for placement in ("top", "model_info")],
)
@pytest.mark.parametrize("limit", [0, 1])
async def test_native_deployment_limits_rejected_at_config_and_runtime(monkeypatch, field, placement, limit):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.auth_utils import get_key_model_rpm_limit, get_key_model_tpm_limit
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.spend_tracking import postgres_accounting

    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "local-test-key"},
    }
    if placement == "top":
        deployment[field] = limit
    else:
        deployment.setdefault(placement, {})[field] = limit
    router = litellm.Router(model_list=[deployment])
    monkeypatch.setattr(proxy_server, "llm_router", router)
    if field.startswith("default_api_key"):
        reader = get_key_model_rpm_limit if "rpm" in field else get_key_model_tpm_limit
        assert reader(UserAPIKeyAuth(), "gpt-4o") == {"gpt-4o": limit}
    with pytest.raises(ValueError, match="deployment"):
        validate_config({"model_list": [deployment]})
    client = SimpleNamespace(_spend_log_spool=object(), db=SimpleNamespace(query_raw=AsyncMock()))
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    with pytest.raises(ValueError, match="deployment"):
        await postgres_accounting.initialize(
            client, {"postgres_admission_accounting": True, "disable_prisma_schema_update": True}, router
        )
    client.db.query_raw.assert_not_awaited()
    assert postgres_accounting.runtime is None
    await postgres_accounting.initialize(None, {"postgres_admission_accounting": False}, router)


@pytest.mark.parametrize("groups", [None, [], ["group"]])
def test_native_deployment_access_groups_remain_guarded(groups):
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"access_groups": groups},
    }
    if groups:
        with pytest.raises(ValueError, match="access-group"):
            validate_config({"model_list": [deployment]})
    else:
        validate_config({"model_list": [deployment]})


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["max_parallel_requests", "global_max_parallel_requests"])
@pytest.mark.parametrize("limit", [0, 1])
async def test_native_general_concurrency_refused_before_startup(field, limit):
    from litellm.proxy.spend_tracking import postgres_accounting

    settings = {"postgres_admission_accounting": True, field: limit}
    with pytest.raises(ValueError, match="concurrency|integration pending"):
        validate_config({"general_settings": settings})
    with pytest.raises(ValueError, match="concurrency"):
        await postgres_accounting.initialize(None, settings, None)
    await postgres_accounting.initialize(None, {**settings, "postgres_admission_accounting": False}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["model_rpm_limit", "model_tpm_limit"])
@pytest.mark.parametrize("limit", [None, {}, {"gpt-4o": 0}, {"gpt-4o": 1}])
@pytest.mark.parametrize("placement", ["field", "metadata"])
async def test_native_key_rate_preparation_and_defaults(native_preparation_io, monkeypatch, field, limit, placement):
    import litellm
    from fastapi import HTTPException
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_helper_fn
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.auth_utils import get_key_model_rpm_limit, get_key_model_tpm_limit

    instance, rows, calls = native_preparation_io
    policy = {field: limit} if placement == "field" else {"metadata": {field: limit}}
    values = {"token": "sk-native-rate-test", "table_name": "key", "accounting_auth": team_admin(), **policy}
    if limit:
        with pytest.raises(HTTPException) as error:
            await generate_key_helper_fn(request_type="key", **values)
        assert error.value.status_code == 503
        assert not rows["key"] and not [call for call in calls if call[0] == "write"]
        with pytest.raises(HTTPException):
            validate_config({"litellm_settings": {"default_key_generate_params": policy}})
        monkeypatch.setattr(litellm, "default_key_generate_params", policy)
        with pytest.raises(HTTPException):
            await postgres_accounting.initialize(None, {"postgres_admission_accounting": True}, None)
    else:
        created = await generate_key_helper_fn(request_type="key", **values)
        key = rows["key"][created["token_id"]]
        reader = get_key_model_rpm_limit if "rpm" in field else get_key_model_tpm_limit
        assert reader(UserAPIKeyAuth(**key)) is None
        validate_config({"litellm_settings": {"default_key_generate_params": policy}})
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    off = await generate_key_helper_fn(request_type="key", **{**values, "token": "sk-native-rate-off"})
    assert rows["key"][off["token_id"]]["metadata"].get(field) == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["model_rpm_limit", "model_tpm_limit"])
@pytest.mark.parametrize("limit", [0, 1])
async def test_native_existing_key_rate_update_and_admission_refused(protocol, field, limit):
    from fastapi import HTTPException

    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    with pytest.raises(HTTPException) as error:
        await instance.update_key(key, {"metadata": {field: {"gpt-4o": limit}}})
    assert error.value.status_code == 503
    await instance.client.db.execute_raw(
        'UPDATE "LiteLLM_VerificationToken" SET metadata=$2::jsonb WHERE token=$1',
        key,
        json.dumps({field: {"gpt-4o": limit}}),
    )
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        with pytest.raises(HTTPException) as error:
            await instance.admit(key, 0.01, 0.001, model="gpt-4o")
        assert error.value.status_code == 503
        assert not request.admitted and not request.dispatched
        assert (
            await instance.client.db.query_raw(
                'SELECT request_id FROM "LiteLLM_AccountingHold" WHERE request_id=$1', request.request_id
            )
            == []
        )
        await request.finish()
        await apply_all(instance)
    finally:
        accounting_request.reset(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_models,team_models,model,allowed",
    [
        (["all-team-models"], ["gpt-4o"], "gpt-4o", True),
        (["all-team-models"], ["gpt-4o"], "other", False),
        (["*"], ["gpt-4o"], "gpt-4o", True),
        (["*"], ["other"], "gpt-4o", False),
        (["gpt-*"], ["gpt-4o"], "gpt-4o", True),
        (["all-proxy-models"], ["gpt-4o"], "gpt-4o", True),
        (["gpt-4o"], ["gpt-4o"], "gpt-4o", True),
        (["other"], ["gpt-4o"], "gpt-4o", False),
        (["gpt-4o"], ["gpt-4o"], "alias", True),
        (["all-team-models"], [], "gpt-4o", True),
        (["all-team-models"], ["all-team-models"], "gpt-4o", True),
    ],
)
async def test_native_locked_key_team_models_and_stale_auth(
    protocol, monkeypatch, key_models, team_models, model, allowed
):
    import litellm
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth, ProxyException
    from litellm.proxy.auth.auth_checks import can_key_call_model

    instance, original, key = protocol
    await instance.reject_component(original.component_id)
    await original.finish()
    await apply_all(instance)
    team = str(uuid4())
    await instance.create_team({"team_id": team, "models": ["*"], "admins": [], "members": []}, [], team_admin())
    await instance.update_key(key, {"models": key_models, "team_id": team}, auth=team_admin())
    await instance.update_team(team, {"models": team_models}, team_admin())
    router = litellm.Router(
        model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "local-test-key"}}]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "model_alias_map", {"alias": "gpt-4o"})
    stale = UserAPIKeyAuth(
        token=key, team_id=team, models=["all-team-models"], team_models=["*"], team_model_aliases={"other": "gpt-4o"}
    )
    assert await can_key_call_model(model, None, stale, router)
    request = await instance.begin()
    context = accounting_request.set(request)
    try:
        if allowed:
            await instance.admit(key, 0.01, 0.001, model=model, auth=stale)
            assert request.admitted and request.team_id == team
            await instance.dispatch()
            await instance.reject_component(request.component_id)
        else:
            with pytest.raises(ProxyException) as error:
                await instance.admit(key, 0.01, 0.001, model=model, auth=stale)
            assert int(error.value.code) == 403
            assert not request.admitted and not request.dispatched
        await request.finish()
        await apply_all(instance)
    finally:
        accounting_request.reset(context)
    if allowed:
        await instance.update_team(team, {"models": ["no-longer-allowed"]}, team_admin())
        later = await instance.begin()
        context = accounting_request.set(later)
        try:
            with pytest.raises(ProxyException) as error:
                await instance.admit(key, 0.01, 0.001, model=model, auth=stale)
            assert int(error.value.code) == 403
            assert not later.admitted and not later.dispatched
            await later.finish()
            await apply_all(instance)
        finally:
            accounting_request.reset(context)


@pytest.fixture
def native_preparation_io(monkeypatch):
    from contextlib import asynccontextmanager
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from prisma import models
    from litellm.proxy import proxy_server
    from litellm.proxy.utils import PrismaClient
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.proxy.spend_tracking.postgres_accounting import _native_policy_data

    rows = {"key": {}, "user": {}, "team": {}}
    calls = []
    native_models = {
        "key": models.LiteLLM_VerificationToken,
        "user": models.LiteLLM_UserTable,
        "team": models.LiteLLM_TeamTable,
    }

    async def query(sql, *args):
        calls.append(("query", sql, args))
        if "clock_timestamp" in sql:
            return [{"now": "2026-09-12T12:00:00"}]
        for kind, table in (("key", "VerificationToken"), ("user", "UserTable"), ("team", "TeamTable")):
            if f'"LiteLLM_{table}"' in sql:
                row = rows[kind].get(args[0])
                return [{"data": deepcopy(row)}] if row else []
        if '"LiteLLM_TeamMembership"' in sql or '"LiteLLM_AccountingScope"' in sql:
            return []
        raise AssertionError(sql)

    async def execute(sql, *args):
        calls.append(("execute", sql, args))
        assert sql.startswith('INSERT INTO "LiteLLM_AccountingScope"'), sql
        return 1

    async def write(kind, **kwargs):
        calls.append(("write", kind, deepcopy(kwargs)))
        data = kwargs["data"]
        create = data.get("create", data)
        identity_field = {"key": "token", "user": "user_id", "team": "team_id"}[kind]
        identity = kwargs.get("where", {}).get(identity_field, create.get(identity_field))
        if kind == "key":
            assert "budget_limits" not in create or create["budget_limits"] is not None
            assert kwargs["include"] == {"litellm_budget_table": True}
            assert data["update"] == {}
        if identity in rows[kind]:
            changes = _native_policy_data(data.get("update", data), native_models[kind])
            for name, value in changes.items():
                if name == "teams" and isinstance(value, dict):
                    rows[kind][identity][name] += value["push"]
                else:
                    rows[kind][identity][name] = value
        else:
            rows[kind][identity] = {
                "spend": 0.0,
                "models": [],
                "teams": [],
                **_native_policy_data(create, native_models[kind]),
            }
        return native_models[kind].model_construct(**deepcopy(rows[kind][identity]))

    tables = {
        "litellm_" + name: SimpleNamespace(
            find_first=AsyncMock(return_value=None),
            find_unique=AsyncMock(return_value=None),
            count=AsyncMock(return_value=0),
        )
        for name in ("verificationtoken", "usertable", "teamtable")
    }
    for kind, name in (("key", "verificationtoken"), ("user", "usertable"), ("team", "teamtable")):

        async def leaf(kind=kind, **kw):
            return await write(kind, **kw)

        for operation in ("upsert", "create", "update"):
            setattr(tables["litellm_" + name], operation, AsyncMock(side_effect=leaf))
    db = SimpleNamespace(**tables, query_raw=AsyncMock(side_effect=query), execute_raw=AsyncMock(side_effect=execute))

    @asynccontextmanager
    async def transaction(**kwargs):
        calls.append(("tx", kwargs))
        yield db
        calls.append(("commit",))

    db.tx = transaction
    client = object.__new__(PrismaClient)
    client.db = db
    client.proxy_logging_obj = SimpleNamespace(failure_handler=AsyncMock())
    instance = PostgresAccounting(client)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(postgres_accounting, "runtime", instance)
    return instance, rows, calls


@pytest.mark.asyncio
async def test_native_preparation_public_shapes_and_duplicate(native_preparation_io, monkeypatch):
    from copy import deepcopy
    from starlette.requests import Request
    from litellm.proxy._types import NewTeamRequest, NewUserRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
    from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_helper_fn
    from litellm.proxy.spend_tracking import postgres_accounting
    from litellm.proxy.utils import hash_token

    instance, rows, calls = native_preparation_io
    auth = team_admin().model_copy(update={"user_id": "creator"})
    result = await new_user(NewUserRequest(user_id="creator", max_budget=0, auto_create_key=False), auth)
    assert result.user_id == "creator"
    result = await new_team(
        NewTeamRequest(team_id="team", max_budget=0.1),
        Request({"type": "http", "method": "POST", "path": "/team/new", "headers": []}),
        auth,
    )
    assert isinstance(result, dict) and result["team_id"] == "team"
    assert rows["user"]["creator"]["teams"] == ["team"]
    assert rows["team"]["team"]["admins"] == rows["team"]["team"]["members"] == rows["team"]["team"]["models"] == []
    assert rows["team"]["team"]["members_with_roles"][0]["user_id"] == "creator"
    data = {
        "token": "sk-native-prepared",
        "user_id": "creator",
        "team_id": "team",
        "table_name": "key",
        "accounting_auth": auth,
    }
    result = await generate_key_helper_fn(request_type="key", **data)
    assert isinstance(result, dict) and result["token_id"] == hash_token(data["token"])
    before = deepcopy(rows["key"])
    calls.clear()
    duplicate = await generate_key_helper_fn(request_type="key", **{**data, "spend": 0.7, "metadata": {"other": True}})
    assert duplicate["token_id"] == result["token_id"] and rows["key"] == before
    native_locks = [c[1] for c in calls if c[0] == "query" and "to_jsonb" in c[1]]
    assert '"LiteLLM_VerificationToken"' in native_locks[0]
    assert '"LiteLLM_UserTable"' in native_locks[1]
    assert '"LiteLLM_TeamTable"' in native_locks[2]
    assert len([c for c in calls if c[0] == "tx"]) == 1
    monkeypatch.setattr(postgres_accounting, "runtime", None)
    off = await generate_key_helper_fn(request_type="key", **{**data, "token": "sk-native-default-off"})
    assert off["token_id"] == hash_token("sk-native-default-off")
    assert rows["key"][off["token_id"]]["metadata"] == {}


@pytest.mark.parametrize("kind", ["key", "user", "team"])
@pytest.mark.parametrize("value", [None, {}, "{}", "null"])
def test_native_preparation_policy_shapes(kind, value):
    from prisma import models
    from litellm.proxy.spend_tracking.postgres_accounting import _native_policy_data

    model = {
        "key": models.LiteLLM_VerificationToken,
        "user": models.LiteLLM_UserTable,
        "team": models.LiteLLM_TeamTable,
    }[kind]
    data = {"model_max_budget": value, "metadata": value, "max_budget": 0, "models": [], "user_id": "{}"}
    getattr(PostgresAccounting, "validate_" + kind)(data)
    result = _native_policy_data(data, model)
    assert result["model_max_budget"] == (None if value in (None, "null") else {})
    assert result["user_id"] == "{}" and result["models"] == [] and result["max_budget"] == 0


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("key", "config", '{"model":"other"}'),
        ("key", "router_settings", '{"num_retries":2}'),
        ("key", "budget_limits", '[{"max_budget":1}]'),
        ("key", "access_group_ids", ["unqualified-group"]),
        ("team", "access_group_ids", ["unqualified-group"]),
        ("user", "model_max_budget", '{"model":1}'),
        ("team", "model_max_budget", '{"model":1}'),
        ("team", "metadata", '{"guardrails":[]}'),
        ("team", "router_settings", '{"num_retries":2}'),
        ("team", "model_aliases", {}),
    ],
)
def test_native_preparation_active_policy_still_denied(kind, field, value):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        getattr(PostgresAccounting, "validate_" + kind)({field: value})
    assert error.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["blocked", "expired", "owner", "role", "source", "destination", "ceiling", "authorized"]
)
async def test_native_preparation_duplicate_fresh_authority(native_preparation_io, failure):
    from copy import deepcopy
    from fastapi import HTTPException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import hash_token

    instance, rows, calls = native_preparation_io
    token = hash_token("sk-target")
    actor_token = hash_token("sk-actor")
    rows["key"][actor_token] = {"token": actor_token, "user_id": "actor", "max_budget": 10}
    rows["user"]["actor"] = {"user_id": "actor", "user_role": "proxy_admin", "spend": 0}
    rows["key"][token] = {
        "token": token,
        "user_id": "original",
        "team_id": "source",
        "spend": 0.03,
        "metadata": {"original": True},
        "budget_duration": "1s",
        "budget_reset_at": "2020-01-01T00:00:00",
    }
    for team in ("source", "destination"):
        rows["team"][team] = {
            "team_id": team,
            "spend": 0,
            "members_with_roles": [{"user_id": "actor", "role": "admin"}],
        }
    if failure == "blocked":
        rows["key"][actor_token]["blocked"] = True
    if failure == "expired":
        rows["key"][actor_token]["expires"] = "2020-01-01T00:00:00"
    if failure == "owner":
        rows["key"][actor_token]["user_id"] = "other"
    if failure in ("role", "source", "destination", "ceiling"):
        rows["user"]["actor"]["user_role"] = "internal_user"
    if failure == "role":
        rows["key"][token]["team_id"] = None
    if failure in ("source", "destination"):
        rows["team"][failure]["members_with_roles"] = []
    auth = UserAPIKeyAuth(token=actor_token, user_id="actor", user_role="proxy_admin")
    before = deepcopy(rows)
    data = {
        "token": "sk-target",
        "user_id": "actor",
        "team_id": "destination",
        "max_budget": 11 if failure == "ceiling" else 1,
    }
    if failure == "authorized":
        result = await instance.create_key(data, auth)
        assert result.token == token and result.spend == 0.03
    else:
        with pytest.raises(HTTPException):
            await instance.create_key(data, auth)
        assert not [c for c in calls if c[0] == "write"]
    assert rows == before


@pytest.mark.asyncio
async def test_native_preparation_user_update_serialized_policy(native_preparation_io):
    instance, rows, _ = native_preparation_io
    await instance.client.insert_data(
        {"user_id": "user", "models": [], "model_max_budget": "{}", "metadata": {"original": True}}, "user"
    )
    result = await instance.client.update_data(
        user_id="user", data={"user_id": "user", "model_max_budget": "{}", "max_budget": 0}, table_name="user"
    )
    assert result["data"].user_id == "user" and rows["user"]["user"]["max_budget"] == 0
    assert rows["user"]["user"]["metadata"] == {"original": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_limits", [None, "[]"])
@pytest.mark.parametrize("serialized", [False, True])
async def test_native_preparation_shared_leaf_preserves_input(native_preparation_io, budget_limits, serialized):
    import json
    from copy import deepcopy
    from prisma import models
    from litellm.proxy.utils import hash_token
    from litellm.proxy.spend_tracking.postgres_accounting import _native_policy_data

    instance, rows, calls = native_preparation_io
    data = {
        "token": "sk-once",
        "models": [],
        "aliases": {},
        "config": {},
        "permissions": {},
        "metadata": {"native": True},
        "model_max_budget": {},
        "router_settings": {},
        "budget_limits": budget_limits,
        "project_id": None,
    }
    if serialized:
        data = {name: json.dumps(value) if isinstance(value, dict) else value for name, value in data.items()}
    before = deepcopy(data)
    prepared = instance.client.prepare_key_insert(data)
    assert prepared["token"] == hash_token(data["token"]) and data == before
    if budget_limits is None:
        assert "budget_limits" not in prepared
    else:
        assert prepared["budget_limits"] == budget_limits
    row = await instance.client.insert_key(prepared, db=instance.client.db)
    assert row.token == hash_token(data["token"])
    assert row.metadata == {"native": True} and row.project_id is None
    assert _native_policy_data(prepared, models.LiteLLM_VerificationToken)["models"] == []
    assert not [c for c in calls if c[0] == "tx"]


@pytest.mark.asyncio
async def test_native_preparation_team_update_and_pre_side_effect_guard(native_preparation_io):
    from copy import deepcopy
    from fastapi import HTTPException
    from litellm.proxy._types import LiteLLM_TeamTable

    instance, rows, calls = native_preparation_io
    prepared = instance.client.jsonify_team_object(
        LiteLLM_TeamTable(team_id="team", max_budget=0.1).model_dump(exclude_none=True)
    )
    await instance.create_team(prepared, [], team_admin())
    result = await instance.update_team("team", {"metadata": "{}", "models": [], "max_budget": 0}, team_admin())
    assert result.team_id == "team" and result.max_budget == 0 and result.models == []
    assert rows["team"]["team"]["spend"] == 0
    calls.clear()
    before = deepcopy(rows)
    with pytest.raises(HTTPException):
        await instance.create_team({**prepared, "team_id": "denied", "model_aliases": {}}, [], team_admin())
    assert rows == before and not calls
