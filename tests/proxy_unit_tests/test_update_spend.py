import asyncio
import os
import sys
from unittest.mock import Mock
from litellm.proxy.utils import _get_redoc_url, _get_docs_url

import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from unittest.mock import MagicMock, patch, AsyncMock


import httpx
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import (
    DB_CONNECTION_ERROR_TYPES,
    ProxyUpdateSpend,
    _spend_log_batch_prefix,
    drain_spend_log_queue,
    update_spend,
    update_spend_logs_job,
)
from litellm.proxy.db.spend_log_queue import (
    SQLiteSpendLogSpool,
    enqueue_spend_log,
    spend_log_queue_stats,
)


class MockPrismaClient:
    def __init__(self):
        # Create AsyncMock for db operations
        self.db = AsyncMock()
        self.db.litellm_spendlogs = AsyncMock()
        self.db.litellm_spendlogs.create_many = AsyncMock()

        # Initialize transaction lists
        self.spend_log_transactions = []
        self.daily_user_spend_transactions = {}

        # Add lock for spend_log_transactions (matches real PrismaClient)
        import asyncio

        self._spend_log_transactions_lock = asyncio.Lock()
        self._spend_log_write_lock = asyncio.Lock()
        self._spend_log_spool = None

    def jsonify_object(self, obj):
        return obj

    def add_spend_log_transaction_to_daily_user_transaction(self, payload):
        # Mock implementation
        pass


def create_mock_proxy_logging():
    print("creating mock proxy logging")
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = (
        AsyncMock()
    )
    print("returning proxy logging obj")
    return proxy_logging_obj


def test_spend_log_batch_prefix_respects_bytes_and_always_makes_progress():
    logs = [
        {"request_id": "first", "payload": "a" * 100},
        {"request_id": "second", "payload": "b" * 100},
    ]

    first_size = len(safe_dumps(logs[0]).encode("utf-8"))
    batch, batch_bytes = _spend_log_batch_prefix(
        logs,
        max_count=10,
        max_bytes=first_size + 2,
    )
    assert batch == [logs[0]]
    assert batch_bytes >= first_size

    oversized, _ = _spend_log_batch_prefix(logs, max_count=10, max_bytes=1)
    assert oversized == [logs[0]]


@pytest.mark.asyncio
async def test_durable_spend_log_spool_survives_process_recreation(tmp_path):
    spool_path = tmp_path / "spend-queue.sqlite3"
    first_process = SQLiteSpendLogSpool(str(spool_path))
    await first_process.enqueue({"request_id": "persisted", "spend": 1.25})

    second_process = SQLiteSpendLogSpool(str(spool_path))
    stats = await second_process.stats()
    batch = await second_process.peek_batch(max_count=10, max_bytes=1024 * 1024)

    assert stats.count == 1
    assert batch.logs == [{"request_id": "persisted", "spend": 1.25}]
    await second_process.acknowledge(batch.durable_row_ids or [])
    assert (await second_process.stats()).count == 0


@pytest.mark.asyncio
async def test_durable_spend_log_spool_group_commits_concurrent_producers(tmp_path):
    spool = SQLiteSpendLogSpool(str(tmp_path / "spend-queue.sqlite3"))

    await asyncio.gather(*(spool.enqueue({"request_id": str(index)}) for index in range(50)))

    batch = await spool.peek_batch(max_count=100, max_bytes=1024 * 1024)
    assert [row["request_id"] for row in batch.logs] == [str(index) for index in range(50)]


@pytest.mark.asyncio
async def test_scheduled_writer_acknowledges_durable_rows_only_after_db_success(tmp_path):
    prisma_client = MockPrismaClient()
    prisma_client._spend_log_spool = SQLiteSpendLogSpool(str(tmp_path / "spend-queue.sqlite3"))
    proxy_logging_obj = create_mock_proxy_logging()
    await enqueue_spend_log(prisma_client, {"request_id": "durable"})

    await update_spend_logs_job(prisma_client, None, proxy_logging_obj)

    assert prisma_client.db.litellm_spendlogs.create_many.await_count == 1
    assert (await spend_log_queue_stats(prisma_client)).count == 0


@pytest.mark.asyncio
async def test_scheduled_writer_retains_durable_rows_and_reduces_batch_after_timeout(tmp_path):
    prisma_client = MockPrismaClient()
    prisma_client._spend_log_spool = SQLiteSpendLogSpool(str(tmp_path / "spend-queue.sqlite3"))
    prisma_client._spend_log_batch_max_count = 1000
    prisma_client._spend_log_batch_max_bytes = 4 * 1024 * 1024
    proxy_logging_obj = create_mock_proxy_logging()
    await enqueue_spend_log(prisma_client, {"request_id": "durable", "payload": "x" * 1024})
    prisma_client.db.litellm_spendlogs.create_many = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(httpx.ReadTimeout):
        await update_spend_logs_job(prisma_client, None, proxy_logging_obj)

    assert (await spend_log_queue_stats(prisma_client)).count == 1
    assert prisma_client._spend_log_batch_max_count == 500
    assert prisma_client._spend_log_batch_max_bytes == 2 * 1024 * 1024

    prisma_client.db.litellm_spendlogs.create_many = AsyncMock(return_value=None)
    await update_spend_logs_job(prisma_client, None, proxy_logging_obj)

    assert (await spend_log_queue_stats(prisma_client)).count == 0
    assert prisma_client._spend_log_batch_max_count == 625


@pytest.mark.asyncio
async def test_update_spend_logs_job_serializes_concurrent_writers():
    """The interval job and queue monitor must not write spend logs concurrently."""
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()
    prisma_client.spend_log_transactions = [{"request_id": "first"}]

    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    active_writes = 0
    max_active_writes = 0
    written_request_ids = []

    async def create_many_side_effect(**kwargs):
        nonlocal active_writes, max_active_writes

        request_id = kwargs["data"][0]["request_id"]
        active_writes += 1
        max_active_writes = max(max_active_writes, active_writes)
        written_request_ids.append(request_id)
        if request_id == "first":
            first_write_started.set()
            await release_first_write.wait()
        active_writes -= 1

    prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=create_many_side_effect
    )

    first_job = asyncio.create_task(
        update_spend_logs_job(prisma_client, None, proxy_logging_obj)
    )
    await first_write_started.wait()

    async with prisma_client._spend_log_transactions_lock:
        prisma_client.spend_log_transactions.append({"request_id": "second"})

    second_job = asyncio.create_task(
        update_spend_logs_job(prisma_client, None, proxy_logging_obj)
    )
    await asyncio.sleep(0)

    assert max_active_writes == 1
    assert written_request_ids == ["first"]

    release_first_write.set()
    await asyncio.gather(first_job, second_job)

    assert max_active_writes == 1
    assert written_request_ids == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_connection_errors(error_type):
    """The scheduled writer requeues transport errors for a later flush."""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create AsyncMock for db_spend_update_writer
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = (
        AsyncMock()
    )

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    create_many_mock = AsyncMock(side_effect=error_type)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    with pytest.raises(type(error_type)):
        await update_spend(prisma_client, None, proxy_logging_obj)

    assert create_many_mock.call_count == 1
    assert len(prisma_client.spend_log_transactions) == 2


@pytest.mark.asyncio
async def test_scheduled_writer_requeues_only_the_byte_bounded_batch():
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()
    prisma_client.spend_log_transactions = [
        {"request_id": "first", "payload": "a" * (3 * 1024 * 1024)},
        {"request_id": "second", "payload": "b" * (3 * 1024 * 1024)},
    ]
    create_many_mock = AsyncMock(side_effect=httpx.ReadTimeout("Request timed out"))
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    with pytest.raises(httpx.ReadTimeout):
        await update_spend_logs_job(prisma_client, None, proxy_logging_obj)

    attempted = create_many_mock.call_args.kwargs["data"]
    assert [entry["request_id"] for entry in attempted] == ["first"]
    assert [entry["request_id"] for entry in prisma_client.spend_log_transactions] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_max_retries_exceeded(error_type):
    """Test that each connection error type properly fails after max retries"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock the database to always fail
    create_many_mock = AsyncMock(side_effect=error_type)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Direct callers can still request inline retries explicitly.
    with pytest.raises(type(error_type)) as exc_info:
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=3,
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    # Verify error message matches
    assert str(exc_info.value) == str(error_type)
    # Verify retry attempts (initial try + 4 retries)
    assert create_many_mock.call_count == 4
    assert len(prisma_client.spend_log_transactions) == 2

    await asyncio.sleep(2)
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.call_count == 1


@pytest.mark.asyncio
async def test_update_spend_logs_non_connection_error():
    """Test handling of non-connection related errors"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock a different type of error (not connection-related)
    unexpected_error = ValueError("Unexpected database error")
    create_many_mock = AsyncMock(side_effect=unexpected_error)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute and verify it raises immediately without retrying
    with pytest.raises(ValueError) as exc_info:
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify error message
    assert str(exc_info.value) == "Unexpected database error"
    # Verify only tried once (no retries for non-connection errors)
    assert create_many_mock.call_count == 1
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.called


@pytest.mark.asyncio
async def test_update_spend_logs_exponential_backoff():
    """Test that exponential backoff is working correctly"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [{"id": "1", "spend": 10}]

    # Track sleep times
    sleep_times = []

    # Mock asyncio.sleep to track delay times
    async def mock_sleep(seconds):
        sleep_times.append(seconds)

    # Mock the database to fail with connection errors
    create_many_mock = AsyncMock(
        side_effect=[
            httpx.ConnectError("Failed to connect"),  # First attempt
            httpx.ConnectError("Failed to connect"),  # Second attempt
            None,  # Third attempt succeeds
        ]
    )

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Apply mocks
    with patch("asyncio.sleep", mock_sleep):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=3,
            prisma_client=prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )

    # Verify exponential backoff
    assert len(sleep_times) == 2  # Should have slept twice
    assert (
        sleep_times[0] >= 1 and sleep_times[0] <= 2
    )  # First retry after 2^0~2^1 seconds
    assert (
        sleep_times[1] >= 2 and sleep_times[1] <= 4
    )  # Second retry after 2^1~2^2 seconds


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_success():
    """
    Test successful processing of multiple batches of spend logs

    Code sets batch size to 1000. This test creates 1500 logs, so it should make 2 batches.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 1500 test spend logs (1.5x BATCH_SIZE)
    prisma_client.spend_log_transactions = [
        {"id": str(i), "spend": 10} for i in range(1500)
    ]

    create_many_mock = AsyncMock(return_value=None)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Each scheduled flush handles one bounded batch.
    await update_spend(prisma_client, None, proxy_logging_obj)
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == 2  # Should have made 2 batch calls

    # Get the actual data from each batch call
    first_batch = create_many_mock.call_args_list[0][1]["data"]
    second_batch = create_many_mock.call_args_list[1][1]["data"]

    # Verify batch sizes
    assert len(first_batch) == 1000
    assert len(second_batch) == 500

    # Verify exact IDs in each batch
    expected_first_batch_ids = {str(i) for i in range(1000)}
    expected_second_batch_ids = {str(i) for i in range(1000, 1500)}

    actual_first_batch_ids = {item["id"] for item in first_batch}
    actual_second_batch_ids = {item["id"] for item in second_batch}

    assert actual_first_batch_ids == expected_first_batch_ids
    assert actual_second_batch_ids == expected_second_batch_ids

    # Verify all logs were processed
    assert len(prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_with_failure():
    """
    Test processing of multiple batches where one batch fails.
    Creates 4000 logs (4 batches) with one batch failing but eventually succeeding after retry.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 4000 test spend logs (4x BATCH_SIZE)
    logs_to_process = [{"id": str(i), "spend": 10} for i in range(4000)]
    prisma_client.spend_log_transactions = []

    # Mock to fail on second batch first attempt, then succeed
    call_count = 0

    async def create_many_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on the second batch's first attempt
        if call_count == 2:
            raise httpx.ConnectError("Failed to connect")
        return None

    create_many_mock = AsyncMock(side_effect=create_many_side_effect)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Exercise the lower-level helper's explicit inline-retry contract. The
    # scheduled job deliberately passes zero retries and requeues instead.
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=3,
        prisma_client=prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging_obj,
        logs_to_process=logs_to_process,
    )

    # Verify
    assert create_many_mock.call_count == 6  # 4 batches + 2 retries for failed batch

    # Verify all batches were processed
    all_processed_logs = []
    for call in create_many_mock.call_args_list:
        all_processed_logs.extend(call[1]["data"])

    # Verify all IDs were processed
    processed_ids = {item["id"] for item in all_processed_logs}

    # these should have ids 0-3999
    print("all processed ids", sorted(processed_ids, key=int))
    expected_ids = {str(i) for i in range(4000)}
    assert processed_ids == expected_ids

    # Verify all logs were cleared from transactions
    assert len(prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_graceful_shutdown_drains_all_spend_batches():
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()
    prisma_client.spend_log_transactions = [{"request_id": str(i), "payload": "x" * 1000} for i in range(1200)]

    remaining = await drain_spend_log_queue(
        prisma_client=prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging_obj,
        timeout_seconds=5,
    )

    assert remaining == 0
    assert len(prisma_client.spend_log_transactions) == 0
    assert prisma_client.db.litellm_spendlogs.create_many.call_count == 2
