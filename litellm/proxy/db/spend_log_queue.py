"""Crash-durable buffering for raw spend-log writes.

The SQLite spool is intentionally local to one proxy process/container. It is
not a reporting database: rows stay here only until PostgreSQL accepts them.
"""

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


@dataclass(frozen=True)
class SpendLogQueueBatch:
    logs: List[Dict[str, Any]]
    serialized_bytes: int
    durable_row_ids: Optional[List[int]] = None

    @property
    def is_durable(self) -> bool:
        return self.durable_row_ids is not None


@dataclass(frozen=True)
class SpendLogQueueStats:
    count: int
    serialized_bytes: int = 0
    oldest_age_seconds: Optional[float] = None


class SQLiteSpendLogSpool:
    """A small WAL-backed FIFO that survives proxy restarts and crashes."""

    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self._operation_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._pending_enqueues: List[Tuple[str, int, asyncio.Future[None]]] = []
        self._enqueue_flush_task: Optional[asyncio.Task[None]] = None
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS spend_log_spool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
        os.chmod(self.path, 0o600)

    async def enqueue(self, payload: Dict[str, Any]) -> None:
        serialized = safe_dumps(payload)
        payload_bytes = len(serialized.encode("utf-8"))
        future = asyncio.get_running_loop().create_future()
        async with self._pending_lock:
            self._pending_enqueues.append((serialized, payload_bytes, future))
            if self._enqueue_flush_task is None:
                self._enqueue_flush_task = asyncio.create_task(self._flush_pending_enqueues())
        await asyncio.shield(future)

    async def _flush_pending_enqueues(self) -> None:
        # Let concurrent callbacks join the same durable transaction. Each
        # caller still waits until its own row is committed before returning.
        await asyncio.sleep(0)
        while True:
            async with self._pending_lock:
                pending = self._pending_enqueues
                self._pending_enqueues = []
                if not pending:
                    self._enqueue_flush_task = None
                    return

            try:
                async with self._operation_lock:
                    await asyncio.to_thread(
                        self._enqueue_many_sync,
                        [(serialized, payload_bytes) for serialized, payload_bytes, _ in pending],
                    )
            except Exception as error:
                for _, _, future in pending:
                    if not future.done():
                        future.set_exception(error)
            else:
                for _, _, future in pending:
                    if not future.done():
                        future.set_result(None)

    def _enqueue_many_sync(self, rows: List[Tuple[str, int]]) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO spend_log_spool(payload, payload_bytes, created_at) VALUES (?, ?, ?)",
                [(serialized, payload_bytes, now) for serialized, payload_bytes in rows],
            )

    async def peek_batch(self, max_count: int, max_bytes: int) -> SpendLogQueueBatch:
        async with self._operation_lock:
            return await asyncio.to_thread(self._peek_batch_sync, max_count, max_bytes)

    def _peek_batch_sync(self, max_count: int, max_bytes: int) -> SpendLogQueueBatch:
        logs: List[Dict[str, Any]] = []
        row_ids: List[int] = []
        serialized_bytes = 2

        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, payload, payload_bytes FROM spend_log_spool ORDER BY id LIMIT ?",
                (max_count,),
            )
            for row_id, serialized, payload_bytes in cursor:
                candidate_bytes = serialized_bytes + int(payload_bytes) + (1 if logs else 0)
                if logs and candidate_bytes > max_bytes:
                    break
                payload = json.loads(serialized)
                if not isinstance(payload, dict):
                    raise ValueError(f"Spend-log spool row {row_id} is not a JSON object")
                row_ids.append(int(row_id))
                logs.append(payload)
                serialized_bytes = candidate_bytes

        return SpendLogQueueBatch(
            logs=logs,
            serialized_bytes=serialized_bytes,
            durable_row_ids=row_ids,
        )

    async def acknowledge(self, row_ids: Sequence[int]) -> None:
        if not row_ids:
            return
        async with self._operation_lock:
            await asyncio.to_thread(self._acknowledge_sync, list(row_ids))

    def _acknowledge_sync(self, row_ids: List[int]) -> None:
        with self._connect() as connection:
            for offset in range(0, len(row_ids), 500):
                chunk = row_ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM spend_log_spool WHERE id IN ({placeholders})",
                    chunk,
                )

    async def stats(self) -> SpendLogQueueStats:
        async with self._operation_lock:
            return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> SpendLogQueueStats:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0), MIN(created_at) FROM spend_log_spool"
            ).fetchone()
        count = int(row[0]) if row is not None else 0
        oldest_at = float(row[2]) if row is not None and row[2] is not None else None
        return SpendLogQueueStats(
            count=count,
            serialized_bytes=int(row[1]) if row is not None else 0,
            oldest_age_seconds=max(0.0, time.time() - oldest_at) if oldest_at is not None else None,
        )


def create_spend_log_spool_from_env() -> Optional[SQLiteSpendLogSpool]:
    path = os.getenv("SPEND_LOG_DURABLE_QUEUE_PATH", "").strip()
    if not path:
        return None
    spool = SQLiteSpendLogSpool(path)
    verbose_proxy_logger.info("Spend tracking - durable queue enabled at %s", spool.path)
    return spool


async def enqueue_spend_log(prisma_client: Any, payload: Dict[str, Any]) -> None:
    """Persist a row, falling back to memory if the local spool is unavailable."""
    spool = getattr(prisma_client, "_spend_log_spool", None)
    if spool is not None:
        try:
            await spool.enqueue(payload)
            return
        except Exception as error:
            verbose_proxy_logger.error(
                "Spend tracking - durable enqueue failed; retaining row in memory. error=%s",
                error,
            )

    async with prisma_client._spend_log_transactions_lock:
        prisma_client.spend_log_transactions.append(payload)


async def spend_log_queue_stats(prisma_client: Any) -> SpendLogQueueStats:
    async with prisma_client._spend_log_transactions_lock:
        memory_count = len(prisma_client.spend_log_transactions)
    spool = getattr(prisma_client, "_spend_log_spool", None)
    if spool is None:
        return SpendLogQueueStats(count=memory_count)
    try:
        durable = await spool.stats()
    except Exception as error:
        verbose_proxy_logger.error(
            "Spend tracking - durable queue stats failed; reporting memory fallback only. error=%s",
            error,
        )
        return SpendLogQueueStats(count=memory_count)
    return SpendLogQueueStats(
        count=memory_count + durable.count,
        serialized_bytes=durable.serialized_bytes,
        oldest_age_seconds=durable.oldest_age_seconds,
    )


async def take_spend_log_batch(
    prisma_client: Any,
    max_count: int,
    max_bytes: int,
) -> SpendLogQueueBatch:
    """Take an in-memory fallback batch, otherwise peek the durable FIFO."""
    async with prisma_client._spend_log_transactions_lock:
        candidates = prisma_client.spend_log_transactions[:max_count]
        batch: List[Dict[str, Any]] = []
        serialized_bytes = 2
        for entry in candidates:
            entry_bytes = len(safe_dumps(entry).encode("utf-8"))
            candidate_bytes = serialized_bytes + entry_bytes + (1 if batch else 0)
            if batch and candidate_bytes > max_bytes:
                break
            batch.append(entry)
            serialized_bytes = candidate_bytes
        if batch:
            prisma_client.spend_log_transactions = prisma_client.spend_log_transactions[len(batch) :]
            return SpendLogQueueBatch(logs=batch, serialized_bytes=serialized_bytes)

    spool = getattr(prisma_client, "_spend_log_spool", None)
    if spool is None:
        return SpendLogQueueBatch(logs=[], serialized_bytes=2)
    return await spool.peek_batch(max_count=max_count, max_bytes=max_bytes)


async def acknowledge_spend_log_batch(prisma_client: Any, batch: SpendLogQueueBatch) -> None:
    if not batch.is_durable:
        return
    spool = getattr(prisma_client, "_spend_log_spool", None)
    if spool is None:
        raise RuntimeError("Durable spend-log queue disappeared before acknowledgement")
    await spool.acknowledge(batch.durable_row_ids or [])


async def release_spend_log_batch(prisma_client: Any, batch: SpendLogQueueBatch) -> None:
    """Return memory rows; durable rows remain in place until acknowledged."""
    if batch.is_durable or not batch.logs:
        return
    async with prisma_client._spend_log_transactions_lock:
        prisma_client.spend_log_transactions = [
            *batch.logs,
            *prisma_client.spend_log_transactions,
        ]
