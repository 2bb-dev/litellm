from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Coroutine, Generic, Optional, Protocol, TypeVar
from uuid import uuid4

from pydantic import JsonValue


class RequestStore(Protocol):
    async def seal(self, request: AccountingRequest) -> None: ...


@dataclass
class AccountingRequest:
    store: RequestStore
    request_id: str = field(default_factory=lambda: str(uuid4()))
    component_id: str = field(default_factory=lambda: str(uuid4()))
    expected: dict[str, str] = field(default_factory=dict)
    observations: dict[str, str] = field(default_factory=dict)
    producers: int = 1
    failed: bool = False
    dispatched: bool = False
    reserved_cost: float = 0.0
    last_component_id: Optional[str] = None
    admitted: bool = False
    input_cost: float = 0.0
    token: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    key_metadata: dict[str, JsonValue] = field(default_factory=dict)

    def acquire(self) -> None:
        if self.producers == 0:
            self.failed = True
            raise RuntimeError("Accounting producer created after sealing")
        self.producers += 1

    async def finish(self, failed: bool = False) -> None:
        if self.producers <= 0:
            raise RuntimeError("Accounting producer finished more than once")
        self.failed = self.failed or failed
        self.producers -= 1
        if self.producers == 0:
            task = asyncio.create_task(self.store.seal(self))
            _finalizers.add(task)
            task.add_done_callback(_finalizers.discard)
            await asyncio.shield(task)


@dataclass
class AccountingCall:
    component_id: Optional[str] = None
    call_type: str = "acompletion"


accounting_call: ContextVar[Optional[AccountingCall]] = ContextVar("accounting_call", default=None)
accounting_request: ContextVar[Optional[AccountingRequest]] = ContextVar("accounting_request", default=None)
T = TypeVar("T")
_finalizers: set[asyncio.Task[None]] = set()


class AccountingProducer(Generic[T]):
    def __init__(self, coroutine: Coroutine[object, object, T], request: AccountingRequest):
        self.coroutine = coroutine
        self.request = request
        self.started = False
        request.acquire()

    async def run(self) -> T:
        self.started = True
        failed = True
        try:
            result = await self.coroutine
            failed = False
            return result
        finally:
            await self.request.finish(failed=failed)

    def discard(self) -> None:
        if self.started:
            return
        self.started = True
        self.coroutine.close()
        self.request.failed = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Without a running owner, the acquired durable obligation stays open.
            return
        task = loop.create_task(self.request.finish(failed=True))
        _finalizers.add(task)
        task.add_done_callback(_finalizers.discard)


def spawn_accounting(coroutine: Coroutine[object, object, T]) -> asyncio.Task[T]:
    request = accounting_request.get()
    if request is None:
        return asyncio.create_task(coroutine)
    producer = AccountingProducer(coroutine, request)
    tracked = producer.run()
    try:
        task = asyncio.create_task(tracked)
    except BaseException:
        tracked.close()
        producer.discard()
        raise
    task.add_done_callback(lambda _: producer.discard())
    return task
