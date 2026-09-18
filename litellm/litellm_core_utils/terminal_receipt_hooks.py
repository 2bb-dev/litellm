"""Private Router/dispatch lifecycle for terminal authority receipts."""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import InstanceOf, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.credential_ownership import (
    STAMP as OWNERSHIP_STAMP,
)
from litellm.litellm_core_utils.credential_ownership import (
    ownership_for_spend,
    planned_ownership_for_terminal,
)
from litellm.litellm_core_utils.terminal_receipt_client import Authority, ReceiptPending, ReceiptUnavailable
from litellm.litellm_core_utils.terminal_receipt_evidence import (
    CONTEXT,
    FIELD,
    STAMP,
    Attempt,
    Envelope,
    Ingress,
    Seal,
    Terminal,
    UUIDText,
    authenticated,
    parse_jws,
)
from litellm.types.utils import ModelResponseStream

if TYPE_CHECKING:
    from litellm.litellm_core_utils.terminal_receipt_oauth import AccountSnapshot
    from litellm.litellm_core_utils.terminal_usage_evidence import UsageEnvelope
    from litellm.litellm_core_utils.terminal_usage_observation import UsageSnapshot

ROOT = "_openorange_terminal_root"
INGRESS_HEADER = "x-openorange-terminal-ingress"
CENTRAL_HEADER = "x-openorange-terminal-central-request"
CALL_CONTEXT = TypeAdapter(InstanceOf[dict[str, object]])
OPAQUE_VALUE = TypeAdapter(object)
_OBJECT = TypeAdapter(dict[str, object])
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_UUID: TypeAdapter[str] = TypeAdapter(UUIDText)
_MODEL: TypeAdapter[str | None] = TypeAdapter(str | None)


def mapping(value: object) -> dict[str, object]:
    try:
        return _OBJECT.validate_python(value)
    except ValidationError:
        return {}


def scrub_headers(kwargs: dict[str, object]) -> None:
    for key in ("headers", "extra_headers"):
        if key in kwargs:
            kwargs[key] = {
                name: value
                for name, value in mapping(kwargs[key]).items()
                if name.lower() not in {INGRESS_HEADER, CENTRAL_HEADER}
            }


@dataclass(slots=True)
class Root:
    authority: Authority
    local_request_id: str
    central_request_id: str
    ingress: str | None
    delegated: bool = False
    claim: Ingress | None = None
    depth: int = 1
    sealed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __deepcopy__(self, memo: object) -> "Root":
        return self

    def claim_ingress(self) -> Ingress:
        with self.lock:
            if self.claim is not None:
                return self.claim
            if self.ingress is None:
                raise ReceiptUnavailable()
            parsed = self.authority.ingress(self.ingress)
            accepted = self.authority.request(
                "POST",
                "/claims/delegated" if self.delegated else "/claims",
                {
                    "ingress_jws": self.ingress,
                    "central_request_id": self.central_request_id,
                },
            )
            if accepted != {
                "correlation_id": parsed.correlation_id,
                "central_request_id": self.central_request_id,
                "producer_id": self.authority.settings.identity,
            }:
                raise ReceiptUnavailable()
            self.claim = parsed
            return parsed

    def leave(self) -> None:
        with self.lock:
            self.depth = max(0, self.depth - 1)
        self.try_seal()

    def try_seal(self) -> None:
        with self.lock:
            if (
                self.depth
                or self.sealed
                or self.claim is None
                or self.authority.settings.role != "producer"
                or self.delegated
            ):
                return
            deadline = time.monotonic() + self.authority.settings.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    result = self.authority.request(
                        "POST",
                        "/correlations/complete",
                        {
                            "correlation_id": self.claim.correlation_id,
                            "central_request_id": self.central_request_id,
                        },
                    )
                    token = result.get("receipt_jws")
                    parsed = parse_jws(token) if isinstance(token, str) else None
                    self.sealed = (
                        set(result) == {"receipt_jws"}
                        and parsed is not None
                        and isinstance(parsed.claims, Seal)
                        and authenticated(parsed, self.authority.settings.issuer, self.authority.keys)
                        and parsed.claims.correlation_id == self.claim.correlation_id
                        and parsed.claims.central_request_id == self.central_request_id
                        and parsed.claims.producer_id == self.authority.settings.identity
                        and parsed.claims.aud == self.claim.aud
                        and parsed.claims.local_request_id == self.claim.local_request_id
                        and parsed.claims.local_attempt_id == self.claim.local_attempt_id
                        and parsed.claims.local_deployment_id == self.claim.local_deployment_id
                    )
                    return
                except ReceiptPending:
                    time.sleep(0.02)
                except ReceiptUnavailable:
                    return


@dataclass(slots=True)
class Session:
    root: Root
    attempt_id: str
    terminal: Terminal
    base: str | None
    envelope: Envelope | None = None
    usage_envelope: "UsageEnvelope | None" = None
    begun: bool = False
    finished: bool = False
    relay: bool = False
    failure_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    local_usage_snapshot: "UsageSnapshot | None" = None
    usage_snapshot: "UsageSnapshot | None" = None
    native_usage_fields: dict[str, object] = field(default_factory=dict, repr=False)
    native_usage_final: bool = False
    usage_final: bool = False
    usage_invalid: bool = False
    failure_persisted: bool = False
    failure_updates_queued: bool = False
    oauth: "AccountSnapshot | None" = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __deepcopy__(self, memo: object) -> "Session":
        return self


def enter_router(kwargs: dict[str, object], authority: Authority | None) -> Root | None:
    if authority is None:
        return None
    metadata = mapping(kwargs.get("metadata"))
    existing = metadata.get(ROOT)
    if type(existing) is Root and existing.authority is authority:
        with existing.lock:
            existing.depth += 1
        return existing
    proxy_headers = mapping(mapping(kwargs.get("proxy_server_request")).get("headers"))
    metadata_headers = mapping(metadata.get("headers"))
    headers = {**proxy_headers, **metadata_headers, **mapping(kwargs.get("headers"))}
    ingress = next((value for name, value in headers.items() if name.lower() == INGRESS_HEADER), None)
    delegated_request = (
        next((value for name, value in headers.items() if name.lower() == CENTRAL_HEADER), None)
        if authority.settings.role == "producer"
        else None
    )
    if authority.settings.role == "producer" and not isinstance(ingress, str):
        return None
    central_request = _UUID.validate_python(delegated_request) if delegated_request is not None else str(uuid4())
    root = Root(
        authority,
        str(uuid4()),
        central_request,
        ingress if isinstance(ingress, str) else None,
        delegated_request is not None,
    )
    kwargs["metadata"] = {**metadata, ROOT: root}
    return root


def select_attempt(
    metadata: dict[str, object], model: object, deployment_id: object, base: object
) -> dict[str, object]:
    root = metadata.get(ROOT)
    if type(root) is not Root or not isinstance(model, str) or not isinstance(deployment_id, str):
        return metadata
    if root.authority.settings.role == "audience" and (
        not model.startswith("litellm_proxy/") or base not in root.authority.settings.upstreams
    ):
        return {name: value for name, value in metadata.items() if name != CONTEXT}
    provider, _, native_model = model.partition("/")
    terminal = Terminal(deployment_id=deployment_id, model=native_model, provider=provider)
    return {**metadata, CONTEXT: Session(root, str(uuid4()), terminal, base if isinstance(base, str) else None)}


def prepare(kwargs: dict[str, object], details: dict[str, object], *, selected_oauth: bool = False) -> None:
    session = mapping(kwargs.get("metadata")).get(CONTEXT)
    if type(session) is not Session:
        scrub_headers(kwargs)
        return
    details[STAMP] = session
    with session.lock:
        if session.begun:
            return
        scrub_headers(kwargs)
        authority = session.root.authority
        if authority.settings.role == "audience":
            session.envelope = Envelope.model_validate_json(
                json.dumps(
                    {
                        "v": 1,
                        "audience": authority.settings.identity,
                        "local_request_id": session.root.local_request_id,
                        "local_attempt_id": session.attempt_id,
                        "local_deployment_id": session.terminal.deployment_id,
                        "correlation_id": None,
                        "receipts": [],
                        "state": "unavailable",
                    }
                )
            )
            if kwargs.get("api_base") != session.base:
                raise ReceiptUnavailable()
            opened = authority.request(
                "POST",
                "/correlations",
                {
                    "local_request_id": session.root.local_request_id,
                    "local_attempt_id": session.attempt_id,
                    "local_deployment_id": session.terminal.deployment_id,
                },
            )
            token = opened.get("ingress_jws")
            if not isinstance(token, str):
                raise ReceiptUnavailable()
            claims = authority.ingress(token)
            if (
                claims.aud != authority.settings.identity
                or claims.local_request_id != session.root.local_request_id
                or claims.local_attempt_id != session.attempt_id
                or claims.local_deployment_id != session.terminal.deployment_id
                or claims.correlation_id != opened.get("correlation_id")
                or claims.exp < time.time()
            ):
                raise ReceiptUnavailable()
            session.envelope = Envelope.model_validate_json(
                json.dumps(
                    {
                        "v": 1,
                        "audience": claims.aud,
                        "local_request_id": claims.local_request_id,
                        "local_attempt_id": claims.local_attempt_id,
                        "local_deployment_id": claims.local_deployment_id,
                        "correlation_id": claims.correlation_id,
                        "receipts": [],
                        "state": "pending",
                    }
                )
            )
            headers = {
                name: value
                for name, value in mapping(kwargs.get("extra_headers")).items()
                if name.lower() not in {INGRESS_HEADER, CENTRAL_HEADER}
            }
            kwargs["extra_headers"] = {**headers, "X-OpenOrange-Terminal-Ingress": token}
            if kwargs.get("stream") is True:
                kwargs["stream_options"] = {**mapping(kwargs.get("stream_options")), "include_usage": True}
        else:
            if session.terminal.provider == "chatgpt" and not selected_oauth:
                return
            claims = session.root.claim_ingress()
            delegate = authority.settings.delegates.get(session.terminal.deployment_id or "")
            if session.terminal.provider == "litellm_proxy" and delegate is not None:
                if session.base not in authority.settings.upstreams or kwargs.get("api_base") != session.base:
                    raise ReceiptUnavailable()
                accepted = authority.request(
                    "POST",
                    "/claims/delegate",
                    {
                        "correlation_id": claims.correlation_id,
                        "central_request_id": session.root.central_request_id,
                        "delegate_producer_id": delegate,
                    },
                )
                if accepted != {"correlation_id": claims.correlation_id, "producer_id": delegate}:
                    raise ReceiptUnavailable()
                headers = {
                    name: value
                    for name, value in mapping(kwargs.get("extra_headers")).items()
                    if name.lower() not in {INGRESS_HEADER, CENTRAL_HEADER}
                }
                kwargs["extra_headers"] = {
                    **headers,
                    "X-OpenOrange-Terminal-Ingress": session.root.ingress,
                    "X-OpenOrange-Terminal-Central-Request": session.root.central_request_id,
                }
                if kwargs.get("stream") is True:
                    kwargs["stream_options"] = {**mapping(kwargs.get("stream_options")), "include_usage": True}
                session.relay = True
                session.begun = True
                return
            planned = planned_ownership_for_terminal(details.get(OWNERSHIP_STAMP), session.terminal.deployment_id)
            accepted = authority.request(
                "POST",
                "/attempts/begin",
                {
                    "correlation_id": claims.correlation_id,
                    "central_request_id": session.root.central_request_id,
                    "terminal_attempt_id": session.attempt_id,
                    "terminal": _JSON.validate_python(session.terminal.model_dump(mode="json")),
                    "planned_ownership": planned,
                },
            )
            token = accepted.get("receipt_jws")
            parsed = parse_jws(token) if isinstance(token, str) else None
            if (
                set(accepted) != {"receipt_jws"}
                or parsed is None
                or not isinstance(parsed.claims, Attempt)
                or not authenticated(parsed, authority.settings.issuer, authority.keys)
                or parsed.claims.event != "started"
                or parsed.claims.aud != claims.aud
                or parsed.claims.local_request_id != claims.local_request_id
                or parsed.claims.local_attempt_id != claims.local_attempt_id
                or parsed.claims.local_deployment_id != claims.local_deployment_id
                or parsed.claims.correlation_id != claims.correlation_id
                or parsed.claims.central_request_id != session.root.central_request_id
                or parsed.claims.producer_id != authority.settings.identity
                or parsed.claims.terminal_attempt_id != session.attempt_id
                or parsed.claims.terminal != session.terminal
                or (
                    parsed.claims.planned_ownership.source != "unknown"
                    and parsed.claims.planned_ownership.model_dump(mode="json") != planned
                )
            ):
                raise ReceiptUnavailable()
        session.begun = True


async def prepare_async(kwargs: dict[str, object], details: dict[str, object]) -> None:
    if type(mapping(kwargs.get("metadata")).get(CONTEXT)) is Session:
        await asyncio.to_thread(prepare, kwargs, details)
    else:
        scrub_headers(kwargs)


def is_bound_relay(details: dict[str, object]) -> bool:
    session = details.get(STAMP)
    return type(session) is Session and (session.root.authority.settings.role == "audience" or session.relay)


def prepare_selected_dispatch(details: dict[str, object]) -> None:
    session = details.get(STAMP)
    if type(session) is Session and session.terminal.provider == "chatgpt" and not session.begun:
        prepare({"metadata": {CONTEXT: session}}, details, selected_oauth=True)


async def prepare_selected_dispatch_async(details: dict[str, object]) -> None:
    session = details.get(STAMP)
    if type(session) is Session and session.terminal.provider == "chatgpt" and not session.begun:
        await asyncio.to_thread(prepare_selected_dispatch, details)


def finish(details: dict[str, object], result: object, outcome: Literal["success", "failure", "unknown"]) -> None:
    session = details.get(STAMP)
    if type(session) is not Session or not session.begun:
        return
    with session.lock:
        session.usage_final = outcome == "success"
        if session.finished:
            return
        authority = session.root.authority
        if authority.settings.role == "producer":
            if session.relay:
                session.finished = True
                session.root.try_seal()
                return
            claims = session.root.claim
            if claims is None:
                return
            from litellm.types.utils import ModelResponse, ResponsesAPIResponse

            model = result.model if isinstance(result, (ModelResponse, ResponsesAPIResponse)) else None
            completed = (
                {"deployment_id": session.terminal.deployment_id, "model": model, "provider": session.terminal.provider}
                if model
                else None
            )
            from litellm.litellm_core_utils.terminal_usage_evidence import terminal_snapshot

            try:
                authority.request(
                    "POST",
                    "/attempts/finish-with-usage",
                    {
                        "finish": {
                            "correlation_id": claims.correlation_id,
                            "central_request_id": session.root.central_request_id,
                            "terminal_attempt_id": session.attempt_id,
                            "outcome": outcome,
                            "completed": _JSON.validate_python(completed),
                            "ownership": ownership_for_spend(
                                details.get(OWNERSHIP_STAMP), session.terminal.deployment_id
                            ),
                        },
                        "usage": terminal_snapshot(session).model_dump(mode="json"),
                    },
                )
                session.finished = True
            except ReceiptUnavailable:
                return
            session.root.try_seal()
        elif session.envelope is not None:
            for _ in range(5):
                session.envelope = authority.retrieve(session.envelope)
                if session.envelope.state != "pending":
                    break
                time.sleep(0.02)
            from litellm.litellm_core_utils.terminal_usage_evidence import refresh_usage

            refresh_usage(session)
            session.finished = True


def evidence_for_spend(value: object) -> dict[str, JsonValue] | None:
    if type(value) is not Session or value.envelope is None:
        return None
    from litellm.litellm_core_utils.terminal_receipt_evidence import safe_evidence

    return safe_evidence(value.envelope.model_dump(mode="json"))


async def finish_async(
    details: dict[str, object], result: object, outcome: Literal["success", "failure", "unknown"]
) -> None:
    if type(details.get(STAMP)) is Session:
        await asyncio.to_thread(finish, details, result, outcome)


def metadata_for_spend(value: object) -> dict[str, JsonValue]:
    evidence = evidence_for_spend(value)
    from litellm.litellm_core_utils.terminal_usage_evidence import usage_evidence_for_spend
    from litellm.litellm_core_utils.terminal_usage_observation import usage_for_spend

    return {
        **({FIELD: evidence} if evidence is not None else {}),
        **usage_for_spend(value),
        **usage_evidence_for_spend(value),
    }


def attempt_row_id(value: object) -> str | None:
    if type(value) is Session and value.root.authority.settings.role == "audience":
        return value.attempt_id
    return None


def failure_was_persisted(value: object) -> bool:
    return type(value) is Session and value.failure_persisted


def mark_failure_persisted(value: object) -> None:
    if type(value) is Session:
        value.failure_persisted = True


def failure_updates_were_queued(value: object) -> bool:
    return type(value) is Session and value.failure_updates_queued


def mark_failure_updates_queued(value: object) -> None:
    if type(value) is Session:
        value.failure_updates_queued = True


def wrap_result(result: object, root: Root | None) -> object:
    if root is None:
        return result
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    if not isinstance(result, CustomStreamWrapper):
        root.leave()
        return result

    class ReceiptStream(CustomStreamWrapper):
        def __init__(self, inner: CustomStreamWrapper) -> None:
            state = mapping(vars(inner))
            super().__init__(  # pyright: ignore[reportUnknownMemberType] -- upstream stream constructor is untyped
                completion_stream=inner,
                model=_MODEL.validate_python(state.get("model")),
                custom_llm_provider=inner.custom_llm_provider,
                logging_obj=inner.logging_obj,
            )
            self.inner = inner
            self._hidden_params = mapping(state.get("_hidden_params"))
            self.left = False

        def leave(self) -> None:
            if not self.left:
                self.left = True
                root.leave()

        def __next__(self) -> ModelResponseStream:
            try:
                return next(self.inner)
            except BaseException:
                self.leave()
                raise

        async def __anext__(self) -> ModelResponseStream:
            try:
                return await self.inner.__anext__()
            except BaseException:
                await asyncio.to_thread(self.leave)
                raise

        async def aclose(self) -> None:
            import anyio

            incomplete = not self.left
            with anyio.CancelScope(shield=True):
                try:
                    await self.inner.aclose()
                finally:
                    try:
                        if incomplete:
                            from litellm.litellm_core_utils.litellm_logging import Logging

                            self.inner._record_partial_usage_for_failure()
                            logger = mapping(vars(self.inner)).get("logging_obj")
                            if isinstance(logger, Logging):
                                await logger.async_failure_handler(  # pyright: ignore[reportUnknownMemberType] -- upstream callback has untyped parameters
                                    RuntimeError("terminal_receipt_stream_incomplete"),
                                    "terminal_receipt_stream_incomplete",
                                )
                    finally:
                        await asyncio.to_thread(self.leave)

    return ReceiptStream(result)
