"""Closed, content-free terminal receipts; retained bytes remain independently verifiable."""

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

FIELD = "openorange_terminal_evidence"
CONTEXT = "_openorange_terminal_context"
STAMP = "_openorange_terminal_stamp"
MAX_JWS = 3072
MAX_EVENTS = 33
MAX_ENVELOPE = 65536
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,255}$")]
UUIDText = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")]
State = Literal["complete", "pending", "unavailable", "invalid", "overflow"]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class Header(Closed):
    alg: Literal["EdDSA"]
    typ: Literal["oo-terminal-receipt+jwt", "oo-terminal-ingress+jwt"]
    kid: UUIDText


class Binding(Closed):
    aud: UUIDText
    local_request_id: UUIDText
    local_attempt_id: UUIDText
    local_deployment_id: Identifier
    correlation_id: UUIDText

    @model_validator(mode="after")
    def content_free(self) -> "Binding":
        if "://" in self.local_deployment_id:
            raise ValueError("invalid_identifier")
        return self


class Ingress(Binding):
    v: Literal[1]
    iss: UUIDText
    exp: Annotated[int, Field(ge=0)]


class Ownership(Closed):
    v: Literal[1]
    source: Literal["platform", "byok", "unknown"]
    provenance: Literal[
        "credential_registration", "route_registration", "unregistered", "credential_override", "ambiguous"
    ]
    registration_id: UUIDText | None
    registration_revision: UUIDText | None
    deployment_id: Identifier | None

    @model_validator(mode="after")
    def coherent(self) -> "Ownership":
        positive = self.source != "unknown"
        if positive != (self.provenance in {"credential_registration", "route_registration"}):
            raise ValueError("invalid_ownership")
        if positive:
            if self.registration_id is None or self.registration_revision is None or self.deployment_id is None:
                raise ValueError("invalid_ownership")
        elif self.registration_id is not None or self.registration_revision is not None:
            raise ValueError("invalid_ownership")
        if self.deployment_id and "://" in self.deployment_id:
            raise ValueError("invalid_identifier")
        return self


class Terminal(Closed):
    deployment_id: Identifier | None
    model: Identifier | None
    provider: Identifier | None

    @model_validator(mode="after")
    def content_free(self) -> "Terminal":
        if any(value is not None and "://" in value for value in (self.deployment_id, self.model, self.provider)):
            raise ValueError("invalid_identifier")
        return self


class Common(Binding):
    v: Literal[1]
    iss: UUIDText
    receipt_id: UUIDText
    central_request_id: UUIDText
    producer_id: UUIDText
    sequence: Annotated[int, Field(ge=1, le=MAX_EVENTS)]
    occurred_at: Annotated[int, Field(ge=0)]


class Attempt(Common):
    terminal_attempt_id: UUIDText
    event: Literal["started", "finished"]
    outcome: Literal["pending", "success", "failure", "unknown"]
    binding: Literal["unobserved", "observed", "unknown"]
    terminal: Terminal
    completed: Terminal | None
    planned_ownership: Ownership
    ownership: Ownership

    @model_validator(mode="after")
    def coherent(self) -> "Attempt":
        if self.ownership.deployment_id != self.terminal.deployment_id:
            raise ValueError("invalid_terminal_binding")
        if self.planned_ownership.deployment_id != self.terminal.deployment_id:
            raise ValueError("invalid_terminal_binding")
        if self.event == "started":
            if (
                self.outcome != "pending"
                or self.binding != "unobserved"
                or self.ownership.source != "unknown"
                or self.completed
            ):
                raise ValueError("invalid_started_event")
        elif self.outcome == "pending" or self.binding == "unobserved":
            raise ValueError("invalid_finished_event")
        if self.binding == "observed":
            if self.ownership.source == "unknown" or self.ownership != self.planned_ownership:
                raise ValueError("invalid_observation")
        elif self.ownership.source != "unknown":
            raise ValueError("invalid_observation")
        if self.completed is not None and (
            self.completed.deployment_id != self.terminal.deployment_id
            or self.completed.provider != self.terminal.provider
            or self.completed.model is None
        ):
            raise ValueError("invalid_completed_identity")
        return self


class Seal(Common):
    event: Literal["seal"]
    attempt_count: Annotated[int, Field(ge=1, le=16)]


Receipt = Attempt | Seal
_RECEIPT: TypeAdapter[Receipt] = TypeAdapter(Annotated[Receipt, Field(discriminator="event")])
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class Envelope(Closed):
    v: Literal[1]
    audience: UUIDText
    local_request_id: UUIDText
    local_attempt_id: UUIDText
    local_deployment_id: Identifier
    correlation_id: UUIDText | None
    receipts: Annotated[tuple[str, ...], Field(max_length=MAX_EVENTS)]
    state: State

    @model_validator(mode="after")
    def content_free(self) -> "Envelope":
        if "://" in self.local_deployment_id:
            raise ValueError("invalid_identifier")
        if self.correlation_id is None and (self.receipts or self.state not in {"unavailable", "invalid"}):
            raise ValueError("invalid_unavailable_context")
        return self


@dataclass(frozen=True, slots=True)
class Parsed:
    original: str
    header: Header
    claims: Receipt | Ingress
    signed: bytes
    signature: bytes


def decode_segment(segment: str) -> bytes:
    decoded = base64.b64decode(segment + "=" * (-len(segment) % 4), altchars=b"-_", validate=True)
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != segment:
        raise ValueError("invalid_base64")
    return decoded


def parse_jws(value: str, *, ingress: bool = False) -> Parsed | None:
    if not value.isascii() or len(value) > MAX_JWS:
        return None
    segments = value.split(".")
    if len(segments) != 3:
        return None
    try:
        header_bytes, payload_bytes, signature = (decode_segment(segment) for segment in segments)
        header = Header.model_validate_json(header_bytes)
        claims = Ingress.model_validate_json(payload_bytes) if ingress else _RECEIPT.validate_json(payload_bytes)
        expected = "oo-terminal-ingress+jwt" if ingress else "oo-terminal-receipt+jwt"
        if header.typ != expected or len(signature) != 64:
            return None
        for original, model in ((header_bytes, header), (payload_bytes, claims)):
            canonical = json.dumps(
                model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            if original != canonical.encode("utf-8"):
                return None
        return Parsed(value, header, claims, f"{segments[0]}.{segments[1]}".encode("ascii"), signature)
    except (ValueError, ValidationError, binascii.Error):
        return None


def authenticated(parsed: Parsed, issuer: str, keys: Mapping[str, Ed25519PublicKey]) -> bool:
    key = keys.get(parsed.header.kid)
    if parsed.claims.iss != issuer or key is None:
        return False
    try:
        key.verify(parsed.signature, parsed.signed)
    except InvalidSignature:
        return False
    return True


def same_binding(claims: Binding, envelope: Envelope) -> bool:
    return (
        claims.aud == envelope.audience
        and claims.local_request_id == envelope.local_request_id
        and claims.local_attempt_id == envelope.local_attempt_id
        and claims.local_deployment_id == envelope.local_deployment_id
        and claims.correlation_id == envelope.correlation_id
    )


def safe_evidence(value: object) -> dict[str, JsonValue]:
    try:
        data = _JSON_OBJECT.validate_python(value)
        if type(data.get("v")) is not int:
            return {"v": 1, "state": "invalid"}
        if (
            set(data) == {"v", "state"}
            and type(data.get("v")) is int
            and data.get("v") == 1
            and data.get("state") in {"invalid", "overflow"}
        ):
            return data
        shell = Envelope.model_validate_json(_JSON_OBJECT.dump_json({**data, "receipts": [], "state": "invalid"}))
        encoded = _JSON_OBJECT.dump_json(data)
        if len(encoded) > MAX_ENVELOPE:
            state = "overflow" if shell.correlation_id is not None else "invalid"
            return _JSON_OBJECT.validate_json(shell.model_copy(update={"state": state}).model_dump_json())
        try:
            envelope = Envelope.model_validate_json(encoded)
        except ValidationError:
            return _JSON_OBJECT.validate_json(shell.model_dump_json())
        parsed = tuple(parse_jws(item) for item in envelope.receipts)
        if any(item is None or not same_binding(item.claims, envelope) for item in parsed):
            return _JSON_OBJECT.validate_json(shell.model_dump_json())
        return _JSON_OBJECT.validate_json(envelope.model_dump_json())
    except (ValidationError, ValueError):
        return {"v": 1, "state": "invalid"}


def evidence_from_metadata(value: object) -> dict[str, JsonValue]:
    try:
        parsed = _JSON.validate_json(value) if isinstance(value, str) else _JSON.validate_python(value)
        if isinstance(parsed, dict) and FIELD in parsed:
            return {FIELD: safe_evidence(parsed[FIELD])}
    except (ValidationError, ValueError):
        pass
    return {}


def verify_set(envelope: Envelope, issuer: str, keys: Mapping[str, Ed25519PublicKey]) -> State:
    parsed = tuple(parse_jws(item) for item in envelope.receipts)
    if any(
        item is None or not same_binding(item.claims, envelope) or not authenticated(item, issuer, keys)
        for item in parsed
    ):
        return "invalid"
    claims = tuple(item.claims for item in parsed if item is not None and isinstance(item.claims, (Attempt, Seal)))
    if len({item.receipt_id for item in claims}) != len(claims):
        return "invalid"
    if any(item.sequence != number for number, item in enumerate(claims, 1)):
        return "invalid"
    if not claims or not isinstance(claims[-1], Seal):
        return "pending"
    seal = claims[-1]
    attempts = tuple(item for item in claims[:-1] if isinstance(item, Attempt))
    started = tuple(item for item in attempts if item.event == "started")
    finished = tuple(item for item in attempts if item.event == "finished")
    if len(attempts) != len(claims) - 1 or len(started) != seal.attempt_count or len(finished) != seal.attempt_count:
        return "invalid"
    if len({item.terminal_attempt_id for item in started}) != len(started):
        return "invalid"
    if len({item.terminal_attempt_id for item in finished}) != len(finished):
        return "invalid"
    if any(item.central_request_id != seal.central_request_id for item in attempts):
        return "invalid"
    for start in started:
        end = next((item for item in finished if item.terminal_attempt_id == start.terminal_attempt_id), None)
        if end is None or (
            end.sequence <= start.sequence
            or end.producer_id != start.producer_id
            or end.terminal != start.terminal
            or end.planned_ownership != start.planned_ownership
        ):
            return "invalid"
    return "complete"
