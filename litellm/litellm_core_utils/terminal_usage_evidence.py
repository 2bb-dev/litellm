"""Closed signed measurement originals, bound to immutable terminal FINISH bytes."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, JsonValue, StringConstraints, TypeAdapter, ValidationError, model_validator

from litellm.litellm_core_utils.terminal_receipt_evidence import (
    MAX_ENVELOPE,
    MAX_JWS,
    Attempt,
    Binding,
    Closed,
    Envelope,
    Identifier,
    State,
    UUIDText,
    decode_segment,
    parse_jws,
    same_binding,
    verify_set,
)
from litellm.litellm_core_utils.terminal_usage_observation import UsageSnapshot

if TYPE_CHECKING:
    from litellm.litellm_core_utils.terminal_receipt_client import Authority

FIELD = "openorange_terminal_usage_evidence"
_JSON = TypeAdapter(dict[str, JsonValue])


class TerminalUsage(UsageSnapshot):
    v: Annotated[int, Field(strict=True, ge=1, le=1)]
    state: Literal["observed", "partial", "unobserved"]

    @model_validator(mode="after")
    def coherent(self) -> "TerminalUsage":
        values = UsageSnapshot.model_validate({name: getattr(self, name) for name in UsageSnapshot.model_fields})
        present = any(getattr(values, name) is not None for name in UsageSnapshot.model_fields)
        if (self.state == "unobserved") == present or len(self.model_dump_json().encode()) > 512:
            raise ValueError("invalid_terminal_usage")
        return self


class UsageHeader(Closed):
    alg: Literal["EdDSA"]
    typ: Literal["oo-terminal-usage+jwt"]
    kid: UUIDText


class UsageClaim(Binding):
    v: Annotated[int, Field(strict=True, ge=1, le=1)]
    iss: UUIDText
    measurement_id: UUIDText
    central_request_id: UUIDText
    producer_id: UUIDText
    terminal_attempt_id: UUIDText
    finish_receipt_id: UUIDText
    finish_sequence: Annotated[int, Field(strict=True, ge=2, le=32)]
    finish_jws_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    occurred_at: Annotated[int, Field(strict=True, ge=1, le=9007199254740991)]
    usage: TerminalUsage


class UsageEnvelope(Closed):
    v: Annotated[int, Field(strict=True, ge=1, le=1)]
    audience: UUIDText
    local_request_id: UUIDText
    local_attempt_id: UUIDText
    local_deployment_id: Identifier
    correlation_id: UUIDText | None
    measurements: Annotated[tuple[str, ...], Field(max_length=16)]
    state: State

    @model_validator(mode="after")
    def coherent(self) -> "UsageEnvelope":
        if "://" in self.local_deployment_id:
            raise ValueError("invalid_identifier")
        if self.correlation_id is None and (self.measurements or self.state not in {"unavailable", "invalid"}):
            raise ValueError("invalid_usage_context")
        return self


@dataclass(frozen=True, slots=True)
class ParsedUsage:
    original: str
    header: UsageHeader
    claims: UsageClaim
    signed: bytes
    signature: bytes


def parse_usage(value: str) -> ParsedUsage | None:
    if not value.isascii() or len(value) > MAX_JWS:
        return None
    segments = value.split(".")
    if len(segments) != 3:
        return None
    try:
        header_bytes, payload_bytes, signature = (decode_segment(segment) for segment in segments)
        header = UsageHeader.model_validate_json(header_bytes)
        claims = UsageClaim.model_validate_json(payload_bytes)
        if len(signature) != 64:
            return None
        for original, model in ((header_bytes, header), (payload_bytes, claims)):
            canonical = json.dumps(
                model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            if original != canonical.encode("utf-8"):
                return None
        return ParsedUsage(value, header, claims, f"{segments[0]}.{segments[1]}".encode("ascii"), signature)
    except (ValueError, ValidationError):
        return None


def usage_authenticated(parsed: ParsedUsage, issuer: str, keys: Mapping[str, Ed25519PublicKey]) -> bool:
    key = keys.get(parsed.header.kid)
    if parsed.claims.iss != issuer or key is None:
        return False
    try:
        key.verify(parsed.signature, parsed.signed)
        return True
    except InvalidSignature:
        return False


def usage_shell(envelope: Envelope) -> UsageEnvelope:
    return UsageEnvelope(
        v=1,
        audience=envelope.audience,
        local_request_id=envelope.local_request_id,
        local_attempt_id=envelope.local_attempt_id,
        local_deployment_id=envelope.local_deployment_id,
        correlation_id=envelope.correlation_id,
        measurements=(),
        state="unavailable" if envelope.correlation_id is None else "pending",
    )


def matching_finish(claim: UsageClaim, finish: Attempt, original: str) -> bool:
    return (
        finish.event == "finished"
        and all(
            getattr(claim, name) == getattr(finish, name)
            for name in (
                "iss",
                "aud",
                "correlation_id",
                "local_request_id",
                "local_attempt_id",
                "local_deployment_id",
                "central_request_id",
                "producer_id",
                "terminal_attempt_id",
            )
        )
        and claim.finish_receipt_id == finish.receipt_id
        and claim.finish_sequence == finish.sequence
        and claim.finish_jws_sha256 == hashlib.sha256(original.encode("ascii")).hexdigest()
    )


def verify_usage_set(
    value: UsageEnvelope, ownership: Envelope, issuer: str, keys: Mapping[str, Ed25519PublicKey]
) -> State:
    original_state = verify_set(ownership, issuer, keys)
    if original_state == "invalid":
        return "invalid"
    if any(
        getattr(value, name) != getattr(ownership, name)
        for name in (
            "audience",
            "local_request_id",
            "local_attempt_id",
            "local_deployment_id",
            "correlation_id",
        )
    ):
        return "invalid"
    finishes = tuple(
        p
        for token in ownership.receipts
        if (p := parse_jws(token)) is not None and isinstance(p.claims, Attempt) and p.claims.event == "finished"
    )
    parsed = tuple(parse_usage(token) for token in value.measurements)
    if any(
        p is None or not usage_authenticated(p, issuer, keys) or not same_binding(p.claims, ownership) for p in parsed
    ):
        return "invalid"
    claims = tuple(p.claims for p in parsed if p is not None)
    if len({p.measurement_id for p in claims}) != len(claims) or len({p.finish_receipt_id for p in claims}) != len(
        claims
    ):
        return "invalid"
    for claim in claims:
        finish = next(
            (p for p in finishes if isinstance(p.claims, Attempt) and p.claims.receipt_id == claim.finish_receipt_id),
            None,
        )
        if finish is None:
            return "invalid" if original_state == "complete" else "pending"
        if not isinstance(finish.claims, Attempt) or not matching_finish(claim, finish.claims, finish.original):
            return "invalid"
    return "complete" if original_state == "complete" and len(claims) == len(finishes) else "pending"


def safe_usage_evidence(value: object) -> dict[str, JsonValue]:
    try:
        data = _JSON.validate_python(value)
        if data in ({"v": 1, "state": "invalid"}, {"v": 1, "state": "overflow"}) and type(data.get("v")) is int:
            return data
        shell = UsageEnvelope.model_validate_json(_JSON.dump_json({**data, "measurements": [], "state": "invalid"}))
        encoded = _JSON.dump_json(data)
        if len(encoded) > MAX_ENVELOPE:
            return shell.model_copy(
                update={"state": "overflow" if shell.correlation_id is not None else "invalid"}
            ).model_dump(mode="json")
        try:
            candidate = UsageEnvelope.model_validate_json(encoded)
        except ValidationError:
            return shell.model_dump(mode="json")
        for token in candidate.measurements:
            parsed = parse_usage(token)
            if parsed is None or any(
                getattr(parsed.claims, name) != getattr(candidate, target)
                for name, target in (
                    ("aud", "audience"),
                    ("local_request_id", "local_request_id"),
                    ("local_attempt_id", "local_attempt_id"),
                    ("local_deployment_id", "local_deployment_id"),
                    ("correlation_id", "correlation_id"),
                )
            ):
                return shell.model_dump(mode="json")
        return candidate.model_dump(mode="json")
    except (ValueError, ValidationError):
        return {"v": 1, "state": "invalid"}


def retrieve_usage(authority: "Authority", previous: UsageEnvelope, ownership: Envelope) -> UsageEnvelope:
    from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable

    if previous.correlation_id is None:
        return previous.model_copy(update={"state": "unavailable"})
    try:
        measurements: tuple[str, ...] = ()
        after = 0
        for _ in range(2):
            page = authority.request("GET", f"/correlations/{previous.correlation_id}/usage?after={after}&limit=16")
            items, next_after = page.get("measurements"), page.get("next_after")
            if (
                set(page) != {"measurements", "next_after", "has_more", "complete"}
                or not isinstance(items, list)
                or any(not isinstance(token, str) for token in items)
                or type(next_after) is not int
                or next_after < after
                or type(page.get("has_more")) is not bool
                or type(page.get("complete")) is not bool
            ):
                return previous.model_copy(update={"measurements": (), "state": "invalid"})
            measurements = (*measurements, *(token for token in items if isinstance(token, str)))
            if len(measurements) > 16:
                return previous.model_copy(update={"measurements": (), "state": "overflow"})
            if page["has_more"] is False:
                break
            if next_after <= after:
                return previous.model_copy(update={"measurements": (), "state": "invalid"})
            after = next_after
        else:
            return previous.model_copy(update={"measurements": (), "state": "overflow"})
        candidate = previous.model_copy(update={"measurements": measurements, "state": "pending"})
        safe = UsageEnvelope.model_validate_json(
            _JSON.dump_json(safe_usage_evidence(candidate.model_dump(mode="json")))
        )
        if safe.state in {"invalid", "overflow"}:
            return safe
        state = verify_usage_set(safe, ownership, authority.settings.issuer, authority.keys)
        return safe.model_copy(update={"state": state, "measurements": () if state == "invalid" else safe.measurements})
    except (ReceiptUnavailable, ValidationError, ValueError):
        keep = (
            previous.measurements
            if verify_usage_set(previous, ownership, authority.settings.issuer, authority.keys) != "invalid"
            else ()
        )
        return previous.model_copy(update={"measurements": keep, "state": "unavailable"})


def terminal_snapshot(value: object) -> TerminalUsage:
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(value) is not Session or value.usage_snapshot is None:
        return TerminalUsage(v=1, state="unobserved")
    fields = value.usage_snapshot.model_dump(mode="json")
    has_values = any(getattr(value.usage_snapshot, name) is not None for name in UsageSnapshot.model_fields)
    state = (
        "unobserved"
        if not has_values
        else "observed"
        if value.native_usage_final and not value.usage_invalid
        else "partial"
    )
    return TerminalUsage.model_validate({"v": 1, "state": state, **fields})


def refresh_usage(value: object) -> None:
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(value) is not Session or value.envelope is None or value.root.authority.settings.role != "audience":
        return
    previous = value.usage_envelope or usage_shell(value.envelope)
    if previous.correlation_id != value.envelope.correlation_id:
        previous = usage_shell(value.envelope)
    value.usage_envelope = retrieve_usage(value.root.authority, previous, value.envelope)
    value.usage_snapshot = None
    if value.usage_envelope.state != "complete" or len(value.usage_envelope.measurements) != 1:
        return
    parsed = parse_usage(value.usage_envelope.measurements[0])
    if parsed is None or parsed.claims.usage.state == "unobserved":
        return
    value.usage_snapshot = UsageSnapshot.model_validate(
        {name: getattr(parsed.claims.usage, name) for name in UsageSnapshot.model_fields}
    )
    value.usage_invalid = parsed.claims.usage.state != "observed"


def usage_evidence_for_spend(value: object) -> dict[str, JsonValue]:
    from litellm.litellm_core_utils.terminal_receipt_hooks import Session

    if type(value) is not Session or value.envelope is None or value.root.authority.settings.role != "audience":
        return {}
    envelope = value.usage_envelope or usage_shell(value.envelope)
    return {FIELD: safe_usage_evidence(envelope.model_dump(mode="json"))}
