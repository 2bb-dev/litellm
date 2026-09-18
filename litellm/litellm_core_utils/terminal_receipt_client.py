"""Explicit Platform receipt authority configuration and authenticated bounded IO."""

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, JsonValue, SecretStr, TypeAdapter, ValidationError, model_validator

from litellm.litellm_core_utils.terminal_receipt_evidence import (
    MAX_ENVELOPE,
    Closed,
    Envelope,
    Identifier,
    Ingress,
    UUIDText,
    authenticated,
    parse_jws,
    verify_set,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])


class ReceiptUnavailable(Exception):
    def __init__(self) -> None:
        super().__init__("terminal_receipt_unavailable")


class ReceiptPending(ReceiptUnavailable):
    pass


class Settings(Closed):
    v: Literal[1]
    role: Literal["audience", "producer"]
    identity: UUIDText
    issuer: UUIDText
    base_url: str
    bearer: SecretStr
    public_keys: dict[UUIDText, str]
    upstreams: tuple[str, ...] = ()
    delegates: dict[Identifier, UUIDText] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=2.0, gt=0, le=10)

    @model_validator(mode="after")
    def endpoints(self) -> "Settings":
        for value in (self.base_url, *self.upstreams):
            url = urlsplit(value)
            if (
                url.username
                or url.password
                or url.query
                or url.fragment
                or not url.hostname
                or (
                    url.scheme != "https"
                    and not (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"})
                )
            ):
                raise ValueError("invalid_receipt_endpoint")
        if not self.public_keys or not self.bearer.get_secret_value():
            raise ValueError("invalid_receipt_configuration")
        return self


class Authority:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys = self._keys(settings.public_keys)

    @staticmethod
    def _keys(values: Mapping[str, str]) -> dict[str, Ed25519PublicKey]:
        result: dict[str, Ed25519PublicKey] = {}
        for kid, pem in values.items():
            key = serialization.load_pem_public_key(pem.encode("ascii"))
            if not isinstance(key, Ed25519PublicKey):
                raise ReceiptUnavailable()
            result[kid] = key
        return result

    def request(
        self, method: Literal["POST", "GET"], path: str, body: Mapping[str, JsonValue] | None = None
    ) -> dict[str, JsonValue]:
        try:
            with (
                httpx.Client(trust_env=False, follow_redirects=False, timeout=self.settings.timeout_seconds) as client,
                client.stream(
                    method,
                    self.settings.base_url.rstrip("/") + path,
                    json=body,
                    headers={"Authorization": "Bearer " + self.settings.bearer.get_secret_value()},
                ) as response,
            ):
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    if len(chunks) + len(chunk) > MAX_ENVELOPE:
                        raise ReceiptUnavailable()
                    chunks.extend(chunk)
                result = _OBJECT.validate_json(bytes(chunks))
                if (
                    response.status_code == 409
                    and path == "/correlations/complete"
                    and result == {"error": "attempts_incomplete"}
                ):
                    raise ReceiptPending()
                if response.status_code != 200:
                    raise ReceiptUnavailable()
                return result
        except (httpx.HTTPError, ValidationError, ValueError):
            raise ReceiptUnavailable() from None

    def ingress(self, token: str) -> Ingress:
        parsed = parse_jws(token, ingress=True)
        if (
            parsed is None
            or not isinstance(parsed.claims, Ingress)
            or not authenticated(parsed, self.settings.issuer, self.keys)
        ):
            raise ReceiptUnavailable()
        return parsed.claims

    def retrieve(self, envelope: Envelope) -> Envelope:
        if envelope.correlation_id is None:
            return envelope.model_copy(update={"state": "unavailable"})
        try:
            receipts: tuple[str, ...] = ()
            after = 0
            for _ in range(3):
                page = self.request("GET", f"/correlations/{envelope.correlation_id}/events?after={after}&limit=16")
                events = page.get("events")
                next_after = page.get("next_after")
                if (
                    set(page) != {"events", "next_after", "has_more", "complete"}
                    or not isinstance(events, list)
                    or len(events) > 16
                    or any(not isinstance(item, str) for item in events)
                    or type(next_after) is not int
                    or next_after < after
                    or type(page.get("has_more")) is not bool
                ):
                    return envelope.model_copy(update={"receipts": (), "state": "invalid"})
                receipts = (*receipts, *(item for item in events if isinstance(item, str)))
                if len(receipts) > 33:
                    return envelope.model_copy(update={"receipts": (), "state": "overflow"})
                if page.get("has_more") is False:
                    break
                if next_after <= after:
                    return envelope.model_copy(update={"receipts": (), "state": "invalid"})
                after = next_after
            else:
                return envelope.model_copy(update={"receipts": (), "state": "overflow"})
            data = {**envelope.model_dump(mode="json"), "receipts": receipts, "state": "pending"}
            encoded = json.dumps(data, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_ENVELOPE:
                return envelope.model_copy(update={"receipts": (), "state": "overflow"})
            candidate = Envelope.model_validate_json(encoded)
            state = verify_set(candidate, self.settings.issuer, self.keys)
            return candidate.model_copy(
                update={"state": state, "receipts": () if state == "invalid" else candidate.receipts}
            )
        except (ReceiptUnavailable, ValidationError, ValueError):
            retained = envelope.receipts if verify_set(envelope, self.settings.issuer, self.keys) != "invalid" else ()
            return envelope.model_copy(update={"receipts": retained, "state": "unavailable"})


def private_file(name: str) -> bytes:
    try:
        path = Path(name)
        if not path.is_absolute():
            raise ReceiptUnavailable()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_uid not in {0, os.getuid()}
                or metadata.st_size > MAX_ENVELOPE
            ):
                raise ReceiptUnavailable()
            data = source.read(MAX_ENVELOPE + 1)
            if len(data) > MAX_ENVELOPE:
                raise ReceiptUnavailable()
        return data
    except OSError:
        raise ReceiptUnavailable() from None


def configured_authority() -> Authority | None:
    name = os.environ.get("OPENORANGE_TERMINAL_RECEIPT_CLIENT_CONFIG_FILE")
    if not name:
        return None
    try:
        return Authority(Settings.model_validate_json(private_file(name)))
    except (ValueError, ValidationError):
        raise ReceiptUnavailable() from None
