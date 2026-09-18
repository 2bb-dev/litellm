"""Public-key-only encryption for retained request content, before durable queues."""

import base64
import json
import logging
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from joserfc import jwe
from joserfc.jwk import RSAKey
from joserfc.registry import HeaderParameter
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, field_validator

from litellm.litellm_core_utils.request_content_mode import (
    CONFIG_ENV,
    encryption_enabled,
    require_protection_marker,
)

INSTANCE_ENV = "OPENORANGE_INSTANCE_UID"
CONTENT_FORMAT = "openorange.request-log.v1"
MAX_CONTENT_BYTES = 16 * 1024 * 1024
MAX_CONFIG_BYTES = 16 * 1024
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


class PublicJWK(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kty: Literal["RSA"]
    n: str
    e: Literal["AQAB"]
    alg: Literal["RSA-OAEP-256"]
    use: Literal["enc"]
    key_ops: tuple[Literal["encrypt"], ...]
    kid: str


class PublicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    instanceUid: str
    publicKey: PublicJWK

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value


class ContentEnvelope(TypedDict):
    format: Literal["openorange.request-log.v1"]
    jwe: str


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    code: Literal[
        "configuration_unavailable",
        "encryption_failed",
        "content_too_large",
        "invalid_record_id",
        "transformation_failed",
    ]


@dataclass(frozen=True, slots=True)
class EncryptionKey:
    instance_uid: str
    kid: str
    key: RSAKey


_runtime_failure: CaptureFailure | None = None


def record_transform_failure() -> CaptureFailure:
    global _runtime_failure
    _runtime_failure = CaptureFailure("transformation_failed")
    logging.getLogger("LiteLLM Proxy").error(
        "Encrypted request content transformation failed",
        extra={"openorange_content_safe": True},
    )
    return _runtime_failure


def jwe_registry() -> jwe.JWERegistry:
    registry = jwe.JWERegistry(
        {
            "v": HeaderParameter("Content format version", "int", required=True),
            "instanceUid": HeaderParameter("Immutable instance", "str", required=True),
            "purpose": HeaderParameter("Content purpose", "str", required=True),
            "recordId": HeaderParameter("Physical spend record", "str", required=True),
        },
        algorithms=("RSA-OAEP-256", "A256GCM"),
    )
    registry.max_ciphertext_length = MAX_CONTENT_BYTES
    registry.max_protected_header_length = 2048
    return registry


def parse_public_config(raw: bytes, instance_uid: str) -> EncryptionKey | CaptureFailure:
    try:
        if len(raw) > MAX_CONFIG_BYTES:
            return CaptureFailure("configuration_unavailable")
        config = PublicConfig.model_validate_json(raw)
        if str(UUID(instance_uid)) != instance_uid or config.instanceUid != instance_uid:
            return CaptureFailure("configuration_unavailable")
        public = config.publicKey
        if public.key_ops != ("encrypt",) or not _B64URL.fullmatch(public.n):
            return CaptureFailure("configuration_unavailable")
        modulus = base64.urlsafe_b64decode(public.n + "=" * (-len(public.n) % 4))
        if (
            not modulus
            or modulus[0] == 0
            or base64.urlsafe_b64encode(modulus).rstrip(b"=").decode("ascii") != public.n
            or not 3072 <= int.from_bytes(modulus, "big").bit_length() <= 4096
        ):
            return CaptureFailure("configuration_unavailable")
        key = RSAKey.import_key(public.model_dump(mode="json"))
        if key.thumbprint() != public.kid:
            return CaptureFailure("configuration_unavailable")
        return EncryptionKey(instance_uid, public.kid, key)
    except (ValueError, TypeError, ValidationError):
        return CaptureFailure("configuration_unavailable")


def encrypt_content(
    key: EncryptionKey, record_id: str, content: Mapping[str, JsonValue]
) -> ContentEnvelope | CaptureFailure:
    if not 1 <= len(record_id) <= 1024 or any(ord(char) < 32 or ord(char) == 127 for char in record_id):
        return CaptureFailure("invalid_record_id")
    try:
        plaintext = json.dumps(content, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(plaintext) > MAX_CONTENT_BYTES:
            return CaptureFailure("content_too_large")
        protected = {
            "alg": "RSA-OAEP-256",
            "enc": "A256GCM",
            "typ": "openorange-request-content+jwe",
            "kid": key.kid,
            "v": 1,
            "instanceUid": key.instance_uid,
            "purpose": "request-log",
            "recordId": record_id,
        }
        token = jwe.encrypt_compact(protected, plaintext, key.key, registry=jwe_registry())
    except Exception:
        return CaptureFailure("encryption_failed")
    else:
        return ContentEnvelope(format=CONTENT_FORMAT, jwe=token)


class RequestContentEncryptor:
    def __init__(self, key: EncryptionKey) -> None:
        self.key = key
        self._failure: CaptureFailure | None = None

    def encrypt(self, record_id: str, content: Mapping[str, JsonValue]) -> ContentEnvelope | CaptureFailure:
        result = encrypt_content(self.key, record_id, content)
        if isinstance(result, CaptureFailure):
            self.record_failure(result)
        return result

    def record_failure(self, failure: CaptureFailure) -> None:
        self._failure = failure
        logging.getLogger("LiteLLM Proxy").error(
            "Encrypted request content capture failed: %s",
            failure.code,
            extra={"openorange_content_safe": True},
        )

    def readiness(self) -> CaptureFailure | None:
        if self._failure is None:
            return None
        if self._failure.code == "transformation_failed":
            return self._failure
        probe = encrypt_content(self.key, "capture-readiness", {"v": 1})
        self._failure = probe if isinstance(probe, CaptureFailure) else None
        return self._failure


@lru_cache(maxsize=2)
def _load_encryptor(
    path: str, instance_uid: str, modified_ns: int, size: int
) -> RequestContentEncryptor | CaptureFailure:
    del modified_ns
    if size > MAX_CONFIG_BYTES:
        return CaptureFailure("configuration_unavailable")
    try:
        with Path(path).open("rb") as source:
            parsed = parse_public_config(source.read(MAX_CONFIG_BYTES + 1), instance_uid)
    except OSError:
        return CaptureFailure("configuration_unavailable")
    if isinstance(parsed, CaptureFailure):
        return parsed
    encryptor = RequestContentEncryptor(parsed)
    encryptor.encrypt("capture-readiness", {"v": 1})
    return encryptor


def configured_encryptor() -> RequestContentEncryptor | CaptureFailure:
    path = os.getenv(CONFIG_ENV, "").strip()
    if not path:
        return CaptureFailure("configuration_unavailable")
    try:
        file_stat = Path(path).lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            return CaptureFailure("configuration_unavailable")
        encryptor = _load_encryptor(path, os.getenv(INSTANCE_ENV, ""), file_stat.st_mtime_ns, file_stat.st_size)
        if isinstance(encryptor, CaptureFailure):
            return encryptor
        if not require_protection_marker(encryptor.key.instance_uid, encryptor.key.kid):
            return CaptureFailure("configuration_unavailable")
        return encryptor
    except OSError:
        return CaptureFailure("configuration_unavailable")


def encryption_readiness() -> CaptureFailure | None:
    if not encryption_enabled():
        return None
    if _runtime_failure is not None:
        return _runtime_failure
    encryptor = configured_encryptor()
    return encryptor if isinstance(encryptor, CaptureFailure) else encryptor.readiness()
