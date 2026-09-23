"""Explicit account registration and private selected ChatGPT credential snapshot."""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import StringConstraints, ValidationError, model_validator

from litellm.litellm_core_utils.credential_ownership import (
    CONTEXT,
    DISPATCH,
    STAMP,
    Registration,
    selected_oauth_proof,
)
from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable, private_file
from litellm.litellm_core_utils.terminal_receipt_evidence import Closed, UUIDText
from litellm.litellm_core_utils.terminal_receipt_hooks import CONTEXT as TERMINAL_CONTEXT
from litellm.litellm_core_utils.terminal_receipt_hooks import STAMP as TERMINAL_STAMP
from litellm.litellm_core_utils.terminal_receipt_hooks import Session, mapping

CONTEXT_FIELD = "_openorange_terminal_oauth"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    account_id: str = field(repr=False)
    key_digest: bytes = field(repr=False)
    api_base: str = field(repr=False)


class AccountRegistration(Closed):
    v: Literal[1]
    source: Literal["platform", "byok"]
    registration_id: UUIDText
    registration_revision: UUIDText
    account_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    api_base: str

    @model_validator(mode="after")
    def endpoint(self) -> "AccountRegistration":
        url = urlsplit(self.api_base)
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or (
                url.scheme != "https"
                and not (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"})
            )
        ):
            raise ValueError("invalid_account_registration")
        return self


def record_account(params: dict[str, object], snapshot: AccountSnapshot | None) -> None:
    metadata = mapping(params.get("litellm_metadata")) or mapping(params.get("metadata"))
    session = metadata.get(TERMINAL_CONTEXT)
    if type(session) is Session:
        with session.lock:
            session.oauth = snapshot


def bind_account(params: dict[str, object], details: dict[str, object], *, responses: bool = False) -> None:
    session = details.get(TERMINAL_STAMP)
    snapshot = session.oauth if type(session) is Session else None
    if type(session) is not Session or session.terminal.provider != "chatgpt" or type(snapshot) is not AccountSnapshot:
        return
    name = os.environ.get("OPENORANGE_CHATGPT_CREDENTIAL_REGISTRATION_FILE")
    if not name:
        return
    try:
        registration = AccountRegistration.model_validate_json(private_file(name))
    except (ReceiptUnavailable, ValidationError, ValueError):
        return
    account_digest = hashlib.sha256(snapshot.account_id.encode()).digest()
    if account_digest.hex() != registration.account_sha256 or snapshot.api_base != registration.api_base:
        return
    declaration = Registration(
        v=1,
        source=registration.source,
        registration_id=registration.registration_id,
        registration_revision=registration.registration_revision,
    )
    metadata = mapping(params.get("litellm_metadata")) or mapping(params.get("metadata"))
    selected = metadata.get(CONTEXT)
    proof = selected_oauth_proof(
        selected, declaration, snapshot.key_digest, account_digest, snapshot.api_base, responses=responses
    )
    details[DISPATCH] = proof
    details[STAMP] = proof
