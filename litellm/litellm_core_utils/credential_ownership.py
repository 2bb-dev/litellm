"""Server-only, per-attempt credential ownership; absence of proof is unknown."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

FIELD = "openorange_credential_ownership"
CONTEXT = "_openorange_credential_selection"
STAMP = "_openorange_credential_stamp"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$")
_MAPPING = TypeAdapter(Mapping[str, object])
_LIST = TypeAdapter(list[object])


class Registration(BaseModel):
    """Explicit control-plane declaration, never inference request metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    v: Literal[1]
    source: Literal["platform", "byok"]
    registration_id: str
    registration_revision: str


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return {}


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value.lower()
    except ValueError:
        return False


def _registration(value: object) -> Registration | None:
    try:
        registration = Registration.model_validate(value)
    except ValidationError:
        return None
    if type(_mapping(value).get("v")) is not int:
        return None
    if not _uuid(registration.registration_id) or not _uuid(registration.registration_revision):
        return None
    return registration


def updated_credential_info(
    current: Mapping[str, object], patch: Mapping[str, object], *, binding_changed: bool
) -> dict[str, object]:
    """A changed binding or declaration requires an explicit fresh revision."""
    previous = _registration(current.get(FIELD))
    proposed = _registration(patch.get(FIELD))
    renewed = proposed is not None and (
        previous is None or proposed.registration_revision != previous.registration_revision
    )
    merged = {**current, **patch}
    if binding_changed or FIELD in patch:
        return {key: value for key, value in merged.items() if key != FIELD or renewed}
    return merged


def unknown(deployment_id: object = None, provenance: str = "unregistered") -> dict[str, JsonValue]:
    return {
        "v": 1,
        "source": "unknown",
        "provenance": provenance if provenance in {"unregistered", "credential_override", "ambiguous"} else "ambiguous",
        "registration_id": None,
        "registration_revision": None,
        "deployment_id": _identifier(deployment_id),
    }


def safe_ownership(value: object, deployment_id: object) -> dict[str, JsonValue]:
    """Bounded projection of an already stamped row, not authentication of JSON."""
    data = _mapping(value)
    selected = _identifier(deployment_id)
    if data.keys() != unknown().keys() or type(data.get("v")) is not int or data.get("v") != 1:
        return unknown(selected)
    if data.get("deployment_id") != selected:
        return unknown(selected, "ambiguous")
    if data.get("source") == "unknown":
        return unknown(selected, str(data.get("provenance")))
    registration = _registration({key: data.get(key) for key in Registration.model_fields})
    if (
        registration is None
        or selected is None
        or data.get("provenance") not in {"credential_registration", "route_registration"}
    ):
        return unknown(selected, "ambiguous")
    return {
        "v": 1,
        "source": registration.source,
        "provenance": str(data["provenance"]),
        "registration_id": registration.registration_id,
        "registration_revision": registration.registration_revision,
        "deployment_id": selected,
    }


@dataclass(frozen=True, slots=True)
class _Selection:
    deployment_id: str | None
    model: str | None
    credential_name: str | None = field(repr=False)
    registration: Registration | None
    key_digest: bytes | None = field(repr=False)
    api_base: str | None = field(repr=False)
    rejection: str | None


@dataclass(frozen=True, slots=True)
class _Stamp:
    source: Literal["platform", "byok", "unknown"]
    provenance: str
    registration_id: str | None
    registration_revision: str | None
    deployment_id: str | None

    def fact(self) -> dict[str, JsonValue]:
        return {
            "v": 1,
            "source": self.source,
            "provenance": self.provenance,
            "registration_id": self.registration_id,
            "registration_revision": self.registration_revision,
            "deployment_id": self.deployment_id,
        }


def _digest(value: object) -> bytes | None:
    if not isinstance(value, str) or not value or value.startswith("os.environ/"):
        return None
    return hashlib.sha256(value.encode()).digest()


def _auth_fields() -> frozenset[str]:
    from litellm.router_utils.clientside_credential_handler import credential_override_fields

    return credential_override_fields() | {
        "api_key",
        "api_base",
        "base_url",
        "client",
        "http_client",
        "headers",
        "litellm_credential_name",
        "custom_llm_provider",
        "api_token",
        "token",
        "shared_session",
    }


def select_credential(deployment: Mapping[str, object], request: Mapping[str, object], deployment_id: object) -> object:
    """Router-owned selection snapshot. Every fallback must replace the snapshot."""
    params = _mapping(deployment.get("litellm_params"))
    info = _mapping(deployment.get("model_info"))
    name = params.get("litellm_credential_name")
    model = params.get("model")
    base = params.get("api_base")
    # The initial slice proves simple API-key transports only. Dynamic/custom auth
    # cannot inherit a static registration merely because the route has one.
    unsupported = any(
        params.get(key) is not None for key in _auth_fields() - {"api_key", "api_base", "litellm_credential_name"}
    )
    return _Selection(
        deployment_id=_identifier(deployment_id),
        model=model if isinstance(model, str) else None,
        credential_name=name if isinstance(name, str) else None,
        registration=_registration(info.get(FIELD)),
        key_digest=_digest(params.get("api_key")),
        api_base=base if isinstance(base, str) else None,
        rejection=(
            "credential_override"
            if any(key in request for key in _auth_fields())
            else "ambiguous"
            if unsupported or (info.get("db_model") is True and not isinstance(name, str))
            else None
        ),
    )


def _stamp(selection: _Selection | None, registration: Registration | None, provenance: str) -> _Stamp:
    return _Stamp(
        source=registration.source if registration else "unknown",
        provenance=provenance,
        registration_id=registration.registration_id if registration else None,
        registration_revision=registration.registration_revision if registration else None,
        deployment_id=selection.deployment_id if selection else None,
    )


def resolve_ownership(
    request: Mapping[str, object],
    credential_values: Mapping[str, object],
    credential_info: Mapping[str, object],
    *,
    credential_ambiguous: bool = False,
) -> object:
    """Called beside credential loading, using the same registry-item snapshot."""
    metadata = _mapping(request.get("litellm_metadata")) or _mapping(request.get("metadata"))
    candidate = metadata.get(CONTEXT)
    selection = candidate if type(candidate) is _Selection else None
    if selection is None:
        return _stamp(None, None, "unregistered")
    if selection.rejection:
        return _stamp(selection, None, selection.rejection)
    if selection.deployment_id is None or credential_ambiguous:
        return _stamp(selection, None, "ambiguous")
    import litellm

    if (
        litellm.headers
        or litellm.client_session is not None
        or litellm.aclient_session is not None
        or getattr(litellm, "network_mock", False)
        or any(request.get(key) is not None for key in ("mock_response", "mock_tool_calls", "mock_timeout"))
    ):
        return _stamp(selection, None, "ambiguous")
    if request.get("model") != selection.model or not (selection.model or "").startswith(("openai/", "venice/")):
        return _stamp(selection, None, "ambiguous")
    if request.get("litellm_credential_name") != selection.credential_name:
        return _stamp(selection, None, "credential_override")
    named = selection.credential_name is not None
    registration = _registration(credential_info.get(FIELD)) if named else selection.registration
    if registration is None:
        return _stamp(selection, None, "unregistered")
    if named and (selection.registration is not None or selection.key_digest is not None):
        return _stamp(selection, None, "ambiguous")
    if any(
        request.get(key) is not None
        for key in _auth_fields() - {"api_key", "api_base", "litellm_credential_name", "client"}
    ):
        return _stamp(selection, None, "credential_override")
    if named and any(key not in {"api_key", "api_base"} for key in credential_values):
        return _stamp(selection, None, "ambiguous")
    expected_key = _digest(credential_values.get("api_key")) if named else selection.key_digest
    actual_key = _digest(request.get("api_key", credential_values.get("api_key")))
    expected_base = selection.api_base or (credential_values.get("api_base") if named else None)
    actual_base = request.get("api_base", credential_values.get("api_base"))
    if (
        expected_key is None
        or not isinstance(expected_base, str)
        or not expected_base
        or actual_key != expected_key
        or actual_base != expected_base
    ):
        return _stamp(selection, None, "credential_override")
    client_rejection = _client_rejection(request.get("client"), expected_key, expected_base)
    if client_rejection:
        return _stamp(selection, None, client_rejection)
    return _stamp(selection, registration, "credential_registration" if named else "route_registration")


def _client_rejection(client: object, expected_key: bytes, expected_base: str) -> str | None:
    if client is not None:
        # Only these concrete SDK clients have the inspected API-key contract.
        import httpx
        from openai import AsyncOpenAI, OpenAI
        from openai._base_client import AsyncHttpxClientWrapper, SyncHttpxClientWrapper

        if not isinstance(client, (OpenAI, AsyncOpenAI)) or type(client) not in {OpenAI, AsyncOpenAI}:
            return "ambiguous"
        if _mapping(vars(client)).get("_custom_headers"):
            return "ambiguous"
        http_client = _mapping(vars(client)).get("_client")
        if not isinstance(http_client, (httpx.Client, httpx.AsyncClient)):
            return "ambiguous"
        if type(http_client) not in {httpx.Client, httpx.AsyncClient, SyncHttpxClientWrapper, AsyncHttpxClientWrapper}:
            return "ambiguous"
        http_state = _mapping(vars(http_client))
        if (
            http_state.get("_auth") is not None
            or any(http_client.event_hooks.values())
            or type(http_state.get("_transport")) not in {httpx.HTTPTransport, httpx.AsyncHTTPTransport}
            or any(
                transport is not None and type(transport) not in {httpx.HTTPTransport, httpx.AsyncHTTPTransport}
                for transport in _mapping(http_state.get("_mounts")).values()
            )
        ):
            return "ambiguous"
        if _digest(client.api_key) != expected_key or str(client.base_url).rstrip("/") != expected_base.rstrip("/"):
            return "credential_override"
        if client.default_headers.get("Authorization") != f"Bearer {client.api_key}":
            return "credential_override"
    return None


def ownership_for_spend(value: object, deployment_id: object) -> dict[str, JsonValue]:
    """JSON callers cannot construct the private Python stamp type."""
    return safe_ownership(value.fact(), deployment_id) if type(value) is _Stamp else unknown(deployment_id)


def strip_ownership(value: object, depth: int = 0) -> object:
    """Remove reserved caller copies, including nested metadata and error objects."""
    if depth >= DEFAULT_MAX_RECURSE_DEPTH:
        return None
    mapping = _mapping(value)
    if isinstance(value, Mapping):
        return {
            key: strip_ownership(item, depth + 1) for key, item in mapping.items() if key not in {FIELD, CONTEXT, STAMP}
        }
    if isinstance(value, list):
        return [strip_ownership(item, depth + 1) for item in _LIST.validate_python(value)]
    return value
