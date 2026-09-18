"""Server-only, per-attempt credential ownership; absence of proof is unknown."""

import hashlib
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID
from weakref import ReferenceType, ref

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

FIELD = "openorange_credential_ownership"
CONTEXT = "_openorange_credential_selection"
STAMP = "_openorange_credential_stamp"
DISPATCH = "_openorange_credential_dispatch"
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
    shared_session: ReferenceType[object] | None = field(default=None, repr=False)


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


@dataclass(frozen=True, slots=True)
class _DispatchProof:
    stamp: _Stamp
    provider: Literal["anthropic", "deepseek", "chatgpt"]
    key_digest: bytes = field(repr=False)
    endpoint: str = field(repr=False)
    account_digest: bytes | None = field(default=None, repr=False)


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


def _plain_shared_session(session: object) -> bool:
    from aiohttp import ClientRequest, ClientResponse, ClientSession, TCPConnector

    if type(session) is not ClientSession or not isinstance(session, ClientSession) or session.closed:
        return False
    state = _mapping(vars(session))
    return not (
        session.auth is not None
        or session.headers
        or session.trust_env
        or session.trace_configs
        or session.cookie_jar
        or state.get("_middlewares")
        or state.get("_default_proxy") is not None
        or state.get("_default_proxy_auth") is not None
        or state.get("_base_url") is not None
        or state.get("_request_class") is not ClientRequest
        or state.get("_response_class") is not ClientResponse
        or "request" in state
        or "_request" in state
        or type(session.connector) is not TCPConnector
    )


def _proxy_shared_session(session: object) -> bool:
    proxy = sys.modules.get("litellm.proxy.proxy_server")
    return (
        session is not None
        and proxy is not None
        and session is getattr(proxy, "shared_aiohttp_session", None)
        and _plain_shared_session(session)
    )


def select_credential(deployment: Mapping[str, object], request: Mapping[str, object], deployment_id: object) -> object:
    """Router-owned selection snapshot. Every fallback must replace the snapshot."""
    params = _mapping(deployment.get("litellm_params"))
    info = _mapping(deployment.get("model_info"))
    name = params.get("litellm_credential_name")
    model = params.get("model")
    base = params.get("api_base")
    shared_session = request.get("shared_session")
    proxy_session = _proxy_shared_session(shared_session)
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
            if any(key in request for key in _auth_fields() if key != "shared_session" or not proxy_session)
            else "ambiguous"
            if unsupported or (info.get("db_model") is True and not isinstance(name, str))
            else None
        ),
        shared_session=ref(shared_session) if proxy_session else None,
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
    # Inspected transports only. ``litellm_proxy/`` is the OpenAI-compatible
    # connection to a configured upstream proxy: a registration there attests
    # the connection credential and the local deployment, never the terminal
    # supplier credential behind that proxy.
    if request.get("model") != selection.model or not (selection.model or "").startswith(
        ("openai/", "veniceai/", "anthropic/", "deepseek/", "litellm_proxy/")
    ):
        return _stamp(selection, None, "ambiguous")
    if request.get("litellm_credential_name") != selection.credential_name:
        return _stamp(selection, None, "credential_override")
    named = selection.credential_name is not None
    registration = _registration(credential_info.get(FIELD)) if named else selection.registration
    if registration is None:
        return _stamp(selection, None, "unregistered")
    if named and (selection.registration is not None or selection.key_digest is not None):
        return _stamp(selection, None, "ambiguous")
    shared_session = selection.shared_session() if selection.shared_session is not None else None
    if request.get("shared_session") is not shared_session or (
        selection.shared_session is not None and not _proxy_shared_session(shared_session)
    ):
        return _stamp(selection, None, "credential_override")
    if any(
        request.get(key) is not None
        for key in _auth_fields() - {"api_key", "api_base", "litellm_credential_name", "client", "shared_session"}
    ):
        return _stamp(selection, None, "credential_override")
    if named and any(key not in {"api_key", "api_base"} for key in credential_values):
        return _stamp(selection, None, "ambiguous")
    expected_key = _digest(credential_values.get("api_key")) if named else selection.key_digest
    actual_key = _digest(request.get("api_key", credential_values.get("api_key")))
    expected_base = selection.api_base or (credential_values.get("api_base") if named else None)
    actual_base = request.get("api_base", credential_values.get("api_base"))
    if expected_key is None or actual_key != expected_key or actual_base != expected_base:
        return _stamp(selection, None, "credential_override")
    provenance = "credential_registration" if named else "route_registration"
    if (selection.model or "").startswith(("anthropic/", "deepseek/")):
        provider = "anthropic" if (selection.model or "").startswith("anthropic/") else "deepseek"
        return _DispatchProof(
            _stamp(selection, registration, provenance),
            provider,
            expected_key,
            _native_endpoint(provider, expected_base),
        )
    if not isinstance(expected_base, str) or not expected_base:
        return _stamp(selection, None, "credential_override")
    client_rejection = _client_rejection(request.get("client"), expected_key, expected_base)
    if client_rejection:
        return _stamp(selection, None, client_rejection)
    return _stamp(selection, registration, provenance)


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
            key: strip_ownership(item, depth + 1)
            for key, item in mapping.items()
            if key
            not in {
                FIELD,
                CONTEXT,
                STAMP,
                DISPATCH,
                "openorange_terminal_evidence",
                "openorange_usage_observation",
                "openorange_terminal_usage_evidence",
                "_openorange_terminal_context",
                "_openorange_terminal_stamp",
                "_openorange_terminal_root",
                "_openorange_terminal_oauth",
            }
            and key.lower() not in {"x-openorange-terminal-ingress", "x-openorange-terminal-central-request"}
        }
    if isinstance(value, list):
        return [strip_ownership(item, depth + 1) for item in _LIST.validate_python(value)]
    return value


def planned_ownership_for_terminal(value: object, deployment_id: object) -> dict[str, JsonValue]:
    selected = value.stamp if type(value) is _DispatchProof else value
    return ownership_for_spend(selected, deployment_id)


def selected_oauth_proof(
    value: object,
    registration: Registration,
    key_digest: bytes,
    account_digest: bytes,
    base: str,
    *,
    responses: bool = False,
) -> object:
    if type(value) is not _Selection:
        return _stamp(None, None, "unregistered")
    if (
        value.rejection is not None
        or value.deployment_id is None
        or not (value.model or "").startswith("chatgpt/")
        or value.credential_name is not None
        or (value.registration is not None and value.registration != registration)
    ):
        return _stamp(value, None, "ambiguous")
    return _DispatchProof(
        _stamp(value, registration, "credential_registration"),
        "chatgpt",
        key_digest,
        base.rstrip("/") + ("/responses" if responses else "/chat/completions"),
        account_digest,
    )


def _native_endpoint(provider: Literal["anthropic", "deepseek"], base: object) -> str:
    default = "https://api.anthropic.com" if provider == "anthropic" else "https://api.deepseek.com/beta"
    root = base.rstrip("/") if isinstance(base, str) and base else default
    suffix = "/v1/messages" if provider == "anthropic" else "/chat/completions"
    return root if root.endswith(suffix) else root + suffix


def _plain_native_transport(client: object) -> bool:
    import httpx

    if type(client) not in {httpx.Client, httpx.AsyncClient}:
        return False
    state = _mapping(vars(client))
    if state.get("_auth") is not None or any(_mapping(state.get("_event_hooks")).values()):
        return False
    transport = state.get("_transport")
    mounts = tuple(_mapping(state.get("_mounts")).values())
    if any(item is not None and type(item) not in {httpx.HTTPTransport, httpx.AsyncHTTPTransport} for item in mounts):
        return False
    if type(transport) in {httpx.HTTPTransport, httpx.AsyncHTTPTransport}:
        return True
    from aiohttp import ClientRequest, ClientResponse, ClientSession, TCPConnector

    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

    if not isinstance(transport, LiteLLMAiohttpTransport) or type(transport) is not LiteLLMAiohttpTransport:
        return False
    session = _mapping(vars(transport)).get("client")
    if not isinstance(session, ClientSession) or type(session) is not ClientSession:
        return False
    session_state = _mapping(vars(session))
    return not (
        session.auth is not None
        or session.trust_env
        or session.trace_configs
        or session_state.get("_middlewares")
        or session_state.get("_request_class") is not ClientRequest
        or session_state.get("_response_class") is not ClientResponse
        or type(session.connector) is not TCPConnector
        or session.headers.get("Authorization")
        or session.headers.get("x-api-key")
        or session.cookie_jar
    )


def ownership_after_http_response(value: object, request: object, client: object, response: object) -> object:
    """Verify native adapter authentication before HTTP status/stream processing.

    A transport exception without a verifiable final request stays unknown; HTTP
    errors with a response retain proof without implying supplier non-execution.
    """
    if type(value) is not _DispatchProof:
        return value
    import httpx

    pending = _Stamp("unknown", "ambiguous", None, None, value.stamp.deployment_id)
    if not isinstance(request, httpx.Request) or not isinstance(response, httpx.Response):
        return pending
    if response.history or response.request is not request or not _plain_native_transport(client):
        return pending
    if request.method != "POST" or str(request.url) != value.endpoint:
        return pending
    auth = request.headers.get_list("authorization")
    keys = request.headers.get_list("x-api-key")
    if value.provider == "chatgpt":
        accounts = request.headers.get_list("chatgpt-account-id")
        if len(accounts) != 1 or _digest(accounts[0]) != value.account_digest:
            return pending
    if len(auth) == 1 and not keys and auth[0].startswith("Bearer "):
        return value.stamp if _digest(auth[0][7:]) == value.key_digest else pending
    if value.provider == "anthropic" and len(keys) == 1 and not auth:
        return value.stamp if _digest(keys[0]) == value.key_digest else pending
    return pending
