"""Actual native Router dispatch through the Platform authority's signer/store."""

import asyncio
import base64
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.terminal_receipt_client import Authority, Settings
from litellm.litellm_core_utils.terminal_receipt_evidence import Envelope, FIELD, parse_jws, verify_set
from litellm.litellm_core_utils.terminal_usage_evidence import (
    FIELD as USAGE_EVIDENCE_FIELD,
    UsageEnvelope,
    parse_usage,
    verify_usage_set,
)
from litellm.litellm_core_utils.terminal_usage_observation import FIELD as USAGE_FIELD, UsageSnapshot
from litellm.proxy.spend_tracking.request_content_metadata import protect_spend_payload
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from tests.test_litellm.litellm_core_utils.test_native_credential_ownership import response_body, route

pytest_plugins = ["tests.test_litellm.litellm_core_utils.test_native_credential_ownership"]


@pytest.fixture
def authority_server(tmp_path):
    checkout = os.environ.get("OPENORANGE_TEST_PLATFORM_CHECKOUT")
    if not checkout:
        pytest.skip("requires the coordinated Platform terminal-receipt source checkout")
    snapshot = tmp_path / "platform-snapshot"
    source_dir = Path(checkout) / "src/server/terminal-receipts"
    target = snapshot / "src/server/terminal-receipts"
    target.mkdir(parents=True)
    (snapshot / "package.json").write_text('{"type":"module"}')
    platform_hashes = {}
    for source in sorted(source_dir.glob("*.ts")):
        if source.name.endswith(".test.ts"):
            continue
        raw = source.read_bytes()
        (target / source.name).write_bytes(raw)
        platform_hashes[source.name] = hashlib.sha256(raw).hexdigest()
    platform_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    issuer, kid, audience, producer, registrar, sidecar = [str(uuid4()) for _ in range(6)]
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    tokens = {role: "synthetic_" + uuid4().hex for role in ("audience", "producer", "registrar", "sidecar")}
    config = {
        "v": 1,
        "issuer": issuer,
        "kid": kid,
        "private_key": private,
        "public_keys": [{"kid": kid, "public_key": public}],
        "database_path": str(tmp_path / "authority.sqlite"),
        "max_correlations": 100,
        "max_registrations": 100,
        "principals": [
            {
                "id": identity,
                "role": "producer" if role == "sidecar" else role,
                "token_sha256": hashlib.sha256(tokens[role].encode()).hexdigest(),
                "audience": audience if role == "audience" else None,
                "delegates": [sidecar] if role == "producer" else [],
            }
            for role, identity in (
                ("audience", audience),
                ("producer", producer),
                ("registrar", registrar),
                ("sidecar", sidecar),
            )
        ],
    }
    config_file = tmp_path / "authority.json"
    config_file.write_text(json.dumps(config))
    config_file.chmod(0o600)
    error_file = (tmp_path / "authority.stderr").open("w+")
    process = subprocess.Popen(
        ["node", "--import", "tsx", str(Path(__file__).with_name("terminal_authority_server.mjs"))],
        cwd=checkout,
        env={
            **os.environ,
            "OPENORANGE_TERMINAL_RECEIPTS_CONFIG_FILE": str(config_file),
            "OPENORANGE_TEST_PLATFORM_CHECKOUT": str(snapshot),
        },
        stdout=subprocess.PIPE,
        stderr=error_file,
        text=True,
    )
    try:
        line = process.stdout.readline()
        if not line:
            error_file.seek(0)
            pytest.fail("Platform authority did not start: " + error_file.read())
        base = f"http://127.0.0.1:{json.loads(line)['port']}/api/v1/inference/terminal-receipts"
        shared = {"v": 1, "issuer": issuer, "base_url": base, "public_keys": {kid: public}}
        terminal = Authority(
            Settings.model_validate_json(
                json.dumps({**shared, "role": "producer", "identity": producer, "bearer": tokens["producer"]})
            )
        )
        yield {
            "source_manifest": {"platform_head": platform_head, "platform_modules": platform_hashes},
            "terminal": terminal,
            "shared": shared,
            "tokens": tokens,
            "audience": audience,
            "producer": producer,
            "sidecar": sidecar,
            "base": base,
            "trust": {
                "v": 1,
                "audience": audience,
                "issuers": [{"issuer": issuer, "keys": [{"kid": kid, "public_key_pem": public}]}],
            },
        }
    finally:
        process.terminate()
        process.wait(timeout=10)
        error_file.close()


class SpendCapture(CustomLogger):
    def __init__(self):
        self.rows = []

    def capture(self, kwargs, response, start, end):
        row = protect_spend_payload(get_logging_payload(kwargs, response, start, end))
        metadata = json.loads(row["metadata"])
        if FIELD in metadata:
            self.rows.append(row)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.capture(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

        await _ProxyDBLogger.async_log_failure_event(self, kwargs, response_obj, start_time, end_time)


@contextmanager
def router_server(event_loop):
    routers = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

            async def complete():
                response = await routers[0].acompletion(
                    model=body["model"],
                    messages=body["messages"],
                    stream=body.get("stream", False),
                    stream_options=body.get("stream_options"),
                    proxy_server_request={"headers": dict(self.headers)},
                )
                if body.get("stream"):
                    chunks = [chunk async for chunk in response]
                    data = "".join("data: " + chunk.model_dump_json() + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                else:
                    data = response.model_dump_json()
                return data.encode()

            try:
                data = asyncio.run_coroutine_threadsafe(complete(), event_loop).result(timeout=15)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(500)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def close_loop_clients():
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    for key, value in tuple(getattr(cache, "cache_dict", {}).items()):
        if key.endswith("-" + str(id(asyncio.get_running_loop()))) and isinstance(value, AsyncHTTPHandler):
            await value.close()
            cache.cache_dict.pop(key, None)


@pytest_asyncio.fixture(loop_scope="function")
async def central_server():
    with router_server(asyncio.get_running_loop()) as server:
        yield server


@pytest_asyncio.fixture(loop_scope="function")
async def sidecar_server():
    with router_server(asyncio.get_running_loop()) as server:
        yield server


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def cleanup_clients():
    yield
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await GLOBAL_LOGGING_WORKER.stop()
    await close_loop_clients()


async def run_router_platform_case(
    monkeypatch,
    tmp_path,
    authority_server,
    native_server,
    central_server,
    provider,
    fallback,
    stream,
    first_source="platform",
    delegated_server=None,
    oauth_fault=None,
    local_fallback=False,
    disconnect=False,
    finish_delay=0,
    reserved_headers=False,
    client_disconnect=False,
    cancelled_scope=False,
    usage_case=None,
    supplier_partial=False,
):
    upstream, calls, replies = native_server
    central, routers = central_server
    capture = SpendCapture()
    from litellm.proxy.proxy_server import proxy_logging_obj

    async def capture_failure_write(**values):
        capture.capture(values["kwargs"], values["completion_response"], values["start_time"], values["end_time"])

    monkeypatch.setattr(
        proxy_logging_obj, "db_spend_update_writer", SimpleNamespace(update_database=capture_failure_write)
    )
    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "input_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "callbacks", [capture])
    native_provider = "deepseek" if provider == "chatgpt" else provider
    routes = [route(native_provider, first_source, upstream, 1), route(native_provider, "byok", upstream, 2)]
    if provider == "chatgpt":
        for entry in routes:
            entry["litellm_params"]["model"] = "chatgpt/gpt-6-astra"
            entry["model_info"]["mode"] = "responses"
        profile = tmp_path / "oauth"
        profile.mkdir()
        (profile / "auth.json").write_text(
            json.dumps(
                {
                    "access_token": "synthetic-selected-oauth-token",
                    "account_id": "synthetic-authorized-account",
                    "expires_at": 4102444800,
                }
            )
        )
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(profile))
        monkeypatch.setenv("CHATGPT_API_BASE", upstream)
    for entry in routes:
        declaration = entry["model_info"]["openorange_credential_ownership"]
        declaration["registration_id"] = str(uuid4())
        declaration["registration_revision"] = str(uuid4())
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                authority_server["base"] + "/registrations",
                headers={"Authorization": "Bearer " + authority_server["tokens"]["registrar"]},
                json={
                    "registration_id": declaration["registration_id"],
                    "registration_revision": declaration["registration_revision"],
                    "producer_id": authority_server["sidecar"] if delegated_server else authority_server["producer"],
                    "deployment_id": entry["model_info"]["id"],
                    "source": declaration["source"],
                },
            )
            assert response.status_code == 200, response.text
    if provider == "chatgpt":
        registration = tmp_path / "oauth-registration.json"
        registration.write_text(
            json.dumps(
                {
                    **routes[0]["model_info"]["openorange_credential_ownership"],
                    "account_sha256": hashlib.sha256(b"synthetic-authorized-account").hexdigest(),
                    "api_base": upstream,
                }
            )
        )
        registration.chmod(0o600)
        monkeypatch.setenv("OPENORANGE_CHATGPT_CREDENTIAL_REGISTRATION_FILE", str(registration))
        if oauth_fault == "missing_registration":
            registration.unlink()
        if oauth_fault in {"account_change", "token_rotation"}:
            data = json.loads((profile / "auth.json").read_text())
            data["account_id" if oauth_fault == "account_change" else "access_token"] = "synthetic-rotated-value"
            (profile / "auth.json").write_text(json.dumps(data))
        if oauth_fault in {"late_auth", "late_account"}:
            from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig

            original_sign = ChatGPTResponsesAPIConfig.sign_request

            def override(self, **kwargs):
                headers, signed = original_sign(self, **kwargs)
                key = "Authorization" if oauth_fault == "late_auth" else "ChatGPT-Account-Id"
                return {
                    **{name: value for name, value in headers.items() if name.lower() != key.lower()},
                    key: "Bearer synthetic-late" if oauth_fault == "late_auth" else "synthetic-late",
                }, signed

            monkeypatch.setattr(ChatGPTResponsesAPIConfig, "sign_request", override)
    terminal_authority = authority_server["terminal"]
    if delegated_server:
        terminal_authority = Authority(
            Settings.model_validate_json(
                json.dumps(
                    {
                        **authority_server["shared"],
                        "role": "producer",
                        "identity": authority_server["sidecar"],
                        "bearer": authority_server["tokens"]["sidecar"],
                    }
                )
            )
        )
    if finish_delay:

        class DelayedFinishAuthority(Authority):
            def request(self, method, path, body=None):
                if path == "/attempts/finish-with-usage":
                    import time

                    time.sleep(finish_delay)
                return super().request(method, path, body)

        terminal_authority = DelayedFinishAuthority(terminal_authority.settings)
    if reserved_headers:
        for entry in routes:
            entry["litellm_params"]["extra_headers"] = {
                "X-OpenOrange-Terminal-Ingress": "synthetic-caller-copy",
                "x-openorange-terminal-central-request": "synthetic-caller-copy",
            }
    terminal_router = litellm.Router(
        model_list=routes,
        num_retries=0,
        fallbacks=[] if local_fallback else [{"group-1": ["group-2"]}],
        terminal_receipt_authority=terminal_authority,
    )
    if delegated_server:
        sidecar_base, sidecar_routers = delegated_server
        sidecar_routers.append(terminal_router)
        relay_authority = Authority(
            authority_server["terminal"].settings.model_copy(
                update={
                    "upstreams": (sidecar_base,),
                    "delegates": {"central-proxy-connection": authority_server["sidecar"]},
                }
            )
        )
        routers.append(
            litellm.Router(
                model_list=[
                    {
                        "model_name": "group-1",
                        "model_info": {"id": "central-proxy-connection"},
                        "litellm_params": {
                            "model": "litellm_proxy/group-1",
                            "api_base": sidecar_base,
                            "api_key": "synthetic-central-connection-key",
                        },
                    }
                ],
                num_retries=0,
                terminal_receipt_authority=relay_authority,
            )
        )
    else:
        routers.append(terminal_router)
    local_authority = Authority(
        Settings.model_validate_json(
            json.dumps(
                {
                    **authority_server["shared"],
                    "role": "audience",
                    "identity": authority_server["audience"],
                    "bearer": authority_server["tokens"]["audience"],
                    "upstreams": [central],
                }
            )
        )
    )
    local = litellm.Router(
        model_list=[
            {
                "model_name": "managed",
                "model_info": {"id": "local-proxy-deployment"},
                "litellm_params": {
                    "model": "litellm_proxy/group-1",
                    "api_key": "synthetic-proxy-connection-key",
                    "api_base": central,
                },
            },
            {
                "model_name": "managed-fallback",
                "model_info": {"id": "local-fallback-deployment"},
                "litellm_params": {
                    "model": "litellm_proxy/group-2",
                    "api_key": "synthetic-proxy-key-2",
                    "api_base": central,
                },
            },
        ],
        num_retries=0,
        fallbacks=[{"managed": ["managed-fallback"]}] if local_fallback else [],
        terminal_receipt_authority=local_authority,
    )
    if supplier_partial:
        partial = response_body(native_provider)
        partial["usage"].update(
            {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
            if native_provider == "anthropic"
            else {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 9}
        )
        replies.append((200, {**partial, "_test_stream_disconnect": True}))
    elif disconnect:
        replies.append((0, {}))
    elif fallback:
        replies.append((429, {"error": {"type": "rate_limit_error", "message": "synthetic"}}))
    body = response_body(native_provider)
    if usage_case == "missing":
        body.pop("usage", None)
    elif usage_case == "zero":
        body["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elif usage_case == "explicit_cache":
        body["usage"].update({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    elif usage_case == "unestablished_writes":
        body["usage"].update(
            {
                "prompt_cache_hit_tokens": 0,
                "cache_creation_input_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 7},
                "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 2},
            }
        )
        body["_test_unestablished_writes"] = True
    elif usage_case in {"responses_write_zero", "responses_write_nonzero"}:
        body["_test_responses_write"] = 0 if usage_case == "responses_write_zero" else 7
    if provider == "chatgpt":
        body["model"] = "gpt-6-astra"
    replies.append((200, body))
    response = await local.acompletion(
        model="managed",
        messages=[{"role": "user", "content": "synthetic"}],
        stream=stream,
        metadata={FIELD: {"state": "forged"}},
        extra_headers={
            "X-OpenOrange-Terminal-Ingress": "synthetic-caller-copy",
            "x-openorange-terminal-central-request": "synthetic-caller-copy",
        },
    )
    if client_disconnect:
        for _ in range(8):
            chunk = await response.__anext__()
            if any(choice.delta.content for choice in chunk.choices):
                break
        if cancelled_scope:
            import anyio

            with anyio.CancelScope() as scope:
                scope.cancel()
                await response.aclose()
        else:
            await response.aclose()
    elif stream:
        response = litellm.stream_chunk_builder([chunk async for chunk in response])
    for _ in range(100):
        if len({item["request_id"] for item in capture.rows}) >= (2 if local_fallback else 1):
            break
        await asyncio.sleep(0.02)
    assert capture.rows, "actual SpendLogs writer did not receive private terminal evidence"
    row = next((item for item in capture.rows if item["model_id"] == "local-fallback-deployment"), capture.rows[-1])
    envelope = Envelope.model_validate_json(json.dumps(json.loads(row["metadata"])[FIELD]))
    assert envelope.local_attempt_id == row["request_id"]
    expected_local = "local-fallback-deployment" if local_fallback else "local-proxy-deployment"
    assert envelope.local_deployment_id == row["model_id"] == expected_local, [
        [
            (item["request_id"], item["model_id"], item["status"], json.loads(item["metadata"])[FIELD]["state"])
            for item in capture.rows
        ],
        [(call["path"], call["body"].get("stream")) for call in calls],
        [
            (parse_jws(token).claims.event, getattr(parse_jws(token).claims, "outcome", None))
            for token in envelope.receipts
        ],
    ]
    if local_fallback:
        envelopes = [
            Envelope.model_validate_json(json.dumps(json.loads(item["metadata"])[FIELD])) for item in capture.rows
        ]
        assert len({item.local_attempt_id for item in envelopes}) == 2, [
            (item["request_id"], item["model_id"], item["status"]) for item in capture.rows
        ]
        assert len({item.local_request_id for item in envelopes}) == 1
        earlier = next(item for item in envelopes if item.local_attempt_id != envelope.local_attempt_id)
        assert earlier.local_deployment_id == "local-proxy-deployment"
        assert earlier.correlation_id != envelope.correlation_id
        assert earlier.receipts, "failed local Router attempt must remain independently retained"
        assert all(parse_jws(token).claims.local_attempt_id == earlier.local_attempt_id for token in earlier.receipts)
        if supplier_partial:
            failed_row = next(item for item in capture.rows if item["request_id"] == earlier.local_attempt_id)
            failed_metadata = json.loads(failed_row["metadata"])
            failed_measurement = UsageEnvelope.model_validate_json(json.dumps(failed_metadata[USAGE_EVIDENCE_FIELD]))
            assert (
                verify_usage_set(failed_measurement, earlier, local_authority.settings.issuer, local_authority.keys)
                == "complete"
            )
            failed_usage = parse_usage(failed_measurement.measurements[0]).claims.usage
            assert failed_usage.state == "partial"
            assert failed_usage.prompt_tokens == 9
            assert failed_usage.cache_read_tokens == 0
            assert failed_usage.cache_write_tokens == (0 if provider == "anthropic" else None)
            assert failed_metadata[USAGE_FIELD]["state"] == "partial"
            assert (
                next(
                    parse_jws(token).claims for token in earlier.receipts if parse_jws(token).claims.event == "finished"
                ).outcome
                == "failure"
            )
    assert envelope.state == "complete", envelope
    assert verify_set(envelope, local_authority.settings.issuer, local_authority.keys) == "complete"
    for name in ("audience", "local_request_id", "local_attempt_id", "correlation_id"):
        assert (
            verify_set(
                envelope.model_copy(update={name: str(uuid4())}), local_authority.settings.issuer, local_authority.keys
            )
            == "invalid"
        )
    assert (
        verify_set(
            envelope.model_copy(update={"local_deployment_id": "other-local"}),
            local_authority.settings.issuer,
            local_authority.keys,
        )
        == "invalid"
    )
    assert (
        verify_set(
            envelope.model_copy(update={"receipts": envelope.receipts[:-1]}),
            local_authority.settings.issuer,
            local_authority.keys,
        )
        == "pending"
    )
    assert (
        verify_set(
            envelope.model_copy(update={"receipts": envelope.receipts[1:]}),
            local_authority.settings.issuer,
            local_authority.keys,
        )
        == "invalid"
    )
    assert (
        verify_set(
            envelope.model_copy(update={"receipts": (envelope.receipts[0], *envelope.receipts)}),
            local_authority.settings.issuer,
            local_authority.keys,
        )
        == "invalid"
    )
    first = envelope.receipts[0].split(".")
    signature = bytearray(base64.urlsafe_b64decode(first[2] + "=" * (-len(first[2]) % 4)))
    signature[0] ^= 1
    tampered = ".".join((*first[:2], base64.urlsafe_b64encode(signature).rstrip(b"=").decode()))
    assert (
        verify_set(
            envelope.model_copy(update={"receipts": (tampered, *envelope.receipts[1:])}),
            local_authority.settings.issuer,
            local_authority.keys,
        )
        == "invalid"
    )
    events = [parse_jws(value).claims for value in envelope.receipts]
    finished = [event for event in events if event.event == "finished"]
    measurement = UsageEnvelope.model_validate_json(json.dumps(json.loads(row["metadata"])[USAGE_EVIDENCE_FIELD]))
    assert measurement.state == "complete", measurement
    assert verify_usage_set(measurement, envelope, local_authority.settings.issuer, local_authority.keys) == "complete"
    assert len(measurement.measurements) == len(finished)
    observations = [parse_usage(token).claims.usage for token in measurement.measurements]
    scalar = json.loads(row["metadata"])[USAGE_FIELD]
    if usage_case == "unestablished_writes":
        assert observations[0].prompt_tokens == 9
        assert observations[0].completion_tokens == 3
        assert observations[0].cache_read_tokens == 0
        assert observations[0].cache_write_tokens is None
        assert observations[0].cache_write_5m_tokens is None
        assert observations[0].cache_write_1h_tokens is None
    if usage_case in {"responses_write_zero", "responses_write_nonzero"}:
        assert observations[0].prompt_tokens == 9
        assert observations[0].completion_tokens == 3
        assert observations[0].cache_read_tokens == 0
        assert observations[0].cache_write_tokens == (0 if usage_case == "responses_write_zero" else 7)
        assert observations[0].cache_write_5m_tokens is None
        assert observations[0].cache_write_1h_tokens is None
    if len(finished) > 1:
        assert scalar["state"] == "unobserved"
        assert all(scalar[name] is None for name in UsageSnapshot.model_fields)
        assert observations[0].state == "unobserved"
    elif usage_case == "missing":
        assert observations[0].state == scalar["state"] == "unobserved"
        assert all(scalar[name] is None for name in UsageSnapshot.model_fields)
    else:
        assert observations[0].state == "observed", observations
        assert scalar["state"] == ("partial" if client_disconnect else "observed"), scalar
        assert all(scalar[name] == getattr(observations[0], name) for name in UsageSnapshot.model_fields)
    expected_source = (
        "unknown"
        if oauth_fault in {"missing_registration", "account_change", "late_auth", "late_account"}
        else first_source
    )
    if disconnect or reserved_headers:
        expected_source = "unknown"
    expected_sources = ["byok"] if local_fallback else ([expected_source, "byok"] if fallback else [expected_source])
    assert [event.ownership.source for event in finished] == expected_sources
    assert [event.outcome for event in finished] == (
        ["failure", "success"] if fallback and not local_fallback else ["success"]
    )
    if delegated_server:
        assert all(event.producer_id == authority_server["sidecar"] for event in finished)
        assert all(event.terminal.deployment_id != "central-proxy-connection" for event in finished)
    if client_disconnect:
        assert row["status"] == "failure"
        assert envelope.receipts
        assert row["completion_tokens"] == 0  # Provider usage has not arrived before this early close.
    elif usage_case == "zero":
        assert response.usage.total_tokens == 0
        assert scalar["prompt_tokens"] == scalar["completion_tokens"] == scalar["total_tokens"] == 0
        assert scalar["cache_write_tokens"] is None
    elif usage_case != "missing":
        assert response.usage.total_tokens == 12
    if usage_case == "explicit_cache":
        assert scalar["prompt_tokens"] == 9
        assert scalar["cache_read_tokens"] == scalar["cache_write_tokens"] == 0
        assert scalar["total_tokens"] is None
    assert all(
        not {"x-openorange-terminal-ingress", "x-openorange-terminal-central-request"}.intersection(
            key.lower() for key in call["headers"]
        )
        for call in calls
    )
    safe_row = {
        key: row[key]
        for key in (
            "request_id",
            "model_id",
            "model",
            "custom_llm_provider",
            "metadata",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "spend",
            "status",
        )
    }
    artifact = os.environ.get("OPENORANGE_TEST_TERMINAL_ARTIFACT_DIR")
    if artifact:
        directory = Path(artifact)
        directory.mkdir(parents=True, exist_ok=True)
        fork_root = Path(__file__).resolve().parents[3]
        modules = (
            "router.py",
            "utils.py",
            "litellm_core_utils/credential_ownership.py",
            "litellm_core_utils/litellm_logging.py",
            "litellm_core_utils/terminal_receipt_client.py",
            "litellm_core_utils/terminal_receipt_evidence.py",
            "litellm_core_utils/terminal_receipt_hooks.py",
            "litellm_core_utils/terminal_receipt_oauth.py",
            "litellm_core_utils/terminal_usage_evidence.py",
            "litellm_core_utils/terminal_usage_observation.py",
            "litellm_core_utils/streaming_handler.py",
            "llms/anthropic/chat/transformation.py",
            "llms/anthropic/chat/handler.py",
            "llms/openai/chat/gpt_transformation.py",
            "llms/openai/openai.py",
            "responses/streaming_iterator.py",
            "llms/chatgpt/authenticator.py",
            "llms/chatgpt/chat/transformation.py",
            "llms/chatgpt/responses/transformation.py",
            "llms/custom_httpx/http_handler.py",
            "llms/custom_httpx/aiohttp_transport.py",
            "llms/custom_httpx/llm_http_handler.py",
            "proxy/hooks/proxy_track_cost_callback.py",
            "proxy/utils.py",
            "proxy/db/db_spend_update_writer.py",
            "proxy/db/spend_log_queue.py",
            "proxy/spend_tracking/request_content_metadata.py",
            "proxy/spend_tracking/spend_tracking_utils.py",
        )
        manifest = {
            **authority_server["source_manifest"],
            "fork_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "fork_modules": {
                name: hashlib.sha256((fork_root / "litellm" / name).read_bytes()).hexdigest() for name in modules
            },
        }
        bundle = {
            "row": safe_row,
            "rows": [{key: item[key] for key in safe_row} for item in capture.rows],
            "trust": authority_server["trust"],
            "source_manifest": manifest,
            "case": {
                "provider": provider,
                "stream": stream,
                "fallback": fallback,
                "first_source": first_source,
                "delegated": bool(delegated_server),
                "oauth_fault": oauth_fault,
                "usage_case": usage_case,
                "local_fallback": local_fallback,
                "supplier_disconnect": disconnect,
                "supplier_partial": supplier_partial,
                "client_disconnect": client_disconnect,
                "cancelled_scope": cancelled_scope,
            },
        }
        path = directory / f"case-{envelope.local_attempt_id}.json"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
            output.write(json.dumps(bundle, sort_keys=True))
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_actual_router_platform_receipts(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider, fallback, stream
):
    await run_router_platform_case(
        monkeypatch, tmp_path, authority_server, native_server, central_server, provider, fallback, stream
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,usage_case,source",
    [
        ("deepseek", "missing", "byok"),
        ("deepseek", "zero", "byok"),
        ("anthropic", "explicit_cache", "byok"),
        ("anthropic", "explicit_cache", "platform"),
    ],
)
async def test_native_missing_usage_and_explicit_zero_stay_distinct(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider, usage_case, source
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        False,
        False,
        first_source=source,
        usage_case=usage_case,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
async def test_confirmed_byok_native_bundle(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        False,
        False,
        first_source="byok",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("stream", [False, True])
async def test_delegated_terminal_router_preserves_actual_supplier_binding(
    monkeypatch, tmp_path, authority_server, native_server, central_server, sidecar_server, provider, stream
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        True,
        stream,
        delegated_server=sidecar_server,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["platform", "byok"])
@pytest.mark.parametrize("stream", [False, True])
async def test_chatgpt_selected_account_through_delegated_router(
    monkeypatch, tmp_path, authority_server, responses_server, central_server, sidecar_server, source, stream
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        responses_server,
        central_server,
        "chatgpt",
        False,
        stream,
        first_source=source,
        delegated_server=sidecar_server,
    )


@pytest.fixture
def responses_server():
    calls, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "headers": dict(self.headers), "body": body})
            code, original = replies.pop(0)
            result = {
                "id": "resp_synthetic",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "model": original["model"],
                "output": [
                    {
                        "id": "msg_synthetic",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "synthetic", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": 9,
                    "output_tokens": 3,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
            if original.get("_test_unestablished_writes"):
                result["usage"].update(
                    {
                        "cache_creation_input_tokens": 7,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                            "cache_creation_tokens": 7,
                        },
                        "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 2},
                    }
                )
            if "_test_responses_write" in original:
                result["usage"]["input_tokens_details"]["cache_write_tokens"] = original["_test_responses_write"]
            events = [
                {
                    "type": "response.created",
                    "response": {**result, "status": "in_progress", "output": [], "usage": None},
                    "sequence_number": 0,
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": result["output"][0],
                    "sequence_number": 1,
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_synthetic",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "synthetic",
                    "sequence_number": 2,
                },
                {"type": "response.completed", "response": result, "sequence_number": 3},
            ]
            data = ("".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, replies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing_registration", "account_change", "token_rotation", "late_auth", "late_account"]
)
async def test_chatgpt_registration_and_actual_wire_changes(
    monkeypatch, tmp_path, authority_server, responses_server, central_server, sidecar_server, fault
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        responses_server,
        central_server,
        "chatgpt",
        False,
        False,
        first_source="byok",
        delegated_server=sidecar_server,
        oauth_fault=fault,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_local_router_fallback_has_distinct_immutable_source_rows(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider, disconnect
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        True,
        False,
        local_fallback=True,
        disconnect=disconnect,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
async def test_supplier_partial_stream_retains_measurements_before_local_fallback(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        True,
        True,
        local_fallback=True,
        supplier_partial=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
async def test_supplier_disconnect_remains_unknown_in_sealed_terminal_fallback(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider
):
    await run_router_platform_case(
        monkeypatch, tmp_path, authority_server, native_server, central_server, provider, True, False, disconnect=True
    )


@pytest.mark.asyncio
async def test_parent_seals_after_delayed_delegated_finish_without_response_sleep(
    monkeypatch, tmp_path, authority_server, native_server, central_server, sidecar_server
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        "anthropic",
        True,
        False,
        delegated_server=sidecar_server,
        finish_delay=0.25,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
async def test_explicit_forwarded_reserved_header_copies_never_reach_supplier(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        False,
        False,
        reserved_headers=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
@pytest.mark.parametrize("cancelled_scope", [False, True])
async def test_local_stream_close_retains_attempt_row_and_supplier_execution_evidence(
    monkeypatch, tmp_path, authority_server, native_server, central_server, provider, cancelled_scope
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        native_server,
        central_server,
        provider,
        False,
        True,
        client_disconnect=True,
        cancelled_scope=cancelled_scope,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek", "chatgpt"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_unestablished_write_aliases_stay_null_in_signed_measurement(
    monkeypatch,
    tmp_path,
    authority_server,
    native_server,
    responses_server,
    central_server,
    sidecar_server,
    provider,
    stream,
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        responses_server if provider == "chatgpt" else native_server,
        central_server,
        provider,
        False,
        stream,
        delegated_server=sidecar_server if provider == "chatgpt" else None,
        usage_case="unestablished_writes",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_case", ["responses_write_zero", "responses_write_nonzero"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_responses_write_measurement_is_signed_before_sdk_defaults(
    monkeypatch,
    tmp_path,
    authority_server,
    responses_server,
    central_server,
    sidecar_server,
    usage_case,
    stream,
):
    await run_router_platform_case(
        monkeypatch,
        tmp_path,
        authority_server,
        responses_server,
        central_server,
        "chatgpt",
        False,
        stream,
        delegated_server=sidecar_server,
        usage_case=usage_case,
    )
