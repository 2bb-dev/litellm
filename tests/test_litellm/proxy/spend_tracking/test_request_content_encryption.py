"""Synthetic-only interoperability, persistence and accounting regression tests."""

import asyncio
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from joserfc import jwe
from joserfc.jwk import RSAKey

import litellm
from litellm._logging import SecretRedactionFilter
from litellm.litellm_core_utils.request_content_mode import encryption_enabled, require_protection_marker
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.spend_log_queue import SQLiteSpendLogSpool
from litellm.proxy.spend_tracking.request_content_encryption import (
    CONFIG_ENV,
    INSTANCE_ENV,
    CaptureFailure,
    EncryptionKey,
    RequestContentEncryptor,
    configured_encryptor,
    encrypt_content,
    encryption_readiness,
    jwe_registry,
    parse_public_config,
)
from litellm.proxy.spend_tracking.request_content_metadata import protect_spend_payload, safe_metadata

FIXTURE = json.loads((Path(__file__).parent / "fixtures/browser_jwe_test_only.json").read_text())
CANARY = "SYNTHETIC_SECRET_DO_NOT_RETAIN_IN_PLAINTEXT with spaces"


@pytest.fixture
def protected_config(monkeypatch, tmp_path):
    from litellm.proxy.spend_tracking import request_content_encryption

    monkeypatch.setattr(request_content_encryption, "_runtime_failure", None)
    config = {"version": 1, "instanceUid": FIXTURE["instanceUid"], "publicKey": copy.deepcopy(FIXTURE["publicKey"])}
    path = tmp_path / "public.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv(CONFIG_ENV, str(path))
    monkeypatch.setenv(INSTANCE_ENV, FIXTURE["instanceUid"])
    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "spend.sqlite"))
    return config


def decrypt(envelope):
    result = jwe.decrypt_compact(envelope["jwe"], RSAKey.import_key(FIXTURE["privateKey"]), registry=jwe_registry())
    return result.protected, json.loads(result.plaintext)


def test_browser_jose_ciphertext_decrypts_with_python_joserfc():
    assert FIXTURE["testOnly"] is True
    header, content = decrypt(FIXTURE["envelope"])
    assert content == FIXTURE["content"]
    assert header == {
        "alg": "RSA-OAEP-256",
        "enc": "A256GCM",
        "typ": "openorange-request-content+jwe",
        "kid": FIXTURE["publicKey"]["kid"],
        "v": 1,
        "instanceUid": FIXTURE["instanceUid"],
        "purpose": "request-log",
        "recordId": FIXTURE["recordId"],
    }


def test_python_encrypts_with_public_key_only(protected_config):
    key = parse_public_config(json.dumps(protected_config).encode(), FIXTURE["instanceUid"])
    assert isinstance(key, EncryptionKey)
    assert not key.key.is_private
    first = encrypt_content(key, "test-row-1", FIXTURE["content"])
    second = encrypt_content(key, "test-row-1", FIXTURE["content"])
    assert not isinstance(first, CaptureFailure)
    assert not isinstance(second, CaptureFailure)
    assert first != second
    header, content = decrypt(first)
    assert header["recordId"] == "test-row-1"
    assert content == FIXTURE["content"]


@pytest.mark.parametrize("length", [697, 1024])
def test_wrapped_response_record_id_survives_capture(protected_config, monkeypatch, length):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    response, kwargs = make_call()
    record_id = "resp_" + "A" * (length - 5)
    response.id = record_id
    raw = get_logging_payload(
        kwargs, response, datetime(2026, 9, 7, tzinfo=timezone.utc), datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc)
    )
    protected = protect_spend_payload(raw)
    assert protected["request_id"] == record_id
    assert protected["spend"] == raw["spend"]
    assert json.loads(protected["metadata"])["openorange_request_log"]["content_status"] == "encrypted"
    envelope = json.loads(protected["proxy_server_request"])
    header, content = decrypt(envelope)
    assert header["recordId"] == record_id
    assert CANARY in json.dumps(content)
    assert CANARY not in json.dumps(protected, default=str)
    assert len(envelope["jwe"].split(".")[0]) <= 2048


@pytest.mark.parametrize("record_id", ["", "r" * 1025, "resp_\ninvalid", "resp_\x7finvalid"])
def test_request_record_id_limit_remains_fail_closed(protected_config, record_id):
    encryptor = configured_encryptor()
    assert isinstance(encryptor, RequestContentEncryptor)
    assert encryptor.encrypt(record_id, {"v": 1}) == CaptureFailure("invalid_record_id")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(version=2),
        lambda config: config.update(version=True),
        lambda config: config.update(instanceUid="different-instance"),
        lambda config: config["publicKey"].update(d="test-private-material"),
        lambda config: config["publicKey"].update(e="Aw"),
        lambda config: config["publicKey"].update(alg="RSA1_5"),
        lambda config: config["publicKey"].update(kid="wrong-thumbprint"),
        lambda config: config["publicKey"].update(key_ops=["decrypt"]),
        lambda config: config["publicKey"].update(n="=" + config["publicKey"]["n"]),
    ],
)
def test_rejects_invalid_public_config(protected_config, mutation):
    mutation(protected_config)
    assert isinstance(
        parse_public_config(json.dumps(protected_config).encode(), FIXTURE["instanceUid"]), CaptureFailure
    )


def test_rejects_noncanonical_instance_and_nonregular_config(protected_config, monkeypatch, tmp_path):
    assert isinstance(parse_public_config(json.dumps(protected_config).encode(), "not-a-uuid"), CaptureFailure)
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path))
    assert isinstance(configured_encryptor(), CaptureFailure)
    link = tmp_path / "public-link.json"
    link.symlink_to(tmp_path / "public.json")
    monkeypatch.setenv(CONFIG_ENV, str(link))
    assert isinstance(configured_encryptor(), CaptureFailure)


def test_missing_configuration_cannot_clear_protected_mode(protected_config, monkeypatch):
    assert require_protection_marker(FIXTURE["instanceUid"], FIXTURE["publicKey"]["kid"])
    monkeypatch.delenv(CONFIG_ENV)
    assert encryption_enabled()
    assert isinstance(encryption_readiness(), CaptureFailure)


def test_legacy_mode_without_marker_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.setenv("SPEND_LOG_DURABLE_QUEUE_PATH", str(tmp_path / "never-enabled.sqlite"))
    assert not encryption_enabled()
    assert encryption_readiness() is None


def test_initial_crypto_failure_recovers_without_touching_config(protected_config):
    with patch(
        "litellm.proxy.spend_tracking.request_content_encryption.jwe.encrypt_compact", side_effect=RuntimeError(CANARY)
    ):
        encryptor = configured_encryptor()
        assert isinstance(encryptor, RequestContentEncryptor)
        assert isinstance(encryption_readiness(), CaptureFailure)
    assert encryption_readiness() is None


def test_oversized_or_unserializable_content_and_invalid_id_fail_safely(protected_config, monkeypatch):
    from litellm.proxy.spend_tracking import request_content_encryption

    encryptor = configured_encryptor()
    assert isinstance(encryptor, RequestContentEncryptor)
    monkeypatch.setattr(request_content_encryption, "MAX_CONTENT_BYTES", 100)
    assert encryptor.encrypt("row", {"v": 1, "request": CANARY * 5}) == CaptureFailure("content_too_large")
    assert encryptor.encrypt("row", {"v": 1, "request": float("nan")}) == CaptureFailure("encryption_failed")
    assert encryptor.encrypt("row\n" + CANARY, {"v": 1}) == CaptureFailure("invalid_record_id")


def test_aes_gcm_tampering_is_detected():
    token = FIXTURE["envelope"]["jwe"].split(".")
    token[3] = ("A" if token[3][0] != "A" else "B") + token[3][1:]
    with pytest.raises(Exception):
        decrypt({"jwe": ".".join(token)})


def test_metadata_allowlist_drops_nested_content_and_arbitrary_numeric_keys():
    raw = {
        "status": "failure",
        "error_information": {"error_message": CANARY},
        "user_api_key_alias": CANARY,
        "openclaw_sender_name": CANARY,
        "usage_object": {
            "prompt_tokens": 6,
            "cache_read_input_tokens": 4,
            CANARY: 18,
            "prompt_tokens_details": {"cached_tokens": 4, "text": CANARY},
        },
        "cost_breakdown": {"input_cost": 0.2, "output_cost": 0.3, "additional_costs": {CANARY: 0.01}},
        "spend_logs_metadata": {"openclaw_session_id": "session-1", "openclaw_cron_id": "cron-1", "payload": CANARY},
    }
    safe = safe_metadata(raw)
    assert CANARY not in json.dumps(safe)
    assert safe["usage_object"] == {
        "prompt_tokens": 6,
        "cache_read_input_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    assert safe["cost_breakdown"] == {"input_cost": 0.2, "output_cost": 0.3}
    assert safe["spend_logs_metadata"]["openclaw_cron_id"] == "cron-1"
    assert safe["status"] == "failure"


def test_usage_allowlist_preserves_only_billing_modality_facts() -> None:
    usage = {
        "prompt_tokens_details": {
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 3},
            "audio_length_seconds": 1.25,
            "character_count": 8,
            "image_count": 2,
        },
        "type": "duration",
        "seconds": 1.25,
        "duration_seconds": 0,
        "audio_duration_seconds": 2.5,
        "audio_seconds": 3.75,
        "characters": 16,
    }
    assert safe_metadata({"additional_usage_values": usage}) == {"additional_usage_values": usage}
    assert safe_metadata({"usage_object": usage}) == {"usage_object": usage}
    for invalid in (True, False, -1, float("inf"), float("nan"), "1", CANARY):
        injected = {
            key: invalid
            for key in (
                "audio_length_seconds",
                "seconds",
                "duration_seconds",
                "audio_duration_seconds",
                "audio_seconds",
                "characters",
                "character_count",
                "image_count",
            )
        }
        assert safe_metadata({"additional_usage_values": injected}) == {"additional_usage_values": {}}
    assert safe_metadata(
        {
            "additional_usage_values": {
                "type": CANARY,
                CANARY: 12,
                "image_count": 1.5,
                "characters": 2.5,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 2, CANARY: 99},
            }
        }
    ) == {"additional_usage_values": {"cache_creation_token_details": {"ephemeral_5m_input_tokens": 2}}}


def test_cache_write_alias_supplies_canonical_sql_fact_without_overriding_explicit_zero() -> None:
    for source, expected in (
        ({"cache_write_tokens": 2}, {"cache_write_tokens": 2, "cache_creation_tokens": 2}),
        ({"cache_write_tokens": 2, "cache_creation_tokens": 0}, {"cache_write_tokens": 2, "cache_creation_tokens": 0}),
        ({"cache_write_tokens": True}, {}),
        ({"cache_write_tokens": -1}, {}),
    ):
        assert safe_metadata({"additional_usage_values": {"prompt_tokens_details": source}}) == {
            "additional_usage_values": {"prompt_tokens_details": expected}
        }


def test_diagnostic_filter_removes_content_extras_and_tracebacks(protected_config):
    record = logging.LogRecord(
        "LiteLLM", logging.ERROR, "test.py", 1, "error: %s", (CANARY,), (ValueError, ValueError(CANARY), None)
    )
    record.payload = {"input": CANARY}
    record.stack_info = CANARY
    SecretRedactionFilter().filter(record)
    assert CANARY not in str(record.__dict__)
    assert record.exc_info is None
    assert record.stack_info is None


def test_db_collector_no_log_exemption_does_not_disable_other_logger_redaction(protected_config, monkeypatch, capsys):
    from litellm.litellm_core_utils.litellm_logging import Logging, emit_standard_logging_payload
    from litellm.litellm_core_utils.redact_messages import (
        should_redact_message_logging,
        redact_message_input_output_from_custom_logger,
    )
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setenv("LITELLM_PRINT_STANDARD_LOGGING_PAYLOAD", "1")
    assert should_redact_message_logging(
        {"litellm_params": {"metadata": {"headers": {"litellm-enable-message-redaction": "true"}}}}
    )
    response, _ = make_call()
    redacted = redact_message_input_output_from_custom_logger(
        SimpleNamespace(model_call_details={}), response, CustomLogger(message_logging=False)
    )
    assert CANARY not in redacted.model_dump_json()
    assert Logging.should_run_callback(SimpleNamespace(), _ProxyDBLogger(), {"no-log": True}, "async_success_handler")
    emit_standard_logging_payload({"messages": CANARY})
    assert CANARY not in capsys.readouterr().out


def make_call(failed=False):
    usage = {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "prompt_tokens_details": {
            "cached_tokens": 5,
            "cache_write_tokens": 2,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 1},
            "audio_length_seconds": 1.25,
            "character_count": 8,
            "image_count": 2,
        },
        "type": "duration",
        "seconds": 1.25,
    }
    response = litellm.ModelResponse(
        id="synthetic-spend-row",
        model="openai/test-model",
        choices=[{"message": {"role": "assistant", "content": CANARY}}],
        usage=usage,
    )
    metadata = {
        "user_api_key": "a" * 64,
        "user_api_key_user_id": "user-1",
        "user_api_key_team_id": "team-1",
        "user_api_key_org_id": "org-1",
        "user_api_key_end_user_id": "end-user-1",
        "status": "failure" if failed else "success",
        "tags": [CANARY],
        "user_api_key_alias": CANARY,
        "error_information": {"error_message": CANARY} if failed else None,
        "spend_logs_metadata": {"openclaw_session_id": "session-1", "openclaw_cron_id": "cron-1", "payload": CANARY},
    }
    kwargs = {
        "model": "test-model",
        "custom_llm_provider": "openai",
        "call_type": "acompletion",
        "litellm_call_id": "synthetic-spend-row",
        "response_cost": 0.25,
        "litellm_params": {
            "metadata": metadata,
            "api_base": "https://example.invalid/" + CANARY,
            "proxy_server_request": {
                "body": {
                    "messages": [{"role": "user", "content": CANARY}],
                    "tools": [{"name": CANARY}],
                    "prompt_cache_key": CANARY,
                }
            },
        },
        "standard_logging_object": {
            "metadata": {"usage_object": usage},
            "response": response.model_dump(),
            "request_tags": [CANARY],
            "prompt_tokens": 9,
            "completion_tokens": 3,
            "total_tokens": 12,
            "model_map_information": {},
            "cost_breakdown": {"input_cost": 0.1, "output_cost": 0.15},
        },
        "combined_usage_object": litellm.Usage(**usage),
    }
    return response, kwargs


@pytest.mark.parametrize(
    "usage,expected",
    [
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 30},
            },
            {"prompt_tokens_details": {"cached_tokens": 30}},
        ),
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 10,
            },
            {"cache_read_input_tokens": 30, "cache_creation_input_tokens": 10},
        ),
    ],
)
def test_actual_logging_payload_preserves_provider_cache_usage_for_sql(protected_config, monkeypatch, usage, expected):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    response, kwargs = make_call(False)
    response.usage = litellm.Usage(**usage)
    row = get_logging_payload(
        kwargs, response, datetime(2026, 9, 7, tzinfo=timezone.utc), datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc)
    )
    protected = protect_spend_payload(row)
    metadata = json.loads(protected["metadata"])
    assert all(metadata["additional_usage_values"][name] == value for name, value in expected.items())
    assert (protected["prompt_tokens"], protected["completion_tokens"], protected["total_tokens"]) == (100, 20, 120)
    assert CANARY not in json.dumps(protected, default=str)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_full_long_content_survives_collection_and_jwe_with_credential_stripping(protected_config, monkeypatch, api):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setenv("MAX_STRING_LENGTH_PROMPT_IN_DB", "2048")
    prompt, answer, arguments = "prompt-" * 1024, "answer-" * 1024, "arguments-" * 1024
    credentials = {"Authorization": "SYNTHETIC_CREDENTIAL_MUST_BE_STRIPPED"}
    response, kwargs = make_call(False)
    body = {
        "messages" if api == "chat" else "input": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": {"name": "test", "description": prompt}}],
        "secret_fields": credentials,
    }
    returned = (
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "tool_calls": [{"function": {"arguments": arguments}}],
                    }
                }
            ]
        }
        if api == "chat"
        else {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": answer}]},
                {"type": "function_call", "arguments": arguments},
            ]
        }
    )
    returned["secret_fields"] = credentials
    kwargs["call_type"] = "acompletion" if api == "chat" else "aresponses"
    kwargs["litellm_params"]["proxy_server_request"]["body"] = body
    kwargs["litellm_params"]["metadata"]["error_information"] = {"error_message": answer, "traceback": arguments}
    kwargs["standard_logging_object"]["response"] = returned
    original = copy.deepcopy(kwargs)
    raw = get_logging_payload(
        kwargs, response, datetime(2026, 9, 7, tzinfo=timezone.utc), datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc)
    )
    protected = protect_spend_payload(raw)
    _, content = decrypt(json.loads(protected["proxy_server_request"]))
    assert content["request"] == {name: value for name, value in body.items() if name != "secret_fields"}
    assert content["response"] == {name: value for name, value in returned.items() if name != "secret_fields"}
    assert content["metadata"]["error_information"] == {"error_message": answer, "traceback": arguments}
    assert "SYNTHETIC_CREDENTIAL_MUST_BE_STRIPPED" not in json.dumps(content)
    assert prompt not in json.dumps(protected, default=str)
    assert kwargs == original


def test_oversized_complete_request_is_capture_failure_not_silently_truncated(protected_config, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.request_content_encryption import MAX_CONTENT_BYTES
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    response, kwargs = make_call(False)
    full_input = "x" * (MAX_CONTENT_BYTES + 1)
    kwargs["litellm_params"]["proxy_server_request"]["body"] = {"input": full_input}
    raw = get_logging_payload(
        kwargs, response, datetime(2026, 9, 7, tzinfo=timezone.utc), datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc)
    )
    assert json.loads(raw["proxy_server_request"])["input"] == full_input
    protected = protect_spend_payload(raw)
    marker = json.loads(protected["metadata"])["openorange_request_log"]
    assert marker["content_status"] == "capture_failed"
    assert marker["failure_code"] == "content_too_large"
    assert protected["proxy_server_request"] == "{}"
    assert protected["spend"] == 0.25


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_failed", [False, True])
@pytest.mark.parametrize("failure_mode", ["none", "crypto", "transform"])
@pytest.mark.parametrize(
    "ownership_source", ["platform", "byok", "unknown", "admission_timeout", "bad_ingress", "retrieval_unavailable"]
)
async def test_real_writer_encrypts_before_sqlite_and_daily_copies_and_keeps_billing(
    protected_config, tmp_path, provider_failed, failure_mode, ownership_source
):
    from litellm.proxy import proxy_server
    from litellm.proxy.utils import PrismaClient

    response, kwargs = make_call(provider_failed)
    from litellm.litellm_core_utils.credential_ownership import (
        CONTEXT,
        FIELD,
        STAMP,
        resolve_ownership,
        select_credential,
    )

    local_source = (
        "unknown"
        if ownership_source in {"admission_timeout", "bad_ingress", "retrieval_unavailable"}
        else ownership_source
    )
    registration = {
        "v": 1,
        "source": local_source,
        "registration_id": "3e15c0c2-feca-4104-a648-8d579315ef51",
        "registration_revision": "e7ca1c3e-b6ea-4ce0-8538-b8f149448038",
    }
    route = {
        "model_info": {"id": "completed-deployment", FIELD: registration},
        "litellm_params": {
            "model": "openai/test-model",
            "api_key": "synthetic-key",
            "api_base": "https://example.invalid/v1",
        },
    }
    kwargs[STAMP] = resolve_ownership(
        {
            **route["litellm_params"],
            "metadata": {CONTEXT: select_credential(route, {}, "completed-deployment")},
        },
        {},
        {},
    )
    kwargs["litellm_params"]["metadata"]["model_info"] = {"id": "completed-deployment"}
    kwargs["litellm_params"]["metadata"][FIELD] = {**registration, "source": "forged"}
    from tests.test_litellm.litellm_core_utils.test_terminal_receipt_evidence import signed_session
    from litellm.litellm_core_utils.terminal_receipt_evidence import FIELD as TERMINAL_FIELD, STAMP as TERMINAL_STAMP

    from tests.test_litellm.litellm_core_utils.test_terminal_usage_evidence import signed_usage
    from litellm.litellm_core_utils.terminal_usage_evidence import FIELD as USAGE_EVIDENCE_FIELD, usage_shell

    terminal_session, measurement, _, _, _ = signed_usage(local_source)
    terminal_session.usage_envelope = measurement
    if ownership_source in {"admission_timeout", "bad_ingress"}:
        from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable
        from litellm.litellm_core_utils.terminal_receipt_evidence import CONTEXT as TERMINAL_CONTEXT, Terminal
        from litellm.litellm_core_utils.terminal_receipt_hooks import prepare

        terminal_session.envelope = None
        terminal_session.begun = False
        terminal_session.finished = False
        terminal_session.terminal = Terminal(
            deployment_id="completed-deployment", model="proxy-model", provider="litellm_proxy"
        )
        with patch.object(
            terminal_session.root.authority,
            "request",
            **(
                {"side_effect": ReceiptUnavailable()}
                if ownership_source == "admission_timeout"
                else {"return_value": {"ingress_jws": CANARY}}
            ),
        ):
            with pytest.raises(ReceiptUnavailable):
                prepare({"metadata": {TERMINAL_CONTEXT: terminal_session}}, kwargs)
        assert terminal_session.envelope is not None
        assert terminal_session.envelope.correlation_id is None
        assert terminal_session.envelope.state == "unavailable"
        terminal_session.usage_envelope = usage_shell(terminal_session.envelope)
    if ownership_source == "retrieval_unavailable":
        from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable

        terminal_session.envelope = terminal_session.envelope.model_copy(
            update={
                "receipts": terminal_session.envelope.receipts[:2],
                "state": "pending",
            }
        )
        expected_receipts = terminal_session.envelope.receipts
        with patch.object(terminal_session.root.authority, "request", side_effect=ReceiptUnavailable()):
            terminal_session.envelope = terminal_session.root.authority.retrieve(terminal_session.envelope)
        assert terminal_session.envelope.receipts == expected_receipts
        assert terminal_session.envelope.state == "unavailable"
    from litellm.litellm_core_utils.terminal_usage_observation import UsageSnapshot, FIELD as USAGE_FIELD

    terminal_session.usage_snapshot = UsageSnapshot(
        prompt_tokens=9,
        completion_tokens=3,
        total_tokens=12,
        cache_read_tokens=5,
        cache_write_tokens=2,
        cache_write_5m_tokens=1,
        cache_write_1h_tokens=1,
    )
    terminal_session.usage_final = not provider_failed
    kwargs[TERMINAL_STAMP] = terminal_session
    kwargs["litellm_params"]["metadata"][TERMINAL_FIELD] = {"state": "forged"}
    kwargs["litellm_params"]["metadata"][USAGE_FIELD] = {"state": "forged", "prompt_tokens": CANARY}
    kwargs["litellm_params"]["metadata"][USAGE_EVIDENCE_FIELD] = {"state": "forged", "measurements": [CANARY]}
    original = copy.deepcopy(kwargs)
    spool = SQLiteSpendLogSpool(str(tmp_path / "spend.sqlite"))
    client = SimpleNamespace(
        _spend_log_spool=spool,
        _spend_log_transactions_lock=asyncio.Lock(),
        spend_log_transactions=[],
        get_request_status=lambda row: PrismaClient.get_request_status(None, row),
    )
    writer = DBSpendUpdateWriter()
    captured = asyncio.Event()
    batch = {}

    async def capture_batch(**values):
        batch.update(values)
        await DBSpendUpdateWriter._batch_database_updates(writer, **values)
        captured.set()

    writer._batch_database_updates = capture_batch
    writer._enqueue_tool_registry_upsert = AsyncMock()
    writer._enqueue_tool_usage_transaction = AsyncMock()
    encryptor = configured_encryptor()
    assert isinstance(encryptor, RequestContentEncryptor)
    with (
        patch.object(proxy_server, "prisma_client", client),
        patch.object(proxy_server.proxy_logging_obj, "db_spend_update_writer", writer),
        patch.object(proxy_server, "disable_spend_logs", False),
        patch.object(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True}),
        patch.object(proxy_server, "user_api_key_cache", SimpleNamespace(async_get_cache=AsyncMock(return_value=None))),
        patch(
            "litellm.proxy.spend_tracking.request_content_encryption.jwe.encrypt_compact",
            side_effect=RuntimeError(CANARY),
        )
        if failure_mode == "crypto"
        else patch(
            "litellm.proxy.spend_tracking.request_content_encryption.jwe.encrypt_compact", wraps=jwe.encrypt_compact
        ),
        patch("litellm.proxy.spend_tracking.request_content_metadata.safe_metadata", side_effect=RuntimeError(CANARY))
        if failure_mode == "transform"
        else patch("litellm.proxy.spend_tracking.request_content_metadata.safe_metadata", wraps=safe_metadata),
    ):
        if provider_failed:
            from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

            collector = _ProxyDBLogger()
            await collector.async_log_failure_event(
                kwargs,
                response,
                datetime(2026, 9, 7, tzinfo=timezone.utc),
                datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc),
            )
            assert terminal_session.failure_persisted
            await collector.async_log_failure_event(kwargs, response, datetime.now(), datetime.now())
        else:
            await writer.update_database(
                token="a" * 64,
                user_id="user-1",
                end_user_id="end-user-1",
                team_id="team-1",
                org_id="org-1",
                kwargs=kwargs,
                completion_response=ValueError(CANARY) if provider_failed else response,
                start_time=datetime(2026, 9, 7, tzinfo=timezone.utc),
                end_time=datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc),
                response_cost=0.25,
            )
        await asyncio.wait_for(captured.wait(), 2)
        rows = (await spool.peek_batch(10, 100000)).logs
        assert len(rows) == 1
        row = rows[0]
        assert CANARY not in json.dumps(row)
        assert CANARY not in json.dumps(batch["payload"], default=str)
        assert batch["response_cost"] == 0.25
        assert batch["hashed_token"] == "a" * 64
        assert batch["user_id"] == "user-1"
        assert batch["team_id"] == "team-1"
        assert batch["org_id"] == "org-1"
        assert batch["payload"]["request_tags"] == "[]"
        daily = await writer._common_add_spend_log_transaction_to_daily_transaction(row, client)
        assert daily["spend"] == 0.25
        assert daily["prompt_tokens"] == 9
        assert daily["completion_tokens"] == 3
        assert daily["cache_read_input_tokens"] == 5
        assert daily["cache_creation_input_tokens"] == 2
        assert daily["failed_requests"] == int(provider_failed)
        assert daily["successful_requests"] == int(not provider_failed)
        marker = json.loads(row["metadata"])["openorange_request_log"]
        for persisted in (row, batch["payload"]):
            persisted_metadata = json.loads(persisted["metadata"])
            assert persisted["request_id"] == terminal_session.attempt_id
            assert persisted_metadata[TERMINAL_FIELD] == terminal_session.envelope.model_dump(mode="json")
            assert persisted_metadata[USAGE_EVIDENCE_FIELD] == terminal_session.usage_envelope.model_dump(mode="json")
            observation = persisted_metadata[USAGE_FIELD]
            assert observation["state"] == ("partial" if provider_failed else "observed")
            assert observation["local_attempt_id"] == terminal_session.attempt_id
            assert observation["prompt_tokens"] == 9
            assert observation["cache_read_tokens"] == 5
            assert observation["cache_write_tokens"] == 2
            assert observation["cache_write_5m_tokens"] == 1
            assert observation["cache_write_1h_tokens"] == 1
            assert persisted_metadata[FIELD]["source"] == local_source
            assert persisted_metadata[FIELD]["deployment_id"] == "completed-deployment"
            assert persisted_metadata[FIELD]["registration_id"] == (
                registration["registration_id"] if local_source != "unknown" else None
            )
            facts = persisted_metadata["additional_usage_values"]
            assert facts["prompt_tokens_details"]["cache_creation_token_details"] == {
                "ephemeral_5m_input_tokens": 1,
                "ephemeral_1h_input_tokens": 1,
            }
            assert facts["prompt_tokens_details"]["audio_length_seconds"] == 1.25
            assert facts["prompt_tokens_details"]["character_count"] == 8
            assert facts["prompt_tokens_details"]["image_count"] == 2
            assert facts["prompt_tokens_details"]["cache_creation_tokens"] == 2
            assert (facts["type"], facts["seconds"]) == ("duration", 1.25)
            assert persisted["model"] == "test-model"
            assert persisted["custom_llm_provider"] == "openai"
        assert marker["content_status"] == ("encrypted" if failure_mode == "none" else "capture_failed")
        if failure_mode != "transform":
            assert marker["facts"]["cron_id"] == "cron-1"
            assert marker["facts"]["prompt_cache_key_present"] is True
        if failure_mode != "none":
            assert row["proxy_server_request"] == "{}"
            assert isinstance(encryption_readiness(), CaptureFailure)
        else:
            header, content = decrypt(json.loads(row["proxy_server_request"]))
            assert header["recordId"] == row["request_id"]
            assert content["request"]["messages"][0]["content"] == CANARY
            assert content["response"]["choices"][0]["message"]["content"] == CANARY
            assert content["request_tags"] == [CANARY]
        writer._enqueue_tool_registry_upsert.assert_not_called()
        writer._enqueue_tool_usage_transaction.assert_not_called()
        increments = writer.spend_update_queue.get_aggregated_db_spend_update_transactions(
            await writer.spend_update_queue.flush_all_updates_from_in_memory_queue()
        )
        assert increments["key_list_transactions"] == {"a" * 64: 0.25}
        assert increments["user_list_transactions"] == {"user-1": 0.25}
        assert increments["team_list_transactions"] == {"team-1": 0.25}
        assert increments["org_list_transactions"] == {"org-1": 0.25}
        assert increments["end_user_list_transactions"] == {"end-user-1": 0.25}
        assert increments["tag_list_transactions"] == {}
        user_daily = await writer.daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
        assert len(user_daily) == 1
        assert next(iter(user_daily.values()))["spend"] == 0.25
        assert kwargs == original
        for file in tmp_path.glob("spend.sqlite*"):
            assert CANARY.encode() not in file.read_bytes()
        reopened = SQLiteSpendLogSpool(str(tmp_path / "spend.sqlite"))
        assert (await reopened.peek_batch(10, 100000)).logs == rows
    if failure_mode == "transform":
        assert encryption_readiness() == CaptureFailure("transformation_failed")
    else:
        assert encryption_readiness() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_mock_provider_output_and_full_retention_survive_no_log(protected_config, monkeypatch, tmp_path, stream):
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

    spool = SQLiteSpendLogSpool(str(tmp_path / "stream.sqlite"))
    client = SimpleNamespace(
        _spend_log_spool=spool, _spend_log_transactions_lock=asyncio.Lock(), spend_log_transactions=[]
    )
    writer = DBSpendUpdateWriter()
    writer._batch_database_updates = AsyncMock()
    finished = asyncio.Event()
    collector = _ProxyDBLogger()

    async def capture(kwargs, response_obj, start_time, end_time):
        await writer.update_database(
            token="a" * 64,
            user_id="user-1",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs=kwargs,
            completion_response=response_obj,
            start_time=start_time,
            end_time=end_time,
            response_cost=0.25,
        )
        finished.set()

    monkeypatch.setattr(collector, "async_log_success_event", capture)
    for name in (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "_async_success_callback", [collector])
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    messages = [{"role": "user", "content": CANARY}]
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=messages,
        stream=stream,
        mock_response=CANARY,
        metadata={"user_api_key_user_id": "user-1"},
        proxy_server_request={"body": {"messages": messages}},
        **{"no-log": True},
    )
    if stream:
        output = "".join([chunk.choices[0].delta.content or "" async for chunk in response])
    else:
        output = response.choices[0].message.content
    assert output == CANARY
    await asyncio.wait_for(finished.wait(), 5)
    rows = (await spool.peek_batch(10, 100000)).logs
    assert len(rows) == 1
    assert CANARY not in json.dumps(rows)
    header, content = decrypt(json.loads(rows[0]["proxy_server_request"]))
    assert header["recordId"] == rows[0]["request_id"]
    assert content["messages"] == messages
    assert content["response"]["choices"][0]["message"]["content"] == CANARY


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_failure", ["missing_database", "missing_spool", "enqueue_error"])
async def test_receipt_failure_recovery_uses_private_context_and_durable_ack(
    protected_config, monkeypatch, tmp_path, initial_failure
):
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.litellm_core_utils.terminal_receipt_evidence import STAMP, FIELD
    from tests.test_litellm.litellm_core_utils.test_terminal_receipt_evidence import signed_session

    session, _, _, _ = signed_session("unknown")
    response, kwargs = make_call(True)
    kwargs[STAMP] = session
    kwargs["litellm_params"]["metadata"]["model_info"] = {"id": session.terminal.deployment_id}
    kwargs["start_time"] = datetime(2026, 9, 7, tzinfo=timezone.utc)
    kwargs["end_time"] = datetime(2026, 9, 7, 0, 0, 1, tzinfo=timezone.utc)
    spool = SQLiteSpendLogSpool(str(tmp_path / "recovery.sqlite"))
    client = SimpleNamespace(
        _spend_log_spool=spool,
        _spend_log_transactions_lock=asyncio.Lock(),
        spend_log_transactions=[],
        get_request_status=lambda row: row["status"],
    )
    writer = DBSpendUpdateWriter()
    batches = AsyncMock()
    writer._batch_database_updates = batches
    writer._enqueue_tool_registry_upsert = AsyncMock()
    writer._enqueue_tool_usage_transaction = AsyncMock()
    proxy_logging = proxy_server.proxy_logging_obj
    monkeypatch.setattr(proxy_logging, "db_spend_update_writer", writer)
    monkeypatch.setattr(proxy_logging, "update_request_status", AsyncMock())
    monkeypatch.setattr(proxy_logging, "alert_types", [])
    monkeypatch.setattr(proxy_server, "prisma_client", None if initial_failure == "missing_database" else client)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    collector = _ProxyDBLogger()
    monkeypatch.setattr(litellm, "callbacks", [collector])
    if initial_failure == "missing_spool":
        client._spend_log_spool = None
    with (
        patch.object(spool, "enqueue", side_effect=OSError("synthetic storage failure"))
        if initial_failure == "enqueue_error"
        else patch.object(spool, "enqueue", wraps=spool.enqueue)
    ):
        await collector.async_log_failure_event(kwargs, response, kwargs["start_time"], kwargs["end_time"])
    await asyncio.sleep(0)
    assert not session.failure_persisted
    assert (await spool.stats()).count == 0
    initial_batches = batches.await_count
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    client._spend_log_spool = spool
    data = {"litellm_call_id": "root-request", "litellm_logging_obj": SimpleNamespace(model_call_details=kwargs)}
    with (
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback._release_budget_reservation", new_callable=AsyncMock
        ) as release,
        patch("litellm.proxy.spend_tracking.request_content_policy.request_policy_failure", return_value=None),
        patch("litellm.proxy.spend_tracking.request_content_policy.protected_profile_failure", return_value=None),
    ):
        await proxy_logging.post_call_failure_hook(
            request_data=data,
            original_exception=ValueError("native failure is not non-execution"),
            user_api_key_dict=UserAPIKeyAuth(request_route="/chat/completions"),
        )
        release.assert_not_awaited()
    await asyncio.sleep(0)
    assert "litellm_logging_obj" not in data
    assert STAMP not in data
    assert session.failure_persisted
    rows = (await spool.peek_batch(10, 100000)).logs
    assert len(rows) == 1
    assert rows[0]["request_id"] == session.attempt_id
    assert json.loads(rows[0]["metadata"])[FIELD]["local_request_id"] == session.root.local_request_id
    assert rows[0]["status"] == "failure"
    assert rows[0]["prompt_tokens"] == 9
    assert batches.await_count == max(initial_batches, 1)
    await asyncio.gather(
        *(
            collector.async_log_failure_event(kwargs, response, kwargs["start_time"], kwargs["end_time"])
            for _ in range(2)
        )
    )
    assert (await spool.stats()).count == 1
