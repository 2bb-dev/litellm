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
    config = {"version": 1, "instanceUid": FIXTURE["instanceUid"], "publicKey": FIXTURE["publicKey"]}
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
async def test_real_writer_encrypts_before_sqlite_and_daily_copies_and_keeps_billing(
    protected_config, tmp_path, provider_failed, failure_mode
):
    from litellm.proxy import proxy_server
    from litellm.proxy.utils import PrismaClient

    response, kwargs = make_call(provider_failed)
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
    encryptor = configured_encryptor()
    assert isinstance(encryptor, RequestContentEncryptor)
    with (
        patch.object(proxy_server, "prisma_client", client),
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
        assert CANARY not in json.dumps(batch["payload_copy"], default=str)
        assert batch["response_cost"] == 0.25
        assert batch["hashed_token"] == "a" * 64
        assert batch["user_id"] == "user-1"
        assert batch["team_id"] == "team-1"
        assert batch["org_id"] == "org-1"
        assert batch["request_tags"] == "[]"
        daily = await writer._common_add_spend_log_transaction_to_daily_transaction(row, client)
        assert daily["spend"] == 0.25
        assert daily["prompt_tokens"] == 9
        assert daily["completion_tokens"] == 3
        assert daily["cache_read_input_tokens"] == 5
        assert daily["cache_creation_input_tokens"] == 2
        assert daily["failed_requests"] == int(provider_failed)
        assert daily["successful_requests"] == int(not provider_failed)
        marker = json.loads(row["metadata"])["openorange_request_log"]
        for persisted in (row, batch["payload_copy"]):
            persisted_metadata = json.loads(persisted["metadata"])
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
