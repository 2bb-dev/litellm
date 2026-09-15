import copy
import hashlib
from uuid import uuid4

import pytest

from litellm.litellm_core_utils.credential_ownership import strip_ownership
from litellm.litellm_core_utils.terminal_usage_evidence import (
    FIELD,
    parse_usage,
    safe_usage_evidence,
    usage_shell,
    verify_usage_set,
)
from tests.test_litellm.litellm_core_utils.test_terminal_receipt_evidence import signed_session


def signed_usage(source="byok"):
    session, sign, (_, finish, _), header = signed_session(source)
    claims = {
        name: finish[name]
        for name in (
            "v",
            "iss",
            "aud",
            "local_request_id",
            "local_attempt_id",
            "local_deployment_id",
            "correlation_id",
            "central_request_id",
            "producer_id",
            "terminal_attempt_id",
            "occurred_at",
        )
    }
    claims.update(
        {
            "measurement_id": str(uuid4()),
            "finish_receipt_id": finish["receipt_id"],
            "finish_sequence": finish["sequence"],
            "finish_jws_sha256": hashlib.sha256(session.envelope.receipts[1].encode("ascii")).hexdigest(),
            "usage": {
                "v": 1,
                "state": "observed",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": None,
                "cache_write_5m_tokens": None,
                "cache_write_1h_tokens": None,
            },
        }
    )
    protected = {**header, "typ": "oo-terminal-usage+jwt"}
    envelope = usage_shell(session.envelope).model_copy(update={"measurements": (sign(claims, protected),)})
    return session, envelope, claims, protected, sign


def verify(session, envelope):
    return verify_usage_set(
        envelope, session.envelope, session.root.authority.settings.issuer, session.root.authority.keys
    )


def test_original_measurement_is_separate_and_requires_every_exact_finish():
    session, envelope, _, _, _ = signed_usage()
    assert verify(session, envelope) == "complete"
    assert safe_usage_evidence(envelope.model_dump(mode="json")) == envelope.model_dump(mode="json")
    assert parse_usage(envelope.measurements[0]).claims.usage.prompt_tokens == 0
    assert parse_usage(envelope.measurements[0]).claims.usage.cache_write_tokens is None
    assert verify(session, envelope.model_copy(update={"measurements": ()})) == "pending"
    assert verify(session, envelope.model_copy(update={"measurements": envelope.measurements * 2})) == "invalid"
    session.envelope = session.envelope.model_copy(update={"receipts": session.envelope.receipts[:-1]})
    assert verify(session, envelope) == "pending"
    assert strip_ownership({"nested": [{FIELD: envelope.model_dump(mode="json")}]}) == {"nested": [{}]}


@pytest.mark.parametrize(
    "field",
    [
        "iss",
        "aud",
        "local_request_id",
        "local_attempt_id",
        "local_deployment_id",
        "correlation_id",
        "central_request_id",
        "producer_id",
        "terminal_attempt_id",
        "finish_receipt_id",
        "finish_sequence",
        "finish_jws_sha256",
    ],
)
def test_valid_signature_cannot_cross_bind_or_replace_original_finish(field):
    session, envelope, claims, header, sign = signed_usage()
    changed = {
        **claims,
        field: 4 if field == "finish_sequence" else "0" * 64 if field == "finish_jws_sha256" else str(uuid4()),
    }
    assert verify(session, envelope.model_copy(update={"measurements": (sign(changed, header),)})) == "invalid"


@pytest.mark.parametrize(
    "fault",
    ["extra", "missing", "boolean", "negative", "unsafe", "unobserved", "type", "header", "signature", "oversize"],
)
def test_unsafe_originals_never_escape_readable_projection(fault):
    _, envelope, original, protected, sign = signed_usage()
    claims, header = copy.deepcopy(original), dict(protected)
    if fault == "extra":
        claims["private_content"] = "SYNTHETIC_DO_NOT_RETAIN"
    elif fault == "missing":
        del claims["usage"]["cache_write_1h_tokens"]
    elif fault in {"boolean", "negative", "unsafe"}:
        claims["usage"]["prompt_tokens"] = {"boolean": True, "negative": -1, "unsafe": 9007199254740992}[fault]
    elif fault == "unobserved":
        claims["usage"]["state"] = "unobserved"
    elif fault == "type":
        header["typ"] = "oo-terminal-receipt+jwt"
    elif fault == "header":
        header["jku"] = "SYNTHETIC_DO_NOT_RETAIN"
    token = sign(claims, header)
    if fault == "signature":
        token = token.rsplit(".", 1)[0] + ".AA"
    elif fault == "oversize":
        token += "a" * 3072
    assert parse_usage(token) is None
    retained = safe_usage_evidence(envelope.model_copy(update={"measurements": (token,)}).model_dump(mode="json"))
    assert retained["state"] == "invalid"
    assert retained["measurements"] == []
    assert retained["local_attempt_id"] == envelope.local_attempt_id


def test_whole_envelope_overflow_and_admission_failure_keep_safe_local_context():
    _, envelope, _, _, _ = signed_usage()
    retained = safe_usage_evidence(envelope.model_copy(update={"measurements": ("a" * 65536,)}).model_dump(mode="json"))
    assert retained["state"] == "overflow"
    assert retained["measurements"] == []
    assert retained["local_request_id"] == envelope.local_request_id
    admission = envelope.model_copy(update={"correlation_id": None, "measurements": (), "state": "unavailable"})
    assert safe_usage_evidence(admission.model_dump(mode="json")) == admission.model_dump(mode="json")
    assert (
        safe_usage_evidence(admission.model_copy(update={"state": "complete"}).model_dump(mode="json"))["state"]
        == "invalid"
    )
