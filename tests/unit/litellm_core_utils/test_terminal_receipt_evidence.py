import base64
import copy
import json
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from litellm.litellm_core_utils.terminal_receipt_client import Authority, Settings
from litellm.litellm_core_utils.terminal_receipt_evidence import Envelope, parse_jws, safe_evidence, verify_set
from litellm.litellm_core_utils.terminal_receipt_hooks import Root, Session


def signed_session(source="byok", local_deployment="completed-deployment"):
    from litellm.litellm_core_utils.terminal_receipt_evidence import Terminal

    key = Ed25519PrivateKey.generate()
    issuer, audience, kid, request, attempt, correlation, central, producer, terminal_attempt = [
        str(uuid4()) for _ in range(9)
    ]
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    authority = Authority(
        Settings.model_validate_json(
            json.dumps(
                {
                    "v": 1,
                    "role": "audience",
                    "identity": audience,
                    "issuer": issuer,
                    "base_url": "http://127.0.0.1:1",
                    "bearer": "synthetic-local-test-token",
                    "public_keys": {kid: public},
                }
            )
        )
    )
    terminal = {"deployment_id": "terminal-deployment", "model": "actual-model", "provider": "actual-provider"}
    unknown = {
        "v": 1,
        "source": "unknown",
        "provenance": "ambiguous",
        "registration_id": None,
        "registration_revision": None,
        "deployment_id": "terminal-deployment",
    }
    ownership = (
        {
            "v": 1,
            "source": source,
            "provenance": "credential_registration",
            "registration_id": str(uuid4()),
            "registration_revision": str(uuid4()),
            "deployment_id": "terminal-deployment",
        }
        if source != "unknown"
        else unknown
    )
    common = {
        "v": 1,
        "iss": issuer,
        "aud": audience,
        "local_request_id": request,
        "local_attempt_id": attempt,
        "local_deployment_id": local_deployment,
        "correlation_id": correlation,
        "central_request_id": central,
        "producer_id": producer,
        "occurred_at": 1,
    }
    began = {
        **common,
        "receipt_id": str(uuid4()),
        "sequence": 1,
        "event": "started",
        "outcome": "pending",
        "binding": "unobserved",
        "terminal_attempt_id": terminal_attempt,
        "terminal": terminal,
        "completed": None,
        "planned_ownership": ownership,
        "ownership": unknown,
    }
    finished = {
        **began,
        "receipt_id": str(uuid4()),
        "sequence": 2,
        "event": "finished",
        "outcome": "success",
        "binding": "observed" if source != "unknown" else "unknown",
        "completed": terminal,
        "ownership": ownership,
    }
    seal = {**common, "receipt_id": str(uuid4()), "sequence": 3, "event": "seal", "attempt_count": 1}
    header = {"alg": "EdDSA", "typ": "oo-terminal-receipt+jwt", "kid": kid}

    def sign(claims, protected=header):
        def encode(value):
            return (
                base64.urlsafe_b64encode(
                    json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                )
                .rstrip(b"=")
                .decode()
            )

        value = encode(protected) + "." + encode(claims)
        return value + "." + base64.urlsafe_b64encode(key.sign(value.encode())).rstrip(b"=").decode()

    envelope = Envelope.model_validate_json(
        json.dumps(
            {
                "v": 1,
                "audience": audience,
                "local_request_id": request,
                "local_attempt_id": attempt,
                "local_deployment_id": local_deployment,
                "correlation_id": correlation,
                "receipts": [sign(began), sign(finished), sign(seal)],
                "state": "complete",
            }
        )
    )
    session = Session(
        Root(authority, request, central, None),
        attempt,
        Terminal(**terminal),
        None,
        envelope=envelope,
        begun=True,
        finished=True,
    )
    return session, sign, (began, finished, seal), header


@pytest.mark.parametrize("source", ["platform", "byok", "unknown"])
def test_closed_evidence_preserves_exact_signed_bytes_and_separate_identity(source):
    session, _, _, _ = signed_session(source)
    evidence = session.envelope
    assert verify_set(evidence, session.root.authority.settings.issuer, session.root.authority.keys) == "complete"
    assert safe_evidence(evidence.model_dump(mode="json")) == evidence.model_dump(mode="json")
    assert parse_jws(evidence.receipts[1]).claims.ownership.source == source
    assert parse_jws(evidence.receipts[1]).claims.terminal.deployment_id != evidence.local_deployment_id


@pytest.mark.parametrize(
    "mutation", ["prompt", "url", "error", "extra_header", "alg", "pending_positive", "wrong_finished_registration"]
)
def test_valid_signature_cannot_make_arbitrary_or_incoherent_claims_readable(mutation):
    session, sign, claims, protected = signed_session()
    body = copy.deepcopy(claims[0])
    header = dict(protected)
    if mutation == "prompt":
        body["prompt"] = "SYNTHETIC_DO_NOT_RETAIN"
    elif mutation == "url":
        body["terminal"]["model"] = "https://secret.invalid/path"
    elif mutation == "error":
        body["error"] = "SYNTHETIC_DO_NOT_RETAIN"
    elif mutation == "extra_header":
        header["jku"] = "https://secret.invalid/path"
    elif mutation == "alg":
        header["alg"] = "HS256"
    elif mutation == "pending_positive":
        body["ownership"] = body["planned_ownership"]
    else:
        body = copy.deepcopy(claims[1])
        body["ownership"] = {**body["ownership"], "registration_revision": str(uuid4())}
    invalid = sign(body, header)
    assert parse_jws(invalid) is None
    envelope = session.envelope.model_dump(mode="json")
    envelope["receipts"] = [invalid]
    retained = safe_evidence(envelope)
    assert retained["receipts"] == []
    assert retained["state"] == "invalid"
    assert retained["local_attempt_id"] == session.attempt_id
    assert "SYNTHETIC_DO_NOT_RETAIN" not in json.dumps(retained)
    assert "secret.invalid" not in json.dumps(retained)


def test_overflow_preserves_only_bounded_local_binding_and_is_idempotent():
    session, _, _, _ = signed_session()
    envelope = session.envelope.model_dump(mode="json")
    envelope["receipts"] = ["x" * 3072] * 33
    retained = safe_evidence(envelope)
    assert retained["receipts"] == []
    assert retained["state"] == "overflow"
    assert retained["local_attempt_id"] == session.attempt_id
    assert safe_evidence(retained) == retained
    assert safe_evidence({"v": 1, "state": "overflow"}) == {"v": 1, "state": "overflow"}


@pytest.mark.parametrize("state", ["complete", "pending", "overflow"])
def test_null_correlation_never_claims_admitted_or_complete_state(state):
    session, _, _, _ = signed_session()
    original = session.envelope.model_dump(mode="json")
    invalid = {**original, "correlation_id": None, "receipts": [], "state": state}
    assert safe_evidence(invalid) == {**invalid, "state": "invalid"}
    missing = dict(invalid)
    del missing["correlation_id"]
    assert safe_evidence(missing) == {"v": 1, "state": "invalid"}


def test_null_correlation_with_oversize_receipt_retains_only_safe_unavailable_context():
    session, _, _, _ = signed_session()
    original = {
        **session.envelope.model_dump(mode="json"),
        "correlation_id": None,
        "receipts": ["SYNTHETIC_UNSAFE" * 6000],
        "state": "unavailable",
    }
    projected = safe_evidence(original)
    assert projected == {**original, "receipts": [], "state": "invalid"}
    assert Envelope.model_validate_json(json.dumps(projected)).correlation_id is None


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "permissions", "relative", "oversize"])
def test_explicit_authority_config_rejects_unsafe_file_bindings(tmp_path, monkeypatch, kind):
    from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable, configured_authority

    session, _, _, _ = signed_session()
    settings = session.root.authority.settings
    data = {**settings.model_dump(mode="json"), "bearer": settings.bearer.get_secret_value()}
    source = tmp_path / "authority.json"
    source.write_text(json.dumps(data))
    source.chmod(0o600)
    path = source
    if kind == "symlink":
        path = tmp_path / "link.json"
        path.symlink_to(source)
    elif kind == "hardlink":
        path = tmp_path / "link.json"
        path.hardlink_to(source)
    elif kind == "permissions":
        source.chmod(0o644)
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        path = source.name
    elif kind == "oversize":
        source.write_text(" " * 65537)
    monkeypatch.setenv("OPENORANGE_TERMINAL_RECEIPT_CLIENT_CONFIG_FILE", str(path))
    with pytest.raises(ReceiptUnavailable, match="^terminal_receipt_unavailable$"):
        configured_authority()


def test_transient_refresh_retains_previously_authenticated_failure_prefix():
    from litellm.litellm_core_utils.terminal_receipt_client import ReceiptUnavailable

    session, _, _, _ = signed_session()
    expected = session.envelope.receipts[:2]

    class FailingRefresh(Authority):
        calls = 0

        def request(self, method, path, body=None):
            self.calls += 1
            if self.calls == 1:
                return {"events": list(expected), "next_after": 2, "has_more": False, "complete": False}
            raise ReceiptUnavailable()

    authority = FailingRefresh(session.root.authority.settings)
    pending = authority.retrieve(session.envelope.model_copy(update={"receipts": (), "state": "pending"}))
    assert pending.receipts == expected
    assert pending.state == "pending"
    failed = authority.retrieve(pending)
    assert failed.state == "unavailable"
    assert failed.receipts == expected
    assert safe_evidence(failed.model_dump(mode="json"))["receipts"] == list(expected)
