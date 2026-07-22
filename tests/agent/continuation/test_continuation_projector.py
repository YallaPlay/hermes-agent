from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.continuation_preview import (
    PROJECTOR_OUTPUT_MAX_BYTES,
    ProjectorRequestKind,
    ProjectorRequestV1,
    ProjectorResponseError,
    ProjectorTransportError,
)
from agent.continuation_projector import (
    BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
    BEDROCK_PROJECTOR_PROTOCOL_V1,
    WORKER_STDERR_MAX_BYTES,
    WORKER_STDOUT_MAX_BYTES,
    BedrockCredentialSnapshotV1,
    BedrockProjectorFailureCode,
    BedrockProjectorTransportError,
    ContinuationPreviewConfigurationError,
    StrictBedrockProjectorV1,
    resolve_bedrock_credential_snapshot,
    resolve_continuation_preview_settings,
)
from agent.continuation_projector_worker import (
    WorkerInputError,
    _classify_bedrock_exception,
    _deny_local_write_audit_event,
    _project_bedrock,
    _validate_request,
)


def _config(*, timeout: float = 2.0) -> dict:
    return {
        "continuation_checkpoint": {"preview_enabled": True},
        "bedrock": {"region": "us-west-2"},
        "auxiliary": {
            "continuation_checkpoint": {
                "provider": "bedrock",
                "model": "global.anthropic.claude-sonnet-5",
                "base_url": "",
                "api_key": "",
                "timeout": timeout,
                "extra_body": {},
                "reasoning_effort": "",
            }
        },
    }


def _settings(*, timeout: float = 2.0):
    return resolve_continuation_preview_settings(_config(timeout=timeout), environ={})


def _credentials() -> BedrockCredentialSnapshotV1:
    return BedrockCredentialSnapshotV1(
        access_key_id="AKIA_TEST_ONLY",
        secret_access_key="secret-test-only",
        session_token="session-test-only",
    )


def _request(attempt: int = 1) -> ProjectorRequestV1:
    return ProjectorRequestV1(
        kind=ProjectorRequestKind.PRIMARY if attempt == 1 else ProjectorRequestKind.REPAIR,
        attempt=attempt,
        prompt=json.dumps({"attempt": attempt, "instruction": "return JSON"}),
    )


def _write_worker(tmp_path: Path, source: str) -> tuple[str, ...]:
    script = tmp_path / "worker.py"
    script.write_text(source, encoding="utf-8")
    return (sys.executable, "-I", "-B", str(script))


def _echo_worker(tmp_path: Path) -> tuple[str, ...]:
    return _write_worker(
        tmp_path,
        f"""
import json, os, sys
request = json.load(sys.stdin)
raw = json.dumps({{
    "attempt": request["attempt"],
    "kind": request["kind"],
    "prompt": request["prompt"],
    "model": request["model"],
    "region": request["region"],
    "max_output_tokens": request["max_output_tokens"],
    "hermes_home": os.environ.get("HERMES_HOME"),
    "inherited_access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
    "worker_home": os.environ.get("HOME"),
}})
json.dump({{
    "protocol": {BEDROCK_PROJECTOR_PROTOCOL_V1!r},
    "ok": True,
    "raw_json": raw,
    "latency_ms": 7,
    "input_tokens": 11,
    "output_tokens": 13,
}}, sys.stdout)
""",
    )


def test_settings_resolve_explicit_bedrock_route_without_mutating_input():
    config = _config(timeout=42)
    before = json.dumps(config, sort_keys=True)

    settings = resolve_continuation_preview_settings(config, environ={})

    assert settings.preview_enabled is True
    assert settings.timeout_seconds == 42
    assert settings.route is not None
    assert settings.route.provider == "bedrock"
    assert settings.route.model == "global.anthropic.claude-sonnet-5"
    assert settings.route.region == "us-west-2"
    assert settings.route.config_source == "auxiliary.continuation_checkpoint"
    assert settings.route.max_output_tokens == BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS
    assert len(settings.config_digest) == 64
    assert len(settings.route.route_digest) == 64
    assert json.dumps(config, sort_keys=True) == before


def test_settings_fall_back_to_pinned_compression_bedrock_route():
    config = {
        "continuation_checkpoint": {"preview_enabled": True},
        "bedrock": {"region": ""},
        "auxiliary": {
            "continuation_checkpoint": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
            "compression": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "timeout": 90,
            },
        },
    }

    settings = resolve_continuation_preview_settings(
        config,
        environ={"AWS_REGION": "eu-west-1"},
    )

    assert settings.route is not None
    assert settings.route.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert settings.route.region == "eu-west-1"
    assert settings.route.config_source == "auxiliary.compression"
    assert settings.timeout_seconds == 90


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"provider": "openrouter"}, "Bedrock"),
        ({"provider": "auto", "model": "some-model"}, "Bedrock"),
        ({"model": ""}, "model"),
        ({"base_url": "https://example.test"}, "base_url"),
        ({"api_key": "must-not-be-used"}, "api_key"),
        ({"extra_body": {"temperature": 0}}, "extra_body"),
        ({"reasoning_effort": "low"}, "reasoning_effort"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": 601}, "timeout"),
    ],
)
def test_enabled_settings_fail_closed_for_unbounded_or_mutable_route(update, match):
    config = _config()
    config["auxiliary"]["continuation_checkpoint"].update(update)

    with pytest.raises(ContinuationPreviewConfigurationError, match=match):
        resolve_continuation_preview_settings(config, environ={})


def test_disabled_settings_do_not_resolve_or_validate_a_route():
    config = {
        "continuation_checkpoint": {"preview_enabled": False},
        "auxiliary": {
            "continuation_checkpoint": {
                "provider": "unsupported",
                "model": "",
                "api_key": "not-read-while-disabled",
            }
        },
    }

    settings = resolve_continuation_preview_settings(config, environ={})

    assert settings.preview_enabled is False
    assert settings.route is None
    assert settings.timeout_seconds is None


def test_preview_enabled_requires_a_real_boolean():
    with pytest.raises(ContinuationPreviewConfigurationError, match="preview_enabled"):
        resolve_continuation_preview_settings(
            {"continuation_checkpoint": {"preview_enabled": "true"}},
            environ={},
        )


def test_resolve_credentials_returns_an_immutable_frozen_snapshot():
    frozen = SimpleNamespace(
        access_key="resolved-key",
        secret_key="resolved-secret",
        token="resolved-token",
    )
    refreshable = SimpleNamespace(get_frozen_credentials=lambda: frozen)
    session = SimpleNamespace(get_credentials=lambda: refreshable)

    result = resolve_bedrock_credential_snapshot(session=session)

    assert result.access_key_id == "resolved-key"
    assert result.secret_access_key == "resolved-secret"
    assert result.session_token == "resolved-token"
    with pytest.raises(AttributeError):
        result.access_key_id = "changed"
    assert "resolved-secret" not in repr(result)


def test_resolve_credentials_fails_closed_when_refresh_or_auth_is_required():
    session = SimpleNamespace(get_credentials=lambda: None)
    with pytest.raises(ContinuationPreviewConfigurationError, match="credentials"):
        resolve_bedrock_credential_snapshot(session=session)


def test_strict_projector_runs_detached_worker_with_scrubbed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/secret/profile")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-key-must-not-leak")
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_echo_worker(tmp_path),
    )

    response = projector(_request())
    payload = json.loads(response.raw_json)

    assert payload == {
        "attempt": 1,
        "kind": "primary",
        "prompt": _request().prompt,
        "model": "global.anthropic.claude-sonnet-5",
        "region": "us-west-2",
        "max_output_tokens": BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
        "hermes_home": None,
        "inherited_access_key": None,
        "worker_home": "/",
    }
    assert response.metadata.projector == "bedrock/global.anthropic.claude-sonnet-5"
    assert response.metadata.latency_ms == 7
    assert response.metadata.input_tokens == 11
    assert response.metadata.output_tokens == 13
    assert response.metadata.config_digest == projector.settings.config_digest
    assert response.metadata.route_digest == projector.settings.route.route_digest
    assert projector.calls == 1
    assert projector.worker_command[1:3] == ("-I", "-B")


def test_strict_projector_allows_only_primary_then_one_repair(tmp_path):
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_echo_worker(tmp_path),
    )

    projector(_request(1))
    projector(_request(2))

    with pytest.raises(ProjectorTransportError, match="at most two"):
        projector(_request(2))
    assert projector.calls == 2


def test_strict_projector_rejects_repair_as_first_call_without_spawning(tmp_path):
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_echo_worker(tmp_path),
    )

    with pytest.raises(ProjectorTransportError, match="call order"):
        projector(_request(2))
    assert projector.calls == 0


def test_cumulative_deadline_kills_the_second_worker_process_group(tmp_path):
    marker = tmp_path / "survived.txt"
    worker = _write_worker(
        tmp_path,
        f"""
import json, subprocess, sys, time
request = json.load(sys.stdin)
if request["attempt"] == 1:
    time.sleep(0.10)
    json.dump({{
        "protocol": {BEDROCK_PROJECTOR_PROTOCOL_V1!r}, "ok": True,
        "raw_json": "{{}}", "latency_ms": 100,
        "input_tokens": 1, "output_tokens": 1,
    }}, sys.stdout)
else:
    subprocess.Popen([
        sys.executable, "-c",
        "import pathlib,time; time.sleep(0.30); pathlib.Path({str(marker)!r}).write_text('alive')",
    ])
    time.sleep(5)
""",
    )
    settings = replace(_settings(), timeout_seconds=0.22)
    projector = StrictBedrockProjectorV1(settings, _credentials(), worker_command=worker)

    projector(_request(1))
    with pytest.raises(TimeoutError):
        projector(_request(2))

    time.sleep(0.45)
    assert marker.exists() is False
    assert projector.calls == 2


def test_cancellation_callback_kills_worker_and_is_typed(tmp_path):
    worker = _write_worker(
        tmp_path,
        "import json,sys,time\njson.load(sys.stdin)\ntime.sleep(5)\n",
    )
    cancelled = threading.Event()
    threading.Timer(0.08, cancelled.set).start()
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=worker,
        cancellation_requested=cancelled.is_set,
    )

    with pytest.raises(concurrent.futures.CancelledError):
        projector(_request())


def test_worker_stdout_and_stderr_are_hard_bounded(tmp_path):
    too_much_stdout = _write_worker(
        tmp_path,
        f"import sys\nsys.stdout.buffer.write(b'x' * {WORKER_STDOUT_MAX_BYTES + 1})\n",
    )
    stdout_projector = StrictBedrockProjectorV1(
        _settings(), _credentials(), worker_command=too_much_stdout
    )
    with pytest.raises(ProjectorResponseError, match="stdout"):
        stdout_projector(_request())

    too_much_stderr = _write_worker(
        tmp_path,
        f"import sys\nsys.stderr.buffer.write(b'x' * {WORKER_STDERR_MAX_BYTES + 1})\n",
    )
    stderr_projector = StrictBedrockProjectorV1(
        _settings(), _credentials(), worker_command=too_much_stderr
    )
    with pytest.raises(ProjectorTransportError, match="stderr"):
        stderr_projector(_request())


@pytest.mark.parametrize(
    ("source", "error_type", "match"),
    [
        ("print('not-json')", ProjectorResponseError, "protocol"),
        (
            f"import json; print(json.dumps({{'protocol': {BEDROCK_PROJECTOR_PROTOCOL_V1!r}, 'ok': False, 'error_code': 'bedrock_error'}}))",
            ProjectorTransportError,
            "bedrock_error",
        ),
        (
            f"import json; print(json.dumps({{'protocol': {BEDROCK_PROJECTOR_PROTOCOL_V1!r}, 'ok': False, 'error_code': 'output_too_large'}}))",
            ProjectorResponseError,
            "output_too_large",
        ),
    ],
)
def test_worker_protocol_failures_are_typed(tmp_path, source, error_type, match):
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_write_worker(tmp_path, source),
    )

    with pytest.raises(error_type, match=match):
        projector(_request())


def test_worker_transport_error_preserves_a_bounded_machine_code(tmp_path):
    source = f"""
import json
print(json.dumps({{
    "protocol": {BEDROCK_PROJECTOR_PROTOCOL_V1!r},
    "ok": False,
    "error_code": "rate_limited",
}}))
"""
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_write_worker(tmp_path, source),
    )

    with pytest.raises(BedrockProjectorTransportError) as raised:
        projector(_request())

    assert raised.value.code is BedrockProjectorFailureCode.RATE_LIMITED


def test_unknown_worker_error_code_cannot_escape_subprocess_text(tmp_path):
    untrusted = "unknown-secret-shaped-worker-detail"
    source = f"""
import json
print(json.dumps({{
    "protocol": {BEDROCK_PROJECTOR_PROTOCOL_V1!r},
    "ok": False,
    "error_code": {untrusted!r},
}}))
"""
    projector = StrictBedrockProjectorV1(
        _settings(),
        _credentials(),
        worker_command=_write_worker(tmp_path, source),
    )

    with pytest.raises(ProjectorResponseError) as raised:
        projector(_request())

    assert untrusted not in str(raised.value)


def _worker_payload(**updates):
    payload = {
        "protocol": BEDROCK_PROJECTOR_PROTOCOL_V1,
        "kind": "primary",
        "attempt": 1,
        "prompt": "Return JSON.",
        "region": "us-east-1",
        "model": "global.anthropic.claude-sonnet-5",
        "max_output_tokens": BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
        "max_output_bytes": PROJECTOR_OUTPUT_MAX_BYTES,
        "connect_timeout_seconds": 10.0,
        "read_timeout_seconds": 30.0,
        "credentials": {
            "access_key_id": "key",
            "secret_access_key": "secret",
            "session_token": "token",
        },
    }
    payload.update(updates)
    return payload


def test_worker_validation_rejects_unknown_fields_and_unbounded_values():
    with pytest.raises(WorkerInputError, match="fields"):
        _validate_request({**_worker_payload(), "fallback_provider": "openrouter"})
    with pytest.raises(WorkerInputError, match="max_output_tokens"):
        _validate_request(_worker_payload(max_output_tokens=BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS + 1))
    with pytest.raises(WorkerInputError, match="max_output_bytes"):
        _validate_request(_worker_payload(max_output_bytes=PROJECTOR_OUTPUT_MAX_BYTES + 1))


def test_worker_calls_bedrock_once_with_static_credentials_and_no_sampling_parameters():
    captured = {}

    class FakeClient:
        def converse(self, **kwargs):
            captured["converse"] = kwargs
            return {
                "output": {"message": {"content": [{"text": '{"objective":"ok"}'}]}},
                "usage": {"inputTokens": 21, "outputTokens": 8},
            }

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            captured["service"] = service
            captured["client"] = kwargs
            return FakeClient()

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    result = _project_bedrock(
        _validate_request(_worker_payload()),
        boto3_module=FakeBoto3,
        config_type=FakeConfig,
        monotonic=lambda: 1.0,
    )

    assert captured["service"] == "bedrock-runtime"
    assert captured["client"]["aws_access_key_id"] == "key"
    assert captured["client"]["aws_secret_access_key"] == "secret"
    assert captured["client"]["aws_session_token"] == "token"
    assert captured["config"]["retries"] == {"total_max_attempts": 1, "mode": "standard"}
    inference = captured["converse"]["inferenceConfig"]
    assert inference == {"maxTokens": BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS}
    assert "temperature" not in json.dumps(captured["converse"]).lower()
    assert "key" not in json.dumps(captured["converse"])
    assert "secret" not in json.dumps(captured["converse"])
    assert result["raw_json"] == '{"objective":"ok"}'
    assert result["input_tokens"] == 21
    assert result["output_tokens"] == 8


def test_worker_enforces_local_byte_and_token_caps_after_one_bedrock_call():
    class FakeBoto3:
        calls = 0

        @classmethod
        def client(cls, _service, **_kwargs):
            class Client:
                def converse(self, **_kwargs):
                    cls.calls += 1
                    return {
                        "output": {
                            "message": {
                                "content": [{"text": "x" * (PROJECTOR_OUTPUT_MAX_BYTES + 1)}]
                            }
                        },
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS + 1,
                        },
                    }

            return Client()

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    with pytest.raises(WorkerInputError, match="output"):
        _project_bedrock(
            _validate_request(_worker_payload()),
            boto3_module=FakeBoto3,
            config_type=FakeConfig,
        )
    assert FakeBoto3.calls == 1


@pytest.mark.parametrize(
    ("error_code", "http_status", "expected"),
    [
        ("ExpiredTokenException", 403, "auth_refresh_required"),
        ("UnknownAuthFailure", 401, "auth_refresh_required"),
        ("AccessDeniedException", 403, "access_denied"),
        ("PaymentRequiredException", 402, "payment_required"),
        ("ThrottlingException", 429, "rate_limited"),
        ("ValidationException", 400, "request_rejected"),
        ("InternalServerException", 500, "provider_unavailable"),
    ],
)
def test_worker_classifies_provider_failures_without_serializing_exception_text(
    error_code, http_status, expected
):
    class FakeClientError(Exception):
        response = {
            "Error": {"Code": error_code},
            "ResponseMetadata": {"HTTPStatusCode": http_status},
        }

    error = FakeClientError("sensitive provider detail must not cross the boundary")

    assert _classify_bedrock_exception(error) == expected


def test_worker_python_audit_hook_rejects_write_intent_but_allows_reads():
    _deny_local_write_audit_event("open", ("/tmp/read-only", "r", 0))
    _deny_local_write_audit_event("open", ("/tmp/read-only", None, os.O_RDONLY))

    with pytest.raises(PermissionError, match="write"):
        _deny_local_write_audit_event("open", ("/tmp/nope", "wb", 0))
    with pytest.raises(PermissionError, match="write"):
        _deny_local_write_audit_event(
            "open", ("/tmp/nope", None, os.O_WRONLY | os.O_CREAT)
        )
    with pytest.raises(PermissionError, match="write"):
        _deny_local_write_audit_event("os.remove", ("/tmp/nope", -1))
