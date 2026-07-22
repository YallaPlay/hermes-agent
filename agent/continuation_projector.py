"""Strict, detached Bedrock projector for read-only continuation previews.

This module owns two boundaries:

* pure resolution from an already-loaded configuration mapping; it never calls
  Hermes config/plugin/provider discovery itself, and
* a one-route subprocess adapter with static credentials, a cumulative
  deadline, no fallback/retry/accounting, and bounded streamed output.

The corresponding worker is intentionally standalone and installs a Python
write-denying audit hook before it imports boto3 or opens the network client.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agent.continuation_preview import (
    PROJECTOR_OUTPUT_MAX_BYTES,
    ProjectorCallMetadataV1,
    ProjectorRequestKind,
    ProjectorRequestV1,
    ProjectorResponseError,
    ProjectorResponseV1,
    ProjectorTransportError,
)
from agent.continuation_projector_worker import (
    BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
    BEDROCK_PROJECTOR_PROTOCOL_V1,
)

WORKER_STDOUT_MAX_BYTES = PROJECTOR_OUTPUT_MAX_BYTES * 6 + 8_192
WORKER_STDERR_MAX_BYTES = 8_192
_WORKER_TRANSPORT_INPUT_MAX_BYTES = 500_000
_DEFAULT_PROJECTOR_TIMEOUT_SECONDS = 600.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$", re.IGNORECASE)


class ContinuationPreviewConfigurationError(ValueError):
    """Preview settings cannot satisfy the strict bounded Bedrock contract."""


class BedrockProjectorFailureCode(str, Enum):
    AUTH_REFRESH_REQUIRED = "auth_refresh_required"
    ACCESS_DENIED = "access_denied"
    PAYMENT_REQUIRED = "payment_required"
    RATE_LIMITED = "rate_limited"
    REQUEST_REJECTED = "request_rejected"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    BEDROCK_ERROR = "bedrock_error"


class BedrockProjectorTransportError(ProjectorTransportError):
    """Typed, content-free failure returned by the detached Bedrock worker."""

    def __init__(self, code: BedrockProjectorFailureCode) -> None:
        self.code = code
        super().__init__(f"strict Bedrock worker transport failed ({code.value})")


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BedrockProjectorRouteV1:
    provider: str
    model: str
    region: str
    config_source: str
    max_output_tokens: int
    route_digest: str

    def __post_init__(self) -> None:
        if self.provider != "bedrock":
            raise ContinuationPreviewConfigurationError("strict route must use Bedrock")
        if not self.model:
            raise ContinuationPreviewConfigurationError("strict Bedrock route requires a model")
        if not _REGION_PATTERN.fullmatch(self.region):
            raise ContinuationPreviewConfigurationError("strict Bedrock route region is invalid")
        if self.config_source not in {
            "auxiliary.continuation_checkpoint",
            "auxiliary.compression",
        }:
            raise ContinuationPreviewConfigurationError("strict Bedrock route source is invalid")
        if self.max_output_tokens != BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS:
            raise ContinuationPreviewConfigurationError("strict Bedrock wire cap is fixed")
        if not _SHA256_PATTERN.fullmatch(self.route_digest):
            raise ContinuationPreviewConfigurationError("strict Bedrock route digest is invalid")


@dataclass(frozen=True)
class ContinuationPreviewSettingsV1:
    preview_enabled: bool
    timeout_seconds: float | None
    route: BedrockProjectorRouteV1 | None
    config_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.preview_enabled, bool):
            raise ContinuationPreviewConfigurationError("preview_enabled must be boolean")
        if not _SHA256_PATTERN.fullmatch(self.config_digest):
            raise ContinuationPreviewConfigurationError("preview config digest is invalid")
        if self.preview_enabled:
            if self.route is None or self.timeout_seconds is None:
                raise ContinuationPreviewConfigurationError(
                    "enabled continuation preview requires a strict route"
                )
            if not 0 < self.timeout_seconds <= 600:
                raise ContinuationPreviewConfigurationError("preview timeout is outside its bound")
        elif self.route is not None or self.timeout_seconds is not None:
            raise ContinuationPreviewConfigurationError(
                "disabled continuation preview cannot resolve a projector route"
            )


@dataclass(frozen=True)
class BedrockCredentialSnapshotV1:
    """Already-refreshed AWS credentials copied before the strict region."""

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.access_key_id, str) or not self.access_key_id:
            raise ContinuationPreviewConfigurationError("Bedrock credentials have no access key")
        if not isinstance(self.secret_access_key, str) or not self.secret_access_key:
            raise ContinuationPreviewConfigurationError("Bedrock credentials have no secret key")
        if self.session_token is not None and not isinstance(self.session_token, str):
            raise ContinuationPreviewConfigurationError(
                "Bedrock credentials have an invalid session token"
            )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContinuationPreviewConfigurationError(f"{path} must be a mapping")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _route_is_configured(route: Mapping[str, Any]) -> bool:
    provider = _text(route.get("provider")).lower()
    return bool(
        (provider and provider != "auto")
        or _text(route.get("model"))
        or _text(route.get("base_url"))
        or _text(route.get("api_key"))
        or route.get("extra_body")
        or _text(route.get("reasoning_effort"))
    )


def _timeout(value: Any, *, default: float) -> float:
    candidate = default if value is None else value
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        raise ContinuationPreviewConfigurationError("strict Bedrock timeout must be numeric")
    result = float(candidate)
    if not 0 < result <= 600:
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock timeout must be greater than zero and at most 600 seconds"
        )
    return result


def _resolve_region(config: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    bedrock = _mapping(config.get("bedrock"), "bedrock")
    region = (
        _text(bedrock.get("region"))
        or _text(environ.get("AWS_REGION"))
        or _text(environ.get("AWS_DEFAULT_REGION"))
        or "us-east-1"
    )
    if not _REGION_PATTERN.fullmatch(region):
        raise ContinuationPreviewConfigurationError("strict Bedrock region is invalid")
    return region


def resolve_continuation_preview_settings(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> ContinuationPreviewSettingsV1:
    """Resolve immutable settings from caller-owned, already-loaded values.

    This function is intentionally pure: it does not load config, discover
    plugins/models, resolve credentials, mutate caches, or inspect AWS files.
    A blank continuation route inherits a *pinned* compression route; there is
    no main-model, provider-auto, or provider fallback inside the strict path.
    """

    if not isinstance(config, Mapping):
        raise ContinuationPreviewConfigurationError("preview config must be a mapping")
    feature = _mapping(config.get("continuation_checkpoint"), "continuation_checkpoint")
    enabled = feature.get("preview_enabled", False)
    if not isinstance(enabled, bool):
        raise ContinuationPreviewConfigurationError(
            "continuation_checkpoint.preview_enabled must be boolean"
        )
    if not enabled:
        return ContinuationPreviewSettingsV1(
            preview_enabled=False,
            timeout_seconds=None,
            route=None,
            config_digest=_digest({"preview_enabled": False}),
        )

    auxiliary = _mapping(config.get("auxiliary"), "auxiliary")
    continuation = _mapping(
        auxiliary.get("continuation_checkpoint"),
        "auxiliary.continuation_checkpoint",
    )
    if _route_is_configured(continuation):
        route_config = continuation
        source = "auxiliary.continuation_checkpoint"
        timeout_default = _DEFAULT_PROJECTOR_TIMEOUT_SECONDS
    else:
        route_config = _mapping(auxiliary.get("compression"), "auxiliary.compression")
        source = "auxiliary.compression"
        timeout_default = 120.0

    provider = _text(route_config.get("provider")).lower()
    model = _text(route_config.get("model"))
    if provider != "bedrock":
        raise ContinuationPreviewConfigurationError(
            "read-only continuation preview requires an explicitly pinned Bedrock route"
        )
    if not model:
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock continuation projector requires a model"
        )
    if _text(route_config.get("base_url")):
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock continuation projector forbids base_url"
        )
    if _text(route_config.get("api_key")):
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock continuation projector forbids api_key"
        )
    extra_body = route_config.get("extra_body", {})
    if extra_body not in (None, {}):
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock continuation projector forbids extra_body"
        )
    if _text(route_config.get("reasoning_effort")):
        raise ContinuationPreviewConfigurationError(
            "strict Bedrock continuation projector forbids reasoning_effort"
        )

    timeout_seconds = _timeout(route_config.get("timeout"), default=timeout_default)
    region = _resolve_region(config, environ if environ is not None else os.environ)
    route_material = {
        "provider": "bedrock",
        "model": model,
        "region": region,
        "config_source": source,
        "max_output_tokens": BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
        "timeout_seconds": timeout_seconds,
        "retries": 0,
        "fallback": False,
        "accounting": False,
    }
    route = BedrockProjectorRouteV1(
        provider="bedrock",
        model=model,
        region=region,
        config_source=source,
        max_output_tokens=BEDROCK_PROJECTOR_MAX_OUTPUT_TOKENS,
        route_digest=_digest(route_material),
    )
    return ContinuationPreviewSettingsV1(
        preview_enabled=True,
        timeout_seconds=timeout_seconds,
        route=route,
        config_digest=_digest(
            {
                "preview_enabled": True,
                "route": route_material,
            }
        ),
    )


def resolve_bedrock_credential_snapshot(*, session: Any = None) -> BedrockCredentialSnapshotV1:
    """Freeze the active AWS credential before entering the strict worker.

    ``get_frozen_credentials`` may refresh an SSO/assume-role source, so callers
    must invoke this while preparing preview settings, never from the worker.
    """

    try:
        if session is None:
            import botocore.session

            session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is None:
            raise ContinuationPreviewConfigurationError(
                "Bedrock credentials are unavailable or require authentication"
            )
        frozen = credentials.get_frozen_credentials()
        if frozen is None:
            raise ContinuationPreviewConfigurationError(
                "Bedrock credentials could not be frozen"
            )
        return BedrockCredentialSnapshotV1(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=getattr(frozen, "token", None),
        )
    except ContinuationPreviewConfigurationError:
        raise
    except Exception as exc:
        raise ContinuationPreviewConfigurationError(
            "Bedrock credentials could not be resolved before the strict preview boundary"
        ) from exc


def _default_worker_command() -> tuple[str, ...]:
    worker = Path(__file__).with_name("continuation_projector_worker.py").resolve()
    return (sys.executable, "-I", "-B", str(worker))


def _worker_environment(source: Mapping[str, str]) -> dict[str, str]:
    allowed = (
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "AWS_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    )
    env = {key: value for key in allowed if (value := source.get(key))}
    env.update(
        {
            "HOME": "/",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONWARNINGS": "ignore",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_METADATA_SERVICE_TIMEOUT": "1",
            "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "1",
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
            "AWS_CONFIG_FILE": os.devnull,
            "BOTO_CONFIG": os.devnull,
        }
    )
    return env


class _BoundedPipeReader(threading.Thread):
    def __init__(self, pipe: Any, limit: int) -> None:
        super().__init__(daemon=True)
        self._pipe = pipe
        self._limit = limit
        self.data = bytearray()
        self.overflow = threading.Event()

    def run(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(65_536)
                if not chunk:
                    return
                remaining = self._limit - len(self.data)
                if len(chunk) > remaining:
                    if remaining > 0:
                        self.data.extend(chunk[:remaining])
                    self.overflow.set()
                    return
                self.data.extend(chunk)
        except (OSError, ValueError):
            return


class _PipeWriter(threading.Thread):
    def __init__(self, pipe: Any, data: bytes) -> None:
        super().__init__(daemon=True)
        self._pipe = pipe
        self._data = data

    def run(self) -> None:
        try:
            self._pipe.write(self._data)
            self._pipe.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                self._pipe.close()
            except (OSError, ValueError):
                pass


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows CI only
            proc.kill()
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProjectorResponseError("worker protocol contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ProjectorResponseError("worker protocol contains a non-JSON constant")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except ProjectorResponseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectorResponseError("worker protocol is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ProjectorResponseError("worker protocol root is not an object")
    return value


class StrictBedrockProjectorV1:
    """Synchronous ``ProjectorV1`` backed by one detached worker per attempt."""

    def __init__(
        self,
        settings: ContinuationPreviewSettingsV1,
        credentials: BedrockCredentialSnapshotV1,
        *,
        worker_command: tuple[str, ...] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        environ: Mapping[str, str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not settings.preview_enabled or settings.route is None or settings.timeout_seconds is None:
            raise ContinuationPreviewConfigurationError(
                "strict Bedrock projector requires enabled preview settings"
            )
        if not isinstance(credentials, BedrockCredentialSnapshotV1):
            raise ContinuationPreviewConfigurationError(
                "strict Bedrock projector requires frozen credentials"
            )
        command = tuple(worker_command or _default_worker_command())
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ContinuationPreviewConfigurationError("worker command is invalid")
        self.settings = settings
        self._credentials = credentials
        self._worker_command = command
        self._cancellation_requested = cancellation_requested or (lambda: False)
        self._environment = _worker_environment(environ if environ is not None else os.environ)
        self._monotonic = monotonic
        self._deadline: float | None = None
        self._calls = 0
        self._call_lock = threading.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def worker_command(self) -> tuple[str, ...]:
        return self._worker_command

    def __call__(self, request: ProjectorRequestV1) -> ProjectorResponseV1:
        if not isinstance(request, ProjectorRequestV1):
            raise ProjectorTransportError("strict projector request has an invalid type")
        if not self._call_lock.acquire(blocking=False):
            raise ProjectorTransportError("strict projector does not allow concurrent calls")
        try:
            expected_attempt = self._calls + 1
            expected_kind = (
                ProjectorRequestKind.PRIMARY
                if expected_attempt == 1
                else ProjectorRequestKind.REPAIR
            )
            if expected_attempt > 2:
                raise ProjectorTransportError("strict projector permits at most two calls")
            if request.attempt != expected_attempt or request.kind is not expected_kind:
                raise ProjectorTransportError("strict projector call order is invalid")
            if self._deadline is None:
                assert self.settings.timeout_seconds is not None
                self._deadline = self._monotonic() + self.settings.timeout_seconds
            if self._cancellation_requested():
                raise concurrent.futures.CancelledError()
            remaining = self._deadline - self._monotonic()
            if remaining <= 0:
                raise TimeoutError("strict projector cumulative deadline expired")

            self._calls += 1
            return self._invoke_worker(request, remaining)
        finally:
            self._call_lock.release()

    def _payload(self, request: ProjectorRequestV1, remaining: float) -> bytes:
        route = self.settings.route
        assert route is not None
        value = {
            "protocol": BEDROCK_PROJECTOR_PROTOCOL_V1,
            "kind": request.kind.value,
            "attempt": request.attempt,
            "prompt": request.prompt,
            "region": route.region,
            "model": route.model,
            "max_output_tokens": route.max_output_tokens,
            "max_output_bytes": request.max_output_bytes,
            "connect_timeout_seconds": min(_DEFAULT_CONNECT_TIMEOUT_SECONDS, remaining),
            "read_timeout_seconds": min(600.0, remaining),
            "credentials": {
                "access_key_id": self._credentials.access_key_id,
                "secret_access_key": self._credentials.secret_access_key,
                "session_token": self._credentials.session_token,
            },
        }
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(payload) > _WORKER_TRANSPORT_INPUT_MAX_BYTES:
            raise ProjectorResponseError("worker input transport exceeds its byte cap")
        return payload

    def _invoke_worker(self, request: ProjectorRequestV1, remaining: float) -> ProjectorResponseV1:
        payload = self._payload(request, remaining)
        try:
            proc = subprocess.Popen(
                self._worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.abspath(os.sep),
                env=self._environment,
                close_fds=True,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise ProjectorTransportError("strict Bedrock worker could not start") from exc
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

        stdout_reader = _BoundedPipeReader(proc.stdout, WORKER_STDOUT_MAX_BYTES)
        stderr_reader = _BoundedPipeReader(proc.stderr, WORKER_STDERR_MAX_BYTES)
        stdin_writer = _PipeWriter(proc.stdin, payload)
        stdout_reader.start()
        stderr_reader.start()
        stdin_writer.start()

        try:
            while proc.poll() is None:
                if stdout_reader.overflow.is_set():
                    raise ProjectorResponseError("worker stdout exceeded its hard bound")
                if stderr_reader.overflow.is_set():
                    raise ProjectorTransportError("worker stderr exceeded its hard bound")
                if self._cancellation_requested():
                    raise concurrent.futures.CancelledError()
                assert self._deadline is not None
                if self._monotonic() >= self._deadline:
                    raise TimeoutError("strict projector cumulative deadline expired")
                time.sleep(0.01)

            if stdout_reader.overflow.is_set():
                raise ProjectorResponseError("worker stdout exceeded its hard bound")
            if stderr_reader.overflow.is_set():
                raise ProjectorTransportError("worker stderr exceeded its hard bound")
            if proc.returncode != 0:
                raise ProjectorTransportError("strict Bedrock worker exited unsuccessfully")
        except BaseException:
            _kill_process_group(proc)
            raise
        finally:
            stdin_writer.join(timeout=0.2)
            stdout_reader.join(timeout=0.2)
            stderr_reader.join(timeout=0.2)
            if stdout_reader.is_alive() or stderr_reader.is_alive():
                _kill_process_group(proc)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass

        # Readers may observe the final bytes between ``poll()`` and ``join()``;
        # enforce both caps again after the pipes have been drained/closed.
        if stdout_reader.overflow.is_set():
            raise ProjectorResponseError("worker stdout exceeded its hard bound")
        if stderr_reader.overflow.is_set():
            raise ProjectorTransportError("worker stderr exceeded its hard bound")

        envelope = _strict_json_object(bytes(stdout_reader.data))
        if envelope.get("protocol") != BEDROCK_PROJECTOR_PROTOCOL_V1:
            raise ProjectorResponseError("worker protocol version is invalid")
        ok = envelope.get("ok")
        if not isinstance(ok, bool):
            raise ProjectorResponseError("worker protocol status is invalid")
        if not ok:
            if set(envelope) != {"protocol", "ok", "error_code"}:
                raise ProjectorResponseError("worker error protocol fields are invalid")
            code = envelope.get("error_code")
            if not isinstance(code, str) or not code:
                raise ProjectorResponseError("worker error protocol code is invalid")
            if code in {
                "protocol_error",
                "invalid_response",
                "output_too_large",
                "output_tokens_exceeded",
            }:
                raise ProjectorResponseError(f"strict Bedrock worker failed: {code}")
            try:
                failure_code = BedrockProjectorFailureCode(code)
            except ValueError as exc:
                raise ProjectorResponseError(
                    "worker error protocol code is not recognized"
                ) from exc
            raise BedrockProjectorTransportError(failure_code)

        required = {
            "protocol",
            "ok",
            "raw_json",
            "latency_ms",
            "input_tokens",
            "output_tokens",
        }
        if set(envelope) != required:
            raise ProjectorResponseError("worker success protocol fields are invalid")
        raw_json = envelope.get("raw_json")
        if not isinstance(raw_json, str):
            raise ProjectorResponseError("worker raw_json is invalid")
        if len(raw_json.encode("utf-8")) > request.max_output_bytes:
            raise ProjectorResponseError("worker raw_json exceeds its local byte cap")
        for name in ("latency_ms", "input_tokens", "output_tokens"):
            value = envelope.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProjectorResponseError(f"worker {name} is invalid")
        route = self.settings.route
        assert route is not None
        if envelope["output_tokens"] > route.max_output_tokens:
            raise ProjectorResponseError("worker output exceeds its local token cap")

        return ProjectorResponseV1(
            raw_json=raw_json,
            metadata=ProjectorCallMetadataV1(
                projector=f"bedrock/{route.model}",
                latency_ms=envelope["latency_ms"],
                input_tokens=envelope["input_tokens"],
                output_tokens=envelope["output_tokens"],
                config_digest=self.settings.config_digest,
                route_digest=route.route_digest,
            ),
        )
