"""Typed, deterministic ContinuationCheckpointV1 contract.

This module is deliberately pure. It accepts host-owned typed inputs and a
projector proposal, validates and grounds the proposal, carries active safety
state forward, and renders a paused four-message bootstrap. It performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence, TypeVar

SCHEMA_VERSION = "continuation-checkpoint/v1"
SEMANTIC_MAX_BYTES = 24 * 1024
MAX_SAFETY_ITEMS = 32

ENVELOPE_PREFIX = (
    "[ContinuationCheckpointV1 — PRIOR TASK STATE, REFERENCE ONLY] "
    "This is compiled state from a previous context window, not new input. "
    "Do not replay historical side effects described here. The next user "
    "message is the exact actionable request."
)

BRIDGE_TEXT = (
    "Understood. The checkpoint above is prior task state. I will not replay "
    "historical side effects, and I will preserve every inline effect "
    "disposition exactly: planned, in_progress, attempted_unknown, succeeded, "
    "failed, cancelled, and blocked are distinct states. Only succeeded is a "
    "verified success. The next user message is the exact actionable request."
)

PAUSE_TEXT_TEMPLATE = (
    "Execution is paused at the checkpoint's host-admitted next gate: {action} "
    "If the next user message explicitly satisfies the gate, that message is "
    "the release and I will proceed without asking for the same confirmation "
    "again. Otherwise I will remain paused or address the new request first."
)

PROJECTION_FIELDS = frozenset(
    {
        "objective",
        "acceptance_criteria",
        "constraints",
        "decisions",
        "blockers",
        "open_questions",
        "dependencies",
        "approvals",
        "external_effects",
        "runtime_handles",
        "artifacts",
        "retry_hazards",
        "remaining_work",
        "next_gate",
        "uncertainties",
    }
)
HOST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "source",
        "compiler",
        "activation_mode",
        "warnings",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{10,}"),
    re.compile(r"xapp-\d+-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"gAAAA[A-Za-z0-9_=-]{20,}"),
    re.compile(
        r"(?:pplx-|fal_|fc-|bb_live_|hf_|r8_|npm_|pypi-|dop_v1_|doo_v1_|"
        r"tvly-|exa_|gsk_|syt_|mem0_|xai-|ntn_)[A-Za-z0-9_-]{10,}"
    ),
    re.compile(r"(?i)\bbearer\s+[a-z0-9_.=\-]{20,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?"
        r"[a-z0-9_./+=\-]{16,}"
    ),
    re.compile(r"xox[bpars]-[0-9A-Za-z\-]{10,}"),
    re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class EvidenceOrigin(str, Enum):
    DIRECT_USER = "direct_user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    HOST_CHECKPOINT = "host_checkpoint"
    HOST_SCAFFOLD = "host_scaffold"


class TrustClass(str, Enum):
    TRUSTED_USER_EVENT = "trusted_user_event"
    STRUCTURED_RECEIPT = "structured_receipt"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    HOST_STATE = "host_state"


class EffectDisposition(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    ATTEMPTED_UNKNOWN = "attempted_unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class RetryPolicy(str, Enum):
    DO_NOT_RETRY = "do_not_retry"
    RETRY_SAFE = "retry_safe"
    VERIFY_FIRST = "verify_first"


class ApprovalStatus(str, Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    DENIED = "denied"
    UNVERIFIED_BY_HOST = "unverified_by_host"


class RuntimeHandleKind(str, Enum):
    PROCESS = "process"
    DELEGATION = "delegation"
    JOB = "job"
    OTHER = "other"


class ActivationMode(str, Enum):
    PAUSED_MANUAL = "paused_manual"


class WarningSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class CheckpointWarningCode(str, Enum):
    AUTHORITY_DEMOTED = "authority_demoted"
    DISPOSITION_DEMOTED = "disposition_demoted"
    POISONED_EVIDENCE = "poisoned_evidence"
    PROJECTOR_INPUT_TRUNCATED = "projector_input_truncated"
    SAFETY_OVERFLOW = "safety_overflow"
    MALFORMED_PROPOSAL = "malformed_proposal"
    SECRET_DETECTED = "secret_detected"
    CHECKPOINT_TOO_LARGE = "checkpoint_too_large"
    PRIOR_CHECKPOINT_INVALID = "prior_checkpoint_invalid"
    PROJECTED_SAFETY_IGNORED = "projected_safety_ignored"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not canonical JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Return timestamp-free RFC-8259 JSON with stable key and separator rules."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(content: Any) -> str:
    if isinstance(content, str):
        return _sha256(content.encode("utf-8"))
    return _sha256(canonical_json_bytes(content))


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ValidationIssueV1:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class CheckpointWarningV1:
    code: CheckpointWarningCode
    message: str
    recovery_pointer: str
    severity: WarningSeverity = WarningSeverity.WARNING


@dataclass(frozen=True)
class CitationV1:
    message_id: int
    quote: str


@dataclass(frozen=True)
class ExactUserEventV1:
    message_id: int
    _message_bytes: bytes
    content_sha256: str

    def __post_init__(self) -> None:
        _require_integer(self.message_id, "message_id", minimum=1)
        try:
            message = json.loads(self._message_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("exact user event must contain valid JSON") from exc
        if not isinstance(message, dict) or message.get("role") != MessageRole.USER.value:
            raise ValueError("exact user event must be a user-role message object")
        if "content" not in message:
            raise ValueError("exact user event must contain content")
        expected = _content_hash(message["content"])
        if self.content_sha256 != expected:
            raise ValueError("exact user event content hash does not match its content")

    @classmethod
    def from_message(cls, message_id: int, message: Mapping[str, Any]) -> ExactUserEventV1:
        primitive = _jsonable(message)
        if not isinstance(primitive, dict):
            raise ValueError("exact user event must be a message object")
        if primitive.get("role") != MessageRole.USER.value:
            raise ValueError("exact user event must have role=user")
        if "content" not in primitive:
            raise ValueError("exact user event must contain content")
        return cls(
            message_id=message_id,
            _message_bytes=canonical_json_bytes(primitive),
            content_sha256=_content_hash(primitive["content"]),
        )

    def to_message(self) -> dict[str, Any]:
        message = json.loads(self._message_bytes)
        if not isinstance(message, dict):  # pragma: no cover - guarded in __post_init__
            raise ValueError("exact user event is not a message object")
        return message


@dataclass(frozen=True)
class LiveUserEventRefV1:
    message_id: int
    content_sha256: str

    def __post_init__(self) -> None:
        _require_integer(self.message_id, "live_user_event_ref.message_id", minimum=1)
        _require_digest(self.content_sha256, "live_user_event_ref.content_sha256")


@dataclass(frozen=True)
class CheckpointSourceV1:
    parent_session_id: str
    lineage_root_session_id: str
    live_user_event_ref: LiveUserEventRefV1
    source_digest: str
    active_message_count: int
    last_active_message_id: int

    def __post_init__(self) -> None:
        _require_nonempty(self.parent_session_id, "parent_session_id")
        _require_nonempty(self.lineage_root_session_id, "lineage_root_session_id")
        _require_digest(self.source_digest, "source_digest")
        _require_integer(self.active_message_count, "active_message_count", minimum=1)
        _require_integer(self.last_active_message_id, "last_active_message_id", minimum=1)

    @classmethod
    def from_event(
        cls,
        *,
        parent_session_id: str,
        lineage_root_session_id: str,
        exact_user_event: ExactUserEventV1,
        source_digest: str,
        active_message_count: int,
        last_active_message_id: int,
    ) -> CheckpointSourceV1:
        return cls(
            parent_session_id=parent_session_id,
            lineage_root_session_id=lineage_root_session_id,
            live_user_event_ref=LiveUserEventRefV1(
                message_id=exact_user_event.message_id,
                content_sha256=exact_user_event.content_sha256,
            ),
            source_digest=source_digest,
            active_message_count=active_message_count,
            last_active_message_id=last_active_message_id,
        )


@dataclass(frozen=True)
class CompilerIdentityV1:
    compiler_version: str
    projector: str
    projection_attempts: int

    def __post_init__(self) -> None:
        _require_nonempty(self.compiler_version, "compiler_version")
        _require_nonempty(self.projector, "projector")
        _require_integer(self.projection_attempts, "projection_attempts", minimum=1)
        if self.projection_attempts > 2:
            raise ValueError("projection_attempts cannot exceed two")


@dataclass(frozen=True)
class EvidenceRecordV1:
    message_id: int
    role: MessageRole
    origin: EvidenceOrigin
    trust_class: TrustClass
    content: str
    effect_disposition: EffectDisposition | None = None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_integer(self.message_id, "evidence.message_id", minimum=1)
        if not isinstance(self.role, MessageRole):
            raise ValueError("evidence.role must be a MessageRole")
        if not isinstance(self.origin, EvidenceOrigin):
            raise ValueError("evidence.origin must be an EvidenceOrigin")
        if not isinstance(self.trust_class, TrustClass):
            raise ValueError("evidence.trust_class must be a TrustClass")
        if not isinstance(self.content, str):
            raise ValueError("evidence.content must be a string")
        if self.content_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.content_sha256
        ):
            raise ValueError("evidence.content_sha256 must be a lowercase SHA-256 digest")
        if self.trust_class is TrustClass.TRUSTED_USER_EVENT and not (
            self.role is MessageRole.USER and self.origin is EvidenceOrigin.DIRECT_USER
        ):
            raise ValueError("trusted user evidence requires direct_user origin and user role")
        if self.trust_class is TrustClass.STRUCTURED_RECEIPT:
            if self.role not in (MessageRole.TOOL, MessageRole.ASSISTANT):
                raise ValueError("structured receipts require tool or assistant role")
            if self.origin not in (EvidenceOrigin.TOOL_RESULT, EvidenceOrigin.ASSISTANT):
                raise ValueError("structured receipts require tool_result or assistant origin")
            if self.effect_disposition not in (
                EffectDisposition.SUCCEEDED,
                EffectDisposition.FAILED,
            ):
                raise ValueError("structured receipts require succeeded or failed disposition")
        elif self.effect_disposition is not None:
            raise ValueError("only structured receipts may carry an effect disposition")


@dataclass(frozen=True)
class DecisionV1:
    decision: str
    rationale: str
    user_confirmed: bool
    citation: CitationV1 | None = None


@dataclass(frozen=True)
class BlockerV1:
    blocker: str
    unblock_condition: str
    owner: str | None = None


@dataclass(frozen=True)
class DependencyV1:
    dependency: str
    state: str
    verification: str | None = None


@dataclass(frozen=True)
class ApprovalV1:
    scope: str
    status: ApprovalStatus
    source_ref: str | None = None
    citation: CitationV1 | None = None


@dataclass(frozen=True)
class ExternalEffectV1:
    effect: str
    disposition: EffectDisposition
    retry_policy: RetryPolicy
    receipt_ref: str | None = None
    recheck_action: str | None = None
    citation: CitationV1 | None = None


@dataclass(frozen=True)
class RuntimeHandleV1:
    kind: RuntimeHandleKind
    id: str
    observed_state: str
    recheck_action: str


@dataclass(frozen=True)
class ArtifactV1:
    ref: str
    kind: str
    observed_state: str | None = None
    verification: str | None = None


@dataclass(frozen=True)
class SafetyConstraintV1:
    id: str
    text: str
    active: bool = True


@dataclass(frozen=True)
class RetryHazardProposalV1:
    operation: str
    risk: str
    safe_recovery: str | None = None


@dataclass(frozen=True)
class RetryHazardV1:
    id: str
    operation: str
    risk: str
    safe_recovery: str | None = None
    active: bool = True


@dataclass(frozen=True)
class RemainingWorkV1:
    item: str
    acceptance_ref: str | None = None


@dataclass(frozen=True)
class NextGateProposalV1:
    action: str
    verification: str
    expected_observation: str
    on_success: str | None = None
    on_failure: str | None = None
    on_unknown: str | None = None
    citation: CitationV1 | None = None


@dataclass(frozen=True)
class NextGateV1:
    action: str
    verification: str
    expected_observation: str
    admitted: bool
    on_success: str | None = None
    on_failure: str | None = None
    on_unknown: str | None = None
    citation: CitationV1 | None = None


@dataclass(frozen=True)
class UncertaintyV1:
    claim: str
    recovery_source: str
    why_uncertain: str | None = None


@dataclass(frozen=True)
class ProjectionV1:
    objective: str
    acceptance_criteria: tuple[str, ...]
    constraints: tuple[str, ...]
    decisions: tuple[DecisionV1, ...]
    blockers: tuple[BlockerV1, ...]
    open_questions: tuple[str, ...]
    dependencies: tuple[DependencyV1, ...]
    approvals: tuple[ApprovalV1, ...]
    external_effects: tuple[ExternalEffectV1, ...]
    runtime_handles: tuple[RuntimeHandleV1, ...]
    artifacts: tuple[ArtifactV1, ...]
    retry_hazards: tuple[RetryHazardProposalV1, ...]
    remaining_work: tuple[RemainingWorkV1, ...]
    next_gate: NextGateProposalV1
    uncertainties: tuple[UncertaintyV1, ...]


@dataclass(frozen=True)
class ContinuationCheckpointV1:
    schema_version: str
    checkpoint_id: str
    source: CheckpointSourceV1
    compiler: CompilerIdentityV1
    activation_mode: ActivationMode
    objective: str
    acceptance_criteria: tuple[str, ...]
    constraints: tuple[SafetyConstraintV1, ...]
    decisions: tuple[DecisionV1, ...]
    blockers: tuple[BlockerV1, ...]
    open_questions: tuple[str, ...]
    dependencies: tuple[DependencyV1, ...]
    approvals: tuple[ApprovalV1, ...]
    external_effects: tuple[ExternalEffectV1, ...]
    runtime_handles: tuple[RuntimeHandleV1, ...]
    artifacts: tuple[ArtifactV1, ...]
    retry_hazards: tuple[RetryHazardV1, ...]
    remaining_work: tuple[RemainingWorkV1, ...]
    next_gate: NextGateV1
    uncertainties: tuple[UncertaintyV1, ...]
    warnings: tuple[CheckpointWarningV1, ...]

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(self)
        if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
            raise TypeError("checkpoint did not serialize to an object")
        return value

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def semantic_bytes(self) -> bytes:
        serialized = self.to_dict()
        semantic = {key: serialized[key] for key in PROJECTION_FIELDS}
        return canonical_json_bytes(semantic)


@dataclass(frozen=True)
class CheckpointBuildResultV1:
    checkpoint: ContinuationCheckpointV1 | None
    warnings: tuple[CheckpointWarningV1, ...]
    issues: tuple[ValidationIssueV1, ...]

    @property
    def renderable(self) -> bool:
        return self.checkpoint is not None and not self.issues


class _ProposalError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        super().__init__(message)
        self.issue = ValidationIssueV1(code=code, path=path, message=message)


_EnumT = TypeVar("_EnumT", bound=Enum)


def _expect_object(
    value: Any,
    path: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    host_fields: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProposalError("invalid_type", path, "expected an object")
    non_string = next((key for key in value if not isinstance(key, str)), None)
    if non_string is not None:
        raise _ProposalError("unknown_field", path, "object keys must be strings")
    missing = sorted(required - value.keys())
    if missing:
        raise _ProposalError("missing_field", f"{path}.{missing[0]}", "required field is missing")
    unknown = sorted(value.keys() - allowed)
    if unknown:
        key = unknown[0]
        code = "host_field" if key in host_fields else "unknown_field"
        raise _ProposalError(code, f"{path}.{key}", "field is not accepted from the projector")
    return value


def _expect_string(
    value: Any,
    path: str,
    *,
    required: bool = True,
    maximum: int = 1000,
) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise _ProposalError("invalid_type", path, "expected a string")
    if required and not value.strip():
        raise _ProposalError("invalid_value", path, "string cannot be empty")
    if len(value) > maximum:
        raise _ProposalError("item_too_large", path, f"string exceeds {maximum} characters")
    return value


def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _ProposalError("invalid_type", path, "expected a boolean")
    return value


def _expect_array(value: Any, path: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise _ProposalError("invalid_type", path, "expected an array")
    if len(value) > maximum:
        raise _ProposalError("too_many_items", path, f"array exceeds {maximum} items")
    return value


def _expect_enum(
    value: Any,
    path: str,
    enum_type: type[_EnumT],
    *,
    allowed: set[str] | None = None,
) -> _EnumT:
    if not isinstance(value, str):
        raise _ProposalError("invalid_type", path, "expected an enum string")
    if allowed is not None and value not in allowed:
        raise _ProposalError("unknown_enum", path, f"unknown enum value: {value}")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _ProposalError("unknown_enum", path, f"unknown enum value: {value}") from exc


def _parse_citation(value: Any, path: str) -> CitationV1 | None:
    if value is None:
        return None
    obj = _expect_object(
        value,
        path,
        allowed=frozenset({"message_id", "quote"}),
        required=frozenset({"message_id", "quote"}),
    )
    message_id = obj["message_id"]
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id < 1:
        raise _ProposalError("invalid_type", f"{path}.message_id", "expected a positive integer")
    quote = _expect_string(obj["quote"], f"{path}.quote", maximum=400)
    assert quote is not None
    if len(quote.strip()) < 3:
        raise _ProposalError("invalid_value", f"{path}.quote", "quote must contain at least 3 characters")
    return CitationV1(message_id=message_id, quote=quote)


def _parse_string_array(value: Any, path: str, maximum: int) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_expect_array(value, path, maximum)):
        parsed = _expect_string(item, f"{path}[{index}]")
        assert parsed is not None
        result.append(parsed)
    return tuple(result)


def _parse_projection(proposal: Mapping[str, Any]) -> ProjectionV1:
    obj = _expect_object(
        proposal,
        "$",
        allowed=PROJECTION_FIELDS,
        required=PROJECTION_FIELDS,
        host_fields=HOST_FIELDS,
    )

    objective = _expect_string(obj["objective"], "$.objective", maximum=2000)
    assert objective is not None

    decisions: list[DecisionV1] = []
    for index, value in enumerate(_expect_array(obj["decisions"], "$.decisions", 32)):
        path = f"$.decisions[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"decision", "rationale", "user_confirmed", "citation"}),
            required=frozenset({"decision", "rationale", "user_confirmed"}),
        )
        decision = _expect_string(item["decision"], f"{path}.decision")
        rationale = _expect_string(item["rationale"], f"{path}.rationale")
        assert decision is not None and rationale is not None
        decisions.append(
            DecisionV1(
                decision=decision,
                rationale=rationale,
                user_confirmed=_expect_bool(item["user_confirmed"], f"{path}.user_confirmed"),
                citation=_parse_citation(item.get("citation"), f"{path}.citation"),
            )
        )

    blockers: list[BlockerV1] = []
    for index, value in enumerate(_expect_array(obj["blockers"], "$.blockers", 16)):
        path = f"$.blockers[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"blocker", "owner", "unblock_condition"}),
            required=frozenset({"blocker", "unblock_condition"}),
        )
        blocker = _expect_string(item["blocker"], f"{path}.blocker")
        condition = _expect_string(item["unblock_condition"], f"{path}.unblock_condition")
        assert blocker is not None and condition is not None
        blockers.append(
            BlockerV1(
                blocker=blocker,
                unblock_condition=condition,
                owner=_expect_string(item.get("owner"), f"{path}.owner", required=False),
            )
        )

    dependencies: list[DependencyV1] = []
    for index, value in enumerate(_expect_array(obj["dependencies"], "$.dependencies", 32)):
        path = f"$.dependencies[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"dependency", "state", "verification"}),
            required=frozenset({"dependency", "state"}),
        )
        dependency = _expect_string(item["dependency"], f"{path}.dependency")
        state = _expect_string(item["state"], f"{path}.state")
        assert dependency is not None and state is not None
        dependencies.append(
            DependencyV1(
                dependency=dependency,
                state=state,
                verification=_expect_string(
                    item.get("verification"), f"{path}.verification", required=False
                ),
            )
        )

    approvals: list[ApprovalV1] = []
    projector_approval_statuses = {
        ApprovalStatus.REQUESTED.value,
        ApprovalStatus.APPROVED.value,
        ApprovalStatus.DENIED.value,
    }
    for index, value in enumerate(_expect_array(obj["approvals"], "$.approvals", 16)):
        path = f"$.approvals[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"scope", "status", "source_ref", "citation"}),
            required=frozenset({"scope", "status"}),
        )
        scope = _expect_string(item["scope"], f"{path}.scope")
        assert scope is not None
        approvals.append(
            ApprovalV1(
                scope=scope,
                status=_expect_enum(
                    item["status"],
                    f"{path}.status",
                    ApprovalStatus,
                    allowed=projector_approval_statuses,
                ),
                source_ref=_expect_string(
                    item.get("source_ref"), f"{path}.source_ref", required=False
                ),
                citation=_parse_citation(item.get("citation"), f"{path}.citation"),
            )
        )

    effects: list[ExternalEffectV1] = []
    for index, value in enumerate(
        _expect_array(obj["external_effects"], "$.external_effects", 64)
    ):
        path = f"$.external_effects[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset(
                {
                    "effect",
                    "disposition",
                    "retry_policy",
                    "receipt_ref",
                    "recheck_action",
                    "citation",
                }
            ),
            required=frozenset({"effect", "disposition", "retry_policy"}),
        )
        effect = _expect_string(item["effect"], f"{path}.effect")
        assert effect is not None
        disposition = _expect_enum(
            item["disposition"], f"{path}.disposition", EffectDisposition
        )
        retry_policy = _expect_enum(item["retry_policy"], f"{path}.retry_policy", RetryPolicy)
        recheck_action = _expect_string(
            item.get("recheck_action"), f"{path}.recheck_action", required=False
        )
        if disposition is EffectDisposition.ATTEMPTED_UNKNOWN and (
            retry_policy is not RetryPolicy.VERIFY_FIRST or not recheck_action
        ):
            raise _ProposalError(
                "invalid_cross_field",
                path,
                "attempted_unknown requires verify_first and recheck_action",
            )
        effects.append(
            ExternalEffectV1(
                effect=effect,
                disposition=disposition,
                retry_policy=retry_policy,
                receipt_ref=_expect_string(
                    item.get("receipt_ref"), f"{path}.receipt_ref", required=False
                ),
                recheck_action=recheck_action,
                citation=_parse_citation(item.get("citation"), f"{path}.citation"),
            )
        )

    handles: list[RuntimeHandleV1] = []
    for index, value in enumerate(
        _expect_array(obj["runtime_handles"], "$.runtime_handles", 64)
    ):
        path = f"$.runtime_handles[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"kind", "id", "observed_state", "recheck_action"}),
            required=frozenset({"kind", "id", "observed_state", "recheck_action"}),
        )
        identifier = _expect_string(item["id"], f"{path}.id")
        observed = _expect_string(item["observed_state"], f"{path}.observed_state")
        recheck = _expect_string(item["recheck_action"], f"{path}.recheck_action")
        assert identifier is not None and observed is not None and recheck is not None
        handles.append(
            RuntimeHandleV1(
                kind=_expect_enum(item["kind"], f"{path}.kind", RuntimeHandleKind),
                id=identifier,
                observed_state=observed,
                recheck_action=recheck,
            )
        )

    artifacts: list[ArtifactV1] = []
    for index, value in enumerate(_expect_array(obj["artifacts"], "$.artifacts", 64)):
        path = f"$.artifacts[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"ref", "kind", "observed_state", "verification"}),
            required=frozenset({"ref", "kind"}),
        )
        reference = _expect_string(item["ref"], f"{path}.ref")
        kind = _expect_string(item["kind"], f"{path}.kind")
        assert reference is not None and kind is not None
        artifacts.append(
            ArtifactV1(
                ref=reference,
                kind=kind,
                observed_state=_expect_string(
                    item.get("observed_state"), f"{path}.observed_state", required=False
                ),
                verification=_expect_string(
                    item.get("verification"), f"{path}.verification", required=False
                ),
            )
        )

    hazards: list[RetryHazardProposalV1] = []
    for index, value in enumerate(
        _expect_array(obj["retry_hazards"], "$.retry_hazards", MAX_SAFETY_ITEMS)
    ):
        path = f"$.retry_hazards[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"operation", "risk", "safe_recovery"}),
            required=frozenset({"operation", "risk"}),
        )
        operation = _expect_string(item["operation"], f"{path}.operation")
        risk = _expect_string(item["risk"], f"{path}.risk")
        assert operation is not None and risk is not None
        hazards.append(
            RetryHazardProposalV1(
                operation=operation,
                risk=risk,
                safe_recovery=_expect_string(
                    item.get("safe_recovery"), f"{path}.safe_recovery", required=False
                ),
            )
        )

    remaining: list[RemainingWorkV1] = []
    for index, value in enumerate(
        _expect_array(obj["remaining_work"], "$.remaining_work", 32)
    ):
        path = f"$.remaining_work[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"item", "acceptance_ref"}),
            required=frozenset({"item"}),
        )
        work_item = _expect_string(item["item"], f"{path}.item")
        assert work_item is not None
        remaining.append(
            RemainingWorkV1(
                item=work_item,
                acceptance_ref=_expect_string(
                    item.get("acceptance_ref"), f"{path}.acceptance_ref", required=False
                ),
            )
        )

    gate_obj = _expect_object(
        obj["next_gate"],
        "$.next_gate",
        allowed=frozenset(
            {
                "action",
                "verification",
                "expected_observation",
                "on_success",
                "on_failure",
                "on_unknown",
                "citation",
            }
        ),
        required=frozenset({"action", "verification", "expected_observation"}),
    )
    gate_action = _expect_string(gate_obj["action"], "$.next_gate.action")
    gate_verification = _expect_string(gate_obj["verification"], "$.next_gate.verification")
    gate_expected = _expect_string(
        gate_obj["expected_observation"], "$.next_gate.expected_observation"
    )
    assert gate_action is not None and gate_verification is not None and gate_expected is not None
    next_gate = NextGateProposalV1(
        action=gate_action,
        verification=gate_verification,
        expected_observation=gate_expected,
        on_success=_expect_string(
            gate_obj.get("on_success"), "$.next_gate.on_success", required=False
        ),
        on_failure=_expect_string(
            gate_obj.get("on_failure"), "$.next_gate.on_failure", required=False
        ),
        on_unknown=_expect_string(
            gate_obj.get("on_unknown"), "$.next_gate.on_unknown", required=False
        ),
        citation=_parse_citation(gate_obj.get("citation"), "$.next_gate.citation"),
    )

    uncertainties: list[UncertaintyV1] = []
    for index, value in enumerate(
        _expect_array(obj["uncertainties"], "$.uncertainties", 32)
    ):
        path = f"$.uncertainties[{index}]"
        item = _expect_object(
            value,
            path,
            allowed=frozenset({"claim", "why_uncertain", "recovery_source"}),
            required=frozenset({"claim", "recovery_source"}),
        )
        claim = _expect_string(item["claim"], f"{path}.claim")
        recovery = _expect_string(item["recovery_source"], f"{path}.recovery_source")
        assert claim is not None and recovery is not None
        uncertainties.append(
            UncertaintyV1(
                claim=claim,
                recovery_source=recovery,
                why_uncertain=_expect_string(
                    item.get("why_uncertain"), f"{path}.why_uncertain", required=False
                ),
            )
        )

    return ProjectionV1(
        objective=objective,
        acceptance_criteria=_parse_string_array(
            obj["acceptance_criteria"], "$.acceptance_criteria", 16
        ),
        constraints=_parse_string_array(obj["constraints"], "$.constraints", MAX_SAFETY_ITEMS),
        decisions=tuple(decisions),
        blockers=tuple(blockers),
        open_questions=_parse_string_array(obj["open_questions"], "$.open_questions", 16),
        dependencies=tuple(dependencies),
        approvals=tuple(approvals),
        external_effects=tuple(effects),
        runtime_handles=tuple(handles),
        artifacts=tuple(artifacts),
        retry_hazards=tuple(hazards),
        remaining_work=tuple(remaining),
        next_gate=next_gate,
        uncertainties=tuple(uncertainties),
    )


def scan_secrets(value: Any) -> tuple[str, ...]:
    try:
        text = canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError):
        return ()
    return tuple(pattern.pattern for pattern in _SECRET_PATTERNS if pattern.search(text))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _constraint_id(text: str) -> str:
    return "constraint_" + _sha256(_normalize_text(text).encode("utf-8"))[:24]


def _hazard_id(hazard: RetryHazardProposalV1 | RetryHazardV1) -> str:
    normalized = {
        "operation": _normalize_text(hazard.operation),
        "risk": _normalize_text(hazard.risk),
        "safe_recovery": _normalize_text(hazard.safe_recovery or ""),
    }
    return "hazard_" + _sha256(canonical_json_bytes(normalized))[:24]


def _normalize_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _citation_covers_record(citation: CitationV1 | None, evidence: EvidenceRecordV1) -> bool:
    if citation is None or citation.message_id != evidence.message_id:
        return False
    return _normalize_quote(citation.quote) == _normalize_quote(evidence.content)


def _recovery_pointer(source: CheckpointSourceV1) -> str:
    return f"parent session {source.parent_session_id}"


def _authority_uncertainty(
    *, claim: str, reason: str, source: CheckpointSourceV1
) -> UncertaintyV1:
    return UncertaintyV1(
        claim=claim[:1000],
        why_uncertain=f"host evidence check failed: {reason}"[:1000],
        recovery_source=_recovery_pointer(source),
    )


def _warning(
    code: CheckpointWarningCode,
    message: str,
    source: CheckpointSourceV1,
    *,
    severity: WarningSeverity = WarningSeverity.WARNING,
) -> CheckpointWarningV1:
    return CheckpointWarningV1(
        code=code,
        message=message,
        recovery_pointer=_recovery_pointer(source),
        severity=severity,
    )


def _ground_projection(
    projection: ProjectionV1,
    evidence: Sequence[EvidenceRecordV1],
    source: CheckpointSourceV1,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[DecisionV1, ...],
    tuple[ApprovalV1, ...],
    tuple[ExternalEffectV1, ...],
    NextGateV1,
    tuple[UncertaintyV1, ...],
    tuple[CheckpointWarningV1, ...],
]:
    index: dict[int, EvidenceRecordV1] = {}
    for record in evidence:
        if record.message_id in index:
            raise _ProposalError(
                "duplicate_evidence_id",
                "$.evidence",
                f"duplicate evidence message_id {record.message_id}",
            )
        index[record.message_id] = record

    warnings: list[CheckpointWarningV1] = []
    uncertainties = list(projection.uncertainties)

    def direct_user_record(
        citation: CitationV1 | None,
    ) -> tuple[EvidenceRecordV1 | None, str | None]:
        if citation is None:
            return None, "no citation was supplied"
        record = index.get(citation.message_id)
        if record is None:
            return None, "cited message was not present in host evidence"
        if not (
            record.role is MessageRole.USER
            and record.origin is EvidenceOrigin.DIRECT_USER
            and record.trust_class is TrustClass.TRUSTED_USER_EVENT
        ):
            return None, "citation did not target structurally trusted direct-user evidence"
        if not _citation_covers_record(citation, record):
            return record, "citation did not cover the complete trusted direct-user event"
        return record, None

    live_record = index.get(source.live_user_event_ref.message_id)
    if live_record is None or not (
        live_record.role is MessageRole.USER
        and live_record.origin is EvidenceOrigin.DIRECT_USER
        and live_record.trust_class is TrustClass.TRUSTED_USER_EVENT
    ):
        raise _ProposalError(
            "live_user_event_missing",
            "$.evidence",
            "host evidence did not contain the trusted live direct-user event",
        )
    live_content_sha256 = live_record.content_sha256 or _content_hash(live_record.content)
    if live_content_sha256 != source.live_user_event_ref.content_sha256:
        raise _ProposalError(
            "live_user_event_mismatch",
            "$.evidence",
            "trusted live-user evidence bytes did not match the host source reference",
        )
    objective = live_record.content
    acceptance_criteria = (
        "Satisfy the exact direct-user event without widening its authority.",
    )

    decisions: list[DecisionV1] = []
    for item in projection.decisions:
        if not item.user_confirmed:
            decisions.append(item)
            continue
        record, reason = direct_user_record(item.citation)
        if reason is None:
            assert record is not None
            item = replace(
                item,
                decision=record.content,
                citation=CitationV1(message_id=record.message_id, quote=record.content),
            )
            reason = "preview does not admit newly projected user-confirmed state"
        decisions.append(
            replace(
                item,
                decision=(record.content if record is not None else "Unverified projected decision."),
                user_confirmed=False,
            )
        )
        warnings.append(
            _warning(
                CheckpointWarningCode.AUTHORITY_DEMOTED,
                "A projected user-confirmed decision was demoted.",
                source,
            )
        )
        uncertainties.append(
            _authority_uncertainty(
                claim=f"User confirmation of decision: {item.decision}",
                reason=reason,
                source=source,
            )
        )

    approvals: list[ApprovalV1] = []
    for item in projection.approvals:
        if item.status is not ApprovalStatus.APPROVED:
            approvals.append(item)
            continue
        record, reason = direct_user_record(item.citation)
        if reason is None:
            assert record is not None
            item = replace(
                item,
                scope=record.content,
                source_ref=f"message:{record.message_id}",
                citation=CitationV1(message_id=record.message_id, quote=record.content),
            )
            reason = "preview does not admit newly projected approval state"
        approvals.append(
            replace(
                item,
                scope=(
                    record.content
                    if record is not None
                    else "Unverified projected approval scope."
                ),
                status=ApprovalStatus.UNVERIFIED_BY_HOST,
            )
        )
        warnings.append(
            _warning(
                CheckpointWarningCode.AUTHORITY_DEMOTED,
                "A projected approval was demoted to unverified_by_host.",
                source,
            )
        )
        uncertainties.append(
            _authority_uncertainty(
                claim=f"Approval status of: {item.scope}", reason=reason, source=source
            )
        )

    effects: list[ExternalEffectV1] = []
    for item in projection.external_effects:
        if item.disposition not in (EffectDisposition.SUCCEEDED, EffectDisposition.FAILED):
            effects.append(item)
            continue
        citation = item.citation
        record = index.get(citation.message_id) if citation is not None else None
        reason: str | None = None
        if citation is None:
            reason = "no structured receipt citation was supplied"
        elif record is None:
            reason = "cited receipt was not present in host evidence"
        elif record.trust_class is not TrustClass.STRUCTURED_RECEIPT:
            reason = "cited output was free-form rather than a structured receipt"
        elif not _citation_covers_record(citation, record):
            reason = "citation did not cover the complete structured receipt"
        if reason is None:
            assert record is not None and record.effect_disposition is not None
            receipt_disposition = record.effect_disposition
            effects.append(
                replace(
                    item,
                    effect=record.content,
                    disposition=receipt_disposition,
                    retry_policy=(
                        RetryPolicy.DO_NOT_RETRY
                        if receipt_disposition is EffectDisposition.SUCCEEDED
                        else RetryPolicy.VERIFY_FIRST
                    ),
                    receipt_ref=f"message:{record.message_id}",
                    recheck_action=(
                        None
                        if receipt_disposition is EffectDisposition.SUCCEEDED
                        else f"Verify the receipt in {source.parent_session_id}."
                    ),
                    citation=CitationV1(message_id=record.message_id, quote=record.content),
                )
            )
            continue
        recheck = item.recheck_action or (
            f"Verify the effect in {source.parent_session_id} or against the live system."
        )
        effects.append(
            replace(
                item,
                disposition=EffectDisposition.ATTEMPTED_UNKNOWN,
                retry_policy=RetryPolicy.VERIFY_FIRST,
                recheck_action=recheck,
            )
        )
        warnings.append(
            _warning(
                CheckpointWarningCode.DISPOSITION_DEMOTED,
                "A projected succeeded/failed disposition was demoted to attempted_unknown.",
                source,
            )
        )
        uncertainties.append(
            _authority_uncertainty(
                claim=f"Outcome of effect: {item.effect}", reason=reason, source=source
            )
        )

    gate = NextGateV1(
        action=live_record.content,
        verification=f"Follow only the exact direct-user event in message {live_record.message_id}.",
        expected_observation="The exact direct-user request is addressed without widened authority.",
        admitted=True,
        citation=CitationV1(message_id=live_record.message_id, quote=live_record.content),
    )

    if projection.constraints:
        warnings.append(
            _warning(
                CheckpointWarningCode.PROJECTED_SAFETY_IGNORED,
                "New projector-authored active constraints were not admitted.",
                source,
            )
        )
        uncertainties.append(
            _authority_uncertainty(
                claim="Projector-authored active constraints",
                reason="new active constraints require host-owned prior state or structural binding",
                source=source,
            )
        )
    if projection.retry_hazards:
        warnings.append(
            _warning(
                CheckpointWarningCode.PROJECTED_SAFETY_IGNORED,
                "New projector-authored active retry hazards were not admitted.",
                source,
            )
        )
        uncertainties.append(
            _authority_uncertainty(
                claim="Projector-authored active retry hazards",
                reason="new active retry hazards require host-owned prior state or structural binding",
                source=source,
            )
        )

    if len(uncertainties) > 32:
        raise _ProposalError(
            "too_many_items",
            "$.uncertainties",
            "grounding demotions exceed the 32-item uncertainty bound",
        )

    return (
        objective,
        acceptance_criteria,
        tuple(decisions),
        tuple(approvals),
        tuple(effects),
        gate,
        tuple(uncertainties),
        tuple(warnings),
    )


_CHECKPOINT_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "source",
        "compiler",
        "activation_mode",
        *PROJECTION_FIELDS,
        "warnings",
    }
)


def _parse_prior_safety_envelope(
    envelope: str,
    source: CheckpointSourceV1,
) -> tuple[tuple[SafetyConstraintV1, ...], tuple[RetryHazardV1, ...]]:
    prefix = ENVELOPE_PREFIX + "\n\n"
    if not isinstance(envelope, str) or not envelope.startswith(prefix):
        raise ValueError("prior safety state did not come from a host checkpoint envelope")
    try:
        payload = json.loads(envelope[len(prefix) :])
    except (TypeError, ValueError) as exc:
        raise ValueError("prior checkpoint envelope contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("prior checkpoint envelope payload must be an object")
    if set(payload) != _CHECKPOINT_TOP_FIELDS:
        raise ValueError("prior checkpoint envelope has an unexpected shape")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prior checkpoint envelope has an unsupported schema")
    if payload.get("activation_mode") != ActivationMode.PAUSED_MANUAL.value:
        raise ValueError("prior checkpoint envelope has an unsupported activation mode")
    checkpoint_id = payload.get("checkpoint_id")
    body = dict(payload)
    body.pop("checkpoint_id", None)
    expected_id = "ccv1_" + _sha256(canonical_json_bytes(body))
    if checkpoint_id != expected_id:
        raise ValueError("prior checkpoint identity is not canonical")
    prior_source = payload.get("source")
    if not isinstance(prior_source, dict):
        raise ValueError("prior checkpoint source is malformed")
    if prior_source.get("lineage_root_session_id") != source.lineage_root_session_id:
        raise ValueError("prior checkpoint is from a different lineage")
    semantic = {key: payload[key] for key in PROJECTION_FIELDS}
    if len(canonical_json_bytes(semantic)) > SEMANTIC_MAX_BYTES:
        raise ValueError("prior checkpoint exceeds the semantic byte bound")
    if scan_secrets(payload):
        raise ValueError("prior checkpoint contains secret-like material")

    constraints_value = payload.get("constraints")
    if not isinstance(constraints_value, list) or len(constraints_value) > MAX_SAFETY_ITEMS:
        raise ValueError("prior checkpoint constraints are malformed")
    constraints: list[SafetyConstraintV1] = []
    seen_constraint_ids: set[str] = set()
    for item in constraints_value:
        if not isinstance(item, dict) or set(item) != {"id", "text", "active"}:
            raise ValueError("prior checkpoint constraint is malformed")
        if not isinstance(item["text"], str) or not isinstance(item["active"], bool):
            raise ValueError("prior checkpoint constraint fields are malformed")
        if item["id"] != _constraint_id(item["text"]):
            raise ValueError("prior checkpoint constraint stable ID is invalid")
        if item["id"] in seen_constraint_ids:
            raise ValueError("prior checkpoint contains duplicate constraint IDs")
        seen_constraint_ids.add(item["id"])
        if item["active"]:
            constraints.append(
                SafetyConstraintV1(id=item["id"], text=item["text"], active=True)
            )

    hazards_value = payload.get("retry_hazards")
    if not isinstance(hazards_value, list) or len(hazards_value) > MAX_SAFETY_ITEMS:
        raise ValueError("prior checkpoint retry hazards are malformed")
    hazards: list[RetryHazardV1] = []
    seen_hazard_ids: set[str] = set()
    for item in hazards_value:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "operation",
            "risk",
            "safe_recovery",
            "active",
        }:
            raise ValueError("prior checkpoint retry hazard is malformed")
        if not isinstance(item["operation"], str) or not isinstance(item["risk"], str):
            raise ValueError("prior checkpoint retry hazard fields are malformed")
        if item["safe_recovery"] is not None and not isinstance(item["safe_recovery"], str):
            raise ValueError("prior checkpoint retry hazard recovery is malformed")
        if not isinstance(item["active"], bool):
            raise ValueError("prior checkpoint retry hazard active flag is malformed")
        hazard = RetryHazardV1(
            id=str(item["id"]),
            operation=item["operation"],
            risk=item["risk"],
            safe_recovery=item["safe_recovery"],
            active=True,
        )
        if hazard.id != _hazard_id(hazard):
            raise ValueError("prior checkpoint retry hazard stable ID is invalid")
        if hazard.id in seen_hazard_ids:
            raise ValueError("prior checkpoint contains duplicate retry hazard IDs")
        seen_hazard_ids.add(hazard.id)
        if item["active"]:
            hazards.append(hazard)

    return tuple(constraints), tuple(hazards)


def _merge_safety_state(
    prior_constraints: Sequence[SafetyConstraintV1],
    prior_hazards: Sequence[RetryHazardV1],
) -> tuple[tuple[SafetyConstraintV1, ...], tuple[RetryHazardV1, ...]]:
    constraints = {item.id: item for item in prior_constraints if item.active}
    hazards = {item.id: item for item in prior_hazards if item.active}
    return tuple(constraints.values()), tuple(hazards.values())


def _checkpoint_id(checkpoint: ContinuationCheckpointV1) -> str:
    body = checkpoint.to_dict()
    body.pop("checkpoint_id")
    return "ccv1_" + _sha256(canonical_json_bytes(body))


def _failure(
    issue: ValidationIssueV1,
    warning_code: CheckpointWarningCode,
    source: CheckpointSourceV1,
    *,
    message: str,
) -> CheckpointBuildResultV1:
    return CheckpointBuildResultV1(
        checkpoint=None,
        warnings=(
            _warning(
                warning_code,
                message,
                source,
                severity=WarningSeverity.ERROR,
            ),
        ),
        issues=(issue,),
    )


def assemble_checkpoint(
    proposal: Mapping[str, Any],
    *,
    source: CheckpointSourceV1,
    compiler: CompilerIdentityV1,
    evidence: Sequence[EvidenceRecordV1],
    prior_checkpoint_envelope: str | None = None,
) -> CheckpointBuildResultV1:
    """Validate and assemble one non-mutating checkpoint, failing closed."""

    try:
        proposal_bytes = canonical_json_bytes(proposal)
    except (TypeError, ValueError) as exc:
        issue = ValidationIssueV1("malformed_json", "$", str(exc))
        return _failure(
            issue,
            CheckpointWarningCode.MALFORMED_PROPOSAL,
            source,
            message="The projector proposal was not canonical JSON.",
        )

    secret_matches = tuple(
        pattern.pattern
        for pattern in _SECRET_PATTERNS
        if pattern.search(proposal_bytes.decode("utf-8"))
    )
    if secret_matches:
        issue = ValidationIssueV1(
            "secret_detected", "$", "secret-like material was present in projector output"
        )
        return _failure(
            issue,
            CheckpointWarningCode.SECRET_DETECTED,
            source,
            message="Secret-like projector output was rejected.",
        )

    try:
        projection = _parse_projection(proposal)
        grounded = _ground_projection(projection, evidence, source)
    except _ProposalError as exc:
        return _failure(
            exc.issue,
            CheckpointWarningCode.MALFORMED_PROPOSAL,
            source,
            message="The projector proposal failed contract validation.",
        )

    prior_constraints: tuple[SafetyConstraintV1, ...] = ()
    prior_hazards: tuple[RetryHazardV1, ...] = ()
    if prior_checkpoint_envelope is not None:
        try:
            prior_constraints, prior_hazards = _parse_prior_safety_envelope(
                prior_checkpoint_envelope, source
            )
        except (TypeError, ValueError) as exc:
            issue = ValidationIssueV1("prior_checkpoint_invalid", "$.prior", str(exc))
            return _failure(
                issue,
                CheckpointWarningCode.PRIOR_CHECKPOINT_INVALID,
                source,
                message="Prior checkpoint safety state could not be validated.",
            )

    constraints, hazards = _merge_safety_state(prior_constraints, prior_hazards)
    if len(constraints) > MAX_SAFETY_ITEMS or len(hazards) > MAX_SAFETY_ITEMS:
        issue = ValidationIssueV1(
            "safety_overflow",
            "$.constraints" if len(constraints) > MAX_SAFETY_ITEMS else "$.retry_hazards",
            "active safety state exceeds its 32-item bound and was not truncated",
        )
        return _failure(
            issue,
            CheckpointWarningCode.SAFETY_OVERFLOW,
            source,
            message="Active constraint or retry-hazard state overflowed; rendering is disabled.",
        )

    (
        objective,
        acceptance_criteria,
        decisions,
        approvals,
        effects,
        next_gate,
        uncertainties,
        warnings,
    ) = grounded
    checkpoint = ContinuationCheckpointV1(
        schema_version=SCHEMA_VERSION,
        checkpoint_id="",
        source=source,
        compiler=compiler,
        activation_mode=ActivationMode.PAUSED_MANUAL,
        objective=objective,
        acceptance_criteria=acceptance_criteria,
        constraints=constraints,
        decisions=decisions,
        blockers=projection.blockers,
        open_questions=projection.open_questions,
        dependencies=projection.dependencies,
        approvals=approvals,
        external_effects=effects,
        runtime_handles=projection.runtime_handles,
        artifacts=projection.artifacts,
        retry_hazards=hazards,
        remaining_work=projection.remaining_work,
        next_gate=next_gate,
        uncertainties=uncertainties,
        warnings=warnings,
    )
    checkpoint = replace(checkpoint, checkpoint_id=_checkpoint_id(checkpoint))

    semantic_size = len(checkpoint.semantic_bytes())
    if semantic_size > SEMANTIC_MAX_BYTES:
        issue = ValidationIssueV1(
            "checkpoint_too_large",
            "$",
            f"canonical semantic checkpoint is {semantic_size} bytes; limit is {SEMANTIC_MAX_BYTES}",
        )
        return _failure(
            issue,
            CheckpointWarningCode.CHECKPOINT_TOO_LARGE,
            source,
            message="The canonical semantic checkpoint exceeded 24 KiB; rendering is disabled.",
        )

    if scan_secrets(checkpoint.to_dict()):
        issue = ValidationIssueV1(
            "secret_detected", "$", "secret-like material survived checkpoint assembly"
        )
        return _failure(
            issue,
            CheckpointWarningCode.SECRET_DETECTED,
            source,
            message="Secret-like material survived assembly; rendering is disabled.",
        )

    return CheckpointBuildResultV1(
        checkpoint=checkpoint,
        warnings=warnings,
        issues=(),
    )


def render_checkpoint_messages(
    checkpoint: ContinuationCheckpointV1,
    exact_user_event: ExactUserEventV1,
) -> list[dict[str, Any]]:
    """Render the canonical user/assistant/user/assistant paused bootstrap."""

    reference = checkpoint.source.live_user_event_ref
    if (
        reference.message_id != exact_user_event.message_id
        or reference.content_sha256 != exact_user_event.content_sha256
    ):
        raise ValueError("exact user event does not match checkpoint source")
    if checkpoint.activation_mode is not ActivationMode.PAUSED_MANUAL:
        raise ValueError("only paused_manual checkpoints are renderable")
    action = checkpoint.next_gate.action.strip()
    if not action.endswith((".", "!", "?")):
        action += "."
    return [
        {
            "role": MessageRole.USER.value,
            "content": ENVELOPE_PREFIX
            + "\n\n"
            + checkpoint.canonical_bytes().decode("utf-8"),
        },
        {"role": MessageRole.ASSISTANT.value, "content": BRIDGE_TEXT},
        exact_user_event.to_message(),
        {
            "role": MessageRole.ASSISTANT.value,
            "content": PAUSE_TEXT_TEMPLATE.format(action=action),
        },
    ]


def render_checkpoint_markdown(checkpoint: ContinuationCheckpointV1) -> str:
    """Render a deterministic inspection view with dispositions always inline."""

    def inline(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    lines = [
        f"# Continuation checkpoint {checkpoint.checkpoint_id}",
        "",
        f"**Objective:** {inline(checkpoint.objective)}",
        "",
    ]

    def section(title: str, values: Sequence[str]) -> None:
        if not values:
            return
        lines.append(f"## {title}")
        lines.extend(f"- {inline(value)}" for value in values)
        lines.append("")

    section("Acceptance criteria", checkpoint.acceptance_criteria)
    section("Constraints", tuple(f"[{item.id}] {item.text}" for item in checkpoint.constraints))
    section(
        "Decisions",
        tuple(
            f"{item.decision} — {item.rationale}"
            + (" [user_confirmed]" if item.user_confirmed else "")
            for item in checkpoint.decisions
        ),
    )
    section(
        "Blockers",
        tuple(f"{item.blocker} — unblock: {item.unblock_condition}" for item in checkpoint.blockers),
    )
    section("Open questions", checkpoint.open_questions)
    section(
        "Dependencies",
        tuple(f"{item.dependency} [{item.state}]" for item in checkpoint.dependencies),
    )
    section(
        "Approvals",
        tuple(f"[{item.status.value}] {item.scope}" for item in checkpoint.approvals),
    )
    section(
        "External effects",
        tuple(
            f"[{item.disposition.value}] {item.effect} (retry: {item.retry_policy.value})"
            for item in checkpoint.external_effects
        ),
    )
    section(
        "Runtime handles",
        tuple(
            f"[{item.kind.value}] {item.id}: {item.observed_state}; recheck: {item.recheck_action}"
            for item in checkpoint.runtime_handles
        ),
    )
    section("Artifacts", tuple(f"{item.ref} ({item.kind})" for item in checkpoint.artifacts))
    section(
        "Retry hazards",
        tuple(f"[{item.id}] {item.operation}: {item.risk}" for item in checkpoint.retry_hazards),
    )
    section("Remaining work", tuple(item.item for item in checkpoint.remaining_work))
    lines.extend(
        [
            "## Next gate",
            f"- **Admitted:** {str(checkpoint.next_gate.admitted).lower()}",
            f"- **Action:** {inline(checkpoint.next_gate.action)}",
            f"- **Verify:** {inline(checkpoint.next_gate.verification)}",
            f"- **Expect:** {inline(checkpoint.next_gate.expected_observation)}",
            "",
        ]
    )
    section(
        "Uncertainties",
        tuple(f"{item.claim} — recover via: {item.recovery_source}" for item in checkpoint.uncertainties),
    )
    return "\n".join(lines)
