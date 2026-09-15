"""Tool result persistence -- preserves large outputs instead of truncating. Layers against
context overflow: (1) per-tool caps inside each tool; (2) ``maybe_persist_tool_result`` —
output over the tool's threshold is persisted and replaced by a preview + path; canonical home
is ALWAYS host-side ``$HERMES_HOME/cache/spillover/{id}.txt`` (works for sessions that never
ran a terminal), remote backends get the translated in-sandbox path (probed for readability)
else a copy in the sandbox temp dir; (3) ``enforce_turn_budget``."""

import hashlib
import logging
import os
import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass

from agent.redact import redact_sensitive_text
from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
SPILLOVER_SUBDIR = "cache/spillover"
SPILLOVER_MAX_AGE_HOURS = 24
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_ARTIFACT_EXTENSION = re.compile(r"(?:\.[A-Za-z0-9_-]+)+")
_MAX_RESULT_FILENAME_STEM = 120
# Artifact paths are quoted verbatim inside the pruned tool-result stub that
# stays in context, so every path character costs tokens for the rest of the
# session. Full sha256/uuid4 hex made a single path 229 chars (~60% of a
# 768-char stub). These prefixes keep the same roles -- scope namespace,
# content digest, rollback nonce -- with collision odds that stay negligible
# for a per-task artifact directory, while cutting the path to ~72 chars.
# Provenance is unaffected: the stub still carries the FULL sha256, and
# nothing parses these path segments back out.
_ARTIFACT_SCOPE_HASH_CHARS = 16
_ARTIFACT_CONTENT_DIGEST_CHARS = 16
_ARTIFACT_ATTEMPT_TOKEN_CHARS = 8


@dataclass(frozen=True)
class PersistedToolArtifact:
    kind: str
    path: str
    chars: int
    sha256: str
    redacted: bool
    created: bool

_spillover_prune_lock = threading.Lock()
_spillover_pruned_homes: set = set()  # profile home keys already swept this process


def get_spillover_dir():
    """Return $HERMES_HOME/cache/spillover as a Path (not created)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / SPILLOVER_SUBDIR


def cleanup_spillover_cache(max_age_hours: int = SPILLOVER_MAX_AGE_HOURS) -> int:
    """Delete spillover files older than *max_age_hours*; returns count removed (same
    contract as the ``cleanup_*_cache`` helpers the gateway housekeeping loop runs hourly)."""
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    try:
        entries = list(get_spillover_dir().iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def _prune_spillover_once() -> None:
    """Best-effort prune, at most once per process PER PROFILE HOME (CLI-only installs never run
    housekeeping; a multiplexed gateway must sweep every profile's ``cache/spillover``, not just the
    first one that spilled)."""
    from hermes_constants import hermes_home_key
    home_key = hermes_home_key()
    with _spillover_prune_lock:
        if home_key in _spillover_pruned_homes:
            return
        _spillover_pruned_homes.add(home_key)
    try:
        if removed := cleanup_spillover_cache():
            logger.debug("Pruned %d expired spillover file(s)", removed)
    except Exception as exc:
        logger.debug("Spillover prune failed: %s", exc)


def _is_host_side_env(env) -> bool:
    """True when this process should write the spill file directly: ``env=None`` (no sandbox
    yet) or the local backend. Remote backends resolve ``read_file`` inside the sandbox."""
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment
        return isinstance(env, LocalEnvironment)
    except Exception:
        return False


def _write_to_spillover(content: str, filename: str):
    """Write host-side to $HERMES_HOME/cache/spillover; returns path str or None."""
    try:
        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / filename
        path.write_text(content, encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Spillover write failed for %s: %s", filename, exc)
        return None
    _prune_spillover_once()
    return str(path)


def _sandbox_visible_spillover_path(host_path: str, env) -> str | None:
    """Path where a remote backend can read *host_path*, or None. Translates via the image
    tools' helper, forces a sync for synced backends, then PROBES readability — a persistent
    container created before spillover joined the mount list lacks the bind mount and must
    fall back to the in-sandbox write."""
    try:
        from tools.credential_files import to_agent_visible_cache_path
        visible = to_agent_visible_cache_path(host_path)
    except Exception as exc:
        logger.debug("Spillover path translation failed: %s", exc)
        return None
    try:
        if (sync_manager := getattr(env, "_sync_manager", None)) is not None:
            sync_manager.sync(force=True)
    except Exception as exc:
        logger.debug("Spillover sync failed: %s", exc)
    try:
        if env.execute(f"test -r {shlex.quote(visible)}", timeout=15).get("returncode", 1) == 0:
            return visible
    except Exception as exc:
        logger.debug("Spillover readability probe failed: %s", exc)
    return None


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    get_temp_dir = getattr(env, "get_temp_dir", None)
    temp_dir = None
    if callable(get_temp_dir):
        try:
            temp_dir = get_temp_dir()
        except Exception as exc:
            logger.debug("Could not resolve env temp dir: %s", exc)
    return f"{temp_dir.rstrip('/') or '/'}/hermes-results" if temp_dir else STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id
    safe_stem = safe_stem or "tool_result"
    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"
    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    last_nl = content.rfind("\n", 0, max_chars)
    return content[:last_nl + 1 if last_nl > max_chars // 2 else max_chars], True


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(); True on success. Content goes through
    stdin, not the command string: Linux ``MAX_ARG_STRLEN`` caps one argv element at 128 KB,
    so a heredoc-in-command silently failed for exactly the oversized results this handles."""
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    return env.execute(cmd, timeout=30, stdin_data=content).get("returncode", 1) == 0


def _publish_immutable_to_sandbox(
    content: str,
    remote_path: str,
    env,
) -> bool | None:
    """Publish verified bytes without replacing an existing artifact.

    Returns ``True`` when this call created the path, ``False`` when an
    identical content-addressed artifact already existed, and ``None`` on any
    transport, integrity, or collision failure.
    """
    storage_dir = os.path.dirname(remote_path)
    artifact_root = os.path.dirname(storage_dir)
    temp_path = f"{remote_path}.tmp-{uuid.uuid4().hex}"
    quoted_artifact_root = shlex.quote(artifact_root)
    quoted_storage_dir = shlex.quote(storage_dir)
    quoted_temp_path = shlex.quote(temp_path)
    quoted_remote_path = shlex.quote(remote_path)
    payload = content.encode("utf-8")
    payload_size = len(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    created_marker = "HERMES_ARTIFACT_CREATED"
    reused_marker = "HERMES_ARTIFACT_REUSED"
    cmd = (
        "{ __hermes_sha256() { "
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum; "
        "elif command -v shasum >/dev/null 2>&1; then shasum -a 256; "
        "else return 127; fi; }; "
        "__hermes_matches() { "
        f"test -f {quoted_remote_path} && test ! -L {quoted_remote_path} && "
        f"__hermes_sha=$(__hermes_sha256 < {quoted_remote_path} 2>/dev/null) && "
        f"test \"${{__hermes_sha%%[[:space:]]*}}\" = {expected_sha256}; }}; "
        f"umask 077 && mkdir -p {quoted_artifact_root} && "
        f"test -d {quoted_artifact_root} && test ! -L {quoted_artifact_root} && "
        f"chmod 700 {quoted_artifact_root} && mkdir -p {quoted_storage_dir} && "
        f"test -d {quoted_storage_dir} && test ! -L {quoted_storage_dir} && "
        f"chmod 700 {quoted_storage_dir} && "
        f"head -c {payload_size} > {quoted_temp_path} && "
        f"test \"$(wc -c < {quoted_temp_path})\" -eq {payload_size} && "
        f"__hermes_temp_sha=$(__hermes_sha256 < {quoted_temp_path} 2>/dev/null) && "
        f"test \"${{__hermes_temp_sha%%[[:space:]]*}}\" = {expected_sha256} && "
        f"chmod 600 {quoted_temp_path} && "
        "__hermes_created=0 && "
        f"if test -e {quoted_remote_path} || test -L {quoted_remote_path}; then "
        f"__hermes_matches && rm -f -- {quoted_temp_path} && "
        f"printf '%s\\n' {reused_marker}; "
        f"elif ln {quoted_temp_path} {quoted_remote_path} 2>/dev/null; then "
        f"__hermes_created=1 && chmod 600 {quoted_remote_path} && __hermes_matches && "
        f"rm -f -- {quoted_temp_path} && printf '%s\\n' {created_marker}; "
        f"else __hermes_matches && rm -f -- {quoted_temp_path} && "
        f"printf '%s\\n' {reused_marker}; fi; "
        f"__hermes_ec=$?; if test \"$__hermes_ec\" -ne 0; then "
        f"rm -f -- {quoted_temp_path} 2>/dev/null || true; "
        f"if test \"$__hermes_created\" -eq 1; then "
        f"rm -f -- {quoted_remote_path} 2>/dev/null || true; fi; fi; "
        f"(exit \"$__hermes_ec\"); }}"
    )
    cleanup_cmd = f"rm -f -- {quoted_temp_path} 2>/dev/null || true"

    try:
        result = env.execute(cmd, timeout=30, stdin_data=content)
    except Exception:
        try:
            env.execute(cleanup_cmd, timeout=30)
        except Exception:
            pass
        return None
    if result.get("returncode", 1) != 0:
        try:
            env.execute(cleanup_cmd, timeout=30)
        except Exception:
            pass
        return None
    output = str(result.get("output", ""))
    if reused_marker in output:
        return False
    if created_marker in output:
        return True
    return None


def persist_tool_artifact(
    content: str,
    *,
    kind: str,
    tool_name: str,
    tool_use_id: str,
    task_scope: str,
    env,
    extension: str,
) -> PersistedToolArtifact | None:
    """Persist a redacted, immutable artifact in the active sandbox."""
    if not _SAFE_ARTIFACT_EXTENSION.fullmatch(extension):
        raise ValueError(f"Unsafe artifact extension: {extension!r}")
    if env is None or not isinstance(task_scope, str) or not task_scope.strip():
        return None

    remote_path = "<unresolved>"
    try:
        persisted_content = redact_sensitive_text(content, force=True)
        if not isinstance(persisted_content, str):
            return None
        payload = persisted_content.encode("utf-8")
        content_sha256 = hashlib.sha256(payload).hexdigest()
        scope_hash = hashlib.sha256(task_scope.encode("utf-8")).hexdigest()
        safe_stem = _safe_result_filename(tool_use_id).removesuffix(".txt")
        # Attempt-unique publication prevents a failed concurrent prune from
        # deleting bytes already referenced by another successful prune. The
        # scope and content digests still make provenance and integrity
        # explicit, while the nonce gives each rollback exclusive ownership.
        # All three are truncated because the path itself is quoted into the
        # in-context stub; the returned artifact keeps the full sha256.
        attempt_token = uuid.uuid4().hex[:_ARTIFACT_ATTEMPT_TOKEN_CHARS]
        remote_path = (
            f"{_resolve_storage_dir(env)}/{scope_hash[:_ARTIFACT_SCOPE_HASH_CHARS]}/"
            f"{content_sha256[:_ARTIFACT_CONTENT_DIGEST_CHARS]}_{safe_stem}"
            f"_{attempt_token}{extension}"
        )
        created = _publish_immutable_to_sandbox(persisted_content, remote_path, env)
        if created is None:
            logger.warning(
                "Sandbox artifact write failed: kind=%s tool=%s path=%s",
                kind,
                tool_name,
                remote_path,
            )
            return None
    except Exception:
        logger.warning(
            "Sandbox artifact write failed: kind=%s tool=%s path=%s",
            kind,
            tool_name,
            remote_path,
        )
        return None

    return PersistedToolArtifact(
        kind=kind,
        path=remote_path,
        chars=len(persisted_content),
        sha256=content_sha256,
        redacted=persisted_content != content,
        created=created,
    )


def remove_created_tool_artifact(artifact: PersistedToolArtifact, env) -> bool:
    """Remove only a verified artifact created by the current prune attempt."""
    if not artifact.created:
        return True
    quoted_path = shlex.quote(artifact.path)
    expected_sha256 = artifact.sha256
    cmd = (
        "{ __hermes_sha256() { "
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum; "
        "elif command -v shasum >/dev/null 2>&1; then shasum -a 256; "
        "else return 127; fi; }; "
        f"test -f {quoted_path} && test ! -L {quoted_path} && "
        f"__hermes_sha=$(__hermes_sha256 < {quoted_path} 2>/dev/null) && "
        f"test \"${{__hermes_sha%%[[:space:]]*}}\" = {expected_sha256} && "
        f"rm -f -- {quoted_path} && test ! -e {quoted_path} && test ! -L {quoted_path}; }}"
    )
    try:
        result = env.execute(cmd, timeout=30)
    except Exception:
        return False
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
        f"Full output saved to: {file_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        + preview + ("\n..." if has_more else "")
        + f"\n{PERSISTED_OUTPUT_CLOSING_TAG}")


_PERSISTED_PATH_RE = re.compile(r"^Full output saved to: (.+)$", re.MULTILINE)


def extract_persisted_path(content: str) -> str | None:
    """File path from a <persisted-output> block, or None (lets the result-reference stubbing
    guard in agent/tool_guardrails.py carry the spillover path instead of leaving it dangling)."""
    match = (_PERSISTED_PATH_RE.search(content)
             if isinstance(content, str) and PERSISTED_OUTPUT_TAG in content else None)
    return match.group(1).strip() if match else None


def maybe_persist_tool_result(content: str, tool_name: str, tool_use_id: str, env=None,
                              config: BudgetConfig = DEFAULT_BUDGET,
                              threshold: int | float | None = None) -> str:
    """Layer 2: persist an oversized result, return preview + path. ``threshold`` overrides
    ``config.resolve_threshold(tool_name)``; falls back to inline truncation when no write
    location succeeds."""
    if threshold is None:
        threshold = config.resolve_threshold(tool_name)
    if threshold == float("inf") or len(content) <= threshold:
        return content
    filename = _safe_result_filename(tool_use_id)
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    def _persisted(path: str, host_suffix: str = "") -> str:
        logger.info("Persisted large tool result: %s (%s, %d chars -> %s%s)",
                    tool_name, tool_use_id, len(content), path, host_suffix)
        return _build_persisted_message(preview, has_more, len(content), path)

    # Always persist host-side first: cache/spillover is the single canonical home.
    host_path = _write_to_spillover(content, filename)
    host_side = _is_host_side_env(env)
    if host_side and host_path is not None:
        return _persisted(host_path)
    if not host_side:
        # Remote backend: reference the mounted/synced path when the sandbox can actually read
        # it, else write into the sandbox temp dir (containers without the spillover mount).
        visible = _sandbox_visible_spillover_path(host_path, env) if host_path else None
        if visible is not None:
            return _persisted(visible, f" [host: {host_path}]")
        remote_path = f"{_resolve_storage_dir(env)}/{filename}"
        try:
            if _write_to_sandbox(content, remote_path, env):
                return _persisted(remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
    logger.info("Inline-truncating large tool result: %s (%d chars, no sandbox write)",
                tool_name, len(content))
    return (f"{preview}\n\n[Truncated: tool response was {len(content):,} chars. "
            "Full output could not be saved to sandbox.]")


def enforce_turn_budget(tool_messages: list[dict], env=None,
                        config: BudgetConfig = DEFAULT_BUDGET) -> list[dict]:
    """Layer 3: persist the largest non-persisted results first until the turn's aggregate is
    under budget. Mutates the list in-place and returns it."""
    sizes = [len(msg.get("content", "")) for msg in tool_messages]
    total_size = sum(sizes)
    candidates = [(i, size) for i, size in enumerate(sizes)
                  if PERSISTED_OUTPUT_TAG not in tool_messages[i].get("content", "")]
    if total_size <= config.turn_budget:
        return tool_messages
    for idx, size in sorted(candidates, key=lambda x: x[1], reverse=True):
        if total_size <= config.turn_budget:
            break
        content = tool_messages[idx]["content"]
        tool_use_id = tool_messages[idx].get("tool_call_id", f"budget_{idx}")
        replacement = maybe_persist_tool_result(
            content=content, tool_name=_BUDGET_TOOL_NAME, tool_use_id=tool_use_id,
            env=env, config=config, threshold=0)
        if replacement != content:
            total_size += len(replacement) - size
            tool_messages[idx]["content"] = replacement
            logger.info("Budget enforcement: persisted tool result %s (%d chars)",
                        tool_use_id, size)
    return tool_messages


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
# ---- END PLUGIN-COMPAT ----
