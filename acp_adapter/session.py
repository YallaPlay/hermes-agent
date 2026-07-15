"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``.  When the editor
reconnects after idle/restart, the ``load_session`` / ``resume_session`` calls
find the persisted session in the database and restore the full conversation
history.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home

import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _translate_acp_cwd(cwd: str) -> str:
    """Translate Windows ACP cwd values when Hermes itself is running in WSL.

    Windows ACP clients can launch ``hermes acp`` inside WSL while still sending
    editor workspaces as Windows drive paths (``E:\\Projects``) or
    ``\\\\wsl.localhost\\`` UNC paths. Store and execute against the POSIX form so
    agents, tools, and persisted ACP sessions all agree on the usable workspace.
    Native Linux/macOS keeps the original cwd unchanged.
    """
    from hermes_constants import translate_cwd_for_wsl_backend

    return translate_cwd_for_wsl_backend(str(cwd))


def _normalize_cwd_for_compare(cwd: str | None) -> str:
    raw = str(cwd or ".").strip()
    if not raw:
        raw = "."
    expanded = os.path.expanduser(raw)

    # Normalize Windows drive paths into the equivalent WSL mount form so
    # ACP history filters match the same workspace across Windows and WSL.
    from hermes_constants import windows_path_to_wsl

    translated = windows_path_to_wsl(expanded)
    if translated is not None:
        expanded = translated
    elif re.match(r"^/mnt/[A-Za-z]/", expanded):
        expanded = f"/mnt/{expanded[5].lower()}/{expanded[7:]}"

    return os.path.normpath(expanded)


def _preview_text(content: Any, limit: int = 60) -> str:
    """Flatten an in-memory message ``content`` value into preview text.

    Multimodal user messages (e.g. a prompt with a screenshot) hold ``content``
    as a list of parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``).
    A naive ``str()`` leaks the list repr into the sessions-list title —
    mirror SessionDB._preview_from_raw: flatten the text parts, fall back to a
    ``[multimodal content]`` placeholder for image-only messages.
    """
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = " ".join(t.strip() for t in parts if t and t.strip()).strip()
        text = text or "[multimodal content]"
    else:
        text = str(content or "").strip()
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


# Slack-bot turns are wrapped in a runtime envelope whose first user message
# starts with this marker (see the Slack runner's build_hermes_slack_prompt).
# Since the Slack bot switched to the ACP provider its sessions persist as
# source='acp', so this content prefix is the only reliable Slack tell.
_SLACK_CONTEXT_PREFIX = "[Slack runtime context]"


def _is_slack_preview(preview: Any) -> bool:
    """True when a session's first-user-message preview marks a Slack session."""
    return str(preview or "").lstrip().startswith(_SLACK_CONTEXT_PREFIX)


def _build_session_title(title: Any, preview: Any, cwd: str | None) -> str:
    explicit = str(title or "").strip()
    if explicit:
        return explicit
    preview_text = str(preview or "").strip()
    if preview_text:
        return preview_text
    leaf = os.path.basename(str(cwd or "").rstrip("/\\"))
    return leaf or "New thread"


def _forked_from_marker(model_config: Any) -> str | None:
    """Extract the ``_forked_from`` lineage marker from a model_config blob.

    Accepts either the raw JSON string (as stored in sessions.model_config) or
    an already-decoded dict. Returns None when absent/unparseable.
    """
    meta = model_config
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(meta, dict):
        return None
    parent = str(meta.get("_forked_from") or "").strip()
    return parent or None


def _format_updated_at(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _updated_at_sort_key(value: Any) -> float:
    if value is None:
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return float("-inf")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(raw)
        except Exception:
            return float("-inf")


def _acp_stderr_print(*args, **kwargs) -> None:
    """Best-effort human-readable output sink for ACP stdio sessions.

    ACP reserves stdout for JSON-RPC frames, so any incidental CLI/status output
    from AIAgent must be redirected away from stdout. Route it to stderr instead.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _register_task_cwd(task_id: str, cwd: str) -> None:
    """Bind a task/session id to the editor's working directory for tools.

    Zed can launch Hermes from a Windows workspace while the ACP process runs
    inside WSL. In that case ACP sends cwd as e.g. ``E:\\Projects\\POTI``;
    local tools need the WSL mount equivalent or subprocess creation fails
    before the command can run.
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import register_task_env_overrides
        register_task_env_overrides(task_id, {"cwd": _translate_acp_cwd(cwd)})
    except Exception:
        logger.debug("Failed to register ACP task cwd override", exc_info=True)


def _acp_base_toolsets() -> List[str]:
    """Return the base toolsets for ACP sessions.

    Honors an explicit ``platform_toolsets.acp`` list in config.yaml so users
    can trim the ACP tool surface (e.g. drop ``browser``) the same way they
    configure other platforms. Falls back to the built-in composite
    ``hermes-acp`` toolset when no list is configured.
    """
    try:
        from hermes_cli.config import load_config

        raw = (load_config().get("platform_toolsets") or {}).get("acp")
        if isinstance(raw, list):
            names = [str(t) for t in raw if t]
            if names:
                return names
    except Exception:
        logger.debug("Failed to read platform_toolsets.acp from config", exc_info=True)
    return ["hermes-acp"]


def _expand_acp_enabled_toolsets(
    toolsets: List[str] | None = None,
    mcp_server_names: List[str] | None = None,
) -> List[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    expanded: List[str] = []
    for name in list(toolsets or _acp_base_toolsets()):
        if name and name not in expanded:
            expanded.append(name)

    for server_name in list(mcp_server_names or []):
        toolset_name = f"mcp-{server_name}"
        if server_name and toolset_name not in expanded:
            expanded.append(toolset_name)

    return expanded


def _apply_effort_to_agent(agent: Any, effort: str) -> None:
    """Mirror a session's reasoning-effort override onto a (fresh) agent.

    ``agent.reasoning_config`` is read on every API call, so mutating it here
    takes effect on the next call (same mechanism as the reasoning_effort
    tool). Empty/invalid levels are a no-op — the agent keeps its default.
    """
    effort = str(effort or "").strip().lower()
    if not effort:
        return
    try:
        from hermes_constants import parse_reasoning_effort

        parsed = parse_reasoning_effort(effort)
        if parsed is not None:
            agent.reasoning_config = parsed
    except Exception:
        logger.debug("Could not apply reasoning effort %r to agent", effort, exc_info=True)


def _clear_task_cwd(task_id: str) -> None:
    """Remove task-specific cwd overrides for an ACP session."""
    if not task_id:
        return
    try:
        from tools.terminal_tool import clear_task_env_overrides
        clear_task_env_overrides(task_id)
    except Exception:
        logger.debug("Failed to clear ACP task cwd override", exc_info=True)


@dataclass
class SessionState:
    """Tracks per-session state for an ACP-managed Hermes agent."""

    session_id: str
    agent: Any  # AIAgent instance
    cwd: str = "."
    model: str = ""
    # ACP session mode (edit-approval policy: default / acceptEdits / dontAsk).
    # Set by set_session_mode; persisted in the model_config JSON blob so it
    # survives an agent restart (e.g. a VS Code window reload respawns the ACP
    # child) instead of reverting to the server default.
    mode: str = ""
    # Session-scoped reasoning effort override (none/minimal/low/medium/high/
    # xhigh), set via the reasoningEffort ACP config option. Empty = provider
    # default. Persisted like `mode` and re-applied to the agent on restore.
    effort: str = ""
    # Authenticated owner (e.g. Cloudflare Access email) supplied by the client
    # on session/new. Persisted to sessions.user_id so the ACP sessions list can
    # filter to a user's own sessions. Soft display key, not an access boundary.
    owner: Optional[str] = None
    # Session this one was forked from (session/fork lineage). Persisted as a
    # ``_forked_from`` marker in the model_config JSON — NOT sessions.
    # parent_session_id, which list_sessions_rich treats as subagent/compression
    # lineage and would hide the fork from session lists. Display-only.
    parent_id: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = None  # threading.Event
    is_running: bool = False
    queued_prompts: List[str] = field(default_factory=list)
    runtime_lock: Any = field(default_factory=Lock)
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the
    shared SessionDB so they survive process restarts and are searchable
    via ``session_search``.
    """

    def __init__(self, agent_factory=None, db=None):
        """
        Args:
            agent_factory: Optional callable that creates an AIAgent-like object.
                           Used by tests. When omitted, a real AIAgent is created
                           using the current Hermes runtime provider configuration.
            db:            Optional SessionDB instance. When omitted, the default
                           SessionDB (``~/.hermes/state.db``) is lazily created.
        """
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use

    # ---- public API ---------------------------------------------------------

    def create_session(self, cwd: str = ".", owner: str | None = None) -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent.

        ``owner`` (an authenticated user identifier, e.g. the Cloudflare Access
        email supplied by the client on ``session/new``) is persisted to
        ``sessions.user_id`` so the ACP sessions list can show a user only
        their own sessions. It is a soft per-user display key, not an access
        boundary.
        """
        import threading

        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(session_id=session_id, cwd=cwd)
        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", "") or "",
            cancel_event=threading.Event(),
        )
        state.owner = (owner or "").strip() or None
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Return the session for *session_id*, or ``None``.

        If the session is not in memory but exists in the database (e.g. after
        a process restart), it is transparently restored.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return state
        # Attempt to restore from database.
        return self._restore(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from memory and database. Returns True if it existed."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        db_existed = self._delete_persisted(session_id)
        if existed or db_existed:
            _clear_task_cwd(session_id)
        return existed or db_existed

    def fork_session(
        self,
        session_id: str,
        cwd: str = ".",
        keep_history: Optional[int] = None,
    ) -> Optional[SessionState]:
        """Deep-copy a session's history into a new session.

        ``keep_history`` limits the copy to the first N history entries
        (``history[:keep_history]``), enabling "fork from here" rewind
        semantics. ``None`` copies the full history. Negative values are
        rejected: ``history[:-N]`` would silently mean "drop the last N
        messages" — a destructive footgun, not a fork prefix.
        """
        import threading

        if keep_history is not None and keep_history < 0:
            raise ValueError("keep_history must be non-negative")

        cwd = _translate_acp_cwd(cwd)
        original = self.get_session(session_id)  # checks DB too
        if original is None:
            return None

        forked_history = (
            original.history if keep_history is None else original.history[:keep_history]
        )
        new_id = str(uuid.uuid4())
        # Carry the parent's provider routing into the fork. Without this the
        # fork agent resolves the config-default provider while keeping the
        # parent's model name — e.g. an openai-codex/gpt-5.6-sol parent forks
        # into bedrock/gpt-5.6-sol and every turn 400s with "The provided
        # model identifier is invalid".
        parent_agent = original.agent
        agent = self._make_agent(
            session_id=new_id,
            cwd=cwd,
            model=original.model or None,
            requested_provider=getattr(parent_agent, "provider", None),
            base_url=getattr(parent_agent, "base_url", None),
            api_mode=getattr(parent_agent, "api_mode", None),
        )
        state = SessionState(
            session_id=new_id,
            agent=agent,
            cwd=cwd,
            model=getattr(agent, "model", original.model) or original.model,
            # Carry the edit-approval mode and reasoning effort into the fork:
            # a fork continues the same working context, so silently reverting
            # to server defaults would surprise the user mid-flow.
            mode=original.mode,
            effort=original.effort,
            # A fork belongs to whoever owns the parent — without this the
            # fork row persists untagged and the strict "My Sessions" owner
            # filter hides it after a reload.
            owner=original.owner,
            # Record fork lineage so clients can nest the fork under its
            # parent in session lists.
            parent_id=session_id,
            history=copy.deepcopy(forked_history),
            cancel_event=threading.Event(),
        )
        _apply_effort_to_agent(agent, original.effort)
        with self._lock:
            self._sessions[new_id] = state
        _register_task_cwd(new_id, cwd)
        self._persist(state)
        logger.info("Forked ACP session %s -> %s", session_id, new_id)
        return state

    def list_sessions(
        self,
        cwd: str | None = None,
        include_archived: bool = False,
        archived_only: bool = False,
        owner: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return lightweight info dicts for all sessions (memory + database)."""
        normalized_cwd = _normalize_cwd_for_compare(cwd) if cwd else None
        owner_filter = (owner or "").strip() or None
        db = self._get_db()
        persisted_rows: dict[str, dict[str, Any]] = {}

        if db is not None:
            try:
                for row in db.list_sessions_rich(
                    source="acp",
                    limit=1000,
                    include_archived=include_archived,
                    archived_only=archived_only,
                    owner=owner_filter,
                ):
                    persisted_rows[str(row["id"])] = dict(row)
            except Exception:
                logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        # Collect in-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                history_len = len(s.history)
                if history_len <= 0:
                    continue
                if archived_only:
                    continue
                if normalized_cwd and _normalize_cwd_for_compare(s.cwd) != normalized_cwd:
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                if owner_filter:
                    # Mirror the DB filter for in-memory rows: STRICT ownership —
                    # show only the caller's own sessions, hide untagged and
                    # other-owner rows. Prefer the in-memory owner, fall back to
                    # the persisted user_id.
                    row_owner = (
                        (s.owner or persisted.get("user_id") or "").strip()
                    )
                    if row_owner != owner_filter:
                        continue
                preview = next(
                    (
                        _preview_text(msg.get("content"))
                        for msg in s.history
                        if msg.get("role") == "user" and _preview_text(msg.get("content"))
                    ),
                    persisted.get("preview") or "",
                )
                parent_id = s.parent_id or _forked_from_marker(persisted.get("model_config"))
                results.append(
                    {
                        "session_id": s.session_id,
                        "cwd": s.cwd,
                        "model": s.model,
                        "history_len": history_len,
                        "user_id": s.owner or persisted.get("user_id") or "",
                        "title": _build_session_title(persisted.get("title"), preview, s.cwd),
                        "updated_at": _format_updated_at(
                            persisted.get("last_active") or persisted.get("started_at") or time.time()
                        ),
                        "archived": False,
                        "parent_id": parent_id,
                        # A fork of a Slack session is an ACP/VS Code session in
                        # its own right — don't inherit the Slack badge.
                        "slack": not parent_id
                        and _is_slack_preview(preview or persisted.get("preview")),
                    }
                )

        # Merge any persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            if sid in seen_ids:
                continue
            message_count = int(row.get("message_count") or 0)
            if message_count <= 0:
                continue
            # Extract cwd from model_config JSON.
            session_cwd = "."
            mc = row.get("model_config")
            if mc:
                try:
                    session_cwd = json.loads(mc).get("cwd", ".")
                except (json.JSONDecodeError, TypeError):
                    pass
            if normalized_cwd and _normalize_cwd_for_compare(session_cwd) != normalized_cwd:
                continue
            parent_id = _forked_from_marker(mc)
            results.append({
                "session_id": sid,
                "cwd": session_cwd,
                "model": row.get("model") or "",
                "history_len": message_count,
                "user_id": row.get("user_id") or "",
                "title": _build_session_title(row.get("title"), row.get("preview"), session_cwd),
                "updated_at": _format_updated_at(row.get("last_active") or row.get("started_at")),
                "archived": bool(row.get("archived")),
                "parent_id": parent_id,
                # Forks of Slack sessions are not Slack sessions — no badge.
                "slack": not parent_id and _is_slack_preview(row.get("preview")),
            })

        results.sort(key=lambda item: _updated_at_sort_key(item.get("updated_at")), reverse=True)
        return results

    def update_cwd(self, session_id: str, cwd: str) -> Optional[SessionState]:
        """Update the working directory for a session and its tool overrides."""
        cwd = _translate_acp_cwd(cwd)
        state = self.get_session(session_id)  # checks DB too
        if state is None:
            return None
        state.cwd = cwd
        _register_task_cwd(session_id, cwd)
        self._persist(state)
        return state

    def cleanup(self) -> None:
        """Remove all sessions (memory and database) and clear task-specific cwd overrides."""
        with self._lock:
            session_ids = list(self._sessions.keys())
            self._sessions.clear()
        for session_id in session_ids:
            _clear_task_cwd(session_id)
            self._delete_persisted(session_id)
        # Also remove any DB-only ACP sessions not currently in memory.
        db = self._get_db()
        if db is not None:
            try:
                rows = db.search_sessions(source="acp", limit=10000)
                for row in rows:
                    sid = row["id"]
                    _clear_task_cwd(sid)
                    db.delete_session(sid)
            except Exception:
                logger.debug("Failed to cleanup ACP sessions from DB", exc_info=True)

    def save_session(self, session_id: str) -> None:
        """Persist the current state of a session to the database.

        Called by the server after prompt completion, slash commands that
        mutate history, and model switches.
        """
        with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            self._persist(state)

    # ---- persistence via SessionDB ------------------------------------------

    def _get_db(self):
        """Lazily initialise and return the SessionDB instance.

        Returns ``None`` if the DB is unavailable (e.g. import error in a
        minimal test environment).

        Note: we resolve ``HERMES_HOME`` dynamically rather than relying on
        the module-level ``DEFAULT_DB_PATH`` constant, because that constant
        is evaluated at import time and won't reflect env-var changes made
        later (e.g. by the test fixture ``_isolate_hermes_home``).
        """
        if self._db_instance is not None:
            return self._db_instance
        try:
            from hermes_state import SessionDB
            hermes_home = get_hermes_home()
            self._db_instance = SessionDB(db_path=hermes_home / "state.db")
            return self._db_instance
        except Exception:
            logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
            return None

    def _persist(self, state: SessionState) -> None:
        """Write session state to the database.

        Creates the session record if it doesn't exist, then replaces all
        stored messages with the current in-memory history.
        """
        db = self._get_db()
        if db is None:
            return

        # Ensure model is a plain string (not a MagicMock or other proxy).
        model_str = str(state.model) if state.model else None
        session_meta = {"cwd": state.cwd}
        provider = getattr(state.agent, "provider", None)
        base_url = getattr(state.agent, "base_url", None)
        api_mode = getattr(state.agent, "api_mode", None)
        if isinstance(provider, str) and provider.strip():
            session_meta["provider"] = provider.strip()
        if isinstance(base_url, str) and base_url.strip():
            session_meta["base_url"] = base_url.strip()
        if isinstance(api_mode, str) and api_mode.strip():
            session_meta["api_mode"] = api_mode.strip()
        # Persist the ACP session mode so it survives an agent restart; omit the
        # default so existing rows stay byte-identical when nothing changed it.
        session_mode = str(getattr(state, "mode", "") or "").strip()
        if session_mode:
            session_meta["mode"] = session_mode
        # Same for the session's reasoning-effort override.
        session_effort = str(getattr(state, "effort", "") or "").strip()
        if session_effort:
            session_meta["effort"] = session_effort
        # Fork lineage marker. Kept in model_config (like _branched_from) rather
        # than parent_session_id, which the session listers treat as
        # subagent/compression lineage and would hide the fork.
        parent_id = str(getattr(state, "parent_id", "") or "").strip()
        if parent_id:
            session_meta["_forked_from"] = parent_id
        cwd_json = json.dumps(session_meta)

        try:
            # Ensure the session record exists.
            existing = db.get_session(state.session_id)
            if existing is None:
                # Persist the FULL metadata blob (provider/base_url/api_mode/
                # mode/effort/_forked_from), not just cwd — a fork or fresh
                # session that isn't prompted again before a process restart
                # would otherwise restore with no provider routing and fall
                # back to the config default, mismatching its model.
                db.create_session(
                    session_id=state.session_id,
                    source="acp",
                    model=model_str,
                    model_config=session_meta,
                    user_id=getattr(state, "owner", None),
                )
            else:
                # Update model_config (contains cwd) if changed.
                try:
                    db.update_session_meta(state.session_id, cwd_json, model_str)
                except Exception:
                    logger.debug("Failed to update ACP session metadata", exc_info=True)

            # When the agent owns persistence to this same SessionDB it has
            # already flushed the live transcript incrementally during
            # run_conversation (append_message), and it preserves pre-compaction
            # turns non-destructively via archive_and_compact() — keeping them on
            # disk as searchable active=0/compacted=1 rows. Calling
            # replace_messages() here would then be a redundant double-write that
            # DELETEs exactly those archived rows (and, after a compression-driven
            # id rotation where agent.session_id no longer equals
            # state.session_id, clobbers the ended parent transcript) — silent
            # data loss for any ACP conversation long enough to compress.
            #
            # Only fall back to the destructive atomic replace when the agent is
            # NOT persisting itself to this DB (e.g. a test agent factory, or a
            # fresh create/fork whose copied history the agent has not flushed
            # yet). That path still rolls back on a mid-rewrite failure so the
            # previously persisted conversation survives (salvaged from #13675).
            agent = state.agent
            agent_db = getattr(agent, "_session_db", None)
            agent_owns_persistence = (
                agent_db is not None
                and agent_db is db
                and bool(getattr(agent, "_session_db_created", False))
            )
            if not agent_owns_persistence:
                # Even when the current agent doesn't "own" persistence, the
                # session on disk may already carry compaction-archived rows —
                # e.g. after a model switch or a /restore, both of which mint a
                # fresh agent with _session_db_created=False (so the check above
                # is False) yet leave the durable archived transcript in place.
                # A full-history replace would DELETE those archived rows just
                # like the owned-agent case. Guard against it: when archived
                # rows exist, replace ONLY the live (active=1) set and leave the
                # archived turns untouched; otherwise the destructive replace is
                # safe (fresh create/fork with no archived history to lose).
                try:
                    has_archived = db.has_archived_messages(state.session_id)
                except Exception:
                    has_archived = False
                db.replace_messages(
                    state.session_id, state.history, active_only=has_archived
                )
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load a session from the database into memory, recreating the AIAgent."""
        import threading

        db = self._get_db()
        if db is None:
            return None

        try:
            row = db.get_session(session_id)
        except Exception:
            logger.debug("Failed to query DB for ACP session %s", session_id, exc_info=True)
            return None

        if row is None:
            return None

        # Only restore ACP sessions.
        if row.get("source") != "acp":
            return None

        # Extract cwd from model_config.
        cwd = "."
        requested_provider = row.get("billing_provider")
        restored_base_url = row.get("billing_base_url")
        restored_api_mode = None
        restored_mode = ""
        restored_effort = ""
        restored_parent_id = ""
        mc = row.get("model_config")
        if mc:
            try:
                meta = json.loads(mc)
                if isinstance(meta, dict):
                    cwd = meta.get("cwd", ".")
                    requested_provider = meta.get("provider") or requested_provider
                    restored_base_url = meta.get("base_url") or restored_base_url
                    restored_api_mode = meta.get("api_mode") or restored_api_mode
                    restored_mode = str(meta.get("mode") or "").strip()
                    restored_effort = str(meta.get("effort") or "").strip()
                    restored_parent_id = str(meta.get("_forked_from") or "").strip()
            except (json.JSONDecodeError, TypeError):
                pass

        model = row.get("model") or None

        # Load conversation history.
        try:
            history = db.get_messages_as_conversation(session_id)
        except Exception:
            logger.warning("Failed to load messages for ACP session %s", session_id, exc_info=True)
            history = []

        try:
            agent = self._make_agent(
                session_id=session_id,
                cwd=cwd,
                model=model,
                requested_provider=requested_provider,
                base_url=restored_base_url,
                api_mode=restored_api_mode,
            )
        except Exception:
            logger.warning("Failed to recreate agent for ACP session %s", session_id, exc_info=True)
            return None

        state = SessionState(
            session_id=session_id,
            agent=agent,
            cwd=cwd,
            model=model or getattr(agent, "model", "") or "",
            mode=restored_mode,
            effort=restored_effort,
            # Rehydrate the owner so forks of a restored session inherit it
            # (and a later _persist doesn't drop it).
            owner=str(row.get("user_id") or "").strip() or None,
            parent_id=restored_parent_id or None,
            history=history,
            cancel_event=threading.Event(),
        )
        _apply_effort_to_agent(agent, restored_effort)
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        logger.info("Restored ACP session %s from DB (%d messages)", session_id, len(history))
        return state

    def _delete_persisted(self, session_id: str) -> bool:
        """Delete a session from the database. Returns True if it existed."""
        db = self._get_db()
        if db is None:
            return False
        try:
            return db.delete_session(session_id)
        except Exception:
            logger.debug("Failed to delete ACP session %s from DB", session_id, exc_info=True)
            return False

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        """Soft-archive (hide) or restore a session. Reversible; not a delete."""
        db = self._get_db()
        if db is None:
            return False
        try:
            return bool(db.set_session_archived(session_id, archived))
        except Exception:
            logger.debug("Failed to set archived on %s", session_id, exc_info=True)
            return False

    def set_session_owner(self, session_id: str, user_id: str) -> bool:
        """Stamp (or clear, when *user_id* is empty) a session's owner in
        state.db. The owner is the authenticated user identifier the ACP
        sessions list filters on. Soft per-user display key, not a boundary.
        Also mirrors the value onto the in-memory session so a subsequent
        ``_persist`` doesn't drop it. Returns True when a row was updated.
        """
        db = self._get_db()
        if db is None:
            return False
        owner = (user_id or "").strip() or None
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                state.owner = owner
        try:
            return bool(db.set_session_owner(session_id, owner or ""))
        except Exception:
            logger.debug("Failed to set owner on %s", session_id, exc_info=True)
            return False

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set (or clear, when *title* is empty/whitespace) a session's canonical
        title in state.db. Returns True when a row was updated.

        Raises ValueError on validation failure or a title-uniqueness conflict —
        callers (the ACP ext-method) surface that to the client so an inline
        rename collision is a visible error, not a silent no-op.
        """
        db = self._get_db()
        if db is None:
            return False
        return bool(db.set_session_title(session_id, title))

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Return the session's canonical title, or None."""
        db = self._get_db()
        if db is None:
            return None
        try:
            return db.get_session_title(session_id)
        except Exception:
            logger.debug("Failed to read title for %s", session_id, exc_info=True)
            return None

    def derive_session_title(self, session_id: str) -> Optional[str]:
        """Derive and persist a title from the session's FIRST user→assistant
        exchange, reusing the same auxiliary-model pass as auto-titling.

        Unlike ``maybe_auto_title`` this is NOT gated to the first turn — it is
        the on-demand backfill for sessions that opened into a long/interrupted
        first turn and never got auto-titled (the classic ``source='acp'``
        untitled case). Runs synchronously (the caller invokes it off-thread) and
        is a no-op when the session already has a title. Returns the new title,
        or None when nothing was set (no exchange yet / already titled / aux
        failure)."""
        db = self._get_db()
        if db is None:
            return None
        try:
            if db.get_session_title(session_id):
                return None  # already titled — never overwrite
            messages = db.get_messages(session_id)
        except Exception:
            logger.debug("derive_session_title: could not read %s", session_id, exc_info=True)
            return None

        def _text(m):
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                )
            return ""

        first_user = next((_text(m) for m in messages if m.get("role") == "user"), "")
        first_assistant = next((_text(m) for m in messages if m.get("role") == "assistant"), "")
        if not first_user or not first_assistant:
            return None  # no completed exchange yet — nothing to title from

        try:
            from agent.title_generator import generate_title
            title = generate_title(first_user, first_assistant)
        except Exception:
            logger.debug("derive_session_title: generation failed for %s", session_id, exc_info=True)
            return None
        if not title:
            return None
        try:
            if db.set_session_title(session_id, title):
                return title
        except ValueError:
            # Extremely unlikely (a derived title colliding with another
            # session); leave untitled rather than error a background derive.
            logger.debug("derive_session_title: title collision for %s", session_id, exc_info=True)
        return None

    # ---- internal -----------------------------------------------------------

    def _make_agent(
        self,
        *,
        session_id: str,
        cwd: str,
        model: str | None = None,
        requested_provider: str | None = None,
        base_url: str | None = None,
        api_mode: str | None = None,
    ):
        if self._agent_factory is not None:
            return self._agent_factory()

        from run_agent import AIAgent
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or default_model)
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        configured_mcp_servers = [
            name
            for name, cfg in (config.get("mcp_servers") or {}).items()
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is not False
        ]

        # Honor agent.disabled_toolsets from config.yaml on the ACP surface,
        # matching the CLI (cli.py reads the same key). Without this, a
        # globally suppressed toolset (e.g. "browser") still loads in editor
        # sessions because only enabled_toolsets is passed to the agent.
        agent_cfg = config.get("agent") or {}
        disabled_toolsets = agent_cfg.get("disabled_toolsets") or None
        if disabled_toolsets is not None:
            disabled_toolsets = [str(t) for t in disabled_toolsets if t]

        kwargs = {
            "platform": "acp",
            "enabled_toolsets": _expand_acp_enabled_toolsets(
                None,  # resolves platform_toolsets.acp or hermes-acp default
                mcp_server_names=configured_mcp_servers,
            ),
            "disabled_toolsets": disabled_toolsets,
            "quiet_mode": True,
            "session_id": session_id,
            "session_db": self._get_db(),
            "model": model or default_model,
        }

        try:
            runtime = resolve_runtime_provider(requested=requested_provider or config_provider)
            kwargs.update(
                {
                    "provider": runtime.get("provider"),
                    "api_mode": api_mode or runtime.get("api_mode"),
                    "base_url": base_url or runtime.get("base_url"),
                    "api_key": runtime.get("api_key"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                }
            )
        except Exception:
            logger.debug("ACP session falling back to default provider resolution", exc_info=True)

        _register_task_cwd(session_id, cwd)
        agent = AIAgent(**kwargs)
        # Codex app-server sessions are spawned lazily on the first turn. Stamp
        # the ACP workspace onto the agent so the Codex runtime starts from the
        # editor/session cwd instead of the Hermes daemon's process cwd.
        agent.session_cwd = cwd
        # ACP stdio transport requires stdout to remain protocol-only JSON-RPC.
        # Route any incidental human-readable agent output to stderr instead.
        agent._print_fn = _acp_stderr_print
        return agent
