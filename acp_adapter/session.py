"""ACP session manager — maps ACP sessions to Hermes AIAgent instances.

Sessions are persisted to the shared SessionDB (``~/.hermes/state.db``) so they
survive process restarts and appear in ``session_search``.  When the editor
reconnects after idle/restart, the ``load_session`` / ``resume_session`` calls
find the persisted session in the database and restore the full conversation
history.
"""
from __future__ import annotations

from hermes_constants import get_hermes_home, strip_owner_note
from agent.context_compressor import is_compaction_summary_message

import copy
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, UTC
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

    # Resolve symlink aliases so equivalent spellings of the same directory
    # compare equal — macOS reports editor workspaces as ``/var/...`` while
    # sessions get stored under ``/private/var/...`` (and ``/tmp`` vs
    # ``/private/tmp``), which made ACP history filters silently drop a
    # workspace's own sessions. ``os.path.realpath`` is lexical for missing
    # paths (strict=False), so cwds that don't exist on this host — e.g.
    # WSL-translated Windows drives — keep the previous normpath behavior.
    # Ported from PrimeIntellect-ai/prime-agent#628.
    try:
        return os.path.realpath(expanded)
    except OSError:
        return os.path.normpath(expanded)


def _preview_text(content: Any, limit: int = 60) -> str:
    """Flatten an in-memory message ``content`` value into preview text.

    Multimodal user messages (e.g. a prompt with a screenshot) hold ``content``
    as a list of parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``).
    A naive ``str()`` leaks the list repr into the sessions-list title —
    mirror hermes_state._flatten_preview_raw: flatten the text parts, fall back
    to a ``[multimodal content]`` placeholder for image-only messages.
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
    # An owned session's first user message carries the authenticated-owner
    # note; stripping it here keeps the owner email out of list titles.
    text = strip_owner_note(text)
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


def _agent_provider_identity(agent: Any) -> str | None:
    """Best provider identity for persist/fork/model-state round-trips.

    Prefers the requested (pre-canonicalization) identity — e.g.
    ``custom:yallaplay-cpa`` — over the canonical ``provider`` attribute,
    which for named custom endpoints is bare ``custom``. Re-resolving bare
    ``custom`` yields the *generic* env/config custom provider (default
    base_url: OpenRouter), silently rerouting the session away from the
    named endpoint the user picked. Non-string / empty values (MagicMock
    test agents, legacy agents) fall through to ``provider``.
    """
    for attr in ("requested_provider", "provider"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except Exception:
        return None


def _progress_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Persisted progress markers for ``session/list`` rows.

    A session can advance in another process (Slack bot, gateway, CLI) that
    this ACP server never observes. Clients compare ``message_count`` against
    the count they last rendered to decide when to re-replay, and show
    ``activity`` (the live heartbeat) while no new message has landed yet.
    """
    fields: Dict[str, Any] = {}
    if row.get("message_count") is not None:
        fields["message_count"] = int(row["message_count"])
    description = str(row.get("last_activity_description") or "").strip()
    at = _format_updated_at(row.get("last_activity_at"))
    if description and at:
        fields["activity"] = {"description": description, "at": at}
    return fields


def _updated_at_sort_key(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return float("-inf")
    for parse in (lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp(), float):
        try:
            return parse(raw)
        except Exception:
            continue
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


def _expand_acp_enabled_toolsets(toolsets: list[str] | None = None,
                                 mcp_server_names: list[str] | None = None) -> list[str]:
    """Return ACP toolsets plus explicit MCP server toolsets for this session."""
    names = [n for n in (["hermes-acp"] if toolsets is None else toolsets) if n]
    names += [f"mcp-{s}" for s in (mcp_server_names or []) if s]
    return list(dict.fromkeys(names))


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
class QueuedPrompt:
    """A prompt deferred until the current ACP turn becomes idle."""

    text: str
    synthetic_notification: bool = False
    completion_sink: Any = None


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
    # Session this one was derived from (fork or clean-context ACP spawn).
    # Persisted via the legacy ``_forked_from`` model_config marker — NOT
    # sessions.parent_session_id, which list_sessions_rich treats as
    # subagent/compression lineage and would hide it from session lists.
    # Display-only.
    parent_id: Optional[str] = None
    # True for delegate-child (source="subagent") sessions restored for
    # observation. These are read-only from ACP's side: the delegate child
    # owns the row and transcript, so _persist must never write them.
    subagent: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: Any = None  # threading.Event
    is_running: bool = False
    # Serialize state-mutating slash commands against turn claims.
    command_op: bool = False
    # Plain strings are normalized by the drain for backward compatibility.
    queued_prompts: List[Any] = field(default_factory=list)
    runtime_lock: Any = field(default_factory=Lock)
    current_prompt_text: str = ""
    interrupted_prompt_text: str = ""
    # ``time.monotonic()`` stamp of the cancel that armed
    # ``interrupted_prompt_text``. The salvage feature exists for clients that
    # implement "stop and send" as cancel-then-submit, which is two protocol
    # calls milliseconds apart — so a buffer is only salvageable inside a short
    # window (see INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC). A zero stamp means
    # "never armed by a real cancel" and is never salvageable.
    interrupted_prompt_at: float = 0.0
    # True once the running turn has produced its final response. A cancel
    # arriving after that point has nothing to salvage: the request was already
    # answered and delivered, so re-attaching it to the NEXT prompt replays
    # finished work.
    response_delivered: bool = False
    # Stable ACP identity for streamed assistant chunks. Persist across turns
    # on the in-memory session state, but close at each turn boundary so a new
    # turn cannot replace the previous assistant bubble.
    message_ids: Any = None
    # Derived work is meaningful before its first model message (lineage,
    # route, cwd, and optional title must survive while the first turn runs).
    # Ordinary blank editor tabs remain ephemeral.
    persist_empty: bool = False


class SessionManager:
    """Thread-safe manager for ACP sessions backed by Hermes AIAgent instances.

    Sessions are held in-memory for fast access **and** persisted to the
    shared SessionDB so they survive process restarts and are searchable
    via ``session_search``.
    """

    def __init__(self, agent_factory=None, db=None):
        """``agent_factory``: AIAgent-like factory (tests); default builds a real AIAgent from
        the runtime provider config. ``db``: SessionDB; default lazily opens ``~/.hermes/state.db``."""
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        # Serializes DB restores: session construction runs off the event loop, so two
        # overlapping session/load for one id must share a single agent build.
        self._restore_lock = threading.Lock()
        self._agent_factory = agent_factory
        self._db_instance = db  # None → lazy-init on first use
        self._cwd_backfilled = False

    # ---- public API ---------------------------------------------------------

    def create_session(
        self,
        cwd: str = ".",
        owner: str | None = None,
        model: str | None = None,
        requested_provider: str | None = None,
        base_url: str | None = None,
        api_mode: str | None = None,
        parent_id: str | None = None,
    ) -> SessionState:
        """Create a new session with a unique ID and a fresh AIAgent.

        ``owner`` (an authenticated user identifier, e.g. the Cloudflare Access
        email supplied by the client on ``session/new``) is persisted to
        ``sessions.user_id`` so the ACP sessions list can show a user only
        their own sessions. It is a soft per-user display key, not an access
        boundary.

        ``model``/``requested_provider``/``base_url``/``api_mode`` let a
        caller pin the new agent's full route instead of the config default.
        A model name is meaningless without its routing, so callers carrying
        a model over from an existing session must pass the whole tuple —
        same invariant as ``fork_session``.

        ``parent_id`` records display-only derivation lineage via the existing
        ``_forked_from`` model-config marker. It does not copy history and does
        not use ``sessions.parent_session_id`` (reserved for hidden subagents).
        """
        import threading

        cwd = _translate_acp_cwd(cwd)
        session_id = str(uuid.uuid4())
        agent = self._make_agent(
            session_id=session_id,
            cwd=cwd,
            model=model,
            requested_provider=requested_provider,
            base_url=base_url,
            api_mode=api_mode,
        )
        state = self._install_state(
            session_id, agent, cwd, getattr(agent, "model", "") or "", [],
            owner=(owner or "").strip() or None,
            parent_id=parent_id,
            persist_empty=bool((parent_id or "").strip()),
        )
        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def _install_state(self, session_id: str, agent: Any, cwd: str, model: str,
                       history: list[dict[str, Any]], *, persist: bool = True,
                       **state_fields: Any) -> SessionState:
        """Install a session, including fork metadata, through the common lifecycle path."""
        state = SessionState(
            session_id=session_id, agent=agent, cwd=cwd, model=model,
            history=history, cancel_event=threading.Event(), **state_fields,
        )
        with self._lock:
            self._sessions[session_id] = state
        _register_task_cwd(session_id, cwd)
        if persist:
            self._persist(state)
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
        with self._restore_lock:
            with self._lock:
                state = self._sessions.get(session_id)  # a concurrent restore may have installed it
            return state if state is not None else self._restore(session_id)

    def get_in_memory(self, session_id: str) -> Optional[SessionState]:
        """Return the in-memory session for *session_id*, or ``None``.

        Unlike ``get_session``, this NEVER restores from the database — it
        answers only "does this process currently host that session?".
        Callers that use in-memory membership as an ownership boundary
        (e.g. deciding whether a completion_queue event belongs to this
        ACP server process) must use this so a lookup can't side-effect a
        DB restore and claim sessions owned by another process.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def live_transcript_history(
        self, session_id: str, repair_alternation: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """Best-effort transcript of a session from the persisted message store.

        Used by ACP history replay: while a turn is RUNNING here, the
        in-memory ``state.history`` only extends at turn end but ``run_agent``
        flushes messages to the DB continuously; while IDLE here, the session
        may have advanced in another process (Slack bot, gateway, CLI) whose
        turns this process never observes. Either way the DB is the faithful
        transcript. ``repair_alternation`` is for callers that adopt the
        result as live conversation state (see
        ``HermesState.get_messages_as_conversation``). Returns None when the
        store is unavailable or the query fails; the caller falls back to
        ``state.history``.
        """
        db = self._get_db()
        if db is None:
            return None
        with self._lock:
            state = self._sessions.get(session_id)
        try:
            head_id = session_id
            if state is not None:
                sid = getattr(state.agent, "session_id", None)
                if isinstance(sid, str) and sid:
                    head_id = sid
            return db.get_messages_as_conversation(
                head_id, repair_alternation=repair_alternation
            )
        except Exception:
            logger.warning(
                "Failed to load live transcript for ACP session %s",
                session_id,
                exc_info=True,
            )
            return None

    def persisted_message_count(self, session_id: str) -> Optional[int]:
        """Persisted message count of the session's live head, or None.

        Same head resolution as ``live_transcript_history`` so the value
        matches what a replay of this session just rendered.
        """
        db = self._get_db()
        if db is None:
            return None
        with self._lock:
            state = self._sessions.get(session_id)
        head_id = session_id
        if state is not None:
            sid = getattr(state.agent, "session_id", None)
            if isinstance(sid, str) and sid:
                head_id = sid
        try:
            # session/list projects a compression root onto its live tip's
            # message_count; resolve the same tip here so a compaction in
            # another process cannot leave the two counts permanently apart
            # (which would make the client re-replay on every list poll).
            chain = db.get_compression_chain(head_id) or [head_id]
            row = db.get_session(chain[-1])
        except Exception:
            logger.debug("Failed to read message count for %s", head_id, exc_info=True)
            return None
        if not row or row.get("message_count") is None:
            return None
        return int(row["message_count"])

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
            # Prefer the requested (pre-canonicalization) identity: a named
            # custom endpoint's ``provider`` is bare ``custom``, which would
            # re-resolve to the generic env/config custom provider
            # (OpenRouter) instead of the parent's actual endpoint.
            requested_provider=(
                _agent_provider_identity(parent_agent)
                or getattr(parent_agent, "provider", None)
            ),
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
            persist_empty=True,
        )
        _apply_effort_to_agent(agent, original.effort)
        with self._lock:
            self._sessions[new_id] = state
        _register_task_cwd(new_id, cwd)
        self._persist(state)
        # Stamp the fork's title from the parent's canonical title ("Parent
        # Title #2") — otherwise the fork row falls back to the first-user-
        # message preview, which is identical to the parent's and useless for
        # telling forks apart. Untitled parents leave the fork untitled (the
        # background derive backfill handles those).
        parent_title = self.get_session_title(session_id)
        if parent_title:
            db = self._get_db()
            if db is not None:
                try:
                    db.set_session_title(
                        new_id, db.get_next_title_in_lineage(parent_title)
                    )
                except ValueError:
                    logger.debug(
                        "fork_session: could not stamp lineage title for %s",
                        new_id,
                        exc_info=True,
                    )
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
                # Always fetch archived rows and filter per-row below. An
                # archived session can still be live in memory (e.g. a fork
                # archived right after creation), and the in-memory branch
                # needs its persisted archived flag to honor the filter —
                # a DB-side filter would just drop the row from this dict.
                for row in db.list_sessions_rich(
                    source="acp",
                    limit=1000,
                    include_archived=True,
                    owner=owner_filter,
                ):
                    persisted_rows[str(row["id"])] = dict(row)
            except Exception:
                logger.debug("Failed to load ACP sessions from DB", exc_info=True)

        cwd_visible_ids: set[str] = set()
        if normalized_cwd:
            def _row_cwd(row: dict[str, Any]) -> str:
                model_config = row.get("model_config")
                if model_config:
                    try:
                        return json.loads(model_config).get("cwd", ".")
                    except (json.JSONDecodeError, TypeError):
                        pass
                return row.get("cwd") or "."

            cwd_visible_ids = {
                sid
                for sid, row in persisted_rows.items()
                if _normalize_cwd_for_compare(_row_cwd(row)) == normalized_cwd
            }
            # Derived ACP sessions stay in the visible conversation family
            # even when they execute in another worktree. This matches the
            # delegate-child contract below and keeps acp_spawn_session output
            # next to its visible parent in workspace-filtered clients.
            changed = True
            while changed:
                changed = False
                for sid, row in persisted_rows.items():
                    if sid in cwd_visible_ids:
                        continue
                    parent_id = _forked_from_marker(row.get("model_config"))
                    if parent_id and parent_id in cwd_visible_ids:
                        cwd_visible_ids.add(sid)
                        changed = True

        def _admits(row_archived: bool) -> bool:
            if archived_only:
                return row_archived
            return include_archived or not row_archived

        # Collect in-memory sessions first.
        with self._lock:
            seen_ids = set(self._sessions.keys())
            results = []
            for s in self._sessions.values():
                if s.subagent:
                    # Delegate-child mirror restored for observation (e.g. the
                    # user opened a subagent transcript). The DB row is
                    # authoritative for these — the subagent branch below lists
                    # them with their real last_active and subagent flag.
                    # Listing them here instead fabricated updated_at=now every
                    # call (their source='subagent' row is absent from the
                    # acp-only persisted fetch, so the wall-clock fallback
                    # fired) and dropped the subagent marker, so hours-old
                    # children pinned to the top of clients' recency sorts for
                    # as long as the worker stayed alive.
                    seen_ids.discard(s.session_id)
                    continue
                history_len = len(s.history)
                if history_len <= 0:
                    # In-memory history is only assigned when a turn FINISHES,
                    # so a spawned/queued session mid-first-turn sits here with
                    # an empty list even though the agent has already flushed
                    # real messages to the DB incrementally. Un-claim the id so
                    # the persisted merge below can surface the DB row (it
                    # still hides truly-empty sessions via message_count <= 0);
                    # otherwise the session is invisible in session/list for
                    # the entire duration of its first turn.
                    seen_ids.discard(s.session_id)
                    continue
                persisted = persisted_rows.get(s.session_id, {})
                # Archived state lives only in the DB row; an in-memory session
                # (e.g. a just-archived fork still resident in the process) must
                # honor it or archiving looks like a no-op in the client.
                row_archived = bool(persisted.get("archived"))
                if not _admits(row_archived):
                    continue
                if normalized_cwd and s.session_id not in cwd_visible_ids:
                    continue
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
                # A compaction summary is not the user's first message: after
                # a compaction it heads the live history and would otherwise
                # become the fallback title.
                preview = next(
                    (
                        _preview_text(msg.get("content"))
                        for msg in s.history
                        if msg.get("role") == "user"
                        and not is_compaction_summary_message(msg)
                        and _preview_text(msg.get("content"))
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
                        # No canonical title yet — candidate for background
                        # title backfill (see HermesACPAgent.list_sessions).
                        "untitled": not str(persisted.get("title") or "").strip(),
                        # Order by LAST ACTIVITY of ANY source — the newest
                        # message on the transcript regardless of role. A long
                        # turn that finally lands assistant/tool messages IS
                        # activity: the row must resort to the top when the turn
                        # ends, not stay frozen at the user's send time (that
                        # made a 40-minute turn look older than it was and
                        # buried the session the user was waiting on).
                        "updated_at": _format_updated_at(
                            persisted.get("last_active")
                            or persisted.get("started_at")
                            or time.time()
                        ),
                        "archived": row_archived,
                        "parent_id": parent_id,
                        # A fork of a Slack session is an ACP/VS Code session in
                        # its own right — don't inherit the Slack badge.
                        # The persisted preview is the durable first user
                        # message; the live history loses the Slack envelope
                        # once compaction replaces the head with a summary.
                        "slack": not parent_id
                        and (
                            _is_slack_preview(persisted.get("preview"))
                            or _is_slack_preview(preview)
                        ),
                        **_progress_fields(persisted),
                    }
                )

        # Merge any persisted sessions not currently in memory.
        for sid, row in persisted_rows.items():
            if sid in seen_ids:
                continue
            row_archived = bool(row.get("archived"))
            if not _admits(row_archived):
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
            if normalized_cwd and sid not in cwd_visible_ids:
                continue
            parent_id = _forked_from_marker(mc)
            results.append({
                "session_id": sid,
                "cwd": session_cwd,
                "model": row.get("model") or "",
                "history_len": message_count,
                "user_id": row.get("user_id") or "",
                "title": _build_session_title(row.get("title"), row.get("preview"), session_cwd),
                "untitled": not str(row.get("title") or "").strip(),
                # Last activity of any role wins for recency (see the in-memory
                # branch).
                "updated_at": _format_updated_at(
                    row.get("last_active") or row.get("started_at")
                ),
                "archived": row_archived,
                "parent_id": parent_id,
                # Forks of Slack sessions are not Slack sessions — no badge.
                "slack": not parent_id and _is_slack_preview(row.get("preview")),
                **_progress_fields(row),
            })

        results.sort(key=lambda item: _updated_at_sort_key(item.get("updated_at")), reverse=True)

        # Surface delegate children (source="subagent") of visible ACP
        # sessions so clients can nest live/completed subagent transcripts
        # under their parent. Children of non-ACP parents (CLI/cron
        # delegations) are excluded — their parents aren't in this list, so
        # the rows would render as orphans. include_children=True is required
        # to defeat list_sessions_rich's default child/delegate filtering.
        if db is not None and not archived_only:
            acp_ids = {item["session_id"] for item in results}
            try:
                subagent_rows = db.list_sessions_rich(
                    source="subagent",
                    limit=1000,
                    include_children=True,
                    include_archived=include_archived,
                    min_message_count=1,
                )
            except Exception:
                subagent_rows = []
                logger.debug("Failed to load subagent sessions from DB", exc_info=True)
            children = []
            for row in subagent_rows:
                parent_sid = str(row.get("parent_session_id") or "")
                if parent_sid not in acp_ids:
                    continue
                sid = str(row["id"])
                if sid in acp_ids:
                    continue
                children.append({
                    "session_id": sid,
                    # Children run in their own terminal-session dirs; they
                    # inherit list visibility via the parent, not their own
                    # cwd, so no cwd filter applies here.
                    "cwd": row.get("cwd") or ".",
                    "model": row.get("model") or "",
                    "history_len": int(row.get("message_count") or 0),
                    "user_id": row.get("user_id") or "",
                    "title": _build_session_title(
                        row.get("title"), row.get("preview"), row.get("cwd") or "."
                    ),
                    "updated_at": _format_updated_at(
                        row.get("last_active") or row.get("started_at")
                    ),
                    "archived": bool(row.get("archived")),
                    "parent_id": parent_sid,
                    "subagent": True,
                    "slack": False,
                    **_progress_fields(row),
                })
            if children:
                results.extend(children)
                results.sort(
                    key=lambda item: _updated_at_sort_key(item.get("updated_at")),
                    reverse=True,
                )
        return results

    def update_cwd(
        self,
        session_id: str,
        cwd: str,
        *,
        persist: bool = True,
    ) -> Optional[SessionState]:
        """Update the working directory for a session and its tool overrides."""
        cwd = _translate_acp_cwd(cwd)
        state = self.get_session(session_id)  # checks DB too
        if state is None:
            return None
        state.cwd = cwd
        _register_task_cwd(session_id, cwd)
        if persist:
            # Persist only metadata here. Rewriting the full transcript can
            # take tens of seconds and creates inactive duplicate rows.
            self._persist(state, persist_messages=False)
        if persist and not state.subagent:
            self._schedule_git_metadata(state, self._claim_cwd_generation(state))
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

    def end_all_sessions(self, end_reason: str = "acp_disconnect") -> int:
        """Stamp ``ended_at`` on every live session (#118216).

        ACP v0.9 has no per-session destroy, so the stdio shutdown that ends
        this process is the session end: the client that drove the
        conversation is gone. Without this writer, source='acp' rows keep
        ``ended_at`` NULL forever and the ended-session guard shared by
        prune/archive (``hermes_state_maintenance``) can never reach them.
        A later load/resume reopens the row (see ``_restore``), the same
        contract the TUI gateway's resume path uses. Best-effort: teardown
        must never raise. Returns the number of sessions ended.
        """
        db = self._get_db()
        if db is None:
            return 0
        with self._lock:
            # Delegate mirrors are observation-only: their owning worker, not
            # this ACP transport, determines when they end.
            session_ids = [sid for sid, state in self._sessions.items() if not state.subagent]
        ended = 0
        for session_id in session_ids:
            try:
                db.end_session(session_id, end_reason)
                ended += 1
            except Exception:
                logger.debug("Failed to end ACP session %s", session_id, exc_info=True)
        if ended:
            logger.info("Ended %d ACP session(s) on shutdown (%s)", ended, end_reason)
        return ended

    # ---- persistence via SessionDB ------------------------------------------

    def _get_db(self):
        """Lazily acquire the process-shared SessionDB; ``None`` if unavailable (e.g. import
        error in a minimal test env). ``HERMES_HOME`` is resolved here, not via the import-time
        ``DEFAULT_DB_PATH``, so test fixtures that change the env var later are honoured. The
        registry handle is the one in-process tools (delegation, session_search, goals) also
        acquire, so the ACP server holds ONE writer on state.db instead of two (#100896)."""
        if self._db_instance is None:
            try:
                from hermes_state_registry import acquire
                self._db_instance = acquire(get_hermes_home() / "state.db")
            except Exception:
                logger.debug("SessionDB unavailable for ACP persistence", exc_info=True)
        if self._db_instance is not None and not self._cwd_backfilled:
            # Rows minted before the adapter wrote the cwd column still carry the workspace
            # in model_config; one idempotent UPDATE per process repairs them (#115705).
            self._cwd_backfilled = True
            try:
                repaired = self._db_instance.backfill_acp_session_cwd()
                if repaired:
                    logger.info("Backfilled cwd for %d ACP session(s) from model_config", repaired)
            except Exception:
                logger.debug("ACP session cwd backfill failed", exc_info=True)
        return self._db_instance

    def _persist(self, state: SessionState, *, persist_messages: bool = True) -> None:
        """Write session state to the database.

        Create the session record if needed and update its metadata. Replace
        stored messages only when ``persist_messages`` is true.
        """
        # Delegate children are observation-only: the child agent owns the
        # row and flushes its own transcript. Persisting here would rewrite
        # model_config (dropping the _delegate_from marker) and the messages.
        if getattr(state, "subagent", False):
            return
        db = self._get_db()
        if db is None:
            return

        # Ensure model is a plain string (not a MagicMock or other proxy).
        model_str = str(state.model) if state.model else None
        session_meta = {"cwd": state.cwd}
        # Persist the requested (pre-canonicalization) provider identity when
        # available: a named custom endpoint canonicalizes to bare ``custom``,
        # and restoring from bare ``custom`` resolves the generic env/config
        # custom provider (default base_url: OpenRouter) instead of the named
        # endpoint the user picked.
        provider = _agent_provider_identity(state.agent) or getattr(
            state.agent, "provider", None
        )
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
                # Fresh editor probes are in-memory drafts. Avoid durable ghost
                # sessions until transcript content exists; copied forks still
                # persist immediately because their history is non-empty.
                if not state.history and not state.persist_empty:
                    return
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
                    cwd=state.cwd,
                )
            else:
                try:
                    db.update_session_meta(
                        state.session_id, cwd_json, model_str, cwd=state.cwd
                    )
                except Exception:
                    logger.debug("Failed to update ACP session metadata", exc_info=True)
                # session/new carries the authenticated owner, but row creation
                # is deferred until the session has history — by which time the
                # agent's _ensure_db_session has usually created the row, and it
                # knows nothing about the ACP owner. Without this backfill the
                # create-time user_id write above never runs for an ordinary
                # conversation and every session persists untagged, hidden
                # behind the strict "My Sessions" owner filter.
                #
                # Fill a gap only: an owner already on the row wins, so this can
                # never transfer ownership of a restored or re-attached session.
                owner = str(getattr(state, "owner", "") or "").strip()
                if owner and not str(existing.get("user_id") or "").strip():
                    try:
                        db.set_session_owner(state.session_id, owner)
                    except Exception:
                        logger.debug(
                            "Failed to backfill ACP session owner", exc_info=True
                        )
                # The create branch above is not the live path: an agent that owns
                # persistence to this same DB flushes the transcript incrementally,
                # so the row already exists by the time we get here.
                # update_session_meta touches only model_config/model, so without
                # this promotion the column stays NULL for the whole session and
                # Desktop files it as unassigned. Claiming a generation keeps the
                # A -> B -> A ordering contract shared with update_cwd().
                if state.cwd:
                    self._schedule_git_metadata(state, self._claim_cwd_generation(state))

            if not persist_messages:
                return

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
                # like the owned-agent case. Guard against it by replacing ONLY
                # the live (active=1) set unconditionally: on a fresh
                # create/fork every row is active=1, so active-only replace is
                # behaviorally identical to the full replace — and when archived
                # rows DO exist they survive. An existence probe here
                # (has_archived_messages) would fail OPEN into the destructive
                # replace on any DB error and can race a concurrent
                # archive_and_compact — the same probe failure mode #80216's
                # /retry fix (gateway/slash_commands.py) deliberately avoids.
                replace_messages = getattr(db, "replace_messages", None)
                if callable(replace_messages):
                    replace_messages(
                        state.session_id, state.history, active_only=True
                    )
        except Exception:
            logger.warning("Failed to persist ACP session %s", state.session_id, exc_info=True)

    def _claim_cwd_generation(self, state: SessionState) -> Optional[int]:
        """Write the cwd column and return its new git-metadata generation.

        Returns None when the row does not exist yet (a contentless session is
        deliberately ephemeral until it has history) or the DB is unavailable.
        """
        db = self._get_db()
        if db is None or not state.cwd:
            return None
        try:
            return db.update_session_cwd(state.session_id, state.cwd)
        except Exception:
            logger.debug("Failed to persist ACP session cwd column for %s",
                         state.session_id, exc_info=True)
            return None

    def _schedule_git_metadata(self, state: SessionState, generation: Optional[int]) -> None:
        """Probe git off the critical path and publish under ``generation``.

        ``session/new`` is on the editor's interactive path; ``git rev-parse``
        on a cold or networked filesystem is not something to put in front of
        the user. The generation guard in ``publish_session_git_metadata``
        means a slow probe for a previous workspace is dropped rather than
        applied to the new one.
        """
        if not generation or not state.cwd:
            return

        session_id, cwd = state.session_id, state.cwd

        def _run() -> None:
            try:
                from tui_gateway import git_probe
                branch, root = git_probe.branch(cwd), git_probe.common_repo_root(cwd)
                if not (branch or root):
                    return
                db = self._get_db()
                if db is not None:
                    db.publish_session_git_metadata(session_id, cwd, generation, branch, root)
            except Exception:
                logger.debug("Failed to publish ACP git metadata for %s", session_id, exc_info=True)

        threading.Thread(target=_run, name=f"acp-git-meta-{session_id[:8]}", daemon=True).start()

    def _restore(self, session_id: str) -> Optional[SessionState]:
        """Load a session from the database into memory, recreating the AIAgent."""
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

        # Restore ACP sessions, plus delegate children (source="subagent")
        # for read-only observation — their transcript is flushed
        # incrementally by the child agent, so a DB restore shows a live
        # child's progress too. Anything else (cli/gateway/cron) stays
        # non-loadable through ACP.
        source = row.get("source")
        if source not in ("acp", "subagent"):
            return None
        is_subagent = source == "subagent"

        # A previous adapter process stamped the row ended at its stdio
        # shutdown (#118216); resuming the conversation reopens it, the same
        # contract the TUI gateway's cold-resume path uses.
        if not is_subagent and row.get("ended_at") is not None:
            try:
                db.reopen_session(session_id)
            except Exception:
                logger.debug("Failed to reopen ACP session %s", session_id, exc_info=True)


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

        # Load conversation history. repair_alternation: this restore feeds
        # LIVE REPLAY — the loaded list becomes the resumed agent's working
        # conversation. A durable ``user;user`` violation left in state.db would
        # otherwise re-fire the pre-request defensive repair on every request
        # for the rest of the session (see hermes_state.get_messages_as_conversation).
        try:
            history = db.get_messages_as_conversation(
                session_id, repair_alternation=not is_subagent
            )
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

        _apply_effort_to_agent(agent, restored_effort)
        state = self._install_state(
            session_id, agent, cwd, model or getattr(agent, "model", "") or "", history,
            persist=False,
            mode=restored_mode,
            effort=restored_effort,
            # Rehydrate the owner so forks of a restored session inherit it
            # (and a later _persist doesn't drop it).
            owner=str(row.get("user_id") or "").strip() or None,
            # Delegate children link to their parent via the DB column (not
            # the _forked_from marker forks use).
            parent_id=(
                (str(row.get("parent_session_id") or "").strip() or None)
                if is_subagent
                else (restored_parent_id or None)
            ),
            subagent=is_subagent,
        )
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

    def promote_session(self, session_id: str) -> bool:
        """Detach an idle ACP fork from its display parent.

        Fork lineage is display-only and lives in ``model_config._forked_from``.
        Removing that marker makes the session a normal top-level ACP session
        without changing its transcript, owner, title, or runtime routing.
        Delegate children remain read-only subagents and cannot be promoted by
        this path.
        """
        db = self._get_db()
        if db is None:
            return False
        row = db.get_session(session_id)
        if row is None:
            return False
        if row.get("source") != "acp":
            raise ValueError("Only ACP fork sessions can be promoted")
        if not _forked_from_marker(row.get("model_config")):
            return False

        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                if state.is_running:
                    raise ValueError("Wait for the session turn to finish before promoting it")
                state.parent_id = None

        db.patch_session_model_config(session_id, {"_forked_from": None})
        updated = db.get_session(session_id)
        return updated is not None and not _forked_from_marker(updated.get("model_config"))

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

        first_user = next(
            (t for m in messages if m.get("role") == "user" and (t := _text(m).strip())), ""
        )
        # Owned sessions carry the authenticated-owner note in their first user
        # message — title from the actual prompt, not the identity envelope.
        first_user = strip_owner_note(first_user)
        # First assistant message WITH text: turns that open with tool calls
        # persist content-less assistant rows first.
        first_assistant = next(
            (t for m in messages if m.get("role") == "assistant" and (t := _text(m).strip())), ""
        )
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
            # Derived-title collision — near-identical first prompts (e.g. two
            # handoff sessions) title identically. Suffix like fork lineage.
            try:
                suffixed = db.get_next_title_in_lineage(title)
                if db.set_session_title(session_id, suffixed):
                    return suffixed
            except ValueError:
                logger.debug(
                    "derive_session_title: title collision for %s", session_id, exc_info=True
                )
        return None

    # ---- internal -----------------------------------------------------------

    def _make_agent(self, *, session_id: str, cwd: str, model: str | None = None,
                    requested_provider: str | None = None, base_url: str | None = None, api_mode: str | None = None,
                    enabled_toolsets: list[str] | None = None, disabled_toolsets: list[str] | None = None):
        """``enabled_toolsets``/``disabled_toolsets`` carry a live session's toolsets into a rebuild; ``None`` derives
        them from config (fresh session)."""
        if self._agent_factory is not None:
            return self._agent_factory()

        from run_agent import AIAgent
        from agent.skill_utils import parse_config_string_list
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.tools_config import _get_platform_tools, enabled_mcp_server_names
        from hermes_constants import resolve_reasoning_config

        config = load_config()
        model_cfg = config.get("model")
        default_model = ""
        config_provider = None
        if isinstance(model_cfg, dict):
            default_model = str(model_cfg.get("default") or default_model)
            config_provider = model_cfg.get("provider")
        elif isinstance(model_cfg, str) and model_cfg.strip():
            default_model = model_cfg.strip()

        if enabled_toolsets is None:
            # The same per-platform resolver as the gateway/cron/api_server: platform_toolsets.acp wins, else
            # hermes-acp; its MCP half (every enabled server, a listed-name allowlist, or none for ``no_mcp``)
            # comes back as bare server names, which ACP keys as ``mcp-<server>`` like its session servers.
            resolved = _get_platform_tools(config, "acp")
            mcp_servers = resolved & enabled_mcp_server_names(config)
            enabled_toolsets = _expand_acp_enabled_toolsets(sorted(resolved - mcp_servers), sorted(mcp_servers))
        kwargs = {
            "platform": "acp", "quiet_mode": True, "session_id": session_id, "session_db": self._get_db(),
            "enabled_toolsets": list(enabled_toolsets),
            # agent.disabled_toolsets is subtracted at tool granularity by the agent, as on the CLI/gateway/cron.
            "disabled_toolsets": (list(disabled_toolsets) if disabled_toolsets is not None
                                  else [name.strip() for name in parse_config_string_list(
                                      (config.get("agent") or {}).get("disabled_toolsets")) if name.strip()] or None),
            "model": model or default_model,
            "cwd": cwd,
            # Same chokepoint as the CLI/gateway/TUI/cron: without it ``agent.reasoning_effort: none`` never
            # reaches an ACP session and the transport applies its default effort (a 400 on non-reasoning
            # models). Resolved against the session's model so per-model overrides apply.
            "reasoning_config": resolve_reasoning_config(config, model or default_model),
        }

        # Honor agent.max_turns on the ACP surface, matching the CLI
        # (cli_agent_setup_mixin passes max_iterations=self.max_turns).
        # Unset/invalid → omit so AIAgent's own default applies.
        try:
            max_turns = int((config.get("agent") or {}).get("max_turns") or 0)
        except (TypeError, ValueError):
            max_turns = 0
        if max_turns > 0:
            kwargs["max_iterations"] = max_turns

        resolve_error: Exception | None = None
        try:
            runtime = resolve_runtime_provider(
                requested=requested_provider or config_provider, target_model=(model or default_model) or None)
            kwargs.update({
                "provider": runtime.get("provider"), "requested_provider": runtime.get("requested_provider"), "api_mode": api_mode or runtime.get("api_mode"),
                "base_url": base_url or runtime.get("base_url"), "api_key": runtime.get("api_key"),
                "credential_pool": runtime.get("credential_pool"),
                "command": runtime.get("command"), "args": list(runtime.get("args") or []),
            })
            # The resolved provider's request body (a custom entry's extra_body); an explicit base_url pointing
            # elsewhere is another endpoint, which must not inherit it.
            if runtime.get("request_overrides") and (not base_url or base_url == runtime.get("base_url")):
                kwargs["request_overrides"] = runtime["request_overrides"]
        except Exception as exc:
            resolve_error = exc
            logger.debug("ACP session falling back to default provider resolution", exc_info=True)

        _register_task_cwd(session_id, cwd)

        # Bounded wait for background MCP discovery so already-spawning fast
        # servers land in the agent's tool snapshot.  ACP entry.py fires
        # discovery in a background daemon thread (start_background_mcp_discovery);
        # the agent snapshots tools once at build (run_agent/agent_init) and
        # never re-reads the registry, so without this join a reachable-but-
        # slow configured server would be invisible for the whole session.
        # ``ensure_mcp_discovery_before_agent_build`` also (re)starts discovery
        # when the entry.py spawn never ran or exited with zero connected
        # servers (the retry-after-zero-connected allowance), making this
        # construction site self-sufficient.  Bounded by
        # ``mcp_discovery_timeout`` (config.yaml, default ~1.5s) so a dead
        # server can't block — servers that miss the bound are picked up by
        # the automatic late-refresh (see HermesACPAgent._schedule_mcp_late_refresh).
        try:
            from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build

            ensure_mcp_discovery_before_agent_build(
                logger=logger,
                thread_name="acp-mcp-discovery",
            )
        except Exception:
            logger.debug("ACP: bounded MCP discovery wait failed", exc_info=True)

        try:
            agent = AIAgent(**kwargs)
        except Exception as exc:
            # The bare-AIAgent fallback dies with "No LLM provider configured. Run `hermes setup`" on a
            # machine that is configured and was working a call earlier; the swallowed resolution
            # failure (revoked OAuth, disabled provider, ...) is the actionable error (#91090).
            if resolve_error is not None:
                raise resolve_error from exc
            raise
        # ACP stdio: stdout is protocol-only JSON-RPC; agent chatter goes to stderr.
        agent.session_cwd = cwd
        agent._print_fn = _acp_stderr_print
        return agent
