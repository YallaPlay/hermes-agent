"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, UTC
import contextlib
import contextvars
import logging
import json
import time
import base64
import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import acp
from acp import RequestError
from acp.schema import (
    AgentCapabilities, AgentMessageChunk, AuthenticateResponse, ClientCapabilities, ForkSessionResponse,
    Implementation, InitializeResponse, ListSessionsResponse, LoadSessionResponse, McpServerHttp, McpServerSse,
    McpServerStdio, ModelInfo, NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionForkCapabilities, SessionInfo, SessionInfoUpdate, SessionListCapabilities,
    SessionConfigOptionSelect, SessionConfigSelectOption, ImageContentBlock, AgentThoughtChunk,
    SessionMode, SessionModeState, SessionModelState, SessionResumeCapabilities, SetSessionConfigOptionResponse,
    SetSessionModeResponse, SetSessionModelResponse, TextContentBlock, Usage, UsageUpdate, UserMessageChunk,
)

from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID, build_auth_methods, detect_provider
from acp_adapter.commands import SlashCommandsMixin, _estimate_tokens
from acp_adapter.content import _MAX_ACP_RESOURCE_BYTES, PromptBlock, _content_blocks_to_openai_user_content, _extract_text
from acp_adapter.events import (
    AssistantMessageIdAllocator, _build_plan_update_from_todo_result, _send_update, flush_open_tool_calls,
    make_message_cb, make_step_cb, make_thinking_cb, make_tool_progress_cb, make_subagent_update_router,
)
from acp_adapter.model_catalog import build_model_state, encode_model_choice
from acp_adapter.permissions import make_approval_callback
from acp_adapter.provenance import session_provenance_meta
from acp_adapter.session import (SessionManager, SessionState, QueuedPrompt,
    _agent_provider_identity, _apply_effort_to_agent, _expand_acp_enabled_toolsets)
from acp_adapter.tools import build_tool_complete, build_tool_start
from agent.context_compressor import (COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor)
from agent.history_media import image_bytes_from_ref, image_refs_from_message, inline_image_bytes_from_content
from agent.interrupt_compat import request_hard_interrupt
from tools.approval_context import reset_hermes_interactive_context, set_hermes_interactive_context

logger = logging.getLogger(__name__)

# Runs the synchronous AIAgent off the event loop.
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="acp-agent")

# ListSessionsRequest has no client-side limit; clients paginate via `cursor`/`next_cursor`.
_LIST_SESSIONS_PAGE_SIZE = 40


def _mcp_server_config(server: McpServerStdio | McpServerHttp | McpServerSse) -> dict:
    if isinstance(server, McpServerStdio):
        return {"command": server.command, "args": list(server.args), "env": {i.name: i.value for i in server.env}}
    return {"url": server.url, "headers": {i.name: i.value for i in server.headers}}


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _bind_guarded(stack: contextlib.ExitStack, label: str, setup: Callable[[], Callable[[], None]]) -> None:
    """Run ``setup`` (returns its teardown) and register the teardown; failures in either half only
    log — the turn must still run without the binding."""
    try:
        teardown = setup()
    except Exception:
        logger.debug("Could not set ACP %s", label, exc_info=True)
        return

    def _teardown() -> None:
        try:
            teardown()
        except Exception:
            logger.debug("Could not restore ACP %s", label, exc_info=True)

    stack.callback(_teardown)


class ModelRejected(ValueError):
    """``switch_model`` refused the requested model (no provider can serve it)."""


@dataclass
class _TurnCallbacks:
    """Per-turn ACP streaming callbacks; all None when no client is connected."""

    tool_progress_cb: Any = None
    reasoning_cb: Any = None
    step_cb: Any = None
    stream_delta_cb: Any = None
    approval_cb: Any = None
    edit_approval_requester: Any = None
    streamed: bool = False
    tool_call_ids: Any = None
    tool_call_meta: Any = None
    subagent_router: Any = None



_ACP_SPAWN_RESULT_MAX_CHARS = 100_000
INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC = 30.0

@dataclass
class _TurnCompletionSink:
    """Exactly-once callback retained with one prompt across queueing."""

    callback: Any
    delivery_emitted: bool = False
    queued: bool = False

    def __call__(
        self,
        final_response: str,
        interrupted: bool,
        stop_reason: str,
        error: str | None = None,
    ) -> None:
        if self.delivery_emitted:
            return
        self.delivery_emitted = True
        try:
            self.callback(final_response, interrupted, stop_reason, error)
        except Exception:
            logger.exception("ACP spawned-turn completion sink failed")


def _fail_queued_completion_sinks(
    entries: list[Any],
    *,
    interrupted: bool,
    error: str,
) -> None:
    """Resolve completion sinks on queued prompts that will not be drained."""
    for entry in entries:
        if not isinstance(entry, QueuedPrompt):
            continue
        sink = entry.completion_sink
        if not isinstance(sink, _TurnCompletionSink) or sink.delivery_emitted:
            continue
        sink.queued = False
        sink(
            "",
            interrupted,
            "cancelled" if interrupted else "failed",
            error,
        )


def _named_slug_for_base_url(base_url: str | None) -> str | None:
    """Map a live agent ``base_url`` back to its named custom-provider slug.

    Recovery path for agents that only know the canonical ``custom`` provider
    (sessions created before ``requested_provider`` was plumbed through, or
    metadata written by older builds). Encoding such a session's current model
    as bare ``custom:<model>`` is a routing hazard: clients cache that id and
    re-apply it to new sessions, where ``resolve_runtime_provider("custom")``
    resolves the *generic* env/config custom provider — default base_url
    ``https://openrouter.ai/api/v1`` — silently rerouting traffic away from
    the named endpoint the user picked.
    """
    url = str(base_url or "").strip().rstrip("/").lower()
    if not url:
        return None
    try:
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return None
    try:
        for entry in get_compatible_custom_providers():
            if not isinstance(entry, dict):
                continue
            entry_url = str(entry.get("base_url", "") or "").strip().rstrip("/").lower()
            if not entry_url or entry_url != url:
                continue
            name = str(entry.get("name", "") or "").strip()
            if not name:
                continue
            provider_key = str(entry.get("provider_key", "") or "").strip()
            return custom_provider_slug(name, provider_key)
    except Exception:
        logger.debug("Could not map base_url to a named custom provider", exc_info=True)
    return None


def _consume_interrupted_prompt(state: Any) -> str:
    """Pop the salvageable interrupted prompt, or "" if there is none.

    Always clears the buffer, including when it is refused for being stale —
    a buffer that outlived its window is dead and must not linger to ambush a
    later prompt. Caller must NOT hold ``state.runtime_lock``.
    """
    with state.runtime_lock:
        text = state.interrupted_prompt_text
        armed_at = state.interrupted_prompt_at
        state.interrupted_prompt_text = ""
        state.interrupted_prompt_at = 0.0
    if not text:
        return ""
    if not armed_at:
        # Never armed by a real cancel (or armed by an older build that did
        # not stamp). Not salvageable — fail closed.
        return ""
    age = time.monotonic() - armed_at
    if age > INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC:
        logger.info(
            "Dropping stale interrupted prompt (%.1fs old, window %.1fs): %s",
            age,
            INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC,
            text[:100],
        )
        return ""
    return text


def _merge_interrupted_prompt(new_text: str, interrupted_prompt: str) -> str:
    """Combine a salvaged interrupted prompt with the user's new message.

    The NEW message leads: it is the live instruction, and the interrupted one
    is background that exists so deictic follow-ups ("not that file") still
    have a referent. Putting the interrupted request first made models read
    the abandoned task as the directive and the user's actual message as an
    afterthought.
    """
    return (
        f"{new_text}\n\n"
        "[Context — your previous request on this session was interrupted "
        "before it finished. It is included below for reference only, so "
        "references like \"that file\" resolve. It is NOT a new instruction; "
        "the message above is what to act on.]\n"
        f"{interrupted_prompt}"
    )


def _group_session_families(
    infos: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group a recency-sorted session list into fork/subagent families.

    A family is a root session plus every descendant reachable through
    ``parent_id`` links WITHIN the list — a child whose parent is absent
    (archived, filtered out) roots its own family, mirroring how clients
    nest lineage. Families keep the order of their freshest member (first
    appearance in the recency-sorted input) and members keep input order.
    Used by ``list_sessions`` to paginate on family boundaries so a parent
    and its children never land on different pages.
    """
    by_id = {s["session_id"]: s for s in infos}

    def _root_of(s: dict[str, Any]) -> str:
        seen: set[str] = set()
        cur = s
        while True:
            sid = cur["session_id"]
            if sid in seen:  # lineage cycle guard
                break
            seen.add(sid)
            parent = cur.get("parent_id")
            if not parent or parent == sid or parent not in by_id:
                break
            cur = by_id[parent]
        return cur["session_id"]

    families: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for s in infos:
        root = _root_of(s)
        if root not in families:
            families[root] = []
            order.append(root)
        families[root].append(s)
    return [families[root] for root in order]


def _apply_session_effort(agent: Any, level: str) -> None:
    """Apply a reasoning-effort level to the LIVE agent.

    ``agent.reasoning_config`` is read on every API call, so this takes effect
    on the next call (same mechanism as the reasoning_effort tool).
    "default" restores the provider default by clearing the override to None —
    unlike the restore-time helper in acp_adapter.session, which can simply
    skip touching a fresh agent.
    """
    try:
        if str(level or "").strip().lower() == "default":
            agent.reasoning_config = None
            return
        from hermes_constants import parse_reasoning_effort

        parsed = parse_reasoning_effort(level)
        if parsed is not None:
            agent.reasoning_config = parsed
    except Exception:
        logger.debug("Could not apply reasoning effort %r", level, exc_info=True)


def _prompt_has_content(user_text: str, user_content: str | list[dict[str, Any]]) -> bool:
    """True when the converted prompt carries anything worth a turn.

    ``user_text`` only sees TextContentBlocks. A prompt that is ONLY a
    non-image attachment (e.g. a .docx uploaded from the composer with no
    caption) converts to a plain text STRING — the ``[Attached file: ...]``
    header plus inlined body or saved-to-cache note — so checking
    ``isinstance(user_content, list)`` alone silently dropped the turn
    (bitten 2026-09-03: .docx uploads vanished with no client feedback).
    Any non-blank converted content, string or structured, is a real prompt.
    Whitespace-only text (``"   "``) stays empty so a blank submit or a
    rewind-without-resend is still refused.
    """
    if user_text:
        return True
    if isinstance(user_content, str):
        return bool(user_content.strip())
    return bool(user_content)


class HermesACPAgent(SlashCommandsMixin, acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _EDIT_APPROVAL_POLICY_CONFIG_ID = "edit_approval_policy"
    _EDIT_APPROVAL_POLICY_DEFAULT = "ask"
    _MODE_DEFAULT = "default"
    # mode id -> (edit approval policy, display name, description)
    _MODES: dict[str, tuple[str, str, str]] = {
        "default": ("ask", "Default", "Ask before edits."),
        "accept_edits": (
            "workspace_session",
            "Accept Edits",
            "Auto-allow workspace and temp-dir edits; still asks for sensitive paths.",
        ),
        "dont_ask": (
            "session", "Don't Ask", "Auto-allow file edits for this session except sensitive paths."
        ),
    }
    _MODE_TO_EDIT_APPROVAL_POLICY = {mode: spec[0] for mode, spec in _MODES.items()}
    _EDIT_APPROVAL_POLICY_TO_MODE = {spec[0]: mode for mode, spec in _MODES.items()}

    # Seconds between notification-watcher drain passes over the process
    # registry's completion queue (background process / delegation events).
    _NOTIFY_DRAIN_INTERVAL_SEC = 2.0

    # Session-scoped reasoning effort, exposed as an ACP select config option.
    # "default" means no override (provider/profile default); the rest mirror
    # tools/reasoning_effort_tool.VALID_LEVELS and the /reasoning command.
    _REASONING_EFFORT_CONFIG_ID = "reasoning_effort"
    _REASONING_EFFORT_DEFAULT = "default"
    _REASONING_EFFORT_LEVELS = (
        ("default", "Default", "Provider/profile default reasoning effort."),
        ("none", "None", "Disable extended reasoning."),
        ("minimal", "Minimal", "Bare-minimum reasoning."),
        ("low", "Low", "Light reasoning for routine work."),
        ("medium", "Medium", "Balanced reasoning."),
        ("high", "High", "Deep reasoning for hard tasks."),
        ("xhigh", "XHigh", "Maximum reasoning effort."),
    )

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None
        # Background watcher draining process/delegation completion events
        # into sessions; started lazily from a running-loop context (see
        # _ensure_notification_watcher).
        self._notification_watcher_task: Optional[asyncio.Task] = None
        # Session ids already sent through the background title backfill —
        # one derive attempt per session per process, so a failing aux model
        # doesn't get hammered on every sessions-list refresh.
        self._title_backfill_attempted: set[str] = set()

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        # Sync method, but in production it runs inside acp.run_agent's event
        # loop (AgentSideConnection.__init__ is constructed in a coroutine),
        # so create_task works; with no running loop (sync tests) the starter
        # skips and prompt()'s defensive call brings the watcher up instead.
        self._ensure_notification_watcher()
        logger.info("ACP client connected")

    async def _send(self, session_id: str, update: Any, *, fail_msg: str, level: int = logging.WARNING) -> bool:
        """``session_update`` that logs instead of raising; False on failure."""
        try:
            await self._conn.session_update(session_id=session_id, update=update)
            return True
        except Exception:
            logger.log(level, fail_msg, session_id, exc_info=True)
            return False

    def _schedule_soon(self, make_coro: Callable[[], Any]) -> None:
        """Run a notification coroutine right after the current response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(asyncio.create_task, make_coro())

    def _session_modes(self, state: SessionState) -> SessionModeState:
        """Edit-approval policy as ACP modes. Zed renders ``config_options`` in the model
        picker's slot; modes (as Claude/Codex use) coexist with the picker."""
        current = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        if current not in self._MODES:
            current = self._MODE_DEFAULT
        return SessionModeState(
            current_mode_id=current,
            available_modes=[SessionMode(id=m, name=n, description=d) for m, (_p, n, d) in self._MODES.items()],
        )

    def _edit_approval_policy_for_state(self, state: SessionState) -> tuple[str, str | None]:
        mode = str(getattr(state, "mode", "") or self._MODE_DEFAULT)
        policy = self._MODE_TO_EDIT_APPROVAL_POLICY.get(mode, self._EDIT_APPROVAL_POLICY_DEFAULT)
        return policy, state.cwd

    def _build_model_state(self, state: SessionState, *, force_refresh: bool = False) -> SessionModelState | None:
        """Authenticated providers + models, from the shared Hermes inventory (same substrate
        as ``hermes model``/TUI/dashboard) so the selector isn't just the current curated list."""
        model = str(state.model or getattr(state.agent, "model", "") or "").strip()
        provider = _agent_provider_identity(state.agent) or detect_provider() or "openrouter"
        if provider == "custom":
            provider = _named_slug_for_base_url(str(getattr(state.agent, "base_url", "") or "")) or provider
        try:
            picker = build_model_state(model, provider, str(getattr(state.agent, "base_url", "") or ""), force_refresh=force_refresh)
            if picker is not None:
                return picker
        except Exception:
            logger.debug("Could not build ACP model state", exc_info=True)

        if not model:
            return None
        choice = encode_model_choice(provider, model)
        return SessionModelState(available_models=[ModelInfo(model_id=choice, name=model)], current_model_id=choice)

    def _switch_model(
        self, state: SessionState, raw_model: str, *, keep_endpoint: bool = False
    ) -> tuple[str | None, str, str]:
        """Rebuild the session agent on a new model -> (old provider, new provider, model).

        Resolution goes through ``hermes_cli.model_switch.switch_model`` seeded with the live
        agent route — the same catalog/alias/credential validation as CLI/gateway/TUI ``/model``
        — so ACP never hands the session a model no provider can serve. ``provider:model`` picker
        ids become ``--provider``. ACP never persists. ``keep_endpoint`` carries base_url/api_mode
        over when the provider is unchanged."""
        from hermes_cli.config import get_compatible_custom_providers, load_config
        from hermes_cli.model_switch import switch_model
        from hermes_cli.models import parse_model_input

        current_provider, current_model = _agent_provider_identity(state.agent), str(state.model or "")
        explicit_provider, model_input = parse_model_input(raw_model, "")
        cfg = load_config()
        result = switch_model(
            raw_input=model_input, explicit_provider=explicit_provider,
            current_provider=current_provider or "openrouter", current_model=current_model,
            current_base_url=str(getattr(state.agent, "base_url", "") or ""),
            current_api_key=str(getattr(state.agent, "api_key", "") or ""),
            user_providers=cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {},
            custom_providers=get_compatible_custom_providers(cfg))
        if not result.success:
            raise ModelRejected(result.error_message or f"Cannot switch to {raw_model}")
        target_provider, new_model = result.target_provider, result.new_model
        endpoint: dict[str, Any] = {}
        if keep_endpoint and not (current_provider and target_provider != current_provider):
            endpoint = {
                "base_url": getattr(state.agent, "base_url", None), "api_mode": getattr(state.agent, "api_mode", None)
            }
        # ACP-provided MCP servers live only on the running agent's toolsets (``_register_session_mcp_servers``);
        # a rebuild that re-derived them from config would silently drop every session MCP tool (#42719).
        agent = self.session_manager._make_agent(
            session_id=state.session_id, cwd=state.cwd, model=new_model,
            requested_provider=target_provider, **endpoint,
            enabled_toolsets=getattr(state.agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(state.agent, "disabled_toolsets", None),
        )
        # Assign only after the rebuild succeeded so a failed switch leaves the session on its
        # working model instead of a model/agent mismatch that persists via save_session.
        _apply_effort_to_agent(agent, getattr(state, "effort", ""))
        state.agent, state.model = agent, new_model
        self.session_manager.save_session(state.session_id)
        from hermes_cli.observability.shared_metrics_events import record_model_switch

        record_model_switch(
            from_provider=current_provider, to_provider=target_provider, surface="acp", from_model=current_model,
            session_id=state.session_id)
        return current_provider, target_provider, new_model

    @staticmethod
    def _build_usage_update(state: SessionState) -> UsageUpdate | None:
        """``usage_update`` for Zed's context indicator: ``size`` = context window, ``used`` =
        estimated request pressure (system prompt + history + tool schemas)."""
        compressor = getattr(state.agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        if size <= 0:
            return None
        try:
            used = _estimate_tokens(state.history, state.agent)
        except Exception:
            logger.debug("Could not estimate ACP native context usage", exc_info=True)
            used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
        return UsageUpdate(session_update="usage_update", size=max(size, 0), used=max(used, 0))

    async def _send_usage_update(self, state: SessionState) -> None:
        if self._conn and (update := self._build_usage_update(state)) is not None:
            await self._send(state.session_id, update, fail_msg="Failed to send ACP usage update for session %s")

    def _provenance_meta(
        self, acp_session_id: str, current_hermes_session_id: str, previous_hermes_session_id: Optional[str] = None
    ) -> Optional[dict]:
        """Best-effort ``_meta.hermes.sessionProvenance`` for an ACP session."""
        try:
            return session_provenance_meta(
                self.session_manager._get_db(), acp_session_id, current_hermes_session_id,
                previous_hermes_session_id=previous_hermes_session_id,
            )
        except Exception:
            logger.debug("Could not build ACP session provenance for %s", acp_session_id, exc_info=True)
            return None

    async def _send_session_info_update(
        self, session_id: str, *,
        current_hermes_session_id: Optional[str] = None, previous_hermes_session_id: Optional[str] = None,
    ) -> None:
        """Session metadata update; pass ``previous_hermes_session_id`` when the internal head
        rotated (compression split) so provenance flags the reason."""
        if not self._conn:
            return
        try:
            row = self.session_manager._get_db().get_session(session_id)
        except Exception:
            logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
            return
        if not row:
            return
        title = row.get("title")
        # `sessions` has no `updated_at`; "now" is right since this fires when the title changed.
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title if isinstance(title, str) and title.strip() else None,
            updated_at=datetime.now(UTC).isoformat(),
            field_meta=self._provenance_meta(
                session_id, current_hermes_session_id or session_id, previous_hermes_session_id
            ),
        )
        await self._send(
            session_id, update, fail_msg="Could not send ACP session info update for %s", level=logging.DEBUG
        )

    async def _register_session_mcp_servers(
        self, state: SessionState, mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return
        try:
            from agent.runtime_cwd import set_session_cwd
            from tools.mcp_tool_discovery import register_mcp_servers

            configs = {s.name: _mcp_server_config(s) for s in mcp_servers}

            def _register_pinned() -> None:
                # new_session/load_session run outside the per-turn cwd pin; the session's logical cwd is
                # the default stdio child cwd (tools/mcp_tool_transport.py::_run_stdio), so pin it here.
                set_session_cwd(state.cwd)
                register_mcp_servers(configs)

            await asyncio.to_thread(_register_pinned)  # to_thread already runs in a copied context
        except Exception:
            logger.warning("Session %s: failed to register ACP MCP servers", state.session_id, exc_info=True)
            return
        try:
            from model_tools import get_tool_definitions
            from agent.memory_manager import inject_memory_provider_tools

            agent = state.agent
            agent.enabled_toolsets = _expand_acp_enabled_toolsets(
                getattr(agent, "enabled_toolsets", None),
                mcp_server_names=[s.name for s in mcp_servers],
            )
            agent.tools = get_tool_definitions(
                enabled_toolsets=agent.enabled_toolsets,
                disabled_toolsets=getattr(agent, "disabled_toolsets", None), quiet_mode=True,
            )
            agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools or []}
            inject_memory_provider_tools(agent)
            if callable(invalidate := getattr(agent, "_invalidate_system_prompt", None)):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id, len(agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration", state.session_id, exc_info=True,
            )

    def _schedule_mcp_late_refresh(self, state: SessionState) -> None:
        """Refresh the tool snapshot when background MCP discovery lands after agent build
        (``_make_agent`` only joins ~1.5s). Waits up to 30s off the critical path, then rebuilds
        via ``refresh_agent_mcp_tools`` (same as ``/reload-mcp``).

        Cache safety: only pre-first-turn (nothing cached yet); afterwards the snapshot stays
        frozen and late servers land via the between-turns prologue refresh
        (``agent/turn_context.py``). No-op if discovery finished, join timed out, registry
        unchanged, or session closed."""
        try:
            from hermes_cli.mcp_startup import mcp_discovery_in_flight
        except Exception:
            return
        if not mcp_discovery_in_flight():
            return
        agent, session_id = state.agent, state.session_id

        def _wait_then_refresh() -> None:
            try:
                from hermes_cli.mcp_startup import join_mcp_discovery

                if not join_mcp_discovery(timeout=30.0):
                    return

                # In-memory only: ``get_session()`` would restore from DB and build a new AIAgent.
                with self.session_manager._lock:
                    current = self.session_manager._sessions.get(session_id)
                if current is None or current.agent is not agent:
                    return

                # ``prompt()`` flips ``is_running`` under ``runtime_lock`` before dispatching, so
                # holding it here closes the window where a refresh would swap ``tools=`` mid-turn.
                with current.runtime_lock:
                    if current.is_running:
                        return
                    if any(int(getattr(agent, k, 0) or 0) > 0 for k in ("_user_turn_count", "_api_call_count")):
                        return

                    from tools.mcp_tool_agent import refresh_agent_mcp_tools

                    added = refresh_agent_mcp_tools(agent, quiet_mode=True)
                if added:
                    logger.info(
                        "Session %s: late MCP refresh added %d tools: %s",
                        session_id, len(added), ", ".join(sorted(added)),
                    )
            except Exception:
                logger.debug("Session %s: late MCP refresh failed", session_id, exc_info=True)

        threading.Thread(target=_wait_then_refresh, name=f"acp-mcp-late-refresh-{session_id}", daemon=True).start()

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self, protocol_version: int | None = None, client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None, **kwargs: Any,
    ) -> InitializeResponse:
        from hermes_cli.version_info import get_version_info

        auth_methods = build_auth_methods()
        logger.info(
            "Initialize from %s (protocol v%s)", client_info.name if client_info else "unknown",
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=get_version_info().base_version),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True, _meta={"hermes": {"keepHistory": True}}),
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(_meta={"hermes": {"keepHistory": True}}), list=SessionListCapabilities(), resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        # Only acknowledge the method_id advertised in initialize().
        if not isinstance(method_id, str):
            return None
        normalized_method = method_id.strip().lower()
        provider = detect_provider()

        if normalized_method == TERMINAL_SETUP_AUTH_METHOD_ID:
            # Terminal auth runs setup out-of-band; succeed only once credentials exist.
            return AuthenticateResponse() if provider else None

        if not provider or normalized_method != provider:
            return None
        return AuthenticateResponse()

    # ---- Session management -------------------------------------------------

    async def _replay_session_history(self, state: SessionState) -> None:
        """Replay persisted user/assistant history during session/load or session/resume.
        ``resume_session`` so that spec-compliant ACP clients receive the
        full transcript within the request's lifetime — see the comment at
        the call sites for the rationale and prior-art citations.

        Replays the conversation as user/assistant chunks, thinking-mode
        thought chunks, plus reconstructed tool-call start/completion
        notifications. Merely restoring server-side state makes Hermes
        remember context, but leaves the editor looking like a clean thread.
        """
        if not self._conn:
            return

        with state.runtime_lock:
            turn_running = state.is_running

        source: list[dict[str, Any]] = state.history
        # Index below which replayed user chunks carry a fork coordinate
        # (_meta.hermes.historyIndex). Defaults to "everything".
        fork_index_limit = len(source)
        if turn_running:
            transcript = self.session_manager.live_transcript_history(state.session_id)
            if transcript is not None and len(transcript) >= len(state.history):
                # Mid-turn: state.history only extends at turn END, but
                # run_agent flushes messages to the DB continuously — the DB
                # is the faithful transcript. Rows past the last completed
                # turn have no stable fork coordinates (compaction may
                # rewrite them before the turn finalizes), so those chunks
                # omit historyIndex. But the prefix that still matches
                # state.history IS stable — fork_session slices
                # state.history, so indices into that prefix remain valid
                # and "fork from an earlier completed turn" keeps working
                # while a turn runs. We stamp only the verified matching
                # prefix: a mid-turn compaction that rewrote older rows
                # breaks the match and stops stamping at the first
                # divergence. A None/shorter transcript means the store is
                # unavailable or resolved the wrong lineage: fail open to
                # the in-memory history below.
                fork_index_limit = self._stable_history_prefix_len(
                    state.history, transcript
                )
                source = transcript
        else:
            transcript = self.session_manager.live_transcript_history(
                state.session_id, repair_alternation=True
            )
            if transcript is not None and len(transcript) > len(state.history):
                # The session advanced in ANOTHER process (Slack bot,
                # gateway, CLI, delegate child): this process cached
                # state.history at restore time and never observes turns it
                # doesn't run, so the persisted store is ahead. Adopt it as
                # the live history — the same source _restore feeds a fresh
                # load — so replay shows the full conversation and the next
                # prompt/_persist build on it instead of clobbering the
                # newer rows. The adopted list IS state.history, so fork
                # coordinates stay valid. A None/shorter transcript means
                # the store is unavailable, resolved the wrong lineage, or
                # was compacted elsewhere: fail open to the in-memory
                # history.
                logger.info(
                    "Session %s advanced outside this process: adopting %d "
                    "persisted rows over %d cached in memory",
                    state.session_id,
                    len(transcript),
                    len(state.history),
                )
                state.history = transcript
                source = state.history

        if not source:
            return

        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        async def _send(update: Any, meta: dict[str, Any] | None = None) -> bool:
            try:
                # ``meta`` becomes the notification-level ``_meta`` (the acp lib
                # forwards session_update **kwargs into SessionNotification's
                # field_meta). We use it to stamp each replayed user turn with
                # its absolute ``state.history`` index so the client can pass
                # that coordinate back as ``_meta.hermes.keepHistory`` on
                # ``session/fork`` — the same list ``fork_session`` slices.
                if meta:
                    await self._conn.session_update(
                        session_id=state.session_id, update=update, **meta
                    )
                else:
                    await self._conn.session_update(session_id=state.session_id, update=update)
                return True
            except Exception:
                logger.warning(
                    "Failed to replay ACP history for session %s",
                    state.session_id,
                    exc_info=True,
                )
                return False

        for index, message in enumerate(source):
            role = str(message.get("role") or "")

            if role == "user":
                text = self._history_message_text(message)
                chunk_meta = self._history_chunk_meta(message, text)
                meta = (
                    {"hermes": {"historyIndex": index}}
                    if index < fork_index_limit
                    else None
                )
                if text:
                    update = self._history_message_update(
                        role=role,
                        text=text,
                        field_meta=chunk_meta,
                    )
                    if update is not None and not await _send(update, meta):
                        return
                # Attachments the user sent with this turn. Emitted after the
                # text so the client's chunk-accumulating block already exists,
                # and with the SAME notification meta so a fork coordinate
                # isn't lost when a turn is image-only.
                for data, mime_type in self._history_message_images(message):
                    image_update = self._history_image_update(
                        role=role,
                        data=data,
                        mime_type=mime_type,
                        field_meta=chunk_meta,
                    )
                    if image_update is not None and not await _send(image_update, meta):
                        return
                continue

            if role == "assistant":
                thought = self._history_reasoning_text(message)
                if thought and not await _send(self._history_thought_update(thought)):
                    return

                text = self._history_message_text(message)
                chunk_meta = self._history_chunk_meta(message, text)
                if text:
                    update = self._history_message_update(
                        role=role,
                        text=text,
                        field_meta=chunk_meta,
                    )
                    if update is not None and not await _send(update):
                        return

                for data, mime_type in self._history_message_images(message):
                    image_update = self._history_image_update(
                        role=role,
                        data=data,
                        mime_type=mime_type,
                        field_meta=chunk_meta,
                    )
                    if image_update is not None and not await _send(image_update):
                        return

                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = self._history_tool_call_id(tool_call)
                        if not tool_call_id:
                            continue
                        tool_name, args = self._history_tool_call_name_args(tool_call)
                        active_tool_calls[tool_call_id] = (tool_name, args)
                        if not await _send(build_tool_start(tool_call_id, tool_name, args)):
                            return
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                tool_name = str(message.get("tool_name") or "").strip()
                function_args: dict[str, Any] | None = None
                if tool_call_id in active_tool_calls:
                    tool_name, function_args = active_tool_calls.pop(tool_call_id)
                if not tool_call_id or not tool_name:
                    continue
                result = message.get("content")
                result_text = result if isinstance(result, str) else None
                complete = build_tool_complete(
                    tool_call_id,
                    tool_name,
                    result=result_text,
                    function_args=function_args,
                )
                # The tool row's persisted time is when the result was flushed,
                # so it stamps the completion; the start event stays unstamped.
                timestamp = self._history_timestamp_iso(message)
                if timestamp:
                    complete.field_meta = {"hermes": {"timestamp": timestamp}}
                if not await _send(complete):
                    return
                if tool_name == "todo":
                    plan_update = _build_plan_update_from_todo_result(result_text)
                    if plan_update is not None and not await _send(plan_update):
                        return

    async def _session_response_fields(self, state: SessionState, replay_verb: str | None = None) -> dict[str, Any]:
        """``models``/``modes``/``field_meta`` for session responses, after an optional history replay;
        schedules command advertisement + usage refresh.

        Per ACP spec, load/resume must stream history via ``session/update`` BEFORE responding
        (Codex/Claude Code/OpenCode/Zed rely on this; deferring via ``call_soon`` broke them).
        Best-effort: a corrupt message must not turn the load into an error."""
        if replay_verb:
            try:
                # Per ACP spec, `session/load` must stream the prior conversation back to the client via
                # `session/update` notifications BEFORE responding, so the client receives the full
                # transcript within the load request's lifetime. Awaiting the replay here matches Codex /
                # Claude Code / OpenCode / Pi and the Zed client (which registers the session-update routing
                # entry before awaiting the loadSession RPC specifically so in-call history replay updates
                # can find the thread). Deferring this via `loop.call_soon` (as we did briefly in May 2026)
                # broke every spec-compliant ACP client that measures notifications synchronously against
                # the load response — see #12285 follow-up.
                await self._replay_session_history(state)
            except Exception:
                logger.warning(
                    f"ACP history replay raised during session/{replay_verb} for %s — "
                    f"{replay_verb} will still succeed, partial transcript may be missing",
                    state.session_id, exc_info=True,
                )
        self._schedule_available_commands_update(state.session_id)
        self._schedule_soon(lambda: self._send_usage_update(state))
        return {
            "models": self._build_model_state(state),
            "modes": self._session_modes(state),
            "config_options": self._session_config_options(state),
            "field_meta": self._merge_hermes_meta(
                self._provenance_meta(state.session_id, getattr(state.agent, "session_id", state.session_id)),
                {**self._turn_status_meta(state), **self._subagent_load_meta(state), **self._message_count_meta(state)},
            ),
        }

    async def _attach_session_mcp(self, state: SessionState, mcp_servers: list | None, log: str, *log_args) -> None:
        await self._register_session_mcp_servers(state, mcp_servers)
        self._schedule_mcp_late_refresh(state)
        logger.info(log, *log_args)

    async def new_session(self, cwd: str, mcp_servers: list | None = None, **kwargs: Any) -> NewSessionResponse:
        # Agent construction (config, memory-provider import, SessionDB) is slow and fully
        # blocking; inline it froze the loop serving every JSON-RPC request (#58083).
        owner = str((kwargs.get("hermes") or {}).get("owner") or "").strip() or None
        state = await asyncio.to_thread(self.session_manager.create_session, cwd=cwd, owner=owner)
        await self._attach_session_mcp(state, mcp_servers, "New session %s (cwd=%s)", state.session_id, cwd)
        return NewSessionResponse(session_id=state.session_id, **await self._session_response_fields(state))

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> LoadSessionResponse | None:
        state = await asyncio.to_thread(self.session_manager.update_cwd, session_id, cwd, persist=False)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._attach_session_mcp(state, mcp_servers, "Loaded session %s", session_id)
        return LoadSessionResponse(**await self._session_response_fields(state, "load"))

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list | None = None, **kwargs: Any
    ) -> ResumeSessionResponse:
        state = await asyncio.to_thread(self.session_manager.update_cwd, session_id, cwd, persist=False)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = await asyncio.to_thread(self.session_manager.create_session, cwd=cwd)
        await self._attach_session_mcp(state, mcp_servers, "Resumed session %s", state.session_id)
        return ResumeSessionResponse(**await self._session_response_fields(state, "resume"))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        # get_session restores a not-in-memory id from the DB (full AIAgent build) and waits
        # on the restore lock — off the loop, like new/load/resume/fork (#58083).
        state = await asyncio.to_thread(self.session_manager.get_session, session_id)
        if not (state and state.cancel_event):
            return
        with state.runtime_lock:
            if state.is_running and state.current_prompt_text and not state.response_delivered:
                state.interrupted_prompt_text = state.current_prompt_text
                state.interrupted_prompt_at = time.monotonic()
            # Cancel + hard-stop under the lock so no other prompt mistakes this turn for
            # redirectable work.
            state.cancel_event.set()
            try:
                if state.agent:
                    request_hard_interrupt(state.agent)
                    # Background delegations are detached from the turn's fan-out; they end with the cancel.
                    from tools.async_delegation import interrupt_for_session
                    interrupt_for_session(parent_session_id=str(getattr(state.agent, "session_id", "") or ""),
                                          reason="acp_cancel")
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
        logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        keep_history = self._keep_history_meta(kwargs)
        state = await asyncio.to_thread(
            self.session_manager.fork_session,
            session_id,
            cwd=cwd,
            keep_history=keep_history,
        )
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(
            session_id=new_id,
            models=self._build_model_state(state) if state is not None else None,
            modes=self._session_modes(state) if state is not None else None,
            config_options=self._session_config_options(state) if state is not None else None,
        )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List ACP sessions with optional ``cwd`` filtering and cursor pagination.

        ``cwd`` is passed through to ``SessionManager.list_sessions`` which already
        normalizes and filters by working directory. ``cursor`` is a ``session_id``
        previously returned as ``next_cursor``; results resume after that entry.
        Server-side page size is capped at ``_LIST_SESSIONS_PAGE_SIZE``; when more
        results remain, ``next_cursor`` is set to the last returned ``session_id``.
        """
        hermes = kwargs.get("hermes") or {}
        archived_only = bool(hermes.get("archivedOnly"))
        include_archived = bool(hermes.get("includeArchived"))
        # Per-user filter: when ownerOnly is set, restrict to the caller's own
        # sessions (plus untagged legacy rows). owner carries the authenticated
        # identity (Cloudflare Access email) the client resolved. Soft display
        # filter, not an access boundary.
        owner = (
            str(hermes.get("owner") or "").strip()
            if hermes.get("ownerOnly")
            else None
        ) or None
        infos = self.session_manager.list_sessions(
            cwd=cwd,
            include_archived=include_archived,
            archived_only=archived_only,
            owner=owner,
        )

        # Group rows into fork/subagent families (a root plus every descendant
        # reachable via parent_id) and paginate on FAMILY boundaries. A naive
        # flat cut can strand children on one page while their parent sits past
        # the cursor; clients that nest lineage (the VS Code sidebar) then
        # render those children ungrouped until the user pages further. Families
        # are ordered by their freshest member (infos is recency-sorted), and a
        # family larger than the page size is still emitted whole — pages may
        # exceed the nominal cap rather than split a family.
        families = _group_session_families(infos)

        if cursor:
            # Resume after the family containing the cursor. The cursor is the
            # last session_id of the previous page, which (with family-atomic
            # pages) is the last member of a family — but tolerate a mid-family
            # cursor from an older server by skipping that whole family.
            for idx, fam in enumerate(families):
                if any(m["session_id"] == cursor for m in fam):
                    families = families[idx + 1:]
                    break
            else:
                # Unknown cursor -> empty page (do not fall back to full list).
                families = []

        infos = []
        consumed = 0
        for fam in families:
            # The cap counts FAMILIES (grouped units), not rows: a page always
            # carries up to _LIST_SESSIONS_PAGE_SIZE groups regardless of how
            # many fork/subagent children each group contains.
            if consumed >= _LIST_SESSIONS_PAGE_SIZE:
                break
            infos.extend(fam)
            consumed += 1
        has_more = consumed < len(families)

        # Opportunistic title backfill: sessions that missed the first-exchange
        # auto-title (long/interrupted first turn, aux failure) otherwise fall
        # back to the first-user-message preview as their list title. Derive
        # real titles in the background — capped per call, once per session per
        # process — and push each result via session_info updates.
        self._schedule_title_backfill(infos)

        sessions = []
        for s in infos:
            updated_at = s.get("updated_at")
            if updated_at is not None and not isinstance(updated_at, str):
                updated_at = str(updated_at)
            hermes_meta: dict[str, Any] = {}
            if s.get("archived"):
                hermes_meta["archived"] = True
            # Fork lineage: id of the session this one was forked from, so
            # clients can nest forks under their parent in session lists.
            if s.get("parent_id"):
                hermes_meta["forkedFrom"] = s["parent_id"]
            # Delegate children: read-only observation sessions nested under
            # their parent (forkedFrom above carries the linkage).
            if s.get("subagent"):
                hermes_meta["isSubagent"] = True
            # Slack-bot sessions (first user message carries the Slack runtime
            # envelope) so clients can badge them, e.g. a Slack icon prefix.
            if s.get("slack"):
                hermes_meta["slack"] = True
            # Session owner (sessions.user_id — the Cloudflare Access email
            # stamped on session/new) so clients can badge rows per user, e.g.
            # owner initials in an "All sessions" list. Soft display key only.
            owner_id = str(s.get("user_id") or "").strip()
            if owner_id:
                hermes_meta["owner"] = owner_id
            # Progress markers for sessions that advance in another process
            # (Slack bot, gateway): clients re-replay an open tab when
            # messageCount moves and show the activity heartbeat meanwhile.
            if s.get("message_count") is not None:
                hermes_meta["messageCount"] = s["message_count"]
            if s.get("activity"):
                hermes_meta["activity"] = s["activity"]
            field_meta = {"hermes": hermes_meta} if hermes_meta else None
            sessions.append(
                SessionInfo(
                    session_id=s["session_id"],
                    cwd=s["cwd"],
                    title=s.get("title"),
                    updated_at=updated_at,
                    field_meta=field_meta,
                )
            )

        next_cursor = sessions[-1].session_id if has_more and sessions else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    # ---- Prompt (core) ------------------------------------------------------

    def _rewrite_prompt_for_interrupt(self, state: SessionState, user_text: str, user_content: Any,
                                      text_only_prompt: bool) -> tuple[str, Any]:
        if (
            text_only_prompt
            and isinstance(user_content, str)
            and user_text.startswith("/steer")
        ):
            steer_text = user_text.split(maxsplit=1)[1].strip() if len(user_text.split(maxsplit=1)) > 1 else ""
            interrupted_prompt = ""
            rewrite_idle = False
            with state.runtime_lock:
                idle_steer = not state.is_running and bool(steer_text)
            if idle_steer:
                interrupted_prompt = _consume_interrupted_prompt(state)
                rewrite_idle = not interrupted_prompt
            if interrupted_prompt:
                user_text = _merge_interrupted_prompt(steer_text, interrupted_prompt)
                user_content = user_text
            elif rewrite_idle:
                user_text = steer_text
                user_content = steer_text
        elif (
            text_only_prompt
            and isinstance(user_content, str)
            and not user_text.startswith("/")
        ):
            # Some ACP clients implement "stop and send" as two protocol calls:
            # cancel the active prompt, then submit plain correction text. Keep
            # the cancelled request attached so deictic follow-ups ("not that
            # file") still have an explicit target — but only inside the
            # salvage window, and only as context BELOW the new instruction.
            interrupted_prompt = ""
            with state.runtime_lock:
                idle = not state.is_running
            if idle:
                interrupted_prompt = _consume_interrupted_prompt(state)
            if interrupted_prompt:
                user_text = _merge_interrupted_prompt(user_text, interrupted_prompt)
                user_content = user_text

        return user_text, user_content

    def _claim_turn_or_queue(self, state: SessionState, session_id: str, user_text: str,
                             user_content: Any, text_only_prompt: bool, *,
                             synthetic_notification: bool = False, completion_sink: Any = None) -> str | None:
        redirected = False
        queued_depth: int | None = None
        with state.runtime_lock:
            if state.is_running or state.command_op:
                if (
                    state.is_running
                    and text_only_prompt
                    and isinstance(user_content, str)
                    and not synthetic_notification
                    and completion_sink is None
                    and getattr(
                        state.agent,
                        "_supports_active_turn_redirect",
                        False,
                    )
                    is True
                    and hasattr(state.agent, "redirect")
                ):
                    try:
                        redirected = bool(state.agent.redirect(user_content))
                    except Exception:
                        logger.debug(
                            "ACP active-turn redirect failed for %s",
                            session_id,
                            exc_info=True,
                        )
                if not redirected:
                    queued_text = user_text or "[Image attachment]"
                    if isinstance(completion_sink, _TurnCompletionSink):
                        completion_sink.queued = True
                    state.queued_prompts.append(
                        QueuedPrompt(
                            text=queued_text,
                            synthetic_notification=synthetic_notification,
                            completion_sink=completion_sink,
                        )
                    )
                    queued_depth = len(state.queued_prompts)
            else:
                state.is_running = True
                state.current_prompt_text = user_text or "[Image attachment]"
                # A real turn is starting, so any prompt salvaged for a later
                # /steer after a prior client cancel is now stale — the user
                # moved on to this turn. Drop it so a correction on a future
                # idle session can't resurrect an abandoned prompt. The salvage
                # paths above already consumed and cleared the field on
                # text-only turns; this closes the gap for turns that skip them
                # (multimodal prompts), which would otherwise leave a live
                # buffer for a later text prompt to pick up. Adopted from
                # upstream PR NousResearch/hermes-agent#56624 — but scoped to
                # real user turns: a SYNTHETIC notification is a background
                # event, not the user moving on, and must leave the buffer
                # intact for the next real prompt (tests/acp/test_server.py::
                # test_notification_does_not_resurrect_cancelled_prompt). The
                # unscoped upstream version breaks that invariant.
                if not synthetic_notification:
                    state.interrupted_prompt_text = ""
                    state.interrupted_prompt_at = 0.0
                state.response_delivered = False

        if redirected:
            return "Redirected the active turn with your correction."
        if queued_depth is not None:
            return f"Queued for the next turn. ({queued_depth} queued)"
        return None

    def _run_agent_turn(
        self, *, state: SessionState, session_id: str, user_text: str, user_content: Any, conn: Any,
        loop: asyncio.AbstractEventLoop, approval_cb: Any, edit_approval_requester: Any,
    ) -> dict:
        """Executor-thread body of one turn, run inside ``contextvars.copy_context()`` so
        ContextVar writes are isolated from concurrent sessions.

        Approval routing is thread-local, so it MUST be bound here, not on the loop thread.
        Interactive routing is a ``tools.approval`` contextvar, not ``HERMES_INTERACTIVE`` in
        os.environ, so concurrent workers can't race a global flag onto the non-interactive
        auto-approve path (GHSA-96vc-wcxf-jjff)."""
        agent = state.agent
        with contextlib.ExitStack() as stack:
            # HERMES_SESSION_KEY scopes per-session caches (interactive sudo password) to this
            # session, not the reused thread. ``cwd`` pins what the system prompt reports as the
            # working directory — otherwise it advertises the Hermes workspace while tools are
            # rooted at the client's project and edits land outside it. ``cron_session=""`` masks
            # any leaked process-global HERMES_CRON_SESSION.
            def _session_context() -> Callable[[], None]:
                from gateway.session_context import clear_session_vars, set_session_vars

                tokens = set_session_vars(
                    session_key=session_id, session_id=session_id, cwd=state.cwd, cron_session="",
                    async_delivery=True, user_id=state.owner or "",
                )
                return lambda: clear_session_vars(tokens)

            def _approval() -> Callable[[], None]:
                from tools import terminal_tool

                previous = terminal_tool._get_approval_callback()
                terminal_tool.set_approval_callback(approval_cb)
                return lambda: terminal_tool.set_approval_callback(previous)

            def _edit_approval() -> Callable[[], None]:
                from acp_adapter.edit_approval import reset_edit_approval_requester, set_edit_approval_requester

                token = set_edit_approval_requester(edit_approval_requester)
                return lambda: reset_edit_approval_requester(token)

            def _spawn() -> Callable[[], None]:
                from acp_adapter.spawn import inject_spawn_session_tool, set_spawn_session_requester, reset_spawn_session_requester

                inject_spawn_session_tool(agent)
                token = set_spawn_session_requester(self._make_spawn_session_requester(loop, state))
                return lambda: reset_spawn_session_requester(token)

            _bind_guarded(stack, "session spawn requester", _spawn)
            _bind_guarded(stack, "session context", _session_context)
            if approval_cb:
                _bind_guarded(stack, "approval callback", _approval)
            if edit_approval_requester:
                _bind_guarded(stack, "edit approval requester", _edit_approval)
            stack.callback(reset_hermes_interactive_context, set_hermes_interactive_context(True))
            # Tools tag side-effects with the ACP session (``kanban_create``); save/restore it.
            stack.callback(_restore_env, "HERMES_SESSION_ID", os.environ.get("HERMES_SESSION_ID"))
            os.environ["HERMES_SESSION_ID"] = session_id

            # Auto-titling fires in the turn prologue; push the title now as a session-info update.
            def _notify_title_update(_title: str, _source: str) -> None:
                if conn:
                    loop.call_soon_threadsafe(asyncio.create_task, self._send_session_info_update(session_id))

            agent._on_session_title = _notify_title_update
            try:
                return agent.run_conversation(
                    user_message=user_content, conversation_history=state.history, task_id=session_id,
                    persist_user_message=user_text or "[Image attachment]",
                )
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}

    async def prompt(self, prompt: list[PromptBlock], session_id: str, **kwargs: Any) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor.

        ``_synthetic_notification`` (read from ``**kwargs``, never part of
        the method signature) is a PRIVATE, internal-only flag set by
        ``_deliver_notification`` when the "prompt" is a synthetic
        background-completion notification rather than typed user input.
        Passing it through ``**kwargs`` keeps the override signature
        identical to ``acp.Agent.prompt`` — the framework's dispatch
        (``build_agent_router`` → ``func(**model_to_kwargs(PromptRequest))``)
        only ever passes ``prompt``/``session_id``/``message_id`` plus
        ``_meta`` keys, so it can never set the flag accidentally. A client
        could only reach it by sending a literal
        ``_meta._synthetic_notification`` key — which merely opts its OWN
        prompt out of slash parsing and interrupt-merge, no privilege
        gained. When True:

        - the busy branch NEVER redirects/steers the running turn — the
          text is appended to ``queued_prompts`` silently (no "Queued for
          the next turn" client ack) and the call returns. Synthetic
          events must never interrupt a running turn, in any interleaving.
        - the ``interrupted_prompt_text`` merge is skipped (and the field
          left intact): a background notification must not resurrect a
          prompt the user deliberately cancelled.
        - slash-command / ``/steer`` interception does not apply —
          notification text is payload, never a command (belt-and-braces
          with the "/" neutralization in ``_deliver_notification``).
        """
        synthetic_notification = bool(kwargs.pop("_synthetic_notification", False))
        completion_sink = kwargs.pop("_turn_completion_sink", None)
        if not callable(completion_sink):
            completion_sink = None
        # Defensive resurrection: if the watcher task ever died (or on_connect
        # ran without a loop), the first prompt from loop context restarts it.
        self._ensure_notification_watcher()
        state = await asyncio.to_thread(self.session_manager.get_session, session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            if completion_sink is not None:
                completion_sink(
                    "",
                    False,
                    "failed",
                    f"ACP session {session_id} was not found.",
                )
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        user_content = _content_blocks_to_openai_user_content(prompt)

        # In-place rewind ("edit & resend"): _meta.hermes.keepHistory on
        # session/prompt truncates the session to the first N history entries
        # before this turn runs — the prompt-side twin of the fork extension,
        # atomic with the submit so a cancel/truncate/send race can't lose the
        # rewind. Refused while a turn is running (the executor thread owns
        # state.history mid-turn); validated before the empty-content early
        # return so an invalid rewind never silently no-ops.
        keep_history = self._keep_history_meta(kwargs)
        text_only_prompt = all(isinstance(block, TextContentBlock) for block in prompt)
        has_content = _prompt_has_content(user_text, user_content)
        if not has_content:
            if keep_history is not None:
                # A truncate with nothing to resend is not a supported shape —
                # the rewind extension is always "edit & resend". Fail loudly
                # instead of dropping history with no replacement turn.
                if completion_sink is not None:
                    completion_sink(
                        "",
                        False,
                        "failed",
                        "The spawned turn requested keepHistory with an empty prompt.",
                    )
                raise acp.RequestError.invalid_params(
                    {"reason": "_meta.hermes.keepHistory requires a non-empty prompt"}
                )
            if completion_sink is not None:
                completion_sink(
                    "",
                    False,
                    "failed",
                    "The spawned turn received an empty prompt.",
                )
            return PromptResponse(stop_reason="end_turn")

        if keep_history is not None:
            with state.runtime_lock:
                if state.is_running or state.command_op:
                    # The executor thread owns state.history mid-turn; queueing
                    # the text while dropping the rewind would silently keep the
                    # turn being rewound away. Clients cancel first, then rewind.
                    if completion_sink is not None:
                        completion_sink(
                            "",
                            False,
                            "failed",
                            "The spawned turn cannot rewind a session while it is running.",
                        )
                    raise acp.RequestError.invalid_params(
                        {"reason": "cannot rewind while a turn is running"}
                    )
                truncated = keep_history < len(state.history)
                if truncated:
                    state.history = state.history[:keep_history]
            if truncated:
                # Mirror /reset and /compact: archive the dropped rows in the
                # message store so a mid-turn DB replay or restart can't
                # resurrect them, then persist the new list.
                self._reconcile_persisted_history(state, state.history)
                self.session_manager.save_session(session_id)

        if not synthetic_notification:
            user_text, user_content = self._rewrite_prompt_for_interrupt(state, user_text, user_content, text_only_prompt)

        # Intercept slash commands — handle locally without calling the LLM.
        # Slash commands are text-only; if the client included images/resources,
        # send the whole multimodal prompt to the agent instead of treating it as
        # an ACP command. Synthetic notifications are payload, never commands —
        # skipped in notification mode (belt-and-braces with the leading-"/"
        # neutralization in _deliver_notification).
        if (
            text_only_prompt
            and isinstance(user_content, str)
            and user_text.startswith("/")
            and not synthetic_notification
        ):
            response_text = await asyncio.to_thread(self._handle_slash_command, user_text, state)
            if response_text is not None:
                if completion_sink is not None:
                    completion_sink(
                        "",
                        False,
                        "failed",
                        "The spawned prompt resolved as a local slash command, not an agent turn.",
                    )
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                    await self._send_usage_update(state)
                await self._drain_queued_prompts(state, session_id, self._conn)
                return PromptResponse(stop_reason="end_turn")

        # If the client sends another regular text prompt while this ACP session
        # is running, route it through the core active-turn redirect. Rich media
        # and older runtimes retain the proven next-turn queue fallback.
        # Synthetic notifications must NEVER redirect (abort/steer) a running
        # turn — in any interleaving, including the TOCTOU window where a user
        # prompt starts a turn between _deliver_notification's idle check and
        # this re-check — so notification mode always takes the queue branch.
        absorbed = self._claim_turn_or_queue(
            state, session_id, user_text, user_content, text_only_prompt,
            synthetic_notification=synthetic_notification, completion_sink=completion_sink,
        )
        if absorbed is not None:
            if self._conn and not synthetic_notification:
                await self._send(session_id, acp.update_agent_message_text(absorbed), fail_msg="Could not acknowledge queued turn for %s")
            return PromptResponse(stop_reason="end_turn")

        # ``state.is_running`` was just set True above. From here to the end
        # of the post-turn tail, ANY escaping raise must reset it — otherwise
        # the session is permanently bricked: every later user message (and
        # every notification delivery) lands in ``queued_prompts`` with no
        # turn ever draining the queue. The executor call and the post-turn
        # tail already carry their own reset handlers (kept — they also emit
        # the idle status update and shape the response); this outer guard
        # extends the same protection to the pre-executor SETUP region
        # (turn-status update, callback-factory construction, spawn-requester
        # wiring), which used to be an unprotected raise window. The inner
        # resets make the except's reset idempotent, never conflicting.
        try:
            # Tell any attached client the turn is live. The panel that issued
            # this prompt() already knows; a panel that merely loaded the session
            # (or one watching a spawned session's first turn) needs the signal
            # to flip its composer into running/steer state.
            await self._send_turn_status_update(state, running=True)

            # First-turn identity: when the session carries an authenticated owner
            # (e.g. the Cloudflare Access email stamped via _meta.hermes.owner at
            # session/new), surface it to the agent ONCE — on the first turn, when
            # history is still empty — as a neutral context line prepended to the
            # user message. Later turns already have it in history, so re-injecting
            # would waste tokens. Text-only prompts only (a plain string user_content);
            # multimodal turns keep their structured content untouched.
            if state.owner and not state.history and isinstance(user_content, str):
                from hermes_constants import ACP_OWNER_NOTE_PREFIX

                owner_note = (
                    f"{ACP_OWNER_NOTE_PREFIX} {state.owner}]\n"
                    "(This is the signed-in user you are talking to, from the "
                    "surface's SSO/identity provider. Use it to address and identify "
                    "them; do not ask who they are.)\n\n"
                )
                user_content = owner_note + user_content

            logger.info("Prompt on session %s: %s", session_id, user_text[:100])

            conn = self._conn
            loop = asyncio.get_running_loop()

            if state.cancel_event:
                state.cancel_event.clear()

            cbs = self._wire_turn_callbacks(state, session_id, conn, loop)

            def _run_agent() -> dict:
                try:
                    return self._run_agent_turn(
                        state=state, session_id=session_id, user_text=user_text, user_content=user_content,
                        conn=conn, loop=loop, approval_cb=cbs.approval_cb,
                        edit_approval_requester=cbs.edit_approval_requester,
                    )
                finally:
                    if cbs.subagent_router is not None:
                        try:
                            cbs.subagent_router.finalize()
                        except Exception:
                            logger.debug("Could not finalize subagent session frames", exc_info=True)
                    self._flush_turn_tool_calls(cbs, session_id, conn, loop)

            try:
                # Snapshot the internal Hermes DB session id before the turn so we
                # can detect a compression-driven session rotation afterwards. The
                # ACP `session_id` stays the stable client handle; agent.session_id
                # is the live internal head that compression may rotate.
                pre_turn_hermes_id = getattr(state.agent, "session_id", None)
                # Wrap the executor call in a fresh copy of the current context so
                # concurrent ACP sessions on the shared ThreadPoolExecutor don't
                # stomp on each other's ContextVar writes (HERMES_SESSION_KEY in
                # particular — used by the interactive sudo password cache scope).
                ctx = contextvars.copy_context()
                result = await loop.run_in_executor(_executor, ctx.run, _run_agent)
            except Exception:
                logger.exception("Executor error for session %s", session_id)
                with state.runtime_lock:
                    state.is_running = False
                    state.current_prompt_text = ""
                    completion_entries = []
                    retained_entries = []
                    for entry in state.queued_prompts:
                        if (
                            isinstance(entry, QueuedPrompt)
                            and isinstance(entry.completion_sink, _TurnCompletionSink)
                        ):
                            completion_entries.append(entry)
                        else:
                            retained_entries.append(entry)
                    state.queued_prompts = retained_entries
                if completion_sink is not None:
                    completion_sink(
                        "",
                        False,
                        "failed",
                        "The ACP executor could not start the spawned turn.",
                    )
                _fail_queued_completion_sinks(
                    completion_entries,
                    interrupted=False,
                    error="The ACP executor failed before queued spawned work could run.",
                )
                try:
                    await self._send_turn_status_update(state, running=False)
                except Exception:
                    logger.debug(
                        "Could not emit idle status after executor failure for %s",
                        session_id,
                        exc_info=True,
                    )
                return PromptResponse(stop_reason="end_turn")

            return await self._finish_turn(
                state, session_id, conn, result, pre_turn_hermes_id, cbs.streamed,
                user_content=user_content, completion_sink=completion_sink,
            )
        except BaseException as exc:
            # Reset (idempotent if an inner handler already did) and flip any
            # attached panel back to idle, then let the error propagate —
            # prompt() has no business swallowing setup failures.
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
                completion_entries = []
                retained_entries = []
                for entry in state.queued_prompts:
                    if (
                        isinstance(entry, QueuedPrompt)
                        and isinstance(entry.completion_sink, _TurnCompletionSink)
                    ):
                        completion_entries.append(entry)
                    else:
                        retained_entries.append(entry)
                state.queued_prompts = retained_entries
            try:
                await self._send_turn_status_update(state, running=False)
            except Exception:
                pass  # best-effort; never mask the original error
            if completion_sink is not None:
                interrupted = isinstance(exc, asyncio.CancelledError)
                completion_sink(
                    "",
                    interrupted,
                    "cancelled" if interrupted else "failed",
                    f"{type(exc).__name__}: {exc}"[:1000],
                )
            interrupted = isinstance(exc, asyncio.CancelledError)
            _fail_queued_completion_sinks(
                completion_entries,
                interrupted=interrupted,
                error=(
                    "The queued spawned turn was cancelled before delivery."
                    if interrupted
                    else f"Queued spawned turn failed: {type(exc).__name__}: {exc}"[:1000]
                ),
            )
            raise


    def _flush_turn_tool_calls(
        self, cbs: _TurnCallbacks, session_id: str, conn: Any, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Close any tool call the turn left open, so no bubble spins after the turn ends."""
        if not conn or cbs.tool_call_ids is None:
            return
        try:
            flush_open_tool_calls(conn, session_id, loop, cbs.tool_call_ids, cbs.tool_call_meta or {})
        except Exception:
            logger.debug("Could not flush open ACP tool calls for %s", session_id, exc_info=True)

    def _wire_turn_callbacks(
        self, state: SessionState, session_id: str, conn: Any, loop: asyncio.AbstractEventLoop
    ) -> _TurnCallbacks:
        """Install the ACP streaming callbacks on the session agent for one turn."""
        cbs = _TurnCallbacks()
        if conn:
            tool_call_ids: dict[str, deque[str]] = defaultdict(deque)
            tool_call_meta: dict[str, dict[str, Any]] = {}
            cbs.tool_call_ids, cbs.tool_call_meta = tool_call_ids, tool_call_meta
            # Shared with the step callback so a runtime that projects
            # ``tool.completed`` closes each call once, not twice.
            turn_state: dict[str, Any] = {}
            cbs.subagent_router = make_subagent_update_router(conn, loop)
            policy_getter = lambda: self._edit_approval_policy_for_state(state)
            cbs.tool_progress_cb = make_tool_progress_cb(
                conn, session_id, loop, tool_call_ids, tool_call_meta, edit_approval_policy_getter=policy_getter,
                turn_state=turn_state, subagent_router=cbs.subagent_router,
            )
            # Per-session allocator: a new turn must never reuse a previous turn's
            # assistant messageId (ACP clients replace the bubble with that id).
            if state.message_ids is None:
                state.message_ids = AssistantMessageIdAllocator()
            state.message_ids.close()  # new turn -> next chunk opens a fresh id
            cbs.reasoning_cb = make_thinking_cb(conn, session_id, loop, state.message_ids)
            cbs.step_cb = make_step_cb(conn, session_id, loop, tool_call_ids, tool_call_meta, turn_state)
            base_step_cb = cbs.step_cb

            def _step_with_usage(api_call_count: int, prev_tools: Any = None) -> None:
                base_step_cb(api_call_count, prev_tools)
                self._notify_midturn_usage(state, loop)

            cbs.step_cb = _step_with_usage
            message_cb = make_message_cb(conn, session_id, loop, state.message_ids)

            def stream_delta_cb(text: str) -> None:
                cbs.streamed = cbs.streamed or bool(text)
                message_cb(text)

            cbs.stream_delta_cb = stream_delta_cb
            # Closes the synthetic permission-request bubble once the user has answered.
            send_update = lambda update: _send_update(conn, session_id, loop, update)
            cbs.approval_cb = make_approval_callback(conn.request_permission, loop, session_id, send_update=send_update)
            try:
                from acp_adapter.edit_approval import make_acp_edit_approval_requester

                cbs.edit_approval_requester = make_acp_edit_approval_requester(
                    conn.request_permission, loop, session_id, auto_approve_getter=policy_getter,
                    send_update=send_update,
                )
            except Exception:
                logger.debug("Could not create ACP edit approval requester", exc_info=True)

        agent = state.agent
        agent.tool_progress_callback = cbs.tool_progress_cb
        # Thought panes get provider reasoning only — no local status updates, no fake accordion.
        agent.thinking_callback = None
        agent.reasoning_callback, agent.step_callback = cbs.reasoning_cb, cbs.step_cb
        agent.stream_delta_callback = cbs.stream_delta_cb
        return cbs

    async def _finish_turn(self, state: SessionState, session_id: str, conn: Any, result: dict,
                           pre_turn_hermes_id: Any, streamed_message: bool, *,
                           user_content: Any = None, completion_sink: Any = None) -> PromptResponse:
        leftover_steer = result.get("pending_steer")

        # Absolute index of this turn's user message in the post-turn history,
        # in the same coordinate space ``fork_session`` slices and history
        # replay stamps as ``_meta.hermes.historyIndex``. Returned to the client
        # so a "fork from this message" affordance can pass it back verbatim as
        # ``_meta.hermes.keepHistory`` without re-deriving it from replay.
        # Recomputed from the FINALIZED ``result["messages"]`` — the agent's
        # ``_persist_user_message_idx`` is stamped before preflight compression,
        # which can replace the message list mid-turn and leave that early index
        # pointing at the wrong entry (or past a summarized-away message). Read
        # before draining queued prompts, since each drained follow-up rewrites
        # ``_persist_user_message_idx`` for its own turn.
        user_history_index = self._final_user_history_index(
            result.get("messages"),
            user_content,
            getattr(state.agent, "_persist_user_message_idx", None),
        )

        # The whole post-turn tail below runs BEFORE ``state.is_running`` is
        # reset. If anything in it raises, the exception escapes ``prompt()``
        # with the session still marked running — permanently bricking it:
        # every subsequent user message hits the ``state.is_running`` guard
        # and is appended to ``queued_prompts`` ("Queued for the next turn.
        # (N queued)") with no turn ever draining the queue. Seen live when
        # a /stop-style cancel landed mid tool call: the interrupted turn
        # returned ``final_response=None`` and ``None.startswith(...)`` blew
        # up here. Guarantee the idle reset with try/finally.
        try:
            if "messages" in result and isinstance(result["messages"], list):
                # An explicit empty list is authoritative (e.g. reset/transform
                # hooks) and must clear the ACP transcript rather than leave
                # stale pre-turn history behind.
                state.history = result["messages"]
                # Persist updated history so sessions survive process restarts.
                self.session_manager.save_session(session_id)

            # Detect a compression-driven internal session rotation. If the agent's
            # DB head moved during the turn, emit a session_info_update carrying
            # _meta.hermes.sessionProvenance so ACP clients can render the boundary
            # and keep old/new ids in lineage. The ACP session_id is unchanged.
            post_turn_hermes_id = getattr(state.agent, "session_id", None)
            if (
                conn
                and post_turn_hermes_id
                and pre_turn_hermes_id
                and post_turn_hermes_id != pre_turn_hermes_id
            ):
                try:
                    await self._send_session_info_update(
                        session_id,
                        current_hermes_session_id=post_turn_hermes_id,
                        previous_hermes_session_id=pre_turn_hermes_id,
                    )
                except Exception:
                    logger.debug(
                        "Could not emit ACP provenance update after rotation for %s",
                        session_id,
                        exc_info=True,
                    )

            # ``final_response`` is ``None`` (key present) for interrupted /
            # recovery turns — ``.get()``'s default doesn't apply, so coerce.
            final_response = result.get("final_response") or ""
            cancelled = bool(state.cancel_event and state.cancel_event.is_set())
            interrupted = bool(result.get("interrupted")) or cancelled
            # Hermes' local "waiting for model response" interrupt status is metadata,
            # not assistant prose — clients get cancellation from stop_reason instead.
            from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

            suppress_interrupt_response = interrupted and final_response.startswith(
                INTERRUPT_WAITING_FOR_MODEL_PREFIX
            )
            if final_response and not interrupted:
                # The turn produced its answer. From here on there is
                # nothing to salvage: a cancel landing in the post-turn
                # tail (history persistence, provenance updates, queued
                # drain) must not arm the interrupted-prompt buffer with a
                # request that was already fulfilled and delivered — doing
                # so makes the NEXT, unrelated prompt replay finished work.
                with state.runtime_lock:
                    state.response_delivered = True
            if completion_sink is not None:
                completion_sink(
                    final_response,
                    interrupted,
                    "cancelled" if interrupted else "end_turn",
                )
            if (
                final_response
                and conn
                and not suppress_interrupt_response
                and (not streamed_message or result.get("response_transformed"))
            ):
                # Deliver the final response when streaming did not already send it,
                # or when a plugin hook transformed the response after streaming
                # finished (e.g. transform_llm_output) — otherwise the appended /
                # rewritten text never reaches the client.
                update = acp.update_agent_message_text(final_response)
                if state.message_ids is not None:
                    if streamed_message and result.get("response_transformed"):
                        update.message_id = state.message_ids.last() or state.message_ids.current()
                    else:
                        update.message_id = state.message_ids.current()
                    state.message_ids.close()
                await conn.session_update(session_id, update)

        finally:
            # Mark this turn idle before draining queued work so recursive
            # prompt() calls can acquire the session — and so an exception in
            # the tail above can never leave the session stuck busy. Queued
            # turns are intentionally run as normal follow-up user prompts,
            # preserving role alternation and history.
            with state.runtime_lock:
                state.is_running = False
                state.current_prompt_text = ""
            # Flip attached-but-not-owner panels back to idle (see the
            # matching running=True update above). Inside the finally so a
            # tail exception can't strand a client in steer mode.
            try:
                await self._send_turn_status_update(state, running=False)
            finally:
                await self._drain_queued_prompts(state, session_id, conn)

        # A steer arriving after the final tool batch cannot be injected into a
        # tool result. Run it as a follow-up after already-queued prompts instead
        # of dropping it at the ACP boundary. Do not echo it again: the client
        # already rendered the steer bubble when it sent the command.
        if isinstance(leftover_steer, str) and leftover_steer.strip():
            await self.prompt(
                prompt=[TextContentBlock(type="text", text=leftover_steer)],
                session_id=session_id,
            )

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        await self._send_usage_update(state)

        stop_reason = "cancelled" if cancelled else "end_turn"
        field_meta = (
            {"hermes": {"userHistoryIndex": user_history_index}}
            if user_history_index is not None
            else None
        )
        return PromptResponse(stop_reason=stop_reason, usage=usage, field_meta=field_meta)

    async def _drain_queued_prompts(self, state: SessionState, session_id: str, conn: Any) -> None:
        while True:
            with state.runtime_lock:
                if state.is_running or state.command_op or not state.queued_prompts:
                    return
                queued_entry = state.queued_prompts.pop(0)
            if isinstance(queued_entry, QueuedPrompt):
                next_prompt = queued_entry.text
                queued_synthetic = queued_entry.synthetic_notification
                queued_sink = queued_entry.completion_sink
                if isinstance(queued_sink, _TurnCompletionSink):
                    queued_sink.queued = False
            else:
                next_prompt = str(queued_entry)
                queued_synthetic = False
                queued_sink = None
            # The echo is best-effort: the text was already popped from the
            # queue, so a raise here (e.g. ConnectionResetError from a client
            # that disconnected mid-turn) escaping prompt() would LOSE it —
            # neither run nor re-queued. Echo failure must never prevent the
            # queued turn from running.
            if conn:
                try:
                    await conn.session_update(
                        session_id,
                        acp.update_user_message_text(next_prompt),
                    )
                except Exception:
                    logger.debug(
                        "Queued-prompt drain echo failed for %s; running the "
                        "queued turn anyway",
                        session_id,
                        exc_info=True,
                    )
            prompt_kwargs = {}
            if queued_synthetic:
                prompt_kwargs["_synthetic_notification"] = True
            if queued_sink is not None:
                prompt_kwargs["_turn_completion_sink"] = queued_sink
            try:
                await self.prompt(
                    prompt=[TextContentBlock(type="text", text=next_prompt)],
                    session_id=session_id,
                    **prompt_kwargs,
                )
            except BaseException as exc:
                interrupted = isinstance(exc, asyncio.CancelledError)
                completion_entries = [queued_entry]
                with state.runtime_lock:
                    retained_entries = []
                    for entry in state.queued_prompts:
                        if (
                            isinstance(entry, QueuedPrompt)
                            and isinstance(entry.completion_sink, _TurnCompletionSink)
                        ):
                            completion_entries.append(entry)
                        else:
                            retained_entries.append(entry)
                    state.queued_prompts = retained_entries
                _fail_queued_completion_sinks(
                    completion_entries,
                    interrupted=interrupted,
                    error=(
                        "The queued spawned turn was cancelled before delivery."
                        if interrupted
                        else f"Queued spawned turn failed: {type(exc).__name__}: {exc}"[:1000]
                    ),
                )
                raise


    # ---- Session settings (ACP protocol methods) -----------------------------

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = await asyncio.to_thread(self.session_manager.get_session, session_id)
        if state is None:
            logger.warning("Session %s: model switch requested for missing session", session_id)
            return None
        # The picker swaps state.agent wholesale; mid-turn that strands the running agent and
        # makes _finish_turn emit a spurious compression-rotation update. Same exclusion as
        # the /model slash command.
        with state.runtime_lock:
            if state.is_running or state.command_op:
                raise acp.RequestError(-32603, "Session is busy; switch models while the session is idle")
            state.command_op = True
        try:
            # switch_model() does synchronous network I/O (models.dev, custom-endpoint probes,
            # ~10 s cold) — off the loop, like the gateway, so other ACP sessions keep flowing.
            try:
                _old, requested_provider, resolved_model = await asyncio.to_thread(
                    self._switch_model, state, model_id, keep_endpoint=True)
            except ModelRejected as exc:
                # A model no provider can serve is a bad ``modelId`` param (-32602), not an agent
                # internal error (-32603): the client attributes it to the request, not to Hermes (#72439).
                # Only the switch_model rejection maps here; a ValueError from the rebuild itself
                # (disabled provider, context window below the floor) stays on the -32603 path.
                from acp.exceptions import RequestError
                raise RequestError.invalid_params({"details": str(exc)}) from exc
        finally:
            with state.runtime_lock:
                state.command_op = False
            # Drain AFTER this response is queued, never inside it: a prompt that arrived
            # mid-switch would otherwise run a whole turn before the client sees the
            # (possibly failed) switch result.
            self._schedule_soon(lambda: self._drain_queued_prompts(state, session_id, self._conn))
        logger.info(
            "Session %s: model switched to %s via provider %s", session_id, resolved_model, requested_provider
        )
        return SetSessionModelResponse()

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = await asyncio.to_thread(self.session_manager.get_session, session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        normalized_mode = str(mode_id or "").strip()
        if normalized_mode not in self._MODES:
            normalized_mode = self._MODE_DEFAULT
        state.mode = normalized_mode
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, normalized_mode)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Accept ACP config option updates even when Hermes has no typed ACP config surface yet."""
        state = await asyncio.to_thread(self.session_manager.get_session, session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        if str(config_id) == self._EDIT_APPROVAL_POLICY_CONFIG_ID:
            mode = self._EDIT_APPROVAL_POLICY_TO_MODE.get(str(value), self._MODE_DEFAULT)
            setattr(state, "mode", mode)
        elif str(config_id) == self._REASONING_EFFORT_CONFIG_ID:
            level = str(value or "").strip().lower()
            valid = {v for v, _, _ in self._REASONING_EFFORT_LEVELS}
            if level not in valid:
                level = self._REASONING_EFFORT_DEFAULT
            # "default" clears the override; the fresh-agent path (restore/fork/
            # model switch) then simply doesn't touch reasoning_config. On the
            # LIVE agent, restore the provider default explicitly.
            state.effort = "" if level == self._REASONING_EFFORT_DEFAULT else level
            _apply_session_effort(state.agent, level)
            logger.info("Session %s: reasoning effort set to %s", session_id, level)
        else:
            options = getattr(state, "config_options", None)
            if not isinstance(options, dict):
                options = {}
            options[str(config_id)] = value
            setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(
            config_options=self._session_config_options(state)
        )

    def _session_config_options(self, state: SessionState) -> list[SessionConfigOptionSelect]:
        """ACP config options for a session: currently the reasoning-effort select."""
        current = str(getattr(state, "effort", "") or "").strip().lower()
        valid = {value for value, _, _ in self._REASONING_EFFORT_LEVELS}
        if current not in valid:
            current = self._REASONING_EFFORT_DEFAULT
        return [
            SessionConfigOptionSelect(
                id=self._REASONING_EFFORT_CONFIG_ID,
                name="Effort",
                description="Reasoning effort for this session.",
                type="select",
                current_value=current,
                options=[
                    SessionConfigSelectOption(value=value, name=name, description=desc)
                    for value, name, desc in self._REASONING_EFFORT_LEVELS
                ],
            )
        ]

    def _notify_midturn_usage(
        self, state: SessionState, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Refresh the client's context indicator between API calls.

        Called from the agent's step callback in the executor thread.
        ``state.history`` is frozen until turn end, so ``_build_usage_update``'s
        history-based estimate would be stale here; the compressor's
        provider-reported ``last_prompt_tokens`` from the just-finished API
        call is the current request pressure.
        """
        if not self._conn:
            return
        compressor = getattr(state.agent, "context_compressor", None)
        size = int(getattr(compressor, "context_length", 0) or 0)
        used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
        if size <= 0 or used <= 0:
            return
        update = UsageUpdate(session_update="usage_update", size=size, used=used)
        from agent.async_utils import safe_schedule_threadsafe

        safe_schedule_threadsafe(
            self._conn.session_update(session_id=state.session_id, update=update),
            loop,
            logger=logger,
            log_message="Failed to send mid-turn ACP usage update",
        )

    @staticmethod
    def _merge_hermes_meta(
        base: Optional[dict], extra: Optional[dict]
    ) -> Optional[dict]:
        """Merge extra keys into a ``_meta.hermes`` payload without clobbering.

        ``base`` is an existing ``{"hermes": {...}}`` meta dict (or ``None``),
        e.g. the provenance meta; ``extra`` is a flat dict of additional keys
        for the ``hermes`` namespace. Returns ``None`` when there is nothing
        to send.
        """
        if not extra:
            return base
        merged = dict(base or {})
        hermes = dict(merged.get("hermes") or {})
        hermes.update(extra)
        merged["hermes"] = hermes
        return merged

    def _turn_status_meta(self, state: SessionState) -> dict:
        """Snapshot the session's live turn state for ``_meta.hermes``.

        A session's turn can be owned by a prompt() call from an earlier
        client interaction while a NEW client attaches via session/load or
        session/resume. Without this, the attaching client renders an idle
        "ready to prompt" composer over a running turn and a second prompt
        starts a concurrent writer on the same transcript (the 2026-07-15
        incident). Read under ``runtime_lock`` for a consistent pair.
        """
        with state.runtime_lock:
            if not state.is_running:
                return {}
            return {
                "isRunning": True,
                "currentPromptText": state.current_prompt_text,
            }

    def _subagent_load_meta(self, state: SessionState) -> dict:
        """``_meta.hermes`` extras for loading a delegate child session.

        Children are observation-only (isSubagent marks the composer
        read-only client-side). A RUNNING child lives in the delegate tool's
        active registry, not in ``SessionManager._sessions`` — so its
        ``state.is_running`` is always False and liveness must come from
        ``list_active_subagents()``.
        """
        if not getattr(state, "subagent", False):
            return {}
        meta: dict[str, Any] = {"isSubagent": True}
        try:
            from tools.delegate_tool import list_active_subagents

            if any(
                str(rec.get("child_session_id") or "") == state.session_id
                for rec in list_active_subagents()
            ):
                meta["isRunning"] = True
        except Exception:
            logger.debug("Could not check live subagent registry", exc_info=True)
        return meta

    def _resolve_notification_session(self, evt: dict) -> Optional[str]:
        """Resolve the in-memory session id a completion event belongs to.

        Background processes and async delegations push completion events
        onto the in-process ``process_registry.completion_queue``. Another
        editor window is a separate ACP process with its OWN queue, so an
        event on OUR queue belongs to us only if it names a session this
        process currently hosts — in-memory membership in
        ``self.session_manager`` is the ownership boundary (never the DB,
        which is shared across processes and would claim foreign sessions).

        Checks ``session_key`` (the ACP session id bound via
        set_session_vars) then ``origin_ui_session_id`` (set on
        async-delegation events) and returns the FIRST id that maps to an
        in-memory, non-subagent session. Returning the matched id (not just
        a bool) is load-bearing: the drain cycle uses it as the delivery
        target, so ownership and targeting can never diverge — previously
        ownership could match on ``session_key`` while delivery targeted a
        foreign/stale ``origin_ui_session_id`` and warn-dropped the event.
        A match counts only when the session is not a delegate child
        (``state.subagent`` falsy) — children are observation-only and
        never receive notifications here.

        Fails closed: empty/missing keys, unknown ids, subagent sessions,
        or any internal error → None. Must never raise.
        """
        try:
            for key in ("session_key", "origin_ui_session_id"):
                sid = str(evt.get(key) or "").strip()
                if not sid:
                    continue
                state = self.session_manager.get_in_memory(sid)
                if state is not None and not getattr(state, "subagent", False):
                    return sid
        except Exception:
            logger.debug("Ownership check failed for event %r", evt, exc_info=True)
        return None

    def _owns_notification_event(self, evt: dict) -> bool:
        """Positive-proof ownership check for ``completion_queue`` events.

        Thin predicate over ``_resolve_notification_session`` (see there
        for the full ownership contract) — kept as a separate bool-returning
        method because it is passed as the ``owns_event`` callback to
        ``process_registry.drain_notifications``.
        """
        return self._resolve_notification_session(evt) is not None

    async def _send_turn_status_update(self, state: SessionState, running: bool) -> None:
        """Notify the client that this session's turn started or ended.

        Rides ``session_info_update`` with ``_meta.hermes.isRunning`` so a
        panel that did NOT start the turn (it attached mid-turn via
        session/load) can flip its composer between running/steer and idle
        states. Clients that don't know the meta ignore a title-less info
        update. Best-effort: failures must never affect the turn.

        ``adoptableTurn`` marks this as a REAL agent turn on a session this
        server owns, which a durable-turn proxy (the YallaPlay ACP broker)
        may adopt into its own per-session turn state when the turn was
        started server-side rather than by a client ``turn/start`` — a
        spawned session's first turn, a background notification delivery, a
        queued-prompt drain. Without it those turns are invisible to the
        proxy: no ring buffering while detached, no cancel-and-grace on
        reap, no liveness in its status probes.

        The flag is deliberately NOT inferable from ``isRunning`` alone:
        ``SubagentUpdateRouter`` emits the same title-less isRunning
        carriers for delegate CHILD sessions, which must never acquire proxy
        turn state (the child's transcript is owned by the delegate tool, and
        its frames are relay-only by design). Only this method sets it.
        """
        if not self._conn:
            return
        update = SessionInfoUpdate(
            session_update="session_info_update",
            updated_at=datetime.now(timezone.utc).isoformat(),
            field_meta={"hermes": {"isRunning": bool(running), "adoptableTurn": True}},
        )
        try:
            await self._conn.session_update(session_id=state.session_id, update=update)
        except Exception:
            logger.debug(
                "Could not send ACP turn status update for %s", state.session_id, exc_info=True
            )


    @staticmethod
    def _flatten_history_text(value: Any) -> str:
        """Normalize a persisted text-or-text-parts value into a single string.

        OpenAI-style assistant content (and provider reasoning fields) can arrive
        as either a scalar string or a list of ``{"text": ...}`` /
        ``{"type": "text", "content": ...}`` parts. Whitespace-only inputs
        collapse to an empty string so callers can treat ``""`` as "nothing to
        emit".
        """
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""

    @classmethod
    def _history_message_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable text from a persisted OpenAI-style message."""
        return cls._flatten_history_text(message.get("content"))

    @classmethod
    def _history_reasoning_text(cls, message: dict[str, Any]) -> str:
        """Extract displayable reasoning/thought text from a persisted assistant message.

        Returns the first non-empty value among ``reasoning_content`` (the
        canonical field used by DeepSeek / Moonshot and the post-#16892
        chat-completions normalizer) and ``reasoning`` (used by the codex
        event projector and several other transports). Both keys are
        actively written by live code paths, so neither branch is
        deprecated — they cover different transports rather than old vs.
        new sessions.
        """
        for key in ("reasoning_content", "reasoning"):
            text = cls._flatten_history_text(message.get(key))
            if text:
                return text
        return ""

    @staticmethod
    def _history_summary_meta(message: dict[str, Any], text: str) -> dict[str, Any] | None:
        """Build the ``_meta`` payload for a replayed compaction summary.

        Compaction summaries are persisted as ordinary history messages —
        standalone handoffs under ``role="user"`` OR ``role="assistant"``
        (the compressor picks whichever role keeps alternation valid), and
        merge-into-tail messages where the summary is appended after the
        first preserved tail message's real content. Without a wire flag,
        ACP frontends render all of these as ordinary turns.

        Two distinct keys under ``_meta.hermes`` (ACP's extensibility
        channel), so clients cannot accidentally hide real content:

        * ``compactionSummary: true`` — the entire chunk is the handoff
          summary. Safe to restyle or collapse wholesale.
        * ``containsCompactionSummary: true`` — a merged-tail message: real
          preserved turn content followed by the summary. Clients may style
          it, but collapsing the whole chunk would hide the preserved
          content, hence the separate key.

        Detection honors the in-process ``_compressed_summary`` flag and
        falls back to content classification, so it also works for a
        DB-reloaded session that lost the in-memory flag.
        """
        kind = ContextCompressor.classify_summary_content(text)
        if kind is None and message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            # Flagged in-process but content didn't classify (e.g. future
            # prefix drift): treat as a standalone summary — the flag is only
            # ever set on summary-bearing messages.
            kind = "standalone"
        if kind == "standalone":
            return {"hermes": {"compactionSummary": True}}
        if kind == "merged":
            return {"hermes": {"containsCompactionSummary": True}}
        return None

    @staticmethod
    def _history_timestamp_iso(message: dict[str, Any]) -> str | None:
        """Return the persisted message timestamp as an ISO-8601 UTC string.

        ``get_messages_as_conversation`` surfaces the ``messages.timestamp``
        column (a ``time.time()`` epoch float) as ``message["timestamp"]``
        when the row has one. Returns ``None`` for messages without a
        usable timestamp (in-memory history entries the DB hasn't stamped,
        or corrupt values) so callers can omit the meta key entirely.
        """
        raw = message.get("timestamp")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
            return None
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    @classmethod
    def _history_chunk_meta(
        cls, message: dict[str, Any], text: str
    ) -> dict[str, Any] | None:
        """Combine the compaction-summary flags and the message timestamp
        into one chunk-level ``_meta.hermes`` payload for history replay.

        Returns ``None`` when neither applies, keeping unflagged,
        untimestamped chunks meta-free (the shape existing clients expect).
        """
        meta = cls._history_summary_meta(message, text)
        timestamp = cls._history_timestamp_iso(message)
        if timestamp:
            if meta is None:
                meta = {"hermes": {}}
            meta["hermes"]["timestamp"] = timestamp
        return meta

    @staticmethod
    def _history_message_update(
        *,
        role: str,
        text: str,
        field_meta: dict[str, Any] | None = None,
    ) -> UserMessageChunk | AgentMessageChunk | None:
        """Build an ACP history replay update for a user/assistant message."""
        block = TextContentBlock(type="text", text=text)
        if role == "user":
            return UserMessageChunk(
                session_update="user_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        if role == "assistant":
            return AgentMessageChunk(
                session_update="agent_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        return None

    @staticmethod
    def _history_image_update(
        *,
        role: str,
        data: bytes,
        mime_type: str,
        field_meta: dict[str, Any] | None = None,
    ) -> UserMessageChunk | AgentMessageChunk | None:
        """Build an ACP history replay update carrying one image.

        Chunked separately from the text so clients that only understand text
        blocks keep rendering the transcript unchanged (they ignore the image
        chunk) while image-capable clients attach it to the same bubble.
        """
        block = ImageContentBlock(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime_type or "image/png",
        )
        if role == "user":
            return UserMessageChunk(
                session_update="user_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        if role == "assistant":
            return AgentMessageChunk(
                session_update="agent_message_chunk",
                content=block,
                field_meta=field_meta,
            )
        return None

    @classmethod
    def _history_message_images(cls, message: dict[str, Any]) -> list[tuple[bytes, str]]:
        """Return replayable ``(bytes, mime)`` images for a history message.

        Two sources, in order:

        1. Image parts still on the in-memory content list (this process ran
           the turn, so the original base64 payload is right there).
        2. ``display_metadata["images"]`` references written by the flush path,
           read back from the image cache — the only source for a DB-sourced
           replay (mid-turn, or a session this process never ran), where
           content is already collapsed to ``[screenshot]`` text.

        Never raises: a missing/expired cache file just yields no image, and
        the text placeholder still explains what was there.
        """
        inline = inline_image_bytes_from_content(
            message.get("content"), max_bytes=_MAX_ACP_RESOURCE_BYTES
        )
        if inline:
            return inline
        images: list[tuple[bytes, str]] = []
        for ref in image_refs_from_message(message):
            loaded = image_bytes_from_ref(ref, max_bytes=_MAX_ACP_RESOURCE_BYTES)
            if loaded is not None:
                images.append(loaded)
        return images

    @staticmethod
    def _history_thought_update(text: str) -> AgentThoughtChunk:
        """Build an ACP history replay update for an assistant thought."""
        return acp.update_agent_thought_text(text)

    @staticmethod
    def _history_tool_call_name_args(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract function name/arguments from an OpenAI-style tool_call."""
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(function.get("name") or tool_call.get("name") or "unknown_tool")
        raw_args = function.get("arguments") or tool_call.get("arguments") or tool_call.get("args") or {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception:
                parsed = {"raw": raw_args}
            raw_args = parsed
        if not isinstance(raw_args, dict):
            raw_args = {}
        return name, raw_args

    @staticmethod
    def _history_tool_call_id(tool_call: dict[str, Any]) -> str:
        """Return the stable provider tool call id for ACP history replay."""
        return str(
            tool_call.get("id")
            or tool_call.get("call_id")
            or tool_call.get("tool_call_id")
            or ""
        ).strip()

    @staticmethod
    def _stable_history_prefix_len(
        history: list[dict[str, Any]], transcript: list[dict[str, Any]]
    ) -> int:
        """Length of the leading run where the DB transcript still matches
        ``state.history``.

        ``fork_session`` slices ``state.history``, so indices into this
        matching prefix are valid fork coordinates even while a turn is
        running. Rows are compared by (role, content) — enough to detect a
        mid-turn compaction that rewrote older rows, at which point stamping
        stops at the first divergence.
        """
        limit = min(len(history), len(transcript))
        for i in range(limit):
            h, t = history[i], transcript[i]
            if str(h.get("role") or "") != str(t.get("role") or ""):
                return i
            if h.get("content") != t.get("content"):
                return i
        return limit

    def _message_count_meta(self, state: SessionState) -> dict:
        """Baseline for the client's out-of-process refresh check.

        Pairs with ``messageCount`` on ``session/list`` rows: the client
        re-replays an open tab when the listed count differs from the count
        its last replay rendered.
        """
        count = self.session_manager.persisted_message_count(state.session_id)
        return {} if count is None else {"messageCount": count}

    @staticmethod
    def _keep_history_meta(kwargs: dict[str, Any]) -> int | None:
        """Extract the optional ``_meta.hermes.keepHistory`` rewind point.

        Neither ``session/fork`` nor ``session/prompt`` has a spec-level
        rewind parameter, so clients pass it through ACP's extensibility
        escape hatch: ``_meta.hermes.keepHistory`` is the number of leading
        history entries to keep (in the same OpenAI-message coordinate space
        that ``session/load`` replays). On fork it limits the copy; on prompt
        it truncates the session in place before the turn runs ("edit &
        resend"). The JSON-RPC router flattens ``_meta`` keys into handler
        kwargs, so the payload arrives here as a ``hermes`` dict. Absent →
        ``None`` → full history (the pre-existing behavior). Invalid
        types/values raise ``invalid_params`` rather than silently keeping
        everything — a client asking for a rewind must not quietly get a
        full copy.
        """
        hermes_meta = kwargs.get("hermes")
        if not isinstance(hermes_meta, dict) or "keepHistory" not in hermes_meta:
            return None
        keep = hermes_meta["keepHistory"]
        # bool is an int subclass; reject it explicitly.
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
            raise acp.RequestError.invalid_params(
                {"reason": "_meta.hermes.keepHistory must be a non-negative integer"}
            )
        return keep

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle Hermes ACP extension methods (client sends `_<method>`)."""
        if method == "setArchived":
            session_id = params.get("sessionId")
            archived = bool(params.get("archived"))
            ok = self.session_manager.set_session_archived(session_id, archived) if session_id else False
            return {"ok": bool(ok)}
        if method == "setOwner":
            # Stamp (or clear, with an empty string) a session's owner
            # (sessions.user_id) — the key the sessions list filters on so a
            # user sees only their own sessions. Soft per-user display key, not
            # an access boundary. Mirrors setArchived; emit an info update so the
            # client sidebar refreshes.
            session_id = params.get("sessionId")
            owner = str(params.get("owner") or "")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            ok = self.session_manager.set_session_owner(session_id, owner)
            if ok:
                await self._send_session_info_update(session_id)
            return {"ok": bool(ok)}
        if method == "promoteSession":
            # Detach a nested ACP fork from its display parent. This preserves
            # the full transcript and routing; it only removes fork lineage so
            # session/list returns the row as a top-level session.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            try:
                ok = self.session_manager.promote_session(session_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if ok:
                await self._send_session_info_update(session_id)
            return {"ok": bool(ok)}
        if method == "setTitle":
            # Set (or clear, with an empty string) a session's CANONICAL title in
            # state.db — the thing session/list, the web UI, sessions-list CLI,
            # and search all read. This is the write path ACP core lacks; the
            # extension's inline rename routes here instead of a VS-Code-local
            # override, so the name is global. Surface a uniqueness conflict as a
            # visible error rather than a silent no-op.
            session_id = params.get("sessionId")
            title = str(params.get("title") or "")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            try:
                ok = self.session_manager.set_session_title(session_id, title)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if ok:
                await self._send_session_info_update(session_id)
            return {"ok": bool(ok), "title": self.session_manager.get_session_title(session_id)}
        if method == "deriveTitle":
            # On-demand backfill for an untitled session (e.g. one that opened
            # into a long first turn and missed the first-exchange auto-title).
            # Reuses the auxiliary-model pass; no-op if already titled or no
            # completed exchange. The aux call is blocking, so run it off the
            # event loop, then push the new title to the client.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            loop = asyncio.get_running_loop()
            title = await loop.run_in_executor(
                None, self.session_manager.derive_session_title, session_id
            )
            if title:
                await self._send_session_info_update(session_id)
            return {"ok": bool(title), "title": title}
        if method == "contextBreakdown":
            # Read-only per-category context-window composition for the client's
            # usage popover. Wraps the same engine the desktop popover and the
            # gateway /usage use; estimates align with usage_update numbers.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            state = await asyncio.to_thread(self.session_manager.get_session, session_id)
            if state is None:
                return {"ok": False, "error": "session not loaded"}
            from agent.context_breakdown import compute_session_context_breakdown

            try:
                breakdown = compute_session_context_breakdown(state.agent, state.history)
            except Exception as exc:  # engine failure must not kill the RPC
                logger.debug("contextBreakdown failed", exc_info=True)
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "breakdown": breakdown}
        if method == "contextReport":
            # Read-only markdown render of the actual composed context — the
            # full text behind a breakdown category (or all of them). Backs the
            # client popover's click-to-view / export-as-markdown actions.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            state = await asyncio.to_thread(self.session_manager.get_session, session_id)
            if state is None:
                return {"ok": False, "error": "session not loaded"}
            from agent.context_breakdown import build_session_context_report

            try:
                report = build_session_context_report(
                    state.agent, state.history, category=params.get("category")
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            except Exception as exc:  # engine failure must not kill the RPC
                logger.debug("contextReport failed", exc_info=True)
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "report": report}
        if method == "accountUsage":
            # Read-only provider account limits (e.g. Codex session/weekly
            # rate-limit windows, Anthropic OAuth usage windows) for the
            # client's usage popover. Wraps the same fetch the gateway /usage
            # command uses. Account-level, not per-session — the sessionId only
            # picks which agent's provider credentials to use. The fetch is a
            # blocking httpx call, so run it off the event loop.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            state = await asyncio.to_thread(self.session_manager.get_session, session_id)
            if state is None:
                return {"ok": False, "error": "session not loaded"}
            from agent.account_usage import fetch_account_usage

            provider = getattr(state.agent, "provider", None)
            if not provider:
                return {"ok": True, "usage": None}
            try:
                snapshot = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: fetch_account_usage(
                        provider,
                        base_url=getattr(state.agent, "base_url", None),
                        api_key=getattr(state.agent, "api_key", None),
                    ),
                )
            except Exception as exc:  # fetch failure must not kill the RPC
                logger.debug("accountUsage fetch failed", exc_info=True)
                return {"ok": False, "error": str(exc)}
            if snapshot is None or not snapshot.available:
                return {"ok": True, "usage": None}
            return {
                "ok": True,
                "usage": {
                    "provider": snapshot.provider,
                    "plan": snapshot.plan,
                    "fetchedAt": snapshot.fetched_at.isoformat(),
                    "windows": [
                        {
                            "label": w.label,
                            "usedPercent": w.used_percent,
                            "resetAt": w.reset_at.isoformat() if w.reset_at else None,
                            "detail": w.detail,
                        }
                        for w in snapshot.windows
                    ],
                    "details": list(snapshot.details),
                },
            }
        if method == "refreshModels":
            # Rebuild the cross-provider model catalog, bypassing the TTL
            # cache, and return a fresh model-selector payload. Lets clients
            # pull the live list when the user opens the model dropdown, so a
            # provider authenticated after session creation (e.g. `hermes auth
            # add openai-codex`) shows up without a window reload. The rebuild
            # hits every authenticated provider's /models endpoint (slow,
            # blocking), so it runs off the event loop.
            session_id = params.get("sessionId")
            if not session_id:
                return {"ok": False, "error": "sessionId required"}
            state = await asyncio.to_thread(self.session_manager.get_session, session_id)
            if state is None:
                return {"ok": False, "error": "session not loaded"}
            try:
                model_state = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._build_model_state(state, force_refresh=True)
                )
            except Exception as exc:  # catalog failure must not kill the RPC
                logger.debug("refreshModels failed", exc_info=True)
                return {"ok": False, "error": str(exc)}
            if model_state is None:
                return {"ok": True, "models": None}
            return {
                "ok": True,
                "models": {
                    "currentModelId": model_state.current_model_id,
                    "availableModels": [
                        {
                            "modelId": m.model_id,
                            "name": m.name,
                            "description": m.description,
                        }
                        for m in model_state.available_models
                    ],
                },
            }
        raise RequestError.method_not_found(f"_{method}")

    _TITLE_BACKFILL_MAX_PER_LIST = 3

    def _schedule_title_backfill(self, infos: list[dict[str, Any]]) -> None:
        """Kick off background title derivation for untitled listed sessions.

        Each candidate gets one attempt per process (derive failures — no
        completed exchange yet, aux-model errors — are not retried on every
        list refresh). The blocking aux call runs off the event loop;
        successes push a session_info update so open sidebars re-render.
        Skips subagent children — their transcripts are delegate work
        products, not user threads worth an aux call.
        """
        candidates = [
            s["session_id"]
            for s in infos
            if s.get("untitled") and not s.get("subagent")
            and s["session_id"] not in self._title_backfill_attempted
        ][: self._TITLE_BACKFILL_MAX_PER_LIST]
        if not candidates:
            return
        self._title_backfill_attempted.update(candidates)
        loop = asyncio.get_running_loop()

        async def _backfill(session_id: str) -> None:
            try:
                title = await loop.run_in_executor(
                    None, self.session_manager.derive_session_title, session_id
                )
                if title:
                    await self._send_session_info_update(session_id)
            except Exception:
                logger.debug(
                    "Background title backfill failed for %s", session_id, exc_info=True
                )

        for sid in candidates:
            asyncio.create_task(_backfill(sid))

    async def _run_spawned_first_turn(
        self,
        session_id: str,
        prompt_text: str,
        parent_session_id: str | None = None,
        deliver_result_to_parent: bool = False,
    ) -> None:
        """Run a freshly spawned session's first turn as a background task.

        Echoes the prompt as a user-message update first (mirrors the queued-
        prompt drain in ``prompt``) so a client that subscribes early sees the
        opening message, then runs the normal prompt path — streaming, turn
        status, auto-title and persistence all behave exactly like a typed
        turn. Exceptions are logged, never raised: the spawning session's
        tool call already returned.
        """
        sink: _TurnCompletionSink | None = None
        if deliver_result_to_parent:
            def _emit_completion(
                final_response: str,
                interrupted: bool,
                stop_reason: str,
                error: str | None,
            ) -> None:
                status = "cancelled" if interrupted or stop_reason == "cancelled" else "completed"
                result_text = str(final_response or "")
                if status == "completed" and not result_text.strip():
                    status = "failed"
                    error = error or "No final assistant response was produced."
                if len(result_text) > _ACP_SPAWN_RESULT_MAX_CHARS:
                    result_text = (
                        result_text[:_ACP_SPAWN_RESULT_MAX_CHARS]
                        + "\n\n[Child result truncated for parent delivery; inspect the child session for the full response.]"
                    )
                child_title = self.session_manager.get_session_title(session_id)
                try:
                    from tools.process_registry import process_registry
                    from tools.session_search_tool import _session_link

                    process_registry.completion_queue.put(
                        {
                            "type": "acp_spawn_completion",
                            "session_key": parent_session_id,
                            "child_session_id": session_id,
                            "child_title": child_title,
                            "child_session_link": _session_link(session_id),
                            "status": status,
                            "result": result_text,
                            "error": error,
                        }
                    )
                except Exception:
                    logger.exception(
                        "acp_spawn_session: could not enqueue completion for %s",
                        session_id,
                    )

            sink = _TurnCompletionSink(_emit_completion)

        try:
            if self._conn:
                await self._conn.session_update(
                    session_id, acp.update_user_message_text(prompt_text)
                )
            response = await self.prompt(
                prompt=[TextContentBlock(type="text", text=prompt_text)],
                session_id=session_id,
                _turn_completion_sink=sink,
            )
            stop_reason = getattr(response, "stop_reason", "end_turn")
            stop_reason = getattr(stop_reason, "value", stop_reason)
            if sink is not None and not sink.delivery_emitted and not sink.queued:
                sink(
                    "",
                    stop_reason == "cancelled",
                    str(stop_reason),
                    "The spawned turn ended before producing a final assistant response.",
                )
        except asyncio.CancelledError:
            if sink is not None and not sink.delivery_emitted and not sink.queued:
                sink("", True, "cancelled", "The spawned turn was cancelled.")
            raise
        except Exception as exc:
            logger.exception("Spawned ACP session %s first turn failed", session_id)
            if sink is not None and not sink.delivery_emitted and not sink.queued:
                detail = f"{type(exc).__name__}: {exc}"
                sink("", False, "failed", detail[:1000])

    async def _deliver_notification(self, session_id: str, text: str) -> None:
        """Deliver a background-completion notification to a session.

        The delivery primitive behind async delivery under ACP: given an
        owned session id and an already-formatted notification text (e.g.
        "[IMPORTANT: Background process finished ...]"), inject it as a
        synthetic turn. Two paths:

        - **Busy** (``state.is_running`` under ``runtime_lock``): append to
          ``state.queued_prompts`` and return SILENTLY — no client
          session_update ack, because synthetic events must not masquerade
          as user input or trigger the "Queued for the next turn" ack. The
          post-turn queued-prompt drain in ``prompt()`` will run it.
          Internal synthetic events must never interrupt/steer a running
          turn (explicit gateway precedent: gateway/run.py's completion
          drain queues instead of steering).
        - **Idle**: echo the text as a user-message update so the
          transcript shows why a turn started, then run a normal prompt
          turn — mirroring ``_run_spawned_first_turn`` exactly (echo is
          skipped when no client is attached; the turn still runs and its
          transcript persists for reattach).

        The prompt is submitted in ``prompt()``'s internal notification
        mode (``_synthetic_notification=True``): no slash/steer parsing,
        no ``interrupted_prompt_text`` merge (a notification must not
        resurrect a user-cancelled prompt), and — critically — the busy
        branch queues SILENTLY instead of redirecting. That makes the
        TOCTOU window between releasing ``runtime_lock`` and ``prompt()``'s
        own ``is_running`` re-check genuinely benign: if a user prompt
        starts a turn in that window, the worst case is the notification
        text landing in ``queued_prompts`` with no client ack — it can
        never abort or steer the user's turn. As belt-and-braces, a
        leading "/" in the text is neutralized with a framing prefix so
        the queued copy can't be command-parsed when the post-turn drain
        replays it as a normal follow-up prompt.

        Never raises: unknown session ids log-and-return, and any failure
        in the echo/prompt path is logged — the caller (the watcher loop)
        must survive any single delivery.
        """
        state = self.session_manager.get_in_memory(session_id)
        if state is None:
            logger.warning(
                "ACP notify: no in-memory session %s; dropping notification",
                session_id,
            )
            return
        # Neutralize a leading "/" so the text can never be intercepted as a
        # slash command — not here (notification mode already skips parsing)
        # but by the queued-prompt drain, which replays queued text through
        # prompt() as a NORMAL user turn. A plain leading space would not
        # survive prompt()'s user_text.strip(), so use a visible framing
        # prefix instead; the original text is preserved verbatim after it.
        if text.lstrip().startswith("/"):
            text = "[notification] " + text
        with state.runtime_lock:
            if state.is_running:
                state.queued_prompts.append(
                    QueuedPrompt(text=text, synthetic_notification=True)
                )
                return
        try:
            # The echo is best-effort on its own: an attached-but-dying conn
            # raising here must not skip the delivery turn (the notification
            # would be silently lost). The prompt below still runs; the turn's
            # transcript persists either way.
            if self._conn:
                try:
                    await self._conn.session_update(
                        session_id, acp.update_user_message_text(text)
                    )
                except Exception:
                    logger.debug(
                        "ACP notify: idle-path echo failed for %s; running the "
                        "delivery turn anyway",
                        session_id,
                        exc_info=True,
                    )
            await self.prompt(
                prompt=[TextContentBlock(type="text", text=text)],
                session_id=session_id,
                _synthetic_notification=True,
            )
        except Exception:
            logger.exception("ACP notify: delivery turn failed for %s", session_id)

    @staticmethod
    def _completion_output_seen_after_spawn(state: SessionState, evt: dict) -> bool:
        """Whether a successful process completion repeats evidence already seen.

        A process notification may finish just after the agent independently
        fetched the same result. Starting another full-context turn for that
        duplicate payload is wasteful. Suppression requires positive provenance:

        1. the event is a successful plain process completion;
        2. the process id appears in an earlier tool result (the spawn boundary);
        3. a later tool result contains either the complete output or an exact
           structured status line proving ``state=complete`` and ``exit_code=0``.

        Old matching text before the spawn boundary, user/assistant prose, empty
        output, arbitrary partial overlap, and failures never qualify. If history
        was compacted or the evidence is ambiguous, return False and preserve
        normal delivery.
        """
        if evt.get("type") != "completion" or evt.get("exit_code") != 0:
            return False
        process_id = str(evt.get("session_id") or "").strip()
        output = str(evt.get("output") or "")
        if not process_id or not output.strip():
            return False

        messages = getattr(state.agent, "_session_messages", None)
        if not isinstance(messages, list):
            return False

        def _content_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            try:
                return json.dumps(content, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                # Defensive only: model-facing tool content should be JSON-safe,
                # but a malformed/cyclic test double must never break delivery.
                return repr(content)

        def _appears_in(candidate: str, haystack: str) -> bool:
            encoded = json.dumps(candidate, ensure_ascii=False)[1:-1]
            return candidate in haystack or encoded in haystack

        spawn_idx = None
        for idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            text = _content_text(message.get("content", ""))
            if process_id in text:
                spawn_idx = idx
                break
        if spawn_idx is None:
            return False

        later_tool_text = []
        for message in messages[spawn_idx + 1 :]:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            later_tool_text.append(_content_text(message.get("content", "")))
        haystack = "\n".join(later_tool_text)
        if not haystack:
            return False

        complete_output = output.strip()
        if len(complete_output) >= 32 and _appears_in(complete_output, haystack):
            return True

        for raw_line in output.splitlines():
            line = raw_line.strip()
            if len(line) < 16 or not line.startswith("{"):
                continue
            try:
                status = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(status, dict):
                continue
            state_value = str(status.get("state") or status.get("status") or "").lower()
            if state_value not in {"complete", "completed", "done", "success", "succeeded"}:
                continue
            if status.get("exit_code") != 0:
                continue
            if _appears_in(line, haystack):
                return True
        return False

    async def _drain_notification_cycle(self) -> None:
        """One drain pass over the process registry's completion queue.

        Pops every event this server can positively prove it owns (see
        ``_owns_notification_event`` — foreign events are re-queued by
        ``drain_notifications`` itself) and delivers each into its session
        via ``_deliver_notification``.

        Delivery protocol: resolve target from ownership → require the target
        IDLE → deliver. Busy targets are requeued so process wait/log can
        still consume ordinary completions before they become synthetic turns.
        Delegation events then claim → complete → deliver for at-most-once
        delivery. See the inline comments for why each step is ordered this way.
        """
        if not self._conn:
            # No client attached — leave events queued for when one is.
            return
        # Local imports match the codebase style and keep the registry /
        # delegation-store import cost off module load.
        from tools.process_registry import process_registry
        from tools.async_delegation import (
            claim_event_delivery,
            complete_event_delivery,
        )

        consumer = f"acp:{os.getpid()}"
        for evt, text in process_registry.drain_notifications(
            owns_event=self._owns_notification_event
        ):
            # The delivery target IS the session ownership matched — never a
            # recomputed origin_ui_session_id-or-session_key expression, which
            # could pick a foreign/stale id when ownership matched on the
            # OTHER key (probe-confirmed: claimed event warn-dropped inside
            # _deliver_notification, then falsely acked 'delivered').
            target = self._resolve_notification_session(evt)
            if target is None:
                # Session evicted between drain's ownership check and here.
                # Requeue for the next 2s cycle instead of warn-dropping —
                # and for delegation events, critically, WITHOUT claiming
                # (a claim we then complete would mark a drop 'delivered').
                # Requeue mirrors drain_notifications' own foreign-event
                # requeue: put THIS event back and keep iterating the rest.
                # No fast-spin: the next pop happens next cycle (2s later).
                process_registry.completion_queue.put(evt)
                continue
            state = self.session_manager.get_in_memory(target)
            if state is not None and state.is_running:
                # Keep the raw event retractable while its target turn runs.
                # In particular, process wait/log may consume a completion
                # before the next drain; drain_notifications can then suppress
                # it instead of leaving a stale synthetic queued prompt.
                process_registry.completion_queue.put(evt)
                continue
            if state is not None and self._completion_output_seen_after_spawn(state, evt):
                logger.info(
                    "ACP notify: suppressed duplicate completion for %s after "
                    "matching process output already seen by the agent",
                    evt.get("session_id") or "unknown",
                )
                continue
            if evt.get("type") == "async_delegation":
                # Durable delegation completions can be restored into EVERY
                # process's completion_queue at registry startup, so the
                # durable claim is the ONLY double-delivery guard: never
                # deliver one without a successful claim. claim=None means
                # another process holds (or already completed) delivery —
                # skip silently.
                claim = claim_event_delivery(evt, consumer)
                if claim is None:
                    continue
                # complete BEFORE deliver — deliberate at-most-once. The
                # delivery turn below can run for MINUTES, and
                # claim_event_delivery lets any other consumer STEAL a claim
                # older than 300s: complete-after-deliver provably
                # double-delivers when a gateway/TUI restarts mid-turn
                # (probe-confirmed: attempts == 2). Completing first shrinks
                # the claim hold to milliseconds, matching every other
                # consumer. It costs no honesty: the target was just verified
                # in-memory and idle, and _deliver_notification never raises,
                # so a post-delivery completion could not detect failure
                # anyway. Benign TOCTOU remains: if a user turn starts
                # between the idle check above and the delivery below, the
                # text lands in RAM queued_prompts after the row was
                # completed — but that window is milliseconds (vs a whole
                # turn before this fix) and the busy-requeue above keeps it
                # from ever spanning a full turn.
                complete_event_delivery(evt, claim)
                await self._deliver_notification(target, text)
            else:
                # Plain completion/watch_match/watch_disabled events are
                # in-process only — no cross-process race, no durable row,
                # no claim needed. The shared busy check above keeps completion
                # events retractable until the target session is idle.
                await self._deliver_notification(target, text)

    async def _notification_watcher(self) -> None:
        """Poll loop: drain completion events into sessions every interval.

        One failing cycle must never kill the loop — everything except
        cancellation is logged and the loop keeps going.
        """
        while True:
            try:
                await self._drain_notification_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ACP notification watcher cycle failed")
            await asyncio.sleep(self._NOTIFY_DRAIN_INTERVAL_SEC)

    def _ensure_notification_watcher(self) -> None:
        """Idempotently start (or resurrect) the notification watcher task.

        Safe to call from sync, no-loop contexts (e.g. on_connect in sync
        tests): with no running loop the start is skipped — prompt() calls
        this again from loop context, so the watcher still comes up.
        """
        task = self._notification_watcher_task
        if task is not None and not task.done():
            return
        try:
            self._notification_watcher_task = asyncio.get_running_loop().create_task(
                self._notification_watcher()
            )
        except RuntimeError:
            # No running event loop — defer to the next loop-context call.
            pass

    def _make_spawn_session_requester(
        self, loop: asyncio.AbstractEventLoop, parent_state: SessionState
    ):
        """Build the sync requester backing the ``acp_spawn_session`` tool.

        Runs on the executor thread of the *parent* session's turn. Creates
        the new session synchronously (SessionManager is thread-safe), then
        schedules its first turn as a background task on the server's event
        loop and returns the real session id immediately — the parent turn
        never blocks on the child. The child is a visibly derived session with
        clean context: it records display lineage to the parent but copies no
        history. It inherits the parent's owner (and cwd unless overridden)
        so per-user session lists show it, plus the parent's edit-approval mode
        and reasoning effort. A spawned session runs its first turn headless
        (no panel attached), so under the default "ask" policy every edit-
        approval request would be silently auto-denied by the client —
        inheriting the parent's mode makes the child at least as capable as
        the session that spawned it. A caller-provided title is stamped best-
        effort before the first turn starts, with #N lineage dedup on collision.
        """

        def _requester(
            prompt_text: str,
            cwd: str | None,
            title: str | None = None,
            provider: str | None = None,
            model: str | None = None,
            *,
            # False by default: a spawned session records no parent and owns its work.
            # See acp_adapter/spawn.py for why routing the result back is wrong here.
            deliver_result_to_parent: bool = False,
        ) -> str:
            from agent.async_utils import safe_schedule_threadsafe

            spawn_cwd = cwd or parent_state.cwd
            # Mirror the fork path (SessionManager.fork_session): carry the
            # parent's FULL model route — model + provider + base_url +
            # api_mode — so the child runs the same model as the session
            # that spawned it instead of silently resolving the config
            # default (a bedrock/fable child of an openai-codex/gpt parent).
            parent_agent = parent_state.agent
            # Requested (pre-canonicalization) identity: bare ``custom`` would
            # re-resolve to the generic env/config custom provider
            # (OpenRouter) instead of the parent's named endpoint.
            parent_provider = _agent_provider_identity(parent_agent)
            spawn_provider = provider or parent_provider
            spawn_model = model or parent_state.model or None
            provider_changed = bool(provider and provider != parent_provider)
            new_state = self.session_manager.create_session(
                cwd=spawn_cwd,
                owner=parent_state.owner,
                model=spawn_model,
                requested_provider=spawn_provider,
                # A base URL and API mode describe the provider transport.
                # Keep them for inherited/same-provider routes, but never leak
                # parent transport metadata into an explicit provider change.
                base_url=(
                    None if provider_changed else getattr(parent_agent, "base_url", None)
                ),
                api_mode=(
                    None if provider_changed else getattr(parent_agent, "api_mode", None)
                ),
                parent_id=parent_state.session_id,
            )
            # Also carry the parent's edit-approval mode and reasoning-effort
            # override, then re-persist so the whole tuple survives an agent
            # restart.
            new_state.mode = parent_state.mode
            new_state.effort = parent_state.effort
            new_state.persist_empty = True
            _apply_effort_to_agent(new_state.agent, parent_state.effort)
            self.session_manager.save_session(new_state.session_id)
            # Stamp the caller-provided title (mirrors fork_session's lineage
            # stamp). The spawning agent knows what the work is, so its title
            # beats the generic post-first-turn auto-title — which never
            # clobbers an existing title (set_auto_title_if_empty). Collisions
            # dedup with a #N suffix; any failure is logged and the spawn
            # proceeds untitled (auto-title picks those up).
            if title:
                db = self.session_manager._get_db()
                if db is not None:
                    try:
                        db.set_session_title(
                            new_state.session_id,
                            db.get_next_title_in_lineage(title),
                        )
                    except Exception:
                        logger.warning(
                            "acp_spawn_session: could not stamp title on %s",
                            new_state.session_id,
                            exc_info=True,
                        )
            future = safe_schedule_threadsafe(
                self._run_spawned_first_turn(
                    new_state.session_id,
                    prompt_text,
                    parent_state.session_id,
                    deliver_result_to_parent,
                ),
                loop,
                logger=logger,
                log_message="acp_spawn_session: failed to schedule first turn",
            )
            if future is None:
                # The session exists but its turn couldn't start; surface the
                # failure to the calling agent rather than returning a dead id.
                raise RuntimeError(
                    "could not schedule the spawned session's first turn"
                )
            return new_state.session_id

        return _requester

    @staticmethod
    def _final_user_history_index(
        messages: Any, user_content: Any, agent_idx: Any
    ) -> int | None:
        """Absolute index of this turn's user message in the FINALIZED history.

        ``agent_idx`` (the agent's ``_persist_user_message_idx``) is captured
        when the turn machinery appends the user message — BEFORE preflight or
        mid-turn context compression, which can replace the message list and
        shift (or summarize away) that message. Trusting it blindly would let
        the returned ``userHistoryIndex`` disagree with the ``historyIndex``
        replay stamps and the prefix ``keepHistory`` slices. So it is used only
        as proof the turn actually appended a user message; the coordinate
        itself comes from rescanning the finalized list from the end — the
        latest content match is the current turn (it was appended last;
        synthetic user messages appended later in the turn, like
        empty-response nudges and length continuations, carry fixed system
        text that cannot equal the submitted content). Returns ``None`` when
        the message cannot be located (e.g. compression merged it into a
        summary), in which case the meta stamp is omitted and the client
        falls back to a full-history fork.
        """
        if isinstance(agent_idx, bool) or not isinstance(agent_idx, int) or agent_idx < 0:
            return None
        if not isinstance(messages, list):
            return None

        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and msg.get("content") == user_content
            ):
                return idx
        return None

    def _reconcile_persisted_history(
        self, state: SessionState, messages: list[dict[str, Any]]
    ) -> None:
        """Mirror an in-memory history rewrite (compact/reset) into the message store.

        save_session deliberately skips replace_messages when the agent owns
        persistence, so a slash-command history rewrite would otherwise leave
        the DB carrying the old rows as the live set — and the mid-turn DB
        replay (and post-restart restore) would resurrect them. Soft-archive
        the old rows and install the new list atomically; best-effort.
        """
        db = self.session_manager._get_db()
        if db is None:
            return
        sid = getattr(state.agent, "session_id", None)
        head_id = sid if isinstance(sid, str) and sid else state.session_id
        try:
            db.archive_and_compact(head_id, messages)
        except Exception:
            logger.warning(
                "Failed to reconcile persisted history for ACP session %s",
                state.session_id,
                exc_info=True,
            )
