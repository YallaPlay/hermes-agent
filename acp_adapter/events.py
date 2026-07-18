"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature that AIAgent expects
for its callbacks. Internally, the callbacks push ACP session updates
to the client via ``conn.session_update()`` using
``asyncio.run_coroutine_threadsafe()`` (since AIAgent runs in a worker
thread while the event loop lives on the main thread).
"""

import asyncio
import json
import logging
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    build_tool_complete,
    build_tool_start,
    make_tool_call_id,
)

logger = logging.getLogger(__name__)


def _json_loads_maybe_prefix(value: str) -> Any:
    """Parse a JSON object even when Hermes appended a human hint after it."""
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)
        return data


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task/todo panel. The
    Hermes agent already maintains task state through the ``todo`` tool, so the
    ACP adapter should expose that state natively instead of only as a generic
    tool-call transcript block.
    """
    if not isinstance(result, str) or not result.strip():
        return None

    try:
        data = _json_loads_maybe_prefix(result)
    except Exception:
        return None

    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    todos = data["todos"]
    if not todos:
        return AgentPlanUpdate(session_update="plan", entries=[])

    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
        # ACP plans only support pending/in_progress/completed. Preserve
        # cancelled tasks as terminal entries instead of dropping them and
        # making the client's full-list replacement lose visible context.
        "cancelled": "completed",
    }
    entries: list[PlanEntry] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        status = status_map.get(raw_status, "pending")
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=status))

    return AgentPlanUpdate(session_update="plan", entries=entries)


def _send_update(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    update: Any,
) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update),
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


# ------------------------------------------------------------------
# Subagent update router
# ------------------------------------------------------------------

def make_subagent_update_router(
    conn: acp.Client,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Route relayed ``subagent.*`` delegate events to their CHILD session.

    Delegated children persist as real sessions and every relayed event
    carries ``child_session_id``; re-emitting them as ACP ``session/update``
    frames addressed to that id gives clients a live per-child transcript.
    Per-child FIFO tool-id state mirrors the parent's ``tool_call_ids``
    discipline. Events without a ``child_session_id`` (possible only for the
    very first ``subagent.start``, which can race the session_ref fill) are
    dropped; the first routed event per child lazily synthesizes the
    ``isRunning: true`` info update so the miss is invisible.

    Call ``router.finalize()`` at parent turn end to close dangling tool
    calls and flip ``isRunning`` off for children that never completed
    (crash/interrupt paths).
    """
    from acp.schema import SessionInfoUpdate

    tool_ids: Dict[str, Dict[str, Deque[str]]] = {}
    running: Dict[str, bool] = {}

    def _send_running(child_id: str, is_running: bool, prompt_text: str | None = None) -> None:
        hermes_meta: Dict[str, Any] = {"isRunning": is_running}
        if prompt_text:
            hermes_meta["currentPromptText"] = prompt_text
        update = SessionInfoUpdate(
            session_update="session_info_update",
            field_meta={"hermes": hermes_meta},
        )
        _send_update(conn, child_id, loop, update)
        running[child_id] = is_running

    def _flush_dangling(child_id: str) -> None:
        for name, queue in tool_ids.get(child_id, {}).items():
            while queue:
                tc_id = queue.popleft()
                _send_update(
                    conn, child_id, loop, build_tool_complete(tc_id, name)
                )
        tool_ids.pop(child_id, None)

    def _router(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        child_id = kwargs.get("child_session_id")
        if not child_id:
            return
        child_id = str(child_id)

        if event_type == "subagent.progress":
            # Batched summary — redundant once per-tool frames stream.
            return

        if not running.get(child_id) and event_type != "subagent.complete":
            _send_running(
                child_id, True, str(kwargs.get("goal") or preview or "") or None
            )
            if event_type == "subagent.start":
                return

        if event_type == "subagent.start":
            return

        if event_type == "subagent.tool":
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {}
            tc_id = make_tool_call_id()
            queues = tool_ids.setdefault(child_id, {})
            queues.setdefault(name or "", deque()).append(tc_id)
            _send_update(conn, child_id, loop, build_tool_start(tc_id, name, args))
            return

        if event_type == "subagent.tool_completed":
            queue = tool_ids.get(child_id, {}).get(name or "")
            if not queue:
                return
            tc_id = queue.popleft()
            result = kwargs.get("result")
            _send_update(
                conn,
                child_id,
                loop,
                build_tool_complete(
                    tc_id, name, result=str(result) if result is not None else None
                ),
            )
            return

        if event_type == "subagent.thinking":
            if preview:
                _send_update(conn, child_id, loop, acp.update_agent_thought_text(preview))
            return

        if event_type == "subagent.text":
            if preview:
                _send_update(conn, child_id, loop, acp.update_agent_message_text(preview))
            return

        if event_type == "subagent.complete":
            _flush_dangling(child_id)
            _send_running(child_id, False)
            return

    def _finalize() -> None:
        """Close dangling child tool calls and running flags at turn end."""
        for child_id in list(tool_ids):
            _flush_dangling(child_id)
        for child_id, is_running in list(running.items()):
            if is_running:
                _send_running(child_id, False)

    _router.finalize = _finalize
    return _router


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    subagent_router: Callable | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and tracks IDs in a FIFO
    queue per tool name so duplicate/parallel same-name calls still complete
    against the correct ACP tool call.  Other event types (``tool.completed``,
    ``reasoning.available``) are silently ignored.
    """

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        # Relayed delegate-child events route to the CHILD session's own
        # transcript; they are never parent tool calls. Events that predate
        # the child's session_ref fill carry no child_session_id and drop.
        if event_type.startswith("subagent."):
            if subagent_router is not None and kwargs.get("child_session_id"):
                subagent_router(event_type, name, preview, args, **kwargs)
            return
        # A completed ``todo`` carries the authoritative task list in its result.
        # Emit the native ACP plan update here, on the live completion event, so
        # the turn's FINAL todo is reflected too. The step callback only observes
        # a tool's result on the *next* step (via ``prev_tools``), which never
        # arrives for the last tool of a turn — leaving the plan panel one todo
        # behind whenever the turn ends on a todo update.
        if event_type == "tool.completed":
            if name == "todo":
                plan_update = _build_plan_update_from_todo_result(kwargs.get("result"))
                if plan_update is not None:
                    _send_update(conn, session_id, loop, plan_update)
            # Emit the ACP tool-call completion live, the instant the tool
            # finishes. The step callback only reports a tool's result on the
            # *next* API step (via ``prev_tools``), so the last tool(s) of a turn
            # — or any turn that ends without a further step — never received a
            # "completed" update and the client spun their spinner forever
            # (until the turn-end sweep marked them failed, which is also wrong:
            # they succeeded). Completing here, correlated by the same FIFO
            # queue, clears the spinner immediately; the step callback's pop is
            # now a redundant no-op for these ids (guarded there).
            queue = tool_call_ids.get(name or "")
            if isinstance(queue, str):
                queue = deque([queue])
                tool_call_ids[name] = queue
            if name and queue:
                tc_id = queue.popleft()
                meta = tool_call_meta.pop(tc_id, {})
                result = kwargs.get("result")
                update = build_tool_complete(
                    tc_id,
                    name,
                    result=str(result) if result is not None else None,
                    function_args=meta.get("args"),
                    snapshot=meta.get("snapshot"),
                )
                _send_update(conn, session_id, loop, update)
                if not queue:
                    tool_call_ids.pop(name, None)
            return
        # Only emit ACP ToolCallStart for tool.started; ignore other event types
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = make_tool_call_id()
        queue = tool_call_ids.get(name)
        if queue is None:
            queue = deque()
            tool_call_ids[name] = queue
        elif isinstance(queue, str):
            queue = deque([queue])
            tool_call_ids[name] = queue
        queue.append(tc_id)

        snapshot = None
        if name in {"write_file", "patch", "skill_manage"}:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        tool_call_meta[tc_id] = {"args": args, "snapshot": snapshot}

        edit_diff = None
        if name in {"write_file", "patch"} and edit_approval_policy_getter is not None:
            try:
                from acp_adapter.edit_approval import build_edit_proposal, should_auto_approve_edit

                proposal = build_edit_proposal(name, args)
                if proposal is not None:
                    policy, cwd = edit_approval_policy_getter()
                    if should_auto_approve_edit(proposal, policy, cwd):
                        edit_diff = proposal
            except Exception:
                logger.debug("Failed to prepare auto-approved ACP edit diff for %s", name, exc_info=True)

        update = build_tool_start(tc_id, name, args, edit_diff=edit_diff)
        _send_update(conn, session_id, loop, update)

    return _tool_progress


# ------------------------------------------------------------------
# Thinking callback
# ------------------------------------------------------------------

def make_thinking_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""

    def _thinking(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_thought_text(text)
        _send_update(conn, session_id, loop, update)

    return _thinking


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if prev_tools and isinstance(prev_tools, list):
            for tool_info in prev_tools:
                tool_name = None
                result = None
                function_args = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                    function_args = tool_info.get("arguments") or tool_info.get("args")
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()
                    meta = tool_call_meta.pop(tc_id, {})
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=str(result) if result is not None else None,
                        function_args=function_args or meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    # NOTE: the native ``plan`` update for ``todo`` results is now
                    # emitted live from the ``tool.completed`` progress event (see
                    # ``make_tool_progress_cb``), which also covers a turn's final
                    # todo. Emitting it here too would double-send and still miss
                    # the last todo, so it is intentionally not done here.
                    if not queue:
                        tool_call_ids.pop(tool_name, None)

    return _step


# ------------------------------------------------------------------
# Agent message callback
# ------------------------------------------------------------------

def make_message_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""

    def _message(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_message_text(text)
        _send_update(conn, session_id, loop, update)

    return _message
