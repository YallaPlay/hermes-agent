"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature AIAgent expects for its
callbacks. AIAgent runs in a worker thread while the event loop lives on the
main thread, so updates are pushed via ``conn.session_update()`` scheduled
thread-safely onto the loop.
"""

import asyncio
import logging
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import _json_loads_maybe, build_tool_complete, build_tool_start, coerce_tool_args, make_tool_call_id

logger = logging.getLogger(__name__)

# ACP plans only support pending/in_progress/completed. Cancelled tasks are kept
# as terminal entries so the client's full-list replacement doesn't drop them.
_PLAN_STATUS = {"pending": "pending", "in_progress": "in_progress", "completed": "completed", "cancelled": "completed"}


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task panel, so the
    todo state is exposed natively rather than only as a tool-call transcript."""
    if not isinstance(result, str) or not result.strip():
        return None
    data = _json_loads_maybe(result)
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    entries: list[PlanEntry] = []
    for item in data["todos"]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=_PLAN_STATUS.get(raw_status, "pending")))
    return AgentPlanUpdate(session_update="plan", entries=entries)


def _send_update(conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, update: Any) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update), loop, logger=logger, log_message="Failed to send ACP update",
    )
    if future is None:
        return
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


class SubagentUpdateRouter:
    """Route relayed ``subagent.*`` delegate events to their child session."""

    def __init__(self, conn: acp.Client, loop: asyncio.AbstractEventLoop) -> None:
        self._conn = conn
        self._loop = loop
        self._tool_ids: Dict[str, Dict[str, Deque[str]]] = {}
        self._running: Dict[str, bool] = {}

    def _send_running(self, child_id: str, is_running: bool, prompt_text: str | None = None) -> None:
        from acp.schema import SessionInfoUpdate

        hermes_meta: Dict[str, Any] = {"isRunning": is_running}
        if prompt_text:
            hermes_meta["currentPromptText"] = prompt_text
        _send_update(
            self._conn,
            child_id,
            self._loop,
            SessionInfoUpdate(session_update="session_info_update", field_meta={"hermes": hermes_meta}),
        )
        self._running[child_id] = is_running

    def _flush_dangling(self, child_id: str) -> None:
        for name, queue in self._tool_ids.get(child_id, {}).items():
            while queue:
                _send_update(self._conn, child_id, self._loop, build_tool_complete(queue.popleft(), name))
        self._tool_ids.pop(child_id, None)

    def __call__(self, event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        child_id = kwargs.get("child_session_id")
        if not child_id:
            return
        child_id = str(child_id)
        if event_type == "subagent.progress":
            return
        if not self._running.get(child_id) and event_type != "subagent.complete":
            self._send_running(child_id, True, str(kwargs.get("goal") or preview or "") or None)
            if event_type == "subagent.start":
                return
        if event_type == "subagent.start":
            return
        if event_type == "subagent.tool":
            tool_args = coerce_tool_args(args)
            tc_id = make_tool_call_id()
            self._tool_ids.setdefault(child_id, {}).setdefault(name or "", deque()).append(tc_id)
            _send_update(self._conn, child_id, self._loop, build_tool_start(tc_id, name, tool_args))
            return
        if event_type == "subagent.tool_completed":
            queue = self._tool_ids.get(child_id, {}).get(name or "")
            if not queue:
                return
            result = kwargs.get("result")
            _send_update(
                self._conn, child_id, self._loop,
                build_tool_complete(queue.popleft(), name, result=str(result) if result is not None else None),
            )
            return
        if event_type == "subagent.thinking" and preview:
            _send_update(self._conn, child_id, self._loop, acp.update_agent_thought_text(preview))
            return
        if event_type == "subagent.text" and preview:
            _send_update(self._conn, child_id, self._loop, acp.update_agent_message_text(preview))
            return
        if event_type == "subagent.complete":
            self._flush_dangling(child_id)
            self._send_running(child_id, False)

    def finalize(self) -> None:
        """Close dangling child tool calls and running flags at turn end."""
        for child_id in list(self._tool_ids):
            self._flush_dangling(child_id)
        for child_id, is_running in list(self._running.items()):
            if is_running:
                self._send_running(child_id, False)


def make_subagent_update_router(conn: acp.Client, loop: asyncio.AbstractEventLoop) -> SubagentUpdateRouter:
    return SubagentUpdateRouter(conn, loop)


def _upgrade_queue(tool_call_ids: Dict[str, Deque[str]], name: str) -> Deque[str] | None:
    """Fetch the per-tool FIFO of pending call IDs, upgrading a legacy bare-string entry in place."""
    queue = tool_call_ids.get(name)
    if isinstance(queue, str):
        queue = tool_call_ids[name] = deque([queue])
    return queue


def make_tool_progress_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
    subagent_router: Callable | None = None,
) -> Callable:
    """Create the live tool/subagent progress bridge."""

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        if event_type.startswith("subagent."):
            if subagent_router is not None and kwargs.get("child_session_id"):
                subagent_router(event_type, name, preview, args, **kwargs)
            return
        if event_type == "tool.completed":
            if name == "todo":
                plan_update = _build_plan_update_from_todo_result(kwargs.get("result"))
                if plan_update is not None:
                    _send_update(conn, session_id, loop, plan_update)
            queue = _upgrade_queue(tool_call_ids, name or "")
            if name and queue:
                tc_id = queue.popleft()
                meta = tool_call_meta.pop(tc_id, {})
                result = kwargs.get("result")
                _send_update(conn, session_id, loop, build_tool_complete(
                    tc_id, name, result=str(result) if result is not None else None,
                    function_args=meta.get("args"), snapshot=meta.get("snapshot"),
                ))
                if not queue:
                    tool_call_ids.pop(name, None)
            return
        if event_type != "tool.started":
            return
        args = coerce_tool_args(args)
        tc_id = make_tool_call_id()
        queue = _upgrade_queue(tool_call_ids, name)
        if queue is None:
            queue = tool_call_ids[name] = deque()
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
        _send_update(conn, session_id, loop, build_tool_start(tc_id, name, args, edit_diff=edit_diff))

    return _tool_progress


class AssistantMessageIdAllocator:
    """Allocates stable per-message ids for streamed assistant chunks."""

    def __init__(self) -> None:
        self._active: str | None = None
        self._last: str | None = None

    def current(self) -> str:
        if self._active is None:
            self._active = self._last = str(uuid.uuid4())
        return self._active

    def last(self) -> str | None:
        return self._last

    def close(self) -> None:
        self._active = None


def _make_text_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, wrap: Callable[[str], Any],
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    def _cb(text: str | None) -> None:
        if text:
            update = wrap(text)
            if message_ids is not None:
                update.message_id = message_ids.current()
            _send_update(conn, session_id, loop, update)
        elif text is None and message_ids is not None:
            message_ids.close()
    return _cb


def make_thinking_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    return _make_text_cb(conn, session_id, loop, acp.update_agent_thought_text, message_ids)


def make_message_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop,
    message_ids: AssistantMessageIdAllocator | None = None,
) -> Callable:
    return _make_text_cb(conn, session_id, loop, acp.update_agent_message_text, message_ids)


def make_step_cb(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a ``step_callback(api_call_count, prev_tools)`` fallback completion bridge."""

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if not isinstance(prev_tools, list):
            return
        for tool_info in prev_tools:
            tool_name = result = function_args = None
            if isinstance(tool_info, dict):
                tool_name = tool_info.get("name") or tool_info.get("function_name")
                result = tool_info.get("result") if "result" in tool_info else tool_info.get("output")
                function_args = tool_info.get("arguments") or tool_info.get("args")
            elif isinstance(tool_info, str):
                tool_name = tool_info
            if not tool_name:
                continue
            queue = _upgrade_queue(tool_call_ids, tool_name)
            if not queue:
                continue
            tc_id = queue.popleft()
            meta = tool_call_meta.pop(tc_id, {})
            _send_update(conn, session_id, loop, build_tool_complete(
                tc_id, tool_name, result=str(result) if result is not None else None,
                function_args=function_args or meta.get("args"), snapshot=meta.get("snapshot"),
            ))
            if not queue:
                tool_call_ids.pop(tool_name, None)
    return _step


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
