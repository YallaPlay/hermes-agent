"""Notification-delivery helpers for the ACP server.

Pure functions over ``SessionState`` and registry completion events, split
out of ``server.py`` (the facade) so the watcher/drain code there stays
focused on scheduling and the claim protocol.
"""

from __future__ import annotations

import json
from typing import Any

from acp_adapter.session import SessionState


def completion_output_seen_after_spawn(state: SessionState, evt: dict) -> bool:
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
