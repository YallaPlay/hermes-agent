"""In-process ACP session spawning.

This module is intentionally isolated from the generic tool registry (the
same precedent as ``edit_approval``): the ACP server binds a spawn requester
in a ContextVar for the duration of one ACP agent run; CLI, gateway, cron and
other runtimes leave it unset, so the tool politely refuses there.

Why in-process: ACP is stdio-only single-client — an external process (e.g. a
detached ``hermes chat -q`` child) can NEVER join the running ACP server, so
its turns are invisible to the VS Code UI and attaching to its session risks
concurrent writers on one transcript (2026-07-15 incident). Spawning the
continuation session INSIDE the ACP server keeps the turn on the server's own
event loop: live streaming, steer, and sidebar surfacing all work.

Trade-off (documented in the handoff skill): an in-process spawn dies with
the ACP process — a window reload kills its in-flight turn. The detached CLI
spawn script remains correct for walk-away durability.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SPAWN_SESSION_TOOL_NAME = "acp_spawn_session"

# OpenAI function-schema shape, matching what agent.tools carries.
SPAWN_SESSION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SPAWN_SESSION_TOOL_NAME,
        "description": (
            "Spawn a NEW independent Hermes session inside this running ACP "
            "server and start its first turn immediately in the background. "
            "Returns the new session id right away — do not wait for the "
            "spawned turn to finish. The session appears in the VS Code "
            "sessions sidebar with live streaming and steer support. Use for "
            "handoff/continuation sessions that should stay visible in this "
            "window. NOT durable across a window reload: the spawned turn "
            "dies with this ACP process — for walk-away work use the "
            "detached CLI spawn instead. If the spawned session will WRITE "
            "to a repo, give it its own worktree via the prompt; two "
            "sessions writing one checkout collide."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The first user message for the new session. Must be "
                        "self-contained — the new session shares none of this "
                        "conversation's context."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory for the new session. Defaults to "
                        "this session's cwd."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Session title (max 100 chars). You know what the "
                        "spawned work is — name it so the sessions sidebar "
                        "shows meaningful text immediately instead of "
                        "waiting for post-first-turn auto-titling. "
                        "Deduplicated with a #N suffix on collision; "
                        "stamping is best-effort and never fails the spawn."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
}

# (prompt_text, cwd_or_none, title_or_none) -> new session id. Raises on failure.
SpawnSessionRequester = Callable[[str, Optional[str], Optional[str]], str]

_SPAWN_SESSION_REQUESTER: ContextVar[SpawnSessionRequester | None] = ContextVar(
    "ACP_SPAWN_SESSION_REQUESTER",
    default=None,
)


def set_spawn_session_requester(requester: SpawnSessionRequester | None) -> Token:
    """Bind an ACP spawn requester for the current context."""

    return _SPAWN_SESSION_REQUESTER.set(requester)


def reset_spawn_session_requester(token: Token) -> None:
    """Restore a previous spawn requester binding."""

    _SPAWN_SESSION_REQUESTER.reset(token)


def clear_spawn_session_requester() -> None:
    """Clear the current requester; primarily used by tests."""

    _SPAWN_SESSION_REQUESTER.set(None)


def get_spawn_session_requester() -> SpawnSessionRequester | None:
    return _SPAWN_SESSION_REQUESTER.get()


def inject_spawn_session_tool(agent: Any) -> bool:
    """Append the spawn tool schema to an ACP-managed agent's tool surface.

    Idempotent. Called by the ACP server only (SessionManager-created agents),
    so the tool is never advertised to CLI/gateway/cron sessions. Mirrors the
    memory-provider injection pattern: append to ``agent.tools`` and add the
    name to ``agent.valid_tool_names``.
    """

    tools = getattr(agent, "tools", None)
    if tools is None:
        return False
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("function", {}).get("name") == SPAWN_SESSION_TOOL_NAME
        ):
            return False
    tools.append(SPAWN_SESSION_TOOL_SCHEMA)
    valid_tool_names = getattr(agent, "valid_tool_names", None)
    if valid_tool_names is None:
        valid_tool_names = set()
        try:
            agent.valid_tool_names = valid_tool_names
        except Exception:
            return True
    valid_tool_names.add(SPAWN_SESSION_TOOL_NAME)
    return True


def maybe_dispatch_spawn_session(
    function_name: str, arguments: dict[str, Any]
) -> str | None:
    """Dispatch an ``acp_spawn_session`` call if this is one.

    Returns ``None`` for every other tool so the normal dispatch continues.
    When the tool is called outside a bound ACP turn (no requester), returns
    a graceful JSON error instead of leaking into the registry.
    """

    if function_name != SPAWN_SESSION_TOOL_NAME:
        return None

    requester = get_spawn_session_requester()
    if requester is None:
        return json.dumps(
            {
                "error": (
                    "acp_spawn_session is only available inside a live ACP "
                    "(VS Code) session. Use the detached CLI spawn script for "
                    "other runtimes."
                )
            },
            ensure_ascii=False,
        )

    prompt_text = str(arguments.get("prompt") or "").strip()
    if not prompt_text:
        return json.dumps({"error": "prompt is required"}, ensure_ascii=False)
    cwd = arguments.get("cwd")
    cwd = str(cwd).strip() if cwd else None
    title = arguments.get("title")
    title = str(title).strip() if title else None

    try:
        session_id = requester(prompt_text, cwd, title)
    except Exception as exc:
        logger.warning("ACP spawn_session requester failed: %s", exc)
        return json.dumps(
            {"error": f"Failed to spawn session: {exc}"}, ensure_ascii=False
        )

    return json.dumps(
        {
            "success": True,
            "session_id": session_id,
            "note": (
                "New session created in this ACP server; its first turn is "
                "running in the background. It will appear in the sessions "
                "sidebar. Do not wait for it here."
            ),
        },
        ensure_ascii=False,
    )
