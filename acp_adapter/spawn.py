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
import inspect
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
            "Spawn a NEW derived, clean-context Hermes session inside this "
            "running ACP server and start its first turn immediately in the "
            "background. The new session is INDEPENDENT: it is shown next to "
            "this one in the sidebar, but it records no parent and does not "
            "copy this conversation history. "
            "Returns the new session id right away. The spawned session owns "
            "its work and reports to the user, not back here; pass "
            "deliver_result_to_parent=true only if you deliberately need its "
            "first-turn result returned to you. For work that must report back, "
            "use delegate_task, whose subagents are real children. "
            "Do not poll for completion. The session appears in the VS Code "
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
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional provider route override. Defaults to this "
                        "session's provider."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional model route override. Defaults to this "
                        "session's model."
                    ),
                },
                "deliver_result_to_parent": {
                    "type": "boolean",
                    "description": (
                        "Return the spawned first-turn result to the spawning "
                        "session. Defaults to FALSE, because a spawned session is "
                        "independent (parent_session_id is NULL) and owns its own "
                        "work. Set true only when you deliberately want the first "
                        "turn reported back, for example a spawned probe whose "
                        "answer the spawner needs; for parented work whose result "
                        "must return, use delegate_task instead."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
}

SpawnSessionRequester = Callable[..., str]

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
    provider = arguments.get("provider")
    provider = str(provider).strip() if provider else None
    model = arguments.get("model")
    model = str(model).strip() if model else None
    # Default false. acp_spawn_session creates a session with parent_session_id = NULL: it is a
# NEW independent session, not a subagent and not a fork. Routing its first-turn result back to
# the spawning session therefore creates a dependency the data model denies, and in practice it
# makes handoff children ask the spawner for approval and end their turn on a question that
# nobody can answer, because ACP cannot inject a reply into a finished turn (observed
# 2026-08-19: a continuation session stalled this way and a third session had to be spawned to
# resume it). Callers who genuinely want the result back must ask for it explicitly.
# delegate_task is the tool for parented work whose result returns to its caller.
    delivery_value = arguments.get("deliver_result_to_parent", False)
    if not isinstance(delivery_value, bool):
        return json.dumps(
            {"error": "deliver_result_to_parent must be a boolean"},
            ensure_ascii=False,
        )
    delivery_supported = True

    try:
        try:
            signature = inspect.signature(requester)
            parameters = signature.parameters.values()
            supports_delivery = (
                "deliver_result_to_parent" in signature.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
            )
        except (TypeError, ValueError):
            supports_delivery = True
        if supports_delivery:
            session_id = requester(
                prompt_text,
                cwd,
                title,
                provider,
                model,
                deliver_result_to_parent=delivery_value,
            )
        else:
            session_id = requester(prompt_text, cwd, title, provider, model)
            delivery_supported = False
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
                "Derived clean-context session created; its first-turn result "
                "will return automatically to this parent. Do not poll it."
                if delivery_value and delivery_supported
                else (
                    "Derived clean-context session created, but automatic parent "
                    "delivery is unavailable for this legacy requester. Inspect "
                    "the child session directly."
                    if delivery_value
                    else "Derived child-owned session created; no parent completion will be sent."
                )
            ),
        },
        ensure_ascii=False,
    )


# ACP injects this schema only into ACP-managed agents. Register the handler separately so calls
# continue through the standard registry/middleware path after model_tools' registry-backed split.
from tools.registry import registry

registry.register(
    name=SPAWN_SESSION_TOOL_NAME,
    toolset="acp-internal",
    schema=SPAWN_SESSION_TOOL_SCHEMA["function"],
    handler=lambda args, **_kwargs: maybe_dispatch_spawn_session(
        SPAWN_SESSION_TOOL_NAME, args
    ),
)
