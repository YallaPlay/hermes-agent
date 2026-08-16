"""Skill write-origin provenance — ContextVar for distinguishing agent-sediment skill writes from foreground user-directed writes.

The curator only consolidates/prunes skills it autonomously created via the
background self-improvement review fork. Skills a user asks a foreground
agent to write belong to the user and must never be auto-curated.

This module exposes a ContextVar that run_agent.py sets before each tool
loop so tool handlers (e.g. skill_manage create) can check whether they
are executing inside the background-review fork.

The signal piggybacks on AIAgent._memory_write_origin, which is already
set to "background_review" for review-fork instances (see
_spawn_background_review in run_agent.py) and defaults to "assistant_tool"
for normal (foreground) agents.

Usage:
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )

    token = set_current_write_origin("background_review")
    try:
        ...  # tool runs here
    finally:
        reset_current_write_origin(token)

    # inside a tool:
    if get_current_write_origin() == "background_review":
        mark_agent_created(skill_name)
"""

import contextvars


_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin",
    default="foreground",
)

# Identity of the active background-review fork, bound alongside the write
# origin at turn setup. Tool dispatch runs each call on a worker thread with a
# *snapshot* of the loop thread's context (propagate_context_to_thread), so
# any per-fork state a tool needs to persist across calls must live in
# process-global storage keyed by this id — ContextVar writes made inside a
# worker die with the snapshot. The read-before-write marks in
# skill_manager_tool are the canonical consumer.
_review_fork_id: contextvars.ContextVar["str | None"] = contextvars.ContextVar(
    "skill_review_fork_id",
    default=None,
)

# The sentinel value the background review fork uses; mirrors
# run_agent.py's AIAgent._memory_write_origin override in
# _spawn_background_review().
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin to the current context.

    Returns a Token the caller must pass to reset_current_write_origin
    in a finally block.
    """
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """Return the active write origin.

    Default: "foreground" — any tool call made by a regular (non-review)
    agent, from the CLI, the gateway, cron, or a subagent.

    "background_review" — the self-improvement review fork; only skills
    created under this origin should be marked agent-created for curator
    management.
    """
    return _write_origin.get()


def is_background_review() -> bool:
    """Convenience: True iff the current write origin is the background
    review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW


def set_current_review_fork_id(fork_id: "str | None") -> "contextvars.Token[str | None]":
    """Bind the active review fork's identity to the current context.

    Pass a stable per-fork value (turn_context uses ``str(id(agent))``) when
    the origin is ``background_review``, or ``None`` for foreground turns.
    """
    return _review_fork_id.set(fork_id)


def get_current_review_fork_id() -> "str | None":
    """Return the active review fork id, or None outside a review fork."""
    return _review_fork_id.get()
