"""Skill write-origin provenance: a ContextVar separating background-review skill writes from foreground
user-directed writes (the curator only curates skills the self-improvement review fork created; skills a user
asked for belong to the user). run_agent.py binds the origin before each tool loop, mirroring
AIAgent._memory_write_origin: ``token = set_current_write_origin(...)`` / ``reset_current_write_origin(token)``."""

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
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """"foreground" for any regular agent (CLI, gateway, cron, subagent); "background_review" for the review fork."""
    return _write_origin.get()


def is_background_review() -> bool:
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
# Attendedness is orthogonal to origin: an explicit ``/refine`` fork IS a background review (every
# curator / skill-ledger / approval guard keyed on ``is_background_review()`` must still apply), but a
# user asked for it, so the unattended-only memory delete gate (#105921) does not.
_review_attended: contextvars.ContextVar[bool] = contextvars.ContextVar("review_attended", default=False)


def set_review_attended(attended: bool) -> contextvars.Token[bool]:
    return _review_attended.set(bool(attended))


def reset_review_attended(token: contextvars.Token[bool]) -> None:
    _review_attended.reset(token)


def is_unattended_review() -> bool:
    return is_background_review() and not _review_attended.get()
