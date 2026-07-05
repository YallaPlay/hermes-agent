"""set_reasoning_effort — let the model tune its own reasoning effort mid-task.

The model calls this FIRST, before doing heavy work, to raise effort (e.g. when
an analytically heavy skill/domain is in play), or to drop it for routine work.

Mechanism (verified against fork runtime 2026-07-05):

* ``agent.reasoning_config`` is a mutable dict read on *every* API call inside
  the turn loop (``agent/chat_completion_helpers.py`` reads ``agent.reasoning_config``
  per call, not cached per turn). Mutating it here therefore takes effect on the
  model's *next* API call — including the next call of the SAME turn — with no
  agent re-init required.
* This is the opposite lifecycle from the ``/reasoning`` slash command
  (``hermes_cli/cli_commands_mixin.py``), which runs *between* turns and must set
  ``self.agent = None`` to force a rebuild. Our tool runs *inside* a turn against
  the live agent object, so it mutates ``agent.reasoning_config`` directly.

Design guarantees:

* Session-scoped only. It NEVER persists to profile config (no ``save_config_value``),
  so a task-local escalation cannot silently change ``agent.reasoning_effort`` for
  future sessions.
* Idempotent. Re-requesting the current level is a cheap no-op (no log spam).
* Provider-honest. Reports the requested level; some providers ignore/remap effort
  (e.g. codex clamps ``minimal`` -> ``low``). We do not claim provider acceptance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Levels the model may request. Mirrors hermes_constants.parse_reasoning_effort
# and the /reasoning slash command's accepted values.
VALID_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _parse_level(level: str) -> Optional[Dict[str, Any]]:
    """Return the reasoning_config dict for *level*, or None if invalid.

    Reuses the single source of truth so this tool and /reasoning produce
    identical config shapes:
      none   -> {"enabled": False}
      <lvl>  -> {"enabled": True, "effort": "<lvl>"}
      bad    -> None
    """
    from hermes_constants import parse_reasoning_effort

    return parse_reasoning_effort(level)


def set_reasoning_effort(agent: Any, level: str, reason: str = "") -> str:
    """Mutate the live agent's reasoning_config. Returns a JSON result string."""
    level = str(level or "").strip().lower()
    reason = str(reason or "").strip()

    if level not in VALID_LEVELS:
        return json.dumps({
            "success": False,
            "error": f"invalid level {level!r}; valid levels: {list(VALID_LEVELS)}",
        })

    parsed = _parse_level(level)
    if parsed is None:
        return json.dumps({
            "success": False,
            "error": f"could not parse reasoning level {level!r}",
        })

    prev = getattr(agent, "reasoning_config", None)

    # Idempotent no-op: already at the requested config.
    if prev == parsed:
        return json.dumps({
            "success": True,
            "level": level,
            "changed": False,
            "note": "already at this effort level",
        })

    agent.reasoning_config = parsed

    # Best-effort: if the agent exposes a back-ref to the CLI/session wrapper,
    # mirror the level so a later in-session agent re-init keeps it. Never persist.
    owner = getattr(agent, "_cli_owner", None)
    if owner is not None and hasattr(owner, "reasoning_config"):
        try:
            owner.reasoning_config = parsed
        except Exception:  # defensive: never let a back-ref quirk break the tool
            pass

    logger.info(
        "set_reasoning_effort: %r -> %r (reason: %s)",
        prev, parsed, reason or "(none given)",
    )

    return json.dumps({
        "success": True,
        "level": level,
        "changed": True,
        "note": (
            "applies from your next model call (may be same turn); "
            "session-scoped, not persisted; provider may ignore/remap the level"
        ),
    })


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

SET_REASONING_EFFORT_SCHEMA = {
    "name": "set_reasoning_effort",
    "description": (
        "Set your own reasoning effort for the rest of this task. Call this FIRST, "
        "before doing the work, when the task is analytically heavy — data analysis, "
        "debugging, multi-file code changes, revenue/cohort/retention reasoning, or "
        "any task where a loaded skill marks its domain as heavy. Use 'high' or "
        "'xhigh' for hard tasks; drop to 'low'/'none' for trivial lookups and routine "
        "formatting. Takes effect on your next model call (may be the same turn). "
        "Session-scoped: it does NOT change profile defaults. Some providers may "
        "ignore or remap the level."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": list(VALID_LEVELS),
                "description": "Reasoning effort level to switch to.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Brief why, for the audit log "
                    "(e.g. 'loaded analytics skill for a cohort query')."
                ),
            },
        },
        "required": ["level"],
    },
}


def check_set_reasoning_effort_requirements() -> bool:
    """No external requirements -- always available."""
    return True


# --- Registry ---
from tools.registry import registry  # noqa: E402

registry.register(
    name="set_reasoning_effort",
    toolset="reasoning",
    schema=SET_REASONING_EFFORT_SCHEMA,
    # Fallback handler for non-agent dispatch paths (no live agent in scope).
    # The real, agent-aware path is the dispatch branch in
    # agent/agent_runtime_helpers.py, which calls set_reasoning_effort(agent, ...).
    handler=lambda args, **kw: json.dumps({
        "success": False,
        "error": "set_reasoning_effort requires an active agent context",
    }),
    check_fn=check_set_reasoning_effort_requirements,
    emoji="⚙️",
)
