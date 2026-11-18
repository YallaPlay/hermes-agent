"""Edit-and-resend on the gateway path (``session/prompt`` ``_meta.hermes.keepHistory``).

Coordinate: the index of a row in the session's live transcript as ``session/load`` replays it
(the owner snapshot's non-archived ``messages`` in stored order). Replay stamps each rewindable user
turn with that index as ``_meta.hermes.historyIndex``; the client sends it back as ``keepHistory``
("keep the first N rows"). Row N must be a rewindable user turn, so the truncate is the owner's
existing ``session.mutate rewind`` to it (the verb ``/undo`` uses): that row and everything after
it are deactivated, then the edited text is submitted as a normal turn.
"""
from __future__ import annotations

import acp

from hermes_cli.gateway_client import GatewayClientError


def keep_history_meta(kwargs) -> int | None:
    """``_meta.hermes.keepHistory`` as main's ``HermesACPAgent._keep_history_meta`` (the router flattens
    ``_meta`` into a ``hermes`` kwarg): absent is None, anything but a non-negative int is refused."""
    hermes_meta = kwargs.get("hermes")
    if not isinstance(hermes_meta, dict) or "keepHistory" not in hermes_meta:
        return None
    keep = hermes_meta["keepHistory"]
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
        raise _invalid("_meta.hermes.keepHistory must be a non-negative integer")
    return keep


def live_rows(snapshot):
    """The replay's coordinate space: live rows in stored order (archived rows are display-only)."""
    return sorted((m for m in snapshot.get("messages") or [] if not m.get("archived")),
                  key=lambda m: m.get("row_id") or 0)


def _rewindable(row) -> bool:
    from hermes_cli.gateway_chat_commands import _user_turns
    return bool(_user_turns({"messages": [row]}))


def replay_updates(live):
    """``(update, meta)`` pairs for ``session/load``: main's replay of ``live``, with each rewindable
    user turn's chunks stamped ``{"hermes": {"historyIndex": i}}``. Non-user runs replay together so
    an assistant tool call still pairs with its tool result."""
    from acp_adapter.server import _history_replay_updates
    run = []
    for index, row in enumerate(live):
        if row.get("role") != "user":
            run.append(row)
            continue
        for update in _history_replay_updates(run):
            yield update, None
        run = []
        meta = {"hermes": {"historyIndex": index}} if _rewindable(row) else None
        for update in _history_replay_updates([row]):
            yield update, meta if getattr(update, "session_update", None) == "user_message_chunk" else None
    for update in _history_replay_updates(run):
        yield update, None


def _invalid(reason):
    return acp.RequestError.invalid_params({"reason": reason})


async def rewind_for_prompt(agent, session_id, keep: int) -> None:
    """Truncate the owner's transcript to its first ``keep`` live rows before the edited turn.

    Retained across a lost reply like ``/undo``: a repeat replays the same rewind instead of deriving a
    new target from an already-shortened transcript. Nothing to drop (``keep`` at or past the end) is a
    plain submit, as main's slice is."""
    def payload(snapshot):
        live = live_rows(snapshot)
        if keep >= len(live):
            return None
        if not _rewindable(live[keep]):
            raise _invalid("_meta.hermes.keepHistory must point at a user turn")
        return {"target_message_id": live[keep]["row_id"]}

    client = await agent._client()
    try:
        result = await agent._mutations.apply(client, session_id, "rewind", payload)
    except GatewayClientError as exc:
        if str(exc) in {"session_busy", "unknown_execution"}:
            agent._mutations.acknowledge(session_id, "rewind", payload)
            raise _invalid("cannot rewind while a turn is running") from exc
        raise
    if result.get("status") != "nothing":
        await agent._adopt(session_id, await client.rpc("session.resume", session_id=session_id))
    agent._mutations.acknowledge(session_id, "rewind", payload)
