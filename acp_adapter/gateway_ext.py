"""Hermes ``_x`` ACP extension methods on the gateway path (``GatewayACPAgent.ext_method``).

FROZEN CONTRACT: the VS Code client (``vscode-hermes/src/acp/AcpClient.ts``) calls these by the
names, params and result shapes main's in-process adapter answers (``acp_adapter/server.py::
ext_method``), so every reply here keeps that shape. Each method is a gateway verb, never a local
write: metadata through ``session.mutate``, reads through the owner's session reads.

Dropped on this path (``method_not_found``, which the client already handles as "old agent"):
``deriveTitle`` (the owner auto-titles every session) and ``promoteSession`` (the gateway keeps no
display-only fork lineage to detach). ``setOwner`` is an acknowledged no-op: a per-user profile
replaces the shared store's soft owner key.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import uuid
from pathlib import Path

import acp

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)

# The category ids and titles of ``agent.context_breakdown.build_session_context_report``.
_REPORT_TITLES = {
    "system_prompt": "System prompt",
    "tool_definitions": "Tool definitions",
    "rules": "Rules (project context)",
    "skills": "Skills index",
    "mcp": "MCP tools",
    "subagent_definitions": "Subagent definitions",
    "memory": "Memory",
    "conversation": "Conversation",
}


async def ext_method(agent, method: str, params: dict) -> dict:
    handler = _HANDLERS.get(method)
    if handler is None:
        raise acp.RequestError.method_not_found(f"_{method}")
    return await handler(agent, params or {})


def _required(params):
    session_id = params.get("sessionId")
    return session_id if isinstance(session_id, str) and session_id else None


async def _target(agent, editor_id):
    """The authority's logical id for an id the editor sent: a held session's alias, else the
    stored lineage's owner (the same read-only lookup ``session/load`` does)."""
    if editor_id in agent._aliases:
        return agent._aliases[editor_id]
    if editor_id in agent._snapshots:
        return editor_id
    from acp_adapter.catalog import logical_session_id
    return await asyncio.to_thread(logical_session_id, agent._home / "state.db", editor_id)


async def _metadata(agent, session_id, operation, payload):
    """A metadata ``session.mutate`` (rename / archive). A held session goes through the editor's
    retained mutations (lost-reply retry, as ``/title``); a sidebar row this editor does not hold is
    resumed without its transcript for the revision fence and detached again."""
    client = await agent._client()
    if session_id in agent._snapshots:
        result = await agent._mutations.apply(client, session_id, operation, payload)
        agent._mutations.acknowledge(session_id, operation, payload)
        return result
    snapshot = await client.rpc("session.resume", session_id=session_id, omit_messages=True)
    try:
        return await client.rpc("session.mutate", session_id=session_id, request_id=uuid.uuid4().hex,
                                expected_revision=snapshot["revision"], operation=operation, payload=payload)
    finally:
        if snapshot.get("subscription_id"):
            with suppress(GatewayClientError):
                await client.rpc("session.detach", session_id=session_id,
                                 subscription_id=snapshot["subscription_id"])


async def _set_archived(agent, params):
    editor_id = _required(params)
    if not editor_id:
        return {"ok": False}
    session_id = await _target(agent, editor_id)
    try:
        await _metadata(agent, session_id, "archive", {"archived": bool(params.get("archived"))})
    except GatewayClientError:
        logger.debug("setArchived refused for %s", session_id, exc_info=True)
        return {"ok": False}
    return {"ok": True}


async def _set_owner(agent, params):
    # Per-user Hermes profiles replace the shared store's soft owner key, so there is nothing to
    # stamp; acknowledged so clients that still send it do not error. Removed in Phase 6.
    return {"ok": True, "ignored": True}


def _title_holder(home, session_id, title):
    """Which other session holds ``title`` (read-only, as ``GatewaySessionStore._title_holder``): the
    owner's ``ValueError`` text never crosses the wire, its bounded reason code does."""
    path = Path(home) / "state.db"
    if not title or not path.exists():
        return None
    from hermes_state import SessionDB
    with SessionDB(db_path=path, read_only=True) as db:
        with db._read_ctx() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE title = ? AND id != ?", (title, session_id)).fetchone()
            return row[0] if row else None


async def _set_title(agent, params):
    editor_id = _required(params)
    if not editor_id:
        return {"ok": False, "error": "sessionId required"}
    title = str(params.get("title") or "")
    session_id = await _target(agent, editor_id)
    try:
        result = await _metadata(agent, session_id, "rename", {"title": title})
    except GatewayClientError as exc:
        holder = await asyncio.to_thread(_title_holder, agent._home, session_id, title.strip())
        error = f"Title '{title.strip()}' is already in use by session {holder}" if holder else str(exc)
        return {"ok": False, "error": error}
    committed = result.get("title", title or None) if isinstance(result, dict) else title or None
    if isinstance(committed, str) and committed:
        await agent._project_title(session_id, agent._editor_id(session_id), committed)
    return {"ok": True, "title": committed}


async def _owner_read(agent, params, method):
    """``(result, None)`` from the owner's read for the editor's session, else ``(None, reply)``."""
    editor_id = _required(params)
    if not editor_id:
        return None, {"ok": False, "error": "sessionId required"}
    session_id = await _target(agent, editor_id)
    client = await agent._client()
    try:
        return await client.rpc(method, session_id=session_id), None
    except GatewayClientError as exc:
        return None, {"ok": False, "error": str(exc)}


async def _context_breakdown(agent, params):
    breakdown, refused = await _owner_read(agent, params, "session.context_breakdown")
    return refused or {"ok": True, "breakdown": breakdown}


def _render_report(breakdown, category):
    """Minimal port of ``build_session_context_report``: the owner exposes the category estimate
    (``session.context_breakdown``), not the composed text, so each section is its estimate and
    detail rows. Same headings, so the client's per-category view and export still line up."""
    if category is not None and category not in _REPORT_TITLES:
        raise ValueError(f"unknown context category: {category!r}")
    by_id = {c.get("id"): c for c in breakdown.get("categories") or [] if isinstance(c, dict)}
    model = breakdown.get("model") or ""
    title = _REPORT_TITLES[category] if category else "Full context"
    out = [f"# Context report — {title}" + (f" ({model})" if model else ""), ""]
    if not category:
        total, limit = breakdown.get("estimated_total"), breakdown.get("context_length")
        if total is not None:
            out += [f"Estimated total: ~{int(total):,} tokens" + (f" of {int(limit):,}" if limit else ""), ""]
    for section_id in [category] if category else list(_REPORT_TITLES):
        entry = by_id.get(section_id) or {}
        out += [f"## {_REPORT_TITLES[section_id]}", "", f"~{int(entry.get('tokens') or 0):,} tokens (estimate)"]
        for row in entry.get("detail") or []:
            if isinstance(row, dict):
                name = row.get("label") or row.get("name") or row.get("id") or ""
                tokens = row.get("tokens")
                out.append(f"- {name}" + (f": ~{int(tokens):,} tokens" if isinstance(tokens, (int, float)) else ""))
        if section_id == "rules":
            for src in breakdown.get("context_files") or []:
                if isinstance(src, dict):
                    out.append(f"- {src.get('label') or src.get('path') or ''}: ~{int(src.get('est_tokens') or 0):,} tokens")
        out.append("")
    out += ["_The gateway exposes category estimates only; the composed text is not available here._", ""]
    return "\n".join(out).strip() + "\n"


async def _context_report(agent, params):
    breakdown, refused = await _owner_read(agent, params, "session.context_breakdown")
    if refused:
        return refused
    try:
        return {"ok": True, "report": _render_report(breakdown, params.get("category") or None)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


async def _account_usage(agent, params):
    usage, refused = await _owner_read(agent, params, "session.account_usage")
    if refused:
        return refused
    snapshot = usage.get("snapshot") if isinstance(usage, dict) else None
    if not snapshot:
        return {"ok": True, "usage": None}
    return {"ok": True, "usage": {
        "provider": snapshot.get("provider"),
        "plan": snapshot.get("plan"),
        "fetchedAt": snapshot.get("fetched_at"),
        "windows": [{"label": w.get("label"), "usedPercent": w.get("used_percent"),
                     "resetAt": w.get("reset_at"), "detail": w.get("detail")}
                    for w in snapshot.get("windows") or []],
        "details": list(snapshot.get("details") or []),
    }}


async def _refresh_models(agent, params):
    info, refused = await _owner_read(agent, params, "session.info")
    if refused:
        return refused
    from acp_adapter import gateway_session_meta
    try:
        # The same picker ``session/new`` advertises (its model ids are what set_session_model takes),
        # rebuilt past the inventory cache for the session's route.
        state = await asyncio.to_thread(gateway_session_meta.model_state, info, force_refresh=True)
    except Exception as exc:  # catalog failure must not kill the RPC, as main
        logger.debug("refreshModels failed", exc_info=True)
        return {"ok": False, "error": str(exc)}
    if state is None:
        return {"ok": True, "models": None}
    return {"ok": True, "models": {
        "currentModelId": state.current_model_id,
        "availableModels": [{"modelId": m.model_id, "name": m.name, "description": m.description}
                            for m in state.available_models],
    }}


_HANDLERS = {
    "setArchived": _set_archived,
    "setOwner": _set_owner,
    "setTitle": _set_title,
    "contextBreakdown": _context_breakdown,
    "contextReport": _context_report,
    "accountUsage": _account_usage,
    "refreshModels": _refresh_models,
}
