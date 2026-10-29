#!/usr/bin/env python3
"""Connection lifecycle tool for managed gateway accounts and local MCP servers.

Disconnecting accounts remains a portal-only user decision.
"""

from typing import Any, Callable, Dict, Optional

from tools.connectors.catalog_tool import MANAGE_CATALOG_SCHEMA, manage_catalog
from tools.connectors.gateway import config as gateway_config
from tools.connectors.managed import run_managed_action
from tools.connectors.mcp import run_mcp_operation
from tools.connectors.targets import ALIAS_PATTERN, ALL_ACTIONS, MCP_ACTIONS, normalize_targets, validate_action
from tools.registry import registry, tool_error


def manage_connections(
    args: dict[str, Any],
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    mcp_backend: Optional[Any] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    connection_callback: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    connectors_available: Optional[Callable[[], bool]] = None,
    card_surface: bool = False,
) -> str:
    action = str(args.get("action") or "status").strip().lower()
    managed, mcp_targets, target_error = normalize_targets(args.get("connectors"))
    if target_error:
        return tool_error(target_error)
    action_error = validate_action(action, managed, mcp_targets)
    if action_error:
        return tool_error(action_error)

    if action in MCP_ACTIONS:
        return run_mcp_operation(
            mcp_targets, action, backend=mcp_backend,
            connection_callback=connection_callback, session_id=session_id, tool_call_id=tool_call_id,
        )

    return run_managed_action(
        action, managed, args,
        client_factory=client_factory, session_id=session_id, tool_call_id=tool_call_id, turn_id=turn_id,
        connection_callback=connection_callback, connectors_available=connectors_available,
        card_surface=card_surface,
    )


MANAGE_CONNECTIONS_SCHEMA = {
    "name": "manage_connections",
    "description": (
        "Connect hosted app accounts or local catalog MCP servers. Default to hosted; use mcp:true only "
        "for an explicit MCP/local-server/install request or an MCP-only catalog entry. Wrong-kind targets "
        "return the correct call. Batch targets when possible.\n\n"
        "Hosted: status lists accounts and connection state; connect authorizes a user-named app "
        "(also use after CONNECTION_REQUIRED); reconnect repairs disconnected accounts only, unless force=true. "
        "For a second account, connect with a new short lowercase alias from the user's words; ask if unsure. "
        "rename names an account. Desktop-only display shows all connectors and waits for Continue (300s); "
        "connectors supplies suggestions. It returns connected/newly_connected and optionally timed_out; "
        "do not repeat the displayed list. Elsewhere display returns status.\n\n"
        "Desktop, TUI and interactive CLI calls wait for the connection card and return each target's "
        "connected/skipped/not_connected status, without links. Without cards, give returned connect_url "
        "links to the USER; never open them yourself. Ask the user to confirm completion, then call status.\n\n"
        "MCP: install adds a catalog server; enable re-enables a configured server; authorize runs OAuth. "
        "Use this tool, never hand-edit mcp_servers config. Calls wait for approval cards where available; "
        "otherwise they run immediately and return any user-facing auth link. Connected tools are listed "
        "and immediately callable through tool_describe/tool_call.\n\n"
        "After a skip or timeout, do not retry automatically: continue without the app or ask in chat. "
        "A new user request permits a new attempt. Disconnect/delete/revoke is user-only: direct the user "
        "to their org's Connectors page in Nous Portal or the desktop app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ALL_ACTIONS),
                "description": (
                    "Defaults to status. connect, reconnect and rename take hosted connector slugs only. "
                    "install, enable and authorize take mcp:true targets only."
                ),
            },
            "connectors": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mcp": {
                                    "type": "boolean",
                                    "description": (
                                        "true = a local MCP server from the catalog; absent or "
                                        "false = a hosted connector account."
                                    ),
                                },
                                "alias": {
                                    "type": "string",
                                    "description": (
                                        "Hosted only: the account's name. connect: a new name adds another "
                                        "account; reconnect: repairs that account; rename: its current name "
                                        "(or its label when unnamed). One aliased target per call."
                                    ),
                                },
                                "to": {
                                    "type": "string",
                                    "pattern": ALIAS_PATTERN,
                                    "description": "rename only: the new name.",
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "description": (
                    "Targets. REQUIRED for every action but status and display "
                    "(e.g. [\"gmail\", {\"name\": \"linear\", \"mcp\": true}]); optional filter for "
                    "status. A bare slug is a hosted connector, so an MCP server needs the object "
                    "form with \"mcp\": true."
                ),
            },
            "force": {
                "type": "boolean",
                "description": "reconnect only: restart the authorization even if the app is connected (account switch).",
            },
        },
        "required": [],
    },
}


registry.register(
    name="manage_connections",
    toolset="connections",
    schema=MANAGE_CONNECTIONS_SCHEMA,
    # The portal gate decides schema presence: an account the portal has not enabled for
    # connectors never sees the tool, so the model cannot call it and read the gateway's
    # 404 back to them. The handler runs the same gate so the RPC path (methods_connectors) and
    # a cached schema agree. Read as a module attribute so tests patch
    # ``gateway.config.connectors_available`` at one seam.
    handler=lambda args, **kw: manage_connections(
        args, session_id=kw.get("session_id"), tool_call_id=kw.get("tool_call_id"),
        connectors_available=gateway_config.connectors_available,
    ),
    check_fn=lambda: gateway_config.connectors_available(),
    emoji="🔗",
)

# Registry dispatch has no card callback, so it answers with the CLI pointer.
registry.register(
    name="manage_catalog",
    toolset="catalog",
    schema=MANAGE_CATALOG_SCHEMA,
    handler=lambda args, **kw: manage_catalog(args, session_id=kw.get("session_id")),
    emoji="🧩",
)
