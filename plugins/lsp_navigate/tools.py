#!/usr/bin/env python3
"""LSP navigation tool — go-to-definition, find-references, symbol search.

Hermes' LSP subsystem was originally diagnostics-only: it fed the post-write
lint delta in ``tools/file_operations.py`` and exposed nothing for reading
code.  That left symbol navigation to ``search_files`` + ``read_file``, which
works but costs a round-trip per hop and cannot distinguish a declaration
from the dozens of textual mentions of the same name.

This tool exposes the navigation half of LSP through the same
``LSPService`` singleton, reusing already-spawned servers so a query after
the first is warm.

Design notes:

- **Capability advertisement is not trustworthy.**  csharp-ls answers
  definition/references/workspace-symbol correctly while advertising none of
  the corresponding ``*Provider`` capabilities in its initialize result.  So
  the tool probes by sending the request and treats ``MethodNotFound`` (and
  only that) as genuine non-support, surfacing ``fallback`` guidance instead
  of a misleading "0 results".
- **One project root is not one repository.**  A monorepo resolves to a
  per-server root per project/solution, and a ``workspace/symbol`` query only
  sees the root it was sent to.  ``workspace_symbols`` therefore takes a list
  of ``roots`` and fans out, merging and deduping the replies, so a caller
  cannot silently under-report by querying one project of several.
- Positions are 0-based to match the LSP spec, but ``read_file`` and grep
  output are 1-based, so the tool accepts ``line`` as 1-based (matching what
  the agent just read on screen) and converts internally.  Mixing the two
  conventions silently returns the wrong symbol.
"""

import json
import os
from typing import Any, Dict, List, Optional

KINDS = ("definition", "references", "document_symbols", "workspace_symbols")

# LSP SymbolKind → readable name, for symbol listings.
_SYMBOL_KINDS = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object",
    20: "key", 21: "null", 22: "enum_member", 23: "struct", 24: "event",
    25: "operator", 26: "type_parameter",
}

MAX_RESULTS = 100


def check_lsp_navigate_requirements() -> bool:
    """Available whenever the LSP service can be constructed."""
    try:
        from agent.lsp import get_service
    except Exception:  # noqa: BLE001
        return False
    try:
        return get_service() is not None
    except Exception:  # noqa: BLE001
        return False


def _rel(path: str, base: Optional[str]) -> str:
    if not base:
        return path
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def _flatten_symbol_tree(
    items: List[Dict[str, Any]], depth: int = 0, out: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Flatten a hierarchical ``DocumentSymbol`` tree into a linear outline.

    ``textDocument/documentSymbol`` may reply with either a flat
    ``SymbolInformation[]`` (carrying ``location``) or a nested
    ``DocumentSymbol[]`` (carrying ``range``/``selectionRange`` and
    ``children``, with no uri).  Rendering only the top level of the nested
    form shows a single useless "the file" entry, so recurse and keep the
    members — those are what makes an outline worth requesting.
    """
    if out is None:
        out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rng = item.get("selectionRange") or item.get("range") or {}
        line = (rng.get("start") or {}).get("line")
        entry: Dict[str, Any] = {}
        if item.get("name"):
            entry["name"] = item["name"]
        if line is not None:
            entry["line"] = int(line) + 1
        kind = item.get("kind")
        if isinstance(kind, int) and kind in _SYMBOL_KINDS:
            entry["kind"] = _SYMBOL_KINDS[kind]
        if depth:
            entry["depth"] = depth
        if item.get("detail"):
            entry["detail"] = str(item["detail"])[:120]
        if entry:
            out.append(entry)
        children = item.get("children")
        if isinstance(children, list) and children and depth < 3:
            _flatten_symbol_tree(children, depth + 1, out)
    return out


def _format_locations(
    items: List[Dict[str, Any]], base: Optional[str]
) -> List[Dict[str, Any]]:
    """Flatten LSP Location / LocationLink / SymbolInformation to a digest.

    Line numbers come back 1-based so they can be handed straight to
    ``read_file(offset=...)`` without another conversion step.
    """
    from agent.lsp.client import uri_to_path

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") if isinstance(item.get("location"), dict) else item
        if not isinstance(loc, dict):
            continue
        uri = loc.get("uri") or loc.get("targetUri") or ""
        rng = loc.get("range") or loc.get("targetSelectionRange") or loc.get("targetRange") or {}
        start = (rng or {}).get("start") or {}
        entry: Dict[str, Any] = {}
        if uri:
            try:
                entry["file"] = _rel(uri_to_path(uri), base)
            except Exception:  # noqa: BLE001
                entry["file"] = uri
        # documentSymbol replies carry no uri (the file is implied by the
        # request), so only emit a location line when there is a real range.
        if start.get("line") is not None:
            entry["line"] = int(start["line"]) + 1
        elif isinstance(item.get("range"), dict):
            inner = (item["range"].get("start") or {}).get("line")
            if inner is not None:
                entry["line"] = int(inner) + 1
        if isinstance(item.get("selectionRange"), dict) and "line" not in entry:
            inner = (item["selectionRange"].get("start") or {}).get("line")
            if inner is not None:
                entry["line"] = int(inner) + 1
        if item.get("name"):
            entry["name"] = item["name"]
        if item.get("containerName"):
            entry["container"] = item["containerName"]
        kind = item.get("kind")
        if isinstance(kind, int) and kind in _SYMBOL_KINDS:
            entry["kind"] = _SYMBOL_KINDS[kind]
        if entry:
            out.append(entry)
    return out


def lsp_navigate_tool(
    kind: str,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
    character: Optional[int] = None,
    symbol: Optional[str] = None,
    query: Optional[str] = None,
    roots: Optional[List[str]] = None,
) -> str:
    """Run one LSP navigation query and return a compact JSON digest."""
    if kind not in KINDS:
        return json.dumps(
            {"error": f"invalid kind '{kind}'", "valid_kinds": list(KINDS)},
            ensure_ascii=False,
        )

    try:
        from agent.lsp import get_service
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"lsp unavailable: {e}"}, ensure_ascii=False)
    svc = get_service()
    if svc is None:
        return json.dumps(
            {
                "error": "lsp service unavailable (disabled, or cwd not in a git workspace)",
                "fallback": "use search_files + read_file",
            },
            ensure_ascii=False,
        )

    abs_path = os.path.abspath(file_path) if file_path else None
    if abs_path and not os.path.isfile(abs_path):
        return json.dumps({"error": f"not a file: {abs_path}"}, ensure_ascii=False)

    # Resolve `symbol` to a position so callers don't have to count columns.
    if kind in ("definition", "references"):
        if abs_path is None:
            return json.dumps(
                {"error": f"{kind} requires file_path"}, ensure_ascii=False
            )
        if line is None:
            return json.dumps(
                {"error": f"{kind} requires line (1-based), or symbol="},
                ensure_ascii=False,
            )
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                src_lines = fh.read().splitlines()
        except OSError as e:
            return json.dumps({"error": f"cannot read {abs_path}: {e}"}, ensure_ascii=False)
        idx = int(line) - 1
        if not (0 <= idx < len(src_lines)):
            return json.dumps(
                {"error": f"line {line} out of range (file has {len(src_lines)} lines)"},
                ensure_ascii=False,
            )
        if character is None:
            if symbol:
                col = src_lines[idx].find(symbol)
                if col < 0:
                    return json.dumps(
                        {
                            "error": f"symbol {symbol!r} not found on line {line}",
                            "line_content": src_lines[idx].strip()[:200],
                        },
                        ensure_ascii=False,
                    )
                character = col
            else:
                # Default to the first non-space char — usually inside the
                # identifier for a declaration line.
                character = len(src_lines[idx]) - len(src_lines[idx].lstrip())

    result = svc.navigate_sync(
        kind,
        file_path=abs_path,
        line=(int(line) - 1) if (line is not None and kind in ("definition", "references")) else None,
        character=character,
        query=query or symbol,
        extra_roots=roots,
    )

    if result.get("error") and not result.get("results"):
        payload: Dict[str, Any] = {"error": result["error"]}
        if result.get("unsupported"):
            payload["fallback"] = (
                "server does not implement this request; use search_files instead"
            )
        if result.get("partial_errors"):
            payload["details"] = result["partial_errors"][:5]
        return json.dumps(payload, ensure_ascii=False)

    base = result.get("root") or (roots[0] if roots else None)
    raw = result.get("results") or []
    if kind == "document_symbols":
        items = _flatten_symbol_tree(raw)
    else:
        items = _format_locations(raw, base)
    out: Dict[str, Any] = {
        "kind": kind,
        "count": len(items),
        "results": items[:MAX_RESULTS],
    }
    if len(items) > MAX_RESULTS:
        out["truncated"] = f"showing {MAX_RESULTS} of {len(items)}"
    if result.get("root"):
        out["root"] = result["root"]
    if result.get("roots"):
        out["roots"] = result["roots"]
    if result.get("partial_errors"):
        out["partial_errors"] = result["partial_errors"][:5]
    if not items:
        out["note"] = (
            "no results — the symbol may be unused, or the query root may not "
            "contain its callers (pass roots=[...] for other projects)"
        )
    return json.dumps(out, ensure_ascii=False)


LSP_NAVIGATE_SCHEMA = {
    "name": "lsp_navigate",
    "description": (
        "Semantic code navigation via a language server: go-to-definition, "
        "find-references, file outline, and project-wide symbol search. "
        "Prefer this over grepping when you need the DECLARATION of a symbol "
        "or its real CALL SITES — it resolves types and scopes, so it does not "
        "return comments, strings, or unrelated same-named identifiers. "
        "Line numbers are 1-based, matching read_file and grep output. "
        "The first query on a project pays a server warm-up (seconds); later "
        "ones are fast. Falls back with a 'fallback' hint when the language "
        "server cannot answer, in which case use search_files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(KINDS),
                "description": (
                    "definition = where a symbol is declared; references = where "
                    "it is used; document_symbols = outline of one file; "
                    "workspace_symbols = find a symbol by name across projects."
                ),
            },
            "file_path": {
                "type": "string",
                "description": (
                    "File containing the symbol. Required for definition, "
                    "references, and document_symbols."
                ),
            },
            "line": {
                "type": "integer",
                "description": (
                    "1-based line of the symbol (as shown by read_file or grep). "
                    "Required for definition and references."
                ),
            },
            "symbol": {
                "type": "string",
                "description": (
                    "Symbol name on that line. Preferred over 'character' — the "
                    "tool locates the column for you. For workspace_symbols this "
                    "acts as the search query."
                ),
            },
            "character": {
                "type": "integer",
                "description": (
                    "0-based column, if you need to override the column that "
                    "'symbol' resolves to."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search string for workspace_symbols.",
            },
            "roots": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "For workspace_symbols: one path per project/solution to "
                    "search. REQUIRED for that kind, because a symbol query only "
                    "sees the project root it is sent to — in a monorepo, pass "
                    "every relevant subproject or results will be incomplete."
                ),
            },
        },
        "required": ["kind"],
    },
}
