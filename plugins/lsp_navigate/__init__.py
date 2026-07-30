"""LSP navigation plugin — bundled, auto-loaded.

Registers one tool, ``lsp_navigate``, exposing the *navigation* half of the
LSP subsystem: go-to-definition, find-references, file outline, and
project-wide symbol search.  Before this existed, ``agent/lsp/`` was
diagnostics-only — it fed the post-write lint delta in
``tools/file_operations.py`` and offered nothing for READING code, so symbol
lookups fell back to grep, which costs a round-trip per hop and cannot tell a
declaration from a same-named mention in a comment.

Why a plugin rather than a ``tools/`` built-in?

- Navigation is **optional capability**, not a foundational one.  It only
  functions for languages whose language server is installed (see
  ``hermes lsp list``), so unlike ``read_file`` or ``terminal`` it cannot be
  relied on unconditionally — ``search_files`` remains the universal fallback.
- Being a plugin makes it **disableable** (``plugins.disabled``), which matters
  because a registered tool costs schema tokens in every session, including the
  many that never touch code.
- ``kind: backend`` + bundled means it still auto-loads with no user opt-in, so
  the default experience matches a built-in.

The engine it drives (``agent/lsp/``) stays in core, because core itself
consumes it for diagnostics.  Only the agent-facing read surface lives here.
"""

from __future__ import annotations

from plugins.lsp_navigate.tools import (
    LSP_NAVIGATE_SCHEMA,
    check_lsp_navigate_requirements,
    lsp_navigate_tool,
)


def _handle(args: dict, **_kw) -> str:
    return lsp_navigate_tool(
        kind=args.get("kind", ""),
        file_path=args.get("file_path"),
        line=args.get("line"),
        character=args.get("character"),
        symbol=args.get("symbol"),
        query=args.get("query"),
        roots=args.get("roots"),
    )


def register(ctx) -> None:
    """Register the ``lsp_navigate`` tool into the ``development`` toolset."""
    ctx.register_tool(
        name="lsp_navigate",
        toolset="development",
        schema=LSP_NAVIGATE_SCHEMA,
        handler=_handle,
        check_fn=check_lsp_navigate_requirements,
        emoji="🧭",
    )
