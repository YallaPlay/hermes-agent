"""Navigation layer: client request normalization, service fan-out, tool digest.

The contracts pinned here are the ones that silently produce WRONG answers
rather than loud failures:

  - ``MethodNotFound`` must surface as "unsupported" (so the caller falls back
    to text search), never as an empty result set.  A server that cannot answer
    and a symbol with no references are different facts.
  - Capability advertisement is untrusted: csharp-ls answers navigation while
    advertising no ``*Provider`` capabilities, so ``supports()`` returning None
    must NOT gate the request.
  - ``workspace/symbol`` only sees the root it was sent to, so a monorepo query
    must fan out across roots and merge — otherwise results are silently partial.
  - Line numbers cross a 1-based (agent/grep/read_file) to 0-based (LSP)
    boundary; an off-by-one silently resolves the wrong symbol.
"""
from __future__ import annotations

import json

import pytest

from agent.lsp.protocol import ERROR_METHOD_NOT_FOUND, ERROR_CONTENT_MODIFIED
from agent.lsp.client import LSPClient, LSPRequestError


def _client() -> LSPClient:
    return LSPClient(
        server_id="fake-ls",
        workspace_root="/ws",
        command=["true"],
    )


class _Recorder:
    """Stand-in for _send_request_with_retry that records calls."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def __call__(self, method, params, *, timeout=None):
        self.calls.append((method, params, timeout))
        if self.exc is not None:
            raise self.exc
        return self.result


# ---------------------------------------------------------------- client


@pytest.mark.asyncio
async def test_method_not_found_returns_none_not_empty():
    """Unsupported must be distinguishable from "no results"."""
    c = _client()
    c._send_request_with_retry = _Recorder(
        exc=LSPRequestError(ERROR_METHOD_NOT_FOUND, "no such method")
    )
    assert await c.definition("/ws/a.cs", 0, 0) is None


@pytest.mark.asyncio
async def test_empty_result_is_empty_list_not_none():
    c = _client()
    c._send_request_with_retry = _Recorder(result=None)
    assert await c.references("/ws/a.cs", 0, 0) == []


@pytest.mark.asyncio
async def test_other_request_errors_propagate():
    """Only MethodNotFound is swallowed; real errors must not look like 'no refs'."""
    c = _client()
    c._send_request_with_retry = _Recorder(
        exc=LSPRequestError(ERROR_CONTENT_MODIFIED, "changed")
    )
    with pytest.raises(LSPRequestError):
        await c.definition("/ws/a.cs", 0, 0)


@pytest.mark.asyncio
async def test_single_location_dict_normalized_to_list():
    c = _client()
    c._send_request_with_retry = _Recorder(result={"uri": "file:///ws/a.cs"})
    out = await c.definition("/ws/a.cs", 1, 2)
    assert isinstance(out, list) and len(out) == 1


@pytest.mark.asyncio
async def test_position_and_context_passed_through():
    c = _client()
    rec = _Recorder(result=[])
    c._send_request_with_retry = rec
    await c.references("/ws/a.cs", 7, 3, include_declaration=True)
    method, params, _ = rec.calls[0]
    assert method == "textDocument/references"
    assert params["position"] == {"line": 7, "character": 3}
    assert params["context"] == {"includeDeclaration": True}


def test_supports_returns_none_when_not_advertised():
    """csharp-ls advertises nothing yet answers; None must mean 'unknown'."""
    c = _client()
    c._initialize_result = {"capabilities": {}}
    assert c.supports("definitionProvider") is None
    c._initialize_result = {"capabilities": {"definitionProvider": True}}
    assert c.supports("definitionProvider") is True
    # Servers may advertise an options object rather than a bool.
    c._initialize_result = {"capabilities": {"referencesProvider": {"workDoneProgress": True}}}
    assert c.supports("referencesProvider") is True


# ---------------------------------------------------------------- service


class _FakeClient:
    def __init__(self, server_id="fake-ls", root="/ws", symbols=None, unsupported=False):
        self.server_id = server_id
        self.workspace_root = root
        self._symbols = symbols or []
        self._unsupported = unsupported
        self.opened = []

    async def open_file(self, path, language_id="plaintext"):
        self.opened.append(path)
        return 1

    async def workspace_symbols(self, query):
        return None if self._unsupported else self._symbols

    async def definition(self, path, line, character):
        return None if self._unsupported else [{"uri": f"file://{path}"}]

    async def references(self, path, line, character, include_declaration=False):
        return [] if not self._unsupported else None

    async def document_symbols(self, path):
        return self._symbols


def _svc(monkeypatch, clients_by_probe):
    """Build a disabled-loop LSPService with _get_or_spawn stubbed."""
    from agent.lsp.manager import LSPService

    svc = LSPService.__new__(LSPService)
    svc._enabled = True
    svc._last_used = {}

    async def fake_spawn(file_path):
        for prefix, client in clients_by_probe.items():
            if str(file_path).startswith(prefix):
                return client
        return None

    svc._get_or_spawn = fake_spawn  # type: ignore[assignment]
    return svc


def _sym(name, uri, line):
    return {"name": name, "location": {"uri": uri, "range": {"start": {"line": line}}}}


@pytest.mark.asyncio
async def test_workspace_symbols_fans_out_across_roots(monkeypatch, tmp_path):
    """A symbol query must reach every root, not just the first."""
    games = tmp_path / "games"
    twin = tmp_path / "clienttwin"
    for d in (games, twin):
        d.mkdir()
        (d / "a.cs").write_text("class A {}")

    c_games = _FakeClient(root=str(games), symbols=[_sym("Foo", "file:///games/a.cs", 1)])
    c_twin = _FakeClient(root=str(twin), symbols=[_sym("Foo", "file:///twin/b.cs", 2)])
    svc = _svc(monkeypatch, {str(games): c_games, str(twin): c_twin})

    out = await svc._workspace_symbols_async("Foo", [str(games), str(twin)])
    assert len(out["results"]) == 2, "both roots must contribute"
    assert set(out["roots"]) == {str(games), str(twin)}


@pytest.mark.asyncio
async def test_workspace_symbols_dedupes_identical_hits(monkeypatch, tmp_path):
    """Two roots sharing a source dir must not double-report a symbol."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "x.cs").write_text("class X {}")
    same = [_sym("Dup", "file:///shared/x.cs", 5)]
    svc = _svc(
        monkeypatch,
        {str(a): _FakeClient(root=str(a), symbols=same),
         str(b): _FakeClient(root=str(b), symbols=list(same))},
    )
    out = await svc._workspace_symbols_async("Dup", [str(a), str(b)])
    assert len(out["results"]) == 1


@pytest.mark.asyncio
async def test_workspace_symbols_reports_partial_failure(monkeypatch, tmp_path):
    """One dead root must not silently shrink the result set."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "x.cs").write_text("class X {}")
    svc = _svc(
        monkeypatch,
        {str(a): _FakeClient(root=str(a), symbols=[_sym("Q", "file:///a/x.cs", 1)]),
         str(b): _FakeClient(root=str(b), unsupported=True)},
    )
    out = await svc._workspace_symbols_async("Q", [str(a), str(b)])
    assert len(out["results"]) == 1
    assert out["partial_errors"], "an unsupported root must be reported, not hidden"


@pytest.mark.asyncio
async def test_workspace_symbols_requires_roots(monkeypatch):
    svc = _svc(monkeypatch, {})
    out = await svc._workspace_symbols_async("Foo", None)
    assert "error" in out


@pytest.mark.asyncio
async def test_navigate_marks_unsupported_kind(monkeypatch, tmp_path):
    src = tmp_path / "a.cs"
    src.write_text("class A {}")
    svc = _svc(monkeypatch, {str(tmp_path): _FakeClient(unsupported=True)})
    out = await svc._navigate_async(
        "definition", file_path=str(src), line=0, character=0, query=None, extra_roots=None
    )
    assert out.get("unsupported") is True


@pytest.mark.asyncio
async def test_navigate_requires_position(monkeypatch, tmp_path):
    src = tmp_path / "a.cs"
    src.write_text("class A {}")
    svc = _svc(monkeypatch, {str(tmp_path): _FakeClient()})
    out = await svc._navigate_async(
        "references", file_path=str(src), line=None, character=None, query=None, extra_roots=None
    )
    assert "requires line and character" in out["error"]


def test_probe_file_for_root_finds_source_file(tmp_path):
    from agent.lsp.manager import LSPService

    svc = LSPService.__new__(LSPService)
    (tmp_path / "obj").mkdir()
    (tmp_path / "obj" / "junk.cs").write_text("// generated")
    deep = tmp_path / "src" / "nested"
    deep.mkdir(parents=True)
    real = deep / "Real.cs"
    real.write_text("class R {}")
    found = svc._probe_file_for_root(str(tmp_path))
    assert found is not None
    assert "obj" not in found, "build output dirs must be skipped"


def test_probe_file_for_root_passthrough_for_file(tmp_path):
    from agent.lsp.manager import LSPService

    svc = LSPService.__new__(LSPService)
    f = tmp_path / "x.cs"
    f.write_text("class X {}")
    assert svc._probe_file_for_root(str(f)) == str(f)


# ---------------------------------------------------------------- tool


def test_tool_rejects_bad_kind():
    from tools.lsp_navigate_tool import lsp_navigate_tool

    out = json.loads(lsp_navigate_tool(kind="teleport"))
    assert "error" in out and "valid_kinds" in out


def test_tool_converts_1based_line_to_0based(monkeypatch, tmp_path):
    """The agent reads 1-based lines; LSP wants 0-based. Off-by-one = wrong symbol."""
    from tools import lsp_navigate_tool as mod

    src = tmp_path / "a.cs"
    src.write_text("// line one\npublic int Target() {}\n")

    captured = {}

    class _Svc:
        def navigate_sync(self, kind, **kw):
            captured.update(kw)
            captured["kind"] = kind
            return {"results": [], "root": str(tmp_path)}

    monkeypatch.setattr(mod, "get_service", lambda: _Svc(), raising=False)
    monkeypatch.setitem(
        __import__("sys").modules, "agent.lsp",
        type("M", (), {"get_service": staticmethod(lambda: _Svc())}),
    )
    out = json.loads(
        mod.lsp_navigate_tool(kind="references", file_path=str(src), line=2, symbol="Target")
    )
    assert captured["line"] == 1, "1-based 2 must become 0-based 1"
    assert captured["character"] == src.read_text().splitlines()[1].index("Target")
    assert out["count"] == 0


def test_tool_symbol_not_on_line_is_an_error(monkeypatch, tmp_path):
    from tools import lsp_navigate_tool as mod

    src = tmp_path / "a.cs"
    src.write_text("class A {}\n")
    monkeypatch.setitem(
        __import__("sys").modules, "agent.lsp",
        type("M", (), {"get_service": staticmethod(lambda: object())}),
    )
    out = json.loads(
        mod.lsp_navigate_tool(kind="definition", file_path=str(src), line=1, symbol="Nope")
    )
    assert "not found on line" in out["error"]


def test_tool_formats_locations_1based(tmp_path):
    from tools.lsp_navigate_tool import _format_locations

    items = [{"uri": f"file://{tmp_path}/a.cs", "range": {"start": {"line": 41}}}]
    out = _format_locations(items, str(tmp_path))
    assert out[0]["line"] == 42, "0-based 41 must render as 1-based 42"
    assert out[0]["file"] == "a.cs"


def test_tool_format_tolerates_null_location():
    """A null/!dict location must be skipped, not crash the digest."""
    from tools.lsp_navigate_tool import _format_locations

    assert _format_locations([{"location": None}, "junk", {}], None) == []  # type: ignore[list-item]


def test_tool_format_handles_location_link():
    from tools.lsp_navigate_tool import _format_locations

    items = [{"targetUri": "file:///ws/b.cs", "targetSelectionRange": {"start": {"line": 3}}}]
    out = _format_locations(items, None)
    assert out[0]["line"] == 4


def test_flatten_symbol_tree_recurses_into_members():
    """A nested DocumentSymbol tree must yield members, not just the file node."""
    from tools.lsp_navigate_tool import _flatten_symbol_tree

    tree = [{
        "name": "StorageGame.cs", "kind": 1,
        "range": {"start": {"line": 0}},
        "children": [{
            "name": "StorageGame", "kind": 5,
            "selectionRange": {"start": {"line": 22}},
            "children": [
                {"name": "GameId", "kind": 7, "selectionRange": {"start": {"line": 30}}},
                {"name": "GetRequestedBotVersion", "kind": 6,
                 "selectionRange": {"start": {"line": 98}}, "detail": "int()"},
            ],
        }],
    }]
    out = _flatten_symbol_tree(tree)
    names = [e["name"] for e in out]
    assert "GetRequestedBotVersion" in names, "members must survive flattening"
    method = next(e for e in out if e["name"] == "GetRequestedBotVersion")
    assert method["line"] == 99, "0-based 98 must render 1-based 99"
    assert method["kind"] == "method"
    assert method["depth"] == 2


def test_flatten_symbol_tree_tolerates_junk():
    from tools.lsp_navigate_tool import _flatten_symbol_tree

    assert _flatten_symbol_tree([None, "x", {}]) == []  # type: ignore[list-item]


def test_tool_reports_unsupported_with_fallback(monkeypatch, tmp_path):
    from tools import lsp_navigate_tool as mod

    src = tmp_path / "a.cs"
    src.write_text("class A {}\n")

    class _Svc:
        def navigate_sync(self, kind, **kw):
            return {"error": "server does not implement definition", "unsupported": True}

    monkeypatch.setitem(
        __import__("sys").modules, "agent.lsp",
        type("M", (), {"get_service": staticmethod(lambda: _Svc())}),
    )
    out = json.loads(
        mod.lsp_navigate_tool(kind="definition", file_path=str(src), line=1, symbol="A")
    )
    assert "fallback" in out and "search_files" in out["fallback"]
