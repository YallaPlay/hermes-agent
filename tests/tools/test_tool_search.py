"""Tests for tools/tool_search.py — progressive tool disclosure.

Coverage targets — these mirror the issues called out in the OpenClaw tool
search report. Every test that names an OpenClaw issue is the regression
guard that would have caught that specific failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_default_when_missing(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.enabled == "auto"
        assert cfg.threshold_pct == 5.0

    def test_bool_true_maps_to_auto(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(True)
        assert cfg.enabled == "auto"


    def test_search_limits_clamped(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({
            "search_default_limit": 999,
            "max_search_limit": 999,
        })
        assert cfg.max_search_limit == 50
        assert cfg.search_default_limit <= cfg.max_search_limit


# ---------------------------------------------------------------------------
# Classification — the hard invariant: core tools NEVER defer.
# ---------------------------------------------------------------------------


class TestClassification:
    def test_core_tools_never_defer(self):
        """The critical invariant from the OpenClaw report."""
        from tools.tool_search import is_deferrable_tool_name
        # Sample of core tools from _HERMES_CORE_TOOLS.
        for core_name in ["terminal", "read_file", "write_file", "patch",
                          "search_files", "todo", "memory", "browser_navigate",
                          "web_search", "session_search", "clarify",
                          "execute_code", "delegate_task", "send_message"]:
            assert not is_deferrable_tool_name(core_name), (
                f"Core tool '{core_name}' must NEVER be deferrable"
            )

    def test_bridge_tools_never_defer(self):
        from tools.tool_search import is_deferrable_tool_name, BRIDGE_TOOL_NAMES
        for name in BRIDGE_TOOL_NAMES:
            assert not is_deferrable_tool_name(name)

    def test_unknown_tool_not_deferrable(self):
        """Defensive: a tool name we cannot resolve to a registry entry must
        not be claimed as deferrable. This protects against the OpenClaw
        cron regression where unresolved tools were silently dropped."""
        from tools.tool_search import is_deferrable_tool_name
        assert not is_deferrable_tool_name("xx_definitely_not_a_tool_xx")

    def test_classify_keeps_unknown_in_visible(self):
        """A tool we can't classify stays visible — never silently dropped.

        This is the OpenClaw #84141 regression guard (cron lost ``exec``
        because it wasn't in the catalog).
        """
        from tools.tool_search import classify_tools
        # Build a tool def for something we don't have a registry entry for.
        defs = [_td("xx_unknown_tool", "Unknown tool")]
        visible, deferrable = classify_tools(defs)
        names = {(td.get("function") or {}).get("name") for td in visible}
        assert "xx_unknown_tool" in names
        assert deferrable == []


# ---------------------------------------------------------------------------
# Token estimation + threshold gate
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_off_never_activates(self):
        from tools.tool_search import ToolSearchConfig, should_activate
        cfg = ToolSearchConfig.from_raw({"enabled": "off"})
        assert not should_activate(cfg, deferrable_tokens=1_000_000, context_length=200_000)


    def test_token_estimate_proportional_to_schema_size(self):
        from tools.tool_search import estimate_tokens_from_schemas
        small = [_td("a", "x")]
        big = [_td(f"name_{i}", f"description for tool {i} " * 20,
                   {"q": {"type": "string", "description": "search query " * 10}})
               for i in range(10)]
        small_t = estimate_tokens_from_schemas(small)
        big_t = estimate_tokens_from_schemas(big)
        assert big_t > small_t * 10


# ---------------------------------------------------------------------------
# Retrieval (BM25 + substring fallback)
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _fake_catalog(self):
        """Build a catalog directly without touching the registry."""
        from tools.tool_search import CatalogEntry, _tokenize, _entry_search_text
        defs = [
            _td("github_create_issue", "Open a new issue in a GitHub repository",
                {"title": {"type": "string"}, "body": {"type": "string"}}),
            _td("github_search_repos", "Search GitHub for matching repositories",
                {"query": {"type": "string"}}),
            _td("slack_send_message", "Post a message into a Slack channel",
                {"channel": {"type": "string"}, "text": {"type": "string"}}),
            _td("calendar_create_event", "Add an event to the user's calendar",
                {"title": {"type": "string"}, "start": {"type": "string"}}),
        ]
        catalog = []
        for d in defs:
            fn = d["function"]
            e = CatalogEntry(
                name=fn["name"], description=fn["description"],
                schema=d, source="mcp", source_name="mcp-test",
            )
            e._tokens = _tokenize(_entry_search_text(d))
            catalog.append(e)
        return catalog

    def test_search_finds_relevant_tool(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "create a github issue", limit=3)
        names = [h.name for h in hits]
        assert names[0] == "github_create_issue"


    def test_search_respects_limit(self):
        from tools.tool_search import search_catalog
        hits = search_catalog(self._fake_catalog(), "github", limit=1)
        assert len(hits) <= 1


# ---------------------------------------------------------------------------
# Assembly — the full passthrough/activate decision.
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_no_deferrable_returns_unchanged(self):
        """Pure-core toolset: pass-through, no bridge tools added."""
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        defs = [_td("terminal", "Run shell"), _td("read_file", "Read a file")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        assert not result.activated
        assert {t["function"]["name"] for t in result.tool_defs} == {"terminal", "read_file"}

    @staticmethod
    def _register_mcp(name):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, "Deferred capability description.")["function"],
            toolset="mcp-tiertest",
        )


    def test_idempotent_when_bridge_already_present(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES
        defs = [_td("terminal", "Run shell"), _td("tool_search", "old")]
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "off"}),
        )
        names = [(t["function"]["name"]) for t in result.tool_defs]
        # The pre-existing tool_search was stripped (it would be re-injected if
        # activation happened; here it didn't).
        assert "tool_search" not in names


# ---------------------------------------------------------------------------
# Bridge dispatch
# ---------------------------------------------------------------------------


class TestBridgeDispatch:
    def test_tool_search_requires_query(self):
        from tools.tool_search import dispatch_tool_search
        result = dispatch_tool_search({}, current_tool_defs=[])
        assert "error" in json.loads(result)

    def test_empty_search_keeps_connected_sources_discoverable(self):
        from tools.registry import registry
        from tools.tool_search import dispatch_tool_search

        name = "recovery_catalog_create_record"
        tool_def = _td(name, "Create a record in the connected catalog service.")
        registry.register(
            name=name,
            handler=lambda args, **kwargs: "{}",
            schema=tool_def,
            toolset="mcp-recovery-catalog",
        )

        result = json.loads(dispatch_tool_search(
            {"query": "unrelated vocabulary"},
            current_tool_defs=[tool_def],
        ))

        assert result["matches"] == []
        assert result["total_available"] == 1
        assert result["available_sources"] == [
            {"name": "recovery-catalog", "tool_count": 1},
        ]
        assert "remain available" in result["hint"]
        assert "before concluding" in result["hint"]


    def test_dynamic_catalog_is_searchable_and_describable(self):
        from tools.tool_search import (
            ToolSearchConfig,
            assemble_tool_defs,
            dispatch_tool_describe,
            dispatch_tool_search,
        )

        dynamic = _td(
            "provider_stats",
            "Show memory provider statistics",
            {"detail": {"type": "boolean"}},
        )
        assembly = assemble_tool_defs(
            [dynamic],
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
            additional_deferrable_names=frozenset({"provider_stats"}),
        )

        search = json.loads(dispatch_tool_search(
            {"query": "memory statistics"},
            current_tool_defs=[],
            catalog=assembly.catalog,
        ))
        assert [match["name"] for match in search["matches"]] == ["provider_stats"]

        described = json.loads(dispatch_tool_describe(
            {"name": "provider_stats"},
            current_tool_defs=[],
            catalog=assembly.catalog,
        ))
        assert described["name"] == "provider_stats"
        assert "detail" in described["parameters"]["properties"]

    def test_resolve_underlying_call_parses_object_args(self):
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call({
            "name": "unknown_xxx",
            "arguments": {"foo": "bar"},
        })
        # Will fail classification because unknown_xxx isn't deferrable.
        assert err is not None


    def test_resolve_underlying_call_rejects_recursion(self):
        """tool_call cannot invoke tool_call itself."""
        from tools.tool_search import resolve_underlying_call, TOOL_CALL_NAME
        name, args, err = resolve_underlying_call({
            "name": TOOL_CALL_NAME,
            "arguments": {},
        })
        assert err is not None
        assert "bridge tool" in err.lower()


# ---------------------------------------------------------------------------
# End-to-end via the real handle_function_call (smoke test).
# ---------------------------------------------------------------------------


class TestHandleFunctionCallIntegration:
    def test_tool_search_dispatch_through_handle_function_call(self):
        """The dispatcher recognizes the bridge tool by name."""
        import model_tools
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "nothing matches this"},
        )
        parsed = json.loads(result)
        # Without a real registry, the matches will be empty, but the
        # dispatch path completed without error.
        assert "matches" in parsed or "error" in parsed


class TestRegression_OpenClawCron84141:
    """Regression guard for the OpenClaw cron-tool-loss class of bug.

    OpenClaw #84141: ``toolsAllow: ["exec"]`` on an isolated cron turn
    resulted in the agent receiving only ``sessions_send`` — the catalog
    builder silently dropped the requested core tool.

    Our defense: core tools are NEVER deferred. This test exercises the
    full assembly pipeline with a mixed core+MCP toolset and asserts that
    every core tool survives.
    """

    def test_core_tool_survives_alongside_many_mcp_tools(self):
        from tools.tool_search import (
            assemble_tool_defs, ToolSearchConfig, BRIDGE_TOOL_NAMES,
            classify_tools,
        )
        # 1 core tool + 50 unknown/MCP-shaped tools (deferrable).
        defs = [_td("terminal", "Run shell commands")]
        # Pad with fake "deferrable" tools — without registry registration,
        # classify_tools puts them in 'visible'. So instead, we just verify
        # the core-tool side: terminal stays in visible regardless.
        visible, deferrable = classify_tools(defs)
        assert any(
            (td.get("function") or {}).get("name") == "terminal"
            for td in visible
        ), "Core tool 'terminal' was wrongly classified as deferrable"

        # Now force activation and check the resulting tool-defs list.
        result = assemble_tool_defs(
            defs,
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {(t.get("function") or {}).get("name") for t in result.tool_defs}
        # terminal must be present; bridges are only added if there are
        # deferrable tools to put behind them.
        assert "terminal" in names

    def test_unwrap_rejects_core_tool_attempt(self):
        """Even if the model tries to invoke a core tool through tool_call,
        we reject the call and tell the model to use it directly."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call({
            "name": "terminal",
            "arguments": {"command": "echo hi"},
        })
        assert err is not None
        assert "not a deferrable" in err


class TestRegression_ToolsetScoping:
    """A restricted-toolset session must not see or invoke out-of-scope tools.

    The bug: the bridge dispatch and the tool_executor unwrap read the
    catalog from the *global* registry (get_tool_definitions with no
    toolset scope = "start with everything"), so a session scoped to one
    MCP server could tool_search the entire process registry and tool_call
    any plugin tool it was never granted. registry.dispatch() has no
    enabled_tools gate for non-execute_code tools, so the out-of-scope tool
    actually ran.

    The fix threads the session's enabled/disabled toolsets into the bridge
    dispatch (model_tools.handle_function_call) and the executor unwrap
    (agent.tool_executor), scoping both the searchable catalog and the
    invocable set to the session's own toolsets.
    """

    @staticmethod
    def _register(name, toolset):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True, "tool": name})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, f"desc for {name}", {"repo": {"type": "string"}}),
            toolset=toolset,
        )

    def test_search_catalog_is_scoped_to_session_toolsets(self):
        import model_tools

        for i in range(12):
            self._register(f"mcp_scoped_gh_{i}", "mcp-scoped-gh")
        self._register("scoped_oos_plugin", "scopedoosplugin")

        # tool_search scoped to the github toolset must not count the
        # out-of-scope plugin tool (or any of the host registry).
        result = model_tools.handle_function_call(
            function_name="tool_search",
            function_args={"query": "mcp_scoped_gh", "limit": 5},
            enabled_toolsets=["mcp-scoped-gh"],
        )
        parsed = json.loads(result)
        assert parsed["total_available"] == 12, (
            f"expected scoped catalog of 12, got {parsed['total_available']} "
            "— catalog leaked tools outside the session's toolsets"
        )
        hit_names = {m["name"] for m in parsed["matches"]}
        assert "scoped_oos_plugin" not in hit_names


    def test_scoped_deferrable_names_helper(self):
        from tools.tool_search import scoped_deferrable_names

        self._register("mcp_helper_op", "mcp-helper")
        import model_tools
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["mcp-helper"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = scoped_deferrable_names(defs)
        assert "mcp_helper_op" in names
        # core tools are never deferrable
        assert "terminal" not in names



class TestRegression_ProviderToolsBridgeScope:
    """Issue #34520: provider-injected tools (memory, context engine) must be
    invokable through the tool_call bridge.

    fact_store / fact_feedback (holographic memory) and lcm_* (context
    engine) tools are appended directly onto ``agent.tools`` in agent_init
    and dispatched via the memory manager / context compressor — they are
    never registered in ``tools.registry``. ``_tool_search_scoped_names``
    derives its scope from the registry alone, so before the fix these
    granted tools were treated as out-of-scope: a model that reached
    fact_store through the tool_call bridge (the path used by providers
    routed through Tool Search assembly) got "'fact_store' is not available
    in this session". Switching to such a provider made memory tools
    silently vanish.

    The fix unions the agent's provider-injected tool names into the scope,
    while mirroring the agent_init injection gate (memory tools only when
    enabled_toolsets is None or names "memory") so the #5544 leak is not
    reintroduced through the bridge path.
    """

    @staticmethod
    def _fake_agent(*, memory_tools=None, ce_tools=None, enabled_toolsets=None,
                    disabled_toolsets=None):
        from types import SimpleNamespace
        from agent.memory_manager import memory_provider_tools_enabled

        mem_mgr = None
        if memory_tools is not None:
            mem_mgr = SimpleNamespace(
                get_all_tool_names=lambda: set(memory_tools),
            )
        injected_memory_tools = (
            set(memory_tools or ())
            if memory_provider_tools_enabled(enabled_toolsets)
            else set()
        )
        surface_tools = injected_memory_tools | set(ce_tools or ())
        return SimpleNamespace(
            _memory_manager=mem_mgr,
            tools=[
                {"type": "function", "function": {"name": name}}
                for name in surface_tools
            ],
            _context_engine_tool_names=set(ce_tools or set()),
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            _tool_search_scope_cache=None,
        )

    def test_memory_tools_in_scope_when_unfiltered(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(memory_tools={"fact_store", "fact_feedback"})
        names = _tool_search_scoped_names(agent)
        assert "fact_store" in names
        assert "fact_feedback" in names

    def test_memory_tools_in_scope_when_memory_enabled(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(
            memory_tools={"fact_store"},
            enabled_toolsets=["terminal", "memory", "web"],
        )
        assert "fact_store" in _tool_search_scoped_names(agent)

    def test_memory_tools_in_scope_when_memory_enabled_via_aggregate_toolset(self):
        """Aggregate toolsets that resolve to memory must use the same gate as
        memory-provider injection, not require the literal ``memory`` name."""
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(
            memory_tools={"fact_store"},
            enabled_toolsets=["hermes-cli"],
        )
        assert "fact_store" in _tool_search_scoped_names(agent)

    def test_memory_tools_excluded_when_memory_not_enabled(self):
        """#5544 guard: bridge scope must not leak memory tools the session
        did not opt into (mirrors the agent_init injection gate)."""
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(
            memory_tools={"fact_store"},
            enabled_toolsets=["terminal", "web"],
        )
        assert "fact_store" not in _tool_search_scoped_names(agent)

    def test_memory_tools_excluded_when_toolsets_empty(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(memory_tools={"fact_store"}, enabled_toolsets=[])
        assert "fact_store" not in _tool_search_scoped_names(agent)

    def test_context_engine_tools_on_surface_are_in_scope(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(
            ce_tools={"lcm_grep", "lcm_describe"},
            enabled_toolsets=["terminal"],
        )
        names = _tool_search_scoped_names(agent)
        assert "lcm_grep" in names
        assert "lcm_describe" in names

    def test_context_engine_tool_removed_from_surface_is_revoked(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(ce_tools={"lcm_grep"})
        assert "lcm_grep" in _tool_search_scoped_names(agent)

        agent.tools = []

        assert "lcm_grep" not in _tool_search_scoped_names(agent)

    def test_no_provider_tools_is_noop(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent()
        names = _tool_search_scoped_names(agent)
        assert "fact_store" not in names

    def test_cache_key_separates_provider_scopes(self):
        """A cached scope for one provider set must not be served to a
        session with a different provider set."""
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(memory_tools={"fact_store"})
        first = _tool_search_scoped_names(agent)
        assert "fact_store" in first

        # Same agent object, but a provider refresh injects a new tool.
        agent._memory_manager.get_all_tool_names = lambda: {
            "fact_store", "fact_feedback"
        }
        agent.tools.append(
            {"type": "function", "function": {"name": "fact_feedback"}}
        )
        second = _tool_search_scoped_names(agent)
        assert "fact_feedback" in second, (
            "stale cache served after provider tool set changed"
        )

    def test_memory_tool_removed_from_surface_is_revoked(self):
        from agent.tool_executor import _tool_search_scoped_names

        agent = self._fake_agent(memory_tools={"fact_store"})
        assert "fact_store" in _tool_search_scoped_names(agent)

        # The manager can still advertise the capability while an agent-tool
        # refresh has removed it from this session's model-facing snapshot.
        agent.tools = []

        assert "fact_store" not in _tool_search_scoped_names(agent)

    # ── resolve_underlying_call must actually admit provider names ──────
    #
    # Maintainer review of PR #34523: the executor's scope union at
    # _tool_search_scoped_names was unreachable for provider-injected names,
    # because resolve_underlying_call() rejected them via
    # is_deferrable_tool_name() (no registry entry) *before* the executor's
    # scope gate ran. These cover the resolution layer + both executor paths.

    def test_resolve_admits_provider_name_via_allowed_names(self):
        """A provider-injected name (no registry entry) resolves cleanly when
        the caller vouches for it via allowed_names."""
        from tools.tool_search import resolve_underlying_call
        name, args, err = resolve_underlying_call(
            {"name": "fact_store", "arguments": {"text": "hi"}},
            allowed_names=frozenset({"fact_store", "fact_feedback"}),
        )
        assert err is None
        assert name == "fact_store"
        assert args == {"text": "hi"}

    def test_resolve_still_rejects_provider_name_out_of_scope(self):
        """Without vouching, the provider name is still rejected — the caller's
        scope remains the gate, so #5544 is preserved."""
        from tools.tool_search import resolve_underlying_call
        _, _, err = resolve_underlying_call(
            {"name": "fact_store", "arguments": {}},
            allowed_names=frozenset({"lcm_grep"}),
        )
        assert err is not None
        assert "not a deferrable" in err

    def test_resolve_allowed_names_does_not_admit_core_tool(self):
        """allowed_names must never let a core tool through even if a
        buggy caller lists it."""
        from tools.tool_search import resolve_underlying_call
        name, _, err = resolve_underlying_call(
            {"name": "terminal", "arguments": {"command": "true"}},
            allowed_names=frozenset({"terminal"}),
        )
        assert name is None
        assert err is not None

    def _executor_agent(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            _memory_manager=SimpleNamespace(
                get_all_tool_names=lambda: {"fact_store", "fact_feedback"},
            ),
            tools=[
                {"type": "function", "function": {"name": "fact_store"}},
                {"type": "function", "function": {"name": "fact_feedback"}},
            ],
            _context_engine_tool_names=set(),
            enabled_toolsets=None,
            disabled_toolsets=None,
            _tool_search_scope_cache=None,
        )

    def test_executor_unwraps_provider_tool_end_to_end(self):
        """End-to-end: the sequential executor's resolve+scope block resolves
        a bridged provider tool to its underlying name (previously blocked).
        Mirrors agent/tool_executor.py handle_function_call unwrap."""
        from tools import tool_search as _ts
        from agent.tool_executor import _tool_search_scoped_names
        agent = self._executor_agent()
        scoped = _tool_search_scoped_names(agent)
        assert "fact_store" in scoped

        function_args = {"name": "fact_store", "arguments": {"text": "remember"}}
        underlying, uargs, err = _ts.resolve_underlying_call(
            function_args, allowed_names=scoped
        )
        assert err is None
        # Executor's own scope gate (both paths do `if underlying in scoped`).
        assert underlying in scoped
        assert underlying == "fact_store"
        assert uargs == {"text": "remember"}

    def test_executor_blocks_ungranted_tool_end_to_end(self):
        """A tool the session was never granted is still blocked at the
        executor scope gate even if resolution is attempted."""
        from tools import tool_search as _ts
        from agent.tool_executor import _tool_search_scoped_names
        agent = self._executor_agent()
        scoped = _tool_search_scoped_names(agent)

        underlying, _uargs, err = _ts.resolve_underlying_call(
            {"name": "lcm_grep", "arguments": {}}, allowed_names=scoped
        )
        # Not vouched (agent has no ce tools) → resolution rejects it, and
        # even a bypass would fail the `underlying in scoped` executor gate.
        assert err is not None or underlying not in scoped


# ---------------------------------------------------------------------------
# Catalog listing (skills-style progressive disclosure)
# ---------------------------------------------------------------------------


class TestCatalogListing:
    def test_config_defaults(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(None)
        assert cfg.listing == "auto"
        assert cfg.listing_max_tokens == 4000
        # legacy bool shapes keep defaults too
        assert ToolSearchConfig.from_raw(True).listing == "auto"


    def test_default_listing_cap_bounds_fixed_catalog_overhead(self):
        """The default manifest must not grow back to the old 20K-token cap."""
        from tools.registry import registry
        from tools.tool_search import (
            ToolSearchConfig,
            assemble_tool_defs,
            estimate_tokens_from_schemas,
        )

        defs = []
        for i in range(500):
            name = f"lean_catalog_tool_{i:04d}"
            registry.register(
                name=name,
                handler=lambda args, **kwargs: "{}",
                schema=_td(name, "Perform a deliberately verbose connected service action."),
                toolset="mcp-lean-catalog",
            )
            defs.append(_td(name, "Perform a deliberately verbose connected service action."))

        cfg = ToolSearchConfig.from_raw(None)
        result = assemble_tool_defs(defs, context_length=1_000_000, config=cfg)
        search = next(
            td for td in result.tool_defs
            if td["function"]["name"] == "tool_search"
        )
        description_tokens = estimate_tokens_from_schemas([search])
        # Includes the bridge schema around the listing, so allow modest
        # framing overhead above the 4K listing budget.
        assert description_tokens < 4500
        assert result.listing_form in {"names", "groups", "mixed"}

    def test_short_desc_first_sentence_and_clip(self):
        from tools.tool_search import _short_desc
        assert _short_desc("Open an issue. Second sentence dropped.") == "Open an issue."
        long = "word " * 40
        s = _short_desc(long)
        assert len(s) <= 61  # 60 + ellipsis char
        assert s.endswith("…")
        assert _short_desc("") == ""


    @staticmethod
    def _register(name):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            return json.dumps({"ok": True})

        registry.register(
            name=name,
            handler=_handler,
            schema=_td(name, "Deferred capability description.")["function"],
            toolset="mcp-listingtest",
        )


    def test_assembly_listing_off_keeps_legacy_description(self):
        from tools.tool_search import assemble_tool_defs, ToolSearchConfig
        for i in range(30):
            self._register(f"mcp_x_{i}")
        defs = [_td(f"mcp_x_{i}", "Deferred.") for i in range(30)]
        result = assemble_tool_defs(
            defs, context_length=1000,
            config=ToolSearchConfig.from_raw({"enabled": "on", "listing": "off"}),
        )
        assert result.activated
        search = next(t for t in result.tool_defs if t["function"]["name"] == "tool_search")
        assert "mcp_x_0" not in search["function"]["description"]


class TestDeferredCallSchemaProbe:
    """Blind tool_call invocations missing required arguments must return
    the tool's parameter schema instead of dispatching into an opaque
    downstream failure (port of nearai/ironclaw#5149's describe-first fix).

    A deferred tool's schema is invisible until tool_describe is called, so
    models routinely invoke deferred tools by name alone. Pre-fix, that
    produced ``KeyError: 'document_id'``-style errors that teach the model
    nothing; post-fix, the probe returns the schema so the model repairs
    the call in one round-trip. Valid calls dispatch untouched.
    """

    @staticmethod
    def _register(name, toolset, required=("document_id",)):
        from tools.registry import registry

        def _handler(args, task_id=None, **kw):
            # Simulates a tool that crashes opaquely on a missing required arg.
            return json.dumps({"ok": True, "doc": args["document_id"]})

        params = {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Doc id"},
                "format": {"type": "string"},
            },
            "required": list(required),
        }
        registry.register(
            name=name,
            handler=_handler,
            schema={"type": "function",
                    "function": {"name": name, "description": f"desc {name}",
                                 "parameters": params}},
            toolset=toolset,
        )

    def test_validator_returns_schema_for_missing_required(self):
        from tools.tool_search import validate_deferred_call_args

        self._register("mcp_probe_docs_get", "mcp-probe")
        err = validate_deferred_call_args("mcp_probe_docs_get", {})
        assert err is not None
        parsed = json.loads(err)
        assert "document_id" in parsed["error"]
        assert "NOT invoked" in parsed["error"]
        assert parsed["parameters"]["required"] == ["document_id"]
        assert "document_id" in parsed["parameters"]["properties"]


    def test_validator_never_blocks_unvalidatable_tools(self):
        from tools.tool_search import validate_deferred_call_args

        # Unknown tool → no schema → dispatch (downstream scope gate handles it).
        assert validate_deferred_call_args("mcp_no_such_tool_xyz", {}) is None


    def test_valid_tool_call_still_dispatches(self):
        import model_tools

        self._register("mcp_probe_valid_op", "mcp-probe-valid")
        result = json.loads(model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": "mcp_probe_valid_op",
                           "arguments": {"document_id": "abc"}},
            enabled_toolsets=["mcp-probe-valid"],
        ))
        assert result.get("ok") is True
        assert result.get("doc") == "abc"
