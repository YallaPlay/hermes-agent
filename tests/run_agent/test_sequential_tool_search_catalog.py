"""Regression tests: sequential tool execution must thread the finalized
Tool Search catalog into bridge dispatch.

The finalized session surface (agent._tool_search_catalog, built by
finalize_agent_tool_surface) carries provider-injected tool schemas
(mnemosyne_*, fact_store, lcm_*) that do NOT exist in tools.registry.
model_tools.handle_function_call reconstructs the catalog registry-only
when tool_search_catalog is None, so any dispatch path that omits the
kwarg silently hides provider tools from tool_search/tool_describe.

The concurrent path (agent_runtime_helpers.invoke_tool) has threaded the
catalog since 157a6049a0, but _should_parallelize_tool_batch() routes
every single-call batch — the normal case for a lone tool_search call —
through execute_tool_calls_sequential, whose handle_function_call
callsites dropped the kwarg. Live symptom: tool_search returned
total_available: 0 in a session whose init logged "10 deferred".
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name, arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _sentinel_catalog():
    from tools.tool_search import CatalogEntry

    return (
        CatalogEntry(
            name="mnemosyne_recall",
            source="memory-provider",
            source_name="mnemosyne",
            description="Recall durable memories",
            schema={"function": {"name": "mnemosyne_recall"}},
        ),
    )


def _run_sequential_bridge_call(agent, tool_name: str):
    """Dispatch one bridge tool call through the sequential executor and
    return the kwargs handle_function_call received."""
    tc = _mock_tool_call(tool_name, json.dumps({"query": "recall memories"}))
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages: list = []
    with patch(
        "run_agent.handle_function_call",
        return_value=json.dumps({"query": "recall memories", "total_available": 1, "matches": []}),
    ) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")
    mock_hfc.assert_called_once()
    return mock_hfc.call_args.kwargs


def test_sequential_quiet_path_threads_tool_search_catalog():
    """quiet_mode branch must pass agent._tool_search_catalog to dispatch."""
    agent = _make_agent("tool_search", "web_search")
    agent.quiet_mode = True
    agent._tool_search_catalog = _sentinel_catalog()

    kwargs = _run_sequential_bridge_call(agent, "tool_search")

    assert kwargs.get("tool_search_catalog") == agent._tool_search_catalog


def test_sequential_verbose_path_threads_tool_search_catalog():
    """non-quiet (else) branch must pass agent._tool_search_catalog too."""
    agent = _make_agent("tool_search", "web_search")
    agent.quiet_mode = False
    agent.tool_progress_mode = "off"
    agent._tool_search_catalog = _sentinel_catalog()

    kwargs = _run_sequential_bridge_call(agent, "tool_search")

    assert kwargs.get("tool_search_catalog") == agent._tool_search_catalog


def test_sequential_missing_catalog_attribute_passes_none():
    """Agents without a finalized surface must degrade to None, not raise."""
    agent = _make_agent("tool_search", "web_search")
    agent.quiet_mode = True
    if hasattr(agent, "_tool_search_catalog"):
        del agent._tool_search_catalog

    kwargs = _run_sequential_bridge_call(agent, "tool_search")

    assert "tool_search_catalog" in kwargs
    assert kwargs["tool_search_catalog"] is None
