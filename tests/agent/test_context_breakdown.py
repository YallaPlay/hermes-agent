"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import (
    build_session_context_report,
    compute_session_context_breakdown,
)


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "",
    volatile: str = "timestamp line",
    tools: list | None = None,
    context_length: int = 200_000,
    last_prompt_tokens: int = 0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
        {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(
        context_length=context_length,
        last_prompt_tokens=last_prompt_tokens,
    )
    return agent, {"stable": stable, "context": context, "volatile": volatile}


def test_breakdown_includes_major_categories():
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    )
    context = "# Project Context\nFollow AGENTS.md"
    volatile = "Current time: now"
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_breakdown_uses_measured_context_when_available():
    agent, parts = _make_agent(last_prompt_tokens=42_000)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["context_used"] == 42_000
    assert data["context_percent"] == 21


def test_breakdown_categories_carry_detail_rows():
    context = (
        "# Project Context\n\nThe following project context files have been loaded"
        " and should be followed:\n\n## .hermes.md\n\nrepo rules here\n\n## AGENTS.md\n\nagent rules"
    )
    stable = (
        "base guidance\n"
        "<available_skills>\n  demo:\n    - hello: hi\n    - other: desc\n</available_skills>"
    )
    history = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "output"},
    ]
    agent, parts = _make_agent(stable=stable, context=context)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    by_id = {item["id"]: item for item in data["categories"]}

    rules_labels = [row["label"] for row in by_id["rules"]["detail"]]
    assert rules_labels == [".hermes.md", "AGENTS.md"]

    tool_labels = {row["label"] for row in by_id["tool_definitions"]["detail"]}
    assert "terminal" in tool_labels

    conv_labels = [row["label"] for row in by_id["conversation"]["detail"]]
    assert conv_labels == ["1 user message", "1 assistant message", "1 tool message"]

    assert by_id["skills"]["detail"] == [{"label": "2 skills indexed", "tokens": None}]


def test_context_report_full_and_single_category():
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(context="# Project Context\n\n## AGENTS.md\n\nrules")

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        full = build_session_context_report(agent, history)
        tools_only = build_session_context_report(
            agent, history, category="tool_definitions"
        )

    assert "## System prompt" in full
    assert "## Conversation" in full
    assert "hello there" in full
    assert "## Rules (project context)" in full

    assert "### terminal" in tools_only
    assert "## System prompt" not in tools_only


def test_context_report_rejects_unknown_category():
    agent, parts = _make_agent()
    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        try:
            build_session_context_report(agent, [], category="nope")
        except ValueError as exc:
            assert "nope" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_context_report_flattens_multimodal_and_tool_calls():
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "terminal", "arguments": "{}"}}],
        },
    ]
    agent, parts = _make_agent()
    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        report = build_session_context_report(agent, history, category="conversation")

    assert "look at this" in report
    assert "[image]" in report
    assert "Tool calls:" in report
