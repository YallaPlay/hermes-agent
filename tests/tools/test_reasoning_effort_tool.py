"""Tests for the set_reasoning_effort agent tool."""

import json

import pytest

from tools.reasoning_effort_tool import set_reasoning_effort, VALID_LEVELS


class FakeAgent:
    """Minimal stand-in for the live agent object."""
    def __init__(self, reasoning_config=None, cli_owner=None):
        self.reasoning_config = reasoning_config
        if cli_owner is not None:
            self._cli_owner = cli_owner


class FakeOwner:
    def __init__(self):
        self.reasoning_config = None


def _load(result: str) -> dict:
    return json.loads(result)


def test_sets_high_effort():
    agent = FakeAgent()
    out = _load(set_reasoning_effort(agent, "high", "analytics cohort query"))
    assert out["success"] is True
    assert out["level"] == "high"
    assert out["changed"] is True
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_none_disables_reasoning():
    agent = FakeAgent(reasoning_config={"enabled": True, "effort": "medium"})
    out = _load(set_reasoning_effort(agent, "none"))
    assert out["success"] is True
    assert agent.reasoning_config == {"enabled": False}


def test_idempotent_noop_when_already_at_level():
    agent = FakeAgent(reasoning_config={"enabled": True, "effort": "high"})
    out = _load(set_reasoning_effort(agent, "high"))
    assert out["success"] is True
    assert out["changed"] is False
    assert "already" in out["note"].lower()


def test_rejects_invalid_level():
    agent = FakeAgent()
    out = _load(set_reasoning_effort(agent, "banana"))
    assert out["success"] is False
    assert "invalid" in out["error"].lower()
    # unchanged
    assert agent.reasoning_config is None


def test_rejects_empty_level():
    agent = FakeAgent()
    out = _load(set_reasoning_effort(agent, ""))
    assert out["success"] is False
    assert agent.reasoning_config is None


@pytest.mark.parametrize("level", VALID_LEVELS)
def test_all_valid_levels_accepted(level):
    agent = FakeAgent()
    out = _load(set_reasoning_effort(agent, level))
    assert out["success"] is True
    assert out["level"] == level


def test_case_and_whitespace_normalized():
    agent = FakeAgent()
    out = _load(set_reasoning_effort(agent, "  HIGH  "))
    assert out["success"] is True
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_propagates_to_cli_owner_when_present():
    owner = FakeOwner()
    agent = FakeAgent(cli_owner=owner)
    set_reasoning_effort(agent, "xhigh")
    assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}
    assert owner.reasoning_config == {"enabled": True, "effort": "xhigh"}


def test_never_persists_to_config(monkeypatch):
    """Guard: the tool must never call save_config_value (session-scoped only)."""
    import cli
    called = {"save": False}

    def _boom(*a, **k):
        called["save"] = True
        return True

    monkeypatch.setattr(cli, "save_config_value", _boom, raising=False)
    agent = FakeAgent()
    set_reasoning_effort(agent, "high")
    assert called["save"] is False


def test_registered_in_agent_loop_tools():
    """The tool must be routed through the agent loop, not the inert registry stub."""
    import model_tools
    assert "set_reasoning_effort" in model_tools._AGENT_LOOP_TOOLS


def test_registered_in_tool_registry():
    from tools.registry import discover_builtin_tools, registry
    discover_builtin_tools()
    entry = registry.get_entry("set_reasoning_effort")
    assert entry is not None
    assert entry.toolset == "reasoning"
