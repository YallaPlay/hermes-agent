"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import base64
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.server import HermesACPAgent, HERMES_VERSION
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes(agent):
    """Edit approval stays on the modes channel (Zed keeps its model picker);
    config_options carries only the reasoning-effort select."""
    resp = await agent.new_session(cwd="/tmp")

    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [(mode.id, mode.name) for mode in resp.modes.available_modes] == [
        ("default", "Default"),
        ("accept_edits", "Accept Edits"),
        ("dont_ask", "Don't Ask"),
    ]
    assert [opt.id for opt in resp.config_options] == ["reasoning_effort"]


@pytest.mark.asyncio
async def test_new_session_advertises_reasoning_effort_config_option(agent):
    resp = await agent.new_session(cwd="/tmp")
    (opt,) = resp.config_options
    assert opt.type == "select"
    assert opt.current_value == "default"
    assert [o.value for o in opt.options] == [
        "default", "none", "minimal", "low", "medium", "high", "xhigh",
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert getattr(state, "mode", None) == "accept_edits"


@pytest.mark.asyncio
async def test_set_config_option_reasoning_effort_applies_and_persists(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "reasoning_effort",
        resp.session_id,
        "high",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert state.effort == "high"
    assert state.agent.reasoning_config == {"enabled": True, "effort": "high"}
    (opt,) = update.config_options
    assert opt.id == "reasoning_effort"
    assert opt.current_value == "high"


@pytest.mark.asyncio
async def test_set_config_option_reasoning_effort_default_clears_override(agent):
    resp = await agent.new_session(cwd="/tmp")
    await agent.set_config_option("reasoning_effort", resp.session_id, "xhigh")
    update = await agent.set_config_option("reasoning_effort", resp.session_id, "default")
    state = agent.session_manager.get_session(resp.session_id)

    assert state.effort == ""
    assert state.agent.reasoning_config is None
    assert update.config_options[0].current_value == "default"


@pytest.mark.asyncio
async def test_set_config_option_reasoning_effort_invalid_falls_back_to_default(agent):
    resp = await agent.new_session(cwd="/tmp")
    await agent.set_config_option("reasoning_effort", resp.session_id, "ludicrous")
    state = agent.session_manager.get_session(resp.session_id)

    assert state.effort == ""
    assert state.agent.reasoning_config is None


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_agent_info(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert resp.agent_info is not None
        assert isinstance(resp.agent_info, Implementation)
        assert resp.agent_info.name == "hermes-agent"
        assert resp.agent_info.version == HERMES_VERSION

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self, agent):
        resp = await agent.initialize(protocol_version=1)
        caps = resp.agent_capabilities
        assert isinstance(caps, AgentCapabilities)
        assert caps.load_session is True
        assert caps.session_capabilities is not None
        assert caps.session_capabilities.fork is not None
        assert caps.session_capabilities.list is not None
        assert caps.session_capabilities.resume is not None

    @pytest.mark.asyncio
    async def test_initialize_capabilities_wire_format(self, agent):
        """Verify the JSON wire format uses correct aliases so ACP clients see the right keys."""
        resp = await agent.initialize(protocol_version=1)
        payload = resp.agent_capabilities.model_dump(by_alias=True, exclude_none=True)
        assert payload["loadSession"] is True
        session_caps = payload["sessionCapabilities"]
        assert "fork" in session_caps
        assert "list" in session_caps
        assert "resume" in session_caps

    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]

    @pytest.mark.asyncio
    async def test_initialize_advertises_terminal_setup_auth_when_no_provider(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: None)
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: None)

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads == [
            {
                "args": ["--setup"],
                "description": (
                    "Open Hermes' interactive model/provider setup in a terminal. "
                    "Use this when Hermes has not been configured on this machine yet."
                ),
                "id": TERMINAL_SETUP_AUTH_METHOD_ID,
                "name": "Configure Hermes provider",
                "type": "terminal",
            }
        ]


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_terminal_setup_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert resp is None


# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:
    @pytest.mark.asyncio
    async def test_new_session_creates_session(self, agent):
        resp = await agent.new_session(cwd="/home/user/project")
        assert isinstance(resp, NewSessionResponse)
        assert resp.session_id
        # Session should be retrievable from the manager
        state = agent.session_manager.get_session(resp.session_id)
        assert state is not None
        assert state.cwd == "/home/user/project"

    @pytest.mark.asyncio
    async def test_new_session_returns_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="gpt-5.4", provider="openai-codex")
        )
        acp_agent = HermesACPAgent(session_manager=manager)

        with patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("gpt-5.4", "recommended"), ("gpt-5.4-mini", "")],
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].model_id == "openai-codex:gpt-5.4"
        assert resp.models.available_models[0].description is not None
        assert "Provider:" in resp.models.available_models[0].description

    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None

    @pytest.mark.asyncio
    async def test_send_available_commands_update(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent._send_available_commands_update("session-123")

        mock_conn.session_update.assert_awaited_once()
        call = mock_conn.session_update.await_args
        assert call.kwargs["session_id"] == "session-123"
        update = call.kwargs["update"]
        assert isinstance(update, AvailableCommandsUpdate)
        assert update.session_update == "available_commands_update"
        assert [cmd.name for cmd in update.available_commands] == [
            "help",
            "model",
            "tools",
            "context",
            "reset",
            "compress",
            "steer",
            "queue",
            "version",
        ]
        model_cmd = next(
            cmd for cmd in update.available_commands if cmd.name == "model"
        )
        assert model_cmd.input is not None
        assert model_cmd.input.root.hint == "model name to switch to"

    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000

    @pytest.mark.asyncio
    async def test_send_usage_update_to_client(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.context_compressor = MagicMock(context_length=100_000)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            await agent._send_usage_update(state)

        mock_conn.session_update.assert_awaited_once()
        call = mock_conn.session_update.await_args
        assert call.kwargs["session_id"] == state.session_id
        update = call.kwargs["update"]
        assert isinstance(update, UsageUpdate)
        assert update.size == 100_000
        assert update.used == 25_000

    @pytest.mark.asyncio
    async def test_notify_midturn_usage_sends_provider_reported_pressure(
        self, agent, mock_manager
    ):
        """Mid-turn refreshes use the compressor's provider-reported prompt
        tokens (state.history is frozen until turn end)."""
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.context_compressor = MagicMock(
            context_length=100_000, last_prompt_tokens=42_000
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        loop = asyncio.get_running_loop()
        await asyncio.to_thread(agent._notify_midturn_usage, state, loop)
        await asyncio.sleep(0)

        mock_conn.session_update.assert_awaited_once()
        call = mock_conn.session_update.await_args
        assert call.kwargs["session_id"] == state.session_id
        update = call.kwargs["update"]
        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 42_000

    @pytest.mark.asyncio
    async def test_notify_midturn_usage_skips_without_real_usage(
        self, agent, mock_manager
    ):
        """No provider-reported prompt tokens yet (first API call still in
        flight, or compaction reset) — send nothing rather than a bogus 0."""
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.context_compressor = MagicMock(
            context_length=100_000, last_prompt_tokens=0
        )
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        loop = asyncio.get_running_loop()
        await asyncio.to_thread(agent._notify_midturn_usage, state, loop)
        await asyncio.sleep(0)

        mock_conn.session_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prompt_step_callback_sends_midturn_usage_update(self, agent):
        """The step callback installed by prompt() must refresh the context
        indicator between API calls, not just at turn end."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.context_compressor = MagicMock(
            context_length=100_000, last_prompt_tokens=37_000
        )

        def _run(**kwargs):
            state.agent.step_callback(1, [])
            return {"final_response": "hi", "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        usage_updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
            if isinstance(
                call.kwargs.get("update") or (call.args[1] if len(call.args) > 1 else None),
                UsageUpdate,
            )
        ]
        assert any(u.used == 37_000 for u in usage_updates)

    @pytest.mark.asyncio
    async def test_cancel_sets_event(self, agent):
        resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(resp.session_id)
        assert not state.cancel_event.is_set()
        await agent.cancel(session_id=resp.session_id)
        assert state.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session_is_noop(self, agent):
        # Should not raise
        await agent.cancel(session_id="does-not-exist")

    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None

    @pytest.mark.asyncio
    async def test_load_session_replays_subagent_child_transcript(self, agent):
        """Loading a delegate child's id streams its DB transcript and marks
        the session read-only running/idle from the live subagent registry."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        db = agent.session_manager._get_db()
        db.create_session(
            "child-replay-1", source="subagent",
            parent_session_id=new_resp.session_id,
        )
        db.append_message("child-replay-1", role="user", content="child goal text")
        db.append_message("child-replay-1", role="assistant", content="child progress")

        resp = await agent.load_session(cwd="/tmp", session_id="child-replay-1")

        assert resp is not None
        # Transcript replayed on the child id.
        replayed = [
            call.kwargs.get("update") if "update" in call.kwargs else call.args[1]
            for call in mock_conn.session_update.await_args_list
            if (call.kwargs.get("session_id") or (call.args[0] if call.args else None))
            == "child-replay-1"
        ]
        texts = [
            u.content.text for u in replayed
            if getattr(u, "session_update", None) in
            ("user_message_chunk", "agent_message_chunk")
        ]
        assert any("child goal text" in t for t in texts)
        assert any("child progress" in t for t in texts)
        # Marked as a subagent session (read-only marker for clients); not
        # running (no live registry entry for it).
        hermes_meta = (resp.field_meta or {}).get("hermes", {})
        assert hermes_meta.get("isSubagent") is True
        assert not hermes_meta.get("isRunning")

    @pytest.mark.asyncio
    async def test_load_session_marks_running_subagent_child(self, agent):
        """A child with a live entry in the subagent registry loads with
        isRunning:true so the client shows the busy indicator."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        db = agent.session_manager._get_db()
        db.create_session(
            "child-running-1", source="subagent",
            parent_session_id=new_resp.session_id,
        )
        db.append_message("child-running-1", role="user", content="running child goal")

        with patch(
            "tools.delegate_tool.list_active_subagents",
            return_value=[{
                "subagent_id": "sub-x",
                "child_session_id": "child-running-1",
                "status": "running",
            }],
        ):
            resp = await agent.load_session(cwd="/tmp", session_id="child-running-1")

        assert resp is not None
        hermes_meta = (resp.field_meta or {}).get("hermes", {})
        assert hermes_meta.get("isSubagent") is True
        assert hermes_meta.get("isRunning") is True

    @pytest.mark.asyncio
    async def test_load_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "system", "content": "hidden system"},
            {"role": "user", "content": "what controls the / slash commands?"},
            {"role": "assistant", "content": "HermesACPAgent._ADVERTISED_COMMANDS controls them."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_search_1",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"pattern":"slash commands","path":"."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_search_1",
                "content": '{"total_count":1,"matches":[{"path":"cli.py","line":42,"content":"slash commands"}]}',
            },
        ]

        mock_conn.session_update.reset_mock()
        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, LoadSessionResponse)
        calls = mock_conn.session_update.await_args_list
        replay_calls = [
            call for call in calls
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"user_message_chunk", "agent_message_chunk"}
        ]
        assert len(replay_calls) == 2
        assert isinstance(replay_calls[0].kwargs["update"], UserMessageChunk)
        assert replay_calls[0].kwargs["update"].content.text == "what controls the / slash commands?"
        assert isinstance(replay_calls[1].kwargs["update"], AgentMessageChunk)
        assert replay_calls[1].kwargs["update"].content.text.startswith("HermesACPAgent")

        tool_updates = [
            call.kwargs["update"]
            for call in calls
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"tool_call", "tool_call_update"}
        ]
        assert len(tool_updates) == 2
        assert isinstance(tool_updates[0], ToolCallStart)
        assert tool_updates[0].tool_call_id == "call_search_1"
        assert tool_updates[0].title == "search: slash commands"
        assert isinstance(tool_updates[1], ToolCallProgress)
        assert tool_updates[1].tool_call_id == "call_search_1"
        assert "Search results" in tool_updates[1].content[0].content.text
        assert "cli.py:42" in tool_updates[1].content[0].content.text

    @pytest.mark.asyncio
    async def test_load_session_stamps_history_index_on_user_chunks(self, agent):
        """Replayed user chunks carry _meta.hermes.historyIndex — the absolute
        state.history coordinate a client passes back as keepHistory on fork."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        # Index 0 is a hidden system message replay skips; the user turns sit at
        # absolute indices 1 and 3, which is exactly what must be stamped.
        state.history = [
            {"role": "system", "content": "hidden system"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)

        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None) == "user_message_chunk"
        ]
        assert len(user_calls) == 2
        assert user_calls[0].kwargs["update"].content.text == "first question"
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 1}
        assert user_calls[1].kwargs["update"].content.text == "second question"
        assert user_calls[1].kwargs["hermes"] == {"historyIndex": 3}

        # Assistant chunks are not stamped — only user turns are fork points.
        assistant_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None) == "agent_message_chunk"
        ]
        assert assistant_calls and all("hermes" not in c.kwargs for c in assistant_calls)

    @pytest.mark.asyncio
    async def test_load_session_flags_compaction_summary_on_replayed_user_chunk(self, agent):
        """A replayed compaction summary must carry _meta.hermes.compactionSummary.

        The handoff is stored role="user" but is not a real user turn; without
        the flag on the wire, ACP frontends render the whole summary as a user
        message. Detection falls back to content, so this holds even for a
        DB-reloaded session that lost the in-process metadata flag.
        """
        from agent.context_compressor import SUMMARY_PREFIX

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        summary_text = SUMMARY_PREFIX + "\n\n## Active Task\nDo the thing."
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": summary_text},
            {"role": "user", "content": "wait 5s and reply ok"},
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        user_chunks = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), UserMessageChunk)
        ]
        assert len(user_chunks) == 2
        # First user chunk is the summary → flagged; second is a real turn → not.
        assert user_chunks[0].field_meta == {"hermes": {"compactionSummary": True}}
        assert user_chunks[1].field_meta is None

    @pytest.mark.asyncio
    async def test_load_session_flags_compaction_summary_on_replayed_assistant_chunk(self, agent):
        """The compressor can emit a standalone summary with role="assistant"
        (whichever role keeps alternation valid), so the assistant replay
        branch must flag it too — not just the user branch.
        """
        from agent.context_compressor import SUMMARY_PREFIX

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        summary_text = SUMMARY_PREFIX + "\n\n## Active Task\nDo the thing."
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "assistant", "content": summary_text},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "on it"},
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        agent_chunks = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentMessageChunk)
        ]
        assert len(agent_chunks) == 2
        assert agent_chunks[0].field_meta == {"hermes": {"compactionSummary": True}}
        assert agent_chunks[1].field_meta is None

    @pytest.mark.asyncio
    async def test_load_session_flags_merged_tail_summary_as_contains_not_standalone(self, agent):
        """A merge-into-tail message carries real preserved content plus the
        summary. It must be flagged containsCompactionSummary — NOT
        compactionSummary — so a client that collapses standalone summaries
        cannot hide the preserved turn content.
        """
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            _SUMMARY_END_MARKER,
            SUMMARY_PREFIX,
        )

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        merged_text = (
            _MERGED_PRIOR_CONTEXT_HEADER
            + "\nplease fix the login bug"
            + "\n\n" + _MERGED_SUMMARY_DELIMITER + "\n\n"
            + SUMMARY_PREFIX + "\n\n## Active Task\nFix login."
            + "\n\n" + _SUMMARY_END_MARKER
        )
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": merged_text},
            {"role": "assistant", "content": "looking at it"},
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        user_chunks = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), UserMessageChunk)
        ]
        assert len(user_chunks) == 1
        assert user_chunks[0].field_meta == {
            "hermes": {"containsCompactionSummary": True}
        }

    @pytest.mark.asyncio
    async def test_load_session_stamps_timestamp_on_replayed_chunks(self, agent):
        """Replayed user/assistant chunks carry _meta.hermes.timestamp (ISO
        UTC) when the persisted message has one, so clients can show when
        each message was sent. Messages without a timestamp stay meta-free,
        and the flag composes with the compaction-summary meta."""
        from agent.context_compressor import SUMMARY_PREFIX

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        summary_text = SUMMARY_PREFIX + "\n\n## Active Task\nDo the thing."
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": "first question", "timestamp": 1783000000.5},
            {"role": "assistant", "content": "first answer", "timestamp": 1783000042.0},
            {"role": "user", "content": "no stamp here"},
            {"role": "user", "content": summary_text, "timestamp": 1783000100.0},
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        user_chunks = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), UserMessageChunk)
        ]
        agent_chunks = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentMessageChunk)
        ]
        assert len(user_chunks) == 3
        assert user_chunks[0].field_meta == {
            "hermes": {"timestamp": "2026-07-02T13:46:40.500000+00:00"}
        }
        # No persisted timestamp → no meta at all (existing client shape).
        assert user_chunks[1].field_meta is None
        # Timestamp composes with the compaction-summary flag.
        assert user_chunks[2].field_meta == {
            "hermes": {
                "compactionSummary": True,
                "timestamp": "2026-07-02T13:48:20+00:00",
            }
        }
        assert len(agent_chunks) == 1
        assert agent_chunks[0].field_meta == {
            "hermes": {"timestamp": "2026-07-02T13:47:22+00:00"}
        }

    @pytest.mark.asyncio
    async def test_load_session_replays_native_plan_for_persisted_todo_tool(self, agent):
        """Persisted todo tool results should rebuild Zed's native plan panel."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_todo_1",
                        "type": "function",
                        "function": {
                            "name": "todo",
                            "arguments": '{"todos":[{"id":"ship","content":"Ship it","status":"in_progress"}]}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_todo_1",
                "content": '{"todos":[{"id":"ship","content":"Ship it","status":"in_progress"}]}',
            },
        ]

        mock_conn.session_update.reset_mock()
        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, LoadSessionResponse)
        relevant_updates = [
            update for update in (call.kwargs["update"] for call in mock_conn.session_update.await_args_list)
            if getattr(update, "session_update", None) in {"tool_call", "tool_call_update", "plan"}
        ]
        assert [getattr(update, "session_update", None) for update in relevant_updates] == [
            "tool_call",
            "tool_call_update",
            "plan",
        ]
        plan = relevant_updates[2]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == ["Ship it"]
        assert [entry.status for entry in plan.entries] == ["in_progress"]

    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )

    @pytest.mark.asyncio
    async def test_load_session_replays_reasoning_thought_before_message(self, agent):
        """Thinking-model thoughts must be replayed via ``agent_thought_chunk``.

        Regression for #12285 — when a session is loaded, persisted assistant
        ``reasoning_content`` / ``reasoning`` fields must surface as ACP
        ``AgentThoughtChunk`` notifications in the same relative position they
        had live (thought streams before the assistant message text), so Zed's
        collapsed Thinking pane rebuilds instead of vanishing on reconnect.
        """
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": "Walk me through it."},
            {
                "role": "assistant",
                "reasoning_content": "Let me think step by step about the request.",
                "content": "Here is the plan.",
            },
            {"role": "user", "content": "And the legacy case?"},
            {
                "role": "assistant",
                # No reasoning_content — exercise the legacy "reasoning" fallback
                # path so sessions persisted before #16892 still replay thoughts.
                "reasoning": "Older sessions stored the trace under the internal key.",
                "content": "Same idea, older field name.",
            },
        ]

        mock_conn.session_update.reset_mock()
        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, LoadSessionResponse)

        replay_kinds = [
            getattr(call.kwargs.get("update"), "session_update", None)
            for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"user_message_chunk", "agent_message_chunk", "agent_thought_chunk"}
        ]
        assert replay_kinds == [
            "user_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
            "user_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
        ]

        thought_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentThoughtChunk)
        ]
        assert len(thought_updates) == 2
        assert thought_updates[0].content.text == "Let me think step by step about the request."
        assert thought_updates[1].content.text == "Older sessions stored the trace under the internal key."

    @pytest.mark.asyncio
    async def test_load_session_replays_reasoning_only_turn(self, agent):
        """Assistant turns with reasoning but no content should still emit a thought.

        Pure reasoning-only assistant entries (e.g. a thinking step before a
        tool-call turn) commonly carry ``reasoning_content`` with empty
        ``content``. The replay must still surface the thought so the editor's
        Thinking pane rebuilds, even when there is no message text to follow.
        """
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {
                "role": "assistant",
                "reasoning_content": "I should call the search tool next.",
                "content": "",
            },
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        thought_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentThoughtChunk)
        ]
        message_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentMessageChunk)
        ]
        assert len(thought_updates) == 1
        assert thought_updates[0].content.text == "I should call the search tool next."
        assert message_updates == []

    @pytest.mark.asyncio
    async def test_load_session_skips_empty_reasoning_fields(self, agent):
        """Empty/whitespace reasoning fields must not produce notifications."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {
                "role": "assistant",
                "reasoning_content": "",
                "reasoning": "   \n\t",
                "content": "Just a regular answer.",
            },
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        thought_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if isinstance(call.kwargs.get("update"), AgentThoughtChunk)
        ]
        assert thought_updates == []

    @pytest.mark.asyncio
    async def test_load_session_replays_thought_then_tool_call_without_message(self, agent):
        """Canonical thinking-model shape: reasoning + tool_call + no body text.

        Thinking models commonly emit a pre-tool thought followed by a
        tool_calls turn with empty ``content``. Replay must emit:
        ``agent_thought_chunk`` then ``tool_call`` then ``tool_call_update``
        for the matching tool result — and crucially, NO ``agent_message_chunk``
        for the empty-text assistant body. Regression for the canonical
        thinking-then-tool flow on #12285.
        """
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": "Find the bug."},
            {
                "role": "assistant",
                "reasoning_content": "I should grep for the function name first.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_grep_1",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"pattern":"foo","path":"."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_grep_1",
                "content": '{"total_count":1,"matches":[{"path":"x.py","line":1,"content":"foo"}]}',
            },
        ]

        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        kinds = [
            getattr(call.kwargs.get("update"), "session_update", None)
            for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {
                "user_message_chunk",
                "agent_thought_chunk",
                "agent_message_chunk",
                "tool_call",
                "tool_call_update",
            }
        ]
        # No agent_message_chunk for the empty-content assistant turn.
        assert "agent_message_chunk" not in kinds
        # Thought must precede the tool_call_start within the assistant turn,
        # and the tool result follows.
        assert kinds == [
            "user_message_chunk",
            "agent_thought_chunk",
            "tool_call",
            "tool_call_update",
        ]

    @pytest.mark.asyncio
    async def test_load_session_replays_history_before_returning_response(self, agent):
        """Per ACP spec, replay must complete BEFORE load_session returns.

        Spec-compliant ACP clients (Codex, Claude Code, OpenCode, Pi, Zed)
        attach their ``session/update`` listeners before awaiting the
        ``loadSession`` RPC and rely on receiving the full transcript within
        the request's lifetime. Deferring replay via ``loop.call_soon`` (the
        prior behavior in May 2026) broke clients that read notification
        counts synchronously against the load response — see #12285 follow-up.
        """
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "hello from history"}]
        events: list[str] = []

        async def replay_records(_state):
            events.append("replay")

        with patch.object(agent, "_replay_session_history", side_effect=replay_records):
            resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
            events.append("returned")

        assert isinstance(resp, LoadSessionResponse)
        # Replay must have happened BEFORE the response was constructed —
        # i.e. before the `events.append("returned")` after the await resolves.
        assert events == ["replay", "returned"]

    @pytest.mark.asyncio
    async def test_resume_session_replays_history_before_returning_response(self, agent):
        """Same spec rationale as ``load_session`` — replay before responding."""
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "hello from history"}]
        events: list[str] = []

        async def replay_records(_state):
            events.append("replay")

        with patch.object(agent, "_replay_session_history", side_effect=replay_records):
            resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
            events.append("returned")

        assert isinstance(resp, ResumeSessionResponse)
        assert events == ["replay", "returned"]

    @pytest.mark.asyncio
    async def test_load_session_survives_replay_helper_exception(self, agent, caplog):
        """A replay helper raising must not turn load_session into an error.

        With awaited replay, an exception in ``_replay_session_history`` now
        propagates into the ``load_session`` handler. The defensive try/except
        guard at the call site must catch and log it so the JSON-RPC client
        still receives a ``LoadSessionResponse`` — partial transcripts are
        acceptable, total load failure is not.
        """
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "hi"}]

        async def boom(_state):
            raise RuntimeError("simulated replay helper crash")

        with caplog.at_level("WARNING", logger="acp_adapter.server"):
            with patch.object(agent, "_replay_session_history", side_effect=boom):
                resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, LoadSessionResponse)
        assert "history replay raised during session/load" in caplog.text

    @pytest.mark.asyncio
    async def test_resume_session_survives_replay_helper_exception(self, agent, caplog):
        """Same guarantee as ``load_session`` for the resume path."""
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "hi"}]

        async def boom(_state):
            raise RuntimeError("simulated replay helper crash")

        with caplog.at_level("WARNING", logger="acp_adapter.server"):
            with patch.object(agent, "_replay_session_history", side_effect=boom):
                resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, ResumeSessionResponse)
        assert "history replay raised during session/resume" in caplog.text

    @pytest.mark.asyncio
    async def test_resume_session_creates_new_if_missing(self, agent):
        resume_resp = await agent.resume_session(cwd="/tmp", session_id="nonexistent")
        assert isinstance(resume_resp, ResumeSessionResponse)

    # ---- mid-turn replay source selection ---------------------------------
    #
    # While a turn is RUNNING, state.history only extends at turn end but the
    # SQLite store is flushed continuously — so replay must read the persisted
    # transcript (via SessionManager.live_transcript_history) instead of the
    # stale in-memory list, and must fail open to state.history whenever the
    # DB path is unavailable or suspiciously short.

    async def _running_replay_state(self, agent, history=None):
        """Create a session, wire a fresh mock conn, and return its state."""
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = history if history is not None else [
            {"role": "user", "content": "kick off the long job"},
        ]
        mock_conn.session_update.reset_mock()
        return state, mock_conn

    @staticmethod
    def _db_transcript():
        """Kickoff + in-flight assistant tool call + tool result rows."""
        return [
            {"role": "user", "content": "kick off the long job"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_live_1",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"pattern":"midturn","path":"."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_live_1",
                "content": '{"total_count":1,"matches":[{"path":"a.py","line":1,"content":"midturn"}]}',
            },
        ]

    @pytest.mark.asyncio
    async def test_load_session_running_turn_replays_db_transcript(self, agent):
        """Running turn: replay reads the persisted transcript, not the stale
        in-memory history — proven by the tool-call updates only the DB has."""
        state, mock_conn = await self._running_replay_state(agent)
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=self._db_transcript()
        )

        await agent._replay_session_history(state)

        tool_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"tool_call", "tool_call_update"}
        ]
        assert len(tool_updates) == 2
        assert isinstance(tool_updates[0], ToolCallStart)
        assert tool_updates[0].tool_call_id == "call_live_1"
        assert isinstance(tool_updates[1], ToolCallProgress)
        assert tool_updates[1].tool_call_id == "call_live_1"
        agent.session_manager.live_transcript_history.assert_called_once_with(
            state.session_id
        )

    @pytest.mark.asyncio
    async def test_running_turn_replay_omits_history_index_meta(self, agent):
        """DB rows have no stable fork coordinates (compaction may rewrite
        them before the turn finalizes), so DB-sourced replay must not stamp
        _meta.hermes.historyIndex on any chunk."""
        state, mock_conn = await self._running_replay_state(agent)
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=self._db_transcript()
        )

        await agent._replay_session_history(state)

        calls = mock_conn.session_update.await_args_list
        assert calls  # replay emitted something
        assert all("hermes" not in call.kwargs for call in calls)

    @pytest.mark.asyncio
    async def test_running_turn_replay_with_empty_memory_history(self, agent):
        """A running first turn: state.history is still empty but the DB
        already holds rows — replay must emit from the DB transcript."""
        state, mock_conn = await self._running_replay_state(agent, history=[])
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=self._db_transcript()
        )

        await agent._replay_session_history(state)

        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert len(user_calls) == 1
        assert user_calls[0].kwargs["update"].content.text == "kick off the long job"

    @pytest.mark.asyncio
    async def test_running_turn_replay_falls_back_when_db_unavailable(self, agent):
        """live_transcript_history returning None → fail open to the memory
        path, unchanged: replay state.history WITH historyIndex meta."""
        state, mock_conn = await self._running_replay_state(agent)
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(return_value=None)

        await agent._replay_session_history(state)

        agent.session_manager.live_transcript_history.assert_called_once_with(
            state.session_id
        )
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert len(user_calls) == 1
        assert user_calls[0].kwargs["update"].content.text == "kick off the long job"
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 0}

    @pytest.mark.asyncio
    async def test_running_turn_replay_falls_back_when_db_shorter(self, agent):
        """A transcript shorter than state.history means the DB head resolved
        to the wrong lineage — distrust it and replay memory with meta."""
        state, mock_conn = await self._running_replay_state(
            agent,
            history=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        )
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=[{"role": "user", "content": "first question"}]
        )

        await agent._replay_session_history(state)

        agent.session_manager.live_transcript_history.assert_called_once_with(
            state.session_id
        )
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert [c.kwargs["update"].content.text for c in user_calls] == [
            "first question",
            "second question",
        ]
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 0}
        assert user_calls[1].kwargs["hermes"] == {"historyIndex": 2}

    @pytest.mark.asyncio
    async def test_running_turn_replay_uses_db_at_equal_length(self, agent):
        """Equal lengths: the DB is still the faithful mid-turn transcript
        (the gate is >=, not >) — replay the DB rows, without index meta."""
        state, mock_conn = await self._running_replay_state(
            agent,
            history=[
                {"role": "user", "content": "kick off the long job"},
                {"role": "assistant", "content": "stale memory snapshot"},
            ],
        )
        state.is_running = True
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=[
                {"role": "user", "content": "kick off the long job"},
                {"role": "assistant", "content": "fresh db snapshot"},
            ]
        )

        await agent._replay_session_history(state)

        agent.session_manager.live_transcript_history.assert_called_once_with(
            state.session_id
        )
        agent_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "agent_message_chunk"
        ]
        assert [c.kwargs["update"].content.text for c in agent_calls] == [
            "fresh db snapshot"
        ]
        assert all(
            "hermes" not in call.kwargs
            for call in mock_conn.session_update.await_args_list
        )

    @pytest.mark.asyncio
    async def test_idle_replay_adopts_longer_db_transcript(self, agent):
        """Idle here, but the persisted store is AHEAD: the session advanced
        in another process (Slack bot, gateway, CLI). Replay must use the DB
        rows and adopt them as state.history so the next prompt builds on
        the newer transcript (2026-07-22 incident: a Slack-owned session
        opened in VS Code froze at a pre-final-answer snapshot forever)."""
        state, mock_conn = await self._running_replay_state(agent)
        state.is_running = False
        db_rows = self._db_transcript()
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=db_rows
        )

        await agent._replay_session_history(state)

        agent.session_manager.live_transcript_history.assert_called_once_with(
            state.session_id, repair_alternation=True
        )
        tool_updates = [
            call.kwargs["update"]
            for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            in {"tool_call", "tool_call_update"}
        ]
        assert len(tool_updates) == 2
        # Unlike the mid-turn DB path, the adopted list IS state.history now,
        # so historyIndex fork coordinates stay valid and keep flowing.
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 0}
        assert state.history == db_rows

    @pytest.mark.asyncio
    async def test_idle_replay_keeps_memory_at_equal_length(self, agent):
        """Idle with DB and memory in sync: stay on the in-memory path so
        fork coordinates (historyIndex meta) keep flowing."""
        state, mock_conn = await self._running_replay_state(
            agent,
            history=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
            ],
        )
        state.is_running = False
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=[
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "db copy of the answer"},
            ]
        )

        await agent._replay_session_history(state)

        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert len(user_calls) == 1
        assert user_calls[0].kwargs["update"].content.text == "first question"
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 0}
        agent_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "agent_message_chunk"
        ]
        assert [c.kwargs["update"].content.text for c in agent_calls] == [
            "first answer"
        ]

    @pytest.mark.asyncio
    async def test_idle_replay_keeps_memory_when_db_unavailable(self, agent):
        """Idle with the store unavailable: fail open to the in-memory path,
        with index meta, and leave state.history untouched."""
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        state, mock_conn = await self._running_replay_state(
            agent, history=list(history)
        )
        state.is_running = False
        agent.session_manager.live_transcript_history = MagicMock(return_value=None)

        await agent._replay_session_history(state)

        assert state.history == history
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert len(user_calls) == 1
        assert user_calls[0].kwargs["hermes"] == {"historyIndex": 0}

    @pytest.mark.asyncio
    async def test_idle_replay_keeps_memory_when_db_shorter(self, agent):
        """Idle with a shorter DB transcript (wrong lineage or compacted
        elsewhere): distrust it — replay memory, don't adopt."""
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        state, mock_conn = await self._running_replay_state(
            agent, history=list(history)
        )
        state.is_running = False
        agent.session_manager.live_transcript_history = MagicMock(
            return_value=[{"role": "user", "content": "first question"}]
        )

        await agent._replay_session_history(state)

        assert state.history == history
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "user_message_chunk"
        ]
        assert [c.kwargs["update"].content.text for c in user_calls] == [
            "first question",
            "second question",
        ]
        assert user_calls[1].kwargs["hermes"] == {"historyIndex": 2}


# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_fork_session_keep_history_meta_slices_prefix(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history.extend(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "reply 2"},
            ]
        )

        # The JSON-RPC router flattens _meta into handler kwargs, so
        # {"_meta": {"hermes": {"keepHistory": 2}}} arrives as hermes={...}.
        fork_resp = await agent.fork_session(
            cwd="/forked",
            session_id=new_resp.session_id,
            hermes={"keepHistory": 2},
        )

        forked = agent.session_manager.get_session(fork_resp.session_id)
        assert len(forked.history) == 2
        assert forked.history[1]["content"] == "reply"
        assert len(state.history) == 4

    @pytest.mark.asyncio
    async def test_fork_session_keep_history_meta_invalid_raises(self, agent):
        new_resp = await agent.new_session(cwd="/original")

        for bad in ("2", -1, True, 1.5):
            with pytest.raises(acp.RequestError):
                await agent.fork_session(
                    cwd="/forked",
                    session_id=new_resp.session_id,
                    hermes={"keepHistory": bad},
                )

    @pytest.mark.asyncio
    async def test_fork_session_router_delivers_keep_history_meta(self, agent):
        """End-to-end through the JSON-RPC router: _meta survives the wire shape."""
        new_resp = await agent.new_session(cwd="/original")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history.extend(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        )
        router = build_agent_router(agent, use_unstable_protocol=True)

        result = await router(
            "session/fork",
            {
                "cwd": "/forked",
                "sessionId": new_resp.session_id,
                "_meta": {"hermes": {"keepHistory": 1}},
            },
            False,
        )

        forked = agent.session_manager.get_session(result.session_id)
        assert len(forked.history) == 1
        assert forked.history[0]["content"] == "first"

    @pytest.mark.asyncio
    async def test_initialize_advertises_fork_keep_history_extension(self, agent):
        resp = await agent.initialize(protocol_version=1)
        fork_caps = resp.agent_capabilities.session_capabilities.fork
        assert fork_caps.field_meta == {"hermes": {"keepHistory": True}}

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"

    @pytest.mark.asyncio
    async def test_list_sessions_passes_cwd_filter(self, agent):
        with patch.object(agent.session_manager, "list_sessions", return_value=[]) as mock_list:
            await agent.list_sessions(cwd="/mnt/e/Projects/AI/browser-link-3")

        mock_list.assert_called_once_with(
            cwd="/mnt/e/Projects/AI/browser-link-3",
            include_archived=False,
            archived_only=False,
            owner=None,
        )

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_first_page(self, agent):
        from acp_adapter import server as acp_server

        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(acp_server._LIST_SESSIONS_PAGE_SIZE + 5)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == acp_server._LIST_SESSIONS_PAGE_SIZE
        assert resp.next_cursor == resp.sessions[-1].session_id

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_no_more(self, agent):
        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(3)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == 3
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_cursor_resumes_after_match(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s3", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="s1")

        assert [s.session_id for s in resp.sessions] == ["s2", "s3"]
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_unknown_cursor_returns_empty(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="does-not-exist")

        assert resp.sessions == []
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_pagination_keeps_fork_family_on_one_page(self, agent):
        from acp_adapter import server as acp_server

        # A parent with enough children to straddle the flat page boundary,
        # then plenty of unrelated sessions. A naive flat cut would strand
        # some children on page 2, splitting the family in nesting clients.
        page = acp_server._LIST_SESSIONS_PAGE_SIZE
        family = [{"session_id": "root", "cwd": "/tmp", "title": None, "updated_at": 0.0}]
        family += [
            {"session_id": f"child{i}", "cwd": "/tmp", "title": None,
             "updated_at": 0.0, "parent_id": "root"}
            for i in range(10)
        ]
        fillers = [
            {"session_id": f"solo{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(page)
        ]
        # Family members occupy positions page-5 .. page+5 of the flat list.
        infos = fillers[: page - 5] + family + fillers[page - 5:]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        ids = [s.session_id for s in resp.sessions]
        family_ids = {m["session_id"] for m in family}
        on_page = family_ids & set(ids)
        # The family is atomic: all members on this page or none.
        assert on_page in (family_ids, set())
        assert resp.next_cursor == ids[-1]

        # The next page starts with whatever was deferred — no family member
        # appears twice and none is lost across the two pages.
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp2 = await agent.list_sessions(cursor=resp.next_cursor)
        ids2 = [s.session_id for s in resp2.sessions]
        assert set(ids) & set(ids2) == set()
        assert family_ids <= set(ids) | set(ids2)
        assert family_ids & set(ids2) in (family_ids, set())

    @pytest.mark.asyncio
    async def test_list_sessions_page_counts_families_not_rows(self, agent):
        from acp_adapter import server as acp_server

        # The page cap counts grouped units: a family with many children is ONE
        # unit, so a page still carries _LIST_SESSIONS_PAGE_SIZE groups even
        # when one group alone exceeds the cap in raw rows.
        page = acp_server._LIST_SESSIONS_PAGE_SIZE
        family = [{"session_id": "root", "cwd": "/tmp", "title": None, "updated_at": 0.0}]
        family += [
            {"session_id": f"child{i}", "cwd": "/tmp", "title": None,
             "updated_at": 0.0, "parent_id": "root"}
            for i in range(page + 5)
        ]
        solos = [
            {"session_id": f"solo{i}", "cwd": "/tmp", "title": None, "updated_at": 0.0}
            for i in range(page)
        ]
        infos = family + solos
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions()

        ids = [s.session_id for s in resp.sessions]
        # Whole family (page+6 rows) + (page-1) solos = page groups on page 1.
        assert {m["session_id"] for m in family} <= set(ids)
        assert len(ids) == (page + 6) + (page - 1)
        assert resp.next_cursor == ids[-1]

        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp2 = await agent.list_sessions(cursor=resp.next_cursor)
        assert [s.session_id for s in resp2.sessions] == [f"solo{page - 1}"]
        assert resp2.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_mid_family_cursor_skips_whole_family(self, agent):
        infos = [
            {"session_id": "root", "cwd": "/tmp", "title": None, "updated_at": 0.0},
            {"session_id": "child1", "cwd": "/tmp", "title": None,
             "updated_at": 0.0, "parent_id": "root"},
            {"session_id": "child2", "cwd": "/tmp", "title": None,
             "updated_at": 0.0, "parent_id": "root"},
            {"session_id": "solo", "cwd": "/tmp", "title": None, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos):
            resp = await agent.list_sessions(cursor="child1")

        # Resuming inside a family (older-server cursor) skips that whole
        # family rather than re-emitting part of it detached from its root.
        assert [s.session_id for s in resp.sessions] == ["solo"]
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_list_sessions_backfills_untitled_rows(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": "prompt preview", "untitled": True,
             "updated_at": 0.0},
            {"session_id": "s2", "cwd": "/tmp", "title": "Real Title", "untitled": False,
             "updated_at": 0.0},
            {"session_id": "s3", "cwd": "/tmp", "title": "child preview", "untitled": True,
             "subagent": True, "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos), \
             patch.object(
                 agent.session_manager, "derive_session_title", return_value="Derived"
             ) as mock_derive, \
             patch.object(agent, "_send_session_info_update") as mock_emit:
            await agent.list_sessions()
            await asyncio.sleep(0.05)  # let the backfill tasks run

        # Only the untitled non-subagent row is derived; success emits an update.
        mock_derive.assert_called_once_with("s1")
        mock_emit.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_list_sessions_backfill_attempts_once_per_session(self, agent):
        infos = [
            {"session_id": "s1", "cwd": "/tmp", "title": "preview", "untitled": True,
             "updated_at": 0.0},
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos), \
             patch.object(
                 agent.session_manager, "derive_session_title", return_value=None
             ) as mock_derive:
            await agent.list_sessions()
            await asyncio.sleep(0.05)
            await agent.list_sessions()
            await asyncio.sleep(0.05)

        # Second list must not retry the failed derive.
        mock_derive.assert_called_once_with("s1")

    @pytest.mark.asyncio
    async def test_list_sessions_backfill_capped_per_call(self, agent):
        infos = [
            {"session_id": f"s{i}", "cwd": "/tmp", "title": f"p{i}", "untitled": True,
             "updated_at": 0.0}
            for i in range(10)
        ]
        with patch.object(agent.session_manager, "list_sessions", return_value=infos), \
             patch.object(
                 agent.session_manager, "derive_session_title", return_value=None
             ) as mock_derive:
            await agent.list_sessions()
            await asyncio.sleep(0.05)

        assert mock_derive.call_count == agent._TITLE_BACKFILL_MAX_PER_LIST

    @pytest.mark.asyncio
    async def test_ext_method_set_archived_delegates_and_returns_ok(self, agent):
        with patch.object(
            agent.session_manager, "set_session_archived", return_value=True
        ) as mock_set:
            result = await agent.ext_method("setArchived", {"sessionId": "s1", "archived": True})
        assert result == {"ok": True}
        mock_set.assert_called_once_with("s1", True)

    @pytest.mark.asyncio
    async def test_ext_method_set_title_delegates_persists_and_emits(self, agent):
        with patch.object(
            agent.session_manager, "set_session_title", return_value=True
        ) as mock_set, patch.object(
            agent.session_manager, "get_session_title", return_value="My Title"
        ), patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method("setTitle", {"sessionId": "s1", "title": "My Title"})
        assert result == {"ok": True, "title": "My Title"}
        mock_set.assert_called_once_with("s1", "My Title")
        mock_emit.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_ext_method_set_title_conflict_returns_error_no_emit(self, agent):
        with patch.object(
            agent.session_manager,
            "set_session_title",
            side_effect=ValueError("Title 'X' is already in use by session s2"),
        ), patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method("setTitle", {"sessionId": "s1", "title": "X"})
        assert result["ok"] is False
        assert "already in use" in result["error"]
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_ext_method_set_title_requires_session_id(self, agent):
        result = await agent.ext_method("setTitle", {"title": "X"})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_ext_method_derive_title_success_persists_and_emits(self, agent):
        with patch.object(
            agent.session_manager, "derive_session_title", return_value="Derived Title"
        ) as mock_derive, patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method("deriveTitle", {"sessionId": "s1"})
        assert result == {"ok": True, "title": "Derived Title"}
        mock_derive.assert_called_once_with("s1")
        mock_emit.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_ext_method_derive_title_noop_returns_false_no_emit(self, agent):
        # Already titled / no exchange yet / aux failure → derive returns None.
        with patch.object(
            agent.session_manager, "derive_session_title", return_value=None
        ), patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method("deriveTitle", {"sessionId": "s1"})
        assert result == {"ok": False, "title": None}
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_ext_method_refresh_models_returns_fresh_payload(self, agent):
        from acp.schema import ModelInfo as AcpModelInfo, SessionModelState

        fresh = SessionModelState(
            available_models=[
                AcpModelInfo(
                    model_id="openai-codex:gpt-5.6-sol",
                    name="gpt-5.6-sol",
                    description="Provider: OpenAI Codex",
                )
            ],
            current_model_id="bedrock:global.anthropic.claude-fable-5",
        )
        state = SimpleNamespace(agent=MagicMock(), model="global.anthropic.claude-fable-5")
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ) as mock_get, patch.object(
            agent, "_build_model_state", return_value=fresh
        ) as mock_build:
            result = await agent.ext_method("refreshModels", {"sessionId": "s1"})
        assert result["ok"] is True
        assert result["models"]["currentModelId"] == "bedrock:global.anthropic.claude-fable-5"
        assert result["models"]["availableModels"] == [
            {
                "modelId": "openai-codex:gpt-5.6-sol",
                "name": "gpt-5.6-sol",
                "description": "Provider: OpenAI Codex",
            }
        ]
        mock_get.assert_called_once_with("s1")
        mock_build.assert_called_once_with(state, force_refresh=True)

    @pytest.mark.asyncio
    async def test_ext_method_refresh_models_unknown_session(self, agent):
        with patch.object(agent.session_manager, "get_session", return_value=None):
            result = await agent.ext_method("refreshModels", {"sessionId": "nope"})
        assert result["ok"] is False
        assert "not loaded" in result["error"]

    @pytest.mark.asyncio
    async def test_ext_method_refresh_models_requires_session_id(self, agent):
        result = await agent.ext_method("refreshModels", {})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_ext_method_refresh_models_none_state_is_ok_null(self, agent):
        state = SimpleNamespace(agent=MagicMock(), model="")
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch.object(agent, "_build_model_state", return_value=None):
            result = await agent.ext_method("refreshModels", {"sessionId": "s1"})
        assert result == {"ok": True, "models": None}

    @pytest.mark.asyncio
    async def test_ext_method_refresh_models_build_failure_returns_error(self, agent):
        state = SimpleNamespace(agent=MagicMock(), model="")
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch.object(
            agent, "_build_model_state", side_effect=RuntimeError("catalog exploded")
        ):
            result = await agent.ext_method("refreshModels", {"sessionId": "s1"})
        assert result["ok"] is False
        assert "catalog exploded" in result["error"]

    @pytest.mark.asyncio
    async def test_ext_method_unknown_raises_method_not_found(self, agent):
        from acp import RequestError
        with pytest.raises(RequestError):
            await agent.ext_method("nope", {})

    @pytest.mark.asyncio
    async def test_ext_method_context_breakdown_returns_payload(self, agent):
        payload = {
            "categories": [
                {"color": "var(--context-usage-system)", "id": "system_prompt", "label": "System prompt", "tokens": 3200},
                {"color": "var(--context-usage-conversation)", "id": "conversation", "label": "Conversation", "tokens": 1500},
            ],
            "context_max": 200000,
            "context_percent": 2,
            "context_used": 4700,
            "estimated_total": 4700,
            "model": "test-model",
        }
        state = SimpleNamespace(agent=MagicMock(), history=[{"role": "user", "content": "hi"}])
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ) as mock_get, patch(
            "agent.context_breakdown.compute_session_context_breakdown",
            return_value=payload,
        ) as mock_compute:
            result = await agent.ext_method("contextBreakdown", {"sessionId": "s1"})
        assert result == {"ok": True, "breakdown": payload}
        mock_get.assert_called_once_with("s1")
        mock_compute.assert_called_once_with(state.agent, state.history)

    @pytest.mark.asyncio
    async def test_ext_method_context_breakdown_unknown_session(self, agent):
        with patch.object(agent.session_manager, "get_session", return_value=None):
            result = await agent.ext_method("contextBreakdown", {"sessionId": "nope"})
        assert result["ok"] is False
        assert "not loaded" in result["error"]

    @pytest.mark.asyncio
    async def test_ext_method_context_breakdown_requires_session_id(self, agent):
        result = await agent.ext_method("contextBreakdown", {})
        assert result == {"ok": False, "error": "sessionId required"}

    @pytest.mark.asyncio
    async def test_ext_method_context_report_returns_markdown(self, agent):
        state = SimpleNamespace(agent=MagicMock(), history=[{"role": "user", "content": "hi"}])
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ) as mock_get, patch(
            "agent.context_breakdown.build_session_context_report",
            return_value="# Context report\n\n## System prompt\n\ntext\n",
        ) as mock_build:
            result = await agent.ext_method(
                "contextReport", {"sessionId": "s1", "category": "system_prompt"}
            )
        assert result["ok"] is True
        assert result["report"].startswith("# Context report")
        mock_get.assert_called_once_with("s1")
        mock_build.assert_called_once_with(
            state.agent, state.history, category="system_prompt"
        )

    @pytest.mark.asyncio
    async def test_ext_method_context_report_unknown_category(self, agent):
        state = SimpleNamespace(agent=MagicMock(), history=[])
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch(
            "agent.context_breakdown.build_session_context_report",
            side_effect=ValueError("unknown context category: 'nope'"),
        ):
            result = await agent.ext_method(
                "contextReport", {"sessionId": "s1", "category": "nope"}
            )
        assert result["ok"] is False
        assert "nope" in result["error"]

    @pytest.mark.asyncio
    async def test_ext_method_context_report_requires_session_id(self, agent):
        result = await agent.ext_method("contextReport", {})
        assert result == {"ok": False, "error": "sessionId required"}

    @pytest.mark.asyncio
    async def test_ext_method_account_usage_returns_windows(self, agent):
        from datetime import datetime, timezone

        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

        reset = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
        snapshot = AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc),
            plan="Pro",
            windows=(
                AccountUsageWindow(label="Session", used_percent=37.0, reset_at=reset),
                AccountUsageWindow(label="Weekly", used_percent=62.5, reset_at=None),
            ),
            details=("Credits balance: $12.00",),
        )
        state = SimpleNamespace(agent=SimpleNamespace(provider="openai-codex", base_url=None, api_key=None))
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch(
            "agent.account_usage.fetch_account_usage", return_value=snapshot
        ) as mock_fetch:
            result = await agent.ext_method("accountUsage", {"sessionId": "s1"})
        assert result["ok"] is True
        usage = result["usage"]
        assert usage["provider"] == "openai-codex"
        assert usage["plan"] == "Pro"
        assert usage["windows"][0] == {
            "label": "Session",
            "usedPercent": 37.0,
            "resetAt": reset.isoformat(),
            "detail": None,
        }
        assert usage["windows"][1]["resetAt"] is None
        assert usage["details"] == ["Credits balance: $12.00"]
        args, kwargs = mock_fetch.call_args
        assert args == ("openai-codex",)

    @pytest.mark.asyncio
    async def test_ext_method_account_usage_none_snapshot_returns_null(self, agent):
        state = SimpleNamespace(agent=SimpleNamespace(provider="bedrock", base_url=None, api_key=None))
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch(
            "agent.account_usage.fetch_account_usage", return_value=None
        ):
            result = await agent.ext_method("accountUsage", {"sessionId": "s1"})
        assert result == {"ok": True, "usage": None}

    @pytest.mark.asyncio
    async def test_ext_method_account_usage_no_provider_skips_fetch(self, agent):
        state = SimpleNamespace(agent=SimpleNamespace(provider=None, base_url=None, api_key=None))
        with patch.object(
            agent.session_manager, "get_session", return_value=state
        ), patch(
            "agent.account_usage.fetch_account_usage"
        ) as mock_fetch:
            result = await agent.ext_method("accountUsage", {"sessionId": "s1"})
        assert result == {"ok": True, "usage": None}
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_ext_method_account_usage_unknown_session(self, agent):
        with patch.object(agent.session_manager, "get_session", return_value=None):
            result = await agent.ext_method("accountUsage", {"sessionId": "nope"})
        assert result["ok"] is False
        assert "not loaded" in result["error"]

    @pytest.mark.asyncio
    async def test_ext_method_account_usage_requires_session_id(self, agent):
        result = await agent.ext_method("accountUsage", {})
        assert result == {"ok": False, "error": "sessionId required"}

    @pytest.mark.asyncio
    async def test_list_sessions_archived_only_forwards_and_stamps_meta(self, agent):
        with patch.object(
            agent.session_manager, "list_sessions",
            return_value=[{
                "session_id": "s1", "cwd": "/tmp", "title": "T",
                "updated_at": 1.0, "archived": True,
            }],
        ) as mock_list:
            resp = await agent.list_sessions(cwd="/tmp", hermes={"archivedOnly": True})
        _, kwargs = mock_list.call_args
        assert kwargs.get("archived_only") is True
        assert kwargs.get("include_archived") is False
        s = resp.sessions[0]
        assert (s.field_meta or {}).get("hermes", {}).get("archived") is True

    @pytest.mark.asyncio
    async def test_list_sessions_stamps_fork_lineage_meta(self, agent):
        with patch.object(
            agent.session_manager, "list_sessions",
            return_value=[
                {"session_id": "fork1", "cwd": "/tmp", "title": "Fork",
                 "updated_at": 2.0, "parent_id": "root1"},
                {"session_id": "root1", "cwd": "/tmp", "title": "Root",
                 "updated_at": 1.0, "parent_id": None},
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp")
        fork, root = resp.sessions
        assert (fork.field_meta or {}).get("hermes", {}).get("forkedFrom") == "root1"
        assert root.field_meta is None

    @pytest.mark.asyncio
    async def test_list_sessions_stamps_subagent_meta(self, agent):
        """Delegate children carry isSubagent + parent linkage so clients can
        nest them and render the read-only composer."""
        with patch.object(
            agent.session_manager, "list_sessions",
            return_value=[
                {"session_id": "child1", "cwd": "/tmp", "title": "Child goal",
                 "updated_at": 2.0, "parent_id": "parent1", "subagent": True},
                {"session_id": "parent1", "cwd": "/tmp", "title": "Parent",
                 "updated_at": 1.0, "parent_id": None},
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp")
        child, parent = resp.sessions
        child_meta = (child.field_meta or {}).get("hermes", {})
        assert child_meta.get("isSubagent") is True
        assert child_meta.get("forkedFrom") == "parent1"
        assert parent.field_meta is None

    @pytest.mark.asyncio
    async def test_list_sessions_stamps_owner_meta(self, agent):
        with patch.object(
            agent.session_manager, "list_sessions",
            return_value=[
                {"session_id": "s1", "cwd": "/tmp", "title": "Mine",
                 "updated_at": 2.0, "user_id": "israel@yallaplay.com"},
                {"session_id": "s2", "cwd": "/tmp", "title": "Untagged",
                 "updated_at": 1.0, "user_id": ""},
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp")
        owned, untagged = resp.sessions
        assert (owned.field_meta or {}).get("hermes", {}).get("owner") == "israel@yallaplay.com"
        assert untagged.field_meta is None

    @pytest.mark.asyncio
    async def test_ext_method_set_owner_delegates_persists_and_emits(self, agent):
        with patch.object(
            agent.session_manager, "set_session_owner", return_value=True
        ) as mock_set, patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method(
                "setOwner", {"sessionId": "s1", "owner": "user@yallaplay.com"}
            )
        assert result == {"ok": True}
        mock_set.assert_called_once_with("s1", "user@yallaplay.com")
        mock_emit.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_ext_method_set_owner_noop_returns_false_no_emit(self, agent):
        # Unknown session id → set returns False → no info update emitted.
        with patch.object(
            agent.session_manager, "set_session_owner", return_value=False
        ), patch.object(agent, "_send_session_info_update") as mock_emit:
            result = await agent.ext_method("setOwner", {"sessionId": "s1", "owner": ""})
        assert result == {"ok": False}
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_ext_method_set_owner_requires_session_id(self, agent):
        result = await agent.ext_method("setOwner", {"owner": "x@y.com"})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_list_sessions_owner_only_forwards_owner(self, agent):
        with patch.object(
            agent.session_manager, "list_sessions",
            return_value=[{
                "session_id": "s1", "cwd": "/tmp", "title": "T",
                "updated_at": 1.0, "user_id": "me@yallaplay.com",
            }],
        ) as mock_list:
            resp = await agent.list_sessions(
                cwd="/tmp", hermes={"ownerOnly": True, "owner": "me@yallaplay.com"}
            )
        _, kwargs = mock_list.call_args
        assert kwargs.get("owner") == "me@yallaplay.com"
        assert len(resp.sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_owner_ignored_without_owner_only(self, agent):
        # owner without ownerOnly must NOT filter (owner passed through as None).
        with patch.object(
            agent.session_manager, "list_sessions", return_value=[],
        ) as mock_list:
            await agent.list_sessions(cwd="/tmp", hermes={"owner": "me@yallaplay.com"})
        _, kwargs = mock_list.call_args
        assert kwargs.get("owner") is None

# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:
    @pytest.mark.asyncio
    async def test_set_session_mode_returns_response(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        resp = await agent.set_session_mode(mode_id="accept_edits", session_id=new_resp.session_id)
        state = agent.session_manager.get_session(new_resp.session_id)

        assert isinstance(resp, SetSessionModeResponse)
        assert getattr(state, "mode", None) == "accept_edits"

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        # The response advertises the full option set (currently the
        # reasoning-effort select) so clients can re-render their pickers.
        assert [opt["id"] for opt in config_result["configOptions"]] == ["reasoning_effort"]

    @pytest.mark.asyncio
    async def test_router_accepts_unstable_model_switch_when_enabled(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent, use_unstable_protocol=True)

        result = await router(
            "session/set_model",
            {"modelId": "gpt-5.4", "sessionId": new_resp.session_id},
            False,
        )
        state = agent.session_manager.get_session(new_resp.session_id)

        assert result == {}
        assert state.model == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_set_session_model_accepts_provider_prefixed_choice(self, tmp_path, monkeypatch):
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        # Pin the parser so this test doesn't depend on live
        # ``_KNOWN_PROVIDER_NAMES`` / ``_PROVIDER_ALIASES`` module state
        # (sibling of the same hardening on
        # ``test_model_switch_uses_requested_provider``).
        monkeypatch.setattr(
            "hermes_cli.models.parse_model_input",
            lambda raw, current: ("anthropic", "claude-sonnet-4-6"),
        )
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, current: None,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = await acp_agent.set_session_model(
                model_id="anthropic:claude-sonnet-4-6",
                session_id=state.session_id,
            )

        assert isinstance(result, SetSessionModelResponse)
        assert state.model == "claude-sonnet-4-6"
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        assert runtime_calls[-1] == "anthropic"


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_returns_end_turn_for_empty_message(self, agent):
        new_resp = await agent.new_session(cwd=".")
        prompt = [TextContentBlock(type="text", text="   ")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_prompt_runs_agent(self, agent):
        """The prompt method should call run_conversation on the agent."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Mock the agent's run_conversation
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Hello! How can I help?",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        })

        # Set up a mock connection
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "end_turn"
        state.agent.run_conversation.assert_called_once()
        assert state.agent.tool_progress_callback is not None
        assert state.agent.step_callback is not None
        assert state.agent.stream_delta_callback is not None
        assert state.agent.reasoning_callback is not None
        assert state.agent.thinking_callback is None

    @pytest.mark.asyncio
    async def test_prompt_injects_authenticated_owner_on_first_turn(self, agent):
        """A session with an authenticated owner surfaces the identity to the
        agent ONCE, on the first turn (empty history), prepended to the user
        message. The persisted user message stays clean (original text)."""
        new_resp = await agent.new_session(cwd=".", hermes={"owner": "kareem@yallaplay.com"})
        state = agent.session_manager.get_session(new_resp.session_id)
        assert state.owner == "kareem@yallaplay.com"
        assert not state.history  # first turn

        captured = {}

        def _run(**kwargs):
            captured.update(kwargs)
            return {"final_response": "hi", "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="hello")],
                           session_id=new_resp.session_id)

        # The owner note is prepended to what the model sees this turn...
        assert "kareem@yallaplay.com" in captured["user_message"]
        assert "authenticated user" in captured["user_message"]
        assert captured["user_message"].endswith("hello")
        # ...but the persisted message is the clean original.
        assert captured["persist_user_message"] == "hello"

    @pytest.mark.asyncio
    async def test_prompt_omits_owner_note_on_later_turns(self, agent):
        """Once history exists, the owner note is NOT re-injected (already in
        history — re-sending wastes tokens)."""
        new_resp = await agent.new_session(cwd=".", hermes={"owner": "kareem@yallaplay.com"})
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [
            {"role": "user", "content": "prior"},
            {"role": "assistant", "content": "ok"},
        ]

        captured = {}

        def _run(**kwargs):
            captured.update(kwargs)
            return {"final_response": "hi", "messages": state.history}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="second")],
                           session_id=new_resp.session_id)

        assert captured["user_message"] == "second"
        assert "authenticated user" not in captured["user_message"]

    @pytest.mark.asyncio
    async def test_prompt_no_owner_note_when_unowned(self, agent):
        """A session without an owner injects nothing."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        assert state.owner is None

        captured = {}

        def _run(**kwargs):
            captured.update(kwargs)
            return {"final_response": "hi", "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="hello")],
                           session_id=new_resp.session_id)

        assert captured["user_message"] == "hello"

    @pytest.mark.asyncio
    async def test_prompt_wires_subagent_router_and_finalizes(self, agent):
        """prompt() builds a subagent router for the turn, hands subagent.*
        events to it via the tool progress callback, and finalizes it at turn
        end so a crashed child can't leave a stuck-running child session."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def _run(**kwargs):
            # Simulate a delegate child relaying events mid-turn, with a
            # child that never sends subagent.complete (crash path).
            cb = state.agent.tool_progress_callback
            cb("subagent.tool", "terminal", "$ ls", {"command": "ls"},
               child_session_id="child-abc")
            return {"final_response": "done", "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "done"},
            ]}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(prompt=[TextContentBlock(type="text", text="go")],
                           session_id=new_resp.session_id)

        child_updates = [
            call.kwargs.get("update") if "update" in call.kwargs else call.args[1]
            for call in mock_conn.session_update.await_args_list
            if (call.kwargs.get("session_id") or (call.args[0] if call.args else None))
            == "child-abc"
        ]
        kinds = [getattr(u, "session_update", None) for u in child_updates]
        # Lazy isRunning:true, tool start, then finalize's dangling completion
        # + isRunning:false at turn end.
        assert kinds == [
            "session_info_update",
            "tool_call",
            "tool_call_update",
            "session_info_update",
        ]
        assert child_updates[0].field_meta["hermes"]["isRunning"] is True
        assert child_updates[-1].field_meta["hermes"]["isRunning"] is False

    @pytest.mark.asyncio
    async def test_prompt_updates_history(self, agent):
        """After a prompt, session history should be updated."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        expected_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hey",
            "messages": expected_history,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert state.history == expected_history

    @pytest.mark.asyncio
    async def test_prompt_returns_user_history_index_meta(self, agent):
        """PromptResponse carries _meta.hermes.userHistoryIndex — the absolute
        post-turn index of this turn's user message, for fork-from-here."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Simulate a turn appended after a prior user/assistant exchange, so the
        # new user message lands at absolute index 2 in the post-turn history.
        def _run(*_args, **_kwargs):
            state.agent._persist_user_message_idx = 2
            return {
                "final_response": "sure",
                "messages": [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "earlier reply"},
                    {"role": "user", "content": "now"},
                    {"role": "assistant", "content": "sure"},
                ],
            }

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="now")],
            session_id=new_resp.session_id,
        )
        assert resp.field_meta == {"hermes": {"userHistoryIndex": 2}}

    @pytest.mark.asyncio
    async def test_prompt_omits_user_history_index_when_unavailable(self, agent):
        """No _persist_user_message_idx (e.g. a bare turn) → no _meta stamp."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        # Ensure the attribute is absent so the getattr fallback fires.
        if hasattr(state.agent, "_persist_user_message_idx"):
            delattr(state.agent, "_persist_user_message_idx")

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hi",
            "messages": [{"role": "user", "content": "hi"}],
        })
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=new_resp.session_id,
        )
        assert resp.field_meta is None

    @pytest.mark.asyncio
    async def test_prompt_user_history_index_recomputed_after_compression(self, agent):
        """Mid-turn compression rewrites the message list AFTER the agent stamps
        ``_persist_user_message_idx`` (agent/turn_context.py sets it pre-preflight).
        The coordinate must come from the finalized result history — and the
        prompt metadata, replay metadata, and fork prefix must all agree."""
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)

        # Finalized post-compression history: a 41-entry pre-turn transcript
        # collapsed to a summary, with the current user turn surviving at
        # absolute index 1 — while the agent's early stamp still says 40.
        compressed_history = [
            {"role": "user", "content": "[Context summary]\nPrevious conversation"},
            {"role": "user", "content": "now"},
            {"role": "assistant", "content": "done"},
        ]

        def _run(*_args, **_kwargs):
            state.agent._persist_user_message_idx = 40  # stale pre-compression index
            return {"final_response": "done", "messages": list(compressed_history)}

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="now")],
            session_id=new_resp.session_id,
        )

        # 1. Prompt metadata reflects the finalized coordinate, not the stale stamp.
        assert resp.field_meta == {"hermes": {"userHistoryIndex": 1}}

        # 2. Replay metadata agrees: session/load stamps the same index on the
        #    replayed chunk for this user turn.
        mock_conn.session_update.reset_mock()
        await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)
        user_calls = [
            call for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None) == "user_message_chunk"
        ]
        stamped = {
            call.kwargs["update"].content.text: call.kwargs["hermes"]["historyIndex"]
            for call in user_calls
        }
        assert stamped["now"] == 1

        # 3. Fork prefix agrees: keepHistory=<returned index> rewinds to just
        #    before this turn's user message.
        fork_resp = await agent.fork_session(
            cwd="/forked",
            session_id=new_resp.session_id,
            hermes={"keepHistory": 1},
        )
        forked = agent.session_manager.get_session(fork_resp.session_id)
        assert forked.history == compressed_history[:1]

    @pytest.mark.asyncio
    async def test_prompt_omits_user_history_index_when_message_summarized_away(self, agent):
        """If compression removed the current user message entirely, no index
        can be honestly returned — omit the stamp (client falls back to a
        full-history fork) rather than pointing at the wrong entry."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def _run(*_args, **_kwargs):
            state.agent._persist_user_message_idx = 40
            return {
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": "[Context summary]\nEverything incl. the last user turn"},
                    {"role": "assistant", "content": "done"},
                ],
            }

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="now")],
            session_id=new_resp.session_id,
        )
        assert resp.field_meta is None

    @pytest.mark.asyncio
    async def test_prompt_sends_final_message_update(self, agent):
        """The final response should be sent as an AgentMessageChunk."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "I can help with that!",
            "messages": [],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="help me")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        # session_update should include the final message (usage_update may follow it)
        mock_conn.session_update.assert_called()
        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        assert any(update.session_update == "agent_message_chunk" for update in updates)

    @pytest.mark.asyncio
    async def test_prompt_suppresses_cancel_interrupt_sentinel(self, agent):
        """ACP cancel status text should not be emitted as assistant output."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        sentinel = "Operation interrupted: waiting for model response (3.3s elapsed)."

        def mock_run(*args, **kwargs):
            state.cancel_event.set()
            return {
                "final_response": sentinel,
                "messages": list(state.history),
                "interrupted": True,
                "completed": False,
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("agent.title_generator.maybe_auto_title") as mock_title:
            prompt = [TextContentBlock(type="text", text="please do a long task")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_texts = [
            update.content.text
            for update in updates
            if update.session_update == "agent_message_chunk"
        ]
        assert resp.stop_reason == "cancelled"
        assert sentinel not in agent_texts
        assert not any(text.startswith("Operation interrupted:") for text in agent_texts)
        mock_title.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_keeps_real_final_response_on_cancelled_turn(self, agent):
        """A cancel flag must not suppress actual assistant/model text."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        final_text = "The actual model answer arrived before cancellation settled."

        def mock_run(*args, **kwargs):
            state.cancel_event.set()
            return {
                "final_response": final_text,
                "messages": [],
                "interrupted": True,
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="finish if you can")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_texts = [
            update.content.text
            for update in updates
            if update.session_update == "agent_message_chunk"
        ]
        assert resp.stop_reason == "cancelled"
        assert final_text in agent_texts

    @pytest.mark.asyncio
    async def test_prompt_cancelled_turn_with_none_response_does_not_brick_session(self, agent):
        """A /stop mid tool call yields ``final_response=None`` (key present).

        Regression: ``result.get("final_response", "")`` kept the ``None``,
        ``None.startswith(...)`` raised, the exception escaped ``prompt()``
        before ``state.is_running`` was reset, and every subsequent message
        was queued forever ("Queued for the next turn. (N queued)")."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            state.cancel_event.set()
            return {
                "final_response": None,
                "messages": [],
                "interrupted": True,
                "completed": False,
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="do something slow")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"
        # The session must be released — not stuck busy queueing forever.
        assert state.is_running is False
        assert state.queued_prompts == []

        # A follow-up prompt must run as a normal turn, not get queued.
        state.cancel_event.clear()
        follow_up_ran = []

        def mock_run_ok(*args, **kwargs):
            follow_up_ran.append(True)
            return {"final_response": "done", "messages": [], "completed": True}

        state.agent.run_conversation = mock_run_ok
        resp2 = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="follow up")],
            session_id=new_resp.session_id,
        )
        assert resp2.stop_reason == "end_turn"
        assert follow_up_ran, "follow-up prompt was queued instead of running"

    @pytest.mark.asyncio
    async def test_prompt_tail_exception_still_releases_session(self, agent):
        """Any exception in the post-turn tail must not leave is_running set."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            return {"final_response": "ok", "messages": [{"role": "user", "content": "x"}], "completed": True}

        state.agent.run_conversation = mock_run

        # Force a failure inside the guarded tail (history persistence).
        agent.session_manager.save_session = MagicMock(side_effect=RuntimeError("disk full"))

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with pytest.raises(RuntimeError):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="hi")],
                session_id=new_resp.session_id,
            )

        assert state.is_running is False
        assert state.current_prompt_text == ""

    @pytest.mark.asyncio
    async def test_prompt_propagates_hermes_session_id_env(self, agent, monkeypatch):
        """ACP must propagate the originating session id to the agent loop
        via ``HERMES_SESSION_ID`` so tools that want to stamp side-effects
        with it (e.g. ``kanban_create``) can read the env var inside
        ``run_conversation``. The variable must be visible during the
        agent call AND restored afterwards so a re-used executor thread
        doesn't leak one session's id into another."""
        # Pre-condition: env is clean.
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        captured: dict[str, str | None] = {}

        def mock_run(user_message, conversation_history=None, task_id=None, **kwargs):
            # Inside the agent loop the env var must reflect the active
            # ACP session id. ``task_id`` is also the session id at this
            # boundary; assert both for symmetry.
            captured["env"] = os.environ.get("HERMES_SESSION_ID")
            captured["task_id"] = task_id
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert captured["env"] == new_resp.session_id, (
            "HERMES_SESSION_ID must be set to the originating ACP session id "
            "while the agent loop is running"
        )
        assert captured["task_id"] == new_resp.session_id
        # Post-condition: must be restored to the prior value (None here).
        assert os.environ.get("HERMES_SESSION_ID") is None, (
            "HERMES_SESSION_ID must be restored after the agent call so "
            "a re-used executor thread doesn't leak the id into the next "
            "session's tools"
        )

    @pytest.mark.asyncio
    async def test_prompt_restores_prior_hermes_session_id(self, agent, monkeypatch):
        """If the env already had HERMES_SESSION_ID set (e.g. nested
        agent loops), the prior value must be restored after the inner
        prompt completes — not popped, not left at the inner id."""
        monkeypatch.setenv("HERMES_SESSION_ID", "outer-sess")

        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        captured: dict[str, str | None] = {}

        def mock_run(*args, **kwargs):
            captured["inner"] = os.environ.get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hi")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert captured["inner"] == new_resp.session_id
        # Outer scope must be restored.
        assert os.environ.get("HERMES_SESSION_ID") == "outer-sess"

    @pytest.mark.asyncio
    async def test_prompt_does_not_duplicate_streamed_final_message(self, agent):
        """If ACP already streamed response chunks, final_response should not be sent again."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            state.agent.stream_delta_callback("streamed answer")
            return {"final_response": "streamed answer", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        agent_chunks = [update for update in updates if update.session_update == "agent_message_chunk"]
        assert len(agent_chunks) == 1
        assert agent_chunks[0].content.text == "streamed answer"

    @pytest.mark.asyncio
    async def test_prompt_delivers_transformed_response_after_streaming(self, agent):
        """If a transform_llm_output plugin hook modifies the response after
        streaming, ACP must deliver the transformed final_response so the
        appended/rewritten text reaches the client.
        """
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            state.agent.stream_delta_callback("original answer")
            return {
                "final_response": "original answer\n\n[plugin appended this]",
                "response_transformed": True,
                "messages": [],
            }

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="hello")]
        await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        # The streamed chunk and the post-stream transformed message should
        # both be present (final delivery is a separate update_agent_message_text
        # call carrying the full transformed text).
        all_texts = [
            getattr(getattr(u, "content", None), "text", None)
            for u in updates
        ]
        assert any(
            text and "[plugin appended this]" in text for text in all_texts
        ), f"expected transformed final to be delivered, got: {all_texts!r}"


    @pytest.mark.asyncio
    async def test_prompt_auto_titles_session(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.model = "gpt-5.6-sol"
        state.agent.provider = "openai-codex"
        state.agent.base_url = "https://chatgpt.example.test/backend-api/codex"
        state.agent.api_key = object()
        state.agent.api_mode = "codex_responses"
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Here is the fix.",
            "messages": [
                {"role": "user", "content": "fix the broken ACP history"},
                {"role": "assistant", "content": "Here is the fix."},
            ],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        with patch("agent.title_generator.maybe_auto_title") as mock_title:
            prompt = [TextContentBlock(type="text", text="fix the broken ACP history")]
            await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        mock_title.assert_called_once()
        assert mock_title.call_args.args[1] == new_resp.session_id
        assert mock_title.call_args.args[2] == "fix the broken ACP history"
        assert mock_title.call_args.args[3] == "Here is the fix."
        assert mock_title.call_args.kwargs["main_runtime"] == {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.example.test/backend-api/codex",
            "api_key": state.agent.api_key,
            "api_mode": "codex_responses",
        }
        assert callable(mock_title.call_args.kwargs["title_callback"])

    @pytest.mark.asyncio
    async def test_prompt_sends_session_info_update_after_auto_title(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "Done.",
            "messages": [
                {"role": "user", "content": "fix zed titles"},
                {"role": "assistant", "content": "Done."},
            ],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        })

        def fake_auto_title(db, session_id, user_text, final_response, history, **kwargs):
            db.set_session_title(session_id, "Fix Zed titles")
            kwargs["title_callback"]("Fix Zed titles")

        with patch("agent.title_generator.maybe_auto_title", side_effect=fake_auto_title):
            mock_conn.session_update.reset_mock()
            await agent.prompt(
                session_id=resp.session_id,
                prompt=[TextContentBlock(type="text", text="fix zed titles")],
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.await_args_list
        ]
        info_updates = [u for u in updates if isinstance(u, SessionInfoUpdate)]
        # Turn start/end status signals (_meta.hermes.isRunning) also ride
        # SessionInfoUpdate but carry no title; the auto-title assertion is
        # about the single TITLED update.
        titled_updates = [u for u in info_updates if u.title is not None]
        assert len(titled_updates) == 1
        assert titled_updates[0].session_update == "session_info_update"
        assert titled_updates[0].title == "Fix Zed titles"

    @pytest.mark.asyncio
    async def test_prompt_populates_usage_from_top_level_run_conversation_fields(self, agent):
        """ACP should map top-level token fields into PromptResponse.usage."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "usage attached",
            "messages": [],
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
            "reasoning_tokens": 7,
            "cache_read_tokens": 11,
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="show usage")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert isinstance(resp, PromptResponse)
        assert resp.usage is not None
        assert resp.usage.input_tokens == 123
        assert resp.usage.output_tokens == 45
        assert resp.usage.total_tokens == 168
        assert resp.usage.thought_tokens == 7
        assert resp.usage.cached_read_tokens == 11

    @pytest.mark.asyncio
    async def test_prompt_cancelled_returns_cancelled_stop_reason(self, agent):
        """If cancel is called during prompt, stop_reason should be 'cancelled'."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)

        def mock_run(*args, **kwargs):
            # Simulate cancel being set during execution
            state.cancel_event.set()
            return {"final_response": "interrupted", "messages": []}

        state.agent.run_conversation = mock_run

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="do something")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "cancelled"


# ---------------------------------------------------------------------------
# prompt rewind (_meta.hermes.keepHistory on session/prompt)
# ---------------------------------------------------------------------------


class TestPromptRewind:
    """In-place rewind: session/prompt with _meta.hermes.keepHistory truncates
    the session history to the first N entries before the turn runs."""

    @staticmethod
    def _seed_history(state):
        state.history.extend(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "mistake"},
                {"role": "assistant", "content": "partial answer"},
            ]
        )

    @staticmethod
    def _wire_agent(agent, state, response="done"):
        captured = {}

        def _run(**kwargs):
            captured.update(kwargs)
            history = kwargs["conversation_history"]
            return {
                "final_response": response,
                "messages": history
                + [
                    {"role": "user", "content": kwargs["user_message"]},
                    {"role": "assistant", "content": response},
                ],
            }

        state.agent.run_conversation = MagicMock(side_effect=_run)
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn
        return captured

    @pytest.mark.asyncio
    async def test_prompt_keep_history_truncates_before_turn(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)
        captured = self._wire_agent(agent, state)

        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="corrected prompt")],
            session_id=new_resp.session_id,
            hermes={"keepHistory": 2},
        )

        assert resp.stop_reason == "end_turn"
        # The turn ran on the truncated prefix — the mistaken turn is gone.
        assert captured["conversation_history"][:2] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert all(m.get("content") != "mistake" for m in state.history)
        assert state.history[-1] == {"role": "assistant", "content": "done"}

    @pytest.mark.asyncio
    async def test_prompt_keep_history_router_wire_shape(self, agent):
        """End-to-end through the JSON-RPC router: _meta survives the wire."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)
        captured = self._wire_agent(agent, state)
        router = build_agent_router(agent, use_unstable_protocol=True)

        result = await router(
            "session/prompt",
            {
                "sessionId": new_resp.session_id,
                "prompt": [{"type": "text", "text": "corrected"}],
                "_meta": {"hermes": {"keepHistory": 2}},
            },
            False,
        )

        assert result.stop_reason == "end_turn"
        assert len(captured["conversation_history"]) == 2

    @pytest.mark.asyncio
    async def test_prompt_keep_history_invalid_raises(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)

        for bad in ("2", -1, True, 1.5):
            with pytest.raises(acp.RequestError):
                await agent.prompt(
                    prompt=[TextContentBlock(type="text", text="x")],
                    session_id=new_resp.session_id,
                    hermes={"keepHistory": bad},
                )
        assert len(state.history) == 4  # untouched

    @pytest.mark.asyncio
    async def test_prompt_keep_history_empty_prompt_raises(self, agent):
        """Truncate-without-resend is not a supported shape."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)

        with pytest.raises(acp.RequestError):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="   ")],
                session_id=new_resp.session_id,
                hermes={"keepHistory": 2},
            )
        assert len(state.history) == 4

    @pytest.mark.asyncio
    async def test_prompt_keep_history_refused_while_running(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)
        state.is_running = True

        with pytest.raises(acp.RequestError):
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="corrected")],
                session_id=new_resp.session_id,
                hermes={"keepHistory": 2},
            )
        assert len(state.history) == 4

    @pytest.mark.asyncio
    async def test_prompt_keep_history_beyond_length_is_noop(self, agent):
        """keepHistory >= len(history) keeps everything (no reconcile write)."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)
        captured = self._wire_agent(agent, state)

        with patch.object(agent, "_reconcile_persisted_history") as reconcile:
            resp = await agent.prompt(
                prompt=[TextContentBlock(type="text", text="follow-up")],
                session_id=new_resp.session_id,
                hermes={"keepHistory": 10},
            )

        assert resp.stop_reason == "end_turn"
        assert len(captured["conversation_history"]) == 4
        reconcile.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_keep_history_reconciles_persisted_rows(self, agent):
        """A real truncation mirrors the rewrite into the message store so a
        DB replay or restart can't resurrect the dropped turn."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        self._seed_history(state)
        self._wire_agent(agent, state)

        with patch.object(agent, "_reconcile_persisted_history") as reconcile:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="corrected")],
                session_id=new_resp.session_id,
                hermes={"keepHistory": 2},
            )

        reconcile.assert_called_once()
        _, messages = reconcile.call_args[0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_initialize_advertises_prompt_keep_history_extension(self, agent):
        resp = await agent.initialize(protocol_version=1)
        prompt_caps = resp.agent_capabilities.prompt_capabilities
        assert prompt_caps.field_meta == {"hermes": {"keepHistory": True}}


# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result

    def test_context_empty(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = []
        result = agent._handle_slash_command("/context", state)
        assert "empty" in result.lower()

    def test_context_with_messages(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = agent._handle_slash_command("/context", state)
        assert "2 messages" in result
        assert "user: 1" in result

    def test_context_shows_usage_and_compression_threshold(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(
            context_length=100_000,
            threshold_tokens=80_000,
        )
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            result = agent._handle_slash_command("/context", state)

        assert "Context usage: ~25,000 / 100,000 tokens (25.0%)" in result
        assert "Compression: ~55,000 tokens until threshold (~80,000, 80%)" in result
        assert "Tip: run /compress" in result

    def test_context_says_compression_due_when_past_threshold(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(
            context_length=100_000,
            threshold_tokens=80_000,
        )

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=82_000,
        ):
            result = agent._handle_slash_command("/context", state)

        assert "Context usage: ~82,000 / 100,000 tokens (82.0%)" in result
        assert "Compression: due now (threshold ~80,000, 80%). Run /compress." in result

    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0

    def test_reset_resets_agent_session_state(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.reset_session_state = MagicMock()

        with patch.object(agent.session_manager, "save_session") as mock_save:
            result = agent._handle_slash_command("/reset", state)

        assert "cleared" in result.lower()
        assert state.history == []
        state.agent.reset_session_state.assert_called_once_with()
        mock_save.assert_called_once_with(state.session_id)

    def test_reset_saves_session_when_agent_state_reset_fails(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.reset_session_state = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch("acp_adapter.server.logger") as mock_logger,
        ):
            result = agent._handle_slash_command("/reset", state)

        assert "cleared" in result.lower()
        assert "state reset failed" in result.lower()
        assert state.history == []
        state.agent.reset_session_state.assert_called_once_with()
        mock_save.assert_called_once_with(state.session_id)
        mock_logger.warning.assert_called_once()

    def test_version(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/version", state)
        assert HERMES_VERSION in result

    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        original_session_db = object()
        state.agent._session_db = original_session_db

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force):
            assert state.agent._session_db is None
            assert messages == state.history
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert state.history == [{"role": "user", "content": "summary"}]
        assert state.agent._session_db is original_session_db
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
            force=True,
        )
        mock_save.assert_called_once_with(state.session_id)

    def test_compress_works_when_auto_compaction_disabled(self, agent, mock_manager):
        """compression.enabled: false disables *automatic* compaction only —
        manual /compress must still compress (matches CLI /compress and the
        gateway handler)."""
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = False
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        state.agent._session_db = None
        state.agent._compress_context = MagicMock(
            return_value=([{"role": "user", "content": "summary"}], "new-system")
        )

        with (
            patch.object(agent.session_manager, "save_session"),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "disabled" not in result.lower()
        assert "Context compressed: 4 -> 1 messages" in result
        state.agent._compress_context.assert_called_once()
        assert state.agent._compress_context.call_args.kwargs.get("force") is True

    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None

    def _owned_persistence_state(self, tmp_path):
        """Real SessionDB + manager whose agent OWNS persistence.

        Mimics a live ACP session: run_agent has already flushed the turn's
        rows itself (agent._session_db is the manager's db and
        _session_db_created=True), so save_session deliberately skips
        replace_messages — the slash command itself must reconcile the DB.
        """
        db = SessionDB(tmp_path / "state.db")

        def factory():
            return SimpleNamespace(
                model="test-model",
                _session_db=db,
                _session_db_created=True,
            )

        manager = SessionManager(agent_factory=factory, db=db)
        acp_agent = HermesACPAgent(session_manager=manager)
        state = manager.create_session(cwd="/work")

        # The agent's own flusher already persisted the live transcript.
        db.append_message(session_id=state.session_id, role="user", content="one")
        db.append_message(session_id=state.session_id, role="assistant", content="two")
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        return acp_agent, state, db

    def test_cmd_compress_reconciles_persisted_history(self, tmp_path):
        """/compress must mirror the in-memory rewrite into the message store:
        active rows become the compacted set, pre-compact rows are archived
        (not deleted) — otherwise mid-turn replay and post-restart restore
        resurrect the uncompacted transcript without the summary."""
        acp_agent, state, db = self._owned_persistence_state(tmp_path)
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        state.agent._compress_context = MagicMock(
            return_value=([{"role": "user", "content": "compacted summary"}], "sys")
        )

        result = acp_agent._handle_slash_command("/compress", state)

        assert "Context compressed" in result
        # Active DB rows must now equal the compacted in-memory history.
        active = [
            m["content"] for m in db.get_messages_as_conversation(state.session_id)
        ]
        assert active == ["compacted summary"]
        # Pre-compact rows survive as archived (soft-archive, not delete).
        all_rows = [
            m["content"]
            for m in db.get_messages(state.session_id, include_inactive=True)
        ]
        assert "one" in all_rows
        assert "two" in all_rows

    def test_cmd_reset_reconciles_persisted_history(self, tmp_path):
        """/reset must empty the ACTIVE row set in the message store, keeping
        the old rows archived — otherwise the DB stays longer than memory and
        every subsequent mid-turn replay resurrects the cleared transcript."""
        acp_agent, state, db = self._owned_persistence_state(tmp_path)

        result = acp_agent._handle_slash_command("/reset", state)

        assert "cleared" in result.lower()
        assert state.history == []
        assert db.get_messages_as_conversation(state.session_id) == []
        # Old rows are archived, not destroyed.
        all_rows = [
            m["content"]
            for m in db.get_messages(state.session_id, include_inactive=True)
        ]
        assert "one" in all_rows
        assert "two" in all_rows

    @pytest.mark.asyncio
    async def test_slash_command_intercepted_in_prompt(self, agent, mock_manager):
        """Slash commands should be handled without calling the LLM."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        prompt = [TextContentBlock(type="text", text="/help")]
        resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"
        updates = [
            call.kwargs.get("update") or call.args[1]
            for call in mock_conn.session_update.call_args_list
        ]
        assert any(update.session_update == "agent_message_chunk" for update in updates)
        assert any(update.session_update == "usage_update" for update in updates)

    @pytest.mark.asyncio
    async def test_unknown_slash_falls_through_to_llm(self, agent, mock_manager):
        """Unknown /commands should be sent to the LLM, not intercepted."""
        new_resp = await agent.new_session(cwd="/tmp")
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        mock_conn.request_permission = AsyncMock(return_value=None)
        agent._conn = mock_conn

        # Mock run_in_executor to avoid actually running the agent
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value={
                "final_response": "I processed /foo",
                "messages": [],
            })
            prompt = [TextContentBlock(type="text", text="/foo bar")]
            resp = await agent.prompt(prompt=prompt, session_id=new_resp.session_id)

        assert resp.stop_reason == "end_turn"

    def test_model_switch_uses_requested_provider(self, tmp_path, monkeypatch):
        """`/model provider:model` should rebuild the ACP agent on that provider."""
        runtime_calls = []

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            runtime_calls.append(requested)
            provider = requested or "openrouter"
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "openrouter/gpt-5"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        # Pin the model-string parser independently of the live
        # ``_KNOWN_PROVIDER_NAMES`` / ``_PROVIDER_ALIASES`` module state.
        # Otherwise any test in the same xdist worker that mutates those
        # globals (e.g. registers a custom provider that shadows
        # ``anthropic``) flakes this one — observed once in CI as
        # ``'custom' == 'anthropic'``.
        monkeypatch.setattr(
            "hermes_cli.models.parse_model_input",
            lambda raw, current: ("anthropic", "claude-sonnet-4-6"),
        )
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, current: None,
        )
        manager = SessionManager(db=SessionDB(tmp_path / "state.db"))

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            acp_agent = HermesACPAgent(session_manager=manager)
            state = manager.create_session(cwd="/tmp")
            result = acp_agent._cmd_model("anthropic:claude-sonnet-4-6", state)

        assert "Provider: anthropic" in result
        assert state.agent.provider == "anthropic"
        assert state.agent.base_url == "https://anthropic.example/v1"
        # ``state.agent.provider == "anthropic"`` plus the base_url check above
        # already prove ``fake_resolve_runtime_provider`` was called with
        # ``requested="anthropic"`` for the model-switch step — the agent's
        # provider/base_url come from that fake's return value. The legacy
        # ``runtime_calls[-1] == "anthropic"`` assertion was flaky in CI
        # under specific xdist-slice scheduling (saw ``'custom' == 'anthropic'``
        # repeatedly) and was redundant with those checks, so it's gone.
        assert "anthropic" in runtime_calls


# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}

    @pytest.mark.asyncio
    async def test_registers_http_servers(self, agent, mock_manager):
        """McpServerHttp servers are converted correctly."""
        from acp.schema import McpServerHttp, HttpHeader

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerHttp(
            name="http-server",
            url="https://api.example.com/mcp",
            headers=[HttpHeader(name="Authorization", value="Bearer tok")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return []

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "http-server" in registered_config
        cfg = registered_config["http-server"]
        assert cfg["url"] == "https://api.example.com/mcp"
        assert cfg["headers"] == {"Authorization": "Bearer tok"}

    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])


class TestBinaryAttachmentPersistence:
    """Binary prompt attachments must be materialized into the document
    cache and the model pointed at the saved path — not dropped."""

    @staticmethod
    def _make_embedded_blob_block(name, blob, mime_type):
        from acp.schema import BlobResourceContents, EmbeddedResourceContentBlock

        return EmbeddedResourceContentBlock(
            type="resource",
            resource=BlobResourceContents(uri=name, blob=blob, mime_type=mime_type),
        )

    def test_embedded_binary_blob_saved_not_dropped(self, tmp_path, monkeypatch):
        from acp_adapter import server as server_mod

        monkeypatch.setattr(
            "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
        )
        block = self._make_embedded_blob_block(
            name="trace.json.gz",
            blob=base64.b64encode(b"\x1f\x8b" + b"\x00" * 64).decode(),
            mime_type="application/x-gzip",
        )
        parts = server_mod._embedded_resource_to_parts(block)
        assert len(parts) == 1 and parts[0]["type"] == "text"
        assert "omitted" not in parts[0]["text"]
        assert "saved to" in parts[0]["text"]
        saved = re.search(r"saved to (\S+)", parts[0]["text"]).group(1)
        assert Path(saved).read_bytes()[:2] == b"\x1f\x8b"
        assert "trace.json.gz" in Path(saved).name

    def test_embedded_binary_blob_save_failure_falls_back(self, monkeypatch):
        from acp_adapter import server as server_mod

        def _boom(data, filename):
            raise OSError("disk full")

        monkeypatch.setattr("gateway.platforms.base.cache_document_from_bytes", _boom)
        block = self._make_embedded_blob_block(
            name="x.bin",
            blob=base64.b64encode(b"\x00\x01").decode(),
            mime_type="application/octet-stream",
        )
        parts = server_mod._embedded_resource_to_parts(block)
        assert "omitted" in parts[0]["text"]

    def test_oversized_image_blob_saved_with_path(self, tmp_path, monkeypatch):
        from acp_adapter import server as server_mod

        monkeypatch.setattr(
            "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
        )
        data = b"\x89PNG" + b"\x00" * (server_mod._MAX_ACP_RESOURCE_BYTES + 1)
        block = self._make_embedded_blob_block(
            name="big.png",
            blob=base64.b64encode(data).decode(),
            mime_type="image/png",
        )
        parts = server_mod._embedded_resource_to_parts(block)
        assert len(parts) == 1 and parts[0]["type"] == "text"
        assert "too large to inline" in parts[0]["text"]
        assert "saved to" in parts[0]["text"]
        saved = re.search(r"saved to (\S+)", parts[0]["text"]).group(1)
        assert Path(saved).read_bytes()[:4] == b"\x89PNG"

    def test_oversized_text_blob_notes_full_file_path(self, tmp_path, monkeypatch):
        from acp_adapter import server as server_mod

        monkeypatch.setattr(
            "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
        )
        data = b"a" * (server_mod._MAX_ACP_RESOURCE_BYTES + 10)
        block = self._make_embedded_blob_block(
            name="big.txt",
            blob=base64.b64encode(data).decode(),
            mime_type="text/plain",
        )
        parts = server_mod._embedded_resource_to_parts(block)
        assert "full file saved to" in parts[0]["text"]
        saved = re.search(r"full file saved to (\S+?)\]?$", parts[0]["text"], re.M).group(1)
        assert Path(saved).stat().st_size == len(data)


class TestResourceLinkPathDisclosure:
    """Non-inlined resource links must name the on-disk path so the model
    can use its tools on the file (no copying — the file already exists)."""

    @staticmethod
    def _make_link_block(path, mime_type=None, name=None):
        from acp.schema import ResourceContentBlock

        return ResourceContentBlock(
            type="resource_link",
            uri=path.as_uri(),
            name=name or path.name,
            mime_type=mime_type,
        )

    def test_binary_file_link_includes_path(self, tmp_path):
        from acp_adapter import server as server_mod

        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        parts = server_mod._resource_link_to_parts(
            self._make_link_block(f, mime_type="application/octet-stream")
        )
        assert len(parts) == 1 and parts[0]["type"] == "text"
        assert str(f) in parts[0]["text"]
        assert "use your tools" in parts[0]["text"].lower()

    def test_oversized_image_link_includes_path(self, tmp_path):
        from acp_adapter import server as server_mod

        f = tmp_path / "big.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * (server_mod._MAX_ACP_RESOURCE_BYTES + 1))
        parts = server_mod._resource_link_to_parts(
            self._make_link_block(f, mime_type="image/png")
        )
        assert len(parts) == 1 and parts[0]["type"] == "text"
        assert "too large to inline" in parts[0]["text"]
        assert f"File is at {f}" in parts[0]["text"]


# ---------------------------------------------------------------------------
# _owns_notification_event — background-notification ownership predicate
# ---------------------------------------------------------------------------


class TestNotificationOwnership:
    """Fail-closed ownership check for completion_queue events.

    The predicate decides whether a background/delegation completion event
    belongs to a session hosted by THIS ACP server process. Only positive
    proof (the event's session id maps to an in-memory, non-subagent
    session) counts; everything else must be rejected.
    """

    def _insert_state(self, manager, session_id, subagent=False):
        """Insert a SessionState directly into the manager's in-memory map."""
        from acp_adapter.session import SessionState

        state = SessionState(
            session_id=session_id,
            agent=MagicMock(name="MockAIAgent"),
            cwd="/tmp",
            subagent=subagent,
        )
        with manager._lock:
            manager._sessions[session_id] = state
        return state

    def test_owns_event_with_matching_session_key(self, agent):
        self._insert_state(agent.session_manager, "sess-owned-1")
        evt = {"session_key": "sess-owned-1"}
        assert agent._owns_notification_event(evt) is True

    def test_rejects_unknown_session_key(self, agent):
        self._insert_state(agent.session_manager, "sess-owned-1")
        evt = {"session_key": "some-other-process-session"}
        assert agent._owns_notification_event(evt) is False

    def test_rejects_event_with_no_keys(self, agent):
        self._insert_state(agent.session_manager, "sess-owned-1")
        assert agent._owns_notification_event({}) is False

    def test_rejects_subagent_session_match(self, agent):
        self._insert_state(agent.session_manager, "child-1", subagent=True)
        evt = {"session_key": "child-1"}
        assert agent._owns_notification_event(evt) is False

    def test_delegation_event_matches_via_origin_ui_session_id(self, agent):
        self._insert_state(agent.session_manager, "ui-sess-1")
        evt = {
            "type": "async_delegation",
            "session_key": "",
            "origin_ui_session_id": "ui-sess-1",
        }
        assert agent._owns_notification_event(evt) is True

    def test_does_not_raise_on_garbage_input(self, agent):
        # None values, wrong types — must return False, never raise.
        assert agent._owns_notification_event({"session_key": None}) is False
        assert agent._owns_notification_event(
            {"session_key": None, "origin_ui_session_id": None}
        ) is False
        assert agent._owns_notification_event({"session_key": 12345}) is False


# ---------------------------------------------------------------------------
# _deliver_notification — synthetic-turn delivery primitive
# ---------------------------------------------------------------------------


class TestNotificationDelivery:
    """Delivery primitive for background-completion notifications.

    Given an owned session id and pre-formatted notification text, the
    method must inject it WITHOUT ever interrupting a running turn:
    busy → queue silently on ``state.queued_prompts``; idle → echo a
    user-message update then run a normal prompt turn (mirroring
    ``_run_spawned_first_turn``). Never raises.
    """

    NOTIFY_TEXT = "[IMPORTANT: Background process finished (session abc)]"

    def _insert_state(self, manager, session_id, is_running=False):
        """Insert a SessionState directly into the manager's in-memory map."""
        from acp_adapter.session import SessionState

        state = SessionState(
            session_id=session_id,
            agent=MagicMock(name="MockAIAgent"),
            cwd="/tmp",
            is_running=is_running,
        )
        with manager._lock:
            manager._sessions[session_id] = state
        return state

    def _attach_conn(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn
        return mock_conn

    @pytest.mark.asyncio
    async def test_busy_session_queues_silently(self, agent):
        # Running turn → append to queued_prompts; NO client ack, NO prompt.
        state = self._insert_state(agent.session_manager, "sess-busy", is_running=True)
        mock_conn = self._attach_conn(agent)
        agent.prompt = AsyncMock()

        await agent._deliver_notification("sess-busy", self.NOTIFY_TEXT)

        assert state.queued_prompts == [self.NOTIFY_TEXT]
        mock_conn.session_update.assert_not_awaited()
        agent.prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_session_echoes_and_prompts(self, agent):
        # Idle → user-message echo, then a normal prompt turn with the text.
        self._insert_state(agent.session_manager, "sess-idle")
        mock_conn = self._attach_conn(agent)
        agent.prompt = AsyncMock()

        await agent._deliver_notification("sess-idle", self.NOTIFY_TEXT)

        agent.prompt.assert_awaited_once()
        kwargs = agent.prompt.await_args.kwargs
        assert kwargs["session_id"] == "sess-idle"
        blocks = kwargs["prompt"]
        assert self.NOTIFY_TEXT in blocks[0].text
        assert mock_conn.session_update.await_count >= 1

    @pytest.mark.asyncio
    async def test_unknown_session_id_is_a_noop(self, agent):
        # No in-memory session → log-and-return, never raise.
        agent.prompt = AsyncMock()

        await agent._deliver_notification("no-such-session", self.NOTIFY_TEXT)

        agent.prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_without_conn_still_prompts(self, agent):
        # Client detached → skip the echo, still run the turn, no raise.
        self._insert_state(agent.session_manager, "sess-noconn")
        agent._conn = None
        agent.prompt = AsyncMock()

        await agent._deliver_notification("sess-noconn", self.NOTIFY_TEXT)

        agent.prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prompt_failure_does_not_propagate(self, agent):
        # A failing delivery turn is logged, never raised to the caller
        # (the watcher loop must survive any single delivery).
        self._insert_state(agent.session_manager, "sess-err")
        self._attach_conn(agent)
        agent.prompt = AsyncMock(side_effect=RuntimeError("boom"))

        await agent._deliver_notification("sess-err", self.NOTIFY_TEXT)
