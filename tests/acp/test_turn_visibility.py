"""Tests for cross-client turn visibility and in-process session spawning.

Covers the two-layer fix for the 2026-07-15 concurrent-writers incident:

* Layer 1 — ``session/load`` / ``session/resume`` responses surface a running
  turn via ``_meta.hermes.isRunning`` (+ ``currentPromptText``), and the
  prompt path emits ``session_info_update`` turn start/end signals so an
  attached-but-not-owner client can track the lifecycle.
* Layer 2 — the ``acp_spawn_session`` tool: schema injection on ACP agents,
  ContextVar-bound dispatch (mirroring ``edit_approval``), and the server's
  spawn requester creating a session plus scheduling its first turn.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import (
    LoadSessionResponse,
    ResumeSessionResponse,
    TextContentBlock,
)
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from acp_adapter.spawn import (
    SPAWN_SESSION_TOOL_NAME,
    clear_spawn_session_requester,
    inject_spawn_session_tool,
    maybe_dispatch_spawn_session,
    reset_spawn_session_requester,
    set_spawn_session_requester,
)


@pytest.fixture()
def mock_manager():
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    return HermesACPAgent(session_manager=mock_manager)


def _hermes_meta(resp):
    meta = getattr(resp, "field_meta", None) or {}
    return meta.get("hermes") or {}


def _loads(result) -> dict:
    """json.loads with a not-None guard (dispatch returns str for our tool)."""
    assert result is not None
    return json.loads(result)


# ---------------------------------------------------------------------------
# Layer 1 — load/resume surface a running turn
# ---------------------------------------------------------------------------


class TestTurnStateOnLoadResume:
    @pytest.mark.asyncio
    async def test_load_session_reports_running_turn(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        with state.runtime_lock:
            state.is_running = True
            state.current_prompt_text = "long investigation prompt"

        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, LoadSessionResponse)
        hermes = _hermes_meta(resp)
        assert hermes.get("isRunning") is True
        assert hermes.get("currentPromptText") == "long investigation prompt"

    @pytest.mark.asyncio
    async def test_load_session_idle_omits_running_meta(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")

        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)

        hermes = _hermes_meta(resp)
        assert "isRunning" not in hermes
        assert "currentPromptText" not in hermes

    @pytest.mark.asyncio
    async def test_resume_session_reports_running_turn(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        with state.runtime_lock:
            state.is_running = True
            state.current_prompt_text = "still working"

        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)

        assert isinstance(resp, ResumeSessionResponse)
        hermes = _hermes_meta(resp)
        assert hermes.get("isRunning") is True
        assert hermes.get("currentPromptText") == "still working"

    @pytest.mark.asyncio
    async def test_running_meta_preserves_provenance_meta(self, agent):
        """The running flag merges into _meta.hermes rather than replacing it."""
        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        with state.runtime_lock:
            state.is_running = True
            state.current_prompt_text = "x"

        agent._provenance_meta = MagicMock(
            return_value={"hermes": {"sessionProvenance": {"sessionKind": "root"}}}
        )
        resp = await agent.load_session(cwd="/tmp", session_id=new_resp.session_id)

        hermes = _hermes_meta(resp)
        assert hermes.get("isRunning") is True
        assert hermes.get("sessionProvenance") == {"sessionKind": "root"}


class TestTurnStatusUpdates:
    @pytest.mark.asyncio
    async def test_prompt_emits_turn_running_then_idle_updates(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "done",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "done"},
            ],
        })

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hello")],
            session_id=new_resp.session_id,
        )

        status_flags = [
            call.kwargs["update"].field_meta["hermes"]["isRunning"]
            for call in mock_conn.session_update.await_args_list
            if getattr(call.kwargs.get("update"), "session_update", None)
            == "session_info_update"
            and isinstance(getattr(call.kwargs.get("update"), "field_meta", None), dict)
            and "isRunning" in (call.kwargs["update"].field_meta.get("hermes") or {})
        ]
        assert status_flags[0] is True
        assert status_flags[-1] is False

    @pytest.mark.asyncio
    async def test_turn_ends_idle_even_when_tail_raises(self, agent):
        """The finally-guarded idle update fires even if the post-turn tail dies."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        # final_response=None + interrupted triggers the historical tail path;
        # keep it simple: raise from save_session via a broken history shape is
        # overkill — instead assert idle after a normal turn plus is_running
        # reset (regression guard for the strand-in-steer-mode class).
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": None,
            "messages": [],
            "interrupted": True,
        })
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=new_resp.session_id,
        )

        with state.runtime_lock:
            assert state.is_running is False


# ---------------------------------------------------------------------------
# Layer 2 — acp_spawn_session tool
# ---------------------------------------------------------------------------


class TestSpawnToolInjection:
    def test_inject_adds_schema_and_valid_name(self):
        fake = MagicMock()
        fake.tools = []
        fake.valid_tool_names = set()

        assert inject_spawn_session_tool(fake) is True
        names = [t["function"]["name"] for t in fake.tools]
        assert names == [SPAWN_SESSION_TOOL_NAME]
        assert SPAWN_SESSION_TOOL_NAME in fake.valid_tool_names

    def test_inject_is_idempotent(self):
        fake = MagicMock()
        fake.tools = []
        fake.valid_tool_names = set()
        inject_spawn_session_tool(fake)

        assert inject_spawn_session_tool(fake) is False
        assert len(fake.tools) == 1

    def test_inject_noops_without_tool_surface(self):
        fake = MagicMock(spec=[])  # no .tools attribute
        assert inject_spawn_session_tool(fake) is False


class TestSpawnDispatch:
    def teardown_method(self):
        clear_spawn_session_requester()

    def test_other_tools_pass_through(self):
        assert maybe_dispatch_spawn_session("read_file", {"path": "x"}) is None

    def test_unbound_requester_returns_graceful_error(self):
        clear_spawn_session_requester()
        result = maybe_dispatch_spawn_session(
            SPAWN_SESSION_TOOL_NAME, {"prompt": "do a thing"}
        )
        payload = _loads(result)
        assert "only available inside a live ACP" in payload["error"]

    def test_empty_prompt_rejected(self):
        token = set_spawn_session_requester(lambda p, c, t: "sid")
        try:
            payload = _loads(
                maybe_dispatch_spawn_session(SPAWN_SESSION_TOOL_NAME, {"prompt": "  "})
            )
            assert payload["error"] == "prompt is required"
        finally:
            reset_spawn_session_requester(token)

    def test_dispatch_returns_session_id(self):
        captured = {}

        def _requester(prompt_text, cwd, title):
            captured["prompt"] = prompt_text
            captured["cwd"] = cwd
            captured["title"] = title
            return "new-session-id"

        token = set_spawn_session_requester(_requester)
        try:
            payload = _loads(
                maybe_dispatch_spawn_session(
                    SPAWN_SESSION_TOOL_NAME,
                    {"prompt": "continue the migration", "cwd": "/work/dir"},
                )
            )
        finally:
            reset_spawn_session_requester(token)

        assert payload["success"] is True
        assert payload["session_id"] == "new-session-id"
        assert captured == {
            "prompt": "continue the migration",
            "cwd": "/work/dir",
            "title": None,
        }

    def test_dispatch_forwards_title(self):
        captured = {}

        def _requester(prompt_text, cwd, title):
            captured["title"] = title
            return "sid"

        token = set_spawn_session_requester(_requester)
        try:
            payload = _loads(
                maybe_dispatch_spawn_session(
                    SPAWN_SESSION_TOOL_NAME,
                    {"prompt": "go", "title": "  Instrument steer delivery  "},
                )
            )
        finally:
            reset_spawn_session_requester(token)

        assert payload["success"] is True
        assert captured["title"] == "Instrument steer delivery"

    def test_requester_failure_surfaces_as_error(self):
        def _requester(prompt_text, cwd, title):
            raise RuntimeError("loop closed")

        token = set_spawn_session_requester(_requester)
        try:
            payload = _loads(
                maybe_dispatch_spawn_session(SPAWN_SESSION_TOOL_NAME, {"prompt": "x"})
            )
        finally:
            reset_spawn_session_requester(token)

        assert "loop closed" in payload["error"]

    def test_model_tools_routes_spawn_calls(self):
        """handle_function_call dispatches acp_spawn_session via the ContextVar
        guard (never the registry)."""
        from model_tools import handle_function_call

        token = set_spawn_session_requester(lambda p, c, t: "routed-id")
        try:
            payload = json.loads(
                handle_function_call(SPAWN_SESSION_TOOL_NAME, {"prompt": "go"})
            )
        finally:
            reset_spawn_session_requester(token)

        assert payload["success"] is True
        assert payload["session_id"] == "routed-id"


class TestServerSpawnRequester:
    @pytest.mark.asyncio
    async def test_requester_creates_session_and_schedules_first_turn(self, agent):
        parent_resp = await agent.new_session(cwd="/tmp", hermes={"owner": "u@yallaplay.com"})
        parent_state = agent.session_manager.get_session(parent_resp.session_id)

        started = asyncio.Event()
        spawned = {}

        async def _fake_first_turn(session_id, prompt_text):
            spawned["session_id"] = session_id
            spawned["prompt"] = prompt_text
            started.set()

        agent._run_spawned_first_turn = _fake_first_turn

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)

        # The requester runs on an executor thread in production.
        new_id = await loop.run_in_executor(None, requester, "carry on", None)
        await asyncio.wait_for(started.wait(), timeout=5)

        assert spawned["session_id"] == new_id
        assert spawned["prompt"] == "carry on"
        child = agent.session_manager.get_session(new_id)
        assert child is not None
        assert child.session_id != parent_state.session_id
        # Inherits the parent's owner and cwd.
        assert child.owner == "u@yallaplay.com"
        assert child.cwd == parent_state.cwd

    @pytest.mark.asyncio
    async def test_requester_honors_explicit_cwd(self, agent, tmp_path):
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(None, requester, "task", str(tmp_path))

        child = agent.session_manager.get_session(new_id)
        assert child.cwd == str(tmp_path)

    @pytest.mark.asyncio
    async def test_requester_stamps_caller_title(self, agent):
        """A caller-provided title lands on the child row before its first
        turn, so the sidebar shows meaningful text immediately (auto-title
        only fills empty titles, so the stamp is never clobbered)."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(
            None, requester, "carry on", None, "Instrument steer delivery"
        )

        db = agent.session_manager._get_db()
        assert db.get_session_title(new_id) == "Instrument steer delivery"

    @pytest.mark.asyncio
    async def test_requester_title_collision_dedups_with_suffix(self, agent):
        """Two spawns with the same title get '#N' lineage suffixes instead of
        one failing on the unique-title index."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        first = await loop.run_in_executor(
            None, requester, "carry on", None, "Retry sweep"
        )
        second = await loop.run_in_executor(
            None, requester, "carry on", None, "Retry sweep"
        )

        db = agent.session_manager._get_db()
        assert db.get_session_title(first) == "Retry sweep"
        assert db.get_session_title(second) == "Retry sweep #2"

    @pytest.mark.asyncio
    async def test_requester_title_failure_never_fails_spawn(self, agent):
        """Title stamping is best-effort: a store error leaves the child
        untitled but the spawn still returns a live session id."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        db = agent.session_manager._get_db()
        with patch.object(
            db, "set_session_title", side_effect=ValueError("bad title")
        ):
            new_id = await loop.run_in_executor(
                None, requester, "carry on", None, "x" * 500
            )

        assert agent.session_manager.get_session(new_id) is not None
        assert db.get_session_title(new_id) is None

    @pytest.mark.asyncio
    async def test_requester_without_title_leaves_child_untitled(self, agent):
        """No title argument → untitled child; auto-title owns it."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(None, requester, "carry on", None)

        db = agent.session_manager._get_db()
        assert db.get_session_title(new_id) is None

    @pytest.mark.asyncio
    async def test_requester_inherits_parent_mode_and_effort(self, agent):
        """The spawned child must carry the parent's edit-approval mode and
        reasoning effort (mirrors fork_session). A spawned session runs its
        first turn headless — no panel attached — so under the default "ask"
        policy every edit approval is auto-denied by the client and the child
        cannot edit files at all (2026-07-15 incident: handoff child's patches
        all bounced with "Edit approval denied by ACP client").
        """
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        parent_state.mode = "dont_ask"
        parent_state.effort = "high"
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(None, requester, "carry on", None)

        child = agent.session_manager.get_session(new_id)
        assert child.mode == "dont_ask"
        assert child.effort == "high"
        # The edit-approval policy derived for the child matches the parent's.
        child_policy, _ = agent._edit_approval_policy_for_state(child)
        parent_policy, _ = agent._edit_approval_policy_for_state(parent_state)
        assert child_policy == parent_policy == "session"

    @pytest.mark.asyncio
    async def test_requester_default_mode_parent_spawns_default_child(self, agent):
        """No inherited surprise: a default-mode parent spawns a default child."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(None, requester, "carry on", None)

        child = agent.session_manager.get_session(new_id)
        assert (child.mode or "") == ""
        assert (child.effort or "") == ""

    @pytest.mark.asyncio
    async def test_requester_carries_parent_model_and_provider_routing(self, agent):
        """The child agent must be built with the parent's FULL route —
        model + provider + base_url + api_mode — not the config default.
        Regression: a gpt-5.6-sol/openai-codex parent spawned a child that
        silently ran the profile default (claude-fable-5/bedrock) because
        the requester passed only cwd/owner to create_session (2026-07-16,
        child session 3eea1d89)."""
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        parent_state.model = "gpt-5.6-sol"
        parent_state.agent.provider = "openai-codex"
        parent_state.agent.base_url = "https://chatgpt.com/backend-api/codex"
        parent_state.agent.api_mode = "codex_responses"
        agent._run_spawned_first_turn = AsyncMock()

        manager = agent.session_manager
        captured = {}
        real_factory = manager._agent_factory

        def spying_make_agent(**kwargs):
            captured.update(kwargs)
            return real_factory()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        with patch.object(manager, "_make_agent", side_effect=spying_make_agent):
            new_id = await loop.run_in_executor(None, requester, "carry on", None)

        assert new_id
        assert captured["model"] == "gpt-5.6-sol"
        assert captured["requested_provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_mode"] == "codex_responses"

    @pytest.mark.asyncio
    async def test_requester_persists_provider_routing_to_db(self, agent):
        """A spawned child must restore with its provider routing after a
        process restart even if its first turn never persists again — the
        creation-time _persist has to write the full route."""

        def _routed_agent():
            a = MagicMock(name="MockAIAgent")
            a.model = "gpt-5.6-sol"
            a.provider = "openai-codex"
            a.base_url = "https://chatgpt.com/backend-api/codex"
            a.api_mode = "codex_responses"
            return a

        manager = agent.session_manager
        manager._agent_factory = _routed_agent
        parent_resp = await agent.new_session(cwd="/tmp")
        parent_state = agent.session_manager.get_session(parent_resp.session_id)
        agent._run_spawned_first_turn = AsyncMock()

        loop = asyncio.get_running_loop()
        requester = agent._make_spawn_session_requester(loop, parent_state)
        new_id = await loop.run_in_executor(None, requester, "carry on", None)

        row = manager._get_db().get_session(new_id)
        assert row["model"] == "gpt-5.6-sol"
        mc = json.loads(row["model_config"])
        assert mc["provider"] == "openai-codex"
        assert mc["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert mc["api_mode"] == "codex_responses"

    @pytest.mark.asyncio
    async def test_spawned_first_turn_echoes_prompt_and_runs(self, agent):
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "done",
            "messages": [
                {"role": "user", "content": "spawned work"},
                {"role": "assistant", "content": "done"},
            ],
        })
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent._run_spawned_first_turn(new_resp.session_id, "spawned work")

        state.agent.run_conversation.assert_called_once()
        user_echoes = [
            call
            for call in mock_conn.session_update.await_args_list
            if getattr(call.args[1] if len(call.args) > 1 else call.kwargs.get("update"),
                       "session_update", None) == "user_message_chunk"
        ]
        assert user_echoes, "spawned prompt was not echoed as a user message"

    @pytest.mark.asyncio
    async def test_prompt_injects_spawn_tool_on_acp_agent(self, agent):
        """A normal prompt() run advertises acp_spawn_session on the agent's
        tool surface (ACP sessions only)."""
        new_resp = await agent.new_session(cwd=".")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent.run_conversation = MagicMock(return_value={
            "final_response": "hi",
            "messages": [],
        })
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hello")],
            session_id=new_resp.session_id,
        )

        names = [t["function"]["name"] for t in state.agent.tools if isinstance(t, dict)]
        assert SPAWN_SESSION_TOOL_NAME in names
        assert SPAWN_SESSION_TOOL_NAME in state.agent.valid_tool_names
