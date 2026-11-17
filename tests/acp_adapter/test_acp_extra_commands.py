"""In-process ACP slash commands beyond the original nine: /goal and its turn loop, /subgoal,
skills, /retry, /undo, /title, /status, /usage, prompt builders, aliases, and the registry fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
from acp.schema import TextContentBlock

from acp_adapter.commands import SlashReply
from tests.acp_adapter.test_acp_commands import make_agent_and_state


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _texts(conn, kind="agent_message_chunk"):
    out = []
    for _sid, update in conn.updates:
        if getattr(update, "session_update", None) == kind:
            out.append(update.content.text)
    return out


async def _send(acp_agent, state, text):
    return await acp_agent.prompt(session_id=state.session_id, prompt=[TextContentBlock(type="text", text=text)])


def _scripted_judge(monkeypatch, decisions):
    """Replace the goal judge with a script of evaluate_after_turn decisions; returns the inputs."""
    from hermes_cli.goals import GoalManager

    judged = []

    def evaluate(self, last_response, **_kwargs):
        judged.append(last_response)
        return decisions.pop(0) if decisions else {"message": "", "should_continue": False}

    monkeypatch.setattr(GoalManager, "evaluate_after_turn", evaluate)
    return judged


# ---- menu --------------------------------------------------------------------------------------

def test_menu_advertises_new_commands():
    acp_agent, _state, _fake, _conn = make_agent_and_state()
    names = {c.name for c in acp_agent._available_commands()}
    assert {"goal", "subgoal", "usage", "status", "title", "undo", "retry", "plan", "learn", "init"} <= names


def test_menu_lists_skills_without_shadowing_builtins(monkeypatch):
    import agent.skill_commands as sc

    monkeypatch.setattr(sc, "get_interactive_skill_commands", lambda: {
        "/my-skill": {"name": "my-skill", "description": "Does a thing"},
        "/plan": {"name": "plan", "description": "a skill that collides"},
    })
    acp_agent, _state, _fake, _conn = make_agent_and_state()
    commands = acp_agent._available_commands()
    assert [c.name for c in commands].count("plan") == 1
    skill = next(c for c in commands if c.name == "my-skill")
    assert skill.description == "Does a thing"


def test_gateway_menu_advertises_goal():
    from acp_adapter.gateway_session_meta import available_commands_update

    names = {c.name for c in available_commands_update().available_commands}
    assert {"goal", "subgoal", "plan", "learn", "init"} <= names


# ---- /goal and the loop --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_goal_kickoff_runs_goal_and_continues_until_judge_stops(monkeypatch):
    acp_agent, state, fake, conn = make_agent_and_state()
    judged = _scripted_judge(monkeypatch, [
        {"message": "↻ Continuing toward goal (1/20)", "should_continue": True,
         "continuation_prompt": "[Continuing toward your standing goal]"},
        {"message": "✓ Goal achieved", "should_continue": False},
    ])

    await _send(acp_agent, state, "/goal ship the feature")

    assert len(fake.runs) == 2
    assert "ship the feature" in fake.runs[0]
    assert fake.runs[1] == "[Continuing toward your standing goal]"
    assert len(judged) == 2
    agent_text = "\n".join(_texts(conn))
    assert "Goal set" in agent_text or "goal" in agent_text.lower()
    assert "✓ Goal achieved" in agent_text
    # The continuation is shown as a user message so the transcript reads naturally.
    assert "[Continuing toward your standing goal]" in _texts(conn, "user_message_chunk")


@pytest.mark.asyncio
async def test_goal_status_and_clear_do_not_run_a_turn(monkeypatch):
    acp_agent, state, fake, conn = make_agent_and_state()
    _scripted_judge(monkeypatch, [])
    await _send(acp_agent, state, "/goal write the docs")
    runs = len(fake.runs)

    await _send(acp_agent, state, "/goal status")
    await _send(acp_agent, state, "/goal clear")
    await _send(acp_agent, state, "/goal status")

    assert len(fake.runs) == runs
    assert "write the docs" in _texts(conn)[-3]
    assert "No active goal" in _texts(conn)[-1] or "No goal" in _texts(conn)[-1]


@pytest.mark.asyncio
async def test_no_goal_means_no_judge_call(monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    judged = _scripted_judge(monkeypatch, [])
    await _send(acp_agent, state, "plain request")
    assert fake.runs == ["plain request"]
    assert judged == []


@pytest.mark.asyncio
async def test_queued_user_prompt_wins_over_goal_continuation(monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    _scripted_judge(monkeypatch, [
        {"message": "", "should_continue": True, "continuation_prompt": "STALE CONTINUATION"},
    ])
    original = fake.run_conversation

    def run(**kwargs):
        result = original(**kwargs)
        if len(fake.runs) == 1:
            state.queued_prompts.append("typed while busy")
        return result

    fake.run_conversation = run
    await _send(acp_agent, state, "/goal finish it")

    assert fake.runs[1] == "typed while busy"
    assert "STALE CONTINUATION" not in fake.runs


@pytest.mark.asyncio
async def test_failed_turn_is_not_judged(monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    judged = _scripted_judge(monkeypatch, [])

    def failing(**kwargs):
        fake.runs.append(kwargs["user_message"])
        return {"final_response": "provider exploded", "failed": True, "messages": []}

    fake.run_conversation = failing
    await _send(acp_agent, state, "/goal finish it")
    assert judged == []


@pytest.mark.asyncio
async def test_goal_judge_error_never_fails_the_turn(monkeypatch):
    from hermes_cli.goals import GoalManager

    acp_agent, state, fake, _conn = make_agent_and_state()

    def boom(self, *_a, **_k):
        raise RuntimeError("judge down")

    monkeypatch.setattr(GoalManager, "evaluate_after_turn", boom)
    response = await _send(acp_agent, state, "/goal finish it")
    assert response.stop_reason == "end_turn"
    assert len(fake.runs) == 1


def test_goal_follows_agent_session_id_after_compression():
    from acp_adapter.commands import _goal_session_key

    _acp_agent, state, fake, _conn = make_agent_and_state()
    fake.session_id = "rotated-child-id"
    assert _goal_session_key(state) == "rotated-child-id"
    fake.session_id = None
    assert _goal_session_key(state) == state.session_id


# ---- /subgoal ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subgoal_add_list_remove(monkeypatch):
    acp_agent, state, _fake, conn = make_agent_and_state()
    _scripted_judge(monkeypatch, [])

    await _send(acp_agent, state, "/subgoal tests pass")
    assert "No goal set" in _texts(conn)[-1]

    await _send(acp_agent, state, "/goal ship it")
    await _send(acp_agent, state, "/subgoal tests pass")
    assert "Added subgoal 1" in _texts(conn)[-1]
    await _send(acp_agent, state, "/subgoal")
    assert "tests pass" in _texts(conn)[-1]
    await _send(acp_agent, state, "/subgoal remove 1")
    assert "Removed subgoal 1" in _texts(conn)[-1]
    await _send(acp_agent, state, "/subgoal remove x")
    assert "needs a number" in _texts(conn)[-1]


# ---- skills ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skill_command_runs_skill_prompt(monkeypatch):
    import agent.skill_commands as sc

    monkeypatch.setattr(sc, "get_interactive_skill_commands",
                        lambda: {"/my-skill": {"name": "my-skill", "description": "d"}})
    monkeypatch.setattr(sc, "split_stacked_skill_commands", lambda args, interactive=True: ([], args))
    monkeypatch.setattr(sc, "build_skill_invocation_message",
                        lambda key, instruction, task_id=None: f"SKILL BODY for {key}: {instruction}")
    acp_agent, state, fake, conn = make_agent_and_state()

    await _send(acp_agent, state, "/my-skill do the thing")

    assert fake.runs == ["SKILL BODY for /my-skill: do the thing"]
    assert any("Loading skill" in text for text in _texts(conn))


# ---- /retry, /undo -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_resends_last_user_message_once():
    acp_agent, state, fake, _conn = make_agent_and_state()
    await _send(acp_agent, state, "first")
    await _send(acp_agent, state, "second")

    await _send(acp_agent, state, "/retry")

    assert fake.runs == ["first", "second", "second"]
    users = [m["content"] for m in state.history if m["role"] == "user"]
    assert users == ["first", "second"]


@pytest.mark.asyncio
async def test_undo_removes_last_turn_without_running():
    acp_agent, state, fake, conn = make_agent_and_state()
    await _send(acp_agent, state, "first")
    await _send(acp_agent, state, "second")

    await _send(acp_agent, state, "/undo")

    assert fake.runs == ["first", "second"]
    assert [m["content"] for m in state.history if m["role"] == "user"] == ["first"]
    assert "Removed message: second" in _texts(conn)[-1]


@pytest.mark.asyncio
async def test_undo_and_retry_refused_mid_turn():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    state.is_running = True
    for text in ("/undo", "/retry"):
        await _send(acp_agent, state, text)
    assert len(state.history) == 2
    assert fake.runs == []


@pytest.mark.asyncio
async def test_undo_with_nothing_to_undo():
    acp_agent, state, _fake, conn = make_agent_and_state()
    await _send(acp_agent, state, "/undo")
    assert "Nothing to undo" in _texts(conn)[-1]
    await _send(acp_agent, state, "/undo abc")
    assert "Usage: /undo" in _texts(conn)[-1]


def test_retry_strips_first_turn_owner_note():
    from acp_adapter.commands import _strip_owner_note
    from hermes_constants import ACP_OWNER_NOTE_PREFIX

    assert _strip_owner_note(f"{ACP_OWNER_NOTE_PREFIX} someone]\n\nreal question") == "real question"
    assert _strip_owner_note("plain") == "plain"


# ---- builders, reads, aliases, fallback ---------------------------------------------------

@pytest.mark.asyncio
async def test_plan_runs_plan_mode_prompt():
    acp_agent, state, fake, _conn = make_agent_and_state()
    await _send(acp_agent, state, "/plan add caching")
    assert len(fake.runs) == 1
    assert fake.runs[0].startswith("[/plan")
    assert "add caching" in fake.runs[0]


def test_status_and_usage_text():
    acp_agent, state, _fake, _conn = make_agent_and_state()
    status = acp_agent._handle_slash_command("/status", state)
    assert f"Session: {state.session_id}" in status
    usage = acp_agent._handle_slash_command("/usage", state)
    assert "Total tokens" in usage or "No active agent" in usage


def test_title_reply_requests_session_info_refresh(monkeypatch):
    acp_agent, state, _fake, _conn = make_agent_and_state()
    titles = {}
    monkeypatch.setattr(acp_agent.session_manager, "set_session_title",
                        lambda sid, title: titles.__setitem__(sid, title) or True)
    monkeypatch.setattr(acp_agent.session_manager, "get_session_title", lambda sid: titles.get(sid))
    reply = acp_agent._handle_slash_command("/title My Work", state)
    assert isinstance(reply, SlashReply) and reply.session_info_changed
    assert titles[state.session_id] == "My Work"
    assert acp_agent._handle_slash_command("/title", state) == "Session title: My Work"


def test_aliases_route_to_builtins():
    acp_agent, state, _fake, _conn = make_agent_and_state()
    state.history = [{"role": "user", "content": "x"}]
    acp_agent._handle_slash_command("/new", state)
    assert state.history == []
    assert "Queued" in (acp_agent._handle_slash_command("/q later", state) or "")


def test_registry_fallback_runs_shared_executor_or_refuses():
    acp_agent, state, _fake, _conn = make_agent_and_state()
    profile = acp_agent._handle_slash_command("/profile", state)
    assert profile and "not available" not in profile
    refused = acp_agent._handle_slash_command("/yolo", state)
    assert refused.startswith("/yolo is not available in the editor")
    assert "session list" in acp_agent._handle_slash_command("/resume", state)


def test_non_registry_slash_text_still_reaches_model():
    acp_agent, state, _fake, _conn = make_agent_and_state()
    assert acp_agent._handle_slash_command("/tmp/foo is broken", state) is None
    assert acp_agent._handle_slash_command("/definitely-not-a-command", state) is None
