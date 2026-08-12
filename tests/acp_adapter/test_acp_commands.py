import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import ImageContentBlock, TextContentBlock

from acp_adapter.server import (
    INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC,
    HermesACPAgent,
)
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self._supports_active_turn_redirect = True
        self.steers = []
        self.redirects = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def redirect(self, text):
        self.redirects.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        if kwargs:
            self.updates.append((kwargs.get("session_id"), kwargs.get("update")))
        else:
            self.updates.append((args[0], args[1]))

    async def request_permission(self, *args, **kwargs):
        return SimpleNamespace(outcome="allow")


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def make_agent_and_state():
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    captured = {}
    sentinel_db = NoopDb()

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m", "provider": "p"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "p",
                "api_mode": "chat_completions",
                "base_url": "u",
                "api_key": "k",
                "command": None,
                "args": [],
            },
        ),
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, CapturingAgent)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["session_id"] == "acp-session"


@pytest.mark.asyncio
async def test_acp_steer_slash_command_injects_into_running_agent():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer prefer the simpler fix")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == ["prefer the simpler fix"]
    assert fake.runs == []








@pytest.mark.asyncio
async def test_acp_cancel_publishes_hard_stop_while_holding_runtime_lock():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "original request"
    observed = {}

    def interrupt():
        acquired = state.runtime_lock.acquire(blocking=False)
        observed["lock_held"] = not acquired
        if acquired:
            state.runtime_lock.release()

    fake.interrupt = interrupt

    await acp_agent.cancel(state.session_id)

    assert observed["lock_held"] is True
    assert state.cancel_event.is_set()
    assert state.interrupted_prompt_text == "original request"


@pytest.mark.asyncio
async def test_acp_normal_turn_clears_stale_interrupted_prompt():
    """A completed normal turn must not leave a salvageable prompt behind.

    Regression: interrupted_prompt_text is set only on a running-cancel and
    cleared only inside the salvage paths. If the user cancels a running turn,
    then sends an ordinary prompt (which runs to completion), the interrupted
    prompt was never cleared — so a later correction on the idle session would
    resurrect and re-run the task the user cancelled and moved on from.
    Starting any real turn must drop the stale salvage buffer.

    Adopted from upstream PR NousResearch/hermes-agent#56624.
    """
    acp_agent, state, fake, _conn = make_agent_and_state()
    # Left over from a prior running-cancel of "refactor the auth module".
    state.interrupted_prompt_text = "refactor the auth module"

    # An ordinary prompt runs to completion in between.
    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="what's the weather")],
    )
    assert state.interrupted_prompt_text == ""
    assert fake.runs == ["what's the weather"]

    # Now a follow-up on the idle session must run ONLY the new text, not the
    # abandoned prompt.
    fake.runs.clear()
    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer be concise")],
    )
    assert fake.runs == ["be concise"]


@pytest.mark.asyncio
async def test_acp_stale_interrupted_prompt_is_not_salvaged_after_window():
    """A cancelled prompt is salvageable only briefly, not indefinitely.

    Regression (live incident, 2026-08-12): a turn was cancelled, its prompt
    armed the salvage buffer, and FOUR HOURS later an unrelated one-line
    request from a different user was merged behind it. The agent read the
    abandoned request as the instruction and rebuilt work nobody asked for.
    """
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.interrupted_prompt_text = "make a clone of the reference sheet and fill it"
    # Armed four hours ago, as in the live incident. Deliberately a fixed
    # absolute age, NOT derived from the window constant — a test that moves
    # with the constant can't detect the constant being widened.
    state.interrupted_prompt_at = time.monotonic() - 4 * 60 * 60
    assert 4 * 60 * 60 > INTERRUPTED_PROMPT_SALVAGE_WINDOW_SEC

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="give me editor access to both sheets")],
    )

    assert fake.runs == ["give me editor access to both sheets"]
    assert "clone of the reference sheet" not in fake.runs[0]
    # The dead buffer is dropped, not left to ambush a later prompt.
    assert state.interrupted_prompt_text == ""
    assert state.interrupted_prompt_at == 0.0


@pytest.mark.asyncio
async def test_acp_fresh_interrupted_prompt_is_salvaged_with_new_message_leading():
    """Legitimate "stop and send" still salvages — but the NEW text leads.

    The interrupted request must survive as referenceable context (so deictic
    corrections resolve), while the live message stays the instruction.
    """
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.interrupted_prompt_text = "refactor the auth module"
    state.interrupted_prompt_at = time.monotonic()

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="not that file, do the login one")],
    )

    assert len(fake.runs) == 1
    merged = fake.runs[0]
    # New instruction leads; interrupted request survives below it as context.
    assert merged.startswith("not that file, do the login one")
    assert "refactor the auth module" in merged
    assert merged.index("not that file") < merged.index("refactor the auth module")
    assert "NOT a new instruction" in merged


@pytest.mark.asyncio
async def test_acp_cancel_after_response_delivered_does_not_arm_salvage():
    """A cancel landing after the turn answered must not arm the buffer.

    The final response is delivered inside the post-turn tail, but is_running
    stays True until the finally block. A cancel in that window used to store
    an already-fulfilled prompt, which the next unrelated prompt then replayed.
    """
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "build the sheets"
    state.response_delivered = True  # turn already produced and sent its answer

    await acp_agent.cancel(state.session_id)

    assert state.interrupted_prompt_text == ""
    assert state.interrupted_prompt_at == 0.0
    assert state.cancel_event.is_set()


@pytest.mark.asyncio
async def test_acp_completed_turn_marks_response_delivered():
    """The delivered flag is actually set by a normal completed turn."""
    acp_agent, state, fake, _conn = make_agent_and_state()

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="hello")],
    )

    assert state.response_delivered is True
    # And a cancel arriving now (turn idle) still arms nothing.
    await acp_agent.cancel(state.session_id)
    assert state.interrupted_prompt_text == ""


@pytest.mark.asyncio
async def test_acp_multimodal_turn_clears_fresh_interrupted_prompt():
    """A turn that skips the salvage branches must still drop the buffer.

    Multimodal prompts bypass both text-only salvage paths, so without the
    turn-start clear a FRESH buffer would survive that turn and be picked up
    by the next text prompt — the user having visibly moved on twice.
    This pins the turn-start clear adopted from upstream PR #56624.
    """
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.interrupted_prompt_text = "refactor the auth module"
    state.interrupted_prompt_at = time.monotonic()  # fresh, inside the window

    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[
            TextContentBlock(type="text", text="what is in this image?"),
            ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png"),
        ],
    )

    assert state.interrupted_prompt_text == ""
    assert state.interrupted_prompt_at == 0.0

    # The following text prompt therefore runs clean.
    fake.runs.clear()
    await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="thanks, now check the logs")],
    )
    assert fake.runs == ["thanks, now check the logs"]






