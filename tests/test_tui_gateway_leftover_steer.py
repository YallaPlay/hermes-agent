"""A /steer that lands after the final tool batch is re-delivered, not dropped.

``AIAgent.run_conversation()`` returns any steer it couldn't inject (one that
arrived after the final tool batch — e.g. during the final API call, when the
model answered with text and no further tool calls) in ``result["pending_steer"]``.
The messaging gateway (``gateway/run.py``) and the CLI (``cli.py``) both
re-deliver it as the next user turn. The desktop/web surface
(``tui_gateway/server.py``) previously did NOT — it accepted the steer via
``session.steer`` (``agent.steer`` always accepts non-empty text and reports
"queued"), then silently lost it because its turn tail never read
``pending_steer``. ``_deliver_leftover_steer`` closes that gap.
"""

import threading
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def test_leftover_steer_dispatched_as_next_turn_when_idle(monkeypatch):
    """No queued prompt beat it and the session is idle → fire a fresh turn."""
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session()

    handled = server._deliver_leftover_steer("r1", "sid", session, "also check the logs")

    assert handled is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "also check the logs"}
    # Claimed the session for the dispatched turn.
    assert session["running"] is True


def test_leftover_steer_restashed_when_fresh_turn_already_running(monkeypatch):
    """A real user turn already claimed the session → don't double-fire; the
    steer is handed back to agent.steer() so that turn's own drain picks it up."""
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fire a new turn")),
    )
    steered = {}
    agent = types.SimpleNamespace(steer=lambda t: steered.setdefault("text", t) or True)
    session = _session(agent=agent, running=True)

    handled = server._deliver_leftover_steer("r1", "sid", session, "late nudge")

    assert handled is True
    assert steered == {"text": "late nudge"}
    # It did not start a new turn; the running turn keeps the session.
    assert session["running"] is True


def test_leftover_steer_noop_on_empty_text(monkeypatch):
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fire")),
    )
    session = _session()
    assert server._deliver_leftover_steer("r1", "sid", session, "   ") is False
    assert session["running"] is False


def test_leftover_steer_releases_running_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session()

    handled = server._deliver_leftover_steer("r1", "sid", session, "go")

    assert handled is True
    # Failure must not leave the session wedged as running.
    assert session["running"] is False


def test_restash_survives_agent_without_steer(monkeypatch):
    """If the agent has no steer() (shouldn't happen, but be defensive), the
    handler still reports handled and does not fire a turn."""
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fire")),
    )
    session = _session(agent=types.SimpleNamespace(), running=True)
    assert server._deliver_leftover_steer("r1", "sid", session, "x") is True
