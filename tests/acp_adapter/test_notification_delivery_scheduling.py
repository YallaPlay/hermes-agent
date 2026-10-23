"""Per-session scheduling of ACP notification deliveries.

Regression for the watcher head-of-line stall: ``_drain_notification_cycle``
used to ``await`` each delivery turn inline, so one session's minutes-long
synthetic turn blocked every other session's completion from even being
claimed. Deliveries are now scheduled as one task per target session:

- a completion for idle session B is delivered while session A's delivery
  turn is still running;
- two events for one session stay ordered and are each delivered exactly
  once (the second is requeued, unclaimed, until the first turn ends);
- a delivery task that raises neither kills the watcher nor blocks other
  sessions;
- cancelling the watcher cancels the delivery tasks it owns.

Everything runs in memory: mock session manager, mock ACP client, patched
``process_registry.drain_notifications``, mocked ``prompt``. The E2E case at
the bottom (real registry queue, real drain, real ``prompt()``; only
``run_conversation`` mocked) also lives here.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionState
from tests.waiting import async_wait_until


def _insert_state(manager, sid, is_running=False):
    state = SessionState(
        session_id=sid,
        agent=MagicMock(name="MockAIAgent"),
        cwd="/tmp",
        is_running=is_running,
    )
    with manager._lock:
        manager._sessions[sid] = state
    return state


def _pop_requeued(evt):
    """True iff evt sits on the real completion_queue; removes it and
    restores every foreign item so parallel tests are unaffected."""
    from tools.process_registry import process_registry

    found = False
    others = []
    while not process_registry.completion_queue.empty():
        try:
            item = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if item is evt:
            found = True
        else:
            others.append(item)
    for item in others:
        process_registry.completion_queue.put(item)
    return found


@pytest.fixture()
def agent(monkeypatch):
    manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    agent = HermesACPAgent(session_manager=manager)
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    agent._conn = conn
    monkeypatch.setattr(type(agent), "_NOTIFY_DRAIN_INTERVAL_SEC", 0.01, raising=True)
    return agent


async def _stop_watcher(agent):
    task = agent._notification_watcher_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    await agent._cancel_notification_deliveries()


class _BlockingPrompt:
    """Fake ``prompt`` whose turn for ``blocked_session`` runs until released,
    flipping ``is_running`` like the real one does."""

    def __init__(self, manager, blocked_session):
        self.manager = manager
        self.blocked_session = blocked_session
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.prompted = []

    async def __call__(self, prompt, session_id, **kw):
        self.prompted.append(session_id)
        state = self.manager.get_in_memory(session_id)
        state.is_running = True
        try:
            if session_id == self.blocked_session:
                self.started.set()
                await self.release.wait()
        finally:
            state.is_running = False


@pytest.mark.asyncio
async def test_idle_session_delivered_while_other_delivery_turn_runs(agent):
    """The inverted head-of-line probe: B must be delivered while A's
    delivery turn is still running, not after it."""
    manager = agent.session_manager
    _insert_state(manager, "sess-A")
    _insert_state(manager, "sess-B")
    fake = _BlockingPrompt(manager, "sess-A")
    agent.prompt = fake

    evt_a = {"type": "completion", "session_key": "sess-A"}
    evt_b = {"type": "completion", "session_key": "sess-B"}
    batches = [[(evt_a, "[done A]")], [(evt_b, "[done B]")]]

    def fake_drain(owns_event):
        return batches.pop(0) if batches else []

    from tools.process_registry import process_registry

    with patch.object(process_registry, "drain_notifications", side_effect=fake_drain):
        agent._ensure_notification_watcher()
        try:
            await asyncio.wait_for(fake.started.wait(), 5)
            await async_wait_until(
                lambda: "sess-B" in fake.prompted,
                timeout=5,
                what="B delivered while A's delivery turn is running",
            )
            assert not fake.release.is_set()
            assert manager.get_in_memory("sess-A").is_running is True
            assert fake.prompted == ["sess-A", "sess-B"]
            assert agent._notification_delivery_in_flight("sess-A")
            fake.release.set()
            await async_wait_until(
                lambda: not agent._notification_delivery_in_flight("sess-A"),
                timeout=5,
                what="A's delivery task to finish",
            )
            assert "sess-A" not in agent._notification_delivery_tasks
        finally:
            fake.release.set()
            await _stop_watcher(agent)


@pytest.mark.asyncio
async def test_drain_cycle_returns_before_delivery_turn_ends(agent):
    """The cycle itself must not await the delivery turn."""
    manager = agent.session_manager
    _insert_state(manager, "sess-slow")
    fake = _BlockingPrompt(manager, "sess-slow")
    agent.prompt = fake
    evt = {"type": "completion", "session_key": "sess-slow"}
    from tools.process_registry import process_registry

    try:
        with patch.object(
            process_registry, "drain_notifications", return_value=[(evt, "[done]")]
        ):
            await asyncio.wait_for(agent._drain_notification_cycle(), timeout=5)
        assert agent._notification_delivery_in_flight("sess-slow")
        await asyncio.wait_for(fake.started.wait(), 5)
        assert not fake.release.is_set()
    finally:
        fake.release.set()
        await _stop_watcher(agent)


@pytest.mark.asyncio
async def test_same_session_events_stay_ordered_and_delivered_once(agent):
    """Two delegation events for one session: the second is requeued
    WITHOUT claiming while the first delivery is in flight (even before
    its prompt() has flipped is_running), then delivered exactly once,
    after the first, with one claim per event."""
    manager = agent.session_manager
    _insert_state(manager, "sess-same")
    fake = _BlockingPrompt(manager, "sess-same")
    agent.prompt = fake
    evt_1 = {"type": "async_delegation", "session_key": "sess-same", "id": 1}
    evt_2 = {"type": "async_delegation", "session_key": "sess-same", "id": 2}
    claims = []
    completes = []
    from tools.process_registry import process_registry

    try:
        with patch(
            "tools.async_delegation.claim_event_delivery",
            side_effect=lambda evt, consumer: claims.append(evt["id"]) or f"claim-{evt['id']}",
        ), patch(
            "tools.async_delegation.complete_event_delivery",
            side_effect=lambda evt, claim: completes.append((evt["id"], claim)),
        ):
            # Both events in ONE batch: the first is scheduled (task created
            # but not yet started), the second must already count as busy.
            with patch.object(
                process_registry,
                "drain_notifications",
                return_value=[(evt_1, "[deleg 1]"), (evt_2, "[deleg 2]")],
            ):
                await asyncio.wait_for(agent._drain_notification_cycle(), timeout=5)
            assert claims == [1]
            assert completes == [(1, "claim-1")]
            assert _pop_requeued(evt_2), "second same-session event was not requeued"
            await asyncio.wait_for(fake.started.wait(), 5)
            assert fake.prompted == ["sess-same"]

            # While the first delivery turn runs, a retry cycle still requeues.
            with patch.object(
                process_registry,
                "drain_notifications",
                return_value=[(evt_2, "[deleg 2]")],
            ):
                await asyncio.wait_for(agent._drain_notification_cycle(), timeout=5)
            assert claims == [1]
            assert _pop_requeued(evt_2)

            # First turn ends → next cycle claims and delivers the second.
            fake.release.set()
            await async_wait_until(
                lambda: not agent._notification_delivery_in_flight("sess-same"),
                timeout=5,
            )
            with patch.object(
                process_registry,
                "drain_notifications",
                return_value=[(evt_2, "[deleg 2]")],
            ):
                await asyncio.wait_for(agent._drain_notification_cycle(), timeout=5)
            await async_wait_until(
                lambda: not agent._notification_delivery_in_flight("sess-same"),
                timeout=5,
            )
            assert claims == [1, 2]
            assert completes == [(1, "claim-1"), (2, "claim-2")]
            assert fake.prompted == ["sess-same", "sess-same"]
            assert not _pop_requeued(evt_2)
    finally:
        fake.release.set()
        _pop_requeued(evt_1)
        _pop_requeued(evt_2)
        await _stop_watcher(agent)


@pytest.mark.asyncio
async def test_delivery_task_exception_is_isolated(agent, caplog):
    """A raising delivery for A: logged, task entry dropped, watcher alive,
    B still delivered."""
    manager = agent.session_manager
    _insert_state(manager, "sess-boom")
    _insert_state(manager, "sess-ok")
    delivered = []

    async def deliver(session_id, text):
        if session_id == "sess-boom":
            raise RuntimeError("delivery exploded")
        delivered.append(session_id)

    agent._deliver_notification = deliver
    evt_boom = {"type": "completion", "session_key": "sess-boom"}
    evt_ok = {"type": "completion", "session_key": "sess-ok"}
    batches = [[(evt_boom, "[boom]"), (evt_ok, "[ok]")]]
    cycles = {"n": 0}

    def fake_drain(owns_event):
        cycles["n"] += 1
        return batches.pop(0) if batches else []

    from tools.process_registry import process_registry

    with patch.object(process_registry, "drain_notifications", side_effect=fake_drain):
        agent._ensure_notification_watcher()
        try:
            await async_wait_until(lambda: delivered == ["sess-ok"], timeout=5)
            await async_wait_until(
                lambda: "sess-boom" not in agent._notification_delivery_tasks, timeout=5
            )
            first = cycles["n"]
            await async_wait_until(lambda: cycles["n"] > first, timeout=5, what="watcher next cycle")
            assert not agent._notification_watcher_task.done()
            assert any(
                "delivery task for sess-boom raised" in rec.getMessage()
                for rec in caplog.records
            )
        finally:
            await _stop_watcher(agent)


@pytest.mark.asyncio
async def test_cancelling_watcher_cancels_in_flight_deliveries(agent):
    manager = agent.session_manager
    _insert_state(manager, "sess-cancel")
    fake = _BlockingPrompt(manager, "sess-cancel")
    agent.prompt = fake
    evt = {"type": "completion", "session_key": "sess-cancel"}
    batches = [[(evt, "[done]")]]

    def fake_drain(owns_event):
        return batches.pop(0) if batches else []

    from tools.process_registry import process_registry

    with patch.object(process_registry, "drain_notifications", side_effect=fake_drain):
        agent._ensure_notification_watcher()
        await asyncio.wait_for(fake.started.wait(), 5)
        delivery = agent._notification_delivery_tasks["sess-cancel"]
        watcher = agent._notification_watcher_task
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(watcher, timeout=5)
        assert delivery.cancelled()
        assert agent._notification_delivery_tasks == {}
        # The fake prompt's finally ran: the session is not left busy.
        assert manager.get_in_memory("sess-cancel").is_running is False


class TestNotificationE2E:
    """End-to-end: a REAL completion event on the REAL registry queue flows
    through the REAL drain cycle (ownership routing + formatting) into a
    REAL synthetic prompt turn. Only ``run_conversation`` — the executor
    boundary — is mocked, matching the file's established pattern.
    """

    @pytest.mark.asyncio
    async def test_completion_event_drains_into_synthetic_turn(self, agent):
        import time

        from tools.process_registry import process_registry

        proc_sid = "proc_test123"  # unique — drain skip-state can't collide

        # HYGIENE: completion_queue is process-global shared state. Stash
        # everything currently queued (ours or leaked from parallel tests)
        # and restore it in teardown so no foreign event is consumed.
        stashed = []
        while not process_registry.completion_queue.empty():
            try:
                stashed.append(process_registry.completion_queue.get_nowait())
            except Exception:
                break
        # Clean consumed/observed skip-state for our unique proc id so the
        # real drain's _drain_should_skip cannot spuriously eat the event.
        process_registry._completion_consumed.discard(proc_sid)
        process_registry._poll_observed.discard(proc_sid)

        try:
            # Real agent + real in-memory idle session.
            new_resp = await agent.new_session(cwd=".")
            state = agent.session_manager.get_session(new_resp.session_id)

            captured = {}

            def _run(**kwargs):
                captured.update(kwargs)
                return {"final_response": "ok", "messages": []}

            state.agent.run_conversation = MagicMock(side_effect=_run)

            mock_conn = MagicMock(spec=acp.Client)
            mock_conn.session_update = AsyncMock()
            agent._conn = mock_conn

            # A REAL completion event, shaped exactly like
            # ProcessRegistry._move_to_finished produces (no drain mock —
            # the real registry drain + formatter run below).
            evt = {
                "type": "completion",
                "session_id": proc_sid,
                "session_key": new_resp.session_id,
                "command": "echo hello world",
                "exit_code": 0,
                "completion_reason": "exited",
                "termination_source": "",
                "output": "hello world",
                "started_at": time.time(),
            }
            process_registry.completion_queue.put(evt)

            await asyncio.wait_for(
                agent._drain_notification_cycle(), timeout=10
            )
            # The cycle only schedules the delivery; wait for the turn itself.
            await asyncio.wait_for(
                agent._notification_delivery_tasks[new_resp.session_id], timeout=10
            )

            # The chain ran end-to-end: event → drain → ownership →
            # format → synthetic turn on OUR session.
            state.agent.run_conversation.assert_called_once()
            user_text = captured["user_message"]
            assert "[IMPORTANT:" in user_text
            assert proc_sid in user_text
            assert "hello world" in user_text
            # The synthetic turn fully completed — session idle again.
            assert state.is_running is False
        finally:
            # The delivery turn goes through prompt(), which may start the
            # real watcher task — never leak it into other tests.
            await _stop_watcher(agent)
            # Drop any leftover copies of OUR event, then restore every
            # stashed foreign event for the tests that own them.
            while not process_registry.completion_queue.empty():
                try:
                    item = process_registry.completion_queue.get_nowait()
                except Exception:
                    break
                if not (
                    isinstance(item, dict)
                    and item.get("session_id") == proc_sid
                ):
                    stashed.append(item)
            for item in stashed:
                process_registry.completion_queue.put(item)
            process_registry._completion_consumed.discard(proc_sid)
            process_registry._poll_observed.discard(proc_sid)
