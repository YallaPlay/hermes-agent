"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import uuid
import warnings
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate

from acp_adapter.events import (
    _send_update,
    make_message_cb,
    make_step_cb,
    make_subagent_update_router,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:



    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids

    def test_completes_tool_live_on_tool_completed(self, mock_conn, event_loop_fixture):
        """A tool.completed event emits the ACP completion immediately.

        The step callback only reports a tool's result on the *next* step, so
        the last tool of a turn never got a "completed" update and the client
        spun forever. tool.completed must clear it live, correlated by the FIFO
        queue and carrying the result.
        """
        from collections import deque

        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "read_file", "Reading /etc/hosts", {"path": "/etc/hosts"})
            assert len(tool_call_ids["read_file"]) == 1
            tc_id = tool_call_ids["read_file"][0]

            cb("tool.completed", "read_file", None, None, result="file contents")

        # Completion built with the started id, args, and the live result.
        mock_btc.assert_called_once_with(
            tc_id, "read_file", result="file contents", function_args={"path": "/etc/hosts"}, snapshot=None,
            is_error=False
        )
        # The id is consumed so the step callback won't double-complete it.
        assert "read_file" not in tool_call_ids

    def test_tool_completed_is_noop_in_step_after_live_completion(self, mock_conn, event_loop_fixture):
        """After a live completion, the step callback has nothing left to pop."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.completed", "terminal", None, None, result="ok")
            assert mock_btc.call_count == 1
            # The next step sees the id already consumed → no second completion.
            step_cb(1, [{"name": "terminal", "result": "ok"}])
            assert mock_btc.call_count == 1


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:



    @pytest.mark.parametrize("raw, expected", [("", ""), (0, "0"), (False, "False")])
    def test_falsey_result_reaches_client_unchanged(self, mock_conn, event_loop_fixture, raw, expected):
        """A present-but-falsey ``result`` is the tool's real output, not a missing key (#10845)."""
        from collections import deque

        cb = make_step_cb(mock_conn, "session-1", event_loop_fixture, {"terminal": deque(["tc-f"])}, {})
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            mock_rcts.return_value = MagicMock(spec=Future)
            cb(1, [{"name": "terminal", "result": raw}])
        mock_btc.assert_called_once_with("tc-f", "terminal", result=expected, function_args=None, snapshot=None)




    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot"):
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_send.assert_called_once()

    @pytest.mark.parametrize("name", ["todo", "todo_list"])
    @pytest.mark.parametrize("live_completion", [False, True])
    def test_todo_plan_emits_once_live_or_on_step_fallback(
        self, mock_conn, event_loop_fixture, name, live_completion,
    ):
        """Live completion sends the last plan immediately; older runtimes use the step fallback."""
        tool_call_ids, tool_call_meta, turn_state = {}, {}, {}
        loop = event_loop_fixture
        progress = make_tool_progress_cb(
            mock_conn, "session-1", loop, tool_call_ids, tool_call_meta, turn_state=turn_state,
        )
        step = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta, turn_state)
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"}'
            '],"summary":{"total":1}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            progress("tool.started", name, None, {})
            mock_send.reset_mock()
            if live_completion:
                progress("tool.completed", name, result=todo_result)
                live_updates = [call.args[3].session_update for call in mock_send.call_args_list]
                assert live_updates == ["plan", "tool_call_update"]
                mock_send.reset_mock()
            step(1, [{"name": name, "result": todo_result}])

        updates = [call.args[3].session_update for call in mock_send.call_args_list]
        assert updates == ([] if live_completion else ["tool_call_update", "plan"])
        assert name not in tool_call_ids

    def test_todo_completion_emits_native_plan_update_live(self, mock_conn, event_loop_fixture):
        """A live ``tool.completed`` event for ``todo`` emits the native ACP plan
        update immediately, covering the final todo of a turn."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture
        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"},'
            '{"id":"patch","content":"Patch renderer","status":"in_progress"},'
            '{"id":"old","content":"Drop stale task","status":"cancelled"}'
            '],"summary":{"total":3}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb("tool.completed", "todo", None, None, result=todo_result)

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == ["plan"]
        plan = updates[0]
        assert isinstance(plan, AgentPlanUpdate)
        assert [entry.content for entry in plan.entries] == [
            "Inspect ACP",
            "Patch renderer",
            "[cancelled] Drop stale task",
        ]
        assert [entry.status for entry in plan.entries] == ["completed", "in_progress", "completed"]
        assert [entry.priority for entry in plan.entries] == ["medium", "medium", "medium"]

    def test_non_todo_completion_emits_nothing(self, mock_conn, event_loop_fixture):
        """A ``tool.completed`` for a non-todo tool must not emit a plan update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture
        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events._send_update") as mock_send:
            cb("tool.completed", "terminal", None, None, result="done")

        mock_send.assert_not_called()

    def test_todo_plan_update_parses_json_with_trailing_hint(self):
        result = '{"todos":[{"id":"ship","content":"Ship ACP plan","status":"pending"}]}\n\n[Hint: persisted]'



# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Subagent update router
# ---------------------------------------------------------------------------


class TestSubagentUpdateRouter:
    def _make_router(self, mock_conn, loop):
        return make_subagent_update_router(mock_conn, loop)

    def test_full_lifecycle_routes_updates_to_child_session(self, mock_conn, event_loop_fixture):
        """start → tool → tool_completed → complete emit updates on the CHILD id."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.start", None, "summarize README",
                   child_session_id="child-1", goal="summarize README")
            router("subagent.tool", "terminal", "$ ls", {"command": "ls"},
                   child_session_id="child-1")
            router("subagent.tool_completed", "terminal", None,
                   child_session_id="child-1", result="ok")
            router("subagent.complete", None, "done", child_session_id="child-1")

        session_ids = [call.args[1] for call in mock_send.call_args_list]
        assert set(session_ids) == {"child-1"}
        updates = [call.args[3] for call in mock_send.call_args_list]
        kinds = [getattr(u, "session_update", None) for u in updates]
        assert kinds == [
            "session_info_update",  # start → isRunning true
            "tool_call",            # tool start
            "tool_call_update",     # tool complete
            "session_info_update",  # complete → isRunning false
        ]
        start_info = updates[0]
        assert start_info.field_meta["hermes"]["isRunning"] is True
        assert start_info.field_meta["hermes"]["currentPromptText"] == "summarize README"
        end_info = updates[-1]
        assert end_info.field_meta["hermes"]["isRunning"] is False
        # A delegate child must NEVER be adoptable by a durable-turn proxy:
        # the delegate tool owns the child's row and transcript, and the
        # child's frames are relay-only by design. Only the server's own
        # _send_turn_status_update sets the flag.
        assert "adoptableTurn" not in start_info.field_meta["hermes"]
        assert "adoptableTurn" not in end_info.field_meta["hermes"]

    def test_fifo_for_duplicate_tool_names(self, mock_conn, event_loop_fixture):
        """Two same-name tools complete in FIFO order against the right ids."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            router("subagent.tool", "terminal", "$ ls", {"command": "ls"},
                   child_session_id="child-1")
            router("subagent.tool", "terminal", "$ pwd", {"command": "pwd"},
                   child_session_id="child-1")
            starts = [
                call.args[3] for call in mock_send.call_args_list
                if getattr(call.args[3], "session_update", None) == "tool_call"
            ]
            first_id = starts[0].tool_call_id
            second_id = starts[1].tool_call_id

            router("subagent.tool_completed", "terminal", None,
                   child_session_id="child-1", result="one")
            router("subagent.tool_completed", "terminal", None,
                   child_session_id="child-1", result="two")

        completed_ids = [call.args[0] for call in mock_btc.call_args_list]
        assert completed_ids == [first_id, second_id]
        assert mock_btc.call_args_list[0].kwargs["result"] == "one"
        assert mock_btc.call_args_list[1].kwargs["result"] == "two"

    def test_events_without_child_session_id_dropped(self, mock_conn, event_loop_fixture):
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.start", None, "goal text")
            router("subagent.tool", "terminal", "$ ls", {"command": "ls"})

        mock_send.assert_not_called()

    def test_thinking_and_text_mapping(self, mock_conn, event_loop_fixture):
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.thinking", None, "pondering...", child_session_id="child-1")
            router("subagent.text", None, "answer chunk", child_session_id="child-1")

        updates = [call.args[3] for call in mock_send.call_args_list]
        kinds = [getattr(u, "session_update", None) for u in updates]
        # First routed event lazily synthesizes the isRunning:true info update.
        assert kinds == [
            "session_info_update",
            "agent_thought_chunk",
            "agent_message_chunk",
        ]
        assert updates[1].content.text == "pondering..."
        assert updates[2].content.text == "answer chunk"

    def test_progress_events_dropped(self, mock_conn, event_loop_fixture):
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.start", None, "goal", child_session_id="child-1")
            mock_send.reset_mock()
            router("subagent.progress", None, "🔀 terminal, file", child_session_id="child-1")

        mock_send.assert_not_called()

    def test_complete_flushes_dangling_tool_ids(self, mock_conn, event_loop_fixture):
        """subagent.complete closes tool calls that never got a completion."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.tool", "terminal", "$ sleep", {"command": "sleep"},
                   child_session_id="child-1")
            router("subagent.complete", None, "done", child_session_id="child-1")

        updates = [call.args[3] for call in mock_send.call_args_list]
        kinds = [getattr(u, "session_update", None) for u in updates]
        # lazy isRunning:true, tool start, flushed completion, isRunning:false
        assert kinds == [
            "session_info_update",
            "tool_call",
            "tool_call_update",
            "session_info_update",
        ]

    def test_lazy_running_synthesis_when_start_missed(self, mock_conn, event_loop_fixture):
        """The first routed event synthesizes isRunning:true when start raced
        the session_ref fill and was dropped."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.tool", "terminal", "$ ls", {"command": "ls"},
                   child_session_id="child-1")

        updates = [call.args[3] for call in mock_send.call_args_list]
        kinds = [getattr(u, "session_update", None) for u in updates]
        assert kinds == ["session_info_update", "tool_call"]
        assert updates[0].field_meta["hermes"]["isRunning"] is True

    def test_independent_state_per_child(self, mock_conn, event_loop_fixture):
        """FIFO queues are keyed by child id — batch children don't cross-pop."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            router("subagent.tool", "terminal", "$ a", {"command": "a"},
                   child_session_id="child-1")
            router("subagent.tool", "terminal", "$ b", {"command": "b"},
                   child_session_id="child-2")
            starts = {
                call.args[1]: call.args[3].tool_call_id
                for call in mock_send.call_args_list
                if getattr(call.args[3], "session_update", None) == "tool_call"
            }
            router("subagent.tool_completed", "terminal", None,
                   child_session_id="child-2", result="b-done")

        assert mock_btc.call_count == 1
        assert mock_btc.call_args.args[0] == starts["child-2"]

    def test_tool_progress_cb_hands_off_subagent_events(self, mock_conn, event_loop_fixture):
        """make_tool_progress_cb routes subagent.* events with child_session_id
        to the router instead of dropping them."""
        router_calls = []

        def fake_router(event_type, name=None, preview=None, args=None, **kwargs):
            router_calls.append((event_type, kwargs.get("child_session_id")))

        cb = make_tool_progress_cb(
            mock_conn, "parent-1", event_loop_fixture, {}, {},
            subagent_router=fake_router,
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb("subagent.start", None, "goal", None, child_session_id="child-1")
            cb("subagent.tool", "terminal", "$ ls", {"command": "ls"},
               child_session_id="child-1")
            # No child id yet (pre session_ref fill) — dropped, not routed.
            cb("subagent.start", None, "goal", None)

        assert router_calls == [
            ("subagent.start", "child-1"),
            ("subagent.tool", "child-1"),
        ]
        # Nothing emitted on the PARENT session for subagent.* events.
        mock_send.assert_not_called()

    def test_router_safe_from_worker_thread(self, mock_conn):
        """Delegate children run in worker threads; frames must arrive on the
        server loop via the threadsafe scheduling path."""
        import threading

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        try:
            router = make_subagent_update_router(mock_conn, loop)

            def _fire():
                router("subagent.tool", "terminal", "$ ls", {"command": "ls"},
                       child_session_id="child-1")
                router("subagent.tool_completed", "terminal", None,
                       child_session_id="child-1", result="ok")

            worker = threading.Thread(target=_fire)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive()

            # info update (lazy running) + tool start + tool complete
            assert mock_conn.session_update.await_count == 3
            session_ids = {
                call.args[0] for call in mock_conn.session_update.await_args_list
            }
            assert session_ids == {"child-1"}
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

    def test_finalize_closes_dangling_state(self, mock_conn, event_loop_fixture):
        """finalize() completes un-popped tool ids and flips isRunning off for
        children whose subagent.complete never arrived (crash/interrupt)."""
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.tool", "terminal", "$ sleep", {"command": "sleep"},
                   child_session_id="child-1")
            mock_send.reset_mock()
            router.finalize()

        updates = [(call.args[1], call.args[3]) for call in mock_send.call_args_list]
        kinds = [(sid, getattr(u, "session_update", None)) for sid, u in updates]
        assert kinds == [
            ("child-1", "tool_call_update"),
            ("child-1", "session_info_update"),
        ]
        info = updates[-1][1]
        assert info.field_meta["hermes"]["isRunning"] is False

    def test_finalize_noop_after_clean_completion(self, mock_conn, event_loop_fixture):
        router = self._make_router(mock_conn, event_loop_fixture)

        with patch("acp_adapter.events._send_update") as mock_send:
            router("subagent.tool", "terminal", "$ ls", {"command": "ls"},
                   child_session_id="child-1")
            router("subagent.tool_completed", "terminal", None,
                   child_session_id="child-1", result="ok")
            router("subagent.complete", None, "done", child_session_id="child-1")
            mock_send.reset_mock()
            router.finalize()

        mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

class TestSendUpdate:
    def test_scheduler_failure_closes_update_coroutine(self, event_loop_fixture):
        """If run_coroutine_threadsafe raises, _send_update must close the coro."""
        created = {"coro": None}

        async def _session_update(session_id, update):
            return None

        conn = MagicMock()

        def _capture_update(session_id, update):
            created["coro"] = _session_update(session_id, update)
            return created["coro"]

        conn.session_update = _capture_update

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                _send_update(conn, "session-1", event_loop_fixture, {"type": "noop"})
            gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        # Only count warnings about THIS test's coroutine; other tests
        #  may emit unrelated
        # "coroutine was never awaited" warnings that bleed through.
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_session_update" in str(w.message)
        ]
        assert runtime_warnings == []


class TestAssistantMessageIds:
    """Streamed chunks carry a per-message ACP messageId; the None flush sentinel starts a new one."""

    def test_deltas_share_one_uuid_until_flush(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        cb = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("Hello ")
            cb("")  # empty delta is ignored, not a flush
            cb("world")
            cb(None)  # flush sentinel — closes the message
            cb("next turn")
        assert sent[0].message_id == sent[1].message_id
        assert sent[2].message_id != sent[0].message_id
        # ACP requires UUID-format message ids.
        assert uuid.UUID(sent[0].message_id) and uuid.UUID(sent[2].message_id)

    def test_thought_chunks_carry_id(self, mock_conn, event_loop_fixture):
        from acp_adapter.events import AssistantMessageIdAllocator

        ids = AssistantMessageIdAllocator()
        think = make_thinking_cb(mock_conn, "s", event_loop_fixture, ids)
        msg = make_message_cb(mock_conn, "s", event_loop_fixture, ids)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            think("pondering")
            msg("answer")
        # Reasoning and answer of the same reply share one message id.
        assert sent[0].message_id == sent[1].message_id

    def test_no_allocator_keeps_legacy_shape(self, mock_conn, event_loop_fixture):
        cb = make_message_cb(mock_conn, "s", event_loop_fixture)
        sent = []
        with patch("acp_adapter.events._send_update",
                   side_effect=lambda c, s, l, u: sent.append(u)):
            cb("text")
        assert sent[0].message_id is None


class TestToolCallsAlwaysReachATerminalStatus:
    """A tool call left ``in_progress`` makes a finished turn look like it ran nothing.

    Two invariants: every call is closed exactly once from its own ``tool.completed``
    (the ``prev_tools`` step closer stands down once completions arrive), and whatever
    is still open at turn end is failed with BOTH per-turn dicts drained together."""

    def _patch(self):
        return patch("acp_adapter.events.asyncio.run_coroutine_threadsafe")

    def test_tool_completed_closes_the_call_once_and_the_step_closer_stands_down(self, mock_conn, event_loop_fixture):
        from collections import deque

        ids, meta, turn_state = {"read": deque(["tc-1", "tc-2"])}, {"tc-1": {"args": {"path": "a"}}}, {}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state=turn_state)
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, turn_state)
        with self._patch() as rcts, patch("acp_adapter.events.build_tool_complete") as btc:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "read", None, None, result="file body")
            step(2, [{"name": "read", "result": "file body", "arguments": '{"path": "a"}'}])
        btc.assert_called_once_with(
            "tc-1", "read", result="file body", function_args={"path": "a"}, snapshot=None, is_error=False,
        )
        assert list(ids["read"]) == ["tc-2"] and "tc-1" not in meta

    def test_step_fallback_coerces_wire_arguments_and_turn_end_flush_fails_what_is_still_open(
        self, mock_conn, event_loop_fixture,
    ):
        """No completion projected: the step closer must survive the JSON-string ``arguments``
        the wire carries (a real ``write_file`` close raised on ``.get`` and was swallowed);
        a denied edit is then failed at turn end, draining ``tool_call_ids`` AND ``tool_call_meta``."""
        from collections import deque

        from acp_adapter.events import flush_open_tool_calls

        ids = {"write_file": deque(["tc-1"]), "edit": deque(["tc-denied"])}
        meta = {"tc-1": {"args": {"path": "a"}, "snapshot": None}, "tc-denied": {"args": {}}}
        step = make_step_cb(mock_conn, "s", event_loop_fixture, ids, meta, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            step(2, [{"name": "write_file", "result": "ok", "arguments": '{"path": "a", "content": "x"}'}])
            assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["completed"]
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 1
            assert flush_open_tool_calls(mock_conn, "s", event_loop_fixture, ids, meta) == 0
        statuses = [c.args[1].status for c in mock_conn.session_update.call_args_list]
        assert statuses == ["completed", "failed"]
        assert ids == {} and meta == {}

    def test_tool_completed_is_error_flag_closes_the_call_as_failed(self, mock_conn, event_loop_fixture):
        """``tool.completed`` carries the executor's ``is_error``; a cancelled tool's plain-text
        result trips no heuristic, so dropping the flag showed an interrupted call green."""
        from collections import deque

        ids = {"terminal": deque(["tc-1"])}
        progress = make_tool_progress_cb(mock_conn, "s", event_loop_fixture, ids, {})
        with self._patch() as rcts:
            rcts.return_value = MagicMock(spec=Future)
            progress("tool.completed", "terminal", None, None, is_error=True,
                     result="[Tool execution cancelled — terminal was skipped. User sent a new message]")
        assert [c.args[1].status for c in mock_conn.session_update.call_args_list] == ["failed"]
