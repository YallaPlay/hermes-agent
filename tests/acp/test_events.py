"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
import gc
import warnings
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import AgentPlanUpdate

from acp_adapter.events import (
    _build_plan_update_from_todo_result,
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
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None

    def test_handles_string_args(self, mock_conn, event_loop_fixture):
        """If args is a JSON string, it should be parsed."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "read_file", "Reading /etc/hosts", '{"path": "/etc/hosts"}')

        assert "read_file" in tool_call_ids

    def test_handles_non_dict_args(self, mock_conn, event_loop_fixture):
        """If args is not a dict, it should be wrapped."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ echo hi", None)

        assert "terminal" in tool_call_ids

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
            tc_id, "read_file", result="file contents", function_args={"path": "/etc/hosts"}, snapshot=None
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


class TestThinkingCallback:
    def test_emits_thought_chunk(self, mock_conn, event_loop_fixture):
        """Thinking callback should emit AgentThoughtChunk."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Analyzing the code...")

        mock_rcts.assert_called_once()

    def test_ignores_empty_text(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_ignores_untracked_tools(self, mock_conn, event_loop_fixture):
        """Tools not in tool_call_ids should be silently ignored."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb(1, [{"name": "unknown_tool", "result": "ok"}])

        mock_rcts.assert_not_called()

    def test_handles_string_tool_info(self, mock_conn, event_loop_fixture):
        """Tool info as a string (just the name) should work."""
        tool_call_ids = {"read_file": "tc-def456"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(2, ["read_file"])

        assert "read_file" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_result_passed_to_build_tool_complete(self, mock_conn, event_loop_fixture):
        """Tool result from prev_tools dict is forwarded to build_tool_complete."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-xyz789"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Provide a result string in the tool info dict
            cb(1, [{"name": "terminal", "result": '{"output": "hello"}'}])

        mock_btc.assert_called_once_with(
            "tc-xyz789", "terminal", result='{"output": "hello"}', function_args=None, snapshot=None
        )

    def test_none_result_passed_through(self, mock_conn, event_loop_fixture):
        """When result is None (e.g. first iteration), None is passed through."""
        from collections import deque

        tool_call_ids = {"web_search": deque(["tc-aaa"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "web_search", "result": None}])

        mock_btc.assert_called_once_with("tc-aaa", "web_search", result=None, function_args=None, snapshot=None)

    def test_step_callback_passes_arguments_and_snapshot(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"write_file": deque(["tc-write"])}
        tool_call_meta = {"tc-write": {"args": {"path": "fallback.txt"}, "snapshot": "snap"}}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "write_file", "result": '{"bytes_written": 23}', "arguments": {"path": "diff-test.txt"}}])

        mock_btc.assert_called_once_with(
            "tc-write",
            "write_file",
            result='{"bytes_written": 23}',
            function_args={"path": "diff-test.txt"},
            snapshot="snap",
        )

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

    def test_step_callback_completes_todo_without_emitting_plan(self, mock_conn, event_loop_fixture):
        """The deferred step callback finalizes the tool call but no longer emits
        the plan update — that now happens live on ``tool.completed`` so the
        turn's final todo is not stranded one step behind."""
        from collections import deque

        tool_call_ids = {"todo": deque(["tc-todo"])}
        loop = event_loop_fixture
        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})
        todo_result = (
            '{"todos":['
            '{"id":"inspect","content":"Inspect ACP","status":"completed"}'
            '],"summary":{"total":1}}'
        )

        with patch("acp_adapter.events._send_update") as mock_send:
            cb(1, [{"name": "todo", "result": todo_result}])

        updates = [call.args[3] for call in mock_send.call_args_list]
        assert [getattr(update, "session_update", None) for update in updates] == [
            "tool_call_update",
        ]

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

        update = _build_plan_update_from_todo_result(result)

        assert isinstance(update, AgentPlanUpdate)
        assert [entry.content for entry in update.entries] == ["Ship ACP plan"]
        assert [entry.status for entry in update.entries] == ["pending"]

    def test_todo_plan_update_with_empty_todos_clears_plan(self):
        update = _build_plan_update_from_todo_result('{"todos":[],"summary":{"total":0}}')

        assert isinstance(update, AgentPlanUpdate)
        assert update.session_update == "plan"
        assert update.entries == []


# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------


class TestMessageCallback:
    def test_emits_agent_message_chunk(self, mock_conn, event_loop_fixture):
        """Message callback should emit AgentMessageChunk."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Here is your answer.")

        mock_rcts.assert_called_once()

    def test_ignores_empty_message(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


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
