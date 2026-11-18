"""Edit-and-resend on the gateway path: ``session/prompt`` ``_meta.hermes.keepHistory`` rewinds the
owner's transcript (``session.mutate rewind``) before the edited turn is submitted, and replayed user
turns carry the ``_meta.hermes.historyIndex`` coordinate the client sends back.

Real authority, SQLite and ``AuthorityConnection.dispatch`` behind ``GatewayACPAgent.prompt``."""
import asyncio

import acp
import pytest

# The editor fixture imports gateway.run, whose bootstrap probes the checkout's git dir for an
# interrupted update. Import it at collection (before the per-test home I/O guard) so a linked
# worktree whose git dir sits under the Hermes home still runs this file alone.
import gateway.run  # noqa: F401

import acp_adapter.server  # noqa: F401 - replay reuses main's transcript projection; import it at collection
from acp_adapter.gateway_rewind import replay_updates
from tests.acp_adapter.test_gateway_retry_command import _editor, _users


def _seed(e):
    for ask, reply in (('first ask', 'first reply'), ('second ask', 'second reply')):
        e.db.append_message(e.sid, 'user', ask)
        e.db.append_message(e.sid, 'assistant', reply)


@pytest.mark.asyncio
async def test_keep_history_rewinds_then_submits_the_edited_turn(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch) as e:
        _seed(e)
        response = await asyncio.wait_for(
            e.agent.prompt([acp.text_block('edited second')], e.sid, hermes={'keepHistory': 2}), 5)
        assert response.stop_reason == 'end_turn'
        assert e.executed == ['edited second']
        # The rewind deactivated the edited turn and its reply (the fake owner persists no new rows).
        assert _users(e.db, e.sid) == ['first ask']
        assert 'second reply' not in [m['content'] for m in e.db.get_messages(e.sid)]


@pytest.mark.asyncio
async def test_keep_history_at_the_end_keeps_everything(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch) as e:
        _seed(e)
        await asyncio.wait_for(e.agent.prompt([acp.text_block('third ask')], e.sid, hermes={'keepHistory': 4}), 5)
        assert e.executed == ['third ask'] and _users(e.db, e.sid) == ['first ask', 'second ask']


@pytest.mark.asyncio
async def test_keep_history_on_a_non_user_row_is_refused_before_anything_changes(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch) as e:
        _seed(e)
        with pytest.raises(acp.RequestError) as caught:
            await asyncio.wait_for(e.agent.prompt([acp.text_block('x')], e.sid, hermes={'keepHistory': 1}), 5)
        assert caught.value.code == -32602
        assert not e.executed and _users(e.db, e.sid) == ['first ask', 'second ask']


@pytest.mark.asyncio
@pytest.mark.parametrize('keep', [-1, True, 'two', 1.5])
async def test_invalid_keep_history_is_invalid_params(tmp_path, monkeypatch, keep):
    async with _editor(tmp_path, monkeypatch) as e:
        _seed(e)
        with pytest.raises(acp.RequestError) as caught:
            await asyncio.wait_for(e.agent.prompt([acp.text_block('x')], e.sid, hermes={'keepHistory': keep}), 5)
        assert caught.value.code == -32602
        assert not e.executed and len(e.db.get_messages(e.sid)) == 4


@pytest.mark.asyncio
async def test_keep_history_with_an_empty_prompt_is_invalid_params(tmp_path, monkeypatch):
    async with _editor(tmp_path, monkeypatch) as e:
        _seed(e)
        with pytest.raises(acp.RequestError) as caught:
            await asyncio.wait_for(e.agent.prompt([acp.text_block('   ')], e.sid, hermes={'keepHistory': 2}), 5)
        assert caught.value.code == -32602
        assert len(e.db.get_messages(e.sid)) == 4


def test_replay_stamps_each_rewindable_user_turn_with_its_live_index():
    live = [{'role': 'user', 'content': 'a', 'row_id': 1},
            {'role': 'assistant', 'content': 'b', 'row_id': 2},
            {'role': 'user', 'content': 'c', 'row_id': 3},
            {'role': 'assistant', 'content': 'd', 'row_id': 4}]
    stamped = [(u.session_update, meta) for u, meta in replay_updates(live)]
    assert stamped == [('user_message_chunk', {'hermes': {'historyIndex': 0}}),
                       ('agent_message_chunk', None),
                       ('user_message_chunk', {'hermes': {'historyIndex': 2}}),
                       ('agent_message_chunk', None)]


def test_replay_keeps_tool_call_pairs_across_assistant_and_tool_rows():
    live = [{'role': 'user', 'content': 'go', 'row_id': 1},
            {'role': 'assistant', 'content': '', 'row_id': 2,
             'tool_calls': [{'id': 't1', 'function': {'name': 'terminal', 'arguments': '{"command": "ls"}'}}]},
            {'role': 'tool', 'content': 'ok', 'tool_call_id': 't1', 'row_id': 3}]
    kinds = [u.session_update for u, _ in replay_updates(live)]
    assert kinds == ['user_message_chunk', 'tool_call', 'tool_call_update']
