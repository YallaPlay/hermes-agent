"""Gateway session/list nests delegate children and forks under their listed parent."""
import asyncio

from acp_adapter.catalog import catalog_sessions
from acp_adapter.gateway_server import GatewayACPAgent
from hermes_state import SessionDB


def _seed(db_path, cwd):
    with SessionDB(db_path) as db:
        db.create_session(session_id='parent', source='acp', cwd=cwd, user_id='a@x.com')
        db.append_message('parent', 'user', 'parent turn')
        db.create_session(session_id='child', source='subagent', cwd='/elsewhere', parent_session_id='parent',
                          model_config={'_delegate_from': 'parent'}, user_id='a@x.com')
        db.append_message('child', 'user', 'delegated goal')
        db.create_session(session_id='fork', source='acp', cwd=cwd, model_config={'_forked_from': 'parent'})
        db.append_message('fork', 'user', '[Slack runtime context] copied envelope')
        db.create_session(session_id='cli-parent', source='cli', cwd=cwd)
        db.append_message('cli-parent', 'user', 'cli turn')
        db.create_session(session_id='cli-child', source='subagent', cwd=cwd, parent_session_id='cli-parent',
                          model_config={'_delegate_from': 'cli-parent'})
        db.append_message('cli-child', 'user', 'cli delegation')


def test_catalog_lists_delegate_children_and_fork_lineage(tmp_path):
    db_path = tmp_path / 'state.db'
    _seed(db_path, str(tmp_path))
    rows = {row['session_id']: row for row in catalog_sessions(db_path, str(tmp_path))}
    assert set(rows) == {'parent', 'child', 'fork'}
    assert rows['child']['subagent'] is True and rows['child']['parent_id'] == 'parent'
    assert rows['fork']['parent_id'] == 'parent' and rows['fork']['slack'] is False
    assert rows['parent']['user_id'] == 'a@x.com'


def test_gateway_list_sessions_stamps_lineage_meta(tmp_path):
    _seed(tmp_path / 'state.db', str(tmp_path))
    agent = GatewayACPAgent.__new__(GatewayACPAgent)
    agent._home = tmp_path
    resp = asyncio.run(agent.list_sessions(cwd=str(tmp_path)))
    meta = {s.session_id: (s.field_meta or {}).get('hermes', {}) for s in resp.sessions}
    assert meta['child']['isSubagent'] is True and meta['child']['forkedFrom'] == 'parent'
    assert meta['fork']['forkedFrom'] == 'parent'
    assert meta['parent']['owner'] == 'a@x.com'
    assert 'isRunning' not in meta['child'] and 'isRunning' not in meta['parent']


class _Editor:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))


async def _no_gateway():
    raise AssertionError('a delegate child must not reach the owner')


def test_gateway_load_session_replays_delegate_child_read_only(tmp_path):
    """A listed child opens observation-only from its stored transcript. The owner's
    ``session.resume`` refuses children, which surfaced as ``Internal error`` in the editor."""
    _seed(tmp_path / 'state.db', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        db.append_message('child', 'assistant', 'child progress')
    agent = GatewayACPAgent.__new__(GatewayACPAgent)
    agent._home, agent._conn, agent._client = tmp_path, _Editor(), _no_gateway
    resp = asyncio.run(agent.load_session(cwd=str(tmp_path), session_id='child', mcp_servers=[]))
    assert resp.field_meta == {'hermes': {'isSubagent': True, 'messageCount': 2}}
    assert {sid for sid, _ in agent._conn.updates} == {'child'}
    kinds = [getattr(u, 'session_update', None) for _, u in agent._conn.updates]
    assert 'user_message_chunk' in kinds and 'agent_message_chunk' in kinds


def test_gateway_list_sessions_marks_live_delegate_children_running(tmp_path):
    import time
    _seed(tmp_path / 'state.db', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        for sid in ('live', 'ended', 'stale'):
            db.create_session(session_id=sid, source='subagent', cwd='/elsewhere', parent_session_id='parent',
                              model_config={'_delegate_from': 'parent'})
            db.append_message(sid, 'user', f'{sid} goal')
        db.touch_session_activity('live', time.time(), description='tool running: terminal')
        db.touch_session_activity('ended', time.time(), description='receiving stream response')
        db.end_session('ended', 'agent_close')
        db.touch_session_activity('stale', time.time() - 3600, description='receiving stream response')
    agent = GatewayACPAgent.__new__(GatewayACPAgent)
    agent._home = tmp_path
    resp = asyncio.run(agent.list_sessions(cwd=str(tmp_path)))
    meta = {s.session_id: (s.field_meta or {}).get('hermes', {}) for s in resp.sessions}
    assert meta['live'].get('isRunning') is True
    assert 'isRunning' not in meta['ended'] and 'isRunning' not in meta['stale']
    agent._conn, agent._client = _Editor(), _no_gateway
    loaded = asyncio.run(agent.load_session(cwd=str(tmp_path), session_id='live', mcp_servers=[]))
    assert loaded.field_meta['hermes']['isRunning'] is True
