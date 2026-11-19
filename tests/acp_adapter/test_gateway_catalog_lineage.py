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
