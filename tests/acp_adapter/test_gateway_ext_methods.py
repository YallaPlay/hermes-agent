"""``_x`` extension methods on the gateway path (``GatewayACPAgent.ext_method``), against a fake owner.

The VS Code client calls the same names, params and result shapes main's in-process adapter
(``acp_adapter/server.py::ext_method``) answers; each test pins one of them.
"""
import acp
import pytest
from unittest.mock import AsyncMock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayRPCError


def _agent(rpc, held=('s',)):
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    for sid in held:
        agent._snapshots[sid] = {'execution_generation': 7, 'revision': 3}
    return agent


def _snapshot(**extra):
    return {'revision': 3, 'execution_generation': 7, 'subscription_id': 'sub-1', 'messages': [], **extra}


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['noSuchMethod', 'deriveTitle', 'promoteSession'])
async def test_unknown_and_dropped_methods_answer_method_not_found(name):
    agent = _agent(AsyncMock(side_effect=AssertionError('no RPC')))
    with pytest.raises(acp.RequestError) as caught:
        await agent.ext_method(name, {'sessionId': 's'})
    assert caught.value.code == -32601


@pytest.mark.asyncio
async def test_set_owner_is_an_acknowledged_no_op():
    agent = _agent(AsyncMock(side_effect=AssertionError('no RPC')))
    assert await agent.ext_method('setOwner', {'sessionId': 's', 'owner': 'a@b.c'}) == {'ok': True, 'ignored': True}


@pytest.mark.asyncio
async def test_set_archived_mutates_a_held_session():
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        if method == 'session.resume':
            return _snapshot()
        assert method == 'session.mutate'
        return {'session_id': 's', 'operation': 'archive', 'revision': 4, 'archived': True}

    agent = _agent(rpc)
    assert await agent.ext_method('setArchived', {'sessionId': 's', 'archived': True}) == {'ok': True}
    mutate = calls[-1][1]
    assert (mutate['operation'], mutate['payload'], mutate['expected_revision']) == ('archive', {'archived': True}, 3)
    assert [m for m, _ in calls] == ['session.resume', 'session.mutate']


@pytest.mark.asyncio
async def test_set_archived_on_an_unheld_session_detaches_the_borrowed_subscription():
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        if method == 'session.resume':
            assert params.get('omit_messages') is True
            return _snapshot()
        if method == 'session.mutate':
            return {'session_id': 'other', 'operation': 'archive', 'revision': 4}
        assert method == 'session.detach' and params == {'session_id': 'other', 'subscription_id': 'sub-1'}
        return {'session_id': 'other', 'subscription_id': 'sub-1', 'detached': True}

    agent = _agent(rpc)
    assert await agent.ext_method('setArchived', {'sessionId': 'other', 'archived': False}) == {'ok': True}
    assert [m for m, _ in calls] == ['session.resume', 'session.mutate', 'session.detach']
    assert 'other' not in agent._snapshots


@pytest.mark.asyncio
async def test_set_archived_refusal_is_ok_false():
    async def rpc(method, **params):
        if method == 'session.resume':
            return _snapshot()
        raise GatewayRPCError('not_found')

    agent = _agent(rpc)
    assert await agent.ext_method('setArchived', {'sessionId': 's', 'archived': True}) == {'ok': False}


@pytest.mark.asyncio
async def test_missing_session_id_is_ok_false_like_main():
    agent = _agent(AsyncMock(side_effect=AssertionError('no RPC')))
    assert await agent.ext_method('setArchived', {'archived': True}) == {'ok': False}
    for name in ('setTitle', 'contextBreakdown', 'contextReport', 'accountUsage', 'refreshModels'):
        assert await agent.ext_method(name, {}) == {'ok': False, 'error': 'sessionId required'}


@pytest.mark.asyncio
async def test_set_title_returns_the_committed_title_and_pushes_it():
    async def rpc(method, **params):
        if method == 'session.resume':
            return _snapshot()
        assert method == 'session.mutate' and params['operation'] == 'rename'
        assert params['payload'] == {'title': 'My chat'} and 'expected_generation' not in params
        return {'session_id': 's', 'operation': 'rename', 'revision': 4, 'title': 'My chat'}

    agent = _agent(rpc)
    assert await agent.ext_method('setTitle', {'sessionId': 's', 'title': 'My chat'}) == {'ok': True, 'title': 'My chat'}
    update = agent._conn.session_update.await_args.kwargs['update']
    assert update.session_update == 'session_info_update' and update.title == 'My chat'


@pytest.mark.asyncio
async def test_set_title_clash_is_ok_false_with_mains_error(monkeypatch):
    async def rpc(method, **params):
        if method == 'session.resume':
            return _snapshot()
        raise GatewayRPCError('internal_error')

    agent = _agent(rpc)
    monkeypatch.setattr('acp_adapter.gateway_ext._title_holder', lambda home, sid, title: 'abc-123')
    assert await agent.ext_method('setTitle', {'sessionId': 's', 'title': 'Dup'}) == {
        'ok': False, 'error': "Title 'Dup' is already in use by session abc-123"}


@pytest.mark.asyncio
async def test_context_breakdown_is_the_owners_payload():
    payload = {'model': 'm', 'context_length': 1000, 'categories': [{'id': 'system_prompt', 'tokens': 5}]}

    async def rpc(method, **params):
        assert (method, params) == ('session.context_breakdown', {'session_id': 's'})
        return payload

    agent = _agent(rpc)
    assert await agent.ext_method('contextBreakdown', {'sessionId': 's'}) == {'ok': True, 'breakdown': payload}


@pytest.mark.asyncio
async def test_context_breakdown_refusal_is_ok_false():
    async def rpc(method, **params):
        raise GatewayRPCError('not_found')

    agent = _agent(rpc)
    assert await agent.ext_method('contextBreakdown', {'sessionId': 's'}) == {'ok': False, 'error': 'not_found'}


@pytest.mark.asyncio
async def test_context_report_renders_the_breakdown_as_markdown():
    payload = {'model': 'gpt-x', 'context_length': 1000, 'estimated_total': 120,
               'categories': [{'id': 'system_prompt', 'label': 'System prompt', 'tokens': 100},
                              {'id': 'conversation', 'label': 'Conversation', 'tokens': 20}],
               'context_files': [{'label': 'AGENTS.md', 'est_tokens': 40}]}

    async def rpc(method, **params):
        assert method == 'session.context_breakdown'
        return payload

    agent = _agent(rpc)
    full = await agent.ext_method('contextReport', {'sessionId': 's'})
    assert full['ok'] and full['report'].startswith('# Context report — Full context (gpt-x)')
    assert '## System prompt' in full['report'] and '100' in full['report'] and 'AGENTS.md' in full['report']
    one = await agent.ext_method('contextReport', {'sessionId': 's', 'category': 'conversation'})
    assert one['ok'] and '## Conversation' in one['report'] and '## System prompt' not in one['report']
    bad = await agent.ext_method('contextReport', {'sessionId': 's', 'category': 'nope'})
    assert bad == {'ok': False, 'error': "unknown context category: 'nope'"}


@pytest.mark.asyncio
async def test_account_usage_maps_the_owners_snapshot_to_mains_shape():
    snapshot = {'provider': 'openai-codex', 'plan': 'Pro', 'fetched_at': '2026-11-19T10:00:00+00:00',
                'windows': [{'label': 'Session', 'used_percent': 42.0, 'reset_at': '2026-11-19T12:00:00+00:00',
                             'detail': None}],
                'details': ['Credits: 5']}

    async def rpc(method, **params):
        assert (method, params) == ('session.account_usage', {'session_id': 's'})
        return {'lines': ['x'], 'nous_lines': [], 'nous_route': False, 'snapshot': snapshot}

    agent = _agent(rpc)
    assert await agent.ext_method('accountUsage', {'sessionId': 's'}) == {'ok': True, 'usage': {
        'provider': 'openai-codex', 'plan': 'Pro', 'fetchedAt': '2026-11-19T10:00:00+00:00',
        'windows': [{'label': 'Session', 'usedPercent': 42.0, 'resetAt': '2026-11-19T12:00:00+00:00', 'detail': None}],
        'details': ['Credits: 5']}}


@pytest.mark.asyncio
async def test_account_usage_without_an_endpoint_is_null_usage():
    async def rpc(method, **params):
        return {'lines': [], 'nous_lines': [], 'nous_route': False}

    agent = _agent(rpc)
    assert await agent.ext_method('accountUsage', {'sessionId': 's'}) == {'ok': True, 'usage': None}


@pytest.mark.asyncio
async def test_refresh_models_rebuilds_the_picker_bypassing_the_cache(monkeypatch):
    from acp.schema import ModelInfo, SessionModelState
    seen = {}

    def fake_state(info, *, force_refresh=False):
        seen.update(info=info, force_refresh=force_refresh)
        return SessionModelState(current_model_id='openai:gpt', available_models=[
            ModelInfo(model_id='openai:gpt', name='gpt', description='d')])

    async def rpc(method, **params):
        assert (method, params) == ('session.info', {'session_id': 's'})
        return {'model': 'gpt', 'provider': 'openai'}

    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', fake_state)
    agent = _agent(rpc)
    assert await agent.ext_method('refreshModels', {'sessionId': 's'}) == {'ok': True, 'models': {
        'currentModelId': 'openai:gpt',
        'availableModels': [{'modelId': 'openai:gpt', 'name': 'gpt', 'description': 'd'}]}}
    assert seen == {'info': {'model': 'gpt', 'provider': 'openai'}, 'force_refresh': True}


@pytest.mark.asyncio
async def test_editor_alias_resolves_to_the_logical_session():
    async def rpc(method, **params):
        assert params['session_id'] == 's'
        return {'categories': []}

    agent = _agent(rpc)
    agent._aliases['editor-held'] = 's'
    assert (await agent.ext_method('contextBreakdown', {'sessionId': 'editor-held'}))['ok']


@pytest.mark.asyncio
async def test_set_title_on_an_unheld_session_resolves_its_logical_id_and_detaches(monkeypatch):
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        assert params['session_id'] == 'root'
        if method == 'session.resume':
            return _snapshot()
        if method == 'session.mutate':
            assert (params['operation'], params['expected_revision']) == ('rename', 3)
            return {'session_id': 'root', 'operation': 'rename', 'revision': 4, 'title': 'T'}
        return {'detached': True}

    seen = []
    monkeypatch.setattr('acp_adapter.catalog.logical_session_id', lambda path, sid: seen.append(sid) or 'root')
    agent = _agent(rpc)
    assert await agent.ext_method('setTitle', {'sessionId': 'compressed-child', 'title': 'T'}) == {'ok': True, 'title': 'T'}
    assert seen == ['compressed-child']
    assert [m for m, _ in calls] == ['session.resume', 'session.mutate', 'session.detach']
    assert 'root' not in agent._snapshots
