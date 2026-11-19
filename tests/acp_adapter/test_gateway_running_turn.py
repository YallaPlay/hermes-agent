"""An editor that loads a session mid-turn (a window reload) sees the turn it did not start, and its end.

Main's in-process server put ``_meta.hermes.isRunning`` (+ ``currentPromptText``) on the load reply and sent a
title-less ``session_info_update`` with ``isRunning`` at every turn start and end (``_turn_status_meta``,
``_send_turn_status_update``). Without them the editor shows an idle composer over a running turn."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent

STARTED = {'admission_id': 'a1', 'status': 'started', 'text': 'fix the bug'}


def _snapshot(running, pending):
    return {'session_id': 's', 'messages': [], 'prompts': [], 'pending': pending, 'running': running,
            'replay_epoch': 'e', 'last_sequence': 10, 'execution_generation': 1, 'revision': 1, 'info': {}}


def _agent(snapshot, monkeypatch):
    monkeypatch.setattr('acp_adapter.catalog.logical_session_id', lambda db, held: 's')

    async def rpc(method, **params):
        return {'session.resume': snapshot, 'session.info': {}, 'config.set': {}, 'session.stats': {}}[method]

    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    gateway = type('Gateway', (), {})()
    gateway.events, gateway.rpc = asyncio.Queue(), rpc
    agent._gateway = gateway

    async def client():
        return gateway
    monkeypatch.setattr(agent, '_client', client)
    return agent, gateway


def _running_updates(agent):
    return [c.kwargs['update'].field_meta['hermes'] for c in agent._conn.session_update.await_args_list
            if getattr(c.kwargs['update'], 'session_update', None) == 'session_info_update'
            and (c.kwargs['update'].field_meta or {}).get('hermes', {}).get('isRunning') is not None]


async def _deliver(agent, gateway, *events):
    for seq, payload in enumerate(events, start=11):
        await gateway.events.put({'params': {'session_id': 's', 'type': 'session.info', 'replay_epoch': 'e',
                                             'seq': seq, 'payload': payload}})
    await gateway.events.put(EOFError('end of the test stream'))
    await asyncio.wait_for(agent._events(), 5)


@pytest.mark.asyncio
async def test_a_load_mid_turn_reports_the_turn_and_its_end(monkeypatch):
    agent, gateway = _agent(_snapshot(True, [STARTED]), monkeypatch)
    response = await agent.load_session(cwd='/', session_id='s')
    assert response.field_meta['hermes']['isRunning'] is True
    assert response.field_meta['hermes']['currentPromptText'] == 'fix the bug'
    assert _running_updates(agent) == []  # the reply carries the running turn; no duplicate update

    await _deliver(agent, gateway, {'pending': [], 'running': False})
    assert _running_updates(agent) == [{'isRunning': False}]


@pytest.mark.asyncio
async def test_an_idle_load_shows_no_turn_until_one_starts(monkeypatch):
    agent, gateway = _agent(_snapshot(False, []), monkeypatch)
    response = await agent.load_session(cwd='/', session_id='s')
    assert 'isRunning' not in ((response.field_meta or {}).get('hermes') or {})

    # The same state again (another surface's unrelated info) repaints nothing; a start and an end do.
    await _deliver(agent, gateway, {'pending': [], 'running': False},
                   {'pending': [STARTED], 'running': True}, {'pending': [], 'running': False})
    assert _running_updates(agent) == [{'isRunning': True, 'currentPromptText': 'fix the bug'},
                                       {'isRunning': False}]
