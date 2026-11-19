"""A session whose frozen provider the profile does not declare moves to the profile's default.

A route naming an unknown provider (a session adopted from another profile's store, or one whose
``providers:`` entry was later removed) fails every turn before an agent exists; the drain resets it
at the next turn boundary and names the move. A declared provider that merely cannot authenticate
is left alone (it still fails closed, #139908). Real SessionStore, real authority, real legacy
adoption and a real config.yaml.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionActor

SID = '20261118_190602_1c358e'


def _profile(home, providers):
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'keep-box', 'default': 'current'},
        'providers': {name: {'name': name, 'base_url': url, 'api_key': key} for name, (url, key) in providers.items()},
        'platform_toolsets': {'acp': []},
    }), encoding='utf-8')


async def _adopt(home, provider, base_url):
    """A legacy ACP row routed through *provider*, adopted by this profile's owner."""
    from gateway.config import GatewayConfig
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    actor = SessionActor('uid:1000', str(home), frozenset({'session:read', 'session:create'}), 'native')
    bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': str(home), 'instance_id': 'fixture'})
    authority.db.create_session(SID, source='acp', model='original', model_config={
        'cwd': str(home), 'provider': provider, 'base_url': base_url, 'api_mode': 'chat_completions'})
    ref = await asyncio.to_thread(resolve_local_target, authority, actor, SID)
    return store, runner, authority, ref, authority.sessions[ref.session_id]


def _route(runner, live):
    model, runtime = runner._resolve_session_agent_runtime(source=live.source, fallback=False, record_notice=False)
    return model, runtime.get('base_url'), runtime.get('api_key')


def _home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)  # gateway.run binds its home at first import
    return home


def test_an_unknown_provider_is_reset_to_the_profile_default_and_announced(monkeypatch):
    from gateway.session_orphaned_route import replace_orphaned_route
    from hermes_cli.auth_constants import AuthError
    home = _home(monkeypatch)
    # Copied from another profile's store: this profile never declared the session's provider.
    _profile(home, {'keep-box': ('http://127.0.0.1:8/v1', 'key-keep')})

    async def scenario():
        store, runner, authority, ref, live = await _adopt(home, 'gone-box', 'http://127.0.0.1:9/v1')
        try:
            with pytest.raises(AuthError) as caught:
                _route(runner, live)
            assert caught.value.code == 'invalid_provider'
            seen = []
            live.event_stream.publish = lambda session_id, payload, **kw: seen.append((kw.get('event_type'), payload))

            await replace_orphaned_route(authority, ref, live)

            assert _route(runner, live) == ('current', 'http://127.0.0.1:8/v1', 'key-keep')
            notice = [p for kind, p in seen if kind == 'session.updated' and p.get('status') == 'route_replaced']
            assert len(notice) == 1
            assert (notice[0]['from_provider'], notice[0]['from_model']) == ('gone-box', 'original')
            assert (notice[0]['provider'], notice[0]['model']) == ('keep-box', 'current')
            seen.clear()
            await replace_orphaned_route(authority, ref, live)  # once moved, the next boundary does nothing
            assert seen == []
        finally:
            store.close_all_db_handles()
    asyncio.run(scenario())


def test_a_declared_provider_that_cannot_authenticate_is_not_replaced(monkeypatch):
    from gateway.session_orphaned_route import replace_orphaned_route
    from hermes_cli.auth_constants import AuthError
    home = _home(monkeypatch)
    _profile(home, {'gone-box': ('http://127.0.0.1:9/v1', 'key-gone'), 'keep-box': ('http://127.0.0.1:8/v1', 'key-keep')})

    async def scenario():
        store, runner, authority, ref, live = await _adopt(home, 'gone-box', 'http://127.0.0.1:9/v1')
        try:
            real = runner._resolve_session_agent_runtime

            def refuse(**kwargs):
                raise AuthError('expired', code='invalid_api_key')
            runner._resolve_session_agent_runtime = refuse
            seen = []
            live.event_stream.publish = lambda session_id, payload, **kw: seen.append(payload)

            await replace_orphaned_route(authority, ref, live)

            runner._resolve_session_agent_runtime = real
            assert _route(runner, live) == ('original', 'http://127.0.0.1:9/v1', 'key-gone')
            assert seen == []
        finally:
            store.close_all_db_handles()
    asyncio.run(scenario())
