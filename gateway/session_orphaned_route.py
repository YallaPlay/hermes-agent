"""A session whose frozen provider this profile no longer declares moves to the profile's default.

A session copied from another profile, or one whose ``providers:`` entry was removed, keeps a frozen
route naming a provider the resolver rejects as unknown (``AuthError(code='invalid_provider')``), so
every turn fails before the agent exists and no credential is ever sent. Nothing can resume that
route. At the next turn boundary the drain commits ``/model reset`` through ``session.mutate``'s own
writer (the profile's configured model and provider, history kept) and names the move to every
viewer, so the move is never silent. A provider that exists but cannot authenticate is not
orphaned: it keeps failing closed (#139908).
"""
import logging

logger = logging.getLogger(__name__)


def _names_known_provider(provider, config):
    """Whether *provider* names a configured entry or a registry provider (reads no credential)."""
    from gateway.session_route_identity import _named_entry
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.models import normalize_provider
    if _named_entry(provider, config):
        return True
    try:
        resolve_provider(normalize_provider(provider))
    except AuthError as exc:
        return exc.code != 'invalid_provider'
    return True


def _orphaned_provider(authority, live):
    """The frozen provider the session runs if this profile does not know it, else None."""
    from gateway.session_policy import policy_for_source, profile_read_scope
    from hermes_cli.auth import AuthError
    policy = policy_for_source(authority.runner, live.source)
    if policy is None or policy.ignore_user_config:
        return None
    with profile_read_scope(policy, authority):
        config = policy.config(authority)
        provider = str((config.get('model') or {}).get('provider') or '')
        if not provider or _names_known_provider(provider, config):
            return None  # every turn passes here: only a name the resolver cannot place pays for a resolve
        try:
            authority.runner._resolve_session_agent_runtime(
                source=live.source, session_key=live.route, fallback=False, record_notice=False)
        except AuthError as exc:
            if exc.code == 'invalid_provider' and provider:
                return provider
    return None


async def replace_orphaned_route(authority, ref, live):
    """Reset an orphaned route before the claim. The drain calls this under the claim gate and
    mutation lock, so no turn is running."""
    import asyncio
    from gateway.session_deferred_pick import _commit
    from gateway.session_mutation_model import configured_default
    from hermes_state_local import local_receipt
    try:
        provider = await asyncio.to_thread(_orphaned_provider, authority, live)
        if provider is None:
            return
        model, profile_provider = configured_default(authority)
        if not model or profile_provider.strip().lower() == provider.strip().lower():
            return  # the profile's own route is the unknown one: a reset would land on it again
        row = await asyncio.to_thread(authority.db.get_session, ref.session_id) or {}
        receipt = await asyncio.to_thread(local_receipt, authority.db, ref.session_id)
    except Exception:
        logger.exception('Orphaned-route check on %s failed', ref.session_id)
        return
    entry = {'principal_id': receipt['principal_id'], 'payload': {'model': 'reset'},
             'application_id': f"orphaned-route:{ref.session_id}:{row.get('runtime_revision')}"}
    result = await _commit(authority, ref, entry, 'model')
    if result is None or result.get('status') in {'refused', 'replayed', 'noop'}:
        if result and result.get('status') == 'refused':
            logger.warning('Session %s runs unknown provider %s; reset refused: %s',
                           ref.session_id, provider, result.get('reason'))
        return
    logger.warning('Session %s ran provider %s, which this profile does not declare; moved to %s/%s',
                   ref.session_id, provider, profile_provider or 'default', model)
    live.event_stream.publish(ref.session_id, {
        'session_id': ref.session_id, 'operation': 'model', 'status': 'route_replaced',
        'from_provider': provider, 'from_model': row.get('model'), 'model': model,
        **({'provider': profile_provider} if profile_provider else {})}, event_type='session.updated')
