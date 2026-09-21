"""The ACP-supplied owner survives deferred row creation.

session/new carries the authenticated owner, but row creation is deferred until
the session has history. By then the agent's ``_ensure_db_session`` has usually
created the row already, and it knows nothing about the ACP owner — so the
create-time ``user_id`` write never runs and the session persists untagged,
invisible behind the strict "My Sessions" filter.
"""
from types import SimpleNamespace

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _manager(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    return db, SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture"))


def test_owner_lands_on_a_row_the_agent_created_first(tmp_path):
    db, manager = _manager(tmp_path)
    state = manager.create_session(cwd=str(tmp_path), owner="me@yallaplay.com")
    # Deferred: session/new leaves no row behind.
    assert db.get_session(state.session_id) is None

    # The first turn: the agent creates the row, with no notion of an ACP owner.
    db.create_session(session_id=state.session_id, source="acp", model="fixture")
    state.history.append({"role": "user", "content": "first prompt"})
    manager.save_session(state.session_id)

    assert db.get_session(state.session_id)["user_id"] == "me@yallaplay.com"
    db.close()


def test_persist_never_overwrites_an_owner_already_on_the_row(tmp_path):
    db, manager = _manager(tmp_path)
    state = manager.create_session(cwd=str(tmp_path), owner="me@yallaplay.com")
    db.create_session(
        session_id=state.session_id, source="acp", model="fixture",
        user_id="someone.else@yallaplay.com",
    )
    state.history.append({"role": "user", "content": "first prompt"})
    manager.save_session(state.session_id)

    # Backfill fills a gap; it is not an ownership transfer.
    assert db.get_session(state.session_id)["user_id"] == "someone.else@yallaplay.com"
    db.close()
