import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _mk(db: SessionDB, sid: str, user_id=None):
    """Create a listable acp session with one message so it surfaces."""
    db.create_session(sid, source="acp", user_id=user_id, model="m")
    db._conn.execute(
        "UPDATE sessions SET message_count = 1, started_at = ? WHERE id = ?",
        (time.time(), sid),
    )
    db._conn.commit()


def test_set_session_owner_stamps_and_clears(db):
    _mk(db, "s1")
    assert db.get_session("s1")["user_id"] in (None, "")

    assert db.set_session_owner("s1", "alice@yallaplay.com") is True
    assert db.get_session("s1")["user_id"] == "alice@yallaplay.com"

    # Empty string clears ownership back to NULL.
    assert db.set_session_owner("s1", "") is True
    assert db.get_session("s1")["user_id"] is None


def test_set_session_owner_unknown_returns_false(db):
    assert db.set_session_owner("nope", "x@y.com") is False


def test_owner_filter_shows_own_and_untagged_hides_others(db):
    _mk(db, "mine", user_id="alice@yallaplay.com")
    _mk(db, "theirs", user_id="bob@yallaplay.com")
    _mk(db, "legacy", user_id=None)  # untagged

    ids = {
        s["id"]
        for s in db.list_sessions_rich(source="acp", owner="alice@yallaplay.com")
    }
    # Own + untagged visible; someone else's hidden.
    assert ids == {"mine", "legacy"}


def test_no_owner_filter_shows_all(db):
    _mk(db, "mine", user_id="alice@yallaplay.com")
    _mk(db, "theirs", user_id="bob@yallaplay.com")

    ids = {s["id"] for s in db.list_sessions_rich(source="acp")}
    assert ids == {"mine", "theirs"}
