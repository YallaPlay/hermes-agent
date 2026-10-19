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


def test_owner_filter_shows_only_own_hides_untagged_and_others(db):
    _mk(db, "mine", user_id="alice@yallaplay.com")
    _mk(db, "theirs", user_id="bob@yallaplay.com")
    _mk(db, "legacy", user_id=None)  # untagged

    ids = {
        s["id"]
        for s in db.list_sessions_rich(source="acp", owner="alice@yallaplay.com")
    }
    # STRICT ownership: only the caller's own sessions. Untagged (legacy /
    # non-interactive) rows AND other owners' rows are hidden under "My".
    assert ids == {"mine"}


def test_no_owner_filter_shows_all(db):
    _mk(db, "mine", user_id="alice@yallaplay.com")
    _mk(db, "theirs", user_id="bob@yallaplay.com")

    ids = {s["id"] for s in db.list_sessions_rich(source="acp")}
    assert ids == {"mine", "theirs"}


def test_owner_filtered_listing_includes_requested_delegate_runs(db):
    _mk(db, "mine", user_id="alice@yallaplay.com")
    _mk(db, "theirs", user_id="bob@yallaplay.com")
    for parent in ("mine", "theirs"):
        db.create_session(
            f"{parent}-worker", source="subagent", parent_session_id=parent,
            model_config={"_delegate_from": parent},
        )

    rows = db.list_sessions_rich(
        owner="alice@yallaplay.com", include_subagents=True,
        project_compression_tips=False,
    )
    assert {row["id"] for row in rows} == {"mine", "mine-worker"}
    assert {row["id"] for row in db.list_sessions_rich(owner="alice@yallaplay.com")} == {"mine"}


@pytest.mark.parametrize("marker", ["_delegate_from", "_branched_from"])
def test_compressed_acp_parent_still_passes_soft_owner_to_children(db, marker):
    db.create_session("parent", source="acp", user_id="alice@yallaplay.com")
    db.end_session("parent", "compression")
    db.create_session(
        "worker", source="subagent", parent_session_id="parent",
        model_config={marker: "parent"},
    )
    db.create_session("grandchild", source="subagent", parent_session_id="worker")

    for sid in ("worker", "grandchild"):
        child = db.get_session(sid)
        assert child["user_id"] == "alice@yallaplay.com"
        assert child["session_key"] is None
        assert child["chat_id"] is None
