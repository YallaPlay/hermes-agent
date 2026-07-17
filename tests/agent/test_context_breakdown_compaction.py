"""Tests for the compaction-events field on the context breakdown payload."""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.context_breakdown import compute_session_context_breakdown
from agent.context_compressor import SUMMARY_PREFIX
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _make_agent(
    *,
    session_db=None,
    session_id="sess-1",
    compression_count=0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.session_id = session_id
    agent._session_db = session_db
    agent.context_compressor = MagicMock(
        context_length=200_000,
        last_prompt_tokens=0,
        compression_count=compression_count,
    )
    return agent, {"stable": "identity", "context": "", "volatile": "now"}


def _summary_text(n: int = 1) -> str:
    return f"{SUMMARY_PREFIX}\nsummary body number {n}"


def test_compaction_none_without_history_or_failure(db):
    db.create_session("sess-1", source="acp")
    agent, parts = _make_agent(session_db=db)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["compaction"] is None


def test_compaction_events_derived_from_summary_markers(db):
    db.create_session("sess-1", source="acp")
    # Simulate two in-place compaction boundaries: turns, boundary 1 summary,
    # more turns, boundary 2 summary. archive_and_compact soft-archives the
    # pre-boundary rows (compacted=1) and inserts the compacted set.
    db.append_message("sess-1", "user", "turn one")
    db.append_message("sess-1", "assistant", "reply one")
    # Explicit timestamps: real boundaries are minutes apart; rows written in
    # the same second are grouped as one logical event, so keep them distinct.
    first_boundary = time.time() - 120
    second_boundary = time.time() - 30
    db.archive_and_compact(
        "sess-1",
        [
            {"role": "user", "content": _summary_text(1), "timestamp": first_boundary},
            {"role": "user", "content": "turn two", "timestamp": first_boundary},
            {"role": "assistant", "content": "reply two", "timestamp": first_boundary},
        ],
    )
    db.archive_and_compact(
        "sess-1",
        [
            {"role": "user", "content": _summary_text(2), "timestamp": second_boundary},
            {"role": "user", "content": "turn three", "timestamp": second_boundary},
        ],
    )
    del first_boundary
    agent, parts = _make_agent(session_db=db, compression_count=2)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    compaction = data["compaction"]
    assert compaction is not None
    assert compaction["count"] == 2
    assert len(compaction["events"]) == 2
    first, second = compaction["events"]
    # Boundary 1 archived the 2 pre-compaction turns; boundary 2 archived the
    # first compacted set — its prior summary handoff is excluded (id ==
    # previous boundary), leaving the 2 real turns (turn two + reply two).
    assert first["messages_before"] == 2
    assert second["messages_before"] == 2
    assert first["summary_tokens"] > 0
    assert first["timestamp"] is not None
    assert "failure" not in compaction


def test_compaction_count_prefers_live_compressor_counter(db):
    # DB-reloaded session: markers absent (e.g. rotation-mode parent holds
    # them) but the live compressor already compacted this process lifetime.
    db.create_session("sess-1", source="acp")
    agent, parts = _make_agent(session_db=db, compression_count=3)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["compaction"]["count"] == 3
    assert data["compaction"]["events"] == []


def test_compaction_failure_cooldown_surfaces(db):
    db.create_session("sess-1", source="acp")
    db.record_compression_failure_cooldown(
        "sess-1", time.time() + 300, error="aux model exploded"
    )
    agent, parts = _make_agent(session_db=db)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    compaction = data["compaction"]
    assert compaction is not None
    assert compaction["count"] == 0
    failure = compaction["failure"]
    assert failure["error"] == "aux model exploded"
    assert 0 < failure["remaining_seconds"] <= 300


def test_compaction_none_without_session_db():
    agent, parts = _make_agent(session_db=None)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["compaction"] is None


def test_compaction_ignores_non_summary_mentions(db):
    # A message that merely quotes the delimiter text must not classify.
    db.create_session("sess-1", source="acp")
    db.append_message(
        "sess-1",
        "user",
        "docs say [END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW] is a marker",
    )
    agent, parts = _make_agent(session_db=db)

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, [])

    assert data["compaction"] is None
