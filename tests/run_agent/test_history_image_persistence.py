"""Flush path persists durable image references (run_agent._flush_messages_to_session_db).

A user turn with an attachment arrives as a content-part LIST. The flush
collapses it to text with `[screenshot]` placeholders — correct for model
context, but historically it left NO handle on the bytes, so a reloaded
session could never render the image again.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent
from agent.history_media import DISPLAY_METADATA_KEY

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7"
    "1kAAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {"name": n, "description": n, "parameters": {}},
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture(autouse=True)
def image_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "images")
    return tmp_path / "images"


def _flush_user_image_turn(agent, content):
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.session_id = "session-img"
    agent._last_flushed_db_idx = 0
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    msg = {"role": "user", "content": content}
    agent._flush_messages_to_session_db([msg], [])
    # The flush batches the turn's rows into ONE append_messages_batch call;
    # assert on the single row dict it carried (same keys the old per-row
    # append_message kwargs had).
    (row,) = agent._session_db.append_messages_batch.call_args.kwargs["messages"]
    return msg, row


def test_flush_keeps_screenshot_placeholder_text(agent):
    """The persisted model-visible content must not change."""
    _msg, kwargs = _flush_user_image_turn(
        agent,
        [
            {"type": "text", "text": "Describe this screenshot"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ],
    )
    assert kwargs["content"] == "Describe this screenshot\n[screenshot]"


def test_flush_persists_image_reference_in_display_metadata(agent, image_cache_dir):
    """The bytes land in the image cache and the row carries the path, so a
    reloaded transcript can render the image instead of a dead placeholder."""
    _msg, kwargs = _flush_user_image_turn(
        agent,
        [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ],
    )
    refs = kwargs["display_metadata"][DISPLAY_METADATA_KEY]
    assert len(refs) == 1
    assert refs[0]["mimeType"] == "image/png"
    # history/ subdir — session-retention lifetime, not the 24h media sweep.
    cached = image_cache_dir / "history" / refs[0]["path"].rsplit("/", 1)[-1]
    assert cached.read_bytes() == PNG_BYTES


def test_flush_stamps_live_message_so_history_rewrites_keep_refs(agent):
    """Compaction/rewind reinsert rows from the in-memory history; the refs
    must ride on the live dict or those rewrites drop the image."""
    msg, _kwargs = _flush_user_image_turn(
        agent, [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]
    )
    assert DISPLAY_METADATA_KEY in msg["display_metadata"]


def test_flush_text_only_turn_leaves_display_metadata_null(agent):
    """Overwhelming majority of rows: no image, no metadata column write."""
    _msg, kwargs = _flush_user_image_turn(agent, "just text")
    assert kwargs["display_metadata"] is None


def test_flush_text_part_list_leaves_display_metadata_null(agent):
    _msg, kwargs = _flush_user_image_turn(
        agent, [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )
    assert kwargs["content"] == "a\nb"
    assert kwargs["display_metadata"] is None


def test_flush_survives_image_cache_failure(agent, monkeypatch):
    """A cache write failure must degrade to the old behavior, never raise —
    _flush_messages_to_session_db swallowing the exception would silently
    drop the whole row."""
    monkeypatch.setattr(
        "gateway.platforms.base.cache_history_image_from_bytes",
        MagicMock(side_effect=RuntimeError("disk full")),
    )
    _msg, kwargs = _flush_user_image_turn(
        agent,
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ],
    )
    assert kwargs["content"] == "hi\n[screenshot]"
    assert kwargs["display_metadata"] is None


def test_flush_preserves_existing_display_metadata_keys(agent):
    """Other producers (async delegation) stamp display_metadata too."""
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.session_id = "session-img"
    agent._last_flushed_db_idx = 0
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    msg = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}],
        "display_metadata": {"delegation_id": "d1"},
    }
    agent._flush_messages_to_session_db([msg], [])
    (row,) = agent._session_db.append_messages_batch.call_args.kwargs["messages"]
    metadata = row["display_metadata"]
    assert metadata["delegation_id"] == "d1"
    assert len(metadata[DISPLAY_METADATA_KEY]) == 1


def test_flush_multimodal_tool_result_still_uses_text_summary(agent):
    """Tool results keep the existing summary path (not the image-ref path):
    their images are already artifact-backed elsewhere."""
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent.session_id = "session-img"
    agent._last_flushed_db_idx = 0
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    tool_msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "tool_name": "vision_analyze",
        "content": [
            {"type": "text", "text": "text_summary here"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ],
    }
    with patch.object(run_agent, "_is_multimodal_tool_result", return_value=True):
        with patch.object(
            run_agent, "_multimodal_text_summary", return_value="text_summary here"
        ):
            agent._flush_messages_to_session_db([tool_msg], [])
    (row,) = agent._session_db.append_messages_batch.call_args.kwargs["messages"]
    assert row["content"] == "text_summary here"
    assert row["display_metadata"] is None
