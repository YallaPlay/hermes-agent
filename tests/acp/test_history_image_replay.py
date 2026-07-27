"""ACP history replay must re-emit images, not just their text placeholder.

Regression cover for: an image attached to a chat turn renders live but
vanishes after a window reload, because `_replay_session_history` only ever
built `TextContentBlock` chunks. Two replay sources have to work:

* in-memory `state.history` — content-part list with base64 data URLs intact.
* DB-sourced rows (mid-turn replay, or a session another process ran) — content
  already collapsed to `[screenshot]` text, with the bytes reachable only via
  the `display_metadata["images"]` references the flush path writes.
"""

import base64

import pytest
from unittest.mock import AsyncMock, MagicMock

import acp
from acp.schema import ImageContentBlock, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from agent.history_media import DISPLAY_METADATA_KEY

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7"
    "1kAAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.fixture()
def agent():
    return HermesACPAgent(
        session_manager=SessionManager(agent_factory=lambda: MagicMock())
    )


@pytest.fixture()
def image_cache(tmp_path, monkeypatch):
    """Real image cache dir; returns a writer for cached image files."""
    cache = tmp_path / "images"
    cache.mkdir()
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", cache)

    def _write(name: str = "img_test.png", data: bytes = PNG_BYTES) -> str:
        path = cache / name
        path.write_bytes(data)
        return str(path)

    return _write


async def _replay(agent, history, *, is_running=False, db_rows=None):
    mock_conn = MagicMock(spec=acp.Client)
    mock_conn.session_update = AsyncMock()
    agent._conn = mock_conn
    resp = await agent.new_session(cwd="/tmp")
    state = agent.session_manager.get_session(resp.session_id)
    state.history = history
    state.is_running = is_running
    agent.session_manager.live_transcript_history = MagicMock(return_value=db_rows)
    mock_conn.session_update.reset_mock()
    await agent._replay_session_history(state)
    return mock_conn.session_update.await_args_list


def _blocks(calls, session_update):
    return [
        call.kwargs["update"].content
        for call in calls
        if getattr(call.kwargs.get("update"), "session_update", None) == session_update
    ]


class TestInMemoryImageReplay:
    @pytest.mark.asyncio
    async def test_user_image_part_replays_as_image_block(self, agent):
        calls = await _replay(
            agent,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                }
            ],
        )
        blocks = _blocks(calls, "user_message_chunk")
        assert isinstance(blocks[0], TextContentBlock)
        assert blocks[0].text == "what is this?"
        assert isinstance(blocks[1], ImageContentBlock)
        assert base64.b64decode(blocks[1].data) == PNG_BYTES
        assert blocks[1].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_image_chunk_carries_fork_coordinate(self, agent):
        """An image-only turn must still be forkable — the coordinate rides on
        every chunk of the entry, not only the text one."""
        calls = await _replay(
            agent,
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]}],
        )
        image_calls = [
            call
            for call in calls
            if isinstance(getattr(call.kwargs.get("update"), "content", None), ImageContentBlock)
        ]
        assert len(image_calls) == 1
        assert image_calls[0].kwargs["hermes"] == {"historyIndex": 0}

    @pytest.mark.asyncio
    async def test_assistant_image_part_replays_as_image_block(self, agent):
        calls = await _replay(
            agent,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "here you go"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                }
            ],
        )
        blocks = _blocks(calls, "agent_message_chunk")
        assert isinstance(blocks[0], TextContentBlock)
        assert isinstance(blocks[1], ImageContentBlock)

    @pytest.mark.asyncio
    async def test_text_only_history_emits_no_image_chunks(self, agent):
        calls = await _replay(agent, [{"role": "user", "content": "hello"}])
        assert all(
            isinstance(getattr(call.kwargs.get("update"), "content", None), TextContentBlock)
            for call in calls
        )

    @pytest.mark.asyncio
    async def test_oversized_inline_image_is_skipped(self, agent):
        from acp_adapter import server as server_mod

        big = b"\x89PNG" + b"\x00" * (server_mod._MAX_ACP_RESOURCE_BYTES + 1)
        url = "data:image/png;base64," + base64.b64encode(big).decode()
        calls = await _replay(
            agent,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "big one"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        )
        assert not [
            call
            for call in calls
            if isinstance(getattr(call.kwargs.get("update"), "content", None), ImageContentBlock)
        ]


class TestPersistedRefImageReplay:
    """The DB row has no bytes — only display_metadata references."""

    @staticmethod
    def _db_row(path):
        return {
            "role": "user",
            "content": "This is the orange rooms\n[screenshot]",
            "display_metadata": {
                DISPLAY_METADATA_KEY: [{"path": path, "mimeType": "image/png"}]
            },
        }

    @pytest.mark.asyncio
    async def test_midturn_db_replay_reads_image_from_cache(self, agent, image_cache):
        path = image_cache()
        calls = await _replay(
            agent, [], is_running=True, db_rows=[self._db_row(path)]
        )
        blocks = _blocks(calls, "user_message_chunk")
        assert isinstance(blocks[0], TextContentBlock)
        assert isinstance(blocks[1], ImageContentBlock)
        assert base64.b64decode(blocks[1].data) == PNG_BYTES

    @pytest.mark.asyncio
    async def test_idle_adopted_db_replay_reads_image_from_cache(
        self, agent, image_cache
    ):
        """Session advanced in another process (Slack bot / gateway): the
        adopted DB rows must render their images too."""
        path = image_cache()
        calls = await _replay(
            agent,
            [],
            is_running=False,
            db_rows=[self._db_row(path), {"role": "assistant", "content": "ok"}],
        )
        assert any(
            isinstance(getattr(call.kwargs.get("update"), "content", None), ImageContentBlock)
            for call in calls
        )

    @pytest.mark.asyncio
    async def test_expired_cache_file_degrades_to_text_only(self, agent, image_cache):
        """The image cache has a 24h TTL cleanup — a missing file must leave
        the text placeholder intact, never raise or abort the replay."""
        image_cache()  # establish the cache dir; the referenced file is absent
        row = self._db_row("/nonexistent/img_gone.png")
        calls = await _replay(agent, [], is_running=True, db_rows=[row])
        blocks = _blocks(calls, "user_message_chunk")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContentBlock)

    @pytest.mark.asyncio
    async def test_ref_outside_cache_roots_is_refused(self, agent, image_cache, tmp_path):
        """display_metadata is data — a path outside the cache roots must not
        become an arbitrary-file-read primitive during transcript rendering."""
        image_cache()
        outside = tmp_path / "secret.png"
        outside.write_bytes(PNG_BYTES)
        calls = await _replay(
            agent, [], is_running=True, db_rows=[self._db_row(str(outside))]
        )
        assert not [
            call
            for call in calls
            if isinstance(getattr(call.kwargs.get("update"), "content", None), ImageContentBlock)
        ]

    @pytest.mark.asyncio
    async def test_inline_parts_win_over_refs(self, agent, image_cache):
        """When both sources are present (in-memory turn already flushed), use
        the in-memory bytes once — never emit the same image twice."""
        path = image_cache()
        calls = await _replay(
            agent,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                    "display_metadata": {
                        DISPLAY_METADATA_KEY: [{"path": path, "mimeType": "image/png"}]
                    },
                }
            ],
        )
        image_blocks = [
            call.kwargs["update"].content
            for call in calls
            if isinstance(getattr(call.kwargs.get("update"), "content", None), ImageContentBlock)
        ]
        assert len(image_blocks) == 1
