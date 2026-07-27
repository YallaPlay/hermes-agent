"""Durable image references in persisted history (agent/history_media.py).

Regression cover for the class of bug where an image attached to a chat turn
renders live but vanishes after a session reload, because the flush path
collapsed the content-part list to ``[screenshot]`` text with no surviving
handle on the bytes.
"""

import base64

import pytest

from agent.history_media import (
    DISPLAY_METADATA_KEY,
    IMAGE_PLACEHOLDER,
    content_parts_to_text_and_image_refs,
    decode_data_url,
    image_refs_from_message,
    merge_image_refs_into_display_metadata,
)

# Smallest valid PNG (1x1). cache_image_from_bytes sniffs magic bytes and
# refuses non-image data, so the payload has to be a real image.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7"
    "1kAAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.fixture(autouse=True)
def image_cache_dir(tmp_path, monkeypatch):
    """Point the shared image cache at a tmp dir.

    ``_resolve_cache_dir`` only honors an override when the module CONSTANT
    differs from its import-time default — setting an env var does nothing.
    """
    monkeypatch.setattr(
        "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "images"
    )
    return tmp_path / "images"


class TestDecodeDataUrl:
    def test_decodes_base64_image_data_url(self):
        decoded = decode_data_url(PNG_DATA_URL)
        assert decoded is not None
        data, mime = decoded
        assert data == PNG_BYTES
        assert mime == "image/png"

    @pytest.mark.parametrize(
        "value",
        [
            "https://example.com/shot.png",
            "/tmp/shot.png",
            "data:image/png,notbase64",
            "data:image/png;base64,!!!!",
            "data:image/png;base64,",
            "",
            None,
        ],
    )
    def test_rejects_non_data_urls_and_garbage(self, value):
        assert decode_data_url(value) is None


class TestContentPartsToTextAndImageRefs:
    def test_persisted_text_keeps_placeholder_shape(self):
        """The model-visible transcript bytes must not change — prompt-cache
        stability and existing expectations depend on the exact placeholder."""
        text, _refs = content_parts_to_text_and_image_refs(
            [
                {"type": "text", "text": "Describe this screenshot"},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ]
        )
        assert text == f"Describe this screenshot\n{IMAGE_PLACEHOLDER}"

    def test_data_url_image_is_cached_and_referenced(self, image_cache_dir):
        _text, refs = content_parts_to_text_and_image_refs(
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ]
        )
        assert len(refs) == 1
        ref = refs[0]
        assert ref["mimeType"] == "image/png"
        assert ref["bytes"] == len(PNG_BYTES)
        cached = image_cache_dir / ref["path"].rsplit("/", 1)[-1]
        assert cached.read_bytes() == PNG_BYTES

    def test_remote_image_keeps_url_without_caching(self):
        _text, refs = content_parts_to_text_and_image_refs(
            [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
        )
        assert refs == [{"url": "https://x.test/a.png", "mimeType": ""}]

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "input_image", "image_url": PNG_DATA_URL},
            {"type": "image", "source": {"data": PNG_DATA_URL}},
            {
                "type": "image",
                "source": {
                    "data": base64.b64encode(PNG_BYTES).decode(),
                    "media_type": "image/png",
                },
            },
        ],
    )
    def test_handles_every_transport_image_shape(self, part):
        _text, refs = content_parts_to_text_and_image_refs([part])
        assert len(refs) == 1
        assert "path" in refs[0]

    def test_undecodable_image_still_emits_placeholder_without_ref(self):
        text, refs = content_parts_to_text_and_image_refs(
            [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@"}},
            ]
        )
        assert text == f"hi\n{IMAGE_PLACEHOLDER}"
        assert refs == []

    def test_cache_failure_never_breaks_the_flush(self, monkeypatch):
        def boom(*_a, **_kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "gateway.platforms.base.cache_image_from_bytes", boom
        )
        text, refs = content_parts_to_text_and_image_refs(
            [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ]
        )
        assert text == f"hi\n{IMAGE_PLACEHOLDER}"
        assert refs == []

    def test_text_only_parts_produce_no_refs(self):
        text, refs = content_parts_to_text_and_image_refs(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        )
        assert text == "a\nb"
        assert refs == []

    def test_empty_content_returns_none_text(self):
        assert content_parts_to_text_and_image_refs([]) == (None, [])


class TestDisplayMetadataRoundTrip:
    def test_merge_preserves_existing_keys(self):
        merged = merge_image_refs_into_display_metadata(
            {"delegation_id": "d1"}, [{"path": "/tmp/a.png"}]
        )
        assert merged == {
            "delegation_id": "d1",
            DISPLAY_METADATA_KEY: [{"path": "/tmp/a.png"}],
        }

    def test_merge_without_refs_keeps_metadata_null(self):
        assert merge_image_refs_into_display_metadata(None, []) is None

    def test_merge_without_refs_preserves_existing_metadata(self):
        assert merge_image_refs_into_display_metadata({"a": 1}, []) == {"a": 1}

    def test_refs_read_back_from_persisted_message(self):
        refs = [{"path": "/tmp/a.png", "mimeType": "image/png"}]
        message = {
            "role": "user",
            "content": "hi\n[screenshot]",
            "display_metadata": {DISPLAY_METADATA_KEY: refs},
        }
        assert image_refs_from_message(message) == refs

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "user", "content": "hi"},
            {"role": "user", "display_metadata": None},
            {"role": "user", "display_metadata": {"images": "nope"}},
            {"role": "user", "display_metadata": {"images": ["nope", 3]}},
        ],
    )
    def test_missing_or_malformed_refs_read_as_empty(self, message):
        assert image_refs_from_message(message) == []
