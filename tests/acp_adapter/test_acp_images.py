import base64

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from acp_adapter.server import (
    HermesACPAgent,
    _content_blocks_to_openai_user_content,
    _extract_text,
    _prompt_has_content,
)


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx_only_prompt():
    # Real .docx bytes are a zip; a NUL byte is enough to trip the binary sniff.
    blob = base64.b64encode(b"PK\x03\x04\x00\x00fake-docx").decode("ascii")
    return [
        EmbeddedResourceContentBlock(
            type="resource",
            resource=BlobResourceContents(uri="report.docx", blob=blob, mimeType=_DOCX_MIME),
        )
    ]


def test_binary_attachment_only_prompt_counts_as_content():
    # Regression: a .docx uploaded with no caption converts to a plain text
    # STRING (attached-file header + saved-to-cache note), has no
    # TextContentBlock, and used to be dropped as an "empty prompt".
    prompt = _docx_only_prompt()
    user_text = _extract_text(prompt).strip()
    user_content = _content_blocks_to_openai_user_content(prompt)

    assert user_text == ""
    assert isinstance(user_content, str)
    assert "[Attached file: report.docx]" in user_content
    assert _prompt_has_content(user_text, user_content) is True


def test_prompt_has_content_rejects_truly_empty_prompts():
    assert _prompt_has_content("", "") is False
    assert _prompt_has_content("", "   \n") is False
    assert _prompt_has_content("", []) is False
    assert _prompt_has_content("hi", "hi") is True
    assert _prompt_has_content("", [{"type": "image_url", "image_url": {"url": "data:,"}}]) is True


def test_acp_image_blocks_convert_to_openai_multimodal_content():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="What is in this image?"),
        ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png"),
    ])

    assert content == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        },
    ]


def test_text_only_acp_blocks_stay_string_for_legacy_prompt_path():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="/help"),
    ])

    assert content == "/help"


def test_acp_resource_link_file_is_inlined_as_text(tmp_path):
    attached = tmp_path / "notes.md"
    attached.write_text("# Notes\n\nAttached file body", encoding="utf-8")

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="Please read this file"),
        ResourceContentBlock(
            type="resource_link",
            name="notes.md",
            title="Project notes",
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert content == (
        "Please read this file\n"
        "[Attached file: Project notes (notes.md)]\n"
        f"URI: {attached.as_uri()}\n\n"
        "# Notes\n\nAttached file body"
    )




@pytest.mark.asyncio
async def test_initialize_advertises_image_prompt_capability():
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities.image is True


# 1x1 transparent PNG — smallest valid image payload for inlining tests.
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)






