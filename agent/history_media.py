"""Durable references for images that appear in persisted history messages.

A user turn carrying an attachment arrives as an OpenAI-style content LIST
(``[{"type": "text", ...}, {"type": "image_url", ...}]``). The session DB
stores one text column per message, so the flush path collapses that list to
text and replaces every image part with a ``[screenshot]`` placeholder — the
base64 payload would bloat the DB (and prompt-cache-stable replay only needs
the text bytes that were sent).

The placeholder alone is a dead end: after the turn, nothing on disk or in
the DB points at the bytes, so any surface rebuilding the transcript from
persisted history (ACP ``session/load`` replay, web UI, CLI resume) can never
show the image again. These helpers keep the model-visible text EXACTLY as
before while writing the bytes to the shared image cache and returning
display-only references for ``messages.display_metadata``.

Display metadata is never sent to a provider (``conversation_loop`` pops
``display_metadata`` from every outgoing copy), so this changes presentation
only — never model context.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Content part types that carry an image across the transports Hermes speaks:
# chat.completions (``image_url``), Responses API (``input_image``) and
# Anthropic-style (``image``).
IMAGE_PART_TYPES = frozenset({"image", "image_url", "input_image"})

# Text stand-in written into the persisted content for each image part.
# Unchanged from the original flush behavior on purpose: it is part of the
# model-visible transcript after a reload, and prompt-cache stability depends
# on those bytes not moving.
IMAGE_PLACEHOLDER = "[screenshot]"

# Key under ``display_metadata`` holding the ordered image references.
DISPLAY_METADATA_KEY = "images"

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)


def _part_image_url(part: dict[str, Any]) -> str:
    """Return the URL/data-URL carried by an image content part, or ''.

    Handles the three shapes: ``{"image_url": {"url": ...}}`` (chat
    completions), ``{"image_url": "..."}`` / ``{"url": ...}`` (Responses API
    and normalized variants), and Anthropic-style
    ``{"source": {"data": ..., "media_type": ...}}``.
    """
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    elif isinstance(image_url, str):
        return image_url
    url = part.get("url")
    if isinstance(url, str):
        return url
    source = part.get("source")
    if isinstance(source, dict):
        data = source.get("data")
        if isinstance(data, str) and data:
            media_type = str(source.get("media_type") or "image/png")
            if data.startswith("data:"):
                return data
            return f"data:{media_type};base64,{data}"
        source_url = source.get("url")
        if isinstance(source_url, str):
            return source_url
    return ""


def decode_data_url(url: str) -> tuple[bytes, str] | None:
    """Decode a base64 ``data:`` URL into ``(bytes, mime_type)``.

    Returns ``None`` for anything that isn't a decodable base64 data URL
    (remote http(s) URLs, plain paths, malformed payloads).
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    match = _DATA_URL_RE.match(url)
    if match is None or not match.group("b64"):
        return None
    payload = match.group("payload") or ""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not data:
        return None
    return data, (match.group("mime") or "image/png")


def _extension_for_mime(mime_type: str) -> str:
    """Map an image mime type to a cache-file extension."""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    return mapping.get((mime_type or "").strip().lower(), ".png")


def _cache_image_bytes(data: bytes, mime_type: str) -> str | None:
    """Write image bytes to the history image cache; return the path or None.

    Uses the ``history/`` cache (session-retention lifetime), NOT the flat
    inbound-media cache — a transcript reference must outlive the 24h sweep
    that transient platform attachments get.

    Imported lazily (the gateway module pulls in platform config) and never
    allowed to fail the caller: persistence of a display-only reference must
    not be able to break a turn's transcript flush.
    """
    try:
        from gateway.platforms.base import cache_history_image_from_bytes

        return cache_history_image_from_bytes(data, _extension_for_mime(mime_type))
    except Exception:
        logger.debug("Could not cache history image (%s)", mime_type, exc_info=True)
        return None


def content_parts_to_text_and_image_refs(
    content: list[Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Collapse a content-part list to persisted text plus image references.

    Returns ``(text, refs)`` where ``text`` is the value to store in
    ``messages.content`` — identical to the historical behavior: text parts
    joined by newlines with one ``[screenshot]`` line per image part, or
    ``None`` when nothing textual remains — and ``refs`` is the display-only
    reference list for ``display_metadata["images"]``.

    Each ref carries ``mimeType`` plus one durable handle:

    * ``path`` — data-URL bytes written to the image cache.
    * ``url`` — remote http(s) image, already durably addressable.

    Image parts whose bytes cannot be decoded or cached contribute their
    placeholder line but no ref, so the persisted text never changes shape
    because caching failed.
    """
    text_lines: list[str] = []
    refs: list[dict[str, Any]] = []

    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_lines.append(str(part.get("text", "")))
            continue
        if part_type not in IMAGE_PART_TYPES:
            continue

        text_lines.append(IMAGE_PLACEHOLDER)
        url = _part_image_url(part)
        if url.startswith(("http://", "https://")):
            refs.append({"url": url, "mimeType": str(part.get("mime_type") or "")})
            continue
        decoded = decode_data_url(url)
        if decoded is None:
            continue
        data, mime_type = decoded
        path = _cache_image_bytes(data, mime_type)
        if path is None:
            continue
        refs.append({"path": path, "mimeType": mime_type, "bytes": len(data)})

    return ("\n".join(text_lines) if text_lines else None), refs


def merge_image_refs_into_display_metadata(
    existing: Any, refs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Attach *refs* to a message's display metadata without losing keys.

    Returns the metadata dict to persist, or ``None`` when there is nothing
    to store (keeping the column NULL for the overwhelming majority of rows).
    """
    if not refs:
        return existing if isinstance(existing, dict) and existing else None
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged[DISPLAY_METADATA_KEY] = refs
    return merged


def image_refs_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the display-only image refs stored on a persisted message."""
    metadata = message.get("display_metadata")
    if not isinstance(metadata, dict):
        return []
    refs = metadata.get(DISPLAY_METADATA_KEY)
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, dict)]


def _safe_image_roots() -> list[Any]:
    """Directories a persisted image reference may be read back from.

    Refs are written by :func:`content_parts_to_text_and_image_refs` into the
    shared image cache, so the cache roots are the only legitimate sources.
    Confining reads to them keeps a hand-edited/imported ``display_metadata``
    from turning transcript rendering into an arbitrary-file-read primitive.
    """
    try:
        from gateway.platforms.base import (
            MEDIA_DELIVERY_SAFE_ROOTS,
            get_history_image_cache_dir,
            get_image_cache_dir,
        )
    except Exception:
        return []
    # get_image_cache_dir() resolves against the ACTIVE profile (and honors a
    # test monkeypatch of the module constant); MEDIA_DELIVERY_SAFE_ROOTS is a
    # frozen import-time tuple covering the legacy/canonical cache layouts.
    # The history/ subdir is covered by its parent, but name it explicitly so
    # a future layout change can't silently drop transcript images.
    roots = [get_image_cache_dir(), get_history_image_cache_dir()]
    roots.extend(MEDIA_DELIVERY_SAFE_ROOTS)
    return roots


def image_bytes_from_ref(
    ref: dict[str, Any], *, max_bytes: int
) -> tuple[bytes, str] | None:
    """Load the bytes for one persisted image ref, or ``None``.

    Returns ``None`` — never raises — for refs that carry no local path, point
    outside the cache roots, no longer exist (the cache has a TTL cleanup), or
    exceed *max_bytes*. Remote ``url`` refs are not fetched: replay must not
    perform network I/O.
    """
    from pathlib import Path

    raw_path = ref.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    try:
        path = Path(raw_path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None

    roots = _safe_image_roots()
    if not roots:
        return None
    if not any(_is_relative_to(path, Path(root)) for root in roots):
        logger.debug("Refusing history image outside the cache roots: %s", path)
        return None

    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None

    mime_type = ref.get("mimeType")
    if not isinstance(mime_type, str) or not mime_type:
        mime_type = _mime_for_extension(path.suffix)
    return data, mime_type


def _is_relative_to(path: Any, root: Any) -> bool:
    """``Path.is_relative_to`` without the 3.9+ requirement gymnastics."""
    try:
        path.relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _mime_for_extension(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get((suffix or "").lower(), "image/png")


def inline_image_bytes_from_content(
    content: Any, *, max_bytes: int
) -> list[tuple[bytes, str]]:
    """Decode image parts still present on an IN-MEMORY history message.

    A session replayed from this process's ``state.history`` (rather than the
    DB) still holds the original content-part list with its base64 data URLs —
    no cache round-trip needed. Oversized payloads are skipped so one huge
    attachment can't stall a session load.
    """
    if not isinstance(content, list):
        return []
    images: list[tuple[bytes, str]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in IMAGE_PART_TYPES:
            continue
        decoded = decode_data_url(_part_image_url(part))
        if decoded is None:
            continue
        data, mime_type = decoded
        if len(data) > max_bytes:
            logger.debug("Skipping oversized inline history image (%d bytes)", len(data))
            continue
        images.append((data, mime_type))
    return images

