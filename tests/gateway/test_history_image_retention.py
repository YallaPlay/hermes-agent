"""Transcript-referenced images expire on the SESSION schedule, not 24h.

Inbound platform media (Discord/Slack attachments) is transient and correctly
swept after 24h. Images a persisted transcript points at are not: a session
reload rebuilds its bubbles from those files, so a 24h sweep would rot live
transcripts into dead `[screenshot]` placeholders. They live under
`<image cache>/history/` with their own retention policy.
"""

import base64
import time

import pytest

from gateway.platforms.base import (
    HISTORY_IMAGE_SUBDIR,
    cache_history_image_from_bytes,
    cleanup_history_image_cache,
    cleanup_image_cache,
    get_history_image_cache_dir,
    get_history_image_retention_days,
    get_image_cache_dir,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7"
    "1kAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def image_cache(tmp_path, monkeypatch):
    """Point the image cache at a tmp dir.

    ``_resolve_cache_dir`` only honors an override when the module CONSTANT
    differs from its import-time default — an env var does nothing.
    """
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "images")
    return tmp_path / "images"


def _age(path, days):
    old = time.time() - days * 86400
    import os

    os.utime(path, (old, old))


class TestHistoryImageCacheDir:
    def test_history_dir_is_a_subdir_of_the_image_cache(self, image_cache):
        assert get_history_image_cache_dir() == image_cache / HISTORY_IMAGE_SUBDIR
        assert get_history_image_cache_dir().is_dir()

    def test_cache_history_image_writes_under_history(self, image_cache):
        path = cache_history_image_from_bytes(PNG_BYTES, ".png")
        assert path.startswith(str(image_cache / HISTORY_IMAGE_SUBDIR))
        from pathlib import Path

        assert Path(path).read_bytes() == PNG_BYTES

    def test_cache_history_image_rejects_non_image_bytes(self):
        """Same magic-byte guard as the inbound path — an HTML error page must
        not be cached as a .png."""
        with pytest.raises(ValueError):
            cache_history_image_from_bytes(b"<html>error</html>", ".png")


class TestInboundSweepLeavesHistoryAlone:
    def test_24h_sweep_does_not_touch_history_images(self, image_cache):
        """THE regression: the inbound-media sweep must not delete an image a
        live session's transcript still references."""
        history_path = cache_history_image_from_bytes(PNG_BYTES, ".png")
        _age(history_path, days=30)

        transient = get_image_cache_dir() / "img_transient.png"
        transient.write_bytes(PNG_BYTES)
        _age(transient, days=30)

        removed = cleanup_image_cache(max_age_hours=24)

        from pathlib import Path

        assert removed == 1
        assert not transient.exists()
        assert Path(history_path).exists()

    def test_inbound_sweep_ignores_the_history_directory_itself(self, image_cache):
        """iterdir() sees the subdirectory; is_file() must skip it (an unlink
        on a directory would raise and abort the sweep)."""
        get_history_image_cache_dir()
        assert cleanup_image_cache(max_age_hours=0) == 0


class TestHistoryRetentionPolicy:
    def test_defaults_to_keep_forever_when_auto_prune_off(self, monkeypatch):
        """Sessions are never pruned by default, so their images must not be
        either — asymmetric deletion is silent data loss."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"sessions": {"auto_prune": False, "retention_days": 90}},
        )
        assert get_history_image_retention_days() is None

    def test_follows_session_retention_days_when_auto_prune_on(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"sessions": {"auto_prune": True, "retention_days": 30}},
        )
        assert get_history_image_retention_days() == 30

    def test_defaults_to_90_days_when_retention_unset(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"sessions": {"auto_prune": True}},
        )
        assert get_history_image_retention_days() == 90

    @pytest.mark.parametrize(
        "sessions",
        [{"auto_prune": True, "retention_days": "abc"}, {"auto_prune": True, "retention_days": 0}],
    )
    def test_malformed_retention_keeps_forever(self, monkeypatch, sessions):
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda *a, **kw: {"sessions": sessions}
        )
        assert get_history_image_retention_days() is None

    def test_unreadable_config_keeps_forever(self, monkeypatch):
        def boom(*_a, **_kw):
            raise RuntimeError("no config")

        monkeypatch.setattr("hermes_cli.config.load_config", boom)
        assert get_history_image_retention_days() is None


class TestHistoryImageSweep:
    def test_sweeps_files_past_the_session_retention_window(self, image_cache):
        old = cache_history_image_from_bytes(PNG_BYTES, ".png")
        fresh = cache_history_image_from_bytes(PNG_BYTES, ".png")
        _age(old, days=100)

        removed = cleanup_history_image_cache(max_age_days=90)

        from pathlib import Path

        assert removed == 1
        assert not Path(old).exists()
        assert Path(fresh).exists()

    def test_keep_forever_policy_is_a_noop(self, image_cache, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"sessions": {"auto_prune": False}},
        )
        ancient = cache_history_image_from_bytes(PNG_BYTES, ".png")
        _age(ancient, days=5000)

        assert cleanup_history_image_cache() == 0

        from pathlib import Path

        assert Path(ancient).exists()

    def test_resolves_policy_from_config_when_unspecified(self, image_cache, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"sessions": {"auto_prune": True, "retention_days": 10}},
        )
        old = cache_history_image_from_bytes(PNG_BYTES, ".png")
        _age(old, days=11)

        assert cleanup_history_image_cache() == 1
