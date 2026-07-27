"""The session-prune startup hook also expires transcript images.

Transcript-referenced images are part of session history. If `sessions.
auto_prune` deletes old sessions but nothing sweeps their images, the cache
grows forever; if a sweep ran while auto_prune was off, images would vanish
from sessions that are still present. Both directions are pinned here.
"""

from unittest.mock import MagicMock, patch

import pytest

import cli


@pytest.fixture()
def session_db():
    db = MagicMock()
    db.get_meta.return_value = "1"  # skip the one-time backfills
    return db


def _run(session_db, sessions_cfg):
    with (
        patch(
            "hermes_cli.config.load_config",
            return_value={"sessions": sessions_cfg},
        ),
        patch("gateway.platforms.base.cleanup_history_image_cache") as sweep,
    ):
        cli._run_state_db_auto_maintenance(session_db)
    return sweep


def test_auto_prune_on_sweeps_images_with_session_retention(session_db):
    sweep = _run(session_db, {"auto_prune": True, "retention_days": 30})
    session_db.maybe_auto_prune_and_vacuum.assert_called_once()
    sweep.assert_called_once_with(max_age_days=30)


def test_auto_prune_on_defaults_to_90_days(session_db):
    sweep = _run(session_db, {"auto_prune": True})
    sweep.assert_called_once_with(max_age_days=90)


def test_auto_prune_off_never_sweeps_images(session_db):
    """Sessions are kept forever in this configuration — deleting their images
    would be silent, asymmetric data loss."""
    sweep = _run(session_db, {"auto_prune": False, "retention_days": 30})
    session_db.maybe_auto_prune_and_vacuum.assert_not_called()
    sweep.assert_not_called()


def test_image_sweep_failure_never_blocks_startup(session_db):
    with (
        patch(
            "hermes_cli.config.load_config",
            return_value={"sessions": {"auto_prune": True, "retention_days": 90}},
        ),
        patch(
            "gateway.platforms.base.cleanup_history_image_cache",
            side_effect=RuntimeError("cache unreadable"),
        ),
    ):
        cli._run_state_db_auto_maintenance(session_db)  # must not raise
    session_db.maybe_auto_prune_and_vacuum.assert_called_once()
