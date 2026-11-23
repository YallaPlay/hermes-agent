"""SessionDB._own_profile_name: which profile owns a store, derived from its path."""
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    return root


def _own(db_dir: Path) -> str | None:
    db = SessionDB(db_dir / "state.db")
    try:
        return db._own_profile_name()
    finally:
        db.close()


def test_profile_under_plain_profiles_dir(native, monkeypatch):
    home = native / "profiles" / "lab"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _own(home) == "lab"


@pytest.mark.platforms("linux")
def test_profile_under_symlinked_profiles_dir(native, tmp_path, monkeypatch):
    """A store behind a symlinked ~/.hermes/profiles still names its profile.

    Returning None here made the gateway reject every session an in-process ACP adapter had
    stamped with the profile name (validate_legacy_row -> not_found).
    """
    real_profiles = tmp_path / "repo" / ".hermes" / "profiles"
    (real_profiles / "lab").mkdir(parents=True)
    (native / "profiles").symlink_to(real_profiles)
    for home in (real_profiles / "lab", native / "profiles" / "lab"):
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert _own(home) == "lab"


def test_default_root_store(native, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert _own(native) == "default"


def test_store_outside_profile_tree_is_none(native, tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    elsewhere = tmp_path / "elsewhere" / "lab"
    elsewhere.mkdir(parents=True)
    assert _own(elsewhere) is None
