"""multiplexer_root_for maps a profile under a symlinked profiles dir to the default root."""
from __future__ import annotations

from pathlib import Path

import hermes_constants
from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for


def _layout(tmp_path: Path, monkeypatch, *, link: bool) -> tuple[Path, Path]:
    default = tmp_path / "home" / ".hermes"
    default.mkdir(parents=True)
    (default / "config.yaml").write_text("{}\n")
    real_root = tmp_path / "repo" / ".hermes"
    real_profiles = real_root / "profiles"
    (real_profiles / "lab").mkdir(parents=True)
    (real_profiles / "lab" / "config.yaml").write_text("{}\n")
    if link:
        (default / "profiles").symlink_to(real_profiles)
    else:
        (default / "profiles").mkdir()
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: default)
    return default, real_profiles / "lab"


def test_resolved_profile_under_symlinked_profiles_maps_to_default_root(tmp_path, monkeypatch):
    default, resolved_home = _layout(tmp_path, monkeypatch, link=True)
    assert multiplexer_root_for(resolved_home) == default


def test_profile_under_unlinked_profiles_keeps_parent_root(tmp_path, monkeypatch):
    _default, resolved_home = _layout(tmp_path, monkeypatch, link=False)
    assert multiplexer_root_for(resolved_home) == resolved_home.parent.parent


def test_classic_layout_unchanged(tmp_path, monkeypatch):
    default = tmp_path / ".hermes"
    (default / "profiles" / "lab").mkdir(parents=True)
    (default / "config.yaml").write_text("{}\n")
    (default / "profiles" / "lab" / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: default)
    assert multiplexer_root_for(default / "profiles" / "lab") == default


def test_symlinked_profiles_found_when_process_home_is_repointed(tmp_path, monkeypatch):
    """A service that runs with HOME=<scratch> (the Slack bot) must still find the account's root.

    The bot spawns `hermes -p <profile> acp` with HOME pointing at its own scratch home. The
    platform default then names <scratch>/.hermes, which has no `profiles` link, and discovery
    looked for the multiplexer in the link target's parent forever ("Gateway starting").
    """
    default, resolved_home = _layout(tmp_path, monkeypatch, link=True)
    scratch = tmp_path / "scratch-home" / ".hermes"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: scratch)
    monkeypatch.setattr(hermes_constants, "account_default_hermes_home", lambda: default)
    assert multiplexer_root_for(resolved_home) == default
