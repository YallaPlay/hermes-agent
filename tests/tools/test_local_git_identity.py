"""Git commit attribution from the authenticated session owner."""

import os
import subprocess

import pytest

import gateway.session_context as sc
from gateway.session_context import _VAR_MAP, set_session_vars
from tools.environments import local_git_identity as gi
from tools.environments.local import _make_run_env

_GIT_KEYS = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for key in _GIT_KEYS:
        monkeypatch.delenv(key, raising=False)
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    monkeypatch.setattr(gi, "_read_committer_config", lambda: ("Claudio", "claudio@yallaplay.com"))
    yield
    for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
        var.set(val)
    sc._session_context_engaged = saved_engaged


def test_email_owner_sets_author_and_configured_committer():
    env = {"HERMES_SESSION_USER_ID": "israel.lot@yallaplay.com"}
    gi.apply_git_identity_env(env)
    assert env["GIT_AUTHOR_NAME"] == "Israel Lot"
    assert env["GIT_AUTHOR_EMAIL"] == "israel.lot@yallaplay.com"
    assert env["GIT_COMMITTER_NAME"] == "Claudio"
    assert env["GIT_COMMITTER_EMAIL"] == "claudio@yallaplay.com"


@pytest.mark.parametrize("owner", ["", "U012ABCDEF", "not an email@x", "a@b"])
def test_non_email_owner_injects_nothing(owner):
    env = {"HERMES_SESSION_USER_ID": owner}
    gi.apply_git_identity_env(env)
    assert not any(k in env for k in _GIT_KEYS)


def test_explicit_exports_win():
    env = {"HERMES_SESSION_USER_ID": "a@yallaplay.com", "GIT_AUTHOR_EMAIL": "me@x.com",
           "GIT_COMMITTER_NAME": "Me"}
    gi.apply_git_identity_env(env)
    assert env["GIT_AUTHOR_EMAIL"] == "me@x.com"
    assert env["GIT_COMMITTER_NAME"] == "Me"


def test_empty_committer_config_leaves_committer_to_gitconfig(monkeypatch):
    monkeypatch.setattr(gi, "_read_committer_config", lambda: ("", ""))
    env = {"HERMES_SESSION_USER_ID": "a@yallaplay.com"}
    gi.apply_git_identity_env(env)
    assert env["GIT_AUTHOR_EMAIL"] == "a@yallaplay.com"
    assert "GIT_COMMITTER_NAME" not in env


def test_session_user_name_preferred_and_header_chars_stripped():
    env = {"HERMES_SESSION_USER_ID": "a@yallaplay.com", "HERMES_SESSION_USER_NAME": "Ann\n<Bee>"}
    gi.apply_git_identity_env(env)
    assert env["GIT_AUTHOR_NAME"] == "AnnBee"


def test_real_commit_through_run_env(tmp_path):
    set_session_vars(session_key="s1", session_id="s1", user_id="israel.lot@yallaplay.com")
    env = _make_run_env({})
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "--allow-empty", "-m", "x"],
                   check=True, env=env)
    out = subprocess.run(["git", "-C", str(tmp_path), "log", "-1", "--format=%an <%ae>|%cn <%ce>"],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert out == "Israel Lot <israel.lot@yallaplay.com>|Claudio <claudio@yallaplay.com>"
