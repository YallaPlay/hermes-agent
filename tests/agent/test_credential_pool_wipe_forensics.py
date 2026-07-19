"""Forensic logging + refresh hardening at credential wipe/quarantine paths.

The openai-codex OAuth credential can silently vanish from a profile's
auth.json: the terminal-refresh quarantine path and the pool-persistence
drop path historically logged only at DEBUG, so a wiped credential left no
trace in errors.log or journald. These tests lock in:

* WARNING-level attribution (provider, entry id/source/label, error code,
  pid, HERMES_PROFILE) at every path that destroys credential state.
* Codex CLI token recovery is attempted BEFORE wiping on a terminal
  refresh error, so a single transient failure cannot destroy the only
  credential when a valid pair exists in ~/.codex/auth.json.
* A manual (independent-account) entry that hits a terminal refresh error
  is marked DEAD (visible in `auth list`, pruned after 24h) instead of
  being left silently in rotation.

Redaction safety: raw token values must never appear in the log output.
"""

from __future__ import annotations

import json
import logging
import os

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


_FAKE_ACCESS = "codex_at_LEAK_CANARY_access_0123456789"
_FAKE_REFRESH = "codex_rt_LEAK_CANARY_refresh_0123456789"


def _codex_auth_store(access_token: str, refresh_token: str) -> dict:
    return {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
            }
        },
    }


def _terminal_error():
    from hermes_cli.auth import AuthError

    return AuthError(
        "Refresh session has been revoked",
        provider="openai-codex",
        code="codex_refresh_failed",
        relogin_required=True,
    )


@pytest.fixture
def codex_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    _write_auth_store(tmp_path, _codex_auth_store(_FAKE_ACCESS, _FAKE_REFRESH))

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod

    # Deterministic: no Codex CLI recovery unless a test opts in.
    monkeypatch.setattr(auth_mod, "_recover_codex_tokens_from_cli", lambda reason: None)

    def _fail(*_a, **_kw):
        raise _terminal_error()

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _fail)

    pool = load_pool("openai-codex")
    assert pool.select() is not None
    return pool


def test_codex_terminal_wipe_logs_warning_with_context(codex_pool, caplog):
    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        assert codex_pool.try_refresh_current() is None

    text = caplog.text
    assert "terminally invalid" in text
    assert "openai-codex" in text
    assert "codex_refresh_failed" in text
    assert f"pid={os.getpid()}" in text
    assert "profile=" in text
    # The quarantined device_code entry removal must be attributed too.
    assert "device_code" in text
    # Raw token material never leaks into the forensic output.
    assert _FAKE_ACCESS not in text
    assert _FAKE_REFRESH not in text


def test_codex_terminal_refresh_recovers_from_cli_before_wiping(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    _write_auth_store(tmp_path, _codex_auth_store(_FAKE_ACCESS, _FAKE_REFRESH))

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod

    recovered_tokens = {
        "access_token": "recovered-access-token",
        "refresh_token": "recovered-refresh-token",
    }
    monkeypatch.setattr(
        auth_mod, "_recover_codex_tokens_from_cli", lambda reason: dict(recovered_tokens)
    )

    def _fail(*_a, **_kw):
        raise _terminal_error()

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _fail)

    pool = load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None

    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        refreshed = pool.try_refresh_current()

    # The entry survives with the recovered token pair instead of being wiped.
    assert refreshed is not None
    assert refreshed.access_token == "recovered-access-token"
    assert refreshed.refresh_token == "recovered-refresh-token"
    assert [entry.source for entry in pool.entries()] == ["device_code"]
    assert "adopting" in caplog.text.lower()

    # auth.json singleton tokens were NOT popped.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    tokens = auth_payload["providers"]["openai-codex"].get("tokens", {})
    assert tokens.get("access_token")


def test_codex_terminal_refresh_marks_manual_entry_dead_not_removed(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    # Pool-only manual entry, empty singleton — the `hermes auth add` shape.
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {"openai-codex": {}},
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "manual1",
                        "label": "openai-codex-oauth-1",
                        "source": "manual:device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": _FAKE_ACCESS,
                        "refresh_token": _FAKE_REFRESH,
                    }
                ]
            },
        },
    )

    from agent.credential_pool import STATUS_DEAD, load_pool
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_recover_codex_tokens_from_cli", lambda reason: None)

    def _fail(*_a, **_kw):
        raise _terminal_error()

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _fail)

    pool = load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None
    assert selected.source == "manual:device_code"

    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        assert pool.try_refresh_current() is None

    # The manual entry must SURVIVE (as DEAD) — absent is not diagnosable.
    entries = pool.entries()
    assert [entry.id for entry in entries] == ["manual1"]
    assert entries[0].last_status == STATUS_DEAD

    # Persisted state matches.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"]
    assert [entry["id"] for entry in persisted] == ["manual1"]
    assert persisted[0]["last_status"] == STATUS_DEAD

    assert "manual" in caplog.text
    assert "DEAD" in caplog.text


def test_write_credential_pool_logs_removed_ids(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "manual1",
                        "label": "openai-codex-oauth-1",
                        "source": "manual:device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": _FAKE_ACCESS,
                        "refresh_token": _FAKE_REFRESH,
                    }
                ]
            },
        },
    )

    from hermes_cli.auth import write_credential_pool

    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        write_credential_pool("openai-codex", [], removed_ids=["manual1"])

    text = caplog.text
    assert "manual1" in text
    assert "manual:device_code" in text
    assert f"pid={os.getpid()}" in text
    assert _FAKE_ACCESS not in text
    assert _FAKE_REFRESH not in text

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert auth_payload["credential_pool"]["openai-codex"] == []


def test_write_credential_pool_no_removed_ids_stays_quiet(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from hermes_cli.auth import write_credential_pool

    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        write_credential_pool(
            "openai-codex",
            [
                {
                    "id": "manual1",
                    "label": "l",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "priority": 0,
                    "access_token": "x",
                }
            ],
        )

    assert "removed" not in caplog.text.lower()


def test_clear_provider_auth_logs_dropped_pool_entries(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {"openai-codex": {"tokens": {"access_token": _FAKE_ACCESS}}},
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "manual1",
                        "label": "openai-codex-oauth-1",
                        "source": "manual:device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": _FAKE_ACCESS,
                        "refresh_token": _FAKE_REFRESH,
                    }
                ]
            },
        },
    )

    from hermes_cli.auth import clear_provider_auth

    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        assert clear_provider_auth("openai-codex") is True

    text = caplog.text
    assert "clear_provider_auth" in text
    assert "manual1" in text
    assert "manual:device_code" in text
    assert f"pid={os.getpid()}" in text
    assert _FAKE_ACCESS not in text

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert "openai-codex" not in auth_payload.get("credential_pool", {})
