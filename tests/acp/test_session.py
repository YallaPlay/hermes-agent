"""Tests for acp_adapter.session — SessionManager and SessionState."""

import contextlib
import io
import json
import time
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from acp_adapter import session as acp_session
from acp_adapter.session import SessionManager, SessionState
from hermes_state import SessionDB


def _mock_agent():
    return MagicMock(name="MockAIAgent")


@pytest.fixture()
def manager():
    """SessionManager with a mock agent factory (avoids needing API keys)."""
    return SessionManager(agent_factory=_mock_agent)


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session_returns_state(self, manager):
        state = manager.create_session(cwd="/tmp/work")
        assert isinstance(state, SessionState)
        assert state.cwd == "/tmp/work"
        assert state.session_id
        assert state.history == []
        assert state.agent is not None

    def test_create_session_registers_task_cwd(self, manager, monkeypatch):
        calls = []
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: calls.append((task_id, cwd)))
        state = manager.create_session(cwd="/tmp/work")
        assert calls == [(state.session_id, "/tmp/work")]


    def test_register_task_cwd_translates_windows_drive_for_wsl_tools(self, monkeypatch):
        captured = {}

        def fake_register_task_env_overrides(task_id, overrides):
            captured["task_id"] = task_id
            captured["overrides"] = overrides

        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        monkeypatch.setattr(
            "tools.terminal_tool.register_task_env_overrides",
            fake_register_task_env_overrides,
        )

        acp_session._register_task_cwd("session-1", r"E:\Projects\AI\paperclip")

        assert captured == {
            "task_id": "session-1",
            "overrides": {"cwd": "/mnt/e/Projects/AI/paperclip"},
        }

    def test_session_ids_are_unique(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        assert s1.session_id != s2.session_id

    def test_get_session(self, manager):
        state = manager.create_session()
        fetched = manager.get_session(state.session_id)
        assert fetched is state

    def test_get_nonexistent_session_returns_none(self, manager):
        assert manager.get_session("does-not-exist") is None

    def test_make_agent_stamps_session_cwd_for_codex_runtime(self, monkeypatch):
        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "acp_adapter.session.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "default": "fake-model",
                    "provider": "fake-provider",
                },
                "mcp_servers": {},
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.session_cwd == "/tmp/project"

    def test_make_agent_passes_config_disabled_toolsets(self, monkeypatch):
        """agent.disabled_toolsets from config.yaml must reach the ACP agent,
        matching the CLI surface. Regression: browser toolset stayed loaded in
        editor sessions despite agent.disabled_toolsets: [browser]."""

        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        cfg = {
            "model": {"default": "fake-model", "provider": "fake-provider"},
            "mcp_servers": {},
            "agent": {"disabled_toolsets": ["browser", ""]},
        }
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr("acp_adapter.session.load_config", lambda: cfg, raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.kwargs["disabled_toolsets"] == ["browser"]

    def test_make_agent_passes_config_max_turns(self, monkeypatch):
        """agent.max_turns from config.yaml must reach the ACP agent as
        max_iterations, matching the CLI surface (cli_agent_setup_mixin
        passes max_iterations=self.max_turns). Without this, editor sessions
        are always capped at the hardcoded default regardless of config."""

        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        cfg = {
            "model": {"default": "fake-model", "provider": "fake-provider"},
            "mcp_servers": {},
            "agent": {"max_turns": 150},
        }
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr("acp_adapter.session.load_config", lambda: cfg, raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.kwargs["max_iterations"] == 150

    def test_make_agent_omits_max_iterations_when_unset(self, monkeypatch):
        """No agent.max_turns in config → don't pass max_iterations at all,
        so AIAgent's own default applies (avoid re-stating it here)."""

        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        cfg = {
            "model": {"default": "fake-model", "provider": "fake-provider"},
            "mcp_servers": {},
            "agent": {},
        }
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr("acp_adapter.session.load_config", lambda: cfg, raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert "max_iterations" not in state.agent.kwargs

    def test_make_agent_ignores_invalid_max_turns(self, monkeypatch):
        """A junk agent.max_turns value must not crash agent creation."""

        class FakeAgent:
            model = "fake-model"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        cfg = {
            "model": {"default": "fake-model", "provider": "fake-provider"},
            "mcp_servers": {},
            "agent": {"max_turns": "not-a-number"},
        }
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr("acp_adapter.session.load_config", lambda: cfg, raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None: {
                "provider": requested,
                "api_mode": "codex_app_server",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
            },
        )
        monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert "max_iterations" not in state.agent.kwargs




# ---------------------------------------------------------------------------
# WSL cwd translation
# ---------------------------------------------------------------------------


class TestWslCwdTranslation:
    def test_translate_acp_cwd_converts_windows_drive_path_when_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd(r"E:\Projects\AI\paperclip") == "/mnt/e/Projects/AI/paperclip"

    def test_translate_acp_cwd_handles_forward_slashes_when_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd("D:/work/project") == "/mnt/d/work/project"

    def test_translate_acp_cwd_leaves_windows_drive_path_unchanged_off_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", False)

        assert acp_session._translate_acp_cwd(r"E:\Projects\AI\paperclip") == r"E:\Projects\AI\paperclip"

    def test_translate_acp_cwd_leaves_posix_path_unchanged_on_wsl(self, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        assert acp_session._translate_acp_cwd("/mnt/e/Projects/AI/paperclip") == "/mnt/e/Projects/AI/paperclip"

    def test_create_session_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)

        state = manager.create_session(cwd=r"E:\Projects\AI\paperclip")

        assert state.cwd == "/mnt/e/Projects/AI/paperclip"

    def test_fork_session_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        original = manager.create_session(cwd="/tmp/base")

        forked = manager.fork_session(original.session_id, cwd=r"D:\work\project")

        assert forked is not None
        assert forked.cwd == "/mnt/d/work/project"

    def test_update_cwd_stores_translated_cwd_on_wsl(self, manager, monkeypatch):
        monkeypatch.setattr("hermes_constants._wsl_detected", True)
        state = manager.create_session(cwd="/tmp/old")

        updated = manager.update_cwd(state.session_id, cwd=r"C:\Users\foo\project")

        assert updated is not None
        assert updated.cwd == "/mnt/c/Users/foo/project"

# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


class TestForkSession:
    def test_fork_session_deep_copies_history(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})
        original.history.append({"role": "assistant", "content": "hi"})

        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None

        # History should be equal in content
        assert len(forked.history) == 2
        assert forked.history[0]["content"] == "hello"

        # But a deep copy — mutating one doesn't affect the other
        forked.history.append({"role": "user", "content": "extra"})
        assert len(original.history) == 2
        assert len(forked.history) == 3

    def test_fork_session_has_new_id(self, manager):
        original = manager.create_session()
        forked = manager.fork_session(original.session_id)
        assert forked is not None
        assert forked.session_id != original.session_id

    def test_fork_session_records_parent_lineage(self, manager):
        original = manager.create_session()
        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None
        assert forked.parent_id == original.session_id
        # Persisted as a _forked_from marker in model_config (NOT
        # parent_session_id, which the listers treat as subagent lineage).
        db = manager._get_db()
        row = db.get_session(forked.session_id)
        mc = json.loads(row["model_config"])
        assert mc["_forked_from"] == original.session_id
        assert row["parent_session_id"] is None

    def test_fork_session_inherits_owner(self, manager):
        """A fork must carry the parent's owner: an untagged fork row is
        hidden by the strict "My Sessions" owner filter after a reload."""
        original = manager.create_session(cwd="/a", owner="me@yallaplay.com")
        forked = manager.fork_session(original.session_id, cwd="/a")
        assert forked is not None
        assert forked.owner == "me@yallaplay.com"
        db = manager._get_db()
        row = db.get_session(forked.session_id)
        assert row["user_id"] == "me@yallaplay.com"

    def test_fork_of_restored_session_inherits_owner(self, manager):
        """Restore drops the in-memory state; the fork must still pick the
        owner up from the persisted row."""
        original = manager.create_session(cwd="/a", owner="me@yallaplay.com")
        original.history.append({"role": "user", "content": "hello"})
        manager.save_session(original.session_id)
        with manager._lock:
            del manager._sessions[original.session_id]

        forked = manager.fork_session(original.session_id, cwd="/a")
        assert forked is not None
        assert forked.owner == "me@yallaplay.com"
        db = manager._get_db()
        assert db.get_session(forked.session_id)["user_id"] == "me@yallaplay.com"

    def test_fork_lineage_surfaces_in_list_sessions(self, manager):
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        forked = manager.fork_session(original.session_id, cwd="/a")
        listing = {s["session_id"]: s for s in manager.list_sessions()}
        assert listing[forked.session_id]["parent_id"] == original.session_id
        assert listing[original.session_id]["parent_id"] is None

    def test_fork_stamps_lineage_title_from_parent(self, manager):
        """A fork of a titled parent gets 'Parent Title #2' immediately, so
        the sidebar never shows the raw first-message preview for it."""
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        manager.save_session(original.session_id)
        db = manager._get_db()
        db.set_session_title(original.session_id, "My Investigation")

        forked = manager.fork_session(original.session_id, cwd="/a")
        assert db.get_session_title(forked.session_id) == "My Investigation #2"

        second = manager.fork_session(original.session_id, cwd="/a")
        assert db.get_session_title(second.session_id) == "My Investigation #3"

    def test_fork_of_untitled_parent_stays_untitled(self, manager):
        """No parent title → no stamp; the background derive backfill owns
        titling in that case."""
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        forked = manager.fork_session(original.session_id, cwd="/a")
        db = manager._get_db()
        assert db.get_session_title(forked.session_id) is None

    def test_fork_lineage_survives_restart(self, manager):
        """DB-only forks (post process restart) still report their parent."""
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        manager.save_session(original.session_id)
        forked = manager.fork_session(original.session_id, cwd="/a")
        fid = forked.session_id
        manager.save_session(fid)

        # Drop the fork from memory: list_sessions merges it from the DB row.
        with manager._lock:
            del manager._sessions[fid]
        listing = {s["session_id"]: s for s in manager.list_sessions()}
        assert listing[fid]["parent_id"] == original.session_id

        # And a restore rehydrates the in-memory lineage too.
        restored = manager.get_session(fid)
        assert restored is not None
        assert restored.parent_id == original.session_id

    def test_archive_in_memory_fork_hides_it_from_active_list(self, manager):
        """Regression: archiving a fork looked like a no-op in VS Code.

        A fork is born live in memory with a full copied history, and the
        in-memory branch of list_sessions used to hardcode archived=False
        (while its seen-id claim also blocked the DB merge), so the archived
        row kept reappearing in the active view and never showed in the
        archived view.
        """
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        manager.save_session(original.session_id)
        forked = manager.fork_session(original.session_id, cwd="/a")
        fid = forked.session_id

        assert manager.set_session_archived(fid, True)

        active_ids = {s["session_id"] for s in manager.list_sessions()}
        assert fid not in active_ids
        assert original.session_id in active_ids

        archived = {s["session_id"]: s for s in manager.list_sessions(archived_only=True)}
        assert fid in archived
        assert archived[fid]["archived"] is True

        # And unarchiving restores it to the active view.
        assert manager.set_session_archived(fid, False)
        active_ids = {s["session_id"] for s in manager.list_sessions()}
        assert fid in active_ids

    def test_include_archived_marks_in_memory_fork_row(self, manager):
        original = manager.create_session(cwd="/a")
        original.history.append({"role": "user", "content": "hello"})
        manager.save_session(original.session_id)
        forked = manager.fork_session(original.session_id, cwd="/a")
        manager.set_session_archived(forked.session_id, True)

        listing = {s["session_id"]: s for s in manager.list_sessions(include_archived=True)}
        assert listing[forked.session_id]["archived"] is True
        assert listing[original.session_id]["archived"] is False

    def test_fork_session_preserves_mode(self, manager):
        original = manager.create_session()
        original.mode = "acceptEdits"
        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None
        assert forked.mode == "acceptEdits"

    def test_fork_session_carries_parent_provider_routing(self, manager):
        """The fork agent must be built with the parent's provider/base_url/
        api_mode — not the config default. Regression: an openai-codex/
        gpt-5.6-sol parent forked into bedrock/gpt-5.6-sol and every turn
        400'd with "The provided model identifier is invalid"."""
        original = manager.create_session(cwd="/a")
        original.agent.provider = "openai-codex"
        original.agent.base_url = "https://chatgpt.com/backend-api/codex"
        original.agent.api_mode = "codex_responses"

        captured = {}
        real_factory = manager._agent_factory

        def spying_make_agent(**kwargs):
            captured.update(kwargs)
            return real_factory()

        with patch.object(manager, "_make_agent", side_effect=spying_make_agent):
            forked = manager.fork_session(original.session_id, cwd="/a")

        assert forked is not None
        assert captured["requested_provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_mode"] == "codex_responses"

    def test_fork_persist_writes_provider_routing_to_db(self, manager):
        """A fork that is never prompted again must still restore with its
        provider routing after a process restart: the create branch of
        _persist has to write the full metadata blob, not just cwd."""

        def _routed_agent():
            a = _mock_agent()
            a.provider = "openai-codex"
            a.base_url = "https://chatgpt.com/backend-api/codex"
            a.api_mode = "codex_responses"
            return a

        manager._agent_factory = _routed_agent
        original = manager.create_session(cwd="/a")

        forked = manager.fork_session(original.session_id, cwd="/a")
        assert forked is not None
        # fork_session persists once at creation; simulate "no further turns".
        db = manager._get_db()
        mc = json.loads(db.get_session(forked.session_id)["model_config"])
        assert mc["provider"] == "openai-codex"
        assert mc["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert mc["api_mode"] == "codex_responses"
        assert mc["_forked_from"] == original.session_id

    def test_fork_session_default_mode_stays_empty(self, manager):
        original = manager.create_session()
        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None
        assert forked.mode == ""

    def test_fork_nonexistent_returns_none(self, manager):
        assert manager.fork_session("bogus-id") is None

    def test_fork_session_keep_history_slices_prefix(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "first"})
        original.history.append({"role": "assistant", "content": "reply"})
        original.history.append({"role": "user", "content": "second"})
        original.history.append({"role": "assistant", "content": "reply 2"})

        forked = manager.fork_session(original.session_id, cwd="/new", keep_history=2)
        assert forked is not None

        assert len(forked.history) == 2
        assert forked.history[0]["content"] == "first"
        assert forked.history[1]["content"] == "reply"
        # Original is untouched.
        assert len(original.history) == 4

        # Still a deep copy — mutating the fork doesn't affect the original.
        forked.history[0]["content"] = "mutated"
        assert original.history[0]["content"] == "first"

    def test_fork_session_keep_history_zero_gives_empty_fork(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})

        forked = manager.fork_session(original.session_id, cwd="/new", keep_history=0)
        assert forked is not None
        assert forked.history == []
        assert len(original.history) == 1

    def test_fork_session_keep_history_beyond_length_copies_all(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})

        forked = manager.fork_session(original.session_id, cwd="/new", keep_history=99)
        assert forked is not None
        assert len(forked.history) == 1

    def test_fork_session_keep_history_none_copies_all(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})
        original.history.append({"role": "assistant", "content": "hi"})

        forked = manager.fork_session(original.session_id, cwd="/new", keep_history=None)
        assert forked is not None
        assert len(forked.history) == 2

    def test_fork_session_keep_history_negative_raises(self, manager):
        original = manager.create_session()
        original.history.append({"role": "user", "content": "hello"})

        with pytest.raises(ValueError, match="non-negative"):
            manager.fork_session(original.session_id, cwd="/new", keep_history=-1)


# ---------------------------------------------------------------------------
# list / cleanup / remove
# ---------------------------------------------------------------------------


class TestListAndCleanup:
    def test_list_sessions_empty(self, manager):
        assert manager.list_sessions() == []

    def test_list_sessions_returns_created(self, manager):
        s1 = manager.create_session(cwd="/a")
        s2 = manager.create_session(cwd="/b")
        s1.history.append({"role": "user", "content": "hello from a"})
        s2.history.append({"role": "user", "content": "hello from b"})
        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert s1.session_id in ids
        assert s2.session_id in ids
        assert len(listing) == 2

    def test_list_sessions_hides_empty_threads(self, manager):
        manager.create_session(cwd="/empty")
        assert manager.list_sessions() == []

    def test_list_sessions_flattens_multimodal_first_message(self, manager):
        """A multimodal first user message (text + image parts) must surface its
        text in the title, not the raw list repr (leaking-code-in-title bug)."""
        s = manager.create_session(cwd="/mm")
        s.history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "sometimes i send a screenshot"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        )
        listing = manager.list_sessions()
        assert len(listing) == 1
        title = listing[0]["title"]
        assert "sometimes i send a screenshot" in title
        assert "[{" not in title and "'type'" not in title

    def test_list_sessions_image_only_first_message_placeholder(self, manager):
        s = manager.create_session(cwd="/imgonly")
        s.history.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ],
            }
        )
        s.history.append({"role": "user", "content": "follow-up words"})
        listing = manager.list_sessions()
        assert len(listing) == 1
        # First user message is image-only → placeholder wins (first match).
        assert listing[0]["title"] == "[multimodal content]"

    def test_list_sessions_strips_owner_note_from_title(self, manager):
        """An owned session's first user message carries the authenticated-owner
        note; the fallback title must show the prompt, not the owner email."""
        s = manager.create_session(cwd="/owned")
        s.history.append(
            {
                "role": "user",
                "content": (
                    "[authenticated user: israel.lot@yallaplay.com]\n"
                    "(This is the signed-in user you are talking to, from the "
                    "surface's SSO/identity provider. Use it to address and "
                    "identify them; do not ask who they are.)\n\n"
                    "help me fix the sessions sidebar"
                ),
            }
        )
        listing = manager.list_sessions()
        assert len(listing) == 1
        title = listing[0]["title"]
        assert title.startswith("help me fix the sessions sidebar")
        assert "authenticated user" not in title
        assert "israel.lot@yallaplay.com" not in title

    def test_list_sessions_marks_untitled_rows(self, manager):
        s = manager.create_session(cwd="/untitled")
        s.history.append({"role": "user", "content": "some prompt"})
        listing = manager.list_sessions()
        assert len(listing) == 1
        assert listing[0]["untitled"] is True

    def test_save_session_preserves_existing_messages_on_encode_failure(self, manager):
        """Regression for #13675: a bad message in state.history must not
        clobber the previously-persisted transcript.  replace_messages()
        wraps DELETE + INSERT in a single rolled-back-on-exception txn.
        """
        state = manager.create_session()
        state.history.append({"role": "user", "content": "original"})
        manager.save_session(state.session_id)

        # Now swap history with a message whose tool_calls is non-JSON-serializable.
        # _execute_write rolls back; the previously persisted "original" stays.
        state.history = [
            {"role": "user", "content": "replacement"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"bad": object()}],
            },
        ]
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"
        assert isinstance(messages[0].get("timestamp"), (int, float))

    def test_save_session_preserves_agent_archived_history(self, tmp_path):
        """Regression: ACP _persist must not destroy compression-archived rows.

        When the agent owns persistence to the same SessionDB, it has already
        flushed the transcript itself and used archive_and_compact() to keep
        pre-compaction turns as searchable active=0/compacted=1 rows. A blind
        replace_messages() here used to DELETE those archived rows (and the FTS
        index entries with them) on every save — silent data loss for any ACP
        conversation long enough to compress.
        """
        db = SessionDB(tmp_path / "state.db")

        def factory():
            # Mimic a live ACP agent: it persists to *this* db and has already
            # created its session row / flushed at least one turn.
            return SimpleNamespace(
                model="test-model",
                _session_db=db,
                _session_db_created=True,
            )

        manager = SessionManager(agent_factory=factory, db=db)
        state = manager.create_session(cwd="/work")

        # Simulate the agent's own persistence: it flushed the live transcript,
        # then compression archived the pre-compaction turns and inserted a
        # compacted summary as the new active set.
        db.append_message(
            session_id=state.session_id, role="user", content="archived needle"
        )
        db.archive_and_compact(
            state.session_id, [{"role": "user", "content": "compacted summary"}]
        )

        # ACP's in-memory history only tracks the post-compaction (active) set.
        state.history = [{"role": "user", "content": "compacted summary"}]
        manager.save_session(state.session_id)

        # The archived pre-compaction turn must survive and stay discoverable.
        contents = [
            m["content"]
            for m in db.get_messages(state.session_id, include_inactive=True)
        ]
        assert "archived needle" in contents
        assert "compacted summary" in contents
        hits = {r["session_id"] for r in db.search_messages("needle")}
        assert state.session_id in hits

    def test_save_session_still_replaces_when_agent_not_self_persisting(self, manager):
        """Agents that don't own DB persistence keep ACP as the source of truth.

        The default fixture's MagicMock agent has a ``_session_db`` that is *not*
        the manager's db, so the destructive replace path stays active and ACP
        history overwrites cleanly (no orphaned rows from a prior save).
        """
        state = manager.create_session()
        db = manager._get_db()

        state.history = [{"role": "user", "content": "v1"}]
        manager.save_session(state.session_id)
        assert [
            m["content"] for m in db.get_messages_as_conversation(state.session_id)
        ] == ["v1"]

        state.history = [{"role": "user", "content": "v2 replaced"}]
        manager.save_session(state.session_id)
        assert [
            m["content"] for m in db.get_messages_as_conversation(state.session_id)
        ] == ["v2 replaced"]

    def test_save_session_preserves_archived_rows_on_model_switch(self, tmp_path):
        """Regression (#50405 W1/W2): a save by a fresh, non-self-persisting
        agent must not destroy compaction-archived rows.

        Model switches and /restore mint a brand-new agent with
        ``_session_db_created=False`` (so it does NOT "own" persistence) and
        then immediately call save_session. If the session had already
        compacted, a blind full-history replace would DELETE the archived
        active=0/compacted=1 rows — the same data loss the owned-agent guard
        prevents. When archived rows exist, _persist must replace only the live
        set (active_only) and leave the archived transcript intact.
        """
        from types import SimpleNamespace

        db = SessionDB(tmp_path / "state.db")
        # Use a mock agent factory so create_session doesn't spin up a real
        # AIAgent (which needs credentials and leaks provider-probe state across
        # xdist workers). The factory's agent does NOT own persistence to db.
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(model="m"), db=db
        )
        state = manager.create_session(cwd="/work")

        # Session flushed a live turn, then compaction archived it.
        db.append_message(
            session_id=state.session_id, role="user", content="archived needle"
        )
        db.archive_and_compact(
            state.session_id, [{"role": "user", "content": "compacted summary"}]
        )

        # Model switch: a fresh agent bound to THIS db but not yet self-created.
        state.agent = SimpleNamespace(
            model="new-model", _session_db=db, _session_db_created=False
        )
        state.history = [{"role": "user", "content": "compacted summary"}]
        manager.save_session(state.session_id)

        # Archived pre-compaction turn survives and stays discoverable.
        contents = [
            m["content"]
            for m in db.get_messages(state.session_id, include_inactive=True)
        ]
        assert "archived needle" in contents
        assert "compacted summary" in contents
        hits = {r["session_id"] for r in db.search_messages("needle")}
        assert state.session_id in hits

    def test_cleanup_clears_all(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        s1.history.append({"role": "user", "content": "one"})
        s2.history.append({"role": "user", "content": "two"})
        assert len(manager.list_sessions()) == 2
        manager.cleanup()
        assert manager.list_sessions() == []

    def test_remove_session(self, manager):
        state = manager.create_session()
        assert manager.remove_session(state.session_id) is True
        assert manager.get_session(state.session_id) is None
        # Removing again returns False
        assert manager.remove_session(state.session_id) is False


# ---------------------------------------------------------------------------
# persistence — sessions survive process restarts (via SessionDB)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Verify that sessions are persisted to SessionDB and can be restored."""

    def test_create_session_includes_registered_mcp_toolsets(self, tmp_path, monkeypatch):
        captured = {}

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            return {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "***",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(model=kwargs.get("model"), enabled_toolsets=kwargs.get("enabled_toolsets"))

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"},
            "mcp_servers": {
                "olympus": {"command": "python", "enabled": True},
                "exa": {"url": "https://exa.ai/mcp"},
                "disabled": {"command": "python", "enabled": False},
            },
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            manager.create_session(cwd="/work")

        assert captured["enabled_toolsets"] == ["hermes-acp", "mcp-olympus", "mcp-exa"]

    def test_create_session_honors_platform_toolsets_acp(self, tmp_path, monkeypatch):
        """platform_toolsets.acp in config replaces the hermes-acp composite."""
        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(model=kwargs.get("model"), enabled_toolsets=kwargs.get("enabled_toolsets"))

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"},
            "platform_toolsets": {"acp": ["web", "terminal", "file"]},
            "mcp_servers": {"olympus": {"command": "python", "enabled": True}},
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **kwargs: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "***",
                "command": None,
                "args": [],
            },
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            manager.create_session(cwd="/work")

        assert captured["enabled_toolsets"] == ["web", "terminal", "file", "mcp-olympus"]

    def test_acp_base_toolsets_fallback_on_empty_or_missing(self, monkeypatch):
        """Missing, non-list, or empty platform_toolsets.acp falls back to hermes-acp."""
        for cfg in ({}, {"platform_toolsets": {}}, {"platform_toolsets": {"acp": "web"}},
                    {"platform_toolsets": {"acp": []}}, {"platform_toolsets": {"acp": ["", None]}}):
            monkeypatch.setattr("hermes_cli.config.load_config", lambda cfg=cfg: cfg)
            assert acp_session._acp_base_toolsets() == ["hermes-acp"]

    def test_create_session_writes_to_db(self, manager):
        state = manager.create_session(cwd="/project")
        db = manager._get_db()
        assert db is not None
        row = db.get_session(state.session_id)
        assert row is not None
        assert row["source"] == "acp"
        # cwd stored in model_config JSON
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/project"

    def test_get_session_restores_from_db(self, manager):
        """Simulate process restart: create session, drop from memory, get again."""
        state = manager.create_session(cwd="/work")
        state.history.append({"role": "user", "content": "hello"})
        state.history.append({"role": "assistant", "content": "hi there"})
        manager.save_session(state.session_id)

        sid = state.session_id

        # Drop from in-memory store (simulates process restart).
        with manager._lock:
            del manager._sessions[sid]

        # get_session should transparently restore from DB.
        restored = manager.get_session(sid)
        assert restored is not None
        assert restored.session_id == sid
        assert restored.cwd == "/work"
        assert len(restored.history) == 2
        assert restored.history[0]["content"] == "hello"
        assert restored.history[1]["content"] == "hi there"
        # Agent should have been recreated.
        assert restored.agent is not None

    def test_mode_survives_restore_from_db(self, manager):
        """A non-default session mode must survive a process restart.

        set_session_mode stores the mode on SessionState and persists it into
        the model_config JSON blob; _restore must read it back so a VS Code
        window reload (which respawns the ACP agent) doesn't revert the mode
        to the server default.
        """
        state = manager.create_session(cwd="/work")
        state.mode = "acceptEdits"
        manager.save_session(state.session_id)
        sid = state.session_id

        # Drop from in-memory store (simulates process restart).
        with manager._lock:
            del manager._sessions[sid]

        restored = manager.get_session(sid)
        assert restored is not None
        assert restored.mode == "acceptEdits"

    def test_default_mode_not_written_to_model_config(self, manager):
        """An unset/default mode leaves no `mode` key in the meta blob."""
        state = manager.create_session(cwd="/work")
        manager.save_session(state.session_id)
        row = manager._get_db().get_session(state.session_id)
        meta = json.loads(row["model_config"])
        assert "mode" not in meta

    def test_effort_survives_restore_from_db(self, manager):
        """A session's reasoning-effort override must survive a restart,
        and be re-applied to the freshly minted agent's reasoning_config."""
        state = manager.create_session(cwd="/work")
        state.effort = "high"
        manager.save_session(state.session_id)
        sid = state.session_id

        with manager._lock:
            del manager._sessions[sid]

        restored = manager.get_session(sid)
        assert restored is not None
        assert restored.effort == "high"
        assert restored.agent.reasoning_config == {"enabled": True, "effort": "high"}

    def test_default_effort_not_written_to_model_config(self, manager):
        """An unset effort leaves no `effort` key in the meta blob and does
        not touch the restored agent's reasoning_config."""
        state = manager.create_session(cwd="/work")
        manager.save_session(state.session_id)
        row = manager._get_db().get_session(state.session_id)
        meta = json.loads(row["model_config"])
        assert "effort" not in meta

    def test_fork_session_preserves_effort(self, manager):
        original = manager.create_session()
        original.effort = "xhigh"
        forked = manager.fork_session(original.session_id, cwd="/new")
        assert forked is not None
        assert forked.effort == "xhigh"
        assert forked.agent.reasoning_config == {"enabled": True, "effort": "xhigh"}

    def test_save_session_updates_db(self, manager):
        state = manager.create_session()
        state.history.append({"role": "user", "content": "test"})
        manager.save_session(state.session_id)

        db = manager._get_db()
        messages = db.get_messages_as_conversation(state.session_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "test"

    def test_remove_session_deletes_from_db(self, manager):
        state = manager.create_session()
        db = manager._get_db()
        assert db.get_session(state.session_id) is not None
        manager.remove_session(state.session_id)
        assert db.get_session(state.session_id) is None

    def test_cleanup_removes_all_from_db(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        db = manager._get_db()
        assert db.get_session(s1.session_id) is not None
        assert db.get_session(s2.session_id) is not None
        manager.cleanup()
        assert db.get_session(s1.session_id) is None
        assert db.get_session(s2.session_id) is None

    def test_list_sessions_includes_db_only(self, manager):
        """Sessions only in DB (not in memory) appear in list_sessions."""
        state = manager.create_session(cwd="/db-only")
        state.history.append({"role": "user", "content": "database only thread"})
        manager.save_session(state.session_id)
        sid = state.session_id

        # Drop from memory.
        with manager._lock:
            del manager._sessions[sid]

        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert sid in ids

    def test_list_sessions_surfaces_in_memory_session_mid_first_turn(self, manager):
        """A session that is IN MEMORY with an empty history but already has
        persisted messages must surface via its DB row.

        In-memory ``state.history`` is only assigned when a turn finishes,
        while the agent flushes messages to the DB incrementally during the
        turn. A spawned session (acp_spawn_session) spends its whole first
        turn in that state; before the fix the empty-history skip also left
        the id claimed in ``seen_ids``, so the DB merge skipped it too and
        the session was invisible in session/list until its first turn ended
        (2026-07-15 incident: spawned handoff session missing from the
        VS Code sidebar despite a healthy DB row).
        """
        state = manager.create_session(cwd="/spawned")
        sid = state.session_id
        assert state.history == []  # mid-turn: nothing assigned yet

        # Simulate the agent's incremental mid-turn flush: messages exist in
        # the DB even though state.history is still empty.
        db = manager._get_db()
        db.append_message(sid, role="user", content="continue the handoff")
        db.append_message(sid, role="assistant", content="on it")

        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert sid in ids

        # Truly-empty sessions (no history AND no persisted messages) must
        # stay hidden — the message_count guard still applies.
        empty = manager.create_session(cwd="/empty-still-hidden")
        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert empty.session_id not in ids

    def test_list_sessions_filters_by_cwd(self, manager):
        keep = manager.create_session(cwd="/keep")
        drop = manager.create_session(cwd="/drop")
        keep.history.append({"role": "user", "content": "keep me"})
        drop.history.append({"role": "user", "content": "drop me"})

        listing = manager.list_sessions(cwd="/keep")
        ids = {s["session_id"] for s in listing}
        assert keep.session_id in ids
        assert drop.session_id not in ids

    def test_list_sessions_includes_subagent_children_of_acp_parents(self, manager):
        """Delegate children (source='subagent', parent_session_id set) of a
        visible ACP session are listed with parent linkage; subagents of
        non-ACP parents (CLI/cron delegations) stay hidden."""
        parent = manager.create_session(cwd="/work")
        parent.history.append({"role": "user", "content": "delegate something"})
        manager.save_session(parent.session_id)

        db = manager._get_db()
        # Child of the visible ACP parent.
        db.create_session(
            "child-sub-1", source="subagent",
            parent_session_id=parent.session_id,
            model="test-model",
        )
        db.append_message("child-sub-1", role="user", content="child goal here")
        db.append_message("child-sub-1", role="assistant", content="working")
        # Orphan subagent: parent is not an ACP session.
        db.create_session("cli-parent", source="cli")
        db.append_message("cli-parent", role="user", content="cli work")
        db.create_session(
            "orphan-sub", source="subagent", parent_session_id="cli-parent",
        )
        db.append_message("orphan-sub", role="user", content="orphan goal")

        listing = manager.list_sessions()
        by_id = {s["session_id"]: s for s in listing}

        assert "child-sub-1" in by_id
        child = by_id["child-sub-1"]
        assert child["parent_id"] == parent.session_id
        assert child.get("subagent") is True
        assert "orphan-sub" not in by_id
        assert "cli-parent" not in by_id

    def test_list_sessions_subagent_children_respect_archived_only(self, manager):
        """archived_only listings don't surface live subagent children."""
        parent = manager.create_session(cwd="/work")
        parent.history.append({"role": "user", "content": "delegate"})
        manager.save_session(parent.session_id)

        db = manager._get_db()
        db.create_session(
            "child-sub-2", source="subagent",
            parent_session_id=parent.session_id,
        )
        db.append_message("child-sub-2", role="user", content="child goal")

        listing = manager.list_sessions(archived_only=True)
        ids = {s["session_id"] for s in listing}
        assert "child-sub-2" not in ids

    def test_list_sessions_hides_empty_subagent_children(self, manager):
        """A subagent row with no flushed messages yet stays hidden (same
        message_count guard as ordinary DB-only rows)."""
        parent = manager.create_session(cwd="/work")
        parent.history.append({"role": "user", "content": "delegate"})
        manager.save_session(parent.session_id)

        db = manager._get_db()
        db.create_session(
            "child-empty", source="subagent",
            parent_session_id=parent.session_id,
        )

        listing = manager.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert "child-empty" not in ids

    def test_get_session_restores_subagent_child_from_db(self, manager):
        """Loading a delegate child's session id restores its DB transcript.
        Children write incrementally, so a mid-run load must surface the
        flushed messages even though the child was never an ACP session."""
        parent = manager.create_session(cwd="/work")
        db = manager._get_db()
        db.create_session(
            "child-load-1", source="subagent",
            parent_session_id=parent.session_id, model="test-model",
            model_config={"_delegate_from": parent.session_id},
        )
        db.append_message("child-load-1", role="user", content="child goal")
        db.append_message("child-load-1", role="assistant", content="progress so far")

        state = manager.get_session("child-load-1")
        assert state is not None
        assert state.subagent is True
        assert [m["content"] for m in state.history] == [
            "child goal", "progress so far",
        ]

    def test_persist_is_noop_for_subagent_sessions(self, manager):
        """ACP must never take write ownership of a delegate child's row:
        persisting would clobber model_config (losing the _delegate_from
        marker that keeps subagent rows out of general session lists) and
        rewrite the transcript the child owns."""
        parent = manager.create_session(cwd="/work")
        db = manager._get_db()
        db.create_session(
            "child-persist-1", source="subagent",
            parent_session_id=parent.session_id,
            model_config={"_delegate_from": parent.session_id},
        )
        db.append_message("child-persist-1", role="user", content="child goal")

        state = manager.get_session("child-persist-1")
        assert state is not None
        manager.save_session("child-persist-1")

        row = db.get_session("child-persist-1")
        meta = json.loads(row["model_config"])
        assert meta.get("_delegate_from") == parent.session_id
        assert row["source"] == "subagent"
        # Transcript untouched.
        messages = db.get_messages_as_conversation("child-persist-1")
        assert [m["content"] for m in messages] == ["child goal"]

    def test_list_sessions_matches_windows_and_wsl_paths(self, manager):
        state = manager.create_session(cwd="/mnt/e/Projects/AI/browser-link-3")
        state.history.append({"role": "user", "content": "same project from WSL"})

        listing = manager.list_sessions(cwd=r"E:\Projects\AI\browser-link-3")
        ids = {s["session_id"] for s in listing}
        assert state.session_id in ids

    def test_list_sessions_prefers_title_then_preview(self, manager):
        state = manager.create_session(cwd="/named")
        state.history.append({"role": "user", "content": "Investigate broken ACP history in Zed"})
        manager.save_session(state.session_id)
        db = manager._get_db()
        db.set_session_title(state.session_id, "Fix Zed ACP history")

        listing = manager.list_sessions(cwd="/named")
        assert listing[0]["title"] == "Fix Zed ACP history"

        db.set_session_title(state.session_id, "")
        listing = manager.list_sessions(cwd="/named")
        assert listing[0]["title"].startswith("Investigate broken ACP history")

    def test_list_sessions_flags_slack_sessions_in_memory(self, manager):
        """A session whose first user message carries the Slack runtime envelope
        is flagged slack=True so clients can badge it (Slack icon prefix)."""
        slack = manager.create_session(cwd="/slack")
        slack.history.append(
            {"role": "user", "content": "[Slack runtime context]\nYou're talking to U123"}
        )
        plain = manager.create_session(cwd="/plain")
        plain.history.append({"role": "user", "content": "ordinary prompt"})

        by_id = {s["session_id"]: s for s in manager.list_sessions()}
        assert by_id[slack.session_id]["slack"] is True
        assert by_id[plain.session_id]["slack"] is False

    def test_list_sessions_flags_slack_sessions_db_only(self, manager):
        state = manager.create_session(cwd="/slack-db")
        state.history.append(
            {"role": "user", "content": "[Slack runtime context]\nrequester: someone"}
        )
        manager.save_session(state.session_id)
        sid = state.session_id
        with manager._lock:
            del manager._sessions[sid]

        by_id = {s["session_id"]: s for s in manager.list_sessions()}
        assert by_id[sid]["slack"] is True

    def test_list_sessions_fork_of_slack_session_not_flagged_in_memory(self, manager):
        """Forking a Slack session yields a first-class ACP session — the fork
        copies the Slack envelope into its history but must NOT carry the
        Slack badge (it is no longer a Slack session)."""
        original = manager.create_session(cwd="/slack")
        original.history.append(
            {"role": "user", "content": "[Slack runtime context]\nYou're talking to U123"}
        )
        forked = manager.fork_session(original.session_id, cwd="/slack")

        by_id = {s["session_id"]: s for s in manager.list_sessions()}
        assert by_id[original.session_id]["slack"] is True
        assert by_id[forked.session_id]["slack"] is False
        assert by_id[forked.session_id]["parent_id"] == original.session_id

    def test_list_sessions_fork_of_slack_session_not_flagged_db_only(self, manager):
        original = manager.create_session(cwd="/slack")
        original.history.append(
            {"role": "user", "content": "[Slack runtime context]\nrequester: someone"}
        )
        forked = manager.fork_session(original.session_id, cwd="/slack")
        manager.save_session(original.session_id)
        manager.save_session(forked.session_id)
        fork_id = forked.session_id
        with manager._lock:
            del manager._sessions[original.session_id]
            del manager._sessions[fork_id]

        by_id = {s["session_id"]: s for s in manager.list_sessions()}
        assert by_id[original.session_id]["slack"] is True
        assert by_id[fork_id]["slack"] is False

    def test_list_sessions_sorted_by_most_recent_activity(self, manager):
        older = manager.create_session(cwd="/ordered")
        older.history.append({"role": "user", "content": "older"})
        manager.save_session(older.session_id)
        time.sleep(0.02)
        newer = manager.create_session(cwd="/ordered")
        newer.history.append({"role": "user", "content": "newer"})
        manager.save_session(newer.session_id)

        listing = manager.list_sessions(cwd="/ordered")
        assert [item["session_id"] for item in listing[:2]] == [newer.session_id, older.session_id]
        assert listing[0]["updated_at"]
        assert listing[1]["updated_at"]

    def test_fork_restores_source_from_db(self, manager):
        """Forking a session that is only in DB should work."""
        original = manager.create_session()
        original.history.append({"role": "user", "content": "context"})
        manager.save_session(original.session_id)

        # Drop original from memory.
        with manager._lock:
            del manager._sessions[original.session_id]

        forked = manager.fork_session(original.session_id, cwd="/fork")
        assert forked is not None
        assert len(forked.history) == 1
        assert forked.history[0]["content"] == "context"
        assert forked.session_id != original.session_id

    def test_update_cwd_restores_from_db(self, manager):
        state = manager.create_session(cwd="/old")
        sid = state.session_id

        with manager._lock:
            del manager._sessions[sid]

        updated = manager.update_cwd(sid, "/new")
        assert updated is not None
        assert updated.cwd == "/new"

        # Should also be persisted in DB.
        db = manager._get_db()
        row = db.get_session(sid)
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/new"

    def test_only_restores_acp_sessions(self, manager):
        """get_session should not restore non-ACP sessions from DB."""
        db = manager._get_db()
        # Manually create a CLI session in the DB.
        db.create_session(session_id="cli-session-123", source="cli", model="test")
        # Should not be found via ACP SessionManager.
        assert manager.get_session("cli-session-123") is None

    def test_sessions_searchable_via_fts(self, manager):
        """ACP sessions stored in SessionDB are searchable via FTS5."""
        state = manager.create_session()
        state.history.append({"role": "user", "content": "how do I configure nginx"})
        state.history.append({"role": "assistant", "content": "Here is the nginx config..."})
        manager.save_session(state.session_id)

        db = manager._get_db()
        results = db.search_messages("nginx")
        assert len(results) > 0
        session_ids = {r["session_id"] for r in results}
        assert state.session_id in session_ids

    def test_tool_calls_persisted(self, manager):
        """Messages with tool_calls should round-trip through the DB."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"}}],
        })
        state.history.append({
            "role": "tool",
            "content": "output here",
            "tool_call_id": "tc_1",
            "name": "terminal",
        })
        manager.save_session(state.session_id)

        # Drop from memory, restore from DB.
        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        assert len(restored.history) == 2
        assert restored.history[0].get("tool_calls") is not None
        assert restored.history[1].get("tool_call_id") == "tc_1"

    def test_assistant_reasoning_fields_persisted(self, manager):
        """ACP session restore should preserve assistant reasoning context."""
        state = manager.create_session()
        state.history.append({
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        })
        manager.save_session(state.session_id)

        with manager._lock:
            del manager._sessions[state.session_id]

        restored = manager.get_session(state.session_id)
        assert restored is not None
        msg = restored.history[0]
        assert isinstance(msg.pop("timestamp", None), (int, float))
        assert restored.history == [{
            "role": "assistant",
            "content": "hello",
            "reasoning": "step-by-step",
            "reasoning_details": [
                {"type": "thinking", "thinking": "first thought"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "id": "rs_123", "encrypted_content": "enc_blob"},
            ],
        }]

    def test_restore_preserves_persisted_provider_snapshot(self, tmp_path, monkeypatch):
        """Restored ACP sessions should keep their original runtime provider."""
        runtime_choice = {"provider": "anthropic"}

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            provider = requested or runtime_choice["provider"]
            return {
                "provider": provider,
                "api_mode": "anthropic_messages" if provider == "anthropic" else "chat_completions",
                "base_url": f"https://{provider}.example/v1",
                "api_key": f"{provider}-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(
                model=kwargs.get("model"),
                provider=kwargs.get("provider"),
                base_url=kwargs.get("base_url"),
                api_mode=kwargs.get("api_mode"),
            )

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": runtime_choice["provider"], "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")
            manager.save_session(state.session_id)

            with manager._lock:
                del manager._sessions[state.session_id]

            runtime_choice["provider"] = "openrouter"
            restored = manager.get_session(state.session_id)

        assert restored is not None
        assert restored.agent.provider == "anthropic"
        assert restored.agent.base_url == "https://anthropic.example/v1"

    def test_acp_agents_route_human_output_to_stderr(self, tmp_path, monkeypatch):
        """ACP agents must keep stdout clean for JSON-RPC stdio transport."""

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            return {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.example/v1",
                "api_key": "test-key",
                "command": None,
                "args": [],
            }

        def fake_agent(**kwargs):
            return SimpleNamespace(model=kwargs.get("model"), _print_fn=None)

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "model": {"provider": "openrouter", "default": "test-model"}
        })
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )
        db = SessionDB(tmp_path / "state.db")

        with patch("run_agent.AIAgent", side_effect=fake_agent):
            manager = SessionManager(db=db)
            state = manager.create_session(cwd="/work")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            state.agent._print_fn("ACP noise")

        assert stdout_buf.getvalue() == ""
        assert stderr_buf.getvalue() == "ACP noise\n"


# ---------------------------------------------------------------------------
# archived support
# ---------------------------------------------------------------------------


def _mgr_with_db(rows, archived_ok=True):
    mgr = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    db = MagicMock(name="db")
    db.list_sessions_rich.return_value = rows
    db.set_session_archived.return_value = archived_ok
    mgr._get_db = lambda: db  # type: ignore[method-assign]
    return mgr, db


def test_list_sessions_fetches_all_and_filters_archived_in_python():
    rows = [
        {
            "id": "s1", "cwd": ".", "model": "m", "message_count": 3,
            "title": "T", "last_active": 1.0, "started_at": 0.0, "archived": 1,
        },
        {
            "id": "s2", "cwd": ".", "model": "m", "message_count": 3,
            "title": "U", "last_active": 1.0, "started_at": 0.0, "archived": 0,
        },
    ]
    mgr, db = _mgr_with_db(rows)
    out = mgr.list_sessions(archived_only=True)
    # The DB fetch is unfiltered (include_archived=True, no archived_only):
    # archived state must be visible for in-memory sessions too, so the
    # archived/active filtering happens per-row in Python.
    _, kwargs = db.list_sessions_rich.call_args_list[0]
    assert kwargs.get("include_archived") is True
    assert "archived_only" not in kwargs
    # only the archived row is surfaced, marked as such
    assert [s["session_id"] for s in out] == ["s1"]
    assert out[0]["archived"] is True


def test_list_sessions_orders_by_last_user_message_not_last_activity():
    # "b" has the most recent USER message even though "a" has fresher overall
    # activity (assistant/tool churn). Ordering must follow the user message.
    rows = [
        {
            "id": "a", "cwd": ".", "model": "m", "message_count": 5,
            "title": "agent-busy", "started_at": 0.0,
            "last_active": 200.0, "last_user_active": 50.0,
        },
        {
            "id": "b", "cwd": ".", "model": "m", "message_count": 5,
            "title": "user-recent", "started_at": 0.0,
            "last_active": 120.0, "last_user_active": 100.0,
        },
    ]
    mgr, _ = _mgr_with_db(rows)
    out = mgr.list_sessions()
    assert [s["session_id"] for s in out] == ["b", "a"]


def test_list_sessions_updated_at_falls_back_without_user_messages():
    rows = [{
        "id": "s1", "cwd": ".", "model": "m", "message_count": 1,
        "title": "T", "started_at": 10.0,
        "last_active": 42.0, "last_user_active": None,
    }]
    mgr, _ = _mgr_with_db(rows)
    out = mgr.list_sessions()
    from datetime import datetime, timezone
    expected = datetime.fromtimestamp(42.0, tz=timezone.utc).isoformat()
    assert out[0]["updated_at"] == expected


def test_set_session_archived_delegates_to_db():
    mgr, db = _mgr_with_db([], archived_ok=True)
    assert mgr.set_session_archived("s1", True) is True
    db.set_session_archived.assert_called_once_with("s1", True)


class TestLiveTranscriptHistory:
    def test_live_transcript_history_returns_db_conversation(self, manager):
        db = manager._get_db()
        sid = "acp-live-transcript-1"
        db.create_session(session_id=sid, source="acp")
        db.append_message(session_id=sid, role="user", content="hello")
        db.append_message(
            session_id=sid,
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        )
        db.append_message(
            session_id=sid,
            role="tool",
            content="ok",
            tool_call_id="call_1",
            tool_name="terminal",
        )

        messages = manager.live_transcript_history(sid)

        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[0]["content"] == "hello"
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2]["tool_call_id"] == "call_1"
        assert messages[2]["content"] == "ok"

    def test_live_transcript_history_resolves_agent_head(self, manager):
        state = manager.create_session()
        agent_id = "agent-head-rotated"
        state.agent.session_id = agent_id
        db = manager._get_db()
        db.create_session(session_id=agent_id, source="cli")
        db.append_message(session_id=agent_id, role="user", content="mid-turn flush")

        messages = manager.live_transcript_history(state.session_id)

        assert [m["content"] for m in messages] == ["mid-turn flush"]

    def test_live_transcript_history_returns_none_on_db_failure(self, manager, monkeypatch):
        db = manager._get_db()

        def boom(*args, **kwargs):
            raise RuntimeError("db exploded")

        monkeypatch.setattr(db, "get_messages_as_conversation", boom)

        assert manager.live_transcript_history("whatever") is None

    def test_live_transcript_history_returns_none_without_db(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_get_db", lambda: None)

        assert manager.live_transcript_history("whatever") is None

    def test_live_transcript_history_swallows_raising_agent_session_id(self, manager):
        """A raising agent.session_id property must not escape (contract: no exception escapes)."""

        class ExplodingAgent:
            @property
            def session_id(self):
                raise RuntimeError("proxy blew up resolving session_id")

        state = manager.create_session()
        state.agent = ExplodingAgent()

        assert manager.live_transcript_history(state.session_id) is None

    def test_live_transcript_history_ignores_non_string_agent_session_id(self, manager):
        """A truthy non-string agent.session_id must not be stringified into a bogus head."""
        state = manager.create_session()
        state.agent.session_id = object()  # truthy, not a str
        sid = state.session_id
        db = manager._get_db()
        db.create_session(session_id=sid, source="acp")
        db.append_message(session_id=sid, role="user", content="under acp id")

        messages = manager.live_transcript_history(sid)

        assert [m["content"] for m in messages] == ["under acp id"]

    def test_live_transcript_history_orders_by_insertion_not_timestamp(self, manager):
        """Rows with descending explicit timestamps must still come back in insertion order."""
        import time as _time

        db = manager._get_db()
        sid = "acp-live-transcript-order"
        db.create_session(session_id=sid, source="acp")
        now = _time.time()
        db.append_message(
            session_id=sid, role="user", content="first", timestamp=now + 100
        )
        db.append_message(
            session_id=sid,
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_ord",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
            timestamp=now,
        )
        db.append_message(
            session_id=sid,
            role="tool",
            content="done",
            tool_call_id="call_ord",
            tool_name="terminal",
            timestamp=now - 100,
        )

        messages = manager.live_transcript_history(sid)

        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[0]["content"] == "first"
        assert messages[1]["tool_calls"][0]["id"] == "call_ord"
        assert messages[2]["content"] == "done"
