"""Tests for agent.title_generator — auto-generated session titles."""

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    generate_title,
    auto_title_session,
    maybe_auto_title,
    _title_language,
    _looks_like_conversational_reply,
)
from hermes_state import SessionDB


class TestGenerateTitle:
    """Unit tests for generate_title()."""




    def test_title_language_reads_config(self):
        cfg = {"auxiliary": {"title_generation": {"language": "  French "}}}

        with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            assert _title_language() == "French"
        with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert _title_language() == ""
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad config")), \
         patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("bad config")):
            assert _title_language() == ""

    def test_default_timeout_delegates_to_auxiliary_config(self):
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Configured Timeout"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            assert generate_title("question", "answer") == "Configured Timeout"

        assert captured_kwargs["task"] == "title_generation"
        assert captured_kwargs["timeout"] is None



    def test_strips_think_blocks(self):
        """Reasoning-model output wrapped in <think>...</think> must not
        leak into the session title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>The user wants a title. I'll summarize the topic "
            "concisely.</think>Debugging Python Import Errors"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("help me fix this import", "Sure...")
            assert title == "Debugging Python Import Errors"
            assert "<think>" not in title
            assert "summarize" not in title

    def test_strips_unterminated_think_block(self):
        """An unterminated <think> block (no close tag) must still be
        stripped so the leaked reasoning doesn't become the title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>Let me reason about a good title for this session"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("hello", "hi there")
            # Everything from the unterminated open tag onward is stripped,
            # leaving nothing → None.
            assert title is None


    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("question", "answer")
            assert len(title) == 80
            assert title.endswith("...")



    def test_invokes_failure_callback_on_exception(self):
        """failure_callback must fire so the user sees a warning (issue #15775)."""
        captured = []

        def _cb(task, exc):
            captured.append((task, exc))

        exc = RuntimeError("openrouter 402: credits exhausted")
        with patch("agent.title_generator.call_llm", side_effect=exc):
            result = generate_title("question", "answer", failure_callback=_cb)

        assert result is None
        assert len(captured) == 1
        assert captured[0][0] == "title generation"
        assert captured[0][1] is exc










    def test_conversational_reply_is_not_stored_as_title(self):
        """End-to-end: the real 2026-07-29 incident reply must yield None so the
        session stays untitled instead of being named after a refusal."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "I'm ready to help! However, I don't see an image attached yet. "
            "Could you please attach the image?"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("look at this screenshot", "...") is None

    def test_conversational_reply_is_rejected_before_truncation(self):
        """The rejection must happen before the 80-char ellipsis truncation —
        otherwise the junk title is merely shortened, not dropped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "Sure, I can help with that. " + "x" * 200
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("q", "a") is None

    def test_legitimate_title_still_returned(self):
        """The guard must not disturb an ordinary title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Slack bot image attachment fix"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("q", "a") == "Slack bot image attachment fix"


# The exact title stored during the 2026-07-29 incident (session
# 8987e113-810c-4557-8d9f-7b1aebf3ea05) before truncation.
_INCIDENT_REPLY = (
    "I'm ready to help! However, I don't see an image attached yet. "
    "Could you please attach the image?"
)


class TestLooksLikeConversationalReply:
    """Unit tests for the _looks_like_conversational_reply() predicate."""

    def test_rejects_the_real_incident_string(self):
        assert _looks_like_conversational_reply(_INCIDENT_REPLY) is True

    def test_rejects_the_incident_string_as_truncated_and_stored(self):
        """The stored form was truncated to 80 chars + ellipsis; the predicate
        runs before truncation, but the truncated text must also be caught in
        case the order ever changes."""
        stored = _INCIDENT_REPLY[:77] + "..."
        assert _looks_like_conversational_reply(stored) is True

    @pytest.mark.parametrize(
        "candidate",
        [
            # first-person / assistant-voice openers
            "I'm ready to help with that",
            "I’m ready to help with that",  # curly apostrophe
            "I don't see an image attached",
            "I do not have access to that file",
            "I can help you with the deploy",
            "I'll take a look at the logs",
            "I need more information about the schema",
            "I'd be happy to help",
            "I apologize for the confusion",
            # willingness / acknowledgement
            "Sure, here is the config",
            "Certainly! Let's look at the data",
            "Of course, I can do that",
            "Happy to help with the migration",
            "Let me check the warehouse for you",
            "No problem, that is easy",
            "Thank you for the clarification",
            # model voice
            "As an AI, I cannot browse the web",
            "As a language model I lack that ability",
            # hedges / presentation
            "It looks like the config is missing",
            "It seems the job never ran",
            "Here's a summary of the changes",
            "Here is the breakdown you asked for",
            "Here are the top three offenders",
            # requests back to the user
            "Could you clarify which environment",
            "Can you share the stack trace",
            "Please provide the session id",
            "Please attach the screenshot",
            # apology / refusal in conversational form
            "Unfortunately I cannot read that file",
            "Sorry, I can't access the database",
            "My apologies for the delay",
        ],
    )
    def test_rejects_conversational_openers(self, candidate):
        assert _looks_like_conversational_reply(candidate) is True

    @pytest.mark.parametrize(
        "candidate",
        [
            "What did you want to analyze?",
            "Which warehouse should I query?",
            "Snowflake cost analysis for July?",
        ],
    )
    def test_rejects_trailing_question_mark(self, candidate):
        assert _looks_like_conversational_reply(candidate) is True

    def test_rejects_refusal_marker_plus_assistant_voice(self):
        """A reply that opens with a noun phrase but still speaks in the
        assistant's voice about a refusal."""
        assert _looks_like_conversational_reply(
            "Unable to proceed because I don't see the attachment"
        ) is True

    def test_rejects_long_sentence_addressing_the_user(self):
        assert _looks_like_conversational_reply(
            "The file you uploaded appears to be empty so nothing was analyzed."
        ) is True

    @pytest.mark.parametrize(
        "candidate",
        [
            # the three mandated survival examples
            "Cannot connect to Snowflake warehouse",
            "Fix I/O timeout in ranking service",
            "Sorry state of the deals config",
            # ordinary titles
            "Slack bot image attachment fix",
            "Snowflake cost analysis for July",
            "Debugging Python Import Errors",
            "Kubernetes Pod Debugging",
            "Setting Up Docker Environment",
            "Session title generator refusal guard",
            "macOS Disk Cleanup",
            "Weather Forecast Lookup",
            "Missions analytics rewrite",
            # keyword-bearing but legitimate
            "Unable-to-pay webhook retry logic",
            "Sorry-state audit of the offer ladder",
            "Can't-repro flake in matchmaking tests",
            "ID mapping for player accounts",
            "Ill-formed JSON in config loader",
            "Season Pass reward collection review",
            "AI agent tool-choice quality metrics",
            "Here-document parsing in shell tool",
            # non-English titles (must survive — prompt allows user's language)
            "Snowflakeのコスト分析",
            "Analyse des coûts Snowflake",
            "تحليل تكاليف Snowflake",
            # long but phrase-like, no terminal punctuation
            "Investigating the intermittent websocket disconnects in the realtime channel",
        ],
    )
    def test_accepts_legitimate_titles(self, candidate):
        assert _looks_like_conversational_reply(candidate) is False

    def test_empty_and_whitespace_are_not_conversational(self):
        """Emptiness is handled by the caller's falsy check, not here."""
        assert _looks_like_conversational_reply("") is False
        assert _looks_like_conversational_reply("   ") is False


class TestAutoTitleSession:
    """Tests for auto_title_session() — the sync worker function."""




    def test_does_not_overwrite_title_set_immediately_before_conditional_write(
        self, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        seen = []

        def generate_after_manual_title(*_args, **_kwargs):
            db.set_session_title("sess-1", "Manual Title")
            return "Auto Title"

        with patch(
            "agent.title_generator.generate_title",
            side_effect=generate_after_manual_title,
        ):
            auto_title_session(
                db,
                "sess-1",
                "hi",
                "hello",
                title_callback=seen.append,
            )

        assert db.get_session_title("sess-1") == "Manual Title"
        assert seen == []

    def test_invokes_title_callback_after_setting_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        db.set_auto_title_if_empty.return_value = True
        seen = []
        with patch("agent.title_generator.generate_title", return_value="Readable Session"):
            auto_title_session(
                db,
                "sess-1",
                "hello",
                "hi there",
                title_callback=seen.append,
            )
        db.set_auto_title_if_empty.assert_called_once_with("sess-1", "Readable Session")
        assert seen == ["Readable Session"]



    def test_body_exception_routed_to_failure_callback(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        seen = []

        boom = ImportError("stale module")
        with patch("agent.title_generator._auto_title_session", side_effect=boom):
            auto_title_session(
                db,
                "sess-1",
                "hi",
                "hello",
                failure_callback=lambda task, exc: seen.append((task, exc)),
            )
        assert seen == [("title generation", boom)]



class TestMaybeAutoTitle:
    """Tests for maybe_auto_title() — the fire-and-forget entry point."""

    def test_skips_if_not_first_exchange(self):
        """Should not fire for conversations with more than 2 user messages."""
        db = MagicMock()
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "response 3"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "third", "response 3", history)
            # Wait briefly for any thread to start
            import time
            time.sleep(0.1)
            mock_auto.assert_not_called()

    def test_fires_on_first_exchange(self):
        """Should fire a background thread for the first exchange."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", "hi there", history)
            # Event-based wait: sleep-sync flaked when the daemon thread
            # wasn't scheduled within the fixed nap on a loaded runner.
            assert called.wait(timeout=10), "auto_title thread never ran"
            mock_auto.assert_called_once_with(
                db,
                "sess-1",
                "hello",
                "hi there",
                failure_callback=None,
                main_runtime=None,
                title_callback=None,
                runtime_validator=None,
            )






class TestAutoTitleDuplicateHandling:
    """Duplicate auto-title handling and not-found hardening (#50537)."""

    def test_dedupes_duplicate_title_via_lineage(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        # Atomic write path: collision raises ValueError, retry persists.
        db.set_auto_title_if_empty.side_effect = [ValueError("in use"), True]
        db.get_next_title_in_lineage.return_value = "Debugging Import Error #2"
        with patch(
            "agent.title_generator.generate_title",
            return_value="Debugging Import Error",
        ):
            seen = []
            auto_title_session(db, "sess-1", "hi", "hello", title_callback=seen.append)
        db.get_next_title_in_lineage.assert_called_once_with("Debugging Import Error")
        assert db.set_auto_title_if_empty.call_args_list[-1][0] == (
            "sess-1",
            "Debugging Import Error #2",
        )
        # callback fires with the actually-persisted (deduped) title
        assert seen == ["Debugging Import Error #2"]



    def test_manual_title_race_skips_without_callback(self):
        # Atomic predicate fails (manual /title landed while generation was in
        # flight) -> nothing persisted, no callback fired.
        from agent.title_generator import _persist_session_title
        db = MagicMock()
        db.set_auto_title_if_empty.return_value = False
        assert _persist_session_title(db, "sess-1", "Some Title") is None
        db.set_session_title.assert_not_called()



class TestRuntimeValidator:
    """runtime_validator gating (#19027): a stale background title request
    must not fire when the session's model/provider changed after spawn."""



    def test_broken_validator_fails_open(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Resilient Title"

        def _bad_validator():
            raise RuntimeError("validator gone")

        with patch("agent.title_generator.call_llm", return_value=mock_response) as mock_llm:
            title = generate_title(
                "question", "answer",
                runtime_validator=_bad_validator,
            )
            assert title == "Resilient Title"
            mock_llm.assert_called_once()

    def test_forwards_runtime_validator_to_worker(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        def _v():
            return True

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", "hi there", history, runtime_validator=_v)
            assert called.wait(timeout=10), "auto_title thread never ran"
            kwargs = mock_auto.call_args.kwargs
            assert kwargs["runtime_validator"] is _v
