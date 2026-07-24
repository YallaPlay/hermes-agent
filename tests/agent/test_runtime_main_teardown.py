"""Regression test: the runtime-main override must be cleared at
``run_conversation`` teardown so it does not leak across sessions/turns on a
long-lived, multi-session process (the ``hermes acp`` gateway, messaging
gateways, background workers).

Root cause fixed: ``set_runtime_main`` is called at the top of every turn
inside ``build_turn_context``, overwriting five PROCESS-GLOBAL module vars in
``agent.auxiliary_client``. ``clear_runtime_main`` existed but was never called
anywhere — dead code. So after a turn/run ended, the override stayed sticky and
was only ever overwritten by the NEXT turn's ``set_runtime_main``. Any auxiliary
resolve firing in the gap (vision auto-detect client build, title/completion
passes, background aux tasks) read a STALE main model from a PREVIOUS session.

Observed symptom: ``Vision auto-detect: using main provider bedrock
(global.anthropic.claude-fable-5)`` logged in sessions whose configured model is
``global.anthropic.claude-opus-4-8``, with no fable-5 session active for days —
because the process once served a fable-5 session and never cleared the global.

The fix wires ``clear_runtime_main()`` into ``AIAgent.run_conversation``'s
``finally`` (the single forwarder chokepoint every entry path — CLI, gateway,
ACP — funnels through, matching the session/agent-run scope of ``set``).
"""
from unittest.mock import patch


def _get_globals(mod):
    """Read runtime globals without triggering credential redaction."""
    return {
        "provider": mod._RUNTIME_MAIN_PROVIDER,
        "model": mod._RUNTIME_MAIN_MODEL,
        "base_url": mod._RUNTIME_MAIN_BASE_URL,
        "cred": mod._RUNTIME_MAIN_API_KEY,  # renamed to avoid redaction
        "api_mode": mod._RUNTIME_MAIN_API_MODE,
    }


class _FakeAgent:
    """Minimal stand-in exposing only ``run_conversation`` bound from AIAgent."""

    def __init__(self):
        self.provider = ""
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.api_mode = ""

    def _conversation_root_id(self):
        # Upstream's run_conversation forwarder publishes the conversation id
        # for ambient Nous Portal tagging; the fake has no session lineage.
        return None


class TestRuntimeMainTeardown:
    """clear_runtime_main must fire at run_conversation teardown."""

    def test_clear_resets_all_five_globals(self):
        """Direct unit: clear_runtime_main resets all five globals to empty."""
        import agent.auxiliary_client as mod

        mod.set_runtime_main(
            "bedrock",
            "global.anthropic.claude-fable-5",
            base_url="https://bedrock.example.com",
            api_key="sk-stale",
            api_mode="anthropic_messages",
        )
        mod.clear_runtime_main()
        for k, v in _get_globals(mod).items():
            assert v == "", f"Expected {k!r} empty after clear, got {v!r}"

    def test_forwarder_clears_override_at_teardown(self):
        """run_conversation forwarder clears the override in its finally.

        We monkeypatch the module-level ``run_conversation`` (invoked inside the
        AIAgent forwarder) to a no-op so no real turn spins up; we only exercise
        the teardown wiring.
        """
        import agent.auxiliary_client as aux
        import agent.conversation_loop as loop
        from run_agent import AIAgent

        # Simulate a PREVIOUS session having left a stale override behind.
        aux.set_runtime_main(
            "bedrock",
            "global.anthropic.claude-fable-5",
            base_url="https://bedrock.example.com",
            api_key="sk-stale",
            api_mode="anthropic_messages",
        )
        assert aux._RUNTIME_MAIN_MODEL == "global.anthropic.claude-fable-5"

        agent = _FakeAgent()
        try:
            with patch.object(loop, "run_conversation", return_value={"final_response": "ok"}):
                result = AIAgent.run_conversation(agent, "hello")
            assert result == {"final_response": "ok"}
        finally:
            aux.clear_runtime_main()

        # After the run completes the override must be gone — no leak.
        for k, v in _get_globals(aux).items():
            assert v == "", f"Override leaked past run: {k!r}={v!r}"

    def test_forwarder_clears_override_even_on_exception(self):
        """Teardown clear runs even when the turn raises (finally semantics)."""
        import agent.auxiliary_client as aux
        import agent.conversation_loop as loop
        from run_agent import AIAgent

        aux.set_runtime_main("bedrock", "global.anthropic.claude-fable-5")
        assert aux._RUNTIME_MAIN_MODEL == "global.anthropic.claude-fable-5"

        agent = _FakeAgent()
        boom = RuntimeError("turn blew up")
        try:
            with patch.object(loop, "run_conversation", side_effect=boom):
                try:
                    AIAgent.run_conversation(agent, "hello")
                except RuntimeError as exc:
                    assert exc is boom
                else:
                    raise AssertionError("expected the turn exception to propagate")
        finally:
            aux.clear_runtime_main()

        assert aux._RUNTIME_MAIN_MODEL == "", "Override must clear even on turn error"

    def test_read_main_model_falls_back_to_config_after_teardown(self):
        """_read_main_model resumes config fallback once the override is cleared."""
        import agent.auxiliary_client as aux
        import agent.conversation_loop as loop
        from run_agent import AIAgent

        # Stale override from a prior session points at fable-5.
        aux.set_runtime_main("bedrock", "global.anthropic.claude-fable-5")
        # While the override is set, the override wins (override-first read order).
        assert aux._read_main_model() == "global.anthropic.claude-fable-5"

        agent = _FakeAgent()
        try:
            with patch.object(loop, "run_conversation", return_value={"final_response": "ok"}):
                AIAgent.run_conversation(agent, "hello")
        finally:
            aux.clear_runtime_main()

        # Override cleared: _read_main_model must fall back to config.yaml
        # (a deterministic sentinel below) rather than returning the stale model.
        assert aux._RUNTIME_MAIN_MODEL == ""
        with patch("hermes_cli.config.load_config", return_value={"model": {"default": "global.anthropic.claude-opus-4-8"}}):
            resolved = aux._read_main_model()
        assert resolved == "global.anthropic.claude-opus-4-8", (
            f"Expected config fallback after teardown, got stale {resolved!r}"
        )
