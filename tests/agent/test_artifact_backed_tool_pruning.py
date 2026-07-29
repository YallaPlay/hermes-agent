"""Artifact-backed behavior for proactive tool-result pruning only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import ARTIFACT_RESULT_TAG, ContextCompressor
from agent.conversation_loop import _run_proactive_tool_result_prune
from tools.tool_result_storage import (
    _ARTIFACT_SCOPE_HASH_CHARS,
    PersistedToolArtifact,
    redact_sensitive_text,
)


class LocalArtifactEnv:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.execute_count = 0

    def get_temp_dir(self) -> str:
        return str(self.temp_dir)

    def execute(self, command, *, timeout=30, stdin_data=None):
        self.execute_count += 1
        proc = subprocess.run(
            ["bash", "-c", command],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {"output": proc.stdout + proc.stderr, "returncode": proc.returncode}


def _compressor(**overrides) -> ContextCompressor:
    kwargs = {
        "model": "test",
        "quiet_mode": True,
        "protect_first_n": 0,
        "protect_last_n": 2,
        "proactive_prune_tokens": 1,
        "proactive_prune_min_result_chars": 200,
        "proactive_prune_min_reclaim_tokens": 0,
        "proactive_prune_artifacts": True,
    }
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return ContextCompressor(**kwargs)


def _messages(*, output: str, arguments: str = '{"cmd":"keep exactly"}', call_id: str = "call-1"):
    return [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "terminal", "arguments": arguments},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "done"},
    ]


def _artifact_metadata(stub: str) -> dict:
    assert stub.startswith(f"{ARTIFACT_RESULT_TAG}\n")
    payload = stub.removeprefix(f"{ARTIFACT_RESULT_TAG}\n").split("\n</tool-result-artifact>", 1)[0]
    return json.loads(payload)


def test_proactive_artifact_success_uses_scope_and_preserves_arguments(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    secret = "sk-" + "a" * 24
    original_output = f"API_KEY={secret}\n" + "result\n" * 2_000
    arguments = json.dumps({"cmd": "python", "payload": "x" * 2_000})
    messages = _messages(output=original_output, arguments=arguments)

    with patch("tools.terminal_tool.get_active_env", return_value=env) as get_env:
        result, count = _compressor().prune_tool_results_only(
            messages,
            current_tokens=10_000,
            task_id="task-scope-A",
        )

    assert result is not messages
    assert count == 1
    get_env.assert_called_once_with("task-scope-A")
    assert result[1]["tool_calls"][0]["function"]["arguments"] == arguments

    metadata = _artifact_metadata(result[2]["content"])
    artifact_path = Path(metadata["path"])
    artifact_bytes = artifact_path.read_bytes()
    sanitized = redact_sensitive_text(original_output, force=True)
    assert artifact_bytes == sanitized.encode("utf-8")
    assert secret not in artifact_bytes.decode("utf-8")
    assert metadata == {
        "v": 1,
        "path": str(artifact_path),
        "chars": len(sanitized),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "redacted": True,
        "tool": "terminal",
        "call_id": "call-1",
        # The stub names the path exactly once -- a prose hint that repeated
        # the path made ~60% of every stub redundant path text.
        "read": "read_file the path above; sha256 is the content digest.",
    }
    assert str(artifact_path) not in metadata["read"]
    scope_hash = hashlib.sha256(b"task-scope-A").hexdigest()
    assert artifact_path.parent.name == scope_hash[:_ARTIFACT_SCOPE_HASH_CHARS]


def test_artifact_mode_never_uses_legacy_lossy_fallback(tmp_path):
    messages = _messages(output="same" * 2_000, arguments=json.dumps({"payload": "x" * 2_000}))
    messages[2]["content"] = messages[2]["content"]
    env = LocalArtifactEnv(tmp_path)

    with (
        patch("tools.terminal_tool.get_active_env", return_value=env),
        patch("agent.context_compressor.persist_tool_artifact", return_value=None),
        patch.object(ContextCompressor, "_prune_old_tool_results") as legacy,
    ):
        result, count = _compressor().prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )

    assert result is messages
    assert count == 0
    assert messages[1]["tool_calls"][0]["function"]["arguments"].endswith('"}')
    legacy.assert_not_called()


def test_artifact_mode_preserves_protected_head_results(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    head_output = "head" * 3_000
    middle_output = "middle" * 2_000
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "head", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "head", "content": head_output},
        {"role": "user", "content": "middle"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "middle", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "middle", "content": middle_output},
        {"role": "user", "content": "tail"},
        {"role": "assistant", "content": "tail reply"},
    ]

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor(protect_first_n=2).prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )

    assert count == 1
    assert result[2]["content"] == head_output
    assert result[5]["content"].startswith(ARTIFACT_RESULT_TAG)


def test_duplicate_call_ids_keep_occurrence_specific_tool_identity(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "dup",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "dup", "content": "first" * 2_000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "dup",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "dup", "content": "second" * 2_000},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "done"},
    ]

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor().prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )

    assert count == 2
    assert _artifact_metadata(result[2]["content"])["tool"] == "terminal"
    assert _artifact_metadata(result[4]["content"])["tool"] == "web_search"


def test_responses_style_call_id_is_used_for_tool_identity(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    messages = _messages(output="output" * 2_000, call_id="wire-id")
    tool_call = messages[1]["tool_calls"][0]
    tool_call["call_id"] = "wire-id"
    tool_call["id"] = "provider-item-id"

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor().prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )

    assert count == 1
    assert _artifact_metadata(result[2]["content"])["tool"] == "terminal"


def test_partial_failure_never_removes_reused_artifact(tmp_path):
    messages = _messages(output="first" * 2_000)
    messages[3:3] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "second" * 2_000},
    ]
    reused = PersistedToolArtifact(
        kind="output",
        path=str(tmp_path / "existing.txt"),
        chars=10_000,
        sha256="a" * 64,
        redacted=False,
        created=False,
    )

    with (
        patch("tools.terminal_tool.get_active_env", return_value=object()),
        patch("agent.context_compressor.persist_tool_artifact", side_effect=[reused, None]),
        patch("agent.context_compressor.remove_created_tool_artifact") as cleanup,
    ):
        result, count = _compressor().prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )

    assert result is messages
    assert count == 0
    cleanup.assert_not_called()


@pytest.mark.parametrize("failure", ["missing-env", "writer", "sanitizer"])
def test_artifact_failures_return_exact_original_object(tmp_path, failure):
    messages = _messages(output="output" * 2_000)
    env = None if failure == "missing-env" else LocalArtifactEnv(tmp_path)

    patches = [patch("tools.terminal_tool.get_active_env", return_value=env)]
    if failure == "writer":
        patches.append(patch("agent.context_compressor.persist_tool_artifact", return_value=None))
    elif failure == "sanitizer":
        patches.append(patch("tools.tool_result_storage.redact_sensitive_text", side_effect=ValueError("bad")))

    with patches[0]:
        with patches[1] if len(patches) > 1 else patch("builtins.id", wraps=id):
            result, count = _compressor().prune_tool_results_only(
                messages, current_tokens=10_000, task_id="task-scope"
            )

    assert result is messages
    assert count == 0


def test_reclaim_gate_rejection_removes_new_artifacts(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    messages = _messages(output="output" * 2_000)

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor(
            proactive_prune_min_reclaim_tokens=1_000_000,
        ).prune_tool_results_only(messages, current_tokens=10_000, task_id="task-scope")

    assert result is messages
    assert count == 0
    assert list((tmp_path / "hermes-results").rglob("*.output.txt")) == []


def test_cleanup_failure_still_fails_closed_with_original_object(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    messages = _messages(output="output" * 2_000)

    with (
        patch("tools.terminal_tool.get_active_env", return_value=env),
        patch("agent.context_compressor.remove_created_tool_artifact", return_value=False),
    ):
        result, count = _compressor(
            proactive_prune_min_reclaim_tokens=1_000_000,
        ).prune_tool_results_only(messages, current_tokens=10_000, task_id="task-scope")

    assert result is messages
    assert count == 0


def test_repeated_prune_is_idempotent_without_rewrite(tmp_path):
    env = LocalArtifactEnv(tmp_path)
    messages = _messages(output="output" * 2_000)
    compressor = _compressor()

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        first, first_count = compressor.prune_tool_results_only(
            messages, current_tokens=10_000, task_id="task-scope"
        )
        calls_after_first = env.execute_count
        artifact_path = Path(_artifact_metadata(first[2]["content"])["path"])
        mtime_after_first = artifact_path.stat().st_mtime_ns
        repeated, repeated_count = compressor.prune_tool_results_only(
            first, current_tokens=10_000, task_id="task-scope"
        )

    assert first_count == 1
    assert repeated is first
    assert repeated_count == 0
    assert env.execute_count == calls_after_first
    assert artifact_path.stat().st_mtime_ns == mtime_after_first


def test_loop_passes_task_scope_only_to_capable_builtin():
    class BuiltinLike:
        supports_proactive_prune_artifacts = True

        def __init__(self):
            self.kwargs = None

        def prune_tool_results_only(self, messages, **kwargs):
            self.kwargs = kwargs
            return messages, 0

    class PluginLike:
        def __init__(self):
            self.kwargs = None

        def prune_tool_results_only(self, messages, **kwargs):
            self.kwargs = kwargs
            return messages, 0

    class StaleSubclassLike:
        supports_proactive_prune_artifacts = True

        def __init__(self):
            self.current_tokens = None

        def prune_tool_results_only(self, messages, current_tokens=None):
            self.current_tokens = current_tokens
            return messages, 0

    messages = [{"role": "user", "content": "hello"}]
    builtin = BuiltinLike()
    plugin = PluginLike()
    stale_subclass = StaleSubclassLike()

    _run_proactive_tool_result_prune(builtin, messages, 123, "effective-task")
    _run_proactive_tool_result_prune(plugin, messages, 123, "effective-task")
    _run_proactive_tool_result_prune(stale_subclass, messages, 123, "effective-task")

    assert builtin.kwargs == {"current_tokens": 123, "task_id": "effective-task"}
    assert plugin.kwargs == {"current_tokens": 123}
    assert stale_subclass.current_tokens == 123


def test_proactive_artifact_skips_results_smaller_than_their_stub(tmp_path):
    """A result the reference stub would not shrink stays verbatim.

    The artifact stub carries two copies of an absolute sandbox path (``path``
    plus the ``read`` hint) alongside a SHA-256, so it has a ~500 char floor.
    ``proactive_prune_min_result_chars`` clamps to 200, so any operator who
    tunes the floor below the stub size opens a window where rewriting a tool
    result makes the transcript BIGGER — the opposite of pruning. Guard the
    invariant directly: never replace content with a longer reference.
    """
    env = LocalArtifactEnv(tmp_path)
    # Comfortably above the 200-char eligibility floor, comfortably below the
    # stub's own length. Sized relative to a real stub so the test cannot rot
    # if the stub layout shrinks further.
    small_output = "s" * 260
    messages = _messages(output=small_output)

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor().prune_tool_results_only(
            messages,
            current_tokens=10_000,
            task_id="task-scope-small",
        )

    # Nothing was rewritten, so the no-op contract hands back the INPUT object.
    assert count == 0
    assert result is messages
    assert result[2]["content"] == small_output
    # The rejected artifact must not be left behind on disk.
    assert list(tmp_path.rglob("*.output.txt")) == []


def test_proactive_artifact_prunes_large_and_skips_small_in_one_pass(tmp_path):
    """One non-beneficial result must not abort the whole pass.

    Regression guard for the "one bad apple" shape: a skipped item is an
    intentional no-op, not a persistence failure, so every other eligible
    result in the same batch must still be pruned.
    """
    env = LocalArtifactEnv(tmp_path)
    big_output = "b" * 20_000
    small_output = "s" * 260
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-big",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-big", "content": big_output},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-small",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-small", "content": small_output},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "done"},
    ]

    with patch("tools.terminal_tool.get_active_env", return_value=env):
        result, count = _compressor().prune_tool_results_only(
            messages,
            current_tokens=10_000,
            task_id="task-scope-mixed",
        )

    assert count == 1
    assert result is not messages
    # Large result pruned to a reference stub...
    assert result[2]["content"].startswith(ARTIFACT_RESULT_TAG)
    # ...small result kept verbatim.
    assert result[4]["content"] == small_output
    # Exactly one artifact survives on disk (the skipped one was cleaned up).
    assert len(list(tmp_path.rglob("*.output.txt"))) == 1
