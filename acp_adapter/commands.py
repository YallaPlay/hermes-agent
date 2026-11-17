"""Headless slash commands for ACP sessions (``/help``, ``/model``, ``/compress`` ...)."""

from __future__ import annotations

import contextvars
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from acp.schema import AvailableCommand, AvailableCommandsUpdate, UnstructuredCommandInput

from acp_adapter.session import SessionState, _expand_acp_enabled_toolsets

logger = logging.getLogger("acp_adapter.server")


def _estimate_tokens(history: list, agent: Any, system_prompt: str | None = None, tools: Any = None) -> int:
    """Rough request-token estimate over history + system prompt + tool schemas."""
    from agent.model_metadata import estimate_request_tokens_rough

    if system_prompt is None:
        system_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    if tools is None:
        tools = getattr(agent, "tools", None) or None
    return estimate_request_tokens_rough(history, system_prompt=system_prompt, tools=tools)


def _queue_prompt(state: SessionState, text: str) -> int:
    with state.runtime_lock:
        state.queued_prompts.append(text)
        return len(state.queued_prompts)


def _queue_if_busy(state: SessionState, text: str) -> int | None:
    """Queue ``text`` behind a live turn and return the queue depth; ``None`` when idle."""
    with state.runtime_lock:
        if not (state.is_running or state.command_op):
            return None
        state.queued_prompts.append(text)
        return len(state.queued_prompts)


@dataclass(frozen=True)
class SlashReply:
    """A command reply that also asks the session to run ``prompt`` next as a normal turn."""
    text: str
    prompt: str | None = None
    session_info_changed: bool = False


def _goal_session_key(state: SessionState) -> str:
    """The id the session's goal row lives under. Compression rotates ``agent.session_id`` and
    ``carry_compression_automation`` moves the goal to the new id, so follow the live agent; the
    ACP id is the key only before the first agent exists."""
    sid = getattr(state.agent, "session_id", None)
    return sid if isinstance(sid, str) and sid else state.session_id


def _goal_manager(state: SessionState):
    """The session's GoalManager (``_goal_session_key`` explains the key)."""
    from hermes_cli import goals
    from hermes_cli.config import load_config_readonly

    try:
        max_turns = int((load_config_readonly().get("goals") or {}).get("max_turns", 20) or 20)
    except Exception:
        max_turns = 20
    return goals.GoalManager(session_id=_goal_session_key(state), default_max_turns=max_turns)


def _is_successful_goal_turn(result: Any, final_response: str) -> bool:
    """Whether a turn produced a response the goal judge can use (tui_gateway parity: failed and
    interrupted turns stay out; a max-iteration handoff is a resumable boundary, not a failure)."""
    from agent.turn_failure_copy import is_max_iteration_handoff

    if not final_response.strip():
        return False
    if not isinstance(result, dict):
        return True
    if result.get("failed") or result.get("interrupted") or result.get("compression_exhausted"):
        return False
    return result.get("completed") is not False or is_max_iteration_handoff(result)


# Commands that mutate shared turn state must not run beside a live turn or beside each
# other: slash dispatch happens on a worker thread while the turn iterates state.history and
# reads agent._session_db, so clearing, rebinding, or swapping state.agent underneath
# run_conversation tears the running turn. Gateway parity: all three are idle-only there.
# The command_op flag is held for the whole handler so a turn cannot claim the session in
# the check-then-act window (the /compress LLM call and /model agent rebuild take seconds).
_MID_TURN_BLOCKED_COMMANDS = frozenset({"reset", "compress", "model", "undo", "retry"})


def _skill_command_entries() -> list[tuple[str, str]]:
    """``(name, description)`` for each interactive skill command; empty when skills are unavailable."""
    try:
        from agent.skill_commands import get_interactive_skill_commands

        commands = get_interactive_skill_commands() or {}
    except Exception:
        logger.debug("Could not list skill commands for ACP", exc_info=True)
        return []
    entries = []
    for key in sorted(commands):
        description = str(commands[key].get("description") or "").strip() or "Run this skill"
        entries.append((key.lstrip("/"), description[:200]))
    return entries


def _skill_invocation(cmd: str, args: str, state: SessionState) -> SlashReply | None:
    """The turn a ``/<skill> [instruction]`` command runs, or None when ``cmd`` is not a skill."""
    try:
        from agent import skill_commands as sc

        commands = sc.get_interactive_skill_commands() or {}
    except Exception:
        logger.debug("Could not resolve skill commands for ACP", exc_info=True)
        return None
    key = f"/{cmd}"
    if key not in commands:
        return None
    # Stacked leading /skill tokens (CLI, TUI and gateway parity).
    extra_keys, instruction = sc.split_stacked_skill_commands(args, interactive=True)
    if extra_keys:
        stacked = sc.build_stacked_skill_invocation_message([key, *extra_keys], instruction, task_id=state.session_id)
        if stacked:
            message, loaded, missing = stacked
            notice = f"⚡ Loading {len(loaded)} stacked skills: {', '.join(loaded)}"
            if missing:
                notice += f"\nSkipped missing skills: {', '.join(missing)}"
            return SlashReply(notice, prompt=message)
    message = sc.build_skill_invocation_message(key, instruction, task_id=state.session_id)
    if not message:
        return SlashReply(f"Could not load skill: {cmd}")
    return SlashReply(f"⚡ Loading skill: {commands[key].get('name', cmd)}", prompt=message)


# Registry commands this adapter cannot serve (terminal UI, multi-session, or messaging-platform
# features). They get a reply instead of reaching the model as text, which would waste a turn.
_EDITOR_ALTERNATIVES = {
    "branch": "Use the editor's fork session action.",
    "resume": "Open the session from the editor's session list.",
    "sessions": "Use the editor's session list.",
}

# Shared executors that are surface plumbing, not commands a user types.
_NON_USER_EXECUTORS = frozenset({"gateway_help", "gateway_commands"})


def _registry_fallback(cmd: str, args: str, state: SessionState) -> str | None:
    """Answer a Hermes registry command this adapter has no handler for.

    A registry-owned executor (``/profile``, ``/bundles`` ...) runs here; any other registry
    command gets a reply that names it. Names outside the registry return None and reach the
    model as normal text (``/tmp/x is broken`` is a message, not a command)."""
    from hermes_cli.commands import resolve_command

    cmd_def = resolve_command(cmd)
    if cmd_def is None:
        return None
    from hermes_cli.observability.shared_metrics_events import record_slash_command
    record_slash_command(command=cmd_def.name, surface="acp")
    if (cmd_def.execute or "") not in _NON_USER_EXECUTORS:
        from hermes_cli.slash_exec import CommandContext, run_execute
        try:
            reply = run_execute(cmd_def, CommandContext(surface="acp", args=args))
        except Exception as exc:
            logger.warning("ACP /%s executor failed", cmd_def.name, exc_info=True)
            return f"/{cmd_def.name} failed: {exc}"
        if reply is not None:
            return reply.text
    hint = _EDITOR_ALTERNATIVES.get(cmd_def.name, "Run it in the Hermes CLI or TUI.")
    return f"/{cmd_def.name} is not available in the editor. {hint}"


class SlashCommandsMixin:
    """Slash-command surface for ``HermesACPAgent``; relies on ``_conn``, ``_send``, ``_schedule_soon``,
    ``session_manager`` and ``_switch_model`` from the host class."""

    # name -> (help text, advertised description, input hint)
    _COMMANDS: dict[str, tuple[str, str, str | None]] = {
        "help": ("Show available commands", "List available commands", None),
        "model": (
            "Show or change current model",
            "Show current model and provider, or switch models",
            "model name to switch to",
        ),
        "tools": ("List available tools", "List available tools with descriptions", None),
        "context": ("Show conversation context info", "Show conversation message counts by role", None),
        "reset": ("Clear conversation history", "Clear conversation history", None),
        "compress": ("Compress conversation context", "Compress conversation context", None),
        "steer": (
            "Inject guidance into the currently running agent turn",
            "Inject guidance into the currently running agent turn",
            "guidance for the active turn",
        ),
        "queue": (
            "Queue a prompt to run after the current turn finishes",
            "Queue a prompt to run after the current turn finishes",
            "prompt to run next",
        ),
        "version": ("Show Hermes version", "Show Hermes version", None),
        "goal": (
            "Set a standing goal Hermes works on across turns until done",
            "Set a standing goal Hermes works on across turns until done (status, pause, resume, clear)",
            "goal text, or status | pause | resume | clear",
        ),
        "subgoal": (
            "Add criteria to the active goal",
            "Add, remove (remove <n>), or clear extra criteria for the active goal",
            "criterion text, or remove <n> | clear",
        ),
        "usage": ("Show token usage for this session", "Show token usage and context for this session", None),
        "status": ("Show session status", "Show session, model, context, and goal status", None),
        "title": ("Show or set the session title", "Show or set the session title", "new title"),
        "undo": ("Remove the last N user turns and their replies", "Remove the last N user turns and their replies",
                 "number of turns"),
        "retry": ("Resend the last user message", "Remove the last exchange and resend the last user message", None),
        "plan": ("Write an implementation plan for a task", "Write a markdown implementation plan for a task",
                 "task to plan"),
        "learn": ("Turn a workflow into a reusable skill", "Turn a workflow from this session into a reusable skill",
                  "what to learn"),
        "init": ("Create or update AGENTS.md for this workspace", "Create or update AGENTS.md for this workspace",
                 "extra instructions"),
    }


    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands = [
            AvailableCommand(name=name, description=desc, input=UnstructuredCommandInput(hint=hint) if hint else None)
            for name, (_help, desc, hint) in cls._COMMANDS.items()
        ]
        for name, description in _skill_command_entries():
            if name not in cls._COMMANDS:  # a built-in owns its name (agent.skill_commands guard parity)
                commands.append(AvailableCommand(
                    name=name, description=description, input=UnstructuredCommandInput(hint="instructions")))
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return
        update = AvailableCommandsUpdate(
            session_update="available_commands_update", available_commands=self._available_commands()
        )
        await self._send(session_id, update, fail_msg="Failed to advertise ACP slash commands for session %s")

    def _schedule_available_commands_update(self, session_id: str) -> None:
        self._schedule_soon(lambda: self._send_available_commands_update(session_id))

    def _handle_slash_command(self, text: str, state: SessionState) -> str | SlashReply | None:
        """Dispatch a slash command; ``None`` for unknown ones so they fall through to the LLM.

        A ``SlashReply`` carrying ``prompt`` asks ``prompt()`` to run that text as a normal turn
        (skills, ``/goal`` kickoff, ``/retry``, ``/plan`` ...)."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        if cmd == "compact":
            cmd = "compress"
        elif cmd in {"new", "clear"}:  # gateway-mode ACP parity: both reset this session
            cmd = "reset"
        elif cmd not in self._COMMANDS:  # registry aliases (/q -> /queue ...)
            from hermes_cli.commands import resolve_command
            cmd_def = resolve_command(cmd) if cmd else None
            if cmd_def is not None and cmd_def.name in self._COMMANDS:
                cmd = cmd_def.name
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd not in self._COMMANDS:
            if not cmd:
                return None
            return _skill_invocation(cmd, args, state) or _registry_fallback(cmd, args, state)
        from hermes_cli.observability.shared_metrics_events import record_slash_command
        record_slash_command(command=cmd, surface="acp")
        mutating = cmd in _MID_TURN_BLOCKED_COMMANDS
        if mutating:
            with state.runtime_lock:
                if state.is_running or state.command_op:
                    return (f"⏳ Session is busy; /{cmd} only works while the session is "
                            "idle. Wait for the current response or cancel first.")
                state.command_op = True
        handler = getattr(self, f"_cmd_{cmd}")

        # Handlers run outside the per-turn cwd-pinning context. ``/compress``
        # and ``/model`` REBUILD the system prompt, so unpinned they'd bake the Hermes install tree
        # into the persisted cached prompt. Pin inside a fresh context: no leak, no teardown.
        def _dispatch() -> str | None:
            try:
                from agent.runtime_cwd import set_session_cwd

                set_session_cwd(state.cwd)
            except Exception:
                logger.debug("Could not pin ACP session cwd for slash command", exc_info=True)
            return handler(args, state)

        try:
            return contextvars.copy_context().run(_dispatch)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"
        finally:
            if mutating:
                with state.runtime_lock:
                    state.command_op = False

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        lines.extend(f"  /{cmd:10s}  {desc}" for cmd, (desc, _adv, _hint) in self._COMMANDS.items())
        skills = [name for name, _desc in _skill_command_entries() if name not in self._COMMANDS]
        if skills:
            lines.extend(["", f"Skills ({len(skills)}): type /<skill-name> [instructions] to load one."])
        lines.extend(["", "Other Hermes commands say where to run them. Any other /text is sent to the model."])
        return "\n".join(lines)

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        current_provider, target_provider, new_model = self._switch_model(state, args)
        provider_label = getattr(state.agent, "provider", None) or target_provider or current_provider or "openrouter"
        logger.info("Session %s: model switched to %s", state.session_id, new_model)
        from hermes_cli.model_picker_usage import large_model_notice
        notice = large_model_notice(new_model, target_provider or "", lang="en")  # window plans only; cache-only read; ACP text is English
        return f"Model switched to: {new_model}\nProvider: {provider_label}" + (f"\n⚠ {notice}" if notice else "")

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            from types import SimpleNamespace
            from agent.memory_manager_tools import inject_memory_provider_tools

            toolsets = _expand_acp_enabled_toolsets(getattr(state.agent, "enabled_toolsets", None))
            tools = get_tool_definitions(
                enabled_toolsets=toolsets,
                disabled_toolsets=getattr(state.agent, "disabled_toolsets", None),
                quiet_mode=True,
            )
            tool_view = SimpleNamespace(
                tools=list(tools or []),
                valid_tool_names={t.get("function", {}).get("name") for t in tools or [] if isinstance(t, dict)},
                enabled_toolsets=toolsets,
                disabled_toolsets=getattr(state.agent, "disabled_toolsets", None),
                _memory_manager=getattr(state.agent, "_memory_manager", None),
            )
            inject_memory_provider_tools(tool_view)
            tools = tool_view.tools
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = (t.get("function") or {}).get("name", "?")
                desc = (t.get("function") or {}).get("description", "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        """Show ACP session context pressure and compression guidance."""
        n_messages = len(state.history)
        roles = Counter(msg.get("role", "unknown") for msg in state.history)

        agent = state.agent
        model = state.model or getattr(agent, "model", "")
        provider = getattr(agent, "provider", None) or "auto"
        compressor = getattr(agent, "context_compressor", None)
        context_length = int(getattr(compressor, "context_length", 0) or 0)
        threshold_tokens = int(getattr(compressor, "threshold_tokens", 0) or 0)

        try:
            approx_tokens = _estimate_tokens(state.history, agent)
        except Exception:
            logger.debug("Could not estimate ACP context usage", exc_info=True)
            approx_tokens = 0

        if threshold_tokens <= 0 and context_length > 0:
            threshold_tokens = int(context_length * 0.80)

        lines = [
            f"Conversation: {n_messages} messages" if n_messages else "Conversation is empty (no messages yet).",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        if model:
            lines.append(f"Model: {model}")
        lines.append(f"Provider: {provider}")

        if approx_tokens > 0 and context_length > 0:
            usage_pct = (approx_tokens / context_length) * 100
            lines.append(f"Context usage: ~{approx_tokens:,} / {context_length:,} tokens ({usage_pct:.1f}%)")
        elif approx_tokens > 0:
            lines.append(f"Context usage: ~{approx_tokens:,} tokens")

        if threshold_tokens > 0 and approx_tokens > 0:
            threshold_pct = (threshold_tokens / context_length) * 100 if context_length > 0 else 0
            pct_note = f", {threshold_pct:.0f}%" if threshold_pct else ""
            if approx_tokens >= threshold_tokens:
                lines.append(f"Compression: due now (threshold ~{threshold_tokens:,}{pct_note}). Run /compress.")
            else:
                remaining = max(threshold_tokens - approx_tokens, 0)
                lines.append(f"Compression: ~{remaining:,} tokens until threshold (~{threshold_tokens:,}{pct_note}).")
        elif threshold_tokens > 0:
            lines.append(f"Compression threshold: ~{threshold_tokens:,} tokens")

        lines.append(
            "Auto-compaction is disabled (compression.enabled: false); /compress still compresses manually."
            if getattr(agent, "compression_enabled", True) is False
            else "Tip: run /compress to compress manually before the threshold."
        )
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        self._reconcile_persisted_history(state, [])
        try:
            reset_session_state = getattr(state.agent, "reset_session_state", None)
            if callable(reset_session_state):
                reset_session_state()
        except Exception:
            logger.warning("ACP session state reset failed for %s", state.session_id, exc_info=True)
            return "Conversation history cleared. Agent session state reset failed; see logs."
        finally:
            self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compress(self, args: str, state: SessionState) -> str:
        """``/compress [here [N] | <focus>] [--preview] [--aggressive]`` through the shared core."""
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.conversation_compression_manual import (
            AGGRESSIVE_UNSUPPORTED, compress_now, parse_compress_args, render_compress_result)

        if not state.history:
            return "Nothing to compress — conversation is empty."
        agent = state.agent
        # No compression_enabled gate: it only disables *automatic* compaction (CLI/gateway parity).
        if not hasattr(agent, "_compress_context"):
            return "Context compression not available for this agent."
        request = parse_compress_args(args)
        if request.aggressive:
            return AGGRESSIVE_UNSUPPORTED
        original_session_db = getattr(agent, "_session_db", None)
        try:
            # Stable ACP session id: suppress _compress_context's SQLite session split.
            agent._session_db = None
            result = compress_now(
                agent, state.history, request, system_message=getattr(agent, "_cached_system_prompt", "") or "",
                task_id=state.session_id)
        except Exception as e:
            return f"Compression failed: {e}"
        finally:
            agent._session_db = original_session_db
        if result.status != "compressed":
            return "\n".join(render_compress_result(result))
        state.history = result.after_messages
        self._reconcile_persisted_history(state, state.history)
        self.session_manager.save_session(state.session_id)
        finalize_context_engine_compression_notification(agent, committed=True)
        return (
            f"Context compressed: {len(result.before_messages)} -> {len(state.history)} messages\n"
            f"~{result.before_tokens:,} -> ~{result.after_tokens:,} tokens"
        )

    def _cmd_steer(self, args: str, state: SessionState) -> str:
        steer_text = args.strip()
        if not steer_text:
            return "Usage: /steer <guidance>"

        if state.is_running and hasattr(state.agent, "steer"):
            try:
                if state.agent.steer(steer_text):
                    preview = steer_text[:80] + ("..." if len(steer_text) > 80 else "")
                    return f"⏩ Steer queued for the active turn: {preview}"
            except Exception as exc:
                logger.warning("ACP steer failed for session %s: %s", state.session_id, exc)
                return f"⚠️ Steer failed: {exc}"

        return f"No active turn — queued for the next turn. ({_queue_prompt(state, steer_text)} queued)"

    def _cmd_queue(self, args: str, state: SessionState) -> str:
        queued_text = args.strip()
        if not queued_text:
            return "Usage: /queue <prompt>"
        return f"Queued for the next turn. ({_queue_prompt(state, queued_text)} queued)"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        from hermes_cli.version_info import get_version_info

        return f"Hermes Agent v{get_version_info().derived_version}"

    # ---- /goal, /subgoal -------------------------------------------------------------------

    def _cmd_goal(self, args: str, state: SessionState) -> str | SlashReply:
        """``/goal <text>`` sets a standing goal and kicks it off; status/pause/resume/clear control it.
        After each turn ``_goal_after_turn`` asks the judge whether to continue."""
        from hermes_cli.goal_command import dispatch_goal_command
        from hermes_cli.goals import last_user_message_content

        mgr = _goal_manager(state)
        # The VS Code user owns this local session, so gate commands need no extra approval
        # (CLI and TUI parity).
        result = dispatch_goal_command(
            mgr, args, authorize_gate=lambda: None,
            last_user_message=last_user_message_content(state.history),
        )
        if result.error or not result.prompt:
            return result.output
        notice = result.output if result.kickoff else f"{result.output}\nContinuing now: taking the next step."
        return SlashReply(notice, prompt=result.prompt)

    def _cmd_subgoal(self, args: str, state: SessionState) -> str:
        """``/subgoal`` lists, ``<text>`` appends, ``remove <n>`` / ``clear`` edit the goal's extra
        criteria. They join the judge and continuation prompts at the next turn boundary."""
        mgr = _goal_manager(state)
        if not mgr.has_goal():
            return "No goal set. Set one with /goal <text> first."
        if not args:
            return f"{mgr.status_line()}\n{mgr.render_subgoals()}"
        verb, _, rest = args.partition(" ")
        verb, rest = verb.lower(), rest.strip()
        try:
            if verb == "remove":
                if not rest:
                    return "Usage: /subgoal remove <n>"
                try:
                    index = int(rest.split()[0])
                except ValueError:
                    return "/subgoal remove needs a number (see /subgoal for the list)."
                return f"✓ Removed subgoal {index}: {mgr.remove_subgoal(index)}"
            if verb == "clear":
                count = mgr.clear_subgoals()
                return f"✓ Cleared {count} subgoal{'s' if count != 1 else ''}." if count else "No subgoals to clear."
            text = mgr.add_subgoal(args)
        except (IndexError, ValueError, RuntimeError) as exc:
            return f"/subgoal: {exc}"
        return f"✓ Added subgoal {len(mgr.state.subgoals) if mgr.state else 0}: {text}"

    def _goal_after_turn(self, state: SessionState, result: Any, final_response: str) -> tuple[str, str | None]:
        """Judge an active goal after a finished turn: ``(verdict message, continuation prompt)``.

        Runs off the event loop (the judge is an LLM call). Failed, interrupted and empty turns are
        never judge input. Any error here is logged and ends the loop; it never fails the turn."""
        try:
            mgr = _goal_manager(state)
            if not mgr.is_active() or not _is_successful_goal_turn(result, final_response):
                return "", None
            from hermes_cli.goals import count_active_delegations, gather_background_processes

            try:
                # Only this session's processes (ACP turns register under the ACP session id).
                background = gather_background_processes(owner_task_id=state.session_id)
                delegations = count_active_delegations(_goal_session_key(state))
            except Exception:
                background, delegations = None, 0
            decision = mgr.evaluate_after_turn(
                final_response, user_initiated=True, background_processes=background,
                active_delegations=delegations)
        except Exception:
            logger.warning("ACP goal continuation hook failed for %s", state.session_id, exc_info=True)
            return "", None
        prompt = decision.get("continuation_prompt") if decision.get("should_continue") else None
        return str(decision.get("message") or ""), prompt or None

    async def _run_goal_continuation(self, state: SessionState, session_id: str, conn: Any, text: str) -> None:
        """Run the goal's next step as a turn. Internal mode: it never redirects a live turn, is
        never parsed as a slash command, and never revives a prompt the user cancelled."""
        from acp import update_user_message_text
        from acp.schema import TextContentBlock

        if conn:
            try:
                await conn.session_update(session_id, update_user_message_text(text))
            except Exception:
                logger.debug("Goal continuation echo failed for %s; running it anyway", session_id, exc_info=True)
        try:
            await self.prompt(prompt=[TextContentBlock(type="text", text=text)], session_id=session_id,
                              _synthetic_notification=True)
        except Exception:
            logger.exception("ACP goal continuation turn failed for %s", session_id)

    # ---- /usage, /status, /title -----------------------------------------------------------

    def _cmd_usage(self, args: str, state: SessionState) -> str:
        from gateway.session_usage import usage_stats

        agent = state.agent
        if agent is None:
            return "No active agent: send a message first."
        stats = usage_stats(agent)

        def n(key: str) -> str:
            return f"{int(stats.get(key) or 0):,}"

        rows = [("Input tokens", n("input_tokens")), ("Output tokens", n("output_tokens"))]
        if stats.get("reasoning_tokens"):
            rows.append(("Reasoning tokens", n("reasoning_tokens")))
        if stats.get("cache_read_tokens"):
            rows.append(("Cache read tokens", n("cache_read_tokens")))
        rows += [("Total tokens", n("total_tokens")), ("API calls", n("api_calls"))]
        if stats.get("context_length"):
            pct = round(int(stats.get("context_tokens") or 0) / int(stats["context_length"]) * 100)
            rows.append(("Current context", f"{n('context_tokens')} / {n('context_length')} ({pct}%)"))
        rows += [("Messages", f"{len(state.history):,}"), ("Compressions", n("compression_count"))]
        lines = ["Session token usage", f"Model: {stats.get('model') or state.model or '(unknown)'}"]
        lines += [f"  {label + ':':<20}{value}" for label, value in rows]
        return "\n".join(lines)

    def _cmd_status(self, args: str, state: SessionState) -> str:
        agent = state.agent
        title = None
        try:
            title = self.session_manager.get_session_title(state.session_id)
        except Exception:
            logger.debug("Could not read ACP session title for /status", exc_info=True)
        lines = [
            f"Session: {state.session_id}" + (f" ({title})" if title else ""),
            f"Model: {state.model or getattr(agent, 'model', '') or '(unknown)'}",
            f"Provider: {getattr(agent, 'provider', None) or 'auto'}",
            f"Workspace: {state.cwd}",
            f"Messages: {len(state.history)}",
            f"Running: {'yes' if state.is_running else 'no'}",
        ]
        with state.runtime_lock:
            queued = len(state.queued_prompts)
        if queued:
            lines.append(f"Queued prompts: {queued}")
        try:
            mgr = _goal_manager(state)
            if mgr.has_goal():
                lines.append(mgr.status_line())
        except Exception:
            logger.debug("Could not read ACP goal state for /status", exc_info=True)
        return "\n".join(lines)

    def _cmd_title(self, args: str, state: SessionState) -> str | SlashReply:
        if not args:
            title = self.session_manager.get_session_title(state.session_id)
            return f"Session title: {title}" if title else "This session has no title. Set one with /title <name>."
        try:
            ok = self.session_manager.set_session_title(state.session_id, args)
        except ValueError as exc:
            return f"Could not set title: {exc}"
        if not ok:
            return "Could not set title: the session is not saved yet. Send a message first."
        title = self.session_manager.get_session_title(state.session_id) or args
        return SlashReply(f"Session title set: {title}", session_info_changed=True)

    # ---- /undo, /retry ---------------------------------------------------------------------

    def _rewind_user_turns(self, state: SessionState, turns_back: int, *, retry: bool) -> tuple[str, Any] | None:
        """Cut history back to before the ``turns_back``-th last user turn: ``(status, content)``,
        or None when there are not enough turns. ``/retry`` checks the turn is resendable as text
        BEFORE cutting, so a refused retry changes nothing."""
        from agent.context_compressor import (
            history_before_user_originated_turn, is_user_originated_turn, retryable_user_text)

        indexes = [i for i, message in enumerate(state.history) if is_user_originated_turn(message)]
        if len(indexes) < turns_back:
            return None
        prefix, live_view = history_before_user_originated_turn(state.history, indexes[-turns_back])
        content = (live_view or {}).get("content")
        if retry:
            content = _strip_owner_note(retryable_user_text(content))
        removed = len(state.history) - len(prefix)
        state.history = list(prefix)
        self._reconcile_persisted_history(state, state.history)
        self.session_manager.save_session(state.session_id)
        return f"{removed} message{'s' if removed != 1 else ''}", content

    def _cmd_undo(self, args: str, state: SessionState) -> str:
        if args and not (args.isdigit() and int(args) > 0):
            return "Usage: /undo [N]"
        turns = int(args or 1)
        rewound = self._rewind_user_turns(state, turns, retry=False)
        if rewound is None:
            return "Nothing to undo: no earlier user message."
        removed, content = rewound
        preview = content if isinstance(content, str) else ""
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        lines = [f"Undid {turns} turn{'s' if turns != 1 else ''} ({removed} removed)."]
        if preview:
            lines.append(f"Removed message: {preview}")
        lines.append("The chat view still shows the removed messages until you reopen the session.")
        return "\n".join(lines)

    def _cmd_retry(self, args: str, state: SessionState) -> str | SlashReply:
        try:
            rewound = self._rewind_user_turns(state, 1, retry=True)
        except ValueError as exc:
            return f"/retry: {exc}"
        if rewound is None or not str(rewound[1] or "").strip():
            return "Nothing to retry: no earlier user message."
        return SlashReply("↻ Retrying the last message.", prompt=rewound[1])

    # ---- /plan, /learn, /init: build a prompt, then run it as a normal turn ----------------

    def _cmd_plan(self, args: str, state: SessionState) -> SlashReply:
        from agent.plan_prompt import build_plan_prompt

        return SlashReply("", prompt=build_plan_prompt(args))

    def _cmd_learn(self, args: str, state: SessionState) -> SlashReply:
        from agent.learn_prompt import build_learn_prompt

        return SlashReply("", prompt=build_learn_prompt(args))

    def _cmd_init(self, args: str, state: SessionState) -> SlashReply:
        from hermes_cli.init_command import build_init_prompt_for_cwd

        return SlashReply("", prompt=build_init_prompt_for_cwd(cwd=state.cwd, extra=args, session_key=state.session_id))


def _strip_owner_note(text: str) -> str:
    """Drop the first-turn owner note ``prompt()`` prepends, so a retried first turn does not carry
    it twice (``prompt()`` adds it again when history is empty)."""
    from hermes_constants import ACP_OWNER_NOTE_PREFIX

    if text.startswith(ACP_OWNER_NOTE_PREFIX):
        _note, sep, rest = text.partition("\n\n")
        if sep:
            return rest
    return text
