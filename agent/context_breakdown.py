"""Live session context-window breakdown for UI surfaces.

Estimates how the next provider request is composed: system prompt tiers,
tool schemas, and conversation history. Uses the same rough char/4 heuristic
as ``agent.model_metadata.estimate_request_tokens_rough`` so numbers align
with compression thresholds — not exact tokenizer counts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

# `- name: description` entries inside the skills index (one per skill).
_SKILL_ENTRY_RE = re.compile(r"^\s*-\s+([\w:-]+)\s*:", re.MULTILINE)

# `## path/to/file.md` section headings inside the project-context block
# (see prompt_builder.build_context_files_prompt — one per loaded file).
_CONTEXT_FILE_RE = re.compile(r"^## (.+)$", re.MULTILINE)

_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}


def _chars_to_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _json_tokens(value: Any) -> int:
    if not value:
        return 0
    return _chars_to_tokens(json.dumps(value, ensure_ascii=False))


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _split_tools(tools: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    builtin: List[dict] = []
    mcp: List[dict] = []
    subagent: List[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> Tuple[str, str]:
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _injected_memory_blocks(agent: Any) -> Tuple[str, str]:
    """Per-turn injections appended to the API-side copy of the user message.

    ``build_turn_context`` stashes the external memory provider's prefetch
    (e.g. Mnemosyne's ``## Mnemosyne Context``) and any ``pre_llm_call``
    plugin context on the agent. Neither ever enters the stored conversation
    history — the loop appends them to a *copy* of the user message at
    request time — so without these the breakdown/report understates what
    the model actually received.
    """
    prefetch = str(getattr(agent, "_last_ext_prefetch_cache", "") or "")
    plugin_ctx = str(getattr(agent, "_last_plugin_user_context", "") or "")
    return prefetch.strip(), plugin_ctx.strip()


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def _flatten_content(content: Any) -> str:
    """Best-effort plain-text view of a message ``content`` field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(str(part.get("text") or ""))
                elif ptype in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(f"[{ptype or 'content'}]")
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def _collect_sections(agent: Any, messages: Optional[List[dict]]) -> Dict[str, Any]:
    """Gather the raw section texts/objects shared by the breakdown and report."""
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    memory_block, user_block = _memory_blocks(agent)
    injected_prefetch, injected_plugin_ctx = _injected_memory_blocks(agent)
    memory_text = "\n\n".join(
        part
        for part in (memory_block, user_block, injected_prefetch, injected_plugin_ctx)
        if part
    ).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)

    return {
        "system_core": system_core,
        "system_tail": system_tail,
        "system_prompt_text": system_prompt_text,
        "context": context,
        "skills_index": skills_index,
        "memory_block": memory_block,
        "user_block": user_block,
        "injected_prefetch": injected_prefetch,
        "injected_plugin_ctx": injected_plugin_ctx,
        "memory_text": memory_text,
        "builtin_tools": builtin_tools,
        "mcp_tools": mcp_tools,
        "subagent_tools": subagent_tools,
        "messages": list(messages or []),
    }


def _tool_detail(tools: Sequence[dict]) -> List[Dict[str, Any]]:
    rows = [
        {"label": _tool_name(tool) or "(unnamed)", "tokens": _json_tokens(tool)}
        for tool in tools
    ]
    rows.sort(key=lambda r: -int(r["tokens"]))
    return rows


def _rules_detail(context: str) -> List[Dict[str, Any]]:
    """One row per loaded context file (## headings in the project-context block)."""
    rows: List[Dict[str, Any]] = []
    headings = list(_CONTEXT_FILE_RE.finditer(context))
    for idx, match in enumerate(headings):
        start = match.start()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(context)
        rows.append(
            {"label": match.group(1).strip(), "tokens": _chars_to_tokens(context[start:end])}
        )
    return rows


def _skills_detail(skills_index: str) -> List[Dict[str, Any]]:
    count = len(_SKILL_ENTRY_RE.findall(skills_index))
    if not count:
        return []
    return [{"label": f"{count} skills indexed", "tokens": None}]


def _memory_detail(
    memory_block: str,
    user_block: str,
    injected_prefetch: str = "",
    injected_plugin_ctx: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if memory_block:
        rows.append({"label": "Agent memory", "tokens": _chars_to_tokens(memory_block)})
    if user_block:
        rows.append({"label": "User profile", "tokens": _chars_to_tokens(user_block)})
    if injected_prefetch:
        rows.append(
            {
                "label": "Memory provider prefetch (injected this turn)",
                "tokens": _chars_to_tokens(injected_prefetch),
            }
        )
    if injected_plugin_ctx:
        rows.append(
            {
                "label": "Plugin context (injected this turn)",
                "tokens": _chars_to_tokens(injected_plugin_ctx),
            }
        )
    return rows


def _conversation_detail(messages: Sequence[dict]) -> List[Dict[str, Any]]:
    if not messages:
        return []
    from agent.model_metadata import estimate_messages_tokens_rough

    by_role: Dict[str, List[dict]] = {}
    for msg in messages:
        role = str(msg.get("role") or "other")
        by_role.setdefault(role, []).append(msg)
    rows: List[Dict[str, Any]] = []
    for role in ("user", "assistant", "tool", "system"):
        group = by_role.pop(role, [])
        if group:
            rows.append(
                {
                    "label": f"{len(group)} {role} message{'s' if len(group) != 1 else ''}",
                    "tokens": estimate_messages_tokens_rough(group),
                }
            )
    for role, group in by_role.items():
        rows.append(
            {
                "label": f"{len(group)} {role} message{'s' if len(group) != 1 else ''}",
                "tokens": estimate_messages_tokens_rough(group),
            }
        )
    return rows


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    from agent.model_metadata import estimate_messages_tokens_rough

    sections = _collect_sections(agent, messages)

    conversation_tokens = estimate_messages_tokens_rough(sections["messages"])

    categories = [
        (
            "system_prompt",
            "System prompt",
            _chars_to_tokens(sections["system_prompt_text"]),
            [],
        ),
        (
            "tool_definitions",
            "Tool definitions",
            _json_tokens(sections["builtin_tools"]),
            _tool_detail(sections["builtin_tools"]),
        ),
        (
            "rules",
            "Rules",
            _chars_to_tokens(sections["context"]),
            _rules_detail(sections["context"]),
        ),
        (
            "skills",
            "Skills",
            _chars_to_tokens(sections["skills_index"]),
            _skills_detail(sections["skills_index"]),
        ),
        (
            "mcp",
            "MCP",
            _json_tokens(sections["mcp_tools"]),
            _tool_detail(sections["mcp_tools"]),
        ),
        (
            "subagent_definitions",
            "Subagent definitions",
            _json_tokens(sections["subagent_tools"]),
            _tool_detail(sections["subagent_tools"]),
        ),
        (
            "memory",
            "Memory",
            _chars_to_tokens(sections["memory_text"]),
            _memory_detail(
                sections["memory_block"],
                sections["user_block"],
                sections["injected_prefetch"],
                sections["injected_plugin_ctx"],
            ),
        ),
        (
            "conversation",
            "Conversation",
            conversation_tokens,
            _conversation_detail(sections["messages"]),
        ),
    ]

    estimated_total = sum(tokens for _, _, tokens, _ in categories)

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    context_used = measured_used if measured_used > 0 else estimated_total
    context_percent = (
        max(0, min(100, round(context_used / context_max * 100)))
        if context_max
        else 0
    )

    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(category_id, "var(--ui-text-tertiary)"),
                "detail": detail,
                "id": category_id,
                "label": label,
                "tokens": tokens,
            }
            for category_id, label, tokens, detail in categories
            if tokens > 0
        ],
        "cache": _session_cache_stats(agent),
        "compaction": _compaction_stats(agent),
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }


def _session_cache_stats(agent: Any) -> Optional[Dict[str, Any]]:
    """Session-cumulative prompt-cache stats from the live agent's counters.

    Hit rate = cache_read / (uncached input + cache_read + cache_write) over
    every API call this session has made so far. ``None`` when the session has
    no prompt traffic yet (or the provider never reports cache usage), so UIs
    can hide the row instead of showing a meaningless 0%.
    """

    def _int(name: str) -> int:
        try:
            return max(0, int(getattr(agent, name, 0) or 0))
        except (TypeError, ValueError):
            return 0

    read = _int("session_cache_read_tokens")
    write = _int("session_cache_write_tokens")
    uncached = _int("session_input_tokens")
    total_prompt = uncached + read + write
    if total_prompt <= 0:
        return None
    return {
        "read_tokens": read,
        "write_tokens": write,
        "uncached_input_tokens": uncached,
        "hit_percent": round(read / total_prompt * 100, 1),
        "api_calls": _int("session_api_calls"),
    }


def _compaction_stats(agent: Any) -> Optional[Dict[str, Any]]:
    """Compaction history + failure state for the live session.

    Events are derived from persisted state (no new columns): each compaction
    leaves a summary-handoff message in ``state.db`` (in-place mode archives
    it at the next boundary; the newest one is still active), classified via
    ``ContextCompressor.classify_summary_content``. Per-event fields:

    - ``timestamp``: epoch seconds the summary row was persisted.
    - ``summary_tokens``: rough chars/4 estimate of the summary handoff size.
    - ``messages_before``: archived rows between the previous boundary and
      this one — the turn count the compaction summarized away. ``None``
      when the transcript carries no archived rows (e.g. rotation-mode
      lineage where the parent session holds them).

    Returns ``None`` when the session has never compacted AND no failure
    cooldown is active, so UIs hide the row entirely. ``count`` prefers the
    live compressor's ``compression_count`` (correct within this process
    lifetime) over the derived event list (correct across restarts) by
    taking the max — a DB-reloaded session has count>0 but a fresh
    in-process counter.
    """
    events: List[Dict[str, Any]] = []
    failure: Optional[Dict[str, Any]] = None
    session_id = getattr(agent, "session_id", None)
    db = getattr(agent, "_session_db", None)

    if db is not None and session_id:
        try:
            from agent.context_compressor import ContextCompressor

            markers = db.get_compaction_marker_messages(session_id)
            boundaries = []
            seen_content: set = set()
            for msg in markers:
                kind = ContextCompressor.classify_summary_content(msg.get("content"))
                if kind is None:
                    continue
                # In-place compaction carries the previous summary forward into
                # each new compacted set, re-inserting an identical row at every
                # subsequent boundary. Those copies are the SAME logical
                # compaction — keep only the earliest occurrence per content.
                content_key = str(msg.get("content") or "").strip()
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                boundaries.append(msg)
            compacted_ids = db.get_compacted_message_ids(session_id)
            # One archive_and_compact write can insert MULTIPLE summary rows
            # (e.g. a standalone handoff plus an assistant-role echo) with
            # near-identical timestamps. Group boundaries written within the
            # same second into ONE logical compaction event.
            groups: List[List[Dict[str, Any]]] = []
            for msg in sorted(boundaries, key=lambda m: float(m.get("timestamp") or 0)):
                ts = float(msg.get("timestamp") or 0)
                if groups and ts - float(groups[-1][-1].get("timestamp") or 0) < 1.0:
                    groups[-1].append(msg)
                else:
                    groups.append([msg])
            prev_boundary_id = 0
            for group in groups:
                group_max_id = max(int(m.get("id") or 0) for m in group)
                archived_between = sum(
                    1
                    for mid in compacted_ids
                    if prev_boundary_id < mid < group_max_id
                    and not any(int(m.get("id") or 0) == mid for m in group)
                )
                lead = group[0]
                events.append(
                    {
                        "timestamp": float(lead.get("timestamp") or 0) or None,
                        "summary_tokens": max(
                            _chars_to_tokens(str(m.get("content") or ""))
                            for m in group
                        ),
                        "messages_before": archived_between if compacted_ids else None,
                    }
                )
                prev_boundary_id = group_max_id
        except Exception:
            events = []
        try:
            failure = db.get_compression_failure_cooldown(session_id)
            if not isinstance(failure, dict):
                failure = None
        except Exception:
            failure = None

    comp = getattr(agent, "context_compressor", None)
    try:
        live_count = max(0, int(getattr(comp, "compression_count", 0) or 0))
    except (TypeError, ValueError):
        live_count = 0
    count = max(live_count, len(events))

    if count <= 0 and failure is None:
        return None
    result: Dict[str, Any] = {"count": count, "events": events}
    if failure:
        result["failure"] = {
            "error": failure.get("error"),
            "cooldown_until": failure.get("cooldown_until"),
            "remaining_seconds": round(failure.get("remaining_seconds") or 0),
        }
    return result


# ---------------------------------------------------------------------------
# Full-text report — the actual composed context as markdown
# ---------------------------------------------------------------------------

_REPORT_TITLES = {
    "system_prompt": "System prompt",
    "tool_definitions": "Tool definitions",
    "rules": "Rules (project context)",
    "skills": "Skills index",
    "mcp": "MCP tools",
    "subagent_definitions": "Subagent definitions",
    "memory": "Memory",
    "conversation": "Conversation",
}


def _fence(text: str, lang: str = "") -> str:
    """Wrap text in a code fence long enough to not collide with its content."""
    longest = 0
    for match in re.finditer(r"`{3,}", text):
        longest = max(longest, len(match.group(0)))
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _render_tools_md(tools: Sequence[dict]) -> str:
    if not tools:
        return "_(none)_"
    out: List[str] = []
    for tool in tools:
        name = _tool_name(tool) or "(unnamed)"
        out.append(f"### {name} (~{_json_tokens(tool):,} tokens)\n")
        out.append(_fence(json.dumps(tool, ensure_ascii=False, indent=2), "json"))
        out.append("")
    return "\n".join(out).strip()


def _render_conversation_md(messages: Sequence[dict]) -> str:
    if not messages:
        return "_(empty)_"
    out: List[str] = []
    for idx, msg in enumerate(messages, 1):
        role = str(msg.get("role") or "?")
        header = f"### {idx}. {role}"
        name = msg.get("name")
        if name:
            header += f" ({name})"
        out.append(header + "\n")
        text = _flatten_content(msg.get("content"))
        if text:
            out.append(_fence(text) if role == "tool" else text)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            out.append("\nTool calls:")
            out.append(_fence(json.dumps(tool_calls, ensure_ascii=False, indent=2), "json"))
        out.append("")
    return "\n".join(out).strip()


def build_session_context_report(
    agent: Any,
    messages: Optional[List[dict]] = None,
    category: Optional[str] = None,
) -> str:
    """Render the actual composed context as a markdown document.

    ``category`` limits the report to one breakdown category id (e.g.
    ``"tool_definitions"``); ``None`` renders every section. Token figures use
    the same char/4 estimate as the breakdown, so they line up with the popover.
    """
    sections = _collect_sections(agent, messages)

    renderers = {
        "system_prompt": lambda: sections["system_prompt_text"] or "_(empty)_",
        "tool_definitions": lambda: _render_tools_md(sections["builtin_tools"]),
        "rules": lambda: sections["context"] or "_(empty)_",
        "skills": lambda: (
            _fence(sections["skills_index"]) if sections["skills_index"] else "_(empty)_"
        ),
        "mcp": lambda: _render_tools_md(sections["mcp_tools"]),
        "subagent_definitions": lambda: _render_tools_md(sections["subagent_tools"]),
        "memory": lambda: sections["memory_text"] or "_(empty)_",
        "conversation": lambda: _render_conversation_md(sections["messages"]),
    }

    if category is not None and category not in renderers:
        raise ValueError(f"unknown context category: {category!r}")

    ids = [category] if category else list(renderers)

    model = getattr(agent, "model", "") or ""
    title = _REPORT_TITLES.get(category or "", "") if category else "Full context"
    out: List[str] = [f"# Context report — {title}" + (f" ({model})" if model else ""), ""]
    for section_id in ids:
        body = renderers[section_id]()
        out.append(f"## {_REPORT_TITLES[section_id]}")
        out.append("")
        out.append(body)
        out.append("")
    return "\n".join(out).strip() + "\n"
