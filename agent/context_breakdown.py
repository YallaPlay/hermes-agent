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
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

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


def _memory_detail(memory_block: str, user_block: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if memory_block:
        rows.append({"label": "Agent memory", "tokens": _chars_to_tokens(memory_block)})
    if user_block:
        rows.append({"label": "User profile", "tokens": _chars_to_tokens(user_block)})
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
            _memory_detail(sections["memory_block"], sections["user_block"]),
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
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }


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
