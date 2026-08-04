"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import re
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm
from hermes_logging import get_session_context, run_with_session_context

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

# Validation callback: () -> bool. Called right before the LLM request in
# generate_title(). Return False to skip — e.g. the user switched models
# after this background thread captured its runtime snapshot, and sending
# the request would reload a model the runtime already evicted (#19027).
RuntimeValidator = Callable[[], bool]

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Write the title in the same language the user is writing in. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

_TITLE_PROMPT_PINNED_LANGUAGE = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Write the title in {language}. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


def _title_language() -> str:
    """Return configured title language, or empty string to match the user."""
    try:
        from hermes_cli.config import load_config_readonly

        return str(
            ((load_config_readonly() or {}).get("auxiliary") or {})
            .get("title_generation", {})
            .get("language", "")
        ).strip()
    except Exception:
        return ""


def _auto_title_enabled() -> bool:
    """Return whether automatic session title generation is enabled."""
    try:
        # Lazy imports, matching _title_language(): title_generator is imported
        # from agent code paths where a module-level hermes_cli import risks
        # circularity, and the read-only loader avoids config-migration writes.
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        config = load_config_readonly()
        title_config = (config.get("auxiliary") or {}).get("title_generation") or {}
        return is_truthy_value(title_config.get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


# --- Conversational-reply rejection -----------------------------------------
#
# The titler asks an auxiliary model for ONLY a title, but the model sometimes
# ANSWERS the exchange instead. The canonical failure (2026-07-29) is a titler
# fed text-only input for a turn that carried an image attachment: the model
# replied "I'm ready to help! However, I don't see an image attached yet. Could
# you please attach the image?" and that reply was truncated to 80 chars and
# stored as the session title.
#
# TRADEOFF: legitimate titles are short noun phrases, so the patterns below are
# anchored at the START of the candidate and are word-boundary aware. Matching a
# bare keyword anywhere in the string would destroy real titles — "Cannot
# connect to Snowflake warehouse", "Fix I/O timeout in ranking service" and
# "Sorry state of the deals config" all contain "refusal" keywords yet are
# perfectly good titles. We accept that a conversational reply which happens to
# open with a noun phrase slips through (the first-line rule above still bounds
# the damage) in exchange for near-zero false positives on real titles.

# Assistant-voice / conversational openers. Anchored at the start of the
# candidate via _CONVERSATIONAL_OPENER_RE below.
#
# NOTE on apostrophes: models emit both straight (') and curly (’) forms, so
# every contraction accepts either. The "I" contractions REQUIRE the apostrophe
# — a bare `i'?d\b` / `i'?ll\b` would also match the acronyms "ID" and "ILL",
# killing legitimate titles like "ID mapping for players".
_APOS = r"['\u2019]"
_CONVERSATIONAL_OPENERS = (
    # first person (apostrophe mandatory — see NOTE above)
    rf"i{_APOS}m\b",  # I'm ready / I'm sorry / I'm unable to
    rf"i{_APOS}ll\b",
    rf"i{_APOS}ve\b",
    rf"i{_APOS}d\b",
    r"i am\b",
    r"i will\b",
    r"i would\b",
    rf"i don{_APOS}?t\b",
    r"i do not\b",
    r"i can\b",
    rf"i can{_APOS}?t\b",
    r"i cannot\b",
    r"i could\b",
    r"i need\b",
    r"i see\b",
    r"i have\b",
    r"i notice\b",
    r"i apologize\b",
    r"i understand\b",
    rf"we{_APOS}?ll need\b",
    # willingness / acknowledgement
    r"sure\b\s*[,.!:]",
    r"certainly\b",
    r"of course\b",
    r"absolutely\b\s*[,.!:]",
    r"happy to\b",
    r"glad to\b",
    r"let me\b",
    r"got it\b\s*[,.!:]",
    r"understood\b\s*[,.!:]",
    r"no problem\b",
    r"okay\b\s*[,.!:]",
    r"ok\b\s*[,.!:]",
    r"thanks\b\s*[,.!:]",
    r"thank you\b",
    # meta / model voice
    r"as an ai\b",
    r"as a language model\b",
    # hedges and presentation
    r"it looks like\b",
    r"it seems\b",
    r"it appears\b",
    rf"here{_APOS}s\b",
    r"here is\b",
    r"here are\b",
    rf"there{_APOS}?s no\b",
    # requests back to the user
    r"could you\b",
    r"can you\b",
    r"would you\b",
    r"please provide\b",
    r"please share\b",
    r"please attach\b",
    r"please let me\b",
    r"please clarify\b",
    # apology / refusal, only in unmistakably conversational form
    r"unfortunately\b",
    r"apologies\b",
    r"my apologies\b",
    r"sorry\b\s*[,.!:]",  # "Sorry, ..." — NOT "Sorry state of the deals config"
)

_CONVERSATIONAL_OPENER_RE = re.compile(
    r"^\W*(?:" + "|".join(_CONVERSATIONAL_OPENERS) + r")",
    re.IGNORECASE,
)

# Refusal markers that only condemn the candidate when it ALSO speaks in the
# assistant's first-person voice somewhere. This catches replies that open with
# a noun phrase ("Unable to proceed — I don't see an attachment") without
# rejecting "Cannot connect to Snowflake warehouse".
_REFUSAL_MARKER_RE = re.compile(
    rf"\b(?:sorry|cannot|can{_APOS}?t|unable to|not able to|don{_APOS}?t have access)\b",
    re.IGNORECASE,
)

_ASSISTANT_VOICE_RE = re.compile(
    rf"\b(?:i{_APOS}m|i am|i{_APOS}ll|i{_APOS}ve|i{_APOS}d|i will|i would|"
    rf"i don{_APOS}?t|i do not|i can|i can{_APOS}?t|i cannot|i need|i see|"
    r"i have|as an ai)\b",
    re.IGNORECASE,
)

# A title is a phrase, not prose. Prose that is long AND sentence-punctuated AND
# addresses the user is conversational; none of those signals rejects on its own.
_SENTENCE_LENGTH_SUSPICION = 60
_ADDRESSES_USER_RE = re.compile(r"\b(?:you|your|you'?re|we|us)\b", re.IGNORECASE)


def _looks_like_conversational_reply(title: str) -> bool:
    """Return True when a candidate title is really a conversational reply.

    Applied to the cleaned candidate inside :func:`generate_title` so a model
    that answers the prompt (or refuses it, or asks a clarifying question)
    leaves the session untitled instead of storing its reply as the title.

    Conservative by construction — see the module comment above for the
    false-positive tradeoff.
    """
    if not title:
        return False
    candidate = title.strip()
    if not candidate:
        return False

    # 1. Assistant-voice / conversational opener at the START of the candidate.
    if _CONVERSATIONAL_OPENER_RE.match(candidate):
        return True

    # 2. A title never asks a question.
    if candidate.rstrip().endswith("?"):
        return True

    # 3. Refusal marker + first-person assistant voice anywhere.
    if _REFUSAL_MARKER_RE.search(candidate) and _ASSISTANT_VOICE_RE.search(candidate):
        return True

    # 4. Long, sentence-punctuated prose that addresses the user. All three
    #    signals are required together: "Snowflake cost analysis for July." is
    #    fine, and so is a long noun phrase without terminal punctuation.
    if (
        len(candidate) > _SENTENCE_LENGTH_SUSPICION
        and candidate.endswith((".", "!"))
        and _ADDRESSES_USER_RE.search(candidate)
    ):
        return True

    return False


def _summarize_user_message(user_message: str) -> str:
    """Collapse a slash-skill-expanded turn back to what the user typed.

    A ``/skill`` invocation expands into a message that embeds the whole skill
    body, so feeding it to the titler verbatim titles the session after the
    *skill's* prose — "Kick off a task in a fresh isolated git worktree" — not
    after the user's request. Reuse the canonical scaffolding parser so the
    model sees ``/work — fix the title leak`` instead.
    """
    if not user_message:
        return ""
    try:
        from agent.skill_commands import describe_skill_invocation

        described = describe_skill_invocation(user_message)
    except Exception:
        logger.debug("Skill-scaffolding summary failed; titling raw", exc_info=True)
        return user_message
    return described if described is not None else user_message


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the main runtime's model when available, falling back to the
    auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    auxiliary call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.

    ``runtime_validator`` is called right before the LLM request. If it
    returns False (e.g. the user's model was switched since the background
    thread captured its runtime snapshot), the call is skipped silently —
    no request is sent, so a stale title request can't reload a model the
    runtime already unloaded (#19027).
    """
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None

    if runtime_validator is not None:
        try:
            if not runtime_validator():
                logger.debug("Title generation skipped: runtime validator returned False")
                return None
        except Exception:
            # Fail open: a broken validator must not disable titling.
            logger.debug("Title runtime validator raised; proceeding", exc_info=True)

    # Truncate long messages to keep the request small
    user_snippet = _summarize_user_message(user_message)[:500]
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    language = _title_language()
    prompt = _TITLE_PROMPT_PINNED_LANGUAGE.format(language=language) if language else _TITLE_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=timeout,
            main_runtime=main_runtime,
        )
        content = response.choices[0].message.content or ""
        # Strip thinking/reasoning blocks that think-enabled models
        # (MiniMax M2.7, DeepSeek, etc.) emit even for simple prompts like
        # title generation. Without this the raw <think>...</think> XML
        # leaks into session titles. Reuses the canonical scrubber so all
        # tag variants (unterminated blocks, orphan closes, mixed case)
        # are handled, not just a single literal <think> pair.
        from agent.agent_runtime_helpers import strip_think_blocks
        title = strip_think_blocks(None, content).strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # A title is one line. A model that ignores "return ONLY the title" and
        # answers the prompt instead (a shell transcript, a bulleted plan) would
        # otherwise be stored verbatim and truncated mid-command. Keep the first
        # non-empty line — the closest thing to a title in that response.
        title = next((line.strip() for line in title.splitlines() if line.strip()), "")
        # A single-line conversational reply survives the first-line rule above
        # ("I'm ready to help! However, I don't see an image attached yet...")
        # and would then be truncated to 80 chars and stored verbatim. Reject it
        # so the session stays untitled and the caller's fallback applies.
        if _looks_like_conversational_reply(title):
            logger.debug(
                "Rejecting conversational reply as session title: %r", title
            )
            return None
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        # Log at WARNING so this shows up in agent.log without debug mode.
        # Full detail at debug level for operators who need the stack.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)
        return None


def _persist_session_title(session_db, session_id, title):
    """Persist a generated title, recovering from duplicate-title collisions.

    The write goes through ``set_auto_title_if_empty`` (predicate + write in
    one transaction) so a manual ``/title`` set while LLM generation was in
    flight is never overwritten — a plain ``set_session_title`` fallback keeps
    older stores working. ``set_session_title`` raises ValueError when the
    title would collide with another session (the unique-title index). Rather
    than swallow it and leave the session untitled (#50537), append a #N
    suffix via get_next_title_in_lineage() when the store supports lineage
    dedup; otherwise re-raise so the caller can decide.

    Returns the title actually persisted, or None when a concurrent manual
    title won the race (nothing was written).
    """
    atomic_fn = getattr(session_db, "set_auto_title_if_empty", None)

    def _set(t):
        if atomic_fn is not None:
            if not atomic_fn(session_id, t):
                # Predicate failed: a title appeared while generation was in
                # flight (manual /title wins), or the session vanished.
                logger.debug(
                    "Skipping auto-generated session title because a title "
                    "was set while generation was in flight"
                )
                return None
            return t
        ok = session_db.set_session_title(session_id, t)
        if ok is False:
            raise RuntimeError(
                f"session {session_id} not found when storing title"
            )
        return t

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    - runtime_validator returns False (model was switched)

    Never lets an exception escape: this is a daemon-thread target, and an
    escaping exception would spray a raw traceback into the user's terminal
    via the default threading excepthook. The canonical trigger is the
    post-``hermes update`` stale-module window, where this function's lazy
    imports read NEW source from disk while already-cached modules
    (``agent.portal_tags`` etc.) are still the OLD version — the resulting
    ImportError repeats on every auto-title attempt until the long-running
    process restarts.
    """
    try:
        _auto_title_session(
            session_db,
            session_id,
            user_message,
            assistant_response,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
            title_callback=title_callback,
            runtime_validator=runtime_validator,
        )
    except Exception as e:
        # WARNING (not debug) so operators see it in agent.log; the message
        # names the likely cause so "restart the process" is discoverable.
        logger.warning(
            "Auto-title failed (harmless; if this started after an update, "
            "restart the running Hermes process): %s",
            e,
        )
        logger.debug("Auto-title traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Auto-title failure_callback raised", exc_info=True)


def _auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Body of :func:`auto_title_session` — see its docstring."""
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    # This runs on a bare daemon thread spawned AFTER the turn's ambient
    # conversation context was reset, so publish it here from the session id
    # we already hold — the title-generation LLM call then carries the same
    # ``conversation=`` Portal tag as the turn it titles. Root-of-lineage for
    # consistency with the agent loop (a no-op on first exchange, where
    # titling happens, but correct if this ever runs on a continuation).
    from agent.aux_accounting import set_accounting_context
    from agent.portal_tags import set_conversation_context

    conversation_id = session_id
    try:
        conversation_id = session_db.get_conversation_root(session_id) or session_id
    except Exception:
        pass
    set_conversation_context(conversation_id)
    # Same for the accounting context, so the title call's token usage is
    # recorded against this session (task='title_generation', #23270).
    set_accounting_context(session_db, session_id)

    title = generate_title(
        user_message,
        assistant_response,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
        runtime_validator=runtime_validator,
    )
    if not title:
        return

    try:
        persisted = _persist_session_title(session_db, session_id, title)
        if persisted is None:
            return
        logger.debug("Auto-generated session title: %s", persisted)
        if title_callback is not None:
            try:
                title_callback(persisted)
            except Exception:
                logger.debug("Auto-title callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Only generates a title when:
    - This appears to be the first user→assistant exchange
    - No title is already set
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    # Count user messages in history to detect first exchange.
    # conversation_history includes the exchange that just happened,
    # so for a first exchange we expect exactly 1 user message
    # (or 2 counting system). Be generous: generate on first 2 exchanges.
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    if user_msg_count > 2:
        return

    # Config read comes after the cheap first-exchange guard so the file
    # isn't touched on every subsequent turn of a long session.
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return

    # Propagate the session context into the auto-title worker thread so its
    # log records (agent.title_generator, agent.auxiliary_client) carry the
    # [session_id] tag — threading.local does not inherit across threads (see
    # hermes_logging.run_with_session_context).
    _sid = get_session_context() or session_id
    thread = threading.Thread(
        target=lambda: run_with_session_context(
            _sid,
            auto_title_session,
            session_db,
            session_id,
            user_message,
            assistant_response,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
            title_callback=title_callback,
            runtime_validator=runtime_validator,
        ),
        daemon=True,
        name="auto-title",
    )
    thread.start()
