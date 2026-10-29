"""Slack titles describe the request, not transport context or quoted history."""

import pytest

from agent.title_generator import build_title_input, derive_title, is_titleable_user_message


@pytest.mark.parametrize("prefix", [
    "[Slack runtime context]\n" + "Formatting and requester details.\n" * 100
    + "[/Slack runtime context]\n\n",
    "[Slack turn tag: abcd]\n\n",
], ids=["first-turn", "resumed-turn"])
@pytest.mark.parametrize("context", [
    "",
    "[thread context — prior messages]\nHi, I'm Claudio.\n"
    "[slack user UOLD] An old request\n[/thread context]\n",
    "[channel instructions: #backend]\nFollow the channel rules.\n[/channel instructions]\n"
    "[channel context — prior messages]\nOld discussion.\n[/channel context]\n"
    "[thread context — prior messages]\nOld reply.\n[/thread context]\n",
    "[channel instructions UPDATED: #backend]\nNew rules.\n[/channel instructions UPDATED]\n"
    "[Scheduled-job context — this thread is pinned to a job]\nJob: jobs/report\n"
    "  title: Daily report\n[thread context — prior messages]\n"
    "[slack user UOLD] An old request\n[/thread context]\n",
], ids=["no-history", "thread", "channel", "scheduled-job"])
def test_slack_request_precedes_title_budget_without_changing_history(prefix, context):
    request = "Create Thanksgiving reward overrides\nKeep the control group unchanged."
    message = prefix + context + "[slack user U123] " + request
    original = message
    assert build_title_input(message) == request
    assert derive_title(message) == "Create Thanksgiving reward overrides"
    assert is_titleable_user_message(message)
    assert message == original


@pytest.mark.parametrize("message, expected", [
    ("Explain [slack user U123] markers", "Explain [slack user U123] markers"),
    ("[Slack turn tag: abcd]\n[slack user U123] Explain this example:\n"
     "[slack user UOTHER] quoted text", "Explain this example:\n[slack user UOTHER] quoted text"),
    ("[Slack turn tag: abcd]\n[slack user U123] " + "x" * 1100, "x" * 1000),
    ("[Slack runtime context]\nOnly setup\n[/Slack runtime context]", ""),
    ("[Slack turn tag: abcd]\n[slack user U123] ", ""),
], ids=["plain-prose", "quoted-header", "budget", "setup-only", "empty-request"])
def test_slack_title_preserves_user_prose_and_skips_context_only_turns(message, expected):
    assert build_title_input(message) == expected
    assert is_titleable_user_message(message) is bool(expected)
    if not expected:
        assert derive_title(message) is None


@pytest.mark.parametrize("marker", ["[Slack runtime context]", "[Slack turn tag: abcd]"])
def test_ordinary_prose_can_start_with_a_slack_marker(marker):
    message = marker + "\nExplain what this marker means."
    assert build_title_input(message) == message
