"""Tests for GitHub Copilot entries shown in the /model picker."""

import os
from unittest.mock import MagicMock, patch

from hermes_cli.model_switch import (
    _credential_pool_is_usable,
    list_authenticated_providers,
)


def test_copilot_picker_checks_persisted_pool_without_refreshing_sources():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.has_available.return_value = True

    with patch("agent.credential_pool.load_pool", return_value=pool) as load_pool:
        assert _credential_pool_is_usable("copilot") is True

    load_pool.assert_called_once_with("copilot", refresh_sources=False)


def test_copilot_picker_uses_raw_token_presence_when_pool_is_empty():
    pool = MagicMock()
    pool.has_credentials.return_value = False

    with (
        patch("agent.credential_pool.load_pool", return_value=pool),
        patch(
            "hermes_cli.copilot_auth.resolve_copilot_token",
            return_value=("raw-token", "gh auth token"),
        ),
        patch("hermes_cli.auth.is_source_suppressed", return_value=False),
        patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            side_effect=AssertionError("picker auth checks must not exchange tokens"),
        ),
    ):
        assert _credential_pool_is_usable("copilot") is True


def test_load_pool_can_skip_external_source_refresh():
    with (
        patch("agent.credential_pool.read_credential_pool", return_value=[]),
        patch("agent.credential_pool._seed_from_singletons") as seed_singletons,
        patch("agent.credential_pool._seed_from_env") as seed_env,
    ):
        from agent.credential_pool import load_pool

        pool = load_pool("copilot", refresh_sources=False)

    assert pool.has_credentials() is False
    seed_singletons.assert_not_called()
    seed_env.assert_not_called()


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_uses_live_catalog_when_available():
    live_models = ["gpt-5.4", "claude-sonnet-4.6", "gemini-3.1-pro-preview"]

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=live_models):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)

    assert copilot is not None
    assert copilot["models"] == live_models
    assert copilot["total_models"] == len(live_models)
