"""Tests for named user-defined provider catalogs in the ACP model selector.

Named endpoints from the ``providers:`` mapping (and legacy
``custom_providers:`` list) are invisible to ``list_available_providers()``,
so ``_authenticated_provider_catalogs`` must append them explicitly for the
ACP model selector to offer them (the TUI ``/model`` picker already does).
"""

from unittest.mock import patch

import pytest

import acp_adapter.server as server_mod
from acp_adapter.server import (
    _authenticated_provider_catalogs,
    _named_custom_provider_catalogs,
)


MANTLE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    """The module-level catalog cache must not leak between tests."""
    server_mod._provider_catalogs_cache = None
    yield
    server_mod._provider_catalogs_cache = None


class TestNamedCustomProviderCatalogs:
    def test_declared_default_model_survives_failed_discovery(self, monkeypatch):
        """Endpoints without a /models route (Bedrock Mantle 404s) keep declared models."""
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "test-key")
        cfg = _cfg(
            providers={
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": MANTLE_URL,
                    "key_env": "BEDROCK_MANTLE_API_KEY",
                    "api_mode": "codex_responses",
                    "default_model": "openai.gpt-5.5",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            (
                "custom:bedrock-mantle",
                "AWS Bedrock Mantle",
                [("openai.gpt-5.5", "")],
            )
        ]

    def test_live_discovery_extends_declared_models(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "model-a",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["model-a", "model-b"],
        ):
            catalogs = _named_custom_provider_catalogs()

        assert len(catalogs) == 1
        slug, label, models = catalogs[0]
        assert slug == "custom:relay"
        assert [m for m, _ in models] == ["model-a", "model-b"]

    def test_declared_models_dict_included(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "model-a",
                    "models": {"model-b": {}, "model-c": {}},
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert [m for m, _ in catalogs[0][2]] == ["model-a", "model-b", "model-c"]

    def test_disabled_provider_skipped(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            providers={
                "off": {
                    "name": "Disabled Endpoint",
                    "base_url": "https://off.example/v1",
                    "key_env": "SOME_KEY",
                    "default_model": "m",
                    "enabled": False,
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_no_credential_and_no_declared_models_skipped(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = _cfg(
            providers={
                "bare": {
                    "name": "Bare",
                    "base_url": "https://bare.example/v1",
                    "key_env": "MISSING_KEY",
                }
            }
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            assert _named_custom_provider_catalogs() == []

    def test_legacy_custom_providers_list_included(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "k")
        cfg = _cfg(
            custom_providers=[
                {
                    "name": "Legacy Endpoint",
                    "base_url": "https://legacy.example/v1",
                    "key_env": "SOME_KEY",
                    "model": "legacy-model",
                }
            ]
        )
        with patch("hermes_cli.config.load_config", return_value=cfg), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _named_custom_provider_catalogs()

        assert catalogs == [
            ("custom:legacy-endpoint", "Legacy Endpoint", [("legacy-model", "")])
        ]


class TestAuthenticatedProviderCatalogsNamedEntries:
    def test_named_providers_appended_to_selector_catalogs(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "test-key")
        cfg = _cfg(
            providers={
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": MANTLE_URL,
                    "key_env": "BEDROCK_MANTLE_API_KEY",
                    "default_model": "openai.gpt-5.5",
                }
            }
        )
        with patch(
            "hermes_cli.models.list_available_providers",
            return_value=[
                {"id": "bedrock", "label": "AWS Bedrock", "authenticated": True}
            ],
        ), patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("global.anthropic.claude-fable-5", "")],
        ), patch(
            "hermes_cli.config.load_config", return_value=cfg
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _authenticated_provider_catalogs("bedrock")

        slugs = [slug for slug, _, _ in catalogs]
        assert slugs == ["bedrock", "custom:bedrock-mantle"]
        named = dict((s, m) for s, _, m in catalogs)["custom:bedrock-mantle"]
        assert named == [("openai.gpt-5.5", "")]

    def test_named_provider_current_sorts_first(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "test-key")
        cfg = _cfg(
            providers={
                "bedrock-mantle": {
                    "name": "AWS Bedrock Mantle",
                    "base_url": MANTLE_URL,
                    "key_env": "BEDROCK_MANTLE_API_KEY",
                    "default_model": "openai.gpt-5.5",
                }
            }
        )
        with patch(
            "hermes_cli.models.list_available_providers",
            return_value=[
                {"id": "bedrock", "label": "AWS Bedrock", "authenticated": True}
            ],
        ), patch(
            "hermes_cli.models.curated_models_for_provider",
            return_value=[("global.anthropic.claude-fable-5", "")],
        ), patch(
            "hermes_cli.config.load_config", return_value=cfg
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=None
        ):
            catalogs = _authenticated_provider_catalogs("custom:bedrock-mantle")

        assert catalogs[0][0] == "custom:bedrock-mantle"

    def test_selector_choice_id_round_trips_through_parse_model_input(self):
        """The encoded choice id must resolve back to the named provider."""
        from hermes_cli.models import parse_model_input

        choice_id = "custom:bedrock-mantle:openai.gpt-5.5"
        provider, model = parse_model_input(choice_id, "bedrock")
        assert provider == "custom:bedrock-mantle"
        assert model == "openai.gpt-5.5"
