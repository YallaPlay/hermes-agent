"""Tests for the config-driven ``disabled_providers`` filter.

Providers listed under ``disabled_providers`` in config.yaml must disappear
from ``list_available_providers`` (and therefore from every picker surface
built on it), even when ambient system credentials would authenticate them
— e.g. copilot piggybacking on a logged-in ``gh`` CLI.
"""

from unittest.mock import patch

from hermes_cli import models


def _provider_ids():
    return {p["id"] for p in models.list_available_providers()}


class TestDisabledProviders:
    def test_no_config_key_keeps_all_providers(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            ids = _provider_ids()
        assert "copilot" in ids
        assert "custom" in ids

    def test_disabled_provider_is_omitted(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"disabled_providers": ["copilot"]},
        ):
            ids = _provider_ids()
        assert "copilot" not in ids
        assert "bedrock" in ids

    def test_string_value_and_aliases_normalize(self):
        # A bare string (not a list) and an alias like "github-copilot" both work.
        with patch(
            "hermes_cli.config.load_config",
            return_value={"disabled_providers": "github-copilot"},
        ):
            ids = _provider_ids()
        assert "copilot" not in ids

    def test_custom_can_be_disabled(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"disabled_providers": ["custom"]},
        ):
            ids = _provider_ids()
        assert "custom" not in ids

    def test_config_error_fails_open(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            ids = _provider_ids()
        assert "copilot" in ids
