"""Tests for the config-driven ``disabled_providers`` filter.

Providers listed under ``disabled_providers`` in config.yaml must disappear
from ``list_available_providers`` (and therefore from every picker surface
built on it), even when ambient system credentials would authenticate them
— e.g. copilot piggybacking on a logged-in ``gh`` CLI.
"""

import json
from unittest.mock import patch

import pytest

from hermes_cli import models


def _provider_ids():
    return {p["id"] for p in models.list_available_providers()}


class TestDisabledProviders:
    def test_no_config_key_keeps_all_providers(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            ids = _provider_ids()
        assert "copilot" in ids
        assert "custom" in ids

    def test_disabled_provider_is_omitted(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"disabled_providers": ["copilot"]},
        ):
            ids = _provider_ids()
        assert "copilot" not in ids
        assert "bedrock" in ids

    def test_string_value_and_aliases_normalize(self):
        # A bare string (not a list) and an alias like "github-copilot" both work.
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"disabled_providers": "github-copilot"},
        ):
            ids = _provider_ids()
        assert "copilot" not in ids

    def test_custom_can_be_disabled(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"disabled_providers": ["custom"]},
        ):
            ids = _provider_ids()
        assert "custom" not in ids

    def test_config_error_fails_open(self):
        with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
            ids = _provider_ids()
        assert "copilot" in ids


@pytest.mark.parametrize("listing", ["list_authenticated_providers", "list_picker_providers"])
@pytest.mark.parametrize("disabled,expected_hidden", [
    (["hidden"], "hidden"), ("hidden", "hidden"), (["custom"], "custom"),
    ("github-copilot", "copilot"),
])
def test_disabled_config_reaches_new_picker(listing, disabled, expected_hidden, monkeypatch, tmp_path):
    """Exercise actual row assembly through the v0.22.2 path and real config loader."""
    from hermes_cli import model_switch_providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "model": {"provider": "custom", "base_url": "http://bare.invalid/v1", "default": "bare-model"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", [])
    monkeypatch.setattr(model_switch_providers, "_build_curated_lists", lambda *a, **kw: {})
    monkeypatch.setattr(model_switch_providers, "_discover_endpoint_models", lambda *a, **kw: (None, False))
    providers = {
        name: {"api": f"http://{name}.invalid/v1", "models": ["test-model"], "discover_models": False}
        for name in ("hidden", "kept", "excluded", "copilot")
    }
    list_providers = getattr(model_switch_providers, listing)
    kwargs = dict(user_providers=providers, excluded_providers=["excluded"], probe_custom_providers=False)
    before = {row["slug"] for row in list_providers(**kwargs)}
    assert {"hidden", "kept", "custom", "copilot"} <= before
    assert "excluded" not in before

    cfg["disabled_providers"] = disabled
    config_path.write_text(json.dumps(cfg))
    after = {row["slug"] for row in list_providers(**kwargs)}
    assert after == before - {expected_hidden}
