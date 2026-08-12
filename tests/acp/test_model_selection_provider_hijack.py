"""Regression tests for ACP model-selection provider routing.

Bug (2026-08-12): switching between two models of the SAME named custom
endpoint sent the request to OpenRouter while the client selector kept
showing the named endpoint.  ``_resolve_model_selection`` consulted
``detect_provider_for_model`` whenever the parsed provider equalled the
current provider — which is exactly what a same-provider switch looks
like — so ``custom:yallaplay-cpa:claude-opus-5`` resolved to
``("openrouter", "anthropic/claude-opus-5")``.

Detection exists to guess a provider for a BARE model name.  It must never
override a provider the user selected explicitly, and it must not fire when
the current endpoint's own catalog already serves the model.
"""

import hermes_cli.models as cli_models
import pytest

from acp_adapter.server import HermesACPAgent

CPA = "custom:yallaplay-cpa"


@pytest.fixture
def detect_spy(monkeypatch):
    """Stub detection so any consultation is visible and deterministic."""
    calls: list[tuple[str, str | None]] = []

    def _fake_detect(model, current_provider=None, *args, **kwargs):
        calls.append((model, current_provider))
        return ("openrouter", f"anthropic/{model}")

    monkeypatch.setattr(cli_models, "detect_provider_for_model", _fake_detect)
    return calls


@pytest.fixture
def cpa_catalog(monkeypatch):
    """Pretend the named CPA endpoint serves the Claude + GPT ids it really does."""
    monkeypatch.setattr(
        "acp_adapter.server._named_custom_provider_catalogs",
        lambda: [
            (
                CPA,
                "YallaPlay CPA",
                [("claude-opus-5", ""), ("claude-fable-5", ""), ("gpt-5.6-sol", "")],
            )
        ],
    )


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-fable-5", "gpt-5.6-sol"],
)
def test_qualified_same_provider_switch_stays_on_named_endpoint(
    detect_spy, cpa_catalog, model
):
    """A picker-encoded id must route to the endpoint it names, not OpenRouter."""
    provider, resolved = HermesACPAgent._resolve_model_selection(
        f"custom:yallaplay-cpa:{model}", CPA
    )

    assert (provider, resolved) == (CPA, model)
    assert detect_spy == [], "detection must not be consulted for a qualified id"


def test_bare_model_served_by_current_endpoint_stays(detect_spy, cpa_catalog):
    """Bare ``/model claude-opus-5`` while on CPA keeps CPA — it serves that id."""
    provider, resolved = HermesACPAgent._resolve_model_selection("claude-opus-5", CPA)

    assert (provider, resolved) == (CPA, "claude-opus-5")
    assert detect_spy == []


def test_bare_model_not_in_catalog_still_detects(detect_spy, cpa_catalog):
    """The bare-name affordance survives: unknown ids may still move provider."""
    provider, resolved = HermesACPAgent._resolve_model_selection("mystery-9", CPA)

    assert (provider, resolved) == ("openrouter", "anthropic/mystery-9")
    assert detect_spy == [("mystery-9", CPA)]


def test_explicit_cross_provider_selection_is_honored(detect_spy, cpa_catalog):
    """An explicit other-provider id switches without consulting detection."""
    provider, resolved = HermesACPAgent._resolve_model_selection(
        "openrouter:x-ai/grok-4.5", CPA
    )

    assert (provider, resolved) == ("openrouter", "x-ai/grok-4.5")
    assert detect_spy == []


def test_catalog_lookup_failure_keeps_the_selected_endpoint(detect_spy, monkeypatch):
    """A broken catalog must fail closed: stay put rather than reroute silently."""

    def _boom():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr("acp_adapter.server._named_custom_provider_catalogs", _boom)

    provider, resolved = HermesACPAgent._resolve_model_selection("claude-opus-5", CPA)

    assert (provider, resolved) == (CPA, "claude-opus-5")
    assert detect_spy == []
