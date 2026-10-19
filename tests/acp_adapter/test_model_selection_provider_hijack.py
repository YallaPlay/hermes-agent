"""Regression tests for ACP model-selection provider routing.

Bug (2026-08-12): switching between two models of the SAME named custom
endpoint sent the request to OpenRouter while the client selector kept
showing the named endpoint. The retired ACP-local selection helper consulted
``detect_provider_for_model`` whenever the parsed provider equalled the
current provider — which is exactly what a same-provider switch looks
like — so ``custom:yallaplay-cpa:claude-opus-5`` resolved to
``("openrouter", "anthropic/claude-opus-5")``.

Detection exists to guess a provider for a BARE model name.  It must never
override a provider the user selected explicitly, and it must not fire when
the current endpoint's own catalog already serves the model.
"""

from types import SimpleNamespace

import hermes_cli.models as cli_models
import hermes_cli.model_switch as shared_switch
import pytest

from acp_adapter.server import HermesACPAgent

CPA = "custom:yallaplay-cpa"


def _resolve_on_live_shared_path(raw_model, provider):
    """Exercise ACP's real adapter and the real shared routing step, not an orphan helper.

    Credentials and model validation are deliberately excluded from these ROUTING tests;
    test_acp_dashboard_model_switch_validation.py checks their acceptance/rejection contract.
    """
    made = {}

    class Manager:
        def _make_agent(self, **kwargs):
            made.update(kwargs)
            return SimpleNamespace(model=kwargs["model"], provider=kwargs["requested_provider"])

        def save_session(self, session_id):
            pass

    agent = HermesACPAgent(session_manager=Manager())
    agent._schedule_mcp_late_refresh = lambda state: None
    state = SimpleNamespace(
        agent=SimpleNamespace(provider="custom", requested_provider=provider,
                              base_url="https://route-test.invalid/v1", api_key="test-only"),
        model="old-model", cwd="/tmp", session_id="route-test", effort="",
    )
    _old, target, resolved = agent._switch_model(state, raw_model)
    assert made["requested_provider"] == target
    assert made["model"] == resolved
    return target, resolved


@pytest.fixture(autouse=True)
def shared_route_without_live_credentials(monkeypatch):
    """Run shared custom-provider routing without network, credentials or model catalogs."""
    monkeypatch.setattr(cli_models, "_configured_custom_provider_ids", lambda: {CPA})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(shared_switch, "resolve_alias", lambda *args: None)
    monkeypatch.setattr(shared_switch, "is_aggregator", lambda provider: False)
    monkeypatch.setattr(shared_switch, "_route_configured_provider", lambda state: False)

    def _route_only(**kwargs):
        explicit = kwargs["explicit_provider"]
        state = shared_switch._Switch(
            raw_input=kwargs["raw_input"], current_provider=kwargs["current_provider"],
            current_model=kwargs["current_model"], current_base_url=kwargs["current_base_url"],
            current_api_key=kwargs["current_api_key"], is_global=False,
            explicit_provider=explicit, user_providers=None, custom_providers=None,
            new_model=kwargs["raw_input"], target_provider=explicit or kwargs["current_provider"],
        )
        if not explicit:
            failure = shared_switch._route_from_model_input(state)
            assert failure is None
        return shared_switch.ModelSwitchResult(
            success=True, new_model=state.new_model, target_provider=state.target_provider,
            base_url=state.current_base_url, api_key=state.current_api_key,
        )

    monkeypatch.setattr(shared_switch, "switch_model", _route_only)


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
        "acp_adapter.model_catalog._named_custom_provider_catalogs",
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
    provider, resolved = _resolve_on_live_shared_path(
        f"custom:yallaplay-cpa:{model}", CPA
    )

    assert (provider, resolved) == (CPA, model)
    assert detect_spy == [], "detection must not be consulted for a qualified id"


def test_bare_model_served_by_current_endpoint_stays(detect_spy, cpa_catalog):
    """Bare ``/model claude-opus-5`` while on CPA keeps CPA — it serves that id."""
    provider, resolved = _resolve_on_live_shared_path("claude-opus-5", CPA)

    assert (provider, resolved) == (CPA, "claude-opus-5")
    assert detect_spy == []


def test_bare_model_not_in_catalog_keeps_custom_route(detect_spy, cpa_catalog):
    """Upstream shared routing never silently re-detects a custom/local provider.

    This supersedes the retired ACP helper's bare-unknown fallback. Model validation
    itself remains owned by switch_model (covered by the upstream ACP/dashboard tests).
    """
    provider, resolved = _resolve_on_live_shared_path("mystery-9", CPA)

    assert (provider, resolved) == (CPA, "mystery-9")
    assert detect_spy == []


def test_explicit_cross_provider_selection_is_honored(detect_spy, cpa_catalog):
    """An explicit other-provider id switches without consulting detection."""
    provider, resolved = _resolve_on_live_shared_path(
        "openrouter:x-ai/grok-4.5", CPA
    )

    assert (provider, resolved) == ("openrouter", "x-ai/grok-4.5")
    assert detect_spy == []


def test_catalog_lookup_failure_keeps_the_selected_endpoint(detect_spy, monkeypatch):
    """A broken catalog must fail closed: stay put rather than reroute silently."""

    def _boom():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr("acp_adapter.model_catalog._named_custom_provider_catalogs", _boom)

    provider, resolved = _resolve_on_live_shared_path("claude-opus-5", CPA)

    assert (provider, resolved) == (CPA, "claude-opus-5")
    assert detect_spy == []
