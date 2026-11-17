"""The fork keeps ``hermes acp`` in-process unless ``acp.execution: gateway`` opts in."""

from acp_adapter.execution_mode import acp_uses_gateway
from acp_adapter.server import HermesACPAgent


def test_default_is_in_process(monkeypatch):
    monkeypatch.delenv("HERMES_ACP_EXECUTION", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert acp_uses_gateway() is False
    agent = HermesACPAgent()
    assert type(agent) is HermesACPAgent


def test_config_opts_into_gateway(monkeypatch):
    monkeypatch.delenv("HERMES_ACP_EXECUTION", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"acp": {"execution": "gateway"}})
    assert acp_uses_gateway() is True


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("HERMES_ACP_EXECUTION", "in_process")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"acp": {"execution": "gateway"}})
    assert acp_uses_gateway() is False
    monkeypatch.setenv("HERMES_ACP_EXECUTION", "gateway")
    assert acp_uses_gateway() is True


def test_unreadable_config_stays_in_process(monkeypatch):
    monkeypatch.delenv("HERMES_ACP_EXECUTION", raising=False)

    def _boom():
        raise OSError("unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert acp_uses_gateway() is False
