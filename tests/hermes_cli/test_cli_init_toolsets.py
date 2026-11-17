"""Explicit toolset validation must not race background plugin discovery.

The enabled-toolset list can be built from the persisted plugin-key cache while
discovery is still registering plugin toolsets, so a plugin toolset (lsp_navigate's
``development``) looked unknown and ``hermes chat -Q`` printed
``Warning: Unknown toolsets: development`` into captured stdout. The gateway now
owns this resolution (``tui_gateway.agent_factory._resolve_explicit_toolsets``).
"""

from unittest.mock import patch

from tui_gateway.agent_factory import _resolve_explicit_toolsets


def _resolve(toolsets, registered, discovered_later):
    printed = []

    def _discover(force=False):
        registered.update(discovered_later)

    with patch("tui_gateway.agent_factory._tui_notice", side_effect=printed.append), \
         patch("hermes_cli.plugins.discover_plugins", side_effect=_discover) as discover, \
         patch("hermes_cli.config.read_raw_config", return_value={}):
        resolved = _resolve_explicit_toolsets(toolsets, lambda name: name in registered)
    discover.assert_called_once()
    return resolved, printed


def test_plugin_toolset_registered_by_inflight_discovery_is_not_unknown():
    resolved, printed = _resolve(["terminal", "development"], {"terminal"}, {"development"})
    assert resolved == ["terminal", "development"]
    assert printed == []


def test_truly_unknown_toolset_still_warns_after_discovery():
    resolved, printed = _resolve(["terminal", "nope"], {"terminal"}, {"development"})
    assert resolved == ["terminal"]
    assert len(printed) == 1 and "nope" in printed[0] and "terminal" not in printed[0]
