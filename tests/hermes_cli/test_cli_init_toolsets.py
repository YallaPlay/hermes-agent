"""Toolset validation at CLI startup must not race background plugin discovery.

The enabled-toolset list can be built from the persisted plugin-key cache while
discovery is still registering plugin toolsets, so a plugin toolset (lsp_navigate's
``development``) looked unknown and ``hermes chat -Q`` printed
``Warning: Unknown toolsets: development`` into captured stdout.
"""

from unittest.mock import patch

from cli import HermesCLI


def _run_init_toolsets(toolsets, registered, discovered_later):
    cli_obj = HermesCLI.__new__(HermesCLI)
    printed = []
    cli_obj._console_print = printed.append

    def _discover(force=False):
        registered.update(discovered_later)

    with patch("cli.validate_toolset", side_effect=lambda name: name in registered), \
         patch("hermes_cli.plugins.discover_plugins", side_effect=_discover):
        cli_obj._init_toolsets(toolsets)
    return printed


def test_plugin_toolset_registered_by_inflight_discovery_is_not_unknown():
    printed = _run_init_toolsets(["terminal", "development"], {"terminal"}, {"development"})
    assert printed == []


def test_truly_unknown_toolset_still_warns_after_discovery():
    printed = _run_init_toolsets(["terminal", "nope"], {"terminal"}, {"development"})
    assert len(printed) == 1 and "nope" in printed[0] and "terminal" not in printed[0]
