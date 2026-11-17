"""Shared fixtures for tests/acp_adapter.

Keeps the ACP server tests offline: ``HermesACPAgent._build_model_state``
calls ``hermes_cli.inventory.build_models_payload``, which (without this
fixture) performs live network fetches — models.dev registry, GitHub model
catalog, Copilot token exchange, Anthropic model list — adding ~3s of real
SSL/socket time to every test that creates or loads a session (~147s total
for test_server.py alone).

Tests that assert model-state behavior re-patch these same attributes with
``unittest.mock.patch`` / ``monkeypatch``; inner patches win, so this
default is transparent to them.

``daemon`` (an ordinary isolated gateway process; ``model_peer`` comes from
``tests/conftest.py``) and ``server_spec`` (a stdio MCP peer) are shared by the
real-wire ACP test files, which receive them by name.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.fixture(autouse=True)
def _offline_model_inventory(monkeypatch):
    """Stub the shared model inventory so ACP tests never hit the network."""
    from hermes_cli import inventory

    class _StubPickerContext:
        def with_overrides(self, **_kwargs):
            return self

    monkeypatch.setattr(inventory, "load_picker_context", lambda: _StubPickerContext())
    monkeypatch.setattr(
        inventory,
        "build_models_payload",
        lambda *_args, **_kwargs: {"providers": []},
    )


# Upstream modules written against the v0.23 gateway-viewer ACP contract: added in v0.23, or
# built around ``GatewayACPAgent``. Regenerate when syncing upstream:
#   git ls-tree -r --name-only <tag> tests/acp_adapter  (new files + files naming GatewayACPAgent)
_UPSTREAM_GATEWAY_CONTRACT_MODULES = frozenset({
    "test_acp_discard_review",
    "test_acp_image_staging_boundary",
    "test_acp_image_strict_reject",
    "test_acp_reattach_unknown_notice",
    "test_acp_unacked_identity_retry",
    "test_adversarial_unknown_follower",
    "test_catalog_read_only",
    "test_gateway_acp_session_end",
    "test_gateway_attach_mode_race",
    "test_gateway_attach_notice_order",
    "test_gateway_attach_shutdown",
    "test_gateway_cancel_admission",
    "test_gateway_catalog_cwd",
    "test_gateway_catalog_local_rows",
    "test_gateway_catalog_reset_owner",
    "test_gateway_client_boundary",
    "test_gateway_compressed_reopen",
    "test_gateway_config_options",
    "test_gateway_config_projection",
    "test_gateway_controls",
    "test_gateway_delivery_shapes",
    "test_gateway_editor_clarify",
    "test_gateway_editor_cwd",
    "test_gateway_editor_mcp",
    "test_gateway_editor_mcp_held",
    "test_gateway_editor_policy",
    "test_gateway_editor_v4a",
    "test_gateway_editor_wire",
    "test_gateway_entry_lifecycle",
    "test_gateway_event_mapping",
    "test_gateway_launch_flags",
    "test_gateway_model_pick_provider",
    "test_gateway_model_provider",
    "test_gateway_new_cold",
    "test_gateway_overlay_home",
    "test_gateway_prompt_content",
    "test_gateway_prompt_reoffer",
    "test_gateway_queued_unknown",
    "test_gateway_reasoning",
    "test_gateway_reconnect",
    "test_gateway_registry_commands",
    "test_gateway_remote_image_submit",
    "test_gateway_replay_gap",
    "test_gateway_retry_command",
    "test_gateway_service_tier",
    "test_gateway_service_tier_reload",
    "test_gateway_session_commands",
    "test_gateway_session_metadata",
    "test_gateway_sessions",
    "test_gateway_slash_text",
    "test_gateway_snapshot_adoption",
    "test_gateway_submit_retry",
    "test_gateway_terminal_projection",
    "test_gateway_terminal_receipt_claim",
    "test_gateway_tool_projection",
    "test_gateway_unacked_slash",
    "test_gateway_voice_stopped",
    "test_gateway_wire_controls",
    "test_named_custom_provider_route",
    "test_policy_identity_compat",
    "test_retained_identity_retirement",
    "test_session_mcp_branch_gate",
    "test_session_mcp_frozen",
    "test_session_mcp_ownership",
    "test_session_mcp_policy",
    "test_session_mcp_selection",
    "test_session_mcp_workspace_move",
})


@pytest.fixture(autouse=True)
def _acp_execution_mode(request, monkeypatch):
    """Pin ``acp_adapter.execution_mode`` so a developer's config.yaml never changes results.

    Fork default (in-process) everywhere except upstream gateway-contract modules."""
    module = request.node.path.stem if hasattr(request.node, "path") else ""
    mode = "gateway" if module in _UPSTREAM_GATEWAY_CONTRACT_MODULES else "in_process"
    monkeypatch.setenv("HERMES_ACP_EXECUTION", mode)


@pytest.fixture
def daemon(tmp_path, model_peer, request):
    """An ordinary isolated ``gateway.run`` process: ``(home, descriptor, env, root)``."""
    from tests.gateway.test_normal_runtime_boot import control

    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    root = Path(__file__).resolve().parents[2]
    model_url = f"http://127.0.0.1:{model_peer.server_port}/v1"
    config = {
        "gateway": {"multiplex_profiles": False},
        "approvals": {"mode": "manual", "timeout": 60},
        "model": {"provider": "custom", "default": "local-wire-stub", "base_url": model_url},
        "auxiliary": {"title_generation": {"enabled": False}},
        # The autouse inventory stub above cannot reach these child processes, and every ACP
        # session/new|load|fork builds its model picker there. Keep them off the internet: a cold
        # catalog fetch blocks the reply for up to 8 s per URL and models.dev for 15 s, and a
        # degraded runner network pushed session/new past the 40 s RPC bound.
        "model_catalog": {"enabled": False},
        "models_dev": {"url": model_url.removesuffix("/v1") + "/models-dev-offline"},
    }
    for key, value in (request.param.items() if isinstance(getattr(request, 'param', None), dict) else ()):
        config.setdefault(key, {}).update(value)
    (home / "config.yaml").write_text(json.dumps(config))
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED="1",
               OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=model_url,
               HERMES_ACP_SKIP_CONFIGURED_MCP="1",
               # Fork: ACP children spawned from this daemon env are gateway viewers
               # (acp_adapter.execution_mode); the fork default would run them in-process.
               HERMES_ACP_EXECUTION="gateway")
    command = [sys.executable, "-m", "gateway.run"]
    if getattr(request, 'param', None) == 'proposed-acp-descriptor':
        # API-owner handoff only: execution/create/policy stay unmodified. This
        # fixture explicitly distinguishes the proposed advert from shipped HEAD.
        command = [sys.executable, '-c', '''
from gateway.session_controls import AuthorityConnection
original = AuthorityConnection.describe
async def describe(self, ref, params):
    result = await original(self, ref, params)
    result['session_create']['sources'].append('acp')
    result['capabilities'].append('acp-editor-policy-v1')
    return result
AuthorityConnection.describe = describe
import runpy
runpy.run_module('gateway.run', run_name='__main__')
''']
    log_path = tmp_path / "gateway.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            descriptor = {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, "identify")
                    if descriptor.get("state") == "ready":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            assert descriptor.get("state") == "ready", log_path.read_text()
            yield home, descriptor, env, root
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.fixture
def server_spec(tmp_path):
    """A stdio MCP server spec whose tool echoes a borrowed credential."""
    script = tmp_path / 'owned_mcp.py'
    script.write_text('''import json, os, sys
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'owned-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':os.environ['BORROWED_CREDENTIAL']}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    return {'name': 'same-editor-name', 'command': sys.executable, 'args': [str(script)],
            'env': [{'name': 'BORROWED_CREDENTIAL', 'value': 'PRIVATE_A'}]}
