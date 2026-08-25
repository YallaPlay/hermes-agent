"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str = "clean") -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()


def _pid_is_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.split()[2] != "Z"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group behavior")
@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["process_tree_exit", "process_tree_hang"])
async def test_client_shutdown_kills_process_group_descendants(tmp_path: Path, script: str):
    """A worker must not survive whether the LSP leader exits or hangs."""
    child_pid_file = tmp_path / "child.pid"
    client = LSPClient(
        server_id="mock-process-tree",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env={
            "MOCK_LSP_SCRIPT": script,
            "MOCK_LSP_CHILD_PID_FILE": str(child_pid_file),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        cwd=str(tmp_path),
    )
    await client.start()
    assert client._proc is not None
    leader_pid = client._proc.pid
    child_pid: int | None = None
    child_survived = True
    try:
        deadline = asyncio.get_running_loop().time() + 3.0
        while not child_pid_file.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        assert child_pid_file.exists(), "mock LSP child did not start"
        child_pid = int(child_pid_file.read_text())

        await client.shutdown()
        deadline = asyncio.get_running_loop().time() + 3.0
        while _pid_is_alive(child_pid) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        child_survived = _pid_is_alive(child_pid)
    finally:
        if child_pid is not None and _pid_is_alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if client.is_running:
            await client.shutdown()
        try:
            os.killpg(leader_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    assert not child_survived, "LSP child survived process-group cleanup"








