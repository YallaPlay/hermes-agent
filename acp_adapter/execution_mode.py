"""Fork switch: does ``hermes acp`` run sessions in-process or view the profile gateway?

Upstream v0.23 made every ACP process a viewer of the profile gateway. The fork keeps the
in-process adapter by default because its extension methods (account usage, context reports,
archive/owner/title edits) and session listing have no gateway equivalent yet.
``acp.execution: gateway`` in config.yaml (or ``HERMES_ACP_EXECUTION=gateway``) opts in.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_VAR = "HERMES_ACP_EXECUTION"


def acp_uses_gateway() -> bool:
    """True when ACP should attach to the gateway; any other value keeps the in-process adapter."""
    mode = os.environ.get(ENV_VAR, "").strip().lower()
    if not mode:
        try:
            from hermes_cli.config import load_config_readonly

            acp_cfg = load_config_readonly().get("acp")
            if isinstance(acp_cfg, dict):
                mode = str(acp_cfg.get("execution") or "").strip().lower()
        except Exception:
            logger.debug("Could not read acp.execution; using the in-process adapter", exc_info=True)
            mode = ""
    return mode == "gateway"
