#!/usr/bin/env python3
"""Hermes no-agent profile script wrapper for the legacy jobs scheduler."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def run_ticks(repo: Path, ticks: int = 2, interval_sec: float = 30.0, *, sleep=time.sleep) -> list[str]:
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.legacy_jobs import scheduler

    alerts: list[str] = []
    started = time.monotonic()
    for i in range(ticks):
        alerts.extend(scheduler.tick(repo))
        if i < ticks - 1:
            target = started + (i + 1) * interval_sec
            delay = target - time.monotonic()
            if delay > 0:
                sleep(delay)
    return alerts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/home/ubuntu/git/yallaplay-hermes-agent")
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="run one tick")
    args = parser.parse_args(argv)
    ticks = 1 if args.once else args.ticks
    try:
        alerts = run_ticks(Path(args.repo), ticks, args.interval_sec)
    except Exception as exc:
        print(f"legacy jobs scheduler failed: {exc}")
        return 1
    if alerts:
        print("\n".join(alerts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
