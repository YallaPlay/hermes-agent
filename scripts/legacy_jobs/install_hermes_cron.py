#!/usr/bin/env python3
"""Prepare the Hermes profile script for the legacy jobs scheduler cron."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = Path("/home/ubuntu/git/yallaplay-hermes-agent")


def profile_home(profile: str | None = None, hermes_home: str | None = None) -> Path:
    if hermes_home:
        return Path(hermes_home).expanduser().resolve()
    if profile:
        return (Path.home() / ".hermes" / "profiles" / profile).resolve()
    return Path.home() / ".hermes" / "profiles" / "claudio-lab"


def wrapper_content(repo: Path, ticks: int = 2, interval_sec: float = 30.0) -> str:
    return f'''#!/usr/bin/env python3
import runpy
import sys
sys.argv = ["hermes_tick.py", "--repo", {str(repo)!r}, "--ticks", "{ticks}", "--interval-sec", "{interval_sec}"]
runpy.run_path({str(repo / "scripts" / "legacy_jobs" / "hermes_tick.py")!r}, run_name="__main__")
'''


def cron_spec(repo: Path) -> dict[str, object]:
    return {
        "name": "legacy-jobs-scheduler",
        "schedule": "every 1m",
        "script": "legacy_jobs_tick.py",
        "no_agent": True,
        "deliver": "local",
        "workdir": str(repo),
    }


def write_profile_script(home: Path, repo: Path, ticks: int = 2, interval_sec: float = 30.0) -> Path:
    target = home / "scripts" / "legacy_jobs_tick.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(wrapper_content(repo, ticks, interval_sec), encoding="utf-8")
    target.chmod(0o755)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--profile", default="claudio-lab")
    parser.add_argument("--hermes-home")
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--write-script", action="store_true")
    parser.add_argument("--create-cron", action="store_true", help="print guarded instruction only; does not create cron")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    home = profile_home(args.profile, args.hermes_home)
    target = home / "scripts" / "legacy_jobs_tick.py"
    if args.write_script:
        target = write_profile_script(home, repo, args.ticks, args.interval_sec)
    print(f"profile_home: {home}")
    print(f"script_target: {target}")
    print("cron_spec:")
    for key, value in cron_spec(repo).items():
        print(f"  {key}: {value}")
    if args.create_cron:
        print("create_cron: not performed by this helper; use the Hermes cronjob tool after human approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
