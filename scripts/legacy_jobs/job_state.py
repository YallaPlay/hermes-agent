#!/usr/bin/env python3
"""Persist a small per-job state blob across legacy scheduler runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
JOBS_DIR = (REPO_DIR / "jobs").resolve()
MAX_STATE_BYTES = 8 * 1024
STATE_FILENAME = "state.txt"


class JobPathError(ValueError):
    pass


def resolve_job_folder(job_path: str | Path, repo_dir: Path = REPO_DIR) -> Path:
    repo_dir = repo_dir.resolve()
    jobs_dir = (repo_dir / "jobs").resolve()
    raw = str(job_path).rstrip("/")
    if not raw:
        raise JobPathError("empty job path")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_dir / path
    resolved = path.resolve(strict=False)
    if resolved.name == "job.md":
        resolved = resolved.parent
    try:
        resolved.relative_to(jobs_dir)
    except ValueError as exc:
        raise JobPathError(f"job path {job_path!r} is outside jobs/") from exc
    return resolved


def state_file(job_path: str | Path, repo_dir: Path = REPO_DIR) -> Path:
    return resolve_job_folder(job_path, repo_dir) / STATE_FILENAME


def read_state(job_path: str | Path, repo_dir: Path = REPO_DIR) -> str:
    path = state_file(job_path, repo_dir)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def truncate_state(text: str) -> tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= MAX_STATE_BYTES:
        return text, False
    return data[-MAX_STATE_BYTES:].decode("utf-8", errors="ignore"), True


def write_state(job_path: str | Path, text: str, repo_dir: Path = REPO_DIR) -> bool:
    path = state_file(job_path, repo_dir)
    text, truncated = truncate_state(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return truncated


def clear_state(job_path: str | Path, repo_dir: Path = REPO_DIR) -> None:
    path = state_file(job_path, repo_dir)
    if path.exists():
        path.unlink()


def _die(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--get", dest="op", action="store_const", const="get")
    group.add_argument("--set", dest="op", action="store_const", const="set")
    group.add_argument("--clear", dest="op", action="store_const", const="clear")
    group.add_argument("--path", dest="op", action="store_const", const="path")
    parser.add_argument("job_path", metavar="JOB_PATH")
    args = parser.parse_args(argv)
    try:
        if args.op == "get":
            sys.stdout.write(read_state(args.job_path))
        elif args.op == "set":
            truncated = write_state(args.job_path, sys.stdin.read())
            if truncated:
                print(f"warning: state truncated to {MAX_STATE_BYTES} bytes", file=sys.stderr)
        elif args.op == "clear":
            clear_state(args.job_path)
        elif args.op == "path":
            print(state_file(args.job_path).as_posix())
    except (OSError, JobPathError) as exc:
        return _die(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
