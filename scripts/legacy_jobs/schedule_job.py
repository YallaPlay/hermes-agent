#!/usr/bin/env python3
"""Manage legacy Slack-created job specs under jobs/slack/<user>/<slug>."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from .scheduler import parse_frontmatter, parse_duration
except ImportError:
    from scheduler import parse_frontmatter, parse_duration

REPO_DIR = Path(__file__).resolve().parents[2]
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title.strip().lower()).strip("_")
    return slug[:80] or "job"


def validate_slug(slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise ValueError("invalid slug")
    return slug


def user_root(user: str, repo: Path = REPO_DIR) -> Path:
    return repo / "jobs" / "slack" / user


def job_folder(user: str, slug: str, repo: Path = REPO_DIR) -> Path:
    return user_root(user, repo) / validate_slug(slug)


def infer_post_mode(channel: str | None, thread_ts: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if channel and thread_ts:
        return "thread"
    if channel:
        return "new_message"
    return "dm"


def validate_schedule(schedule: str | None, every: str | None) -> None:
    if not schedule and not every:
        raise ValueError("one of --schedule or --every is required")
    if every and parse_duration(every) < 60:
        raise ValueError("minimum interval is 1 minute")


def render_job_md(frontmatter: dict[str, Any], body: str) -> str:
    return "---\n" + yaml.safe_dump(frontmatter, sort_keys=False).strip() + "\n---\n\n" + body.rstrip() + "\n"


def read_job(path: Path) -> tuple[dict[str, Any], str]:
    return parse_frontmatter(path.read_text(encoding="utf-8"))


def create_job(args: argparse.Namespace, body: str, repo: Path = REPO_DIR) -> Path:
    slug = validate_slug(args.slug or slugify(args.title))
    validate_schedule(args.schedule, args.every)
    folder = job_folder(args.slack_user, slug, repo)
    folder.mkdir(parents=True, exist_ok=True)
    fm: dict[str, Any] = {"title": args.title, "created_by": args.slack_user, "post_mode": infer_post_mode(args.channel, args.thread_ts, args.post_mode)}
    if args.schedule:
        fm["schedule"] = args.schedule
    if args.every:
        fm["every"] = args.every
    if args.channel:
        fm["channel"] = args.channel
    if args.thread_ts:
        fm["thread_ts"] = args.thread_ts
    if args.expires_at:
        fm["expires_at"] = args.expires_at
    (folder / "job.md").write_text(render_job_md(fm, body), encoding="utf-8")
    return folder / "job.md"


def list_jobs(user: str, repo: Path = REPO_DIR) -> list[str]:
    rows: list[str] = []
    root = user_root(user, repo)
    if not root.exists():
        return rows
    for spec in sorted(root.glob("*/job.md")):
        fm, _ = read_job(spec)
        target = fm.get("channel") or fm.get("post_mode") or "dm"
        sched = fm.get("schedule") or fm.get("every") or ""
        rows.append("\t".join([spec.parent.name, str(sched), str(target), str(fm.get("title") or spec.parent.name)]))
    return rows


def soft_delete(folder: Path, repo: Path = REPO_DIR) -> Path:
    dest_root = repo / "jobs" / "_deleted"
    dest_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    dest = dest_root / f"{ts}-slack__{folder.parent.name}__{folder.name}"
    shutil.move(str(folder), str(dest))
    return dest


def update_job(args: argparse.Namespace, body: str | None, repo: Path = REPO_DIR) -> Path:
    folder = job_folder(args.slack_user, args.slug, repo)
    spec = folder / "job.md"
    fm, old_body = read_job(spec)
    if args.pause:
        fm["paused"] = True
    if args.resume:
        fm.pop("paused", None)
    for attr, key in [("title", "title"), ("schedule", "schedule"), ("every", "every"), ("channel", "channel"), ("thread_ts", "thread_ts"), ("post_mode", "post_mode"), ("expires_at", "expires_at")]:
        value = getattr(args, attr, None)
        if value is not None:
            fm[key] = value
    validate_schedule(fm.get("schedule"), fm.get("every"))
    spec.write_text(render_job_md(fm, body if body else old_body), encoding="utf-8")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_DIR))
    parser.add_argument("--slack-user", required=True)
    parser.add_argument("--slug")
    parser.add_argument("--title")
    parser.add_argument("--schedule")
    parser.add_argument("--every")
    parser.add_argument("--channel")
    parser.add_argument("--thread-ts")
    parser.add_argument("--post-mode", choices=["log", "dm", "new_message", "thread"])
    parser.add_argument("--expires-at")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show")
    parser.add_argument("--modify", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    try:
        if args.list:
            print("\n".join(list_jobs(args.slack_user, repo)))
            return 0
        if args.show:
            print((job_folder(args.slack_user, args.show, repo) / "job.md").read_text(encoding="utf-8"), end="")
            return 0
        if args.delete:
            if not args.slug:
                raise ValueError("--slug is required for --delete")
            print(soft_delete(job_folder(args.slack_user, args.slug, repo), repo).as_posix())
            return 0
        body = sys.stdin.read()
        if args.modify or args.pause or args.resume:
            if not args.slug:
                raise ValueError("--slug is required for modify/pause/resume")
            print(update_job(args, body if body.strip() else None, repo).as_posix())
            return 0
        if not args.title:
            raise ValueError("--title is required to create a job")
        print(create_job(args, body, repo).as_posix())
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
