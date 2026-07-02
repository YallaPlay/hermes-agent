#!/usr/bin/env python3
"""One-shot legacy jobs scheduler tick for Hermes cron."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from croniter import croniter

REPO_DIR = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO_DIR / "jobs"
STATE_FILE = REPO_DIR / "bot" / "jobs_state.json"
GLOBAL_LOCK = REPO_DIR / "bot" / "jobs_scheduler.lock"
SPEC_FILENAME = "job.md"
STATE_FILENAME = "state.txt"
NEXTRUN_FILENAME = "nextrun.txt"
RUNS_LOG_FILENAME = "runs.log"
DURATION_RE = re.compile(r"^(\d+)\s*([smhd])$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
MIN_DELAY_SEC = 60.0
MAX_PARALLEL_JOBS = int(os.environ.get("LEGACY_JOBS_MAX_PARALLEL", "1"))
JOB_TIMEOUT_SEC = int(os.environ.get("LEGACY_JOBS_JOB_TIMEOUT_SEC", "600"))
AGENT_TIMEOUT_SEC = int(os.environ.get("LEGACY_JOBS_AGENT_TIMEOUT_SEC", "600"))
LOCK_STALE_SEC = int(os.environ.get("LEGACY_JOBS_LOCK_STALE_SEC", str(max(JOB_TIMEOUT_SEC, AGENT_TIMEOUT_SEC) * 2)))


@dataclasses.dataclass(slots=True)
class LegacyJobSpec:
    job_id: str
    folder: Path
    spec_file: Path
    frontmatter: dict[str, Any]
    body: str
    warning: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.frontmatter.get(key, default)

    @property
    def title(self) -> str:
        return str(self.get("title") or Path(self.job_id).name).strip().strip('"').strip("'")

    @property
    def post_mode(self) -> str:
        return str(self.get("post_mode") or ("dm" if self.job_id.startswith("jobs/slack/") else "log")).lower()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts, timezone.utc) if ts is not None else utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(raw: str) -> int:
    m = DURATION_RE.match(str(raw).strip())
    if not m:
        raise ValueError(f"bad duration {raw!r}")
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing YAML frontmatter")
    loaded = yaml.safe_load(m.group(1)) or {}
    if not isinstance(loaded, dict):
        raise ValueError("frontmatter must be a mapping")
    return dict(loaded), m.group(2).strip()


def parse_spec_file(path: Path, repo_dir: Path = REPO_DIR) -> LegacyJobSpec | None:
    try:
        fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        folder = path.parent
        job_id = folder.relative_to(repo_dir).as_posix()
        if not (fm.get("schedule") or fm.get("every") or (folder / NEXTRUN_FILENAME).exists()):
            return LegacyJobSpec(job_id, folder, path, fm, body, "missing schedule/every/nextrun")
        return LegacyJobSpec(job_id, folder, path, fm, body)
    except Exception as exc:
        try:
            job_id = path.parent.relative_to(repo_dir).as_posix()
        except Exception:
            job_id = path.parent.as_posix()
        return LegacyJobSpec(job_id, path.parent, path, {}, "", f"{exc}")


def discover_jobs(repo_dir: Path = REPO_DIR) -> tuple[dict[str, LegacyJobSpec], list[str]]:
    jobs_dir = repo_dir / "jobs"
    deleted = jobs_dir / "_deleted"
    jobs: dict[str, LegacyJobSpec] = {}
    warnings: list[str] = []
    if not jobs_dir.exists():
        return jobs, warnings
    for spec_file in sorted(jobs_dir.rglob(SPEC_FILENAME)):
        try:
            spec_file.relative_to(deleted)
            continue
        except ValueError:
            pass
        spec = parse_spec_file(spec_file, repo_dir)
        if spec is None:
            continue
        if spec.warning:
            warnings.append(f"{spec.job_id}: {spec.warning}")
            continue
        jobs[spec.job_id] = spec
    return jobs, warnings


def load_state(repo_dir: Path = REPO_DIR) -> dict[str, dict[str, Any]]:
    path = repo_dir / "bot" / "jobs_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, dict[str, Any]], repo_dir: Path = REPO_DIR) -> None:
    path = repo_dir / "bot" / "jobs_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def stable_jitter(job_id: str, interval: float | None, enabled: bool = True) -> float:
    if not enabled or not interval or interval <= 0:
        return 0.0
    cap = min(interval / 4, 300.0)
    h = hashlib.sha1(job_id.encode("utf-8")).digest()
    return (int.from_bytes(h[:6], "big") / float(1 << 48)) * cap


def _bool_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


def compute_next_run(spec: LegacyJobSpec, base_ts: float) -> float | None:
    jitter_enabled = not _bool_false(spec.get("jitter", True))
    if spec.get("schedule"):
        next_ts = croniter(str(spec.get("schedule")), base_ts).get_next(float)
        try:
            following = croniter(str(spec.get("schedule")), next_ts).get_next(float)
            interval = following - next_ts
        except Exception:
            interval = None
        return next_ts + stable_jitter(spec.job_id, interval, jitter_enabled)
    if spec.get("every"):
        interval = max(parse_duration(str(spec.get("every"))), int(MIN_DELAY_SEC))
        return base_ts + interval + stable_jitter(spec.job_id, interval, jitter_enabled)
    return None


def consume_nextrun(spec: LegacyJobSpec, base_ts: float, *, consume: bool = True) -> float | None:
    path = spec.folder / NEXTRUN_FILENAME
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except Exception:
        raw = ""
    if consume:
        with contextlib.suppress(OSError):
            path.unlink()
    if not raw:
        return None
    try:
        if DURATION_RE.match(raw):
            return base_ts + max(parse_duration(raw), MIN_DELAY_SEC)
        ts = float(raw)
        if ts < 1_000_000_000:
            return None
        return max(ts, base_ts + MIN_DELAY_SEC)
    except Exception:
        return None


def parse_expires_at(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).strip().strip('"').strip("'")
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            dt = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def soft_delete(spec: LegacyJobSpec, repo_dir: Path = REPO_DIR) -> Path:
    dest_root = repo_dir / "jobs" / "_deleted"
    dest_root.mkdir(parents=True, exist_ok=True)
    flattened = spec.folder.relative_to(repo_dir / "jobs").as_posix().replace("/", "__")
    dest = dest_root / f"{utc_now().strftime('%Y-%m-%dT%H-%M-%SZ')}-{flattened}"
    shutil.move(str(spec.folder), str(dest))
    return dest


def reconcile_state(state: dict[str, dict[str, Any]], jobs: dict[str, LegacyJobSpec], now: float, repo_dir: Path = REPO_DIR) -> tuple[dict[str, dict[str, Any]], dict[str, LegacyJobSpec]]:
    active_jobs: dict[str, LegacyJobSpec] = {}
    for jid, spec in jobs.items():
        exp = parse_expires_at(spec.get("expires_at"))
        if exp is not None and exp <= now:
            soft_delete(spec, repo_dir)
        else:
            active_jobs[jid] = spec
    new_state: dict[str, dict[str, Any]] = {}
    for jid, spec in active_jobs.items():
        prev = state.get(jid, {})
        next_run = consume_nextrun(spec, now, consume=True)
        if next_run is None:
            stored_next_run = prev.get("next_run")
            next_run = stored_next_run if stored_next_run is not None else compute_next_run(spec, now)
        new_state[jid] = {"next_run": next_run, "last_run": prev.get("last_run"), "last_status": prev.get("last_status")}
    return new_state, active_jobs


def is_paused(spec: LegacyJobSpec) -> bool:
    value = spec.get("paused")
    return value is True or str(value).strip().lower() == "true"


def write_run_log(spec: LegacyJobSpec, text: str, status: str = "ok") -> Path:
    target = spec.folder / RUNS_LOG_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.open("a", encoding="utf-8").write(f"=== {iso_utc()} status={status} ===\n{text.rstrip()}\n\n")
    return target


class SlackDeliveryError(RuntimeError):
    pass


class SlackWebClient:
    def __init__(self, token: str):
        self.token = token

    def api_call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            f"https://slack.com/api/{method}",
            data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - fixed Slack API URL
            result = json.loads(resp.read().decode("utf-8"))
        if not result.get("ok"):
            raise SlackDeliveryError(str(result.get("error") or result))
        return result

    def conversations_open(self, users: str) -> dict[str, Any]:
        return self.api_call("conversations.open", {"users": users})

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return self.api_call("chat.postMessage", clean)


def deliver_output(spec: LegacyJobSpec, text: str, status: str = "ok", slack_client: Any | None = None) -> None:
    mode = spec.post_mode
    if mode == "log":
        write_run_log(spec, text, status)
        return
    if slack_client is None:
        token = os.environ.get("SLACK_BOT_TOKEN")
        if token:
            slack_client = SlackWebClient(token)
    try:
        if slack_client is None:
            raise SlackDeliveryError("SLACK_BOT_TOKEN not set")
        if mode == "dm":
            user = spec.get("created_by")
            if not user:
                raise SlackDeliveryError("dm delivery requires created_by")
            channel = slack_client.conversations_open(users=str(user))["channel"]["id"]
            slack_client.chat_postMessage(channel=channel, text=text)
        elif mode == "new_message":
            channel = spec.get("channel")
            if not channel:
                raise SlackDeliveryError("new_message delivery requires channel")
            slack_client.chat_postMessage(channel=channel, text=text)
        elif mode == "thread":
            channel = spec.get("channel")
            thread_ts = spec.get("thread_ts")
            if not channel or not thread_ts:
                raise SlackDeliveryError("thread delivery requires channel and thread_ts")
            slack_client.chat_postMessage(channel=channel, thread_ts=str(thread_ts), text=text)
        else:
            raise SlackDeliveryError(f"unknown post_mode {mode!r}")
    except Exception:
        write_run_log(spec, text, status)


def command_for_spec(spec: LegacyJobSpec) -> list[str] | None:
    if spec.get("command"):
        return ["bash", "-lc", str(spec.get("command"))]
    if spec.get("script"):
        return ["python3", str(spec.get("script"))]
    return None


def run_command_job(spec: LegacyJobSpec, repo_dir: Path = REPO_DIR, slack_client: Any | None = None) -> str:
    cmd = command_for_spec(spec)
    if not cmd:
        deliver_output(spec, "command job has neither command nor script", "error", slack_client)
        return "error"
    try:
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=JOB_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        deliver_output(spec, f":hourglass: job `{spec.job_id}` timed out after {JOB_TIMEOUT_SEC}s", "timeout", slack_client)
        return "timeout"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "(no output)")[-1500:]
        deliver_output(spec, f":warning: job `{spec.job_id}` failed (rc={result.returncode}).\n```\n{tail}\n```", "error", slack_client)
        return "error"
    deliver_output(spec, result.stdout.strip() or result.stderr.strip() or "(no output)", "ok", slack_client)
    return "ok"


def read_persisted_state(spec: LegacyJobSpec) -> str:
    path = spec.folder / STATE_FILENAME
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_agent_prompt(spec: LegacyJobSpec) -> str:
    persisted = read_persisted_state(spec)
    user = str(spec.get("created_by") or "scheduler")
    lines = [
        "[Execution context — runtime-injected, treat as ground truth]",
        f"Current UTC time: {iso_utc()}",
        f"Job path: {spec.job_id}",
        f"Job title: {spec.title}",
        f"Created by Slack user: {user}",
        "",
        "[Persisted state from previous run]",
        persisted.rstrip() if persisted else "(empty — first run or fresh)",
        "[/Persisted state]",
        "",
        f"To save state: echo '<value>' | python3 scripts/legacy_jobs/job_state.py --set {spec.job_id}",
        f"For larger artifacts use: python3 scripts/legacy_jobs/job_files.py write|read|list|delete|path {spec.job_id} <name>",
        "",
        spec.body,
        "",
        "Format the final reply concisely for Slack/Hermes delivery.",
    ]
    return "\n".join(lines)


def hermes_profile() -> str:
    return os.environ.get("LEGACY_JOBS_HERMES_PROFILE") or os.environ.get("HERMES_PROFILE") or "claudio-lab"


def hermes_command(prompt: str) -> list[str]:
    return ["hermes", "--profile", hermes_profile(), "chat", "-q", prompt, "--source", "legacy-cron"]


def run_agent_job(spec: LegacyJobSpec, repo_dir: Path = REPO_DIR, slack_client: Any | None = None) -> str:
    if not spec.body.strip():
        return "empty-body"
    prompt = build_agent_prompt(spec)
    try:
        result = subprocess.run(hermes_command(prompt), cwd=repo_dir, capture_output=True, text=True, timeout=AGENT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        deliver_output(spec, f":hourglass: job `{spec.job_id}` timed out after {AGENT_TIMEOUT_SEC}s", "timeout", slack_client)
        return "timeout"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "(no output)")[-1500:]
        deliver_output(spec, f":warning: job `{spec.job_id}` failed (rc={result.returncode}).\n```\n{tail}\n```", "error", slack_client)
        return "error"
    deliver_output(spec, result.stdout.strip() or "(empty reply)", "ok", slack_client)
    return "ok"


def per_job_lock_path(spec: LegacyJobSpec) -> Path:
    return spec.folder / ".lock"


def _lock_is_fresh(path: Path, now: float, stale_sec: float = LOCK_STALE_SEC) -> bool:
    return path.exists() and (now - path.stat().st_mtime) < stale_sec


@contextlib.contextmanager
def lock_file(path: Path, stale_sec: float = LOCK_STALE_SEC):
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if _lock_is_fresh(path, now, stale_sec):
        yield False
        return
    if path.exists():
        with contextlib.suppress(OSError):
            path.unlink()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield True
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def run_job(spec: LegacyJobSpec, repo_dir: Path = REPO_DIR, slack_client: Any | None = None) -> str:
    with lock_file(per_job_lock_path(spec)) as acquired:
        if not acquired:
            return "in-flight"
        job_type = str(spec.get("type") or "claude").lower()
        if job_type in {"command", "script", "shell"}:
            return run_command_job(spec, repo_dir, slack_client)
        return run_agent_job(spec, repo_dir, slack_client)


def tick(repo_dir: Path = REPO_DIR, *, slack_client: Any | None = None) -> list[str]:
    alerts: list[str] = []
    with lock_file(repo_dir / "bot" / "jobs_scheduler.lock") as acquired:
        if not acquired:
            return alerts
        state = load_state(repo_dir)
        jobs, warnings = discover_jobs(repo_dir)
        alerts.extend(warnings)
        now = time.time()
        state, jobs = reconcile_state(state, jobs, now, repo_dir)
        for jid, spec in sorted(jobs.items()):
            row = state.get(jid, {})
            if is_paused(spec) or row.get("next_run") is None or row["next_run"] > now:
                continue
            status = run_job(spec, repo_dir, slack_client)
            finished = time.time()
            override = consume_nextrun(spec, finished, consume=True)
            state[jid] = {
                "next_run": override if override is not None else compute_next_run(spec, finished),
                "last_run": finished,
                "last_status": status,
            }
        save_state(state, repo_dir)
    return alerts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_DIR))
    args = parser.parse_args(argv)
    alerts = tick(Path(args.repo).resolve())
    if alerts:
        print("\n".join(alerts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
