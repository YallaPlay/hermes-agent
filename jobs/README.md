# Legacy Jobs on Hermes Cron

Recurring jobs keep the legacy Claudio `jobs/**/job.md` folder model while Hermes cron drives the scheduler.

Hermes runs one `no_agent` cron job named `legacy-jobs-scheduler` on `every 1m`. Its profile script calls `scripts/legacy_jobs/hermes_tick.py`, which performs two scheduler ticks separated by about 30 seconds. This preserves the legacy 30s poll cadence without patching Hermes core.

## Layout

- `jobs/system/<slug>/job.md` — system maintenance jobs, normally `post_mode: log`.
- `jobs/team/<slug>/job.md` — repo-authored team jobs.
- `jobs/slack/<user>/<slug>/job.md` — Slack/user-created jobs; gitignored.
- `jobs/_deleted/` — soft-delete archive; gitignored.

The job id is the containing folder path relative to the repo, for example `jobs/team/daily_dau`.

Runtime files next to a job spec are gitignored:

- `state.txt` — opaque persisted state, capped at 8 KiB.
- `data/` — larger per-job scratch data.
- `nextrun.txt` — one-shot dynamic schedule override.
- `pin.txt` — legacy Slack thread pin compatibility.
- `runs.log` / `runs/` — local delivery output.
- `.lock` — per-job single-flight lock.

`bot/jobs_state.json` remains the scheduler-owned runtime state file for legacy compatibility. Do not edit it manually.

## Spec format

```markdown
---
title: "Daily DAU sanity check"
type: claude              # claude (default) | command | script | shell
schedule: "0 9 * * 1-5"  # cron, OR
# every: 1h               # duration: <int><s|m|h|d>, minimum effective interval 1m
post_mode: log            # log | dm | new_message | thread
created_by: U0123ABC      # required for dm
channel: C0123XYZ         # required for new_message/thread
thread_ts: "1700000000.1" # required for thread
expires_at: "2026-06-01T00:00:00Z"
jitter: false             # optional; default true
---

Job prompt or human notes.
```

A job is valid when it has one of `schedule`, `every`, or a pending `nextrun.txt`. Slack-created jobs default to `post_mode: dm`; all others default to `post_mode: log`.

## Job types

- `claude` or missing `type`: runs the body through Hermes CLI as a one-shot agent job.
- `command`, `script`, or `shell`: deterministic no-LLM job. `command:` runs through `bash -lc` from repo root. `script:` runs as `python3 <script>` from repo root.

## Persisted state

Use the repo-local helpers from command/agent jobs:

- `python3 scripts/legacy_jobs/job_state.py --get jobs/team/foo`
- `echo value | python3 scripts/legacy_jobs/job_state.py --set jobs/team/foo`
- `python3 scripts/legacy_jobs/job_state.py --clear jobs/team/foo`
- `python3 scripts/legacy_jobs/job_files.py write jobs/team/foo results.csv < data.csv`
- `python3 scripts/legacy_jobs/job_files.py read jobs/team/foo results.csv`

## Dynamic self-scheduling

A job or helper process may write `nextrun.txt` in the job folder. The scheduler consumes and deletes it once. Contents may be a duration (`4m`, `3h`, `2d`) or epoch seconds. Values below the 60s minimum are raised to 60s.

## Delivery

`post_mode: log` appends to `<job_folder>/runs.log`. Slack modes are attempted only when Slack credentials are available; failures always fall back to `runs.log`. The Hermes cron wrapper itself stays silent for normal idle/successful ticks so local cron delivery is quiet.
