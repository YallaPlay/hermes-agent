# yallaplay-hermes-agent

Fresh Hermes-based pilot for Claudio, YallaPlay's internal interactive agent.

This repo intentionally starts without Slack or scheduler integration. The first milestone is to rebuild the interactive Claudio capabilities on Hermes: knowledge lookup, analytics workflows, engineering/repo investigation, liveops reads, collaboration reads, observability reads, and safe knowledge capture.

## Layout

- `.hermes.md` - project context, routing, and safety rules loaded by Hermes in this repo.
- `SOUL.md` - Claudio identity and default interaction style for a Hermes profile.
- `skills/` - Hermes skills for domain workflows and safety policy.
- `tools/` - thin executable wrappers around YallaPlay APIs/CLIs.
- `config/` - non-secret examples only.
- `docs/` - pilot notes and migration plan.
- `yallaplay-wiki/` - git submodule; central knowledge bin for findings, definitions, methodology, queries, and schema notes.

## Pilot Scope

In scope now:

- Interactive CLI usage through Hermes.
- Read-only or explicitly gated tools.
- Wiki-backed knowledge lookup and capture.
- Provider/model experimentation, including Bedrock where available.

Out of scope for the first milestone:

- Slack bot routing.
- Recurring scheduler/jobs.
- Ungated writes to production systems.
- Migrating historical Codex/Claude session state.

## Setup Sketch

```bash
git submodule update --init --recursive
bash scripts/bootstrap_profile.sh claudio-lab
claudio-lab setup
claudio-lab chat
```

`bootstrap_profile.sh` keeps repo-owned YallaPlay skills as the source of truth
by symlinking `skills/yallaplay/*` into the target Hermes profile's
`skills/claudio-authored/` category for a clear visual split from bundled
Hermes skills. In an already-running TUI, run `/reload-skills` after changing
those repo skill files.

Use this as a lab profile first. Do not point production Slack or scheduled jobs at it until the interactive workflows have been tested.
