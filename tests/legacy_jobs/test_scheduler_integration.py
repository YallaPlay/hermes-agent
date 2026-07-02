from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from scripts.legacy_jobs import hermes_tick


class SchedulerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        folder = self.repo / "jobs" / "team" / "_smoke_command"
        folder.mkdir(parents=True)
        (self.repo / "bot").mkdir()
        (folder / "job.md").write_text('''---
title: "Legacy scheduler smoke command"
type: command
every: 1m
post_mode: log
jitter: false
command: "printf 'legacy scheduler smoke ok'"
---

Command smoke job for the Hermes legacy scheduler port.
''')
        (self.repo / "bot" / "jobs_state.json").write_text(
            '{"jobs/team/_smoke_command": {"next_run": 0, "last_run": null, "last_status": null}}'
        )
        self.folder = folder

    def tearDown(self):
        self.tmp.cleanup()

    def test_scheduler_runs_log_only_command_job_end_to_end(self):
        rc = hermes_tick.main(["--repo", str(self.repo), "--once"])
        self.assertEqual(rc, 0)
        self.assertIn("legacy scheduler smoke ok", (self.folder / "runs.log").read_text())
        state = json.loads((self.repo / "bot" / "jobs_state.json").read_text())
        self.assertEqual(state["jobs/team/_smoke_command"]["last_status"], "ok")


if __name__ == "__main__":
    unittest.main()
