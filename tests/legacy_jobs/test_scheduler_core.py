from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock

from scripts.legacy_jobs import scheduler


class SchedulerDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "jobs" / "team" / "a").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write_job(self, rel, fm="every: 1m", body="body"):
        path = self.repo / rel / "job.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\n{fm}\n---\n\n{body}\n")
        return path

    def test_discovers_job_md_at_any_depth(self):
        self.write_job("jobs/team/nested/foo")
        jobs, warnings = scheduler.discover_jobs(self.repo)
        self.assertIn("jobs/team/nested/foo", jobs)
        self.assertEqual(warnings, [])

    def test_skips_deleted_archive(self):
        self.write_job("jobs/_deleted/old")
        jobs, _ = scheduler.discover_jobs(self.repo)
        self.assertEqual(jobs, {})

    def test_job_id_is_folder_path(self):
        self.write_job("jobs/team/foo")
        jobs, _ = scheduler.discover_jobs(self.repo)
        self.assertEqual(jobs["jobs/team/foo"].job_id, "jobs/team/foo")

    def test_defaults_post_mode_by_root(self):
        self.write_job("jobs/team/foo")
        self.write_job("jobs/slack/U123/foo")
        jobs, _ = scheduler.discover_jobs(self.repo)
        self.assertEqual(jobs["jobs/team/foo"].post_mode, "log")
        self.assertEqual(jobs["jobs/slack/U123/foo"].post_mode, "dm")

    def test_empty_or_bad_frontmatter_is_skipped_with_warning(self):
        bad = self.repo / "jobs" / "team" / "bad" / "job.md"
        bad.parent.mkdir(parents=True)
        bad.write_text("no frontmatter")
        jobs, warnings = scheduler.discover_jobs(self.repo)
        self.assertEqual(jobs, {})
        self.assertTrue(warnings)


class SchedulerStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, rel="jobs/team/foo", fm=None):
        fm = fm or {"every": "1m", "jitter": False}
        folder = self.repo / rel
        folder.mkdir(parents=True, exist_ok=True)
        return scheduler.LegacyJobSpec(rel, folder, folder / "job.md", fm, "body")

    def test_seeds_next_run_for_every(self):
        s = self.spec()
        state, _ = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertEqual(state[s.job_id]["next_run"], 1060)

    def test_seeds_next_run_for_cron(self):
        s = self.spec(fm={"schedule": "0 9 * * *", "jitter": False})
        state, _ = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertIsNotNone(state[s.job_id]["next_run"])

    def test_paused_job_not_due(self):
        s = self.spec(fm={"every": "1m", "paused": True})
        self.assertTrue(scheduler.is_paused(s))

    def test_expired_job_soft_deleted(self):
        s = self.spec(fm={"every": "1m", "expires_at": "1970-01-01T00:00:01Z"})
        state, jobs = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertEqual(state, {})
        self.assertEqual(jobs, {})
        self.assertTrue((self.repo / "jobs" / "_deleted").exists())

    def test_nextrun_duration_consumed(self):
        s = self.spec()
        (s.folder / "nextrun.txt").write_text("4m")
        state, _ = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertEqual(state[s.job_id]["next_run"], 1240)
        self.assertFalse((s.folder / "nextrun.txt").exists())

    def test_nextrun_epoch_consumed(self):
        s = self.spec()
        (s.folder / "nextrun.txt").write_text("2000000000")
        state, _ = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertEqual(state[s.job_id]["next_run"], 2000000000)

    def test_bad_nextrun_removed_and_ignored(self):
        s = self.spec()
        (s.folder / "nextrun.txt").write_text("bad")
        state, _ = scheduler.reconcile_state({}, {s.job_id: s}, 1000, self.repo)
        self.assertEqual(state[s.job_id]["next_run"], 1060)
        self.assertFalse((s.folder / "nextrun.txt").exists())

    def test_jitter_stable_per_job(self):
        self.assertEqual(scheduler.stable_jitter("jobs/team/foo", 3600), scheduler.stable_jitter("jobs/team/foo", 3600))


class SchedulerDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.folder = self.repo / "jobs" / "team" / "foo"
        self.folder.mkdir(parents=True)
        self.spec = scheduler.LegacyJobSpec("jobs/team/foo", self.folder, self.folder / "job.md", {"post_mode": "log"}, "")

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_delivery_appends_runs_log(self):
        scheduler.deliver_output(self.spec, "hello", "ok")
        self.assertIn("hello", (self.folder / "runs.log").read_text())

    def test_unknown_post_mode_falls_back_to_log(self):
        self.spec.frontmatter["post_mode"] = "weird"
        scheduler.deliver_output(self.spec, "hello", "ok")
        self.assertIn("hello", (self.folder / "runs.log").read_text())

    def test_empty_output_still_logs_if_job_ran(self):
        scheduler.deliver_output(self.spec, "", "ok")
        self.assertIn("status=ok", (self.folder / "runs.log").read_text())


class SchedulerCommandJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.folder = self.repo / "jobs" / "team" / "foo"
        self.folder.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def make_spec(self, fm):
        fm = {"type": "command", "post_mode": "log", **fm}
        return scheduler.LegacyJobSpec("jobs/team/foo", self.folder, self.folder / "job.md", fm, "")

    def log(self):
        return (self.folder / "runs.log").read_text()

    def test_command_job_success_logs_stdout(self):
        status = scheduler.run_command_job(self.make_spec({"command": "printf ok"}), self.repo)
        self.assertEqual(status, "ok")
        self.assertIn("ok", self.log())

    def test_command_job_uses_stderr_when_stdout_empty(self):
        status = scheduler.run_command_job(self.make_spec({"command": "printf err >&2"}), self.repo)
        self.assertEqual(status, "ok")
        self.assertIn("err", self.log())

    def test_command_job_nonzero_logs_error(self):
        status = scheduler.run_command_job(self.make_spec({"command": "echo bad >&2; exit 2"}), self.repo)
        self.assertEqual(status, "error")
        self.assertIn("rc=2", self.log())

    def test_script_job_runs_python_file(self):
        script = self.repo / "hello.py"
        script.write_text("print('script ok')")
        status = scheduler.run_command_job(self.make_spec({"script": "hello.py"}), self.repo)
        self.assertEqual(status, "ok")
        self.assertIn("script ok", self.log())

    def test_command_job_timeout_logs_timeout(self):
        with mock.patch.object(scheduler, "JOB_TIMEOUT_SEC", 1):
            status = scheduler.run_command_job(self.make_spec({"command": "sleep 2"}), self.repo)
        self.assertEqual(status, "timeout")


class SchedulerAgentJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.folder = self.repo / "jobs" / "team" / "foo"
        self.folder.mkdir(parents=True)
        self.spec = scheduler.LegacyJobSpec("jobs/team/foo", self.folder, self.folder / "job.md", {"post_mode": "log", "title": "T"}, "do work")

    def tearDown(self):
        self.tmp.cleanup()

    def test_agent_prompt_includes_runtime_envelope(self):
        prompt = scheduler.build_agent_prompt(self.spec)
        self.assertIn("Job path: jobs/team/foo", prompt)
        self.assertIn("Job title: T", prompt)

    def test_agent_prompt_includes_persisted_state(self):
        (self.folder / "state.txt").write_text("old")
        self.assertIn("old", scheduler.build_agent_prompt(self.spec))

    def test_agent_job_success_logs_final_stdout(self):
        fake = subprocess_result(0, "done", "")
        with mock.patch.object(scheduler.subprocess, "run", return_value=fake):
            status = scheduler.run_agent_job(self.spec, self.repo)
        self.assertEqual(status, "ok")
        self.assertIn("done", (self.folder / "runs.log").read_text())

    def test_agent_job_nonzero_logs_error(self):
        fake = subprocess_result(1, "", "boom")
        with mock.patch.object(scheduler.subprocess, "run", return_value=fake):
            status = scheduler.run_agent_job(self.spec, self.repo)
        self.assertEqual(status, "error")
        self.assertIn("boom", (self.folder / "runs.log").read_text())

    def test_empty_body_is_skipped(self):
        self.spec.body = ""
        self.assertEqual(scheduler.run_agent_job(self.spec, self.repo), "empty-body")


def subprocess_result(returncode, stdout, stderr):
    cp = mock.Mock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


if __name__ == "__main__":
    unittest.main()
