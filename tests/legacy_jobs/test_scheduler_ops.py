from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts.legacy_jobs import scheduler
from scripts.legacy_jobs import hermes_tick
from scripts.legacy_jobs import install_hermes_cron
from scripts.legacy_jobs import schedule_job


class SchedulerLockingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.folder = self.repo / "jobs" / "team" / "foo"
        self.folder.mkdir(parents=True)
        self.spec = scheduler.LegacyJobSpec("jobs/team/foo", self.folder, self.folder / "job.md", {"type": "command", "command": "printf ok", "post_mode": "log"}, "")

    def tearDown(self):
        self.tmp.cleanup()

    def test_global_lock_causes_second_tick_to_exit_silent(self):
        lock = self.repo / "bot" / "jobs_scheduler.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("held")
        self.assertEqual(scheduler.tick(self.repo), [])

    def test_stale_global_lock_is_recovered(self):
        lock = self.repo / "bot" / "jobs_scheduler.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("stale")
        old = 1000
        import os
        os.utime(lock, (old, old))
        with mock.patch.object(scheduler, "LOCK_STALE_SEC", 1):
            self.assertEqual(scheduler.tick(self.repo), [])
        self.assertFalse(lock.exists())

    def test_per_job_lock_prevents_duplicate_job(self):
        (self.folder / ".lock").write_text("held")
        self.assertEqual(scheduler.run_job(self.spec, self.repo), "in-flight")

    def test_per_job_lock_cleared_after_success(self):
        scheduler.run_job(self.spec, self.repo)
        self.assertFalse((self.folder / ".lock").exists())

    def test_per_job_lock_cleared_after_failure(self):
        self.spec.frontmatter["command"] = "exit 2"
        scheduler.run_job(self.spec, self.repo)
        self.assertFalse((self.folder / ".lock").exists())


class HermesTickTests(unittest.TestCase):
    def test_wrapper_runs_two_ticks_by_default(self):
        calls = []
        with mock.patch.object(scheduler, "tick", side_effect=lambda repo: calls.append(repo) or []):
            hermes_tick.run_ticks(Path('/tmp/repo'), sleep=lambda _: None)
        self.assertEqual(len(calls), 2)

    def test_wrapper_sleeps_between_ticks(self):
        sleeps = []
        with mock.patch.object(scheduler, "tick", return_value=[]):
            hermes_tick.run_ticks(Path('/tmp/repo'), ticks=2, interval_sec=0.01, sleep=sleeps.append)
        self.assertEqual(len(sleeps), 1)

    def test_wrapper_once_runs_one_tick(self):
        with mock.patch.object(hermes_tick, "run_ticks", return_value=[]) as run:
            self.assertEqual(hermes_tick.main(["--repo", "/tmp/repo", "--once"]), 0)
            self.assertEqual(run.call_args.args[1], 1)

    def test_wrapper_silent_on_success(self):
        with mock.patch.object(hermes_tick, "run_ticks", return_value=[]):
            self.assertEqual(hermes_tick.main(["--repo", "/tmp/repo", "--once"]), 0)

    def test_wrapper_alerts_on_scheduler_exception(self):
        with mock.patch.object(hermes_tick, "run_ticks", side_effect=RuntimeError("boom")):
            self.assertEqual(hermes_tick.main(["--repo", "/tmp/repo", "--once"]), 1)


class InstallHermesCronTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "profile"
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "scripts" / "legacy_jobs").mkdir(parents=True)
        (self.repo / "scripts" / "legacy_jobs" / "hermes_tick.py").write_text("print('x')")

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_prints_expected_paths(self):
        spec = install_hermes_cron.cron_spec(self.repo)
        self.assertEqual(spec["script"], "legacy_jobs_tick.py")
        self.assertTrue(spec["no_agent"])

    def test_write_script_creates_profile_script(self):
        target = install_hermes_cron.write_profile_script(self.home, self.repo)
        self.assertTrue(target.exists())
        self.assertIn("hermes_tick.py", target.read_text())

    def test_does_not_create_cron_by_default(self):
        self.assertEqual(install_hermes_cron.main(["--repo", str(self.repo), "--hermes-home", str(self.home)]), 0)
        self.assertFalse((self.home / "cron" / "jobs.json").exists())


class SlackDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.folder = self.repo / "jobs" / "slack" / "U1" / "foo"
        self.folder.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, fm):
        return scheduler.LegacyJobSpec("jobs/slack/U1/foo", self.folder, self.folder / "job.md", fm, "")

    def test_dm_delivery_opens_conversation(self):
        client = FakeSlack()
        scheduler.deliver_output(self.spec({"post_mode": "dm", "created_by": "U1"}), "hi", slack_client=client)
        self.assertEqual(client.opened, "U1")
        self.assertEqual(client.messages[0]["channel"], "D1")

    def test_new_message_delivery_posts_channel(self):
        client = FakeSlack()
        scheduler.deliver_output(self.spec({"post_mode": "new_message", "channel": "C1"}), "hi", slack_client=client)
        self.assertEqual(client.messages[0]["channel"], "C1")

    def test_thread_delivery_uses_thread_ts(self):
        client = FakeSlack()
        scheduler.deliver_output(self.spec({"post_mode": "thread", "channel": "C1", "thread_ts": "123.4"}), "hi", slack_client=client)
        self.assertEqual(client.messages[0]["thread_ts"], "123.4")

    def test_missing_token_falls_back_to_log(self):
        scheduler.deliver_output(self.spec({"post_mode": "dm", "created_by": "U1"}), "hi")
        self.assertIn("hi", (self.folder / "runs.log").read_text())

    def test_slack_api_error_falls_back_to_log(self):
        client = FakeSlack(fail=True)
        scheduler.deliver_output(self.spec({"post_mode": "new_message", "channel": "C1"}), "hi", slack_client=client)
        self.assertIn("hi", (self.folder / "runs.log").read_text())


class FakeSlack:
    def __init__(self, fail=False):
        self.fail = fail
        self.opened = None
        self.messages = []

    def conversations_open(self, users):
        if self.fail:
            raise RuntimeError("no")
        self.opened = users
        return {"channel": {"id": "D1"}}

    def chat_postMessage(self, **kwargs):
        if self.fail:
            raise RuntimeError("no")
        self.messages.append(kwargs)
        return {"ok": True, "ts": "1.0"}


class ScheduleJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_job_writes_job_md(self):
        args = mock.Mock(slack_user="U1", slug="daily", title="Daily", schedule="0 9 * * *", every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        path = schedule_job.create_job(args, "body", self.repo)
        self.assertTrue(path.exists())

    def test_create_infers_dm_post_mode_without_channel(self):
        self.assertEqual(schedule_job.infer_post_mode(None, None, None), "dm")

    def test_create_infers_new_message_with_channel(self):
        self.assertEqual(schedule_job.infer_post_mode("C1", None, None), "new_message")

    def test_create_infers_thread_with_channel_and_thread_ts(self):
        self.assertEqual(schedule_job.infer_post_mode("C1", "1.0", None), "thread")

    def test_list_jobs_tsv(self):
        args = mock.Mock(slack_user="U1", slug="daily", title="Daily", schedule="0 9 * * *", every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        schedule_job.create_job(args, "body", self.repo)
        self.assertIn("daily\t0 9 * * *", schedule_job.list_jobs("U1", self.repo)[0])

    def test_pause_resume_updates_frontmatter(self):
        args = mock.Mock(slack_user="U1", slug="daily", title="Daily", schedule="0 9 * * *", every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        schedule_job.create_job(args, "body", self.repo)
        pause = mock.Mock(slack_user="U1", slug="daily", pause=True, resume=False, title=None, schedule=None, every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        schedule_job.update_job(pause, None, self.repo)
        fm, _ = schedule_job.read_job(self.repo / "jobs" / "slack" / "U1" / "daily" / "job.md")
        self.assertTrue(fm["paused"])
        resume = mock.Mock(slack_user="U1", slug="daily", pause=False, resume=True, title=None, schedule=None, every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        schedule_job.update_job(resume, None, self.repo)
        fm, _ = schedule_job.read_job(self.repo / "jobs" / "slack" / "U1" / "daily" / "job.md")
        self.assertNotIn("paused", fm)

    def test_delete_soft_deletes_folder(self):
        args = mock.Mock(slack_user="U1", slug="daily", title="Daily", schedule="0 9 * * *", every=None, channel=None, thread_ts=None, post_mode=None, expires_at=None)
        schedule_job.create_job(args, "body", self.repo)
        dest = schedule_job.soft_delete(self.repo / "jobs" / "slack" / "U1" / "daily", self.repo)
        self.assertTrue(dest.exists())

    def test_rejects_path_traversal_slug(self):
        with self.assertRaises(ValueError):
            schedule_job.validate_slug("../bad")


if __name__ == "__main__":
    unittest.main()
