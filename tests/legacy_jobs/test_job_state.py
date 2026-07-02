from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.legacy_jobs import job_state


class JobStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "jobs" / "team" / "foo").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_get_clear_state(self):
        job_state.write_state("jobs/team/foo", "abc", self.repo)
        self.assertEqual(job_state.read_state("jobs/team/foo", self.repo), "abc")
        job_state.clear_state("jobs/team/foo", self.repo)
        self.assertEqual(job_state.read_state("jobs/team/foo", self.repo), "")

    def test_accepts_job_md_path(self):
        path = self.repo / "jobs" / "team" / "foo" / "job.md"
        path.write_text("---\nevery: 1m\n---\n")
        self.assertEqual(job_state.state_file("jobs/team/foo/job.md", self.repo), path.parent / "state.txt")

    def test_rejects_outside_jobs_path(self):
        with self.assertRaises(job_state.JobPathError):
            job_state.state_file("README.md", self.repo)

    def test_truncates_to_8kb(self):
        text = "a" * (job_state.MAX_STATE_BYTES + 10)
        truncated = job_state.write_state("jobs/team/foo", text, self.repo)
        self.assertTrue(truncated)
        data = job_state.read_state("jobs/team/foo", self.repo).encode()
        self.assertEqual(len(data), job_state.MAX_STATE_BYTES)


if __name__ == "__main__":
    unittest.main()
