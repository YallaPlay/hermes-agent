"""Concurrent reviews must not reset or clear another fork's read-before-write marks."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from tools.file_tools import read_file_tool
from tools.skill_manager_guards import (
    _background_review_read_before_write_guard,
    _reset_background_review_read_marks,
    clear_background_review_read_marks,
)
from tools.skill_provenance import (
    BACKGROUND_REVIEW,
    reset_current_write_origin,
    set_current_review_fork_id,
    set_current_write_origin,
)


def test_parallel_review_reset_and_cleanup_preserve_other_forks_reads(tmp_path):
    target = tmp_path / "SKILL.md"
    target.write_text("Shared skill content\n", encoding="utf-8")
    first_read, second_finished = Event(), Event()

    def guard():
        return _background_review_read_before_write_guard("demo", target, "patch", "SKILL.md")

    def review(key):
        origin = set_current_write_origin(BACKGROUND_REVIEW)
        fork = set_current_review_fork_id(key)
        try:
            if key == "first":
                _reset_background_review_read_marks()
                # A tool-worker snapshot reads, then another snapshot checks the write.
                assert "Shared skill content" in contextvars.copy_context().run(read_file_tool, str(target))
                first_read.set()
                assert second_finished.wait(5), "second review did not finish"
                assert contextvars.copy_context().run(guard) is None
            else:
                assert first_read.wait(5), "first review did not read"
                _reset_background_review_read_marks()
                refused = guard()
                assert refused is not None and refused["_read_before_write_required"] is True
                assert "Shared skill content" in contextvars.copy_context().run(read_file_tool, str(target))
                assert contextvars.copy_context().run(guard) is None
                clear_background_review_read_marks(key)
                refused = guard()
                assert refused is not None and refused["_read_before_write_required"] is True
                second_finished.set()
        finally:
            clear_background_review_read_marks(key)
            fork.var.reset(fork)
            reset_current_write_origin(origin)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(review, "first")
        second = workers.submit(review, "second")
        first.result(timeout=10)
        second.result(timeout=10)
