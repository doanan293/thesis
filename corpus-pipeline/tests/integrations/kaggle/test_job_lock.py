import threading

import pytest

from corpus_pipeline.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
)


def test_same_target_lock_fails_immediately(tmp_path):
    target = tmp_path / "results" / "scores.jsonl"
    lock_root = tmp_path / "locks"
    with (
        kaggle_job_lock(target, lock_root=lock_root),
        pytest.raises(RuntimeError, match="already has an active Kaggle job"),
        kaggle_job_lock(target, lock_root=lock_root),
    ):
        raise AssertionError("contended lock must not be entered")


def test_distinct_targets_can_lock_concurrently(tmp_path):
    lock_root = tmp_path / "locks"
    with (
        kaggle_job_lock(tmp_path / "a", lock_root=lock_root),
        kaggle_job_lock(tmp_path / "b", lock_root=lock_root),
    ):
        assert True


def test_lock_releases_after_exception(tmp_path):
    target = tmp_path / "result"
    lock_root = tmp_path / "locks"
    with (
        pytest.raises(ValueError, match="boom"),
        kaggle_job_lock(target, lock_root=lock_root),
    ):
        raise ValueError("boom")
    with kaggle_job_lock(target, lock_root=lock_root):
        assert True


def test_cache_lock_serializes_writers(tmp_path):
    entered = threading.Event()
    finished = threading.Event()
    target = tmp_path / "cache.jsonl"
    lock_root = tmp_path / "cache-locks"

    def second_writer():
        with kaggle_cache_lock(target, lock_root=lock_root):
            entered.set()
        finished.set()

    with kaggle_cache_lock(target, lock_root=lock_root):
        thread = threading.Thread(target=second_writer)
        thread.start()
        assert not entered.wait(0.1)
    assert entered.wait(1.0)
    assert finished.wait(1.0)
    thread.join()
