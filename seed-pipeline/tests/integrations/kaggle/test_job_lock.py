import threading
from pathlib import Path

import pytest

from seed_pipeline.integrations.kaggle.job_lock import (
    KaggleAccountBusy,
    kaggle_account_lock,
    kaggle_cache_lock,
    kaggle_job_lock,
    lock_file_name,
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


def test_lock_file_is_named_after_the_target_inside_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    target = data / "cache" / "text_embeddings" / "qwen3_embedding_4b_fp16.jsonl"

    assert (
        lock_file_name(target, "job", data_dir=data)
        == "cache__text_embeddings__qwen3_embedding_4b_fp16.jsonl.job.lock"
    )


def test_lock_file_for_a_target_outside_data_is_marked_external(tmp_path: Path) -> None:
    data = tmp_path / "data"
    target = tmp_path / "elsewhere" / "run.json"
    expected = "_external/" + target.resolve().as_posix().lstrip("/")

    assert lock_file_name(target, "cache", data_dir=data) == (
        expected.replace("/", "__") + ".cache.lock"
    )


def test_job_and_cache_locks_on_one_target_can_nest(tmp_path: Path) -> None:
    target = tmp_path / "cache.jsonl"
    lock_root = tmp_path / "locks"

    with (
        kaggle_job_lock(target, lock_root=lock_root),
        kaggle_cache_lock(target, lock_root=lock_root),
    ):
        kinds = sorted(path.name.rsplit(".", 2)[-2] for path in lock_root.iterdir())

    assert kinds == ["cache", "job"]


def test_account_lock_uses_one_file_per_account(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"

    with kaggle_account_lock("acc2", lock_root=lock_root) as profile:
        assert profile == "acc2"
        assert (lock_root / "kaggle-accounts" / "acc2.lock").is_file()


def test_two_jobs_cannot_hold_the_same_account(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"
    entered = threading.Event()
    release = threading.Event()

    def first_job() -> None:
        with kaggle_account_lock("acc3", lock_root=lock_root):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=first_job)
    thread.start()
    assert entered.wait(2)
    try:
        with (
            pytest.raises(KaggleAccountBusy, match="acc3"),
            kaggle_account_lock("acc3", lock_root=lock_root),
        ):
            raise AssertionError("a held account must not be entered")
        with kaggle_account_lock("acc1", lock_root=lock_root):
            pass
    finally:
        release.set()
        thread.join()

    with kaggle_account_lock("acc3", lock_root=lock_root):
        pass


@pytest.mark.parametrize("profile", ["auto", "../acc1", "acc0", ""])
def test_account_lock_rejects_names_that_are_not_profiles(
    profile: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="invalid Kaggle account profile"):
        kaggle_account_lock(profile, lock_root=tmp_path)
