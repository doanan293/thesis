from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from seed_pipeline.config.paths import WORK_DIR


def _lock_path(target: Path, lock_root: Path) -> tuple[Path, str]:
    canonical = str(Path(target).resolve(strict=False))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Path(lock_root) / f"{digest}.lock", canonical


@contextmanager
def _acquire(
    target: Path,
    *,
    lock_root: Path,
    non_blocking: bool,
) -> Generator[Path, None, None]:
    lock_path, canonical = _lock_path(target, lock_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Local target {canonical} already has an active Kaggle job; "
                "wait for it to finish or choose a different run/job"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(canonical)
        handle.flush()
        try:
            yield Path(target)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def kaggle_job_lock(
    target: Path,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[Path]:
    root = lock_root if lock_root is not None else (WORK_DIR / "kaggle-job-locks")
    return _acquire(target, lock_root=root, non_blocking=True)


def kaggle_cache_lock(
    target: Path,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[Path]:
    root = lock_root if lock_root is not None else (WORK_DIR / "kaggle-cache-locks")
    return _acquire(target, lock_root=root, non_blocking=False)
