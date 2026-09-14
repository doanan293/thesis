from __future__ import annotations

import fcntl
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal

from seed_pipeline.config.paths import DATA_DIR, LOCK_DIR

LockKind = Literal["job", "cache"]
MAX_LOCK_FILE_NAME_BYTES = 255


def lock_file_name(
    target: Path, kind: LockKind, *, data_dir: Path | None = None
) -> str:
    """Name a lock after its target: the path under data/ with "/" replaced by "__".

    The kind is part of the name because one command holds a job lock and a cache lock on
    the same target, and flock locks belong to the open handle, not to the process.
    """
    resolved = Path(target).resolve(strict=False)
    base = Path(DATA_DIR if data_dir is None else data_dir).resolve(strict=False)
    try:
        relative = resolved.relative_to(base).as_posix()
    except ValueError:
        relative = "_external/" + resolved.as_posix().lstrip("/")
    name = f"{relative.replace('/', '__')}.{kind}.lock"
    if len(name.encode("utf-8")) > MAX_LOCK_FILE_NAME_BYTES:
        raise ValueError(f"Lock target path is too long for a lock file name: {target}")
    return name


@contextmanager
def _acquire(
    target: Path,
    *,
    kind: LockKind,
    lock_root: Path,
    non_blocking: bool,
) -> Generator[Path, None, None]:
    canonical = str(Path(target).resolve(strict=False))
    lock_path = Path(lock_root) / lock_file_name(target, kind)
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
    root = LOCK_DIR if lock_root is None else lock_root
    return _acquire(target, kind="job", lock_root=root, non_blocking=True)


def kaggle_cache_lock(
    target: Path,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[Path]:
    root = LOCK_DIR if lock_root is None else lock_root
    return _acquire(target, kind="cache", lock_root=root, non_blocking=False)
