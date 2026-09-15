from __future__ import annotations

import fcntl
import os
from collections.abc import Generator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Literal

from seed_pipeline.config.paths import DATA_DIR, LOCK_DIR
from seed_pipeline.integrations.kaggle.config import ACCOUNT_NAME

LockKind = Literal["job", "cache", "publish"]
MAX_LOCK_FILE_NAME_BYTES = 255
ACCOUNT_LOCK_DIRNAME = "kaggle-accounts"


class KaggleAccountBusy(RuntimeError):
    """Another local command holds the session lock of this Kaggle account."""

    def __init__(self, profile: str):
        super().__init__(
            f"Kaggle account {profile} already runs a session for another local job"
        )
        self.profile = profile


class _LockHeldError(Exception):
    """A non-blocking flock found the lock taken."""


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
def _flock(
    lock_path: Path, content: str, *, non_blocking: bool
) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise _LockHeldError(str(lock_path)) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(content)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    with ExitStack() as stack:
        # Only the acquisition is translated; errors from the body pass through.
        try:
            stack.enter_context(_flock(lock_path, canonical, non_blocking=non_blocking))
        except _LockHeldError as exc:
            raise RuntimeError(
                f"Local target {canonical} already has an active Kaggle job; "
                "wait for it to finish or choose a different run/job"
            ) from exc
        yield Path(target)


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


def kaggle_account_lock(
    profile: str,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[str]:
    """Hold the session lock of one Kaggle account, failing at once if it is taken.

    Parallel `seed rerank` commands for different models share the account pool;
    this lock keeps each account to one GPU session started from this machine.
    """
    if ACCOUNT_NAME.fullmatch(profile) is None:
        raise ValueError(f"invalid Kaggle account profile: {profile!r}")
    root = LOCK_DIR if lock_root is None else lock_root
    return _account_lock(profile, Path(root) / ACCOUNT_LOCK_DIRNAME / f"{profile}.lock")


@contextmanager
def _account_lock(profile: str, lock_path: Path) -> Generator[str, None, None]:
    with ExitStack() as stack:
        try:
            stack.enter_context(
                _flock(lock_path, f"{profile} pid={os.getpid()}\n", non_blocking=True)
            )
        except _LockHeldError as exc:
            raise KaggleAccountBusy(profile) from exc
        yield profile
