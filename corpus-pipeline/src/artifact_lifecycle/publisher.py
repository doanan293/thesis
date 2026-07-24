from __future__ import annotations

import fcntl
import shutil
from collections.abc import Callable
from pathlib import Path

from artifact_lifecycle.contract import validate_contract_directory


class PublishError(RuntimeError):
    """Raised when final contract activation or recovery is ambiguous."""


def _paths(final_dir: Path) -> tuple[Path, Path, Path]:
    final_dir = Path(final_dir)
    return (
        final_dir,
        final_dir.parent / f".{final_dir.name}.next",
        final_dir.parent / f".{final_dir.name}.previous",
    )


def recover_publish(final_dir: Path) -> None:
    final, next_dir, previous = _paths(final_dir)
    if next_dir.exists():
        shutil.rmtree(next_dir)
    if not final.exists() and previous.exists():
        previous.replace(final)
    elif final.exists() and previous.exists():
        try:
            validate_contract_directory(final)
        except Exception:
            shutil.rmtree(final)
            previous.replace(final)
        else:
            shutil.rmtree(previous)
    elif not final.exists() and not previous.exists():
        return


def publish_contract(
    candidate_dir: Path,
    final_dir: Path,
    *,
    validate: Callable[[Path], object] = validate_contract_directory,
    failpoint: Callable[[str], None] | None = None,
) -> None:
    candidate_dir = Path(candidate_dir)
    final, next_dir, previous = _paths(final_dir)
    final.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final.parent / f".{final.name}.publish.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        recover_publish(final)
        validate(candidate_dir)
        if next_dir.exists():
            shutil.rmtree(next_dir)
        shutil.copytree(candidate_dir, next_dir)
        validate(next_dir)
        had_previous = final.exists()
        if had_previous:
            if previous.exists():
                raise PublishError(f"Ambiguous previous final directory: {previous}")
            final.replace(previous)
        try:
            if failpoint is not None:
                failpoint("after_backup")
            next_dir.replace(final)
            validate(final)
            if previous.exists():
                shutil.rmtree(previous)
        except Exception:
            if final.exists():
                shutil.rmtree(final)
            if previous.exists():
                previous.replace(final)
            if next_dir.exists():
                shutil.rmtree(next_dir)
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
