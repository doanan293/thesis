from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"


def _process_start_ticks(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return fields[21]


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    boot_id: str
    process_start_ticks: str

    @classmethod
    def current(cls) -> ProcessIdentity:
        pid = os.getpid()
        return cls(pid, _boot_id(), _process_start_ticks(pid))

    def to_dict(self, command: str) -> dict[str, object]:
        return {
            "pid": self.pid,
            "boot_id": self.boot_id,
            "process_start_ticks": self.process_start_ticks,
            "created_at": datetime.now(UTC).isoformat(),
            "command": command,
        }

    def is_alive(self) -> bool:
        try:
            return (
                self.boot_id == _boot_id()
                and self.process_start_ticks == _process_start_ticks(self.pid)
            )
        except (FileNotFoundError, IndexError, OSError):
            return False


class TemporaryWorkspace:
    PREFIX = "corpus-pipeline-kaggle-"

    def __init__(self, temp_root: Path, command: str):
        self.temp_root = Path(temp_root).resolve()
        self.command = command
        self.path: Path | None = None
        self._created_temp_root = False

    @property
    def kernel_dir(self) -> Path:
        assert self.path is not None
        return self.path / "kernel"

    @property
    def incoming_dir(self) -> Path:
        assert self.path is not None
        return self.path / "incoming"

    @property
    def checkpoint_dir(self) -> Path:
        assert self.path is not None
        return self.path / "checkpoint-dataset"

    @classmethod
    def collect_stale(cls, temp_root: Path) -> None:
        root = Path(temp_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for candidate in root.glob(f"{cls.PREFIX}*"):
            if not candidate.is_dir() or candidate.parent != root:
                continue
            try:
                payload = json.loads(
                    (candidate / "workspace.json").read_text(encoding="utf-8")
                )
                identity = ProcessIdentity(
                    pid=int(payload["pid"]),
                    boot_id=str(payload["boot_id"]),
                    process_start_ticks=str(payload["process_start_ticks"]),
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if not identity.is_alive():
                shutil.rmtree(candidate, ignore_errors=True)

    def __enter__(self) -> TemporaryWorkspace:
        self._created_temp_root = not self.temp_root.exists()
        self.collect_stale(self.temp_root)
        self.path = Path(tempfile.mkdtemp(prefix=self.PREFIX, dir=str(self.temp_root)))
        for child in (self.kernel_dir, self.incoming_dir, self.checkpoint_dir):
            child.mkdir()
        (self.path / "workspace.json").write_text(
            json.dumps(
                ProcessIdentity.current().to_dict(self.command),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.path is None:
            return
        try:
            shutil.rmtree(self.path)
        except OSError as cleanup_error:
            print(
                f"Warning: could not clean Kaggle workspace "
                f"{self.path}: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
            return
        if self._created_temp_root:
            with suppress(OSError):
                self.temp_root.rmdir()


@contextmanager
def unwind_on_sigterm():
    previous = signal.getsignal(signal.SIGTERM)

    def raise_interrupt(signum, frame):
        del signum, frame
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
