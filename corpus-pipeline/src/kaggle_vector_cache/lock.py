from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kaggle_vector_cache.temp_workspace import ProcessIdentity


@dataclass(frozen=True)
class RunResult:
    model: str
    run_count: int
    total: int
    complete: int
    missing: int
    cache_path: Path | None = None
    kernel_status: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.total > 0 and self.missing == 0


class ModelRunLock:
    def __init__(self, path: Path, model: str):
        self.path = Path(path)
        self.model = model
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self._owner_is_alive():
            with contextlib.suppress(OSError):
                self.path.unlink()
        payload = json.dumps(
            ProcessIdentity.current().to_dict(command="run")
            | {"model": self.model, "started_at": datetime.now(UTC).isoformat()}
        ).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                f"A vector-cache run is already running for {self.model}; lock: {self.path}"
            ) from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        self.acquired = True
        return self

    def _owner_is_alive(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            identity = ProcessIdentity(
                pid=int(payload["pid"]),
                boot_id=str(payload["boot_id"]),
                process_start_ticks=str(payload["process_start_ticks"]),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return True
        return identity.is_alive()

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired and self.path.exists():
            with contextlib.suppress(OSError):
                self.path.unlink()
            with contextlib.suppress(OSError):
                self.path.parent.rmdir()
        self.acquired = False
