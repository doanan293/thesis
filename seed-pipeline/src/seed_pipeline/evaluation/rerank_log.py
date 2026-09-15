"""Append-only, timestamped log of every `seed rerank` invocation for one model."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from seed_pipeline.config.paths import rerank_log_path


def _local_now() -> datetime:
    return datetime.now().astimezone()


class RerankLog:
    """One timestamped line per event, kept under data/work so it survives reboots."""

    def __init__(
        self,
        path: Path,
        *,
        echo: TextIO | None = None,
        now: Callable[[], datetime] = _local_now,
    ) -> None:
        self.path = Path(path)
        self._echo = echo
        self._now = now

    def __call__(self, message: str) -> None:
        stamp = self._now().isoformat(timespec="seconds")
        lines = [f"{stamp} {part}" for part in message.splitlines() or [""]]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(f"{line}\n" for line in lines)
        if self._echo is not None:
            for line in lines:
                print(line, file=self._echo, flush=True)


def open_rerank_log(model: str, *, echo: bool) -> RerankLog:
    return RerankLog(rerank_log_path(model), echo=sys.stderr if echo else None)
