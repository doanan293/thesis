from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class CheckpointIdentityError(ValueError):
    """Raised when a checkpoint belongs to another job identity."""


class CheckpointStore:
    def __init__(
        self, data_path: Path, identity: dict[str, Any], key_fn: Callable[[dict], str]
    ):
        self.data_path = Path(data_path)
        self.identity = dict(identity)
        self.key_fn = key_fn
        self.identity_path = self.data_path.with_suffix(
            self.data_path.suffix + ".identity.json"
        )
        self.records: dict[str, dict] = {}
        self._load()

    @classmethod
    def open(
        cls, data_path: Path, identity: dict[str, Any], key_fn: Callable[[dict], str]
    ):
        return cls(data_path, identity, key_fn)

    def _load(self) -> None:
        if self.identity_path.is_file():
            stored = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if stored != self.identity:
                raise CheckpointIdentityError(
                    "checkpoint identity does not match current job"
                )
        if self.data_path.is_file():
            for line in self.data_path.open(encoding="utf-8"):
                if line.strip():
                    record = json.loads(line)
                    self.records[self.key_fn(record)] = record

    def missing(self, records: Iterable[dict]) -> list[dict]:
        return [record for record in records if self.key_fn(record) not in self.records]

    def append_batch(self, records: Iterable[dict]) -> None:
        for record in records:
            self.records[self.key_fn(record)] = dict(record)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for record in self.records.values():
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.data_path)
        identity_temp = self.identity_path.with_suffix(
            self.identity_path.suffix + ".tmp"
        )
        identity_temp.write_text(
            json.dumps(self.identity, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(identity_temp, self.identity_path)

    def append(self, record: dict) -> None:
        self.append_batch([record])

    @property
    def complete(self) -> int:
        return len(self.records)
