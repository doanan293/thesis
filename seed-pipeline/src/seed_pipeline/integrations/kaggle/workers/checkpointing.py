from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class CheckpointIdentityError(ValueError):
    """Raised when a checkpoint belongs to another job identity."""


class JournalCorruptionError(ValueError):
    """Raised when an inference journal is malformed or inconsistent."""


class AppendOnlyJournal:
    """Crash-tolerant append-only result journal with O(1) key lookup."""

    schema_version = 1

    def __init__(
        self,
        path: Path,
        identity: dict[str, Any],
        key_fn: Callable[[dict], str],
        fingerprint_fn: Callable[[dict], str],
    ):
        self.path = Path(path)
        self.identity = dict(identity)
        self.key_fn = key_fn
        self.fingerprint_fn = fingerprint_fn
        self.records: dict[str, dict] = {}
        self.bytes_appended = 0
        self._load()

    @classmethod
    def open(
        cls,
        path: Path,
        identity: dict[str, Any],
        key_fn: Callable[[dict], str],
        fingerprint_fn: Callable[[dict], str],
        seed_path: Path | None = None,
    ) -> AppendOnlyJournal:
        target = Path(path)
        if not target.exists() and seed_path is not None and Path(seed_path).is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed_path, target)
        return cls(target, identity, key_fn, fingerprint_fn)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            nonempty_lines = [
                line_number
                for line_number, line in enumerate(handle, start=1)
                if line.strip()
            ]
        last_nonempty = nonempty_lines[-1] if nonempty_lines else None
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError as exc:
                    if line_number == last_nonempty:
                        break
                    raise JournalCorruptionError(
                        f"Invalid journal JSON at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(envelope, dict):
                    raise JournalCorruptionError(
                        f"Journal row must be an object at {self.path}:{line_number}"
                    )
                if envelope.get("schema_version") != self.schema_version:
                    raise JournalCorruptionError(
                        f"Unsupported journal schema at {self.path}:{line_number}"
                    )
                if envelope.get("identity") != self.identity:
                    raise JournalCorruptionError(
                        f"Journal identity mismatch at {self.path}:{line_number}"
                    )
                payload = envelope.get("payload")
                key = envelope.get("key")
                fingerprint = envelope.get("fingerprint")
                if not isinstance(payload, dict) or not isinstance(key, str):
                    raise JournalCorruptionError(
                        f"Invalid journal row at {self.path}:{line_number}"
                    )
                if key != self.key_fn(payload) or fingerprint != self.fingerprint_fn(
                    payload
                ):
                    raise JournalCorruptionError(
                        f"Journal key or fingerprint mismatch at {self.path}:{line_number}"
                    )
                self.records[key] = payload

    def _envelope(self, record: dict) -> dict:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "key": self.key_fn(record),
            "fingerprint": self.fingerprint_fn(record),
            "payload": dict(record),
        }

    def append_batch(self, records: Iterable[dict]) -> None:
        materialized = [dict(record) for record in records]
        encoded = [
            json.dumps(self._envelope(record), ensure_ascii=False, sort_keys=True)
            + "\n"
            for record in materialized
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(encoded)
            handle.flush()
        self.bytes_appended += sum(len(item.encode("utf-8")) for item in encoded)
        for record in materialized:
            self.records[self.key_fn(record)] = record

    def reusable(self, current_records: Iterable[dict]) -> dict[str, dict]:
        reusable: dict[str, dict] = {}
        for current in current_records:
            key = self.key_fn(current)
            cached = self.records.get(key)
            if cached is not None and self.fingerprint_fn(
                cached
            ) == self.fingerprint_fn(current):
                reusable[key] = cached
        return reusable

    def compact(self, current_records: Iterable[dict], destination: Path) -> int:
        selected = []
        for current in current_records:
            key = self.key_fn(current)
            cached = self.records.get(key)
            if cached is not None and self.fingerprint_fn(
                cached
            ) == self.fingerprint_fn(current):
                selected.append(cached)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, destination)
        return len(selected)


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
            with self.data_path.open(encoding="utf-8") as handle:
                for line in handle:
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
