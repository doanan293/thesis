from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKSUM_FIELD = "record_sha256"


class CacheRecordError(ValueError):
    """Raised when a checksummed JSONL cache is invalid."""


@dataclass(frozen=True)
class ValidatedSubset:
    total: int
    complete: int
    missing: int
    sha256: str | None

    @property
    def is_complete(self) -> bool:
        return self.missing == 0


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in record.items() if key != CHECKSUM_FIELD}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CacheRecordError("record contains non-canonical JSON values") from exc
    return encoded.encode("utf-8")


def seal_record(record: Mapping[str, Any], schema: str) -> dict[str, Any]:
    if not schema:
        raise CacheRecordError("cache schema must not be empty")
    sealed = dict(record)
    sealed.pop(CHECKSUM_FIELD, None)
    sealed["cache_schema"] = str(schema)
    sealed[CHECKSUM_FIELD] = hashlib.sha256(_canonical_bytes(sealed)).hexdigest()
    return sealed


def verify_record(
    record: Mapping[str, Any], *, path: Path, line_number: int
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise CacheRecordError(
            f"cache record must be an object at {path}:{line_number}"
        )
    schema = record.get("cache_schema")
    checksum = record.get(CHECKSUM_FIELD)
    if not isinstance(schema, str) or not schema:
        raise CacheRecordError(
            f"cache record missing cache_schema at {path}:{line_number}"
        )
    if not isinstance(checksum, str):
        raise CacheRecordError(
            f"cache record missing {CHECKSUM_FIELD} at {path}:{line_number}"
        )
    expected = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    if not hmac.compare_digest(checksum, expected):
        raise CacheRecordError(f"checksum mismatch at {path}:{line_number}")
    return dict(record)


def _rewrite_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_records(
    path: Path, *, allow_legacy: bool = False
) -> tuple[list[dict[str, Any]], bool]:
    path = Path(path)
    if not path.exists():
        return [], False
    raw_lines = path.read_bytes().splitlines(keepends=True)
    last_nonempty = max(
        (index for index, raw in enumerate(raw_lines) if raw.strip()),
        default=-1,
    )
    records: list[dict[str, Any]] = []
    migrated = False
    for index, raw in enumerate(raw_lines):
        if not raw.strip():
            continue
        line_number = index + 1
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == last_nonempty and not raw.endswith(b"\n"):
                _rewrite_bytes(
                    path,
                    b"".join(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ).encode("utf-8")
                        + b"\n"
                        for item in records
                    ),
                )
                return records, migrated
            raise CacheRecordError(
                f"invalid JSON in {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise CacheRecordError(
                f"cache record must be an object at {path}:{line_number}"
            )
        if CHECKSUM_FIELD not in record or "cache_schema" not in record:
            if not allow_legacy:
                raise CacheRecordError(f"legacy cache record at {path}:{line_number}")
            records.append(record)
            migrated = True
            continue
        records.append(verify_record(record, path=path, line_number=line_number))
    return records, migrated


def _encode_records(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def rewrite_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    _rewrite_bytes(Path(path), _encode_records(records))


def append_record(
    path: Path, record: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    sealed = seal_record(record, schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_encode_records([sealed]))
        handle.flush()
        os.fsync(handle.fileno())
    return sealed


def merge_records(
    path: Path,
    incoming: Iterable[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], Hashable],
    equivalent: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
) -> list[dict[str, Any]]:
    existing, _ = load_records(Path(path))
    by_key: dict[Hashable, dict[str, Any]] = {}
    for record in existing:
        record_key = key(record)
        previous = by_key.get(record_key)
        if previous is not None and not equivalent(previous, record):
            raise CacheRecordError(f"conflicting record for key {record_key!r}")
        by_key[record_key] = record
    for record in incoming:
        verified = verify_record(record, path=Path(path), line_number=0)
        record_key = key(verified)
        previous = by_key.get(record_key)
        if previous is not None and not equivalent(previous, verified):
            raise CacheRecordError(f"conflicting record for key {record_key!r}")
        by_key[record_key] = previous or verified
    merged = sorted(by_key.values(), key=lambda item: _canonical_bytes(item))
    rewrite_records(Path(path), merged)
    return merged


def subset_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    payload = sorted(_canonical_bytes(record) for record in records)
    digest = hashlib.sha256()
    for item in payload:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()
