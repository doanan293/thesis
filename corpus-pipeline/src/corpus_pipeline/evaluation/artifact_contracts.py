from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}


class ArtifactContractError(RuntimeError):
    """Raised when an evaluation artifact violates its contract."""


def sha256_file(path: Path) -> str:
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stat.st_size >= 1024 * 1024:
            cached = _DIGEST_CACHE.get(key)
            if cached is not None:
                return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        if stat.st_size >= 1024 * 1024:
            _DIGEST_CACHE[key] = value
        return value
    except OSError as exc:
        raise ArtifactContractError(f"Cannot read artifact file: {path}") from exc


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactContractError(f"Artifact file is missing: {path}")
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ArtifactContractError(f"Cannot open artifact file: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ArtifactContractError(
                    f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ArtifactContractError(
                    f"JSONL record must be a JSON object at {path}:{line_number}"
                )
            yield record


@dataclass(frozen=True)
class Completion:
    total: int
    complete: int
    missing: int

    def __post_init__(self) -> None:
        if min(self.total, self.complete, self.missing) < 0:
            raise ArtifactContractError("Completion counts must be non-negative")
        if self.complete + self.missing != self.total:
            raise ArtifactContractError(
                "Completion counts must satisfy complete + missing == total"
            )

    @property
    def is_complete(self) -> bool:
        return self.total == self.complete and self.missing == 0


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    artifact_type: str
    created_at: str
    data_filename: str
    data_sha256: str
    record_count: int
    identity: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        data_path: Path,
        record_count: int,
        identity: dict[str, Any],
    ) -> ArtifactManifest:
        data_path = Path(data_path)
        if record_count < 0:
            raise ArtifactContractError("record_count must be non-negative")
        return cls(
            schema_version=1,
            artifact_type=str(artifact_type),
            created_at=datetime.now(UTC).isoformat(),
            data_filename=data_path.name,
            data_sha256=sha256_file(data_path),
            record_count=int(record_count),
            identity=dict(identity),
        )

    @classmethod
    def create_from_digest(
        cls,
        *,
        artifact_type: str,
        data_path: Path,
        record_count: int,
        identity: dict[str, Any],
        data_sha256: str,
    ) -> ArtifactManifest:
        if record_count < 0:
            raise ArtifactContractError("record_count must be non-negative")
        return cls(
            schema_version=1,
            artifact_type=str(artifact_type),
            created_at=datetime.now(UTC).isoformat(),
            data_filename=Path(data_path).name,
            data_sha256=data_sha256,
            record_count=int(record_count),
            identity=dict(identity),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        required = {
            "schema_version",
            "artifact_type",
            "created_at",
            "data_filename",
            "data_sha256",
            "record_count",
            "identity",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ArtifactContractError(
                f"Manifest is missing required fields: {', '.join(missing)}"
            )
        if int(payload["schema_version"]) != 1:
            raise ArtifactContractError(
                f"Unsupported artifact manifest schema: {payload['schema_version']}"
            )
        if not isinstance(payload["identity"], dict):
            raise ArtifactContractError("Manifest identity must be an object")
        try:
            record_count = int(payload["record_count"])
        except (TypeError, ValueError) as exc:
            raise ArtifactContractError(
                "Manifest record_count must be an integer"
            ) from exc
        return cls(
            schema_version=1,
            artifact_type=str(payload["artifact_type"]),
            created_at=str(payload["created_at"]),
            data_filename=str(payload["data_filename"]),
            data_sha256=str(payload["data_sha256"]),
            record_count=record_count,
            identity=dict(payload["identity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> ArtifactManifest:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid artifact manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"Artifact manifest must be an object: {path}")
    return ArtifactManifest.from_dict(payload)


def validate_data_file(path: Path, manifest: ArtifactManifest | dict[str, Any]) -> int:
    path = Path(path)
    parsed = (
        manifest
        if isinstance(manifest, ArtifactManifest)
        else ArtifactManifest.from_dict(manifest)
    )
    if parsed.data_filename != path.name:
        raise ArtifactContractError(
            f"Manifest data filename mismatch: {parsed.data_filename} != {path.name}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != parsed.data_sha256:
        raise ArtifactContractError(
            f"Artifact checksum mismatch for {path}: {actual_hash} != {parsed.data_sha256}"
        )
    count = 0
    for _record in iter_jsonl_objects(path):
        count += 1
    if count != parsed.record_count:
        raise ArtifactContractError(
            f"Artifact record count mismatch for {path}: {count} != {parsed.record_count}"
        )
    return count


def require_finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ArtifactContractError(f"{field} must be finite")
    return number


def atomic_promote(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise ArtifactContractError(f"Cannot promote missing artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    except OSError as exc:
        raise ArtifactContractError(
            f"Unable to atomically promote {source} to {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination
