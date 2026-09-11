from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROFILE_SCHEMA_VERSION = 3


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_int(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"expected an integer value, got {type(value).__name__}")


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class RuntimeCandidate:
    server_slots: int
    concurrency: int
    request_batch_size: int
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int

    def __post_init__(self) -> None:
        values = (
            self.server_slots,
            self.concurrency,
            self.request_batch_size,
            self.context_per_slot,
            self.logical_batch_size,
            self.physical_batch_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("runtime candidate values must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "server_slots": self.server_slots,
            "concurrency": self.concurrency,
            "request_batch_size": self.request_batch_size,
            "context_per_slot": self.context_per_slot,
            "logical_batch_size": self.logical_batch_size,
            "physical_batch_size": self.physical_batch_size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeCandidate:
        names = (
            "server_slots",
            "concurrency",
            "request_batch_size",
            "context_per_slot",
            "logical_batch_size",
            "physical_batch_size",
        )
        missing = [name for name in names if name not in payload]
        if missing:
            raise ValueError(f"runtime candidate is missing: {', '.join(missing)}")
        try:
            values = {name: _to_int(payload[name]) for name in names}
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime candidate values must be integers") from exc
        return cls(**values)


@dataclass(frozen=True)
class RuntimeSearchSpace:
    candidates: tuple[RuntimeCandidate, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("runtime search space must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("runtime search space contains duplicate candidates")

    @property
    def sha256(self) -> str:
        return canonical_sha256([candidate.to_dict() for candidate in self.candidates])

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class EmbeddingRuntimeSearchSpaces:
    query: RuntimeSearchSpace
    corpus: RuntimeSearchSpace


@dataclass(frozen=True)
class RuntimeProfileIdentity:
    payload: dict[str, object]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        workload: str,
        model: str,
        model_sha256: str,
        runtime_sha256: str,
        inference_cache_policy_sha256: str,
        machine_shape: str,
        topology: str,
        search_space: RuntimeSearchSpace,
    ) -> RuntimeProfileIdentity:
        if not workload.strip() or not model.strip() or not machine_shape.strip():
            raise ValueError("runtime profile identity fields must not be empty")
        payload: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "workload": workload,
            "model": model,
            "model_sha256": _require_sha256(model_sha256, "model_sha256"),
            "runtime_sha256": _require_sha256(runtime_sha256, "runtime_sha256"),
            "inference_cache_policy_sha256": _require_sha256(
                inference_cache_policy_sha256, "inference_cache_policy_sha256"
            ),
            "machine_shape": machine_shape,
            "topology": topology,
            "search_space_sha256": search_space.sha256,
        }
        return cls(payload, canonical_sha256(payload))


@dataclass(frozen=True)
class RuntimeProfile:
    identity: RuntimeProfileIdentity
    selected: RuntimeCandidate
    sample_count: int
    measurements: tuple[dict[str, object], ...]
    benchmark_job_sha256: str
    created_at: str
    checksum: str

    @classmethod
    def create(
        cls,
        identity: RuntimeProfileIdentity,
        selected: RuntimeCandidate,
        *,
        sample_count: int,
        measurements: Sequence[Mapping[str, object]],
        benchmark_job_sha256: str,
    ) -> RuntimeProfile:
        if sample_count < 1:
            raise ValueError("runtime profile sample_count must be positive")
        benchmark_job_sha256 = _require_sha256(
            benchmark_job_sha256, "benchmark_job_sha256"
        )
        normalized = tuple(dict(measurement) for measurement in measurements)
        if not any(
            measurement.get("status") == "ok"
            and measurement.get("candidate") == selected.to_dict()
            for measurement in normalized
        ):
            raise ValueError("selected runtime candidate was not measured successfully")
        created_at = datetime.now(UTC).isoformat()
        body = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "identity": identity.payload,
            "identity_sha256": identity.sha256,
            "selected": selected.to_dict(),
            "sample_count": sample_count,
            "measurements": list(normalized),
            "benchmark_job_sha256": benchmark_job_sha256,
            "created_at": created_at,
        }
        return cls(
            identity,
            selected,
            sample_count,
            normalized,
            benchmark_job_sha256,
            created_at,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "identity": self.identity.payload,
            "identity_sha256": self.identity.sha256,
            "selected": self.selected.to_dict(),
            "sample_count": self.sample_count,
            "measurements": list(self.measurements),
            "benchmark_job_sha256": self.benchmark_job_sha256,
            "created_at": self.created_at,
        }
        return body | {"checksum": self.checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeProfile:
        if _to_int(payload.get("schema_version", -1)) != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported runtime profile schema")
        identity_payload = payload.get("identity")
        if not isinstance(identity_payload, dict):
            raise ValueError("runtime profile identity must be an object")
        identity_sha256 = _require_sha256(
            payload.get("identity_sha256"), "identity_sha256"
        )
        if canonical_sha256(identity_payload) != identity_sha256:
            raise ValueError("runtime profile identity checksum mismatch")
        identity = RuntimeProfileIdentity(identity_payload, identity_sha256)
        selected_payload = payload.get("selected")
        if not isinstance(selected_payload, dict):
            raise ValueError("runtime profile selected candidate must be an object")
        selected = RuntimeCandidate.from_dict(selected_payload)
        measurements_payload = payload.get("measurements")
        if not isinstance(measurements_payload, list):
            raise ValueError("runtime profile measurements must be a list")
        measurements = tuple(
            dict(item) for item in measurements_payload if isinstance(item, dict)
        )
        if len(measurements) != len(measurements_payload):
            raise ValueError("runtime profile measurements must contain objects")
        sample_count = _to_int(payload.get("sample_count", 0))
        benchmark_job_sha256 = _require_sha256(
            payload.get("benchmark_job_sha256"), "benchmark_job_sha256"
        )
        created_at = str(payload.get("created_at", ""))
        checksum = _require_sha256(payload.get("checksum"), "checksum")
        body = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "identity": identity_payload,
            "identity_sha256": identity_sha256,
            "selected": selected.to_dict(),
            "sample_count": sample_count,
            "measurements": list(measurements),
            "benchmark_job_sha256": benchmark_job_sha256,
            "created_at": created_at,
        }
        if canonical_sha256(body) != checksum:
            raise ValueError("runtime profile checksum mismatch")
        return cls(
            identity,
            selected,
            sample_count,
            measurements,
            benchmark_job_sha256,
            created_at,
            checksum,
        )


class RuntimeProfileStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, identity: RuntimeProfileIdentity, *, model_slug: str) -> Path:
        workload = str(identity.payload["workload"])
        return self.root / workload / model_slug / f"{identity.sha256}.json"

    def load(
        self, identity: RuntimeProfileIdentity, *, model_slug: str
    ) -> RuntimeProfile | None:
        path = self.path(identity, model_slug=model_slug)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = RuntimeProfile.from_dict(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if profile.identity != identity:
            return None
        return profile

    def save(self, profile: RuntimeProfile, *, model_slug: str) -> Path:
        destination = self.path(profile.identity, model_slug=model_slug)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            # The handle closes before the atomic replacement below.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    profile.to_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination
