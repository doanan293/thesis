from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.integrations.kaggle.config import OwnerConfiguration
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

type JSONValue = (
    bool | int | float | str | Sequence["JSONValue"] | Mapping[str, "JSONValue"] | None
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputFile:
    key: str
    source_path: Path
    filename: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("input key must not be empty")
        if not self.source_path.is_file():
            raise ValueError(f"input file is missing: {self.source_path}")
        if self.filename == "dependency_manifest.json":
            raise ValueError("reserved mounted filename: dependency_manifest.json")
        if not self.filename.strip() or Path(self.filename).name != self.filename:
            raise ValueError(f"invalid mounted input filename: {self.filename!r}")
        actual_digest = _sha256_file(self.source_path)
        if actual_digest != self.sha256:
            raise ValueError(
                f"input file sha256 mismatch for {self.key}: "
                f"{actual_digest} != {self.sha256}"
            )

    @classmethod
    def create(
        cls, key: str, source_path: Path, *, filename: str | None = None
    ) -> InputFile:
        source = Path(source_path)
        if not key.strip():
            raise ValueError("input key must not be empty")
        if not source.is_file():
            raise ValueError(f"input file is missing: {source}")
        mounted_name = source.name if filename is None else filename
        return cls(key, source, mounted_name, _sha256_file(source))

    def descriptor(self) -> dict[str, str]:
        return {"filename": self.filename, "sha256": self.sha256}


@dataclass(frozen=True)
class InputBundle:
    files: tuple[InputFile, ...]
    sha256: str

    @classmethod
    def create(cls, files: Sequence[InputFile]) -> InputBundle:
        items = tuple(files)
        if not items:
            raise ValueError("input bundle must contain at least one file")
        by_key = {item.key: item for item in items}
        if len(by_key) != len(items):
            duplicate = next(
                item.key
                for item in items
                if sum(other.key == item.key for other in items) > 1
            )
            raise ValueError(f"duplicate input key: {duplicate}")
        filenames = [item.filename for item in items]
        if len(set(filenames)) != len(filenames):
            duplicate = next(
                filename for filename in filenames if filenames.count(filename) > 1
            )
            raise ValueError(f"duplicate mounted filename: {duplicate}")
        descriptors = {key: by_key[key].descriptor() for key in sorted(by_key)}
        return cls(
            items,
            canonical_sha256({"schema_version": 1, "files": descriptors}),
        )

    def file(self, key: str) -> InputFile:
        try:
            return next(item for item in self.files if item.key == key)
        except StopIteration as exc:
            raise KeyError(f"input bundle has no logical key {key!r}") from exc

    def descriptors(self) -> dict[str, dict[str, str]]:
        return {
            key: self.file(key).descriptor()
            for key in sorted(item.key for item in self.files)
        }


class StageName(StrEnum):
    CORPUS_EMBED = "corpus-embed"
    QUERY_EMBED = "query-embed"
    RERANK = "rerank"
    RERANK_BENCHMARK = "rerank-benchmark"
    QUERY_EMBED_BENCHMARK = "query-embed-benchmark"
    CORPUS_EMBED_BENCHMARK = "corpus-embed-benchmark"


class KernelPresence(StrEnum):
    ABSENT = "absent"
    EXISTS = "exists"
    UNKNOWN = "unknown"


class KernelStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class KernelRemoteState:
    reference: str
    presence: KernelPresence
    status: KernelStatus | None = None
    detail: str = ""


class ActionVerb(StrEnum):
    REUSE = "reuse"
    ATTACH = "attach"
    DOWNLOAD = "download"
    CHECKPOINT = "checkpoint"
    FINALIZE = "finalize"
    CREATE = "create"
    UPDATE = "update"
    WAIT = "wait"
    RESUME = "resume"
    SUBMIT = "submit"
    SYNC = "sync"


def reuse_payload_of(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Identity fields that decide whether checkpointed records can be reused.

    Inputs are matched record by record and a runtime profile changes only speed, so
    neither belongs to the reuse identity.
    """
    reuse: dict[str, JSONValue] = {
        key: value for key, value in payload.items() if key != "input_sha256"
    }
    runtime_parameters = reuse.get("runtime_parameters")
    if isinstance(runtime_parameters, Mapping):
        reuse["runtime_parameters"] = {
            key: value
            for key, value in runtime_parameters.items()
            if key != "runtime_profile"
        }
    return reuse


@dataclass(frozen=True)
class JobIdentity:
    payload: dict[str, JSONValue]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        stage: StageName,
        contract_version: int,
        model: str,
        model_sha256: str,
        input_sha256: str,
        runtime_parameters: Mapping[str, JSONValue],
    ) -> JobIdentity:
        payload: dict[str, JSONValue] = {
            "stage": stage.value,
            "contract_version": contract_version,
            "model": model,
            "model_sha256": model_sha256,
            "input_sha256": input_sha256,
            "runtime_parameters": dict(runtime_parameters),
        }
        return cls(payload, canonical_sha256(payload))

    @property
    def reuse_payload(self) -> dict[str, JSONValue]:
        return reuse_payload_of(self.payload)

    @property
    def reuse_sha256(self) -> str:
        return canonical_sha256(self.reuse_payload)


@dataclass(frozen=True)
class ReconcileAction:
    resource_kind: str
    reference: str
    verb: ActionVerb
    reason: str


@dataclass(frozen=True)
class StageRequest:
    stage: StageName
    model: str
    input_path: Path
    output_dir: Path
    gguf_root: Path
    owners: OwnerConfiguration
    force: bool = False
    check_only: bool = False
    max_runs: int = 10
    total_budget_seconds: int = 21_600
    benchmark_items: int | None = None
    runtime_profile: RuntimeCandidate | None = None
    # False: ignore account checkpoints and existing kernels (a forced command's
    # later sessions) without republishing dependencies.
    resume_remote: bool = True


@dataclass(frozen=True)
class StageJob:
    stage: StageName
    contract_version: int
    model: str
    identity: JobIdentity
    input_bundle: InputBundle
    output_dir: Path
    local_cache_path: Path
    data_filename: str
    expected_total: int
    worker_module: str
    worker_config: dict[str, JSONValue]


@dataclass(frozen=True)
class CloudArtifact:
    data_path: Path
    manifest_path: Path
    identity: JobIdentity
    completion: Completion
    checkpoint_path: Path | None = None
    strict_identity_match: bool = True
    producing_job_sha256: str | None = None
    diagnostic_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    job: StageJob
    completion: Completion
    actions: tuple[ReconcileAction, ...]
    artifact_path: Path | None
    run_count: int
