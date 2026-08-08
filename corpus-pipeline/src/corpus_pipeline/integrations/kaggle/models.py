from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration

type JSONValue = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StageName(StrEnum):
    CORPUS_EMBED = "corpus-embed"
    QUERY_EMBED = "query-embed"
    RERANK = "rerank"


class ActionVerb(StrEnum):
    REUSE = "reuse"
    CREATE = "create"
    UPDATE = "update"
    WAIT = "wait"
    RESUME = "resume"
    SUBMIT = "submit"
    SYNC = "sync"


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
        return {
            key: value
            for key, value in self.payload.items()
            if key != "input_sha256"
        }

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


@dataclass(frozen=True)
class StageJob:
    stage: StageName
    contract_version: int
    model: str
    identity: JobIdentity
    input_path: Path
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


@dataclass(frozen=True)
class PipelineResult:
    job: StageJob
    completion: Completion
    actions: tuple[ReconcileAction, ...]
    artifact_path: Path | None
    run_count: int
