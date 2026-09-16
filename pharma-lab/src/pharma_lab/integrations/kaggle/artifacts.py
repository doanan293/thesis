from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharma_lab.integrations.kaggle.errors import KagglePipelineError
from pharma_lab.integrations.kaggle.models import (
    CloudArtifact,
    Completion,
    JobIdentity,
    reuse_payload_of,
)
from pharma_lab.runtime.runtime_profiles import canonical_sha256


class ArtifactContractError(KagglePipelineError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactContractError(f"Cannot read artifact file: {path}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class CloudArtifactManifest:
    schema_version: int
    artifact_type: str
    created_at: str
    data_filename: str
    data_sha256: str
    record_count: int
    identity: dict[str, Any]
    total: int
    complete: int
    missing: int
    reuse_sha256: str | None = None
    checkpoint_filename: str | None = None
    checkpoint_sha256: str | None = None
    runtime: dict[str, Any] | None = None

    @property
    def completion(self) -> Completion:
        return Completion(self.total, self.complete, self.missing)


def _read_jsonl_count(path: Path) -> int:
    count = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ArtifactContractError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ArtifactContractError(
                        f"JSONL record must be an object at {path}:{line_number}"
                    )
                count += 1
    except OSError as exc:
        raise ArtifactContractError(f"Cannot open artifact file: {path}") from exc
    return count


def _load_manifest(path: Path) -> CloudArtifactManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid artifact manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError("Artifact manifest must be an object")
    required = {
        "schema_version",
        "artifact_type",
        "created_at",
        "data_filename",
        "data_sha256",
        "record_count",
        "identity",
        "total",
        "complete",
        "missing",
    }
    missing = sorted(required - payload.keys())
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
    return CloudArtifactManifest(
        schema_version=1,
        artifact_type=str(payload["artifact_type"]),
        created_at=str(payload["created_at"]),
        data_filename=str(payload["data_filename"]),
        data_sha256=str(payload["data_sha256"]),
        record_count=int(payload["record_count"]),
        identity=dict(payload["identity"]),
        total=int(payload["total"]),
        complete=int(payload["complete"]),
        missing=int(payload["missing"]),
        reuse_sha256=(
            str(payload["reuse_sha256"])
            if payload.get("reuse_sha256") is not None
            else None
        ),
        checkpoint_filename=(
            str(payload["checkpoint_filename"])
            if payload.get("checkpoint_filename") is not None
            else None
        ),
        checkpoint_sha256=(
            str(payload["checkpoint_sha256"])
            if payload.get("checkpoint_sha256") is not None
            else None
        ),
        runtime=(
            dict(payload["runtime"])
            if isinstance(payload.get("runtime"), dict)
            else None
        ),
    )


def load_cloud_artifact(
    data_path: Path,
    manifest_path: Path,
    expected_identity: JobIdentity,
    *,
    allow_partial: bool = False,
    allow_reuse: bool = False,
    expected_artifact_type: str | None = None,
) -> CloudArtifact:
    data_path, manifest_path = Path(data_path), Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    if (
        expected_artifact_type is not None
        and manifest.artifact_type != expected_artifact_type
    ):
        raise ArtifactContractError(
            "Cloud artifact type mismatch: "
            f"{manifest.artifact_type} != {expected_artifact_type}"
        )
    if manifest.data_filename != data_path.name:
        raise ArtifactContractError("Manifest data filename mismatch")
    if sha256_file(data_path) != manifest.data_sha256:
        raise ArtifactContractError("Cloud artifact data checksum mismatch")
    if _read_jsonl_count(data_path) != manifest.record_count:
        raise ArtifactContractError("Cloud artifact record count mismatch")
    identity_payload = dict(manifest.identity)
    manifest_job_sha256 = str(identity_payload.pop("job_sha256", ""))
    if identity_payload:
        derived_job_sha256 = canonical_sha256(identity_payload)
        derived_reuse_sha256 = canonical_sha256(reuse_payload_of(identity_payload))
        if manifest_job_sha256 != derived_job_sha256:
            raise ArtifactContractError("Cloud artifact identity hash mismatch")
        if (
            manifest.reuse_sha256 is not None
            and manifest.reuse_sha256 != derived_reuse_sha256
        ):
            raise ArtifactContractError("Cloud artifact reuse identity hash mismatch")
    else:
        derived_job_sha256 = manifest_job_sha256
        derived_reuse_sha256 = manifest.reuse_sha256
    strict_identity_match = derived_job_sha256 == expected_identity.sha256
    if not strict_identity_match and (
        not allow_reuse
        or derived_reuse_sha256 is None
        or derived_reuse_sha256 != expected_identity.reuse_sha256
    ):
        raise ArtifactContractError("Cloud artifact job identity mismatch")
    checkpoint_path = None
    if manifest.checkpoint_filename is not None:
        checkpoint_path = manifest_path.with_name(manifest.checkpoint_filename)
        if not checkpoint_path.is_file():
            raise ArtifactContractError(
                f"Cloud artifact checkpoint is missing: {checkpoint_path}"
            )
        if (
            manifest.checkpoint_sha256 is None
            or sha256_file(checkpoint_path) != manifest.checkpoint_sha256
        ):
            raise ArtifactContractError("Cloud artifact checkpoint checksum mismatch")
    if not allow_partial and not manifest.completion.is_complete:
        raise ArtifactContractError(
            f"Cannot load incomplete artifact: complete={manifest.complete}, total={manifest.total}, missing={manifest.missing}"
        )
    diagnostic_paths = [
        candidate
        for candidate in (
            manifest_path.with_name("telemetry.json"),
            manifest_path.with_name("benchmark_report.md"),
            *sorted(manifest_path.parent.glob("server-*.log")),
        )
        if candidate.is_file()
    ]
    return CloudArtifact(
        data_path,
        manifest_path,
        expected_identity,
        manifest.completion,
        checkpoint_path=checkpoint_path,
        strict_identity_match=strict_identity_match,
        producing_job_sha256=str(manifest.identity.get("job_sha256"))
        if manifest.identity.get("job_sha256") is not None
        else None,
        diagnostic_paths=tuple(diagnostic_paths),
    )


def promote_complete_artifact(artifact: CloudArtifact, destination: Path) -> Path:
    if not artifact.completion.is_complete:
        raise ArtifactContractError("Cannot promote incomplete artifact")
    load_cloud_artifact(artifact.data_path, artifact.manifest_path, artifact.identity)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="artifact-promote-", dir=str(destination.parent)
    ) as raw:
        staged = Path(raw) / destination.name
        shutil.copy2(artifact.data_path, staged)
        os.replace(staged, destination)
        staged_manifest = Path(raw) / "manifest.json"
        shutil.copy2(artifact.manifest_path, staged_manifest)
        os.replace(staged_manifest, destination.with_name("manifest.json"))
        for diagnostic in artifact.diagnostic_paths:
            staged_diagnostic = Path(raw) / diagnostic.name
            shutil.copy2(diagnostic, staged_diagnostic)
            os.replace(staged_diagnostic, destination.with_name(diagnostic.name))
    return destination
