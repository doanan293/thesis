from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_pipeline.artifacts.io import copy_hash_and_count
from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    load_manifest,
    validate_data_file,
    write_json,
)


@dataclass(frozen=True)
class ArtifactCompletion(Completion):
    total: int
    complete: int
    missing: int


@dataclass(frozen=True)
class ArtifactBundle:
    root: Path
    data_path: Path
    manifest_path: Path
    manifest: ArtifactManifest
    completion: ArtifactCompletion


def _completion_from_manifest(
    manifest_path: Path, manifest: ArtifactManifest
) -> ArtifactCompletion:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    total = int(payload.get("total", manifest.record_count))
    complete = int(payload.get("complete", manifest.record_count))
    missing = int(payload.get("missing", total - complete))
    return ArtifactCompletion(total, complete, missing)


def load_bundle(
    root: Path,
    *,
    expected_type: str,
    require_complete: bool,
) -> ArtifactBundle:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest.artifact_type != expected_type:
        raise ArtifactContractError(
            f"Artifact type mismatch: {manifest.artifact_type} != {expected_type}"
        )
    data_path = root / manifest.data_filename
    validate_data_file(data_path, manifest)
    completion = _completion_from_manifest(manifest_path, manifest)
    if require_complete and not completion.is_complete:
        raise ArtifactContractError(
            "Artifact is incomplete: "
            f"complete={completion.complete}, total={completion.total}, "
            f"missing={completion.missing}"
        )
    return ArtifactBundle(root, data_path, manifest_path, manifest, completion)


def _atomic_copy(source_path: Path, destination_path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source_path, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_bundle(
    output_dir: Path,
    *,
    artifact_type: str,
    source_path: Path,
    identity: dict[str, Any],
    total: int,
    complete: int,
) -> ArtifactBundle:
    if min(total, complete) < 0 or complete > total:
        raise ArtifactContractError("Invalid artifact completion counts")
    source_path = Path(source_path)
    if not source_path.is_file():
        raise ArtifactContractError(f"Artifact source is missing: {source_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / f".{source_path.name}.staged"
    data_sha256, observed_count = copy_hash_and_count(source_path, staged_path)
    if observed_count != complete:
        staged_path.unlink(missing_ok=True)
        raise ArtifactContractError(
            f"Artifact row count mismatch: expected {complete}, observed {observed_count}"
        )
    data_path = output_dir / f"{source_path.stem}.{data_sha256}.jsonl"
    os.replace(staged_path, data_path)
    manifest = ArtifactManifest.create_from_digest(
        artifact_type=artifact_type,
        data_path=data_path,
        record_count=complete,
        identity=identity,
        data_sha256=data_sha256,
    )
    payload = manifest.to_dict()
    payload.update(
        total=total,
        complete=complete,
        missing=total - complete,
    )
    manifest_path = output_dir / "manifest.json"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=output_dir
    )
    os.close(fd)
    temporary_manifest = Path(temporary_name)
    try:
        write_json(temporary_manifest, payload)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    for candidate in output_dir.glob("*.jsonl"):
        if candidate != data_path:
            candidate.unlink(missing_ok=True)
    return load_bundle(
        output_dir,
        expected_type=artifact_type,
        require_complete=False,
    )
