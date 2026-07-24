from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    pass


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManifestError(f"Manifest must be a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_schema(manifest: dict) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema: {manifest.get('schema_version')}"
        )


def validate_checkpoint_manifest(
    manifest: dict,
    model: str,
    corpus_sha256: str = "",
    vector_dimension: int | None = None,
) -> None:
    require_schema(manifest)
    if manifest.get("model") != model:
        raise ManifestError(
            f"Checkpoint model mismatch: {manifest.get('model')} != {model}"
        )
    if corpus_sha256 and manifest.get("corpus_sha256") != corpus_sha256:
        raise ManifestError(
            "Checkpoint corpus hash mismatch: "
            f"{manifest.get('corpus_sha256')} != {corpus_sha256}"
        )
    total = int(manifest.get("total", -1))
    complete = int(manifest.get("complete", -1))
    missing = int(manifest.get("missing", -1))
    if min(total, complete, missing) < 0 or complete + missing != total:
        raise ManifestError("Checkpoint completion counts are invalid")
    actual_dimension = int(manifest.get("vector_dim", 0))
    if actual_dimension < 1:
        raise ManifestError("Checkpoint vector dimension is invalid")
    if vector_dimension is not None and actual_dimension != vector_dimension:
        raise ManifestError(
            f"Checkpoint vector dimension mismatch: "
            f"{actual_dimension} != {vector_dimension}"
        )
    if not manifest.get("cache_sha256"):
        raise ManifestError("Checkpoint cache hash is missing")


def validate_checkpoint_artifact(
    manifest: dict,
    cache_path: Path,
    *,
    model: str,
    corpus_sha256: str,
    vector_dimension: int,
) -> None:
    validate_checkpoint_manifest(
        manifest,
        model=model,
        corpus_sha256=corpus_sha256,
        vector_dimension=vector_dimension,
    )
    actual = file_sha256(cache_path)
    if actual != manifest["cache_sha256"]:
        raise ManifestError(
            f"Checkpoint cache checksum mismatch: "
            f"{actual} != {manifest['cache_sha256']}"
        )


def build_corpus_manifest(path: Path, row_count: int) -> dict:
    if row_count < 0:
        raise ManifestError("Corpus row count must be non-negative")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "file": Path(path).name,
        "sha256": file_sha256(path),
        "row_count": int(row_count),
    }


def build_checkpoint_manifest(
    model: str,
    vector_dim: int,
    corpus_sha256: str,
    cache_path: Path,
    total: int,
    complete: int,
    missing: int,
    kernel_version: str,
    autotune_profile_path: Path | None = None,
    interruption: str | None = None,
) -> dict:
    if min(total, complete, missing) < 0 or complete + missing != total:
        raise ManifestError("Checkpoint completion counts are inconsistent")
    if vector_dim < 1:
        raise ManifestError("Checkpoint vector dimension must be positive")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": model,
        "vector_dim": int(vector_dim),
        "corpus_sha256": corpus_sha256,
        "cache_file": Path(cache_path).name,
        "cache_sha256": file_sha256(cache_path),
        "total": int(total),
        "complete": int(complete),
        "missing": int(missing),
        "kernel_version": str(kernel_version),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if autotune_profile_path is not None:
        manifest["autotune_profile_file"] = Path(autotune_profile_path).name
        manifest["autotune_profile_sha256"] = file_sha256(autotune_profile_path)
    if interruption:
        manifest["interruption"] = str(interruption)[:2000]
    return manifest
