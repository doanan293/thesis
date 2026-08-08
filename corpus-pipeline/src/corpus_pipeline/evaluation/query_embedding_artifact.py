from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    Completion,
    canonical_sha256,
    iter_jsonl_objects,
    sha256_file,
    write_json,
)
from corpus_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
    query_hash,
    validate_embedding,
)


@dataclass(frozen=True)
class QuerySnapshot:
    data_path: Path
    manifest_path: Path
    manifest: ArtifactManifest
    row_count: int


@dataclass(frozen=True)
class QueryCacheArtifact:
    data_path: Path
    manifest_path: Path
    manifest: ArtifactManifest
    completion: Completion

    @property
    def root(self) -> Path:
        return self.data_path.parent


def migrate_query_bundle(
    source: Path,
    destination: QueryEmbeddingCache,
    *,
    model: str,
    vector_dim: int,
    model_sha256: str,
) -> int:
    """Merge a legacy query JSONL file or bundle into a flat cache."""
    source = Path(source)
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise QueryEmbeddingCacheError(
                f"Legacy query bundle manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = source / str(manifest["data_filename"])
    if not source.is_file():
        raise QueryEmbeddingCacheError(f"Legacy query cache is missing: {source}")
    migrated = 0
    for line_number, record in enumerate(iter_jsonl_objects(source), start=1):
        try:
            destination.set(
                model,
                str(record["query_id"]),
                str(record["query"]),
                validate_embedding(record["embedding"], source, line_number),
            )
        except (KeyError, QueryEmbeddingCacheError) as exc:
            raise QueryEmbeddingCacheError(
                f"Invalid legacy query record at {source}:{line_number}"
            ) from exc
        migrated += 1
    if destination.vector_dim != vector_dim:
        raise QueryEmbeddingCacheError(
            f"Migrated query dimension mismatch: {destination.vector_dim} != {vector_dim}"
        )
    if destination.model_sha256 not in ("", model_sha256):
        raise QueryEmbeddingCacheError("Migrated query model digest mismatch")
    return migrated


def _query_key(model: str, row: dict[str, Any]) -> tuple[str, str, str]:
    query_id = str(row.get("query_id") or "")
    query_text = str(row.get("query") or "")
    if not query_id or not query_text.strip():
        raise ArtifactContractError(
            "Every evaluation query requires query_id and query"
        )
    return str(model), query_id, query_hash(query_text)


def _query_rows(eval_path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl_objects(Path(eval_path)))
    if not rows:
        raise ArtifactContractError(f"Evaluation query snapshot is empty: {eval_path}")
    keys = [_query_key("snapshot", row)[1:] for row in rows]
    if len(set(keys)) != len(keys):
        raise ArtifactContractError(
            "Evaluation query snapshot contains duplicate query keys"
        )
    return rows


def build_query_snapshot(eval_path: Path, output_dir: Path) -> QuerySnapshot:
    eval_path = Path(eval_path)
    rows = _query_rows(eval_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "queries.jsonl"
    shutil.copy2(eval_path, data_path)
    manifest = ArtifactManifest.create(
        artifact_type="query_snapshot",
        data_path=data_path,
        record_count=len(rows),
        identity={"eval_sha256": sha256_file(data_path)},
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest.to_dict())
    return QuerySnapshot(data_path, manifest_path, manifest, len(rows))


def query_cache_identity(
    eval_path: Path,
    *,
    model: str,
    gguf_sha256: str,
    vector_dimension: int,
) -> dict[str, Any]:
    if vector_dimension < 1:
        raise ArtifactContractError("vector_dimension must be positive")
    return {
        "eval_sha256": sha256_file(Path(eval_path)),
        "model": str(model),
        "gguf_sha256": str(gguf_sha256),
        "vector_dimension": int(vector_dimension),
        "cache_schema": "query-embedding-v1",
    }


def inspect_query_cache(
    *,
    eval_path: Path,
    cache_path: Path,
    model: str,
    vector_dimension: int,
) -> Completion:
    rows = _query_rows(Path(eval_path))
    expected = {_query_key(model, row) for row in rows}
    cache = QueryEmbeddingCache(Path(cache_path))
    complete = 0
    for key in expected:
        vector = cache.records.get(key)
        if vector is None:
            continue
        if len(vector) != vector_dimension:
            raise ArtifactContractError(
                f"Query embedding dimension mismatch for {key[1]}: "
                f"{len(vector)} != {vector_dimension}"
            )
        complete += 1
    return Completion(len(expected), complete, len(expected) - complete)


def finalize_query_cache(
    *,
    eval_path: Path,
    cache_path: Path,
    model: str,
    gguf_sha256: str,
    vector_dimension: int,
    output_dir: Path,
    require_complete: bool = False,
    job_sha256: str | None = None,
) -> QueryCacheArtifact:
    rows = _query_rows(Path(eval_path))
    expected = {_query_key(model, row) for row in rows}
    cache = QueryEmbeddingCache(Path(cache_path))
    completion = inspect_query_cache(
        eval_path=eval_path,
        cache_path=cache_path,
        model=model,
        vector_dimension=vector_dimension,
    )
    if require_complete and not completion.is_complete:
        raise QueryEmbeddingCacheError(
            f"Query embedding cache is missing {completion.missing} records"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "query_embeddings.jsonl"
    cache.compact_to(data_path, expected)
    identity = query_cache_identity(
        eval_path,
        model=model,
        gguf_sha256=gguf_sha256,
        vector_dimension=vector_dimension,
    )
    identity["logical_sha256"] = canonical_sha256(identity)
    if job_sha256 is not None:
        identity["job_sha256"] = str(job_sha256)
    manifest = ArtifactManifest.create(
        artifact_type="query_embeddings",
        data_path=data_path,
        record_count=completion.complete,
        identity=identity,
    )
    manifest_payload = manifest.to_dict()
    manifest_payload["total"] = completion.total
    manifest_payload["complete"] = completion.complete
    manifest_payload["missing"] = completion.missing
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest_payload)
    return QueryCacheArtifact(data_path, manifest_path, manifest, completion)
