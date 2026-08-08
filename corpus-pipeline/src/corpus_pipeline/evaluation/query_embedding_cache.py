from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import corpus_pipeline.evaluation.query_hash as _query_hash
from corpus_pipeline.cache.jsonl_records import (
    CacheRecordError,
    ValidatedSubset,
    append_record,
    load_records,
    rewrite_records,
    seal_record,
)
from corpus_pipeline.cache.jsonl_records import (
    subset_sha256 as record_subset_sha256,
)
from corpus_pipeline.config.paths import QUERY_EMBEDDING_CACHE_DIR

model_slug = _query_hash.model_slug
normalize_query_for_hash = _query_hash.normalize_query_for_hash
query_hash = _query_hash.query_hash


class QueryEmbeddingCacheError(RuntimeError):
    pass


def default_query_embedding_cache_path(eval_path: Path, model_name: str) -> Path:
    del eval_path
    return QUERY_EMBEDDING_CACHE_DIR / f"{model_slug(model_name)}.jsonl"


def validate_embedding(
    value, path: Path | None = None, line_number: int | None = None
) -> list[float]:
    location = ""
    if path is not None and line_number is not None:
        location = f" at {path}:{line_number}"
    if not isinstance(value, list) or not value:
        raise QueryEmbeddingCacheError(f"embedding must be a non-empty list{location}")
    vector: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise QueryEmbeddingCacheError(
                f"embedding values must be numeric{location}"
            )
        vector.append(float(item))
    return vector


@dataclass
class QueryEmbeddingCache:
    path: Path
    vector_dim: int | None = None
    model_sha256: str = ""

    def __post_init__(self):
        self.path = Path(self.path)
        self.model_sha256 = str(self.model_sha256)
        self.records: dict[tuple[str, str, str], list[float]] = {}
        self.record_metadata: dict[tuple[str, str, str], dict] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            records, has_legacy = load_records(self.path, allow_legacy=True)
        except CacheRecordError as exc:
            raise QueryEmbeddingCacheError(str(exc)) from exc
        migrated = False
        normalized_records: list[dict] = []
        for line_number, record in enumerate(records, start=1):
            normalized = self._normalize_record(record, line_number)
            if "record_sha256" not in record or "cache_schema" not in record:
                normalized = seal_record(normalized, "query-embedding-v2")
                migrated = True
            normalized_records.append(normalized)
            self._load_record(normalized, line_number)
        if has_legacy or migrated:
            rewrite_records(self.path, normalized_records)

    def _normalize_record(self, record: dict, line_number: int) -> dict:
        for field in ("model", "query_id", "embedding"):
            if field not in record:
                raise QueryEmbeddingCacheError(
                    f"Cache record missing required field '{field}' in {self.path}:{line_number}"
                )
        query = str(record.get("query") or "")
        if not query.strip():
            raise QueryEmbeddingCacheError(
                f"Cache record requires query text in {self.path}:{line_number}"
            )
        vector = validate_embedding(record["embedding"], self.path, line_number)
        dimension = int(record.get("vector_dim") or len(vector))
        if self.vector_dim is None:
            self.vector_dim = dimension
        if dimension != self.vector_dim or len(vector) != self.vector_dim:
            raise QueryEmbeddingCacheError(
                f"Query embedding dimension mismatch at {self.path}:{line_number}: "
                f"{len(vector)} != {self.vector_dim}"
            )
        record_model_sha = str(record.get("model_sha256") or "")
        if self.model_sha256 and record_model_sha not in ("", self.model_sha256):
            raise QueryEmbeddingCacheError(
                f"Model digest mismatch in {self.path}:{line_number}"
            )
        normalized = {
            **record,
            "model": str(record["model"]),
            "model_sha256": self.model_sha256 or record_model_sha,
            "query_id": str(record["query_id"]),
            "query_hash": query_hash(query),
            "query": query,
            "vector_dim": self.vector_dim,
            "embedding": vector,
        }
        return normalized

    def _load_record(self, record: dict, line_number: int) -> None:
        if record.get("query_hash") != query_hash(str(record.get("query") or "")):
            raise QueryEmbeddingCacheError(
                f"Query hash mismatch in {self.path}:{line_number}"
            )
        vector = validate_embedding(record["embedding"], self.path, line_number)
        key = (str(record["model"]), str(record["query_id"]), str(record["query_hash"]))
        # Append-only checkpoints may contain a newer replacement for a key;
        # the last validated record is authoritative.
        self.records[key] = vector
        self.record_metadata[key] = dict(record)

    def prune_to_queries(self, model: str, rows: list[dict]) -> dict:
        allowed_keys = {
            (
                str(model),
                str(row.get("query_id") or ""),
                query_hash(str(row.get("query") or "")),
            )
            for row in rows
        }
        kept = {
            key: vector for key, vector in self.records.items() if key in allowed_keys
        }
        removed = len(self.records) - len(kept)
        if removed == 0:
            return {"kept": len(kept), "removed": 0}

        self.records = kept
        self.record_metadata = {key: self.record_metadata[key] for key in kept}
        if self.records:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                for model_name, query_id, query_hash_value in sorted(self.records):
                    key = (model_name, query_id, query_hash_value)
                    record = dict(self.record_metadata[key])
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif self.path.exists():
            self.path.write_text("", encoding="utf-8")
        return {"kept": len(kept), "removed": removed}

    def get(self, model: str, query_id: str, query_text: str) -> list[float] | None:
        key = (str(model), str(query_id), query_hash(query_text))
        vector = self.records.get(key)
        if vector is None:
            return None
        return list(vector)

    def set(
        self, model: str, query_id: str, query_text: str, embedding: list[float]
    ) -> list[float]:
        vector = validate_embedding(embedding)
        if self.vector_dim is None:
            self.vector_dim = len(vector)
        if len(vector) != self.vector_dim:
            raise QueryEmbeddingCacheError(
                f"Query embedding dimension mismatch: {len(vector)} != {self.vector_dim}"
            )
        key = (str(model), str(query_id), query_hash(query_text))
        record = {
            "model": str(model),
            "model_sha256": self.model_sha256,
            "query_id": str(query_id),
            "query_hash": key[2],
            "query": str(query_text),
            "vector_dim": self.vector_dim,
            "embedding": vector,
            "created_at": datetime.now(UTC).isoformat(),
        }
        record = append_record(self.path, record, schema="query-embedding-v2")
        self.records[key] = vector
        self.record_metadata[key] = record
        return list(vector)

    @staticmethod
    def default_path(model_name: str) -> Path:
        return QUERY_EMBEDDING_CACHE_DIR / f"{model_slug(model_name)}.jsonl"

    def replace_keys(self, keys: set[tuple[str, str, str]]) -> None:
        kept = [
            key_record
            for key, key_record in self.record_metadata.items()
            if key not in keys
        ]
        rewrite_records(
            self.path, sorted(kept, key=lambda item: json.dumps(item, sort_keys=True))
        )
        self.records.clear()
        self.record_metadata.clear()
        self._load()

    def validate_subset(self, rows: list[dict], model: str) -> ValidatedSubset:
        expected = {
            (
                str(model),
                str(row.get("query_id") or ""),
                query_hash(str(row.get("query") or "")),
            )
            for row in rows
        }
        found = [
            self.record_metadata[key]
            for key in sorted(expected)
            if key in self.record_metadata
        ]
        for record in found:
            if record.get("model") != str(model):
                raise QueryEmbeddingCacheError(
                    "Query cache contains an unexpected model"
                )
        missing = len(expected) - len(found)
        return ValidatedSubset(
            total=len(expected),
            complete=len(found),
            missing=missing,
            sha256=record_subset_sha256(found) if not missing else None,
        )

    def subset_sha256(self, rows: list[dict], model: str) -> str:
        subset = self.validate_subset(rows, model)
        if not subset.is_complete:
            raise QueryEmbeddingCacheError(
                f"Query embedding cache is missing {subset.missing} records"
            )
        return str(subset.sha256)

    def compact_to(
        self,
        destination: Path,
        allowed_keys: set[tuple[str, str, str]],
    ) -> Path:
        allowed_models = {item[0] for item in allowed_keys}
        unexpected = {
            key
            for key in self.records
            if key[0] in allowed_models and key not in allowed_keys
        }
        if unexpected:
            raise QueryEmbeddingCacheError(
                f"Query cache contains {len(unexpected)} unexpected keys"
            )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for key in sorted(allowed_keys):
                record = self.record_metadata.get(key)
                if record is None:
                    continue
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        return destination

    def get_or_embed(
        self,
        model: str,
        query_id: str,
        query_text: str,
        embed_fn: Callable[[str], list[float]],
    ) -> list[float]:
        cached = self.get(model, query_id, query_text)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        embedding = embed_fn(query_text)
        return self.set(model, query_id, query_text, embedding)
