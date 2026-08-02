from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from corpus_pipeline.config.paths import QUERY_EMBEDDING_CACHE_DIR


class QueryEmbeddingCacheError(RuntimeError):
    pass


def normalize_query_for_hash(query: str) -> str:
    return str(query).strip()


def query_hash(query: str) -> str:
    normalized = normalize_query_for_hash(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def model_slug(model_name: str) -> str:
    return (
        model_name.replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def default_query_embedding_cache_path(eval_path: Path, model_name: str) -> Path:
    return (
        QUERY_EMBEDDING_CACHE_DIR
        / model_slug(model_name)
        / f"{Path(eval_path).stem}.jsonl"
    )


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

    def __post_init__(self):
        self.path = Path(self.path)
        self.records: dict[tuple[str, str, str], list[float]] = {}
        self.record_metadata: dict[tuple[str, str, str], dict] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QueryEmbeddingCacheError(
                        f"Invalid JSON in {self.path}:{line_number}: {exc}"
                    ) from exc
                self._load_record(record, line_number)

    def _load_record(self, record: dict, line_number: int) -> None:
        for field in ("model", "query_id", "query_hash", "embedding"):
            if field not in record:
                raise QueryEmbeddingCacheError(
                    f"Cache record missing required field '{field}' in {self.path}:{line_number}"
                )
        vector = validate_embedding(record["embedding"], self.path, line_number)
        key = (str(record["model"]), str(record["query_id"]), str(record["query_hash"]))
        self.records[key] = vector
        self.record_metadata[key] = {
            "model": key[0],
            "query_id": key[1],
            "query_hash": key[2],
            "query": str(record.get("query") or ""),
            "embedding": vector,
            "created_at": str(record.get("created_at") or ""),
        }

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
        key = (str(model), str(query_id), query_hash(query_text))
        record = {
            "model": str(model),
            "query_id": str(query_id),
            "query_hash": key[2],
            "query": str(query_text),
            "embedding": vector,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[key] = vector
        self.record_metadata[key] = record
        return list(vector)

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
