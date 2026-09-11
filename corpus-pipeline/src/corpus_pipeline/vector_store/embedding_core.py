from __future__ import annotations

import builtins
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from corpus_pipeline.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    load_records,
    rewrite_records,
    seal_record,
)
from corpus_pipeline.cache.jsonl_records import (
    subset_sha256 as record_subset_sha256,
)
from corpus_pipeline.corpus.metadata.qdrant_payload_contract import (
    compact_validated_runtime_payload,
)
from corpus_pipeline.runtime.client import LlamaCppClient, LlamaCppRequestError

QDRANT_POINT_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "corpus-pipeline:qdrant-point-id:v1"
)


class VectorInputError(ValueError):
    """Raised when canonical vector input is unavailable or malformed."""


DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"


@dataclass(frozen=True)
class QdrantDependencies:
    document_type: type
    point_struct_type: type
    helper_type: type


def load_qdrant_dependencies() -> QdrantDependencies:
    from qdrant_client.models import Document, PointStruct

    from corpus_pipeline.vector_store.qdrant_client_helper import QdrantClientHelper

    return QdrantDependencies(
        document_type=Document,
        point_struct_type=PointStruct,
        helper_type=QdrantClientHelper,
    )


@dataclass(frozen=True)
class EmbeddingCollectionResult:
    embeddings_by_cache_key: dict[int, list[float]] | None
    cache_hits: int
    embedded_count: int
    stopped_early: bool = False


@dataclass
class EmbeddingProgressReporter:
    total: int
    started_at: float
    clock: Callable[[], float]
    last_reported_processed: int = 0
    next_threshold: int = 100

    def update(
        self,
        processed: int,
        cache_hits: int,
        embedded: int,
        *,
        final: bool = False,
    ) -> None:
        first_batch = self.last_reported_processed == 0 and processed > 0
        crossed_threshold = processed >= self.next_threshold
        if not (first_batch or crossed_threshold or final):
            return
        if processed == self.last_reported_processed:
            return
        while self.next_threshold <= processed:
            self.next_threshold += 100
        elapsed = max(self.clock() - self.started_at, 1e-9)
        percent = 100.0 * processed / self.total if self.total else 100.0
        print(
            f"Embedding progress: {processed}/{self.total} ({percent:.1f}%), "
            f"cache_hits={cache_hits}, embedded={embedded}, "
            f"speed={processed / elapsed:.1f} chunks/s",
            flush=True,
        )
        self.last_reported_processed = processed


@dataclass(frozen=True)
class EmbeddingCacheCompletion:
    total: int
    complete: int
    missing: int

    @property
    def is_complete(self) -> bool:
        return self.missing == 0


def runtime_stop_deadline(
    max_runtime_seconds: float | None,
    stop_margin_seconds: float,
    clock=None,
) -> float | None:
    if max_runtime_seconds is None:
        return None
    clock = clock or time.monotonic
    return clock() + max_runtime_seconds - stop_margin_seconds


def runtime_budget_exhausted(deadline: float | None, clock=None) -> bool:
    clock = clock or time.monotonic
    return deadline is not None and clock() >= deadline


class EmbeddingError(RuntimeError):
    pass


class EmbeddingCacheError(RuntimeError):
    pass


def parse_server_urls(values: list[str] | None) -> list[str]:
    urls: list[str] = []
    for value in values or []:
        for item in value.split(","):
            normalized = item.strip().rstrip("/")
            if normalized and normalized not in urls:
                urls.append(normalized)
    return urls


def model_slug(model_name: str) -> str:
    return (
        model_name.replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def canonical_json_hash(value) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def qdrant_point_id(chunk_id, cache_key: int) -> int | str:
    if isinstance(chunk_id, int) and not isinstance(chunk_id, bool) and chunk_id >= 0:
        return chunk_id

    chunk_id_text = str(chunk_id or "").strip()
    if not chunk_id_text:
        return cache_key
    if chunk_id_text.isdecimal():
        return int(chunk_id_text)

    try:
        return str(uuid.UUID(chunk_id_text))
    except ValueError:
        return str(uuid.uuid5(QDRANT_POINT_ID_NAMESPACE, chunk_id_text))


def validate_cached_embedding(
    value,
    vector_dim: int,
    path: Path | None = None,
    line_number: int | None = None,
) -> list[float]:
    location = (
        f" at {path}:{line_number}"
        if path is not None and line_number is not None
        else ""
    )
    if not isinstance(value, list) or len(value) != vector_dim:
        raise EmbeddingCacheError(
            f"embedding must be a list of {vector_dim} numbers{location}"
        )
    vector: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise EmbeddingCacheError(f"embedding values must be numeric{location}")
        vector.append(float(item))
    return vector


class ChunkEmbeddingCache:
    def __init__(
        self,
        path: Path,
        vector_dim: int,
        allow_truncated_final_record: bool = False,
        model_sha256: str = "",
    ):
        self.path = Path(path)
        self.vector_dim = vector_dim
        self.allow_truncated_final_record = allow_truncated_final_record
        self.model_sha256 = model_sha256
        self.records: dict[tuple[str, int, str, str, int], list[float]] = {}
        self.content_records: dict[tuple[str, str, str, int], list[float]] = {}
        self.record_data: dict[tuple[str, int, str, str, int], dict] = {}
        self.content_record_data: dict[tuple[str, str, str, int], dict] = {}
        self.skipped_legacy_records = 0
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw_records, has_legacy = load_records(self.path, allow_legacy=True)
        except CacheRecordError as exc:
            raise EmbeddingCacheError(str(exc)) from exc
        migrated_records: list[dict] = []
        migrated = False
        for line_number, record in enumerate(raw_records, start=1):
            if "payload_hash" not in record:
                self.skipped_legacy_records += 1
                continue
            if self.model_sha256 and record.get("model_sha256") not in (
                None,
                "",
                self.model_sha256,
            ):
                raise EmbeddingCacheError(
                    f"Model digest mismatch in {self.path}:{line_number}"
                )
            key = self._record_key(record, line_number)
            vector = validate_cached_embedding(
                record["embedding"],
                self.vector_dim,
                self.path,
                line_number,
            )
            normalized = {
                **record,
                "model": key[0],
                "model_sha256": self.model_sha256,
                "chunk_key": key[1],
                "text_hash": key[2],
                "payload_hash": key[3],
                "vector_dim": key[4],
                "embedding": vector,
            }
            if "record_sha256" not in record or "cache_schema" not in record:
                normalized = seal_record(normalized, "corpus-embedding-v2")
                migrated = True
            migrated_records.append(normalized)
            self.records[key] = vector
            content_key = (key[0], key[2], key[3], key[4])
            self.content_records[content_key] = vector
            self.record_data[key] = normalized
            self.content_record_data[content_key] = normalized
        if has_legacy or migrated:
            rewrite_records(self.path, migrated_records)

    def _record_key(
        self, record: dict, line_number: int
    ) -> tuple[str, int, str, str, int]:
        for field in (
            "model",
            "chunk_key",
            "text_hash",
            "payload_hash",
            "vector_dim",
            "embedding",
        ):
            if field not in record:
                raise EmbeddingCacheError(
                    f"Cache record missing required field '{field}' in {self.path}:{line_number}"
                )
        return (
            str(record["model"]),
            int(record["chunk_key"]),
            str(record["text_hash"]),
            str(record["payload_hash"]),
            int(record["vector_dim"]),
        )

    def get(
        self, model: str, chunk_key: int, text: str, payload_hash_value: str
    ) -> list[float] | None:
        key = (
            model,
            chunk_key,
            text_hash(text),
            payload_hash_value,
            self.vector_dim,
        )
        vector = self.records.get(key)
        if vector is None:
            content_key = (key[0], key[2], key[3], key[4])
            vector = self.content_records.get(content_key)
        if vector is None:
            self.misses += 1
            return None
        self.hits += 1
        return list(vector)

    def set(
        self,
        model: str,
        chunk_key: int,
        text: str,
        payload_hash_value: str,
        embedding: list[float],
    ) -> list[float]:
        vector = validate_cached_embedding(embedding, self.vector_dim)
        key = (
            model,
            chunk_key,
            text_hash(text),
            payload_hash_value,
            self.vector_dim,
        )
        record = {
            "model": key[0],
            "model_sha256": self.model_sha256,
            "chunk_key": key[1],
            "text_hash": key[2],
            "payload_hash": key[3],
            "vector_dim": key[4],
            "embedding": vector,
            "created_at": datetime.now(UTC).isoformat(),
        }
        record = append_record(self.path, record, schema="corpus-embedding-v2")
        self.records[key] = vector
        content_key = (key[0], key[2], key[3], key[4])
        self.content_records[content_key] = vector
        self.record_data[key] = record
        self.content_record_data[content_key] = record
        return list(vector)

    def prune(
        self, valid_keys: builtins.set[tuple[str, int, str, str, int]]
    ) -> tuple[int, int]:
        return len(self.record_data), 0

    def replace_keys(self, keys: builtins.set[tuple[str, int, str, str, int]]) -> None:
        kept = [record for key, record in self.record_data.items() if key not in keys]
        rewrite_records(
            self.path, sorted(kept, key=lambda item: self._record_key(item, 0))
        )
        self.records.clear()
        self.content_records.clear()
        self.record_data.clear()
        self.content_record_data.clear()
        self.skipped_legacy_records = 0
        self._load()

    def expected_records(self, points: list[dict], model: str) -> list[dict]:
        records = []
        for point in points:
            key = (
                model,
                int(point["cache_key"]),
                text_hash(point["embedding_text"]),
                str(point["payload_hash"]),
                self.vector_dim,
            )
            record = self.record_data.get(key)
            if record is None:
                content_key = (key[0], key[2], key[3], key[4])
                record = self.content_record_data.get(content_key)
            if record is not None:
                records.append(record)
        return records

    def subset_sha256(self, points: list[dict], model: str) -> str:
        records = self.expected_records(points, model)
        if len(records) != len(points):
            raise EmbeddingCacheError("Embedding cache is incomplete")
        return record_subset_sha256(records)


def get_embedding(
    text: str,
    model_name: str,
    vector_dimension: int,
    client: LlamaCppClient,
    max_retries: int = 3,
) -> list:
    return client.embed([text], model_name, vector_dimension)[0]


def partition_indexed_texts(
    texts: list[str],
    worker_count: int,
) -> list[list[tuple[int, str]]]:
    partitions: list[list[tuple[int, str]]] = [[] for _ in range(worker_count)]
    for index, value in enumerate(texts):
        partitions[index % worker_count].append((index, value))
    return partitions


def get_embeddings_from_clients(
    texts: list[str],
    model_name: str,
    vector_dimension: int,
    clients: list[LlamaCppClient],
    point_contexts: list[dict] | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if not clients:
        raise EmbeddingError("No llama.cpp embedding server is configured")
    partitions = partition_indexed_texts(texts, min(len(clients), len(texts)))
    ordered: list[list[float] | None] = [None] * len(texts)

    def call_worker(worker_index: int):
        indexed = partitions[worker_index]
        try:
            vectors = clients[worker_index].embed(
                [text for _, text in indexed], model_name, vector_dimension
            )
        except LlamaCppRequestError as exc:
            relative_index = exc.failed_input_index
            if (
                relative_index is not None
                and 0 <= relative_index < len(indexed)
                and point_contexts is not None
            ):
                original_index = indexed[relative_index][0]
                point = point_contexts[original_index]
                chunk_id = point.get("payload", {}).get("chunk_id", "<unknown>")
                raise EmbeddingError(
                    f"Embedding failed for cache_key={point['cache_key']} "
                    f"chunk_id={chunk_id}: {exc}"
                ) from exc
            raise
        return indexed, vectors

    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        results = list(executor.map(call_worker, range(len(partitions))))
    for indexed, vectors in results:
        if len(indexed) != len(vectors):
            raise EmbeddingError("llama.cpp worker returned an unexpected vector count")
        for (index, _), vector in zip(indexed, vectors, strict=False):
            ordered[index] = validate_cached_embedding(vector, vector_dimension)
    if any(vector is None for vector in ordered):
        raise EmbeddingError("Parallel llama.cpp result join left missing vectors")
    return [vector for vector in ordered if vector is not None]


def canonical_metadata_payload(chunk: dict) -> dict:
    chunk_id = chunk.get("chunk_id", "<unknown>")
    return compact_validated_runtime_payload(
        chunk, label=f"Canonical metadata record {chunk_id}"
    )


def chunk_payload_hash(chunk: dict) -> str:
    return canonical_json_hash(canonical_metadata_payload(chunk))


def normalize_canonical_metadata_record(chunk: dict, cache_key: int) -> dict:
    payload = canonical_metadata_payload(chunk)
    chunk_id = payload["chunk_id"]
    embedding_text = payload["embedding_text"]
    return {
        "point_id": qdrant_point_id(chunk_id, cache_key),
        "cache_key": cache_key,
        "embedding_text": embedding_text,
        "payload": payload,
        "payload_hash": canonical_json_hash(payload),
    }


def normalize_input_records(records: list[dict]) -> list[dict]:
    return [
        normalize_canonical_metadata_record(record, cache_key=index)
        for index, record in enumerate(records, start=1)
    ]


def chunked(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_normalized_input_points(input_path: Path) -> list[dict]:
    return list(iter_normalized_input_points(input_path))


def iter_normalized_input_points(input_path: Path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise VectorInputError(f"missing canonical metadata input: {input_path}")
    with input_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorInputError(f"invalid JSON at {input_path}:{index}") from exc
            if not isinstance(record, dict):
                raise VectorInputError(
                    f"record at {input_path}:{index} is not an object"
                )
            yield normalize_canonical_metadata_record(record, cache_key=index)


def expected_cache_keys(
    points: list[dict], model: str, vector_dim: int
) -> set[tuple[str, int, str, str, int]]:
    return {
        (
            model,
            int(point["cache_key"]),
            text_hash(point["embedding_text"]),
            point["payload_hash"],
            vector_dim,
        )
        for point in points
    }


def prune_embedding_cache(
    embedding_cache: ChunkEmbeddingCache | None,
    points: list[dict],
    model: str,
    vector_dim: int,
) -> tuple[int, int]:
    if embedding_cache is None:
        return 0, 0
    kept, removed = embedding_cache.prune(
        expected_cache_keys(points, model, vector_dim)
    )
    print(f"Pruned embedding cache: kept {kept}, removed {removed} stale records.")
    return kept, removed


def detect_cache_vector_dimension(cache_path: Path, model: str) -> int:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise EmbeddingCacheError(f"Embedding cache not found: {cache_path}")

    dimensions: set[int] = set()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingCacheError(
                    f"Invalid JSON in {cache_path}:{line_number}: {exc}"
                ) from exc
            if str(record.get("model")) != model or "payload_hash" not in record:
                continue
            if "vector_dim" not in record:
                raise EmbeddingCacheError(
                    f"Cache record missing required field 'vector_dim' in {cache_path}:{line_number}"
                )
            dimensions.add(int(record["vector_dim"]))

    if not dimensions:
        raise EmbeddingCacheError(
            f"No embedding cache records found for model {model} in {cache_path}"
        )
    if len(dimensions) != 1:
        raise EmbeddingCacheError(
            f"Embedding cache contains multiple vector dimensions for model {model}: {sorted(dimensions)}"
        )
    return next(iter(dimensions))


def collect_complete_cached_embeddings(
    points: list[dict],
    model: str,
    embedding_cache: ChunkEmbeddingCache,
) -> dict[int, list[float]]:
    vectors_by_cache_key: dict[int, list[float]] = {}
    missing: list[dict] = []

    for point in points:
        cache_key = int(point["cache_key"])
        vector = embedding_cache.get(
            model, cache_key, point["embedding_text"], point["payload_hash"]
        )
        if vector is None:
            missing.append(point)
            continue
        vectors_by_cache_key[cache_key] = vector

    if missing:
        samples = ", ".join(
            f"{int(point['cache_key'])}:{point['payload'].get('chunk_id', '<unknown>')}"
            for point in missing[:5]
        )
        raise EmbeddingCacheError(
            f"Missing {len(missing)} embedding cache records for model {model}. "
            f"Sample chunk_key:chunk_id values: {samples}"
        )

    return vectors_by_cache_key


def summarize_embedding_cache_completion(
    points: list[dict],
    model: str,
    embedding_cache: ChunkEmbeddingCache | None,
) -> EmbeddingCacheCompletion:
    complete = 0
    if embedding_cache is not None:
        for point in points:
            vector = embedding_cache.get(
                model,
                int(point["cache_key"]),
                point["embedding_text"],
                point["payload_hash"],
            )
            if vector is not None:
                complete += 1
    total = len(points)
    return EmbeddingCacheCompletion(
        total=total, complete=complete, missing=total - complete
    )


def inspect_embedding_cache_completion(
    input_path: Path,
    cache_path: Path,
    model: str,
) -> EmbeddingCacheCompletion:
    points = read_normalized_input_points(input_path)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return EmbeddingCacheCompletion(
            total=len(points), complete=0, missing=len(points)
        )

    try:
        vector_dim = detect_cache_vector_dimension(cache_path, model)
    except EmbeddingCacheError as exc:
        if "No embedding cache records found" not in str(exc):
            raise
        return EmbeddingCacheCompletion(
            total=len(points), complete=0, missing=len(points)
        )

    embedding_cache = ChunkEmbeddingCache(
        cache_path,
        vector_dim,
        allow_truncated_final_record=True,
    )
    return summarize_embedding_cache_completion(points, model, embedding_cache)


def collect_or_create_embeddings(
    points: list[dict],
    args,
    vector_dim: int,
    embedding_cache: ChunkEmbeddingCache | None,
    progress_clock: Callable[[], float] = time.monotonic,
    retain_embeddings: bool = True,
) -> EmbeddingCollectionResult:
    cache_hits = 0
    embedded_count = 0
    embeddings_by_cache_key: dict[int, list[float]] | None = (
        {} if retain_embeddings else None
    )
    stopped_early = False
    progress = EmbeddingProgressReporter(
        total=len(points),
        started_at=progress_clock(),
        clock=progress_clock,
    )

    pending_batches: list[tuple[list[dict], list[str]]] = []

    def commit_embeddings(
        points_to_embed: list[dict], new_embeddings: list[list[float]]
    ) -> None:
        nonlocal embedded_count
        for point, vector_embedding in zip(
            points_to_embed, new_embeddings, strict=True
        ):
            cache_key = int(point["cache_key"])
            input_text = point["embedding_text"]
            if embedding_cache:
                vector_embedding = embedding_cache.set(
                    args.model,
                    cache_key,
                    input_text,
                    point["payload_hash"],
                    vector_embedding,
                )
            if embeddings_by_cache_key is not None:
                embeddings_by_cache_key[cache_key] = vector_embedding
            embedded_count += 1

        progress.update(
            cache_hits + embedded_count,
            cache_hits,
            embedded_count,
        )

    for batch in chunked(points, args.input_batch_size):
        if runtime_budget_exhausted(getattr(args, "runtime_stop_deadline", None)):
            stopped_early = True
            break

        points_to_embed = []
        texts_to_embed = []

        for point in batch:
            cache_key = int(point["cache_key"])
            input_text = point["embedding_text"]
            cached_embedding = (
                embedding_cache.get(
                    args.model, cache_key, input_text, point["payload_hash"]
                )
                if embedding_cache
                else None
            )
            if cached_embedding is not None:
                if embeddings_by_cache_key is not None:
                    embeddings_by_cache_key[cache_key] = cached_embedding
                cache_hits += 1
                continue
            points_to_embed.append(point)
            texts_to_embed.append(input_text)

        if not points_to_embed:
            continue
        if args.mock:
            new_embeddings = [
                [0.01 * (int(point["cache_key"]) % 100)] * vector_dim
                for point in points_to_embed
            ]
            commit_embeddings(points_to_embed, new_embeddings)
        else:
            pending_batches.append((points_to_embed, texts_to_embed))

    if pending_batches and not runtime_budget_exhausted(
        getattr(args, "runtime_stop_deadline", None)
    ):
        import asyncio

        from corpus_pipeline.integrations.kaggle.workers.scheduling import (
            stream_map_ordered,
        )

        async def operation(client, _index, item):
            batch_points, batch_texts = item
            vectors = await asyncio.to_thread(
                client.embed, batch_texts, args.model, vector_dim
            )
            return batch_points, vectors

        def on_completed(completed):
            for _index, (batch_points, vectors) in completed:
                commit_embeddings(batch_points, vectors)

        scheduled = asyncio.run(
            stream_map_ordered(
                pending_batches,
                args.llama_clients,
                max(1, int(getattr(args, "embedding_concurrency", 1))),
                operation,
                getattr(args, "runtime_stop_deadline", float("inf")) or float("inf"),
                on_completed=on_completed,
            )
        )
        stopped_early = scheduled.stopped_early
    elif pending_batches:
        stopped_early = True

    progress.update(
        cache_hits + embedded_count,
        cache_hits,
        embedded_count,
        final=True,
    )

    return EmbeddingCollectionResult(
        embeddings_by_cache_key=embeddings_by_cache_key,
        cache_hits=cache_hits,
        embedded_count=embedded_count,
        stopped_early=stopped_early,
    )
