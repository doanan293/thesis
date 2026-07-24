from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config.paths import PROJECT_ROOT, RAG_FINAL_CHUNKS_PATH, VECTOR_EMBEDDING_CACHE_DIR
from model_runtime.catalog import EMBEDDING_MODELS, ModelKind, require_model
from model_runtime.client import LlamaCppClient, LlamaCppRequestError
from model_runtime.compose import LlamaCppComposeManager, resolve_server
from rag_metadata.qdrant_payload_contract import compact_validated_runtime_payload

CANONICAL_METADATA_PATH = str(RAG_FINAL_CHUNKS_PATH)
VECTOR_DIMENSION = 1024
DEFAULT_QDRANT_BATCH_SIZE = 1000
DEFAULT_REQUEST_TIMEOUT = 900
DEFAULT_COMPOSE_FILE = PROJECT_ROOT.parent / "docker-compose.yml"
DEFAULT_GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
QDRANT_POINT_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "corpus-pipeline:qdrant-point-id:v1"
)
DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"

# Compatibility hooks for focused tests. Production dependencies are imported
# only when an upload path actually touches Qdrant.
Document = None
PointStruct = None
QdrantClientHelper = None


@dataclass(frozen=True)
class QdrantDependencies:
    document_type: type
    point_struct_type: type
    helper_type: type


def load_qdrant_dependencies() -> QdrantDependencies:
    global Document, PointStruct, QdrantClientHelper
    if Document is None or PointStruct is None:
        from qdrant_client.models import Document as QdrantDocument
        from qdrant_client.models import PointStruct as QdrantPointStruct

        Document = QdrantDocument
        PointStruct = QdrantPointStruct
    if QdrantClientHelper is None:
        from vector_store.qdrant_client_helper import QdrantClientHelper as Helper

        QdrantClientHelper = Helper
    return QdrantDependencies(
        document_type=Document,
        point_struct_type=PointStruct,
        helper_type=QdrantClientHelper,
    )


@dataclass(frozen=True)
class EmbeddingCollectionResult:
    embeddings_by_cache_key: dict[int, list[float]]
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


def default_embedding_cache_path(model_name: str) -> Path:
    return VECTOR_EMBEDDING_CACHE_DIR / f"{model_slug(model_name)}.jsonl"


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()


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
        return int(cache_key)
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
        self, path: Path, vector_dim: int, allow_truncated_final_record: bool = False
    ):
        self.path = Path(path)
        self.vector_dim = vector_dim
        self.allow_truncated_final_record = allow_truncated_final_record
        self.records: dict[tuple[str, int, str, str, int], list[float]] = {}
        self.record_data: dict[tuple[str, int, str, str, int], dict] = {}
        self.skipped_legacy_records = 0
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        last_nonempty_line = max(
            (
                line_number
                for line_number, line in enumerate(lines, start=1)
                if line.strip()
            ),
            default=0,
        )
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if (
                    self.allow_truncated_final_record
                    and line_number == last_nonempty_line
                ):
                    self._truncate_after_line(lines, line_number)
                    break
                raise EmbeddingCacheError(
                    f"Invalid JSON in {self.path}:{line_number}: {exc}"
                ) from exc
            if "payload_hash" not in record:
                self.skipped_legacy_records += 1
                continue
            key = self._record_key(record, line_number)
            vector = validate_cached_embedding(
                record["embedding"],
                self.vector_dim,
                self.path,
                line_number,
            )
            self.records[key] = vector
            self.record_data[key] = {
                **record,
                "model": key[0],
                "chunk_key": key[1],
                "text_hash": key[2],
                "payload_hash": key[3],
                "vector_dim": key[4],
                "embedding": vector,
            }

    def _truncate_after_line(self, lines: list[str], bad_line_number: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines[: bad_line_number - 1])
        os.replace(temp_path, self.path)

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
            str(model),
            int(chunk_key),
            text_hash(text),
            str(payload_hash_value),
            self.vector_dim,
        )
        vector = self.records.get(key)
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
            str(model),
            int(chunk_key),
            text_hash(text),
            str(payload_hash_value),
            self.vector_dim,
        )
        record = {
            "model": key[0],
            "chunk_key": key[1],
            "text_hash": key[2],
            "payload_hash": key[3],
            "vector_dim": key[4],
            "embedding": vector,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[key] = vector
        self.record_data[key] = record
        return list(vector)

    def prune(
        self, valid_keys: builtins.set[tuple[str, int, str, str, int]]
    ) -> tuple[int, int]:
        kept_records = {
            key: record for key, record in self.record_data.items() if key in valid_keys
        }
        kept = len(kept_records)
        removed = len(self.record_data) - kept + self.skipped_legacy_records
        if removed == 0 and self.skipped_legacy_records == 0:
            return kept, removed

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in kept_records.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_path, self.path)
        self.record_data = kept_records
        self.records = {
            key: list(record["embedding"]) for key, record in kept_records.items()
        }
        self.skipped_legacy_records = 0
        return kept, removed


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
        "cache_key": int(cache_key),
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
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run build_rag_metadata.py first.")
        sys.exit(1)
    print(f"Reading canonical metadata records from {input_path}...")
    with open(input_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return normalize_input_records(records)


def expected_cache_keys(
    points: list[dict], model: str, vector_dim: int
) -> set[tuple[str, int, str, str, int]]:
    return {
        (
            str(model),
            int(point["cache_key"]),
            text_hash(point["embedding_text"]),
            point["payload_hash"],
            int(vector_dim),
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
            if str(record.get("model")) != str(model) or "payload_hash" not in record:
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
) -> EmbeddingCollectionResult:
    cache_hits = 0
    embedded_count = 0
    embeddings_by_cache_key: dict[int, list[float]] = {}
    stopped_early = False
    progress = EmbeddingProgressReporter(
        total=len(points),
        started_at=progress_clock(),
        clock=progress_clock,
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
                embeddings_by_cache_key[cache_key] = cached_embedding
                cache_hits += 1
                continue
            points_to_embed.append(point)
            texts_to_embed.append(input_text)

        if not points_to_embed:
            new_embeddings = []
        elif args.mock:
            new_embeddings = [
                [0.01 * (int(point["cache_key"]) % 100)] * vector_dim
                for point in points_to_embed
            ]
        else:
            new_embeddings = get_embeddings_from_clients(
                texts_to_embed,
                args.model,
                vector_dim,
                args.llama_clients,
                point_contexts=points_to_embed,
            )

        for point, vector_embedding in zip(
            points_to_embed, new_embeddings, strict=False
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
            embeddings_by_cache_key[cache_key] = vector_embedding
            embedded_count += 1

        progress.update(
            cache_hits + embedded_count,
            cache_hits,
            embedded_count,
        )

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


def build_qdrant_point(
    point: dict,
    vector_embedding: list[float],
    dependencies: QdrantDependencies | None = None,
):
    dependencies = dependencies or load_qdrant_dependencies()
    return dependencies.point_struct_type(
        id=point["point_id"],
        vector={
            DENSE_VECTOR_NAME: vector_embedding,
            BM25_SPARSE_VECTOR_NAME: dependencies.document_type(
                text=point["embedding_text"],
                model=BM25_MODEL_NAME,
            ),
        },
        payload=point["payload"],
    )


def upsert_points_from_embeddings(
    points: list[dict],
    embeddings_by_cache_key: dict[int, list[float]],
    q_client,
    qdrant_batch_size: int,
    mock: bool,
    dependencies: QdrantDependencies | None = None,
) -> int:
    dependencies = dependencies or load_qdrant_dependencies()
    pending_points = []
    prepared_count = 0
    upserted_count = 0

    def flush_points() -> None:
        nonlocal upserted_count
        if not pending_points:
            return
        q_client.insert_chunks_batch(pending_points)
        upserted_count += len(pending_points)
        pending_points.clear()

    for prepared_count, point in enumerate(points, start=prepared_count + 1):
        cache_key = int(point["cache_key"])
        pending_points.append(
            build_qdrant_point(point, embeddings_by_cache_key[cache_key], dependencies)
        )

        if prepared_count % 100 == 0:
            mode_str = "Mock" if mock else "llama.cpp"
            print(f"  [{mode_str}] Prepared embeddings for {prepared_count} chunks...")

        if len(pending_points) >= qdrant_batch_size:
            flush_points()

    flush_points()
    return upserted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest vectors into Qdrant.")
    parser.add_argument(
        "--mock", action="store_true", help="Use deterministic mock embeddings."
    )
    parser.add_argument(
        "--model",
        choices=sorted(EMBEDDING_MODELS),
        default="qwen3-embedding:0.6b-fp16",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input unified chunks JSONL path.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=None,
        help="Texts per llama.cpp request; defaults to the model catalog value.",
    )
    parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=DEFAULT_QDRANT_BATCH_SIZE,
        help="Number of prepared points to upsert into Qdrant per request.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Seconds to wait for each llama.cpp request.",
    )
    parser.add_argument(
        "--server-mode",
        choices=("compose", "external"),
        default="compose",
        help="Start the local Compose service or use external llama.cpp servers.",
    )
    parser.add_argument(
        "--llama-server-url",
        action="append",
        default=[],
        help="External llama.cpp base URL; repeat for multiple workers.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=DEFAULT_COMPOSE_FILE,
        help="Docker Compose file used in compose mode.",
    )
    parser.add_argument(
        "--gguf-root",
        type=Path,
        default=DEFAULT_GGUF_ROOT,
        help="Directory containing canonical GGUF files.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Stop cache-only ingest cleanly after this wall-clock budget.",
    )
    parser.add_argument(
        "--stop-margin-seconds",
        type=float,
        default=600.0,
        help="Seconds reserved before max runtime for clean output export.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional JSONL cache path for corpus chunk embeddings.",
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable corpus embedding cache reads and writes.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Generate or update the corpus embedding cache without touching Qdrant.",
    )
    parser.add_argument(
        "--upload-from-cache",
        action="store_true",
        help="Upload Qdrant points from an existing complete embedding cache without calling llama.cpp.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = require_model(args.model)
    if spec.kind is not ModelKind.EMBEDDING:
        parser.error("--model must select an embedding model")
    if args.input_batch_size is None:
        args.input_batch_size = spec.local_request_batch_size
    args.llama_server_url = parse_server_urls(args.llama_server_url)
    if args.input_batch_size < 1:
        parser.error("--input-batch-size must be >= 1")
    if args.qdrant_batch_size < 1:
        parser.error("--qdrant-batch-size must be >= 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be > 0")
    if args.stop_margin_seconds < 0:
        parser.error("--stop-margin-seconds must be >= 0")
    if (
        args.max_runtime_seconds is not None
        and args.stop_margin_seconds >= args.max_runtime_seconds
    ):
        parser.error("--stop-margin-seconds must be less than --max-runtime-seconds")
    if args.max_runtime_seconds is not None and not args.cache_only:
        parser.error("--max-runtime-seconds requires --cache-only")
    if args.cache_only and args.upload_from_cache:
        parser.error("--cache-only and --upload-from-cache are mutually exclusive")
    if args.cache_only and args.no_embedding_cache:
        parser.error("--cache-only requires the embedding cache")
    if args.upload_from_cache and args.no_embedding_cache:
        parser.error("--upload-from-cache requires the embedding cache")
    if (
        args.server_mode == "external"
        and not args.llama_server_url
        and not args.mock
        and not args.upload_from_cache
    ):
        parser.error("--server-mode external requires --llama-server-url")
    args.runtime_stop_deadline = runtime_stop_deadline(
        args.max_runtime_seconds,
        args.stop_margin_seconds,
    )
    return args


def main():
    args = parse_args()
    spec = require_model(args.model)

    input_path = args.input or Path(CANONICAL_METADATA_PATH)

    model_suffix = args.model.replace(":", "_").replace("-", "_").replace(".", "_")

    points = read_normalized_input_points(input_path)

    if args.upload_from_cache:
        qdrant_dependencies = load_qdrant_dependencies()
        cache_path = args.embedding_cache or default_embedding_cache_path(args.model)
        vector_dim = detect_cache_vector_dimension(cache_path, args.model)
        embedding_cache = ChunkEmbeddingCache(cache_path, vector_dim)
        print(f"Using embedding cache at {embedding_cache.path}")
        embeddings_by_cache_key = collect_complete_cached_embeddings(
            points, args.model, embedding_cache
        )

        q_client = qdrant_dependencies.helper_type(
            host="localhost",
            port=6333,
            collection_name=f"thesis_chunks_{model_suffix}",
            vector_size=vector_dim,
        )
        q_client.init_collection()
        q_client.clear_collection()
        upserted_count = upsert_points_from_embeddings(
            points,
            embeddings_by_cache_key,
            q_client,
            args.qdrant_batch_size,
            args.mock,
            qdrant_dependencies,
        )
        print(
            "Vector Upload Complete. "
            f"Total chunks: {len(points)}, "
            f"cache hits: {len(points)}, "
            f"upserted: {upserted_count}."
        )
        return 0

    vector_dim = VECTOR_DIMENSION if args.mock else spec.vector_dimension
    if vector_dim is None:
        raise EmbeddingError(
            f"Embedding dimension is missing from the model catalog: {args.model}"
        )

    args.llama_clients = []
    if not args.mock:
        manager = LlamaCppComposeManager(args.compose_file)
        endpoints = resolve_server(
            args.server_mode,
            args.llama_server_url,
            spec,
            manager,
            args.gguf_root,
        )
        args.llama_clients = [
            LlamaCppClient(endpoint, timeout=args.request_timeout)
            for endpoint in endpoints
        ]
        print(f"Using llama.cpp embedding server(s): {', '.join(endpoints)}")

    embedding_cache = None
    if not args.no_embedding_cache and not args.mock:
        embedding_cache = ChunkEmbeddingCache(
            args.embedding_cache or default_embedding_cache_path(args.model),
            vector_dim,
            allow_truncated_final_record=args.cache_only,
        )
        print(f"Using embedding cache at {embedding_cache.path}")

    if embedding_cache:
        prune_embedding_cache(embedding_cache, points, args.model, vector_dim)

    collection = collect_or_create_embeddings(
        points,
        args,
        vector_dim,
        embedding_cache,
    )
    embeddings_by_cache_key = collection.embeddings_by_cache_key
    cache_hits = collection.cache_hits
    embedded_count = collection.embedded_count

    if args.cache_only:
        missing_count = len(points) - cache_hits - embedded_count
        if collection.stopped_early:
            print(
                "Vector Cache Partial. "
                f"Total chunks: {len(points)}, "
                f"cache hits: {cache_hits}, "
                f"embedded: {embedded_count}, "
                f"missing: {missing_count}."
            )
        else:
            print(
                "Vector Cache Complete. "
                f"Total chunks: {len(points)}, "
                f"cache hits: {cache_hits}, "
                f"embedded: {embedded_count}."
            )
        return 0

    qdrant_dependencies = load_qdrant_dependencies()
    q_client = qdrant_dependencies.helper_type(
        host="localhost",
        port=6333,
        collection_name=f"thesis_chunks_{model_suffix}",
        vector_size=vector_dim,
    )
    q_client.init_collection()
    q_client.clear_collection()
    upserted_count = upsert_points_from_embeddings(
        points,
        embeddings_by_cache_key,
        q_client,
        args.qdrant_batch_size,
        args.mock,
        qdrant_dependencies,
    )

    print(
        "Vector Ingestion Complete. "
        f"Total chunks: {len(points)}, "
        f"cache hits: {cache_hits}, "
        f"embedded: {embedded_count}, "
        f"upserted: {upserted_count}."
    )


if __name__ == "__main__":
    main()
