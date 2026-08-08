from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from corpus_pipeline.cache.jsonl_records import merge_records
from corpus_pipeline.config.paths import WORK_DIR
from corpus_pipeline.evaluation.artifact_contracts import sha256_file
from corpus_pipeline.runtime.catalog import ModelKind, require_model
from corpus_pipeline.runtime.client import LlamaCppClient
from corpus_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server
from corpus_pipeline.vector_store.ingest_vectors import (
    DEFAULT_COMPOSE_FILE,
    DEFAULT_GGUF_ROOT,
    ChunkEmbeddingCache,
    collect_or_create_embeddings,
    expected_cache_keys,
    prune_embedding_cache,
    read_normalized_input_points,
    summarize_embedding_cache_completion,
)


@dataclass(frozen=True)
class ChunkEmbeddingRequest:
    chunks_path: Path
    model: str
    cache_path: Path
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float


@dataclass(frozen=True)
class EmbeddingStageResult:
    cache_path: Path | None
    actions: tuple[str, ...]
    incomplete: bool = False


class ChunkEmbeddingBackend(Protocol):
    def run(self, request: ChunkEmbeddingRequest) -> EmbeddingStageResult:
        raise NotImplementedError


def chunk_embedding_identity(request: ChunkEmbeddingRequest) -> dict:
    spec = require_model(request.model)
    return {
        "input_sha256": sha256_file(request.chunks_path),
        "model": request.model,
        "model_sha256": spec.sha256,
        "vector_dimension": spec.vector_dimension,
    }


class LocalChunkEmbeddingBackend:
    def __init__(
        self,
        *,
        compose_file: Path = DEFAULT_COMPOSE_FILE,
        gguf_root: Path = DEFAULT_GGUF_ROOT,
        server_mode: str = "compose",
        llama_server_urls: list[str] | None = None,
    ) -> None:
        self.compose_file = compose_file
        self.gguf_root = gguf_root
        self.server_mode = server_mode
        self.llama_server_urls = llama_server_urls or []

    def run(self, request: ChunkEmbeddingRequest) -> EmbeddingStageResult:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError("--model must select an embedding model")
        if request.dry_run:
            return EmbeddingStageResult(None, ("dry-run",))
        points = read_normalized_input_points(request.chunks_path)
        if request.force:
            request.cache_path.unlink(missing_ok=True)
        cache = ChunkEmbeddingCache(
            request.cache_path,
            spec.vector_dimension or 0,
            model_sha256=spec.sha256,
        )
        if request.force:
            cache.replace_keys(
                expected_cache_keys(points, request.model, spec.vector_dimension or 0)
            )
        prune_embedding_cache(cache, points, request.model, spec.vector_dimension or 0)
        complete_before_run = summarize_embedding_cache_completion(
            points, request.model, cache
        ).is_complete
        if complete_before_run:
            endpoints = []
        else:
            manager = LlamaCppComposeManager(self.compose_file)
            endpoints = resolve_server(
                self.server_mode,
                self.llama_server_urls,
                spec,
                manager,
                self.gguf_root,
            )
        args = SimpleNamespace(
            model=request.model,
            input_batch_size=spec.local_request_batch_size,
            mock=False,
            llama_clients=[
                LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
                for endpoint in endpoints
            ],
            runtime_stop_deadline=None,
        )
        collection = collect_or_create_embeddings(
            points,
            args,
            spec.vector_dimension or 0,
            cache,
            retain_embeddings=False,
        )
        return EmbeddingStageResult(
            request.cache_path,
            ("local",),
            collection.stopped_early,
        )


class KaggleChunkEmbeddingBackend:
    def __init__(
        self,
        runner: Callable[[ChunkEmbeddingRequest], EmbeddingStageResult] | None = None,
    ):
        self.runner = runner or self._run_kaggle_orchestrator

    def run(self, request: ChunkEmbeddingRequest) -> EmbeddingStageResult:
        return self.runner(request)

    @staticmethod
    def _run_kaggle_orchestrator(
        request: ChunkEmbeddingRequest,
    ) -> EmbeddingStageResult:
        from corpus_pipeline.integrations.kaggle.models import StageName
        from corpus_pipeline.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        result = run_kaggle_stage(
            stage=StageName.CORPUS_EMBED,
            model=request.model,
            input_path=request.chunks_path,
            output_dir=WORK_DIR
            / "kaggle-chunk-embeddings"
            / require_model(request.model).slug,
            gguf_root=DEFAULT_GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
        )
        if result.artifact_path is None:
            return EmbeddingStageResult(
                None,
                tuple(action.reason for action in result.actions),
                incomplete=not result.completion.is_complete,
            )
        remote = ChunkEmbeddingCache(
            result.artifact_path,
            spec.vector_dimension or 0,
            model_sha256=spec.sha256,
        )
        merge_records(
            request.cache_path,
            remote.record_data.values(),
            key=lambda record: (
                str(record["model"]),
                int(record["chunk_key"]),
                str(record["text_hash"]),
                str(record["payload_hash"]),
                int(record["vector_dim"]),
            ),
            equivalent=lambda left, right: left["embedding"] == right["embedding"],
        )
        return EmbeddingStageResult(
            request.cache_path,
            tuple(action.reason for action in result.actions),
            incomplete=not result.completion.is_complete,
        )
