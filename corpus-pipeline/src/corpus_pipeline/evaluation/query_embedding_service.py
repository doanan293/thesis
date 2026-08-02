from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from corpus_pipeline.artifacts.bundle import ArtifactBundle, publish_bundle
from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.evaluation.preload_query_embeddings import preload_embeddings
from corpus_pipeline.evaluation.query_embedding_artifact import (
    finalize_query_cache,
    query_cache_identity,
)
from corpus_pipeline.runtime.catalog import ModelKind, require_model
from corpus_pipeline.runtime.client import LlamaCppClient
from corpus_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server
from corpus_pipeline.vector_store.ingest_vectors import (
    DEFAULT_COMPOSE_FILE,
    DEFAULT_GGUF_ROOT,
)


@dataclass(frozen=True)
class QueryEmbeddingRequest:
    evaluation_path: Path
    model: str
    output_dir: Path
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float


@dataclass(frozen=True)
class QueryEmbeddingStageResult:
    bundle: ArtifactBundle | None
    actions: tuple[str, ...]
    incomplete: bool = False


class QueryEmbeddingBackend(Protocol):
    def run(self, request: QueryEmbeddingRequest) -> QueryEmbeddingStageResult:
        raise NotImplementedError


def query_embedding_identity(request: QueryEmbeddingRequest) -> dict:
    spec = require_model(request.model)
    identity = query_cache_identity(
        request.evaluation_path,
        model=request.model,
        gguf_sha256=spec.sha256,
        vector_dimension=spec.vector_dimension or 0,
    )
    identity["logical_sha256"] = canonical_sha256(identity)
    return identity


def query_checkpoint_path(request: QueryEmbeddingRequest) -> Path:
    identity = query_embedding_identity(request)
    return request.output_dir / ".checkpoints" / f"{identity['logical_sha256']}.jsonl"


class LocalQueryEmbeddingBackend:
    def __init__(
        self,
        *,
        compose_file: Path = DEFAULT_COMPOSE_FILE,
        gguf_root: Path = DEFAULT_GGUF_ROOT,
    ):
        self.compose_file = compose_file
        self.gguf_root = gguf_root

    def run(self, request: QueryEmbeddingRequest) -> QueryEmbeddingStageResult:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError("--model must select an embedding model")
        if request.dry_run:
            return QueryEmbeddingStageResult(None, ("dry-run",))
        manager = LlamaCppComposeManager(self.compose_file)
        endpoint = resolve_server("compose", [], spec, manager, self.gguf_root)[0]
        client = LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
        partial_cache = query_checkpoint_path(request)
        if request.force:
            partial_cache.unlink(missing_ok=True)
        stats = preload_embeddings(
            eval_path=request.evaluation_path,
            model_name=request.model,
            cache_path=partial_cache,
            embed_batch_fn=lambda queries: client.embed(
                queries, request.model, spec.vector_dimension
            ),
            batch_size=spec.local_request_batch_size,
            force=request.force,
        )
        artifact = finalize_query_cache(
            eval_path=request.evaluation_path,
            cache_path=partial_cache,
            model=request.model,
            gguf_sha256=spec.sha256,
            vector_dimension=spec.vector_dimension or 0,
            output_dir=request.output_dir,
            require_complete=False,
        )
        bundle = publish_bundle(
            request.output_dir,
            artifact_type="query_embeddings",
            source_path=artifact.data_path,
            identity=artifact.manifest.identity,
            total=stats["total"],
            complete=stats["cached"] + stats["processed"],
        )
        return QueryEmbeddingStageResult(
            bundle,
            ("local",),
            incomplete=not artifact.completion.is_complete,
        )


class KaggleQueryEmbeddingBackend:
    def __init__(
        self,
        runner: Callable[[QueryEmbeddingRequest], QueryEmbeddingStageResult]
        | None = None,
    ):
        self.runner = runner or self._run_kaggle_orchestrator

    def run(self, request: QueryEmbeddingRequest) -> QueryEmbeddingStageResult:
        return self.runner(request)

    @staticmethod
    def _run_kaggle_orchestrator(
        request: QueryEmbeddingRequest,
    ) -> QueryEmbeddingStageResult:
        from corpus_pipeline.integrations.kaggle.models import StageName
        from corpus_pipeline.integrations.kaggle.service import run_kaggle_stage

        result = run_kaggle_stage(
            stage=StageName.QUERY_EMBED,
            model=request.model,
            input_path=request.evaluation_path,
            output_dir=request.output_dir,
            gguf_root=DEFAULT_GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
        )
        if result.artifact_path is None:
            return QueryEmbeddingStageResult(
                None,
                tuple(action.reason for action in result.actions),
                incomplete=not result.completion.is_complete,
            )
        bundle = publish_bundle(
            request.output_dir,
            artifact_type="query_embeddings",
            source_path=result.artifact_path,
            identity=query_embedding_identity(request),
            total=result.completion.total,
            complete=result.completion.complete,
        )
        return QueryEmbeddingStageResult(
            bundle,
            tuple(action.reason for action in result.actions),
            incomplete=not result.completion.is_complete,
        )
