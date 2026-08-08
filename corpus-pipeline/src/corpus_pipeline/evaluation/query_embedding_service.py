from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from corpus_pipeline.cache.jsonl_records import merge_records
from corpus_pipeline.config.paths import WORK_DIR, query_embedding_bundle_dir
from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.evaluation.preload_query_embeddings import preload_embeddings
from corpus_pipeline.evaluation.query_embedding_artifact import (
    migrate_query_bundle,
    query_cache_identity,
)
from corpus_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    query_hash,
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
    cache_path: Path | None
    subset_sha256: str | None
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
            return QueryEmbeddingStageResult(None, None, ("dry-run",))
        manager = LlamaCppComposeManager(self.compose_file)
        endpoint = resolve_server("compose", [], spec, manager, self.gguf_root)[0]
        client = LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
        partial_cache = Path(request.output_dir)
        cache = QueryEmbeddingCache(
            partial_cache,
            vector_dim=spec.vector_dimension or 0,
            model_sha256=spec.sha256,
        )
        rows = _read_query_rows(request.evaluation_path)
        if not partial_cache.exists():
            legacy_bundle = query_embedding_bundle_dir(
                request.model, _evaluation_sha256(request.evaluation_path)
            )
            if legacy_bundle.is_dir():
                migrate_query_bundle(
                    legacy_bundle,
                    cache,
                    model=request.model,
                    vector_dim=spec.vector_dimension or 0,
                    model_sha256=spec.sha256,
                )
        if request.force:
            cache.replace_keys(_query_keys(rows, request.model))
        preload_embeddings(
            eval_path=request.evaluation_path,
            model_name=request.model,
            cache_path=partial_cache,
            embed_batch_fn=lambda queries: client.embed(
                queries, request.model, spec.vector_dimension
            ),
            batch_size=spec.local_request_batch_size,
            force=False,
            vector_dim=spec.vector_dimension,
            model_sha256=spec.sha256,
        )
        cache = QueryEmbeddingCache(
            partial_cache,
            vector_dim=spec.vector_dimension or 0,
            model_sha256=spec.sha256,
        )
        subset = cache.validate_subset(rows, request.model)
        return QueryEmbeddingStageResult(
            partial_cache,
            subset.sha256,
            ("local",),
            incomplete=not subset.is_complete,
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

        remote_dir = (
            WORK_DIR / "kaggle-query-embeddings" / require_model(request.model).slug
        )
        result = run_kaggle_stage(
            stage=StageName.QUERY_EMBED,
            model=request.model,
            input_path=request.evaluation_path,
            output_dir=remote_dir,
            gguf_root=DEFAULT_GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
        )
        if result.artifact_path is None:
            return QueryEmbeddingStageResult(
                None,
                None,
                tuple(action.reason for action in result.actions),
                incomplete=not result.completion.is_complete,
            )
        spec = require_model(request.model)
        remote = QueryEmbeddingCache(
            result.artifact_path,
            vector_dim=spec.vector_dimension,
            model_sha256=spec.sha256,
        )
        merge_records(
            request.output_dir,
            remote.record_metadata.values(),
            key=lambda record: (
                str(record["model"]),
                str(record["query_id"]),
                str(record["query_hash"]),
            ),
            equivalent=lambda left, right: left["embedding"] == right["embedding"],
        )
        local = QueryEmbeddingCache(
            request.output_dir,
            vector_dim=spec.vector_dimension,
            model_sha256=spec.sha256,
        )
        subset = local.validate_subset(
            _read_query_rows(request.evaluation_path), request.model
        )
        return QueryEmbeddingStageResult(
            request.output_dir,
            subset.sha256,
            tuple(action.reason for action in result.actions),
            incomplete=not subset.is_complete,
        )


def _evaluation_sha256(path: Path) -> str:
    from corpus_pipeline.evaluation.artifact_contracts import sha256_file

    return sha256_file(path)


def _read_query_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _query_keys(rows: list[dict], model: str) -> set[tuple[str, str, str]]:
    return {
        (
            str(model),
            str(row.get("query_id") or ""),
            query_hash(str(row.get("query") or "")),
        )
        for row in rows
    }
