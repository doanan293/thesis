"""Retrieval evaluation through the backend RetrievalService (spec §10.4)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.retrieval.models import Hit, Query, QueryOrigin
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.infrastructure.composition import build_retrieval_service
from pharma_agent.infrastructure.settings import Settings

from seed_pipeline.bundle.chunks import gold_chunk_label
from seed_pipeline.bundle.export import COLLECTION_KEY
from seed_pipeline.config.paths import query_embedding_cache_path
from seed_pipeline.evaluation.artifact_contracts import sha256_file
from seed_pipeline.evaluation.cached_query_embedder import (
    CachedQueryEmbedder,
    cached_query_embedder,
)
from seed_pipeline.evaluation.query_embedding_cache import QueryEmbeddingCache
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifact,
    CandidateArtifactReader,
    build_candidate_artifact,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace
from seed_pipeline.runtime.catalog import require_model

RETRIEVERS = ("bm25", "dense", "hybrid")


class SearchService(Protocol):
    async def search(
        self, queries: Sequence[Query], rerank_query: str
    ) -> SearchResult: ...


class RetrievalBackend(Protocol):
    @property
    def service(self) -> SearchService: ...

    async def aclose(self) -> None: ...


class BackendFactory(Protocol):
    def __call__(
        self, settings: Settings, *, embedder: CachedQueryEmbedder | None = None
    ) -> RetrievalBackend: ...


@dataclass(frozen=True)
class RetrieveRequest:
    evaluation_path: Path
    run_root: Path
    retriever: str
    candidate_k: int
    rrf_k: int
    limit: int | None
    force: bool
    collection: str = COLLECTION_KEY
    prefetch_k: int | None = None
    artifact_root: Path | None = None
    backend_env_file: Path | None = None
    query_embeddings: Path | None = None


@dataclass(frozen=True)
class RetrieveResult:
    workspace: RunWorkspace
    artifact: CandidateArtifact


def resolve_prefetch_k(
    retriever: str, candidate_k: int, prefetch_k: int | None
) -> int | None:
    if retriever != "hybrid":
        if prefetch_k is not None:
            raise ValueError("--prefetch-k is only valid with --retriever hybrid")
        return None
    effective = candidate_k if prefetch_k is None else prefetch_k
    if effective < candidate_k:
        raise ValueError("--prefetch-k must be greater than or equal to --candidate-k")
    return effective


def evaluation_settings(base: Settings, request: RetrieveRequest) -> Settings:
    rerank = base.retrieval.rerank.model_copy(
        update={
            "protocol": "none",
            "top_n": request.candidate_k,
            "max_candidates": request.candidate_k,
        }
    )
    retrieval = base.retrieval.model_copy(
        update={
            "mode": request.retriever,
            "candidate_k": request.candidate_k,
            "prefetch_k": request.prefetch_k or request.candidate_k,
            "rrf_k": request.rrf_k,
            "collections": [request.collection],
            "rerank": rerank,
        }
    )
    langfuse = base.langfuse.model_copy(update={"public_key": None, "secret_key": None})
    return base.model_copy(update={"retrieval": retrieval, "langfuse": langfuse})


def query_cache_for(
    request: RetrieveRequest, *, model: str, dimension: int
) -> QueryEmbeddingCache:
    if request.query_embeddings is not None:
        return QueryEmbeddingCache(request.query_embeddings, vector_dim=dimension)
    spec = require_model(model)
    if spec.vector_dimension != dimension:
        raise ValueError(
            f"Backend embedding dimension {dimension} does not match "
            f"{spec.vector_dimension} for {model}"
        )
    return QueryEmbeddingCache(
        query_embedding_cache_path(model),
        vector_dim=dimension,
        model_sha256=spec.sha256,
    )


def candidate_from_hit(hit: Hit, rank: int, source: str) -> RetrievalCandidate:
    label = gold_chunk_label(hit.section_key, hit.ordinal)
    return RetrievalCandidate(
        chunk_id=label,
        score=hit.fusion_score,
        rank=rank,
        source=source,
        payload={
            "chunk_id": label,
            "section_id": hit.section_key,
            "chunk_index": hit.ordinal,
        },
        document_text=hit.embedding_text,
    )


async def search_hits(service: SearchService, text: str) -> list[Hit]:
    result = await service.search(
        [Query(text=text, origin=QueryOrigin.INITIAL)], rerank_query=text
    )
    if result.error is not None:
        raise RuntimeError(f"backend retrieval failed: {result.error}")
    return [item.hit for item in result.items]


async def current_release_id(service: SearchService, probe_query: str) -> str:
    releases = {str(hit.release_id) for hit in await search_hits(service, probe_query)}
    if len(releases) != 1:
        raise RuntimeError(
            "Expected hits from exactly one published release, found "
            f"{sorted(releases)}; run `pharma-agent corpus import ... --publish` first"
        )
    return releases.pop()


def load_query_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid evaluation JSON at {path}:{line_number}"
                    ) from exc
                if (
                    not isinstance(row, dict)
                    or not row.get("query_id")
                    or not str(row.get("query") or "").strip()
                ):
                    raise ValueError(
                        "Evaluation row requires query_id and query at "
                        f"{path}:{line_number}"
                    )
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    except OSError as exc:
        raise ValueError(f"Evaluation JSONL is missing: {path}") from exc
    if not rows:
        raise ValueError(f"Evaluation JSONL is empty: {path}")
    return rows


class BackendCandidateRetriever:
    def __init__(
        self,
        runner: asyncio.Runner,
        service: SearchService,
        *,
        source: str,
        release_id: str,
    ) -> None:
        self._runner = runner
        self._service = service
        self._source = source
        self._release_id = release_id

    def search_batch(
        self, rows: Sequence[dict[str, Any]], limit: int
    ) -> list[list[RetrievalCandidate]]:
        return self._runner.run(self._search_rows(rows, limit))

    async def _search_rows(
        self, rows: Sequence[dict[str, Any]], limit: int
    ) -> list[list[RetrievalCandidate]]:
        return list(
            await asyncio.gather(*(self._search_row(row, limit) for row in rows))
        )

    async def _search_row(
        self, row: dict[str, Any], limit: int
    ) -> list[RetrievalCandidate]:
        hits = await search_hits(self._service, str(row["query"]))
        releases = {str(hit.release_id) for hit in hits} - {self._release_id}
        if releases:
            raise RuntimeError(
                f"Query {row['query_id']} returned release {sorted(releases)} but the "
                f"run is pinned to release {self._release_id}; start a new --run"
            )
        return [
            candidate_from_hit(hit, rank, self._source)
            for rank, hit in enumerate(hits[:limit], start=1)
        ]


def run_retrieval(
    request: RetrieveRequest,
    *,
    backend_factory: BackendFactory = build_retrieval_service,
) -> RetrieveResult:
    if request.candidate_k < 1 or request.rrf_k < 1:
        raise ValueError("candidate_k and rrf_k must be >= 1")
    if request.retriever not in RETRIEVERS:
        raise ValueError("retriever must be bm25, dense or hybrid")
    request = replace(
        request,
        prefetch_k=resolve_prefetch_k(
            request.retriever, request.candidate_k, request.prefetch_k
        ),
    )
    rows = load_query_rows(request.evaluation_path, request.limit)
    base = (
        Settings(_env_file=request.backend_env_file)
        if request.backend_env_file is not None
        else Settings()
    )
    settings = evaluation_settings(base, request)
    embedding = settings.retrieval.embedding
    embedder: CachedQueryEmbedder | None = None
    query_embeddings_sha256: str | None = None
    if request.retriever != "bm25":
        embedder, query_embeddings_sha256 = cached_query_embedder(
            query_cache_for(
                request, model=embedding.model, dimension=embedding.dimension
            ),
            rows,
            model=embedding.model,
            dimension=embedding.dimension,
        )
    stack = backend_factory(settings, embedder=embedder)
    with asyncio.Runner() as runner:
        try:
            release_id = runner.run(
                current_release_id(stack.service, str(rows[0]["query"]))
            )
            identity = RunIdentity(
                evaluation_path=str(request.evaluation_path.resolve()),
                evaluation_sha256=sha256_file(request.evaluation_path),
                collection_name=settings.retrieval.qdrant_collection,
                embedding_model=embedding.model,
                query_embeddings_sha256=query_embeddings_sha256,
                retriever=request.retriever,
                candidate_k=request.candidate_k,
                rrf_k=request.rrf_k,
                limit=request.limit,
                prefetch_k=request.prefetch_k,
                release_id=release_id,
                chunker_version=CHUNKER_VERSION,
            )
            workspace = RunWorkspace.open_or_create(
                request.run_root,
                identity,
                artifact_root=request.artifact_root,
                force=request.force,
            )
            reusable = _reusable_artifact(workspace, request)
            if reusable is not None:
                workspace.record_candidates(reusable)
                return RetrieveResult(workspace, reusable)
            artifact = build_candidate_artifact(
                rows=rows,
                retriever=BackendCandidateRetriever(
                    runner,
                    stack.service,
                    source=request.retriever,
                    release_id=release_id,
                ),
                output_path=workspace.candidates_dir / "candidates.jsonl",
                identity=asdict(identity),
                candidate_k=request.candidate_k,
            )
            workspace.record_candidates(artifact)
            return RetrieveResult(workspace, artifact)
        finally:
            runner.run(stack.aclose())


def _reusable_artifact(
    workspace: RunWorkspace, request: RetrieveRequest
) -> CandidateArtifact | None:
    data_path = workspace.candidates_dir / "candidates.jsonl"
    manifest_path = workspace.candidates_dir / "manifest.json"
    if request.force or not data_path.is_file() or not manifest_path.is_file():
        return None
    reader = CandidateArtifactReader(data_path, manifest_path)
    query_count = 0
    pair_count = 0
    for record in reader:
        query_count += 1
        pair_count += len(record["candidates"])
    if reader.manifest.identity.get("candidate_k") != request.candidate_k:
        return None
    return CandidateArtifact(
        data_path, manifest_path, reader.manifest, query_count, pair_count
    )
