from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from qdrant_client import QdrantClient

from corpus_pipeline.evaluation.artifact_contracts import sha256_file
from corpus_pipeline.evaluation.dump_retrieval_candidates import load_query_rows
from corpus_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifact,
    CandidateArtifactReader,
    build_candidate_artifact,
)
from corpus_pipeline.evaluation.retrievers import (
    DenseQdrantRetriever,
    QdrantBm25Retriever,
    QdrantHybridRetriever,
)
from corpus_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace
from corpus_pipeline.runtime.catalog import require_model


def model_collection_name(model_name: str) -> str:
    return "thesis_chunks_" + require_model(model_name).slug


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


@dataclass(frozen=True)
class RetrieveRequest:
    evaluation_path: Path
    query_embeddings_dir: Path | None
    run_root: Path
    embedding_model: str
    qdrant_url: str
    retriever: str
    candidate_k: int
    rrf_k: int
    limit: int | None
    force: bool
    prefetch_k: int | None = None


@dataclass(frozen=True)
class RetrieveResult:
    workspace: RunWorkspace
    artifact: CandidateArtifact


def _qdrant_client(url: str) -> QdrantClient:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--qdrant-url must be an http(s) URL")
    return QdrantClient(url=url)


def build_retriever_for_request(
    request: RetrieveRequest,
    rows: list[dict],
    query_cache: QueryEmbeddingCache | None,
):
    collection_name = model_collection_name(request.embedding_model)
    client = _qdrant_client(request.qdrant_url)
    collections = {item.name for item in client.get_collections().collections}
    if collection_name not in collections:
        raise ValueError(
            f"Qdrant collection '{collection_name}' is missing; ingest vectors for "
            f"{request.embedding_model} first"
        )
    if request.retriever == "bm25":
        return QdrantBm25Retriever(client, collection_name)
    if query_cache is None:
        raise ValueError("Dense and hybrid retrieval require --query-embeddings")
    cache = query_cache
    dimension = require_model(request.embedding_model).vector_dimension
    if dimension is None:
        raise ValueError("Embedding dimension is missing")
    for row in rows:
        vector = cache.get(
            request.embedding_model, str(row["query_id"]), str(row["query"])
        )
        if vector is None:
            raise QueryEmbeddingCacheError(
                f"Query embedding cache is missing {row['query_id']}"
            )
        if len(vector) != dimension:
            raise QueryEmbeddingCacheError(
                f"Query embedding dimension for {row['query_id']} is {len(vector)}, expected {dimension}"
            )

    def embed_query(row):
        if not isinstance(row, dict):
            raise QueryEmbeddingCacheError(
                "Cached retrieval requires the full query row"
            )
        vector = cache.get(
            request.embedding_model, str(row["query_id"]), str(row["query"])
        )
        if vector is None:
            raise QueryEmbeddingCacheError(
                f"Query embedding cache is missing {row['query_id']}"
            )
        return vector

    retriever_type = (
        QdrantHybridRetriever if request.retriever == "hybrid" else DenseQdrantRetriever
    )
    kwargs = (
        {
            "rrf_k": request.rrf_k,
            "prefetch_k": request.prefetch_k or request.candidate_k,
        }
        if request.retriever == "hybrid"
        else {}
    )
    return retriever_type(client, collection_name, embed_query=embed_query, **kwargs)


def run_retrieval(request: RetrieveRequest) -> RetrieveResult:
    if request.candidate_k < 1 or request.rrf_k < 1:
        raise ValueError("candidate_k and rrf_k must be >= 1")
    if request.retriever not in {"dense", "bm25", "hybrid"}:
        raise ValueError("retriever must be dense, bm25, or hybrid")
    request = replace(
        request,
        prefetch_k=resolve_prefetch_k(
            request.retriever, request.candidate_k, request.prefetch_k
        ),
    )
    rows = load_query_rows(request.evaluation_path, request.limit)
    query_cache = None
    query_embeddings_sha256 = None
    if request.retriever in {"dense", "hybrid"}:
        if request.query_embeddings_dir is None:
            raise ValueError("Dense and hybrid retrieval require --query-embeddings")
        spec = require_model(request.embedding_model)
        query_cache = QueryEmbeddingCache(
            request.query_embeddings_dir,
            vector_dim=spec.vector_dimension,
            model_sha256=spec.sha256,
        )
        subset = query_cache.validate_subset(rows, request.embedding_model)
        if not subset.is_complete:
            raise QueryEmbeddingCacheError(
                f"Query embedding cache is missing {subset.missing} records"
            )
        query_embeddings_sha256 = subset.sha256
    collection_name = model_collection_name(request.embedding_model)
    identity = RunIdentity(
        evaluation_path=str(request.evaluation_path.resolve()),
        evaluation_sha256=sha256_file(request.evaluation_path),
        collection_name=collection_name,
        embedding_model=request.embedding_model,
        query_embeddings_sha256=query_embeddings_sha256,
        retriever=request.retriever,
        candidate_k=request.candidate_k,
        rrf_k=request.rrf_k,
        limit=request.limit,
        prefetch_k=request.prefetch_k,
    )
    workspace = RunWorkspace.open_or_create(
        request.run_root, identity, force=request.force
    )
    existing_data = workspace.candidates_dir / "candidates.jsonl"
    existing_manifest = workspace.candidates_dir / "manifest.json"
    if not request.force and existing_data.is_file() and existing_manifest.is_file():
        reader = CandidateArtifactReader(existing_data, existing_manifest)
        query_count = 0
        pair_count = 0
        for record in reader:
            query_count += 1
            pair_count += len(record["candidates"])
        if reader.manifest.identity.get("candidate_k") == request.candidate_k:
            artifact = CandidateArtifact(
                existing_data,
                existing_manifest,
                reader.manifest,
                query_count,
                pair_count,
            )
            workspace.record_candidates(artifact)
            return RetrieveResult(workspace, artifact)
    retriever = build_retriever_for_request(request, rows, query_cache)
    artifact = build_candidate_artifact(
        rows=rows,
        retriever=retriever,
        output_path=workspace.candidates_dir / "candidates.jsonl",
        identity=asdict(identity),
        candidate_k=request.candidate_k,
    )
    workspace.record_candidates(artifact)
    return RetrieveResult(workspace, artifact)
