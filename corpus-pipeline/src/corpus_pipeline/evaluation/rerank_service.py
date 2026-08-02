from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.artifacts.bundle import ArtifactBundle, load_bundle, publish_bundle
from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.evaluation.rerank_score_cache import (
    RerankScoreCache,
    finalize_rerank_cache,
    prompt_contract_hash,
)
from corpus_pipeline.evaluation.rerankers import LlamaCppReranker
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.evaluation.run_workspace import RunWorkspace
from corpus_pipeline.runtime.catalog import ModelKind, require_model
from corpus_pipeline.runtime.client import LlamaCppClient
from corpus_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server
from corpus_pipeline.vector_store.ingest_vectors import (
    DEFAULT_COMPOSE_FILE,
    DEFAULT_GGUF_ROOT,
)


@dataclass(frozen=True)
class RerankRequest:
    run_root: Path
    candidates_dir: Path | None
    output_dir: Path | None
    model: str
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float


@dataclass(frozen=True)
class RerankStageResult:
    bundle: ArtifactBundle | None
    actions: tuple[str, ...]
    incomplete: bool = False


def rerank_checkpoint_path(output_dir: Path, candidate_sha256: str, model: str) -> Path:
    spec = require_model(model)
    identity = {
        "candidate_data_sha256": candidate_sha256,
        "reranker": model,
        "model_sha256": spec.sha256,
        "protocol": spec.reranker_protocol,
        "request_contract_sha256": prompt_contract_hash(
            protocol=spec.reranker_protocol or ""
        ),
    }
    return Path(output_dir) / ".checkpoints" / f"{canonical_sha256(identity)}.jsonl"


class LocalRerankBackend:
    def run(self, request: RerankRequest) -> RerankStageResult:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        workspace = RunWorkspace.open_or_create(
            request.run_root, _identity(request), force=request.force
        )
        candidates_dir = request.candidates_dir or workspace.candidates_dir
        candidate_bundle = load_bundle(
            candidates_dir, expected_type="retrieval_candidates", require_complete=True
        )
        if request.dry_run:
            return RerankStageResult(None, ("dry-run",))
        output_dir = request.output_dir or workspace.rerank_scores_dir
        partial = rerank_checkpoint_path(
            output_dir, candidate_bundle.manifest.data_sha256, request.model
        )
        if request.force:
            partial.unlink(missing_ok=True)
        cache = RerankScoreCache(partial)
        manager = LlamaCppComposeManager(DEFAULT_COMPOSE_FILE)
        endpoint = resolve_server("compose", [], spec, manager, DEFAULT_GGUF_ROOT)[0]
        reranker = LlamaCppReranker(
            spec, LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
        )
        processed = 0
        for record in CandidateArtifactReader.from_data_path(
            candidate_bundle.data_path
        ):
            row = {"query_id": record["query_id"], "query": record["query"]}
            candidates = [_candidate(item) for item in record["candidates"]]
            for candidate in candidates:
                if (
                    cache.key_for(request.model, row, candidate) in cache.records
                    and not request.force
                ):
                    continue
                scored = reranker.rerank(row["query"], [candidate])[0]
                cache.set(
                    request.model,
                    row,
                    candidate,
                    scored.rerank_score or 0.0,
                    protocol=spec.reranker_protocol,
                )
                processed += 1
        data_path, _manifest_path, completion = finalize_rerank_cache(
            candidate_data_path=candidate_bundle.data_path,
            candidate_manifest_path=candidate_bundle.manifest_path,
            partial_cache_path=partial,
            output_dir=output_dir,
            reranker=request.model,
            gguf_sha256=spec.sha256,
            protocol=spec.reranker_protocol or "",
            request_contract_sha256=prompt_contract_hash(
                protocol=spec.reranker_protocol or ""
            ),
            require_complete=False,
        )
        bundle = publish_bundle(
            output_dir,
            artifact_type="rerank_score_cache",
            source_path=data_path,
            identity={
                "candidate_data_sha256": candidate_bundle.manifest.data_sha256,
                "reranker": request.model,
            },
            total=completion.total,
            complete=completion.complete,
        )
        workspace.record_rerank_scores(bundle)
        return RerankStageResult(
            bundle, (f"scored={processed}",), incomplete=not completion.is_complete
        )


def _candidate(item: dict) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=str(item["chunk_id"]),
        score=float(item["retrieval_score"]),
        rank=int(item["retrieval_rank"]),
        source=str(item["source"]),
        payload=dict(item.get("payload") or {}),
        document_text=str(item["document_text"]),
        document_hash=str(item["document_hash"]),
    )


def _identity(request: RerankRequest):
    from corpus_pipeline.evaluation.run_workspace import load_run_record

    return load_run_record(request.run_root / "run.json").identity


class KaggleRerankBackend:
    def __init__(
        self, runner: Callable[[RerankRequest], RerankStageResult] | None = None
    ):
        self.runner = runner or self._run_kaggle

    def run(self, request: RerankRequest) -> RerankStageResult:
        return self.runner(request)

    @staticmethod
    def _run_kaggle(request: RerankRequest) -> RerankStageResult:
        from corpus_pipeline.integrations.kaggle.models import StageName
        from corpus_pipeline.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        workspace = RunWorkspace.open_or_create(
            request.run_root, _identity(request), force=request.force
        )
        candidates_dir = request.candidates_dir or workspace.candidates_dir
        candidate_bundle = load_bundle(
            candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        output_dir = request.output_dir or workspace.rerank_scores_dir
        result = run_kaggle_stage(
            stage=StageName.RERANK,
            model=request.model,
            input_path=candidate_bundle.data_path,
            output_dir=output_dir,
            gguf_root=DEFAULT_GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
        )
        actions = tuple(action.reason for action in result.actions)
        if result.artifact_path is None:
            return RerankStageResult(
                None,
                actions,
                incomplete=not result.completion.is_complete,
            )
        data_path, _manifest_path, completion = finalize_rerank_cache(
            candidate_data_path=candidate_bundle.data_path,
            candidate_manifest_path=candidate_bundle.manifest_path,
            partial_cache_path=result.artifact_path,
            output_dir=output_dir,
            reranker=request.model,
            gguf_sha256=spec.sha256,
            protocol=spec.reranker_protocol or "",
            request_contract_sha256=prompt_contract_hash(
                protocol=spec.reranker_protocol or ""
            ),
            require_complete=False,
        )
        bundle = publish_bundle(
            output_dir,
            artifact_type="rerank_score_cache",
            source_path=data_path,
            identity={
                "candidate_data_sha256": candidate_bundle.manifest.data_sha256,
                "reranker": request.model,
            },
            total=completion.total,
            complete=completion.complete,
        )
        workspace.record_rerank_scores(bundle)
        return RerankStageResult(
            bundle,
            actions,
            incomplete=not completion.is_complete,
        )
