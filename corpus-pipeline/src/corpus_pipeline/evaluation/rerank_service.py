from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.artifacts.bundle import load_bundle
from corpus_pipeline.cache.jsonl_records import merge_records
from corpus_pipeline.config.paths import WORK_DIR, rerank_score_cache_path
from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
    migrate_legacy_rerank,
)
from corpus_pipeline.evaluation.rerank_score_cache import (
    RerankScoreCache,
)
from corpus_pipeline.evaluation.rerankers import LlamaCppReranker, Reranker
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.evaluation.run_workspace import RunWorkspace
from corpus_pipeline.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
)
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
    model: str
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    benchmark: bool = False
    benchmark_pairs: int = 512
    artifact_root: Path | None = None
    kaggle_account: str | None = None


@dataclass(frozen=True)
class RerankStageResult:
    artifact_dir: Path | None
    variant_sha256: str
    subset_sha256: str | None
    actions: tuple[str, ...]
    incomplete: bool = False
    benchmark_report: Path | None = None
    benchmark_levels: int = 0


def rerank_checkpoint_path(output_dir: Path, candidate_sha256: str, model: str) -> Path:
    spec = require_model(model)
    identity = {
        "candidate_data_sha256": candidate_sha256,
        "reranker": model,
        "model_sha256": spec.sha256,
        "protocol": spec.reranker_protocol,
        "request_contract_sha256": (
            spec.rerank_contract.sha256 if spec.rerank_contract is not None else ""
        ),
    }
    return Path(output_dir) / ".checkpoints" / f"{canonical_sha256(identity)}.jsonl"


def _cleanup_completed_stage_artifact(
    artifact_path: Path, staging_root: Path
) -> None:
    root = Path(staging_root).resolve()
    artifact_dir = Path(artifact_path).resolve().parent
    if artifact_dir == root or not artifact_dir.is_relative_to(root):
        raise ValueError(
            f"completed artifact is outside Kaggle rerank staging root: "
            f"{artifact_dir}"
        )
    shutil.rmtree(artifact_dir)


def _rerank_record_key(record: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(record["reranker"]),
        str(record.get("model_sha256") or ""),
        str(record.get("request_contract_sha256") or ""),
        str(record["query_id"]),
        str(record["query_hash"]),
        str(record["chunk_id"]),
        str(record["document_hash"]),
    )


def _merge_remote_rerank_scores(
    cache_path: Path,
    remote: RerankScoreCache,
    *,
    model_sha256: str,
    request_contract_sha256: str,
) -> None:
    """Make a completed Kaggle job authoritative for keys it produced."""
    local = RerankScoreCache(
        cache_path,
        model_sha256=model_sha256,
        request_contract_sha256=request_contract_sha256,
    )
    local.replace_keys(set(remote.record_metadata))
    merge_records(
        cache_path,
        remote.record_metadata.values(),
        key=_rerank_record_key,
        equivalent=lambda left, right: left["score"] == right["score"],
    )


class LocalRerankBackend:
    def __init__(
        self,
        reranker_factory: Callable[[object, float], Reranker] | None = None,
    ):
        self.reranker_factory = reranker_factory or self._default_reranker

    @staticmethod
    def _default_reranker(spec, timeout: float) -> Reranker:
        manager = LlamaCppComposeManager(DEFAULT_COMPOSE_FILE)
        endpoint = resolve_server("compose", [], spec, manager, DEFAULT_GGUF_ROOT)[0]
        return LlamaCppReranker(spec, LlamaCppClient(endpoint, timeout=timeout))

    def run(self, request: RerankRequest) -> RerankStageResult:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        if request.benchmark:
            raise ValueError("benchmark requires --backend kaggle")
        workspace = RunWorkspace.open_or_create(
            request.run_root,
            _identity(request),
            artifact_root=request.artifact_root,
            force=request.force,
        )
        candidates_dir = request.candidates_dir or workspace.candidates_dir
        candidate_bundle = load_bundle(
            candidates_dir, expected_type="retrieval_candidates", require_complete=True
        )
        from corpus_pipeline.evaluation.variant_identity import RerankVariantIdentity

        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = workspace.variant_records().get(identity.sha256)
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, identity.sha256, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        try:
            migrated = migrate_legacy_rerank(workspace, candidate_bundle)
        except (OSError, RuntimeError, ValueError):
            migrated = None
        if migrated is not None:
            digest, migrated_record = migrated
            bundle = load_registered_rerank_bundle(workspace, digest, migrated_record)
            if digest == identity.sha256:
                return RerankStageResult(
                    bundle.root,
                    digest,
                    bundle.manifest.data_sha256,
                    ("reuse=migrated legacy variant",),
                )
        if request.dry_run:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (f"target rerank variant {identity.sha256}", "dry-run"),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model, spec.sha256, contract)
        cache = RerankScoreCache(
            cache_path,
            model_sha256=spec.sha256,
            request_contract_sha256=contract,
        )
        expected = cache.expected_keys_from_candidates(
            candidate_bundle.data_path, request.model
        )
        if request.force:
            cache.replace_keys(expected)
        reranker = self.reranker_factory(spec, request.request_timeout_seconds)
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
        subset = cache.validate_subset(candidate_bundle.data_path, request.model)
        if not subset.is_complete:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (f"scored={processed}",),
                incomplete=True,
            )
        # Different model jobs may finish concurrently; serialize the
        # read-modify-write of the shared run registry while publishing.
        with kaggle_cache_lock(request.run_root / "run.json"):
            bundle = finalize_run_rerank_bundle(
                workspace=workspace,
                candidate_bundle=candidate_bundle,
                cache_path=cache_path,
                identity=identity,
            )
        return RerankStageResult(
            bundle.root,
            identity.sha256,
            subset.sha256,
            (f"scored={processed}",),
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
        spec = require_model(request.model)
        # Kaggle staging/runtime resources are model-scoped under WORK_DIR,
        # so serialize the same model across runs while allowing variants
        # for different models to proceed concurrently.
        lock_target = WORK_DIR / "kaggle-rerank-jobs" / spec.slug
        with kaggle_job_lock(lock_target):
            return KaggleRerankBackend._run_kaggle_unlocked(request)

    @staticmethod
    def _run_kaggle_unlocked(request: RerankRequest) -> RerankStageResult:
        from corpus_pipeline.integrations.kaggle.models import StageName
        from corpus_pipeline.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        workspace = RunWorkspace.open_or_create(
            request.run_root,
            _identity(request),
            artifact_root=request.artifact_root,
            force=request.force,
        )
        candidates_dir = request.candidates_dir or workspace.candidates_dir
        candidate_bundle = load_bundle(
            candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        from corpus_pipeline.evaluation.variant_identity import RerankVariantIdentity

        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = workspace.variant_records().get(identity.sha256)
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, identity.sha256, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        try:
            with kaggle_cache_lock(request.run_root / "run.json"):
                migrated = migrate_legacy_rerank(workspace, candidate_bundle)
        except (OSError, RuntimeError, ValueError):
            migrated = None
        if migrated is not None:
            digest, migrated_record = migrated
            bundle = load_registered_rerank_bundle(workspace, digest, migrated_record)
            if digest == identity.sha256:
                return RerankStageResult(
                    bundle.root,
                    digest,
                    bundle.manifest.data_sha256,
                    ("reuse=migrated legacy variant",),
                )
        from corpus_pipeline.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )

        resolution = ensure_runtime_profile(
            workload="rerank",
            benchmark_stage=StageName.RERANK_BENCHMARK.value,
            model=request.model,
            input_path=candidate_bundle.data_path,
            gguf_root=DEFAULT_GGUF_ROOT,
            budget_seconds=request.budget_seconds,
            dry_run=request.dry_run,
            force=request.force,
            kaggle_account=request.kaggle_account,
        )
        if resolution.profile is None:
            return RerankStageResult(None, identity.sha256, None, (f"profile={resolution.action}",), incomplete=True)
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model, spec.sha256, contract)
        remote_dir = WORK_DIR / "kaggle-rerank-scores" / spec.slug
        result = run_kaggle_stage(
            stage=StageName.RERANK,
            model=request.model,
            input_path=candidate_bundle.data_path,
            output_dir=remote_dir,
            gguf_root=DEFAULT_GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
            runtime_profile=resolution.profile.selected,
            kaggle_account=request.kaggle_account,
        )
        actions = tuple(
            f"{action.verb.value} {action.resource_kind} {action.reference}: {action.reason}"
            for action in result.actions
        )
        if result.artifact_path is None:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (f"target rerank variant {identity.sha256}", *actions),
                incomplete=not result.completion.is_complete,
            )
        remote = RerankScoreCache(
            result.artifact_path,
            model_sha256=spec.sha256,
            request_contract_sha256=contract,
        )
        with kaggle_cache_lock(cache_path):
            _merge_remote_rerank_scores(
                cache_path,
                remote,
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )
            local = RerankScoreCache(
                cache_path,
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )
            subset = local.validate_subset(candidate_bundle.data_path, request.model)
        if not subset.is_complete:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                actions,
                incomplete=True,
            )
        with kaggle_cache_lock(request.run_root / "run.json"):
            bundle = finalize_run_rerank_bundle(
                workspace=workspace,
                candidate_bundle=candidate_bundle,
                cache_path=cache_path,
                identity=identity,
            )
        _cleanup_completed_stage_artifact(result.artifact_path, remote_dir)
        return RerankStageResult(
            bundle.root,
            identity.sha256,
            subset.sha256,
            actions,
        )
