from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.artifacts.bundle import load_bundle
from seed_pipeline.cache.jsonl_records import merge_records
from seed_pipeline.config.paths import (
    COMPOSE_FILE,
    GGUF_ROOT,
    WORK_DIR,
    rerank_score_cache_path,
)
from seed_pipeline.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
)
from seed_pipeline.evaluation.rerank_score_cache import (
    RerankScoreCache,
)
from seed_pipeline.evaluation.rerankers import LlamaCppReranker, Reranker
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunConflictError,
    RunWorkspace,
    load_run_record,
)
from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity
from seed_pipeline.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
)
from seed_pipeline.runtime.catalog import ModelKind, require_model
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server


@dataclass(frozen=True)
class RerankRequest:
    run_root: Path
    model: str
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    benchmark: bool = False
    benchmark_pairs: int = 512
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


def _cleanup_completed_stage_artifact(artifact_path: Path, staging_root: Path) -> None:
    root = Path(staging_root).resolve()
    artifact_dir = Path(artifact_path).resolve().parent
    if artifact_dir == root or not artifact_dir.is_relative_to(root):
        raise ValueError(
            f"completed artifact is outside Kaggle rerank staging root: {artifact_dir}"
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


def _workspace(request: RerankRequest) -> RunWorkspace:
    record = load_run_record(request.run_root / "run.json")
    return RunWorkspace(request.run_root, record.identity, record.origin)


def _existing_variant(
    workspace: RunWorkspace, slug: str, identity_sha256: str, *, force: bool
) -> RerankVariantRecord | None:
    """The registered variant to reuse, or None when it must be (re)built."""
    existing = workspace.variant_records().get(slug)
    if existing is None or force:
        return None
    if existing.variant_sha256 != identity_sha256:
        raise RunConflictError(
            f"Rerank variant {workspace.rerank_dir(slug)} already exists for "
            "different inputs; use --force"
        )
    return existing


class LocalRerankBackend:
    def __init__(
        self,
        reranker_factory: Callable[[object, float], Reranker] | None = None,
    ):
        self.reranker_factory = reranker_factory or self._default_reranker

    @staticmethod
    def _default_reranker(spec, timeout: float) -> Reranker:
        manager = LlamaCppComposeManager(COMPOSE_FILE)
        endpoint = resolve_server("compose", [], spec, manager, GGUF_ROOT)[0]
        return LlamaCppReranker(spec, LlamaCppClient(endpoint, timeout=timeout))

    def run(self, request: RerankRequest) -> RerankStageResult:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        if request.benchmark:
            raise ValueError("benchmark requires --backend kaggle")
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model)
        cache = RerankScoreCache(
            cache_path, model_sha256=spec.sha256, request_contract_sha256=contract
        )
        if request.dry_run:
            missing = cache.validate_subset(
                candidate_bundle.data_path, request.model
            ).missing
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (
                    f"target={workspace.rerank_dir(spec.slug)}",
                    f"missing_pairs={missing}",
                    "dry-run",
                ),
            )
        expected = cache.expected_keys_from_candidates(
            candidate_bundle.data_path, request.model
        )
        if request.force:
            cache.replace_keys(expected)
        # A complete cache must not start a model server, so the reranker is created
        # only when the first pair without a score is found.
        reranker: Reranker | None = None
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
                if reranker is None:
                    reranker = self.reranker_factory(
                        spec, request.request_timeout_seconds
                    )
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
                force=request.force,
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
        from seed_pipeline.integrations.kaggle.models import StageName
        from seed_pipeline.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model)
        missing = (
            RerankScoreCache(
                cache_path, model_sha256=spec.sha256, request_contract_sha256=contract
            )
            .validate_subset(candidate_bundle.data_path, request.model)
            .missing
        )
        missing_pairs = f"missing_pairs={missing}"
        from seed_pipeline.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )

        resolution = ensure_runtime_profile(
            workload="rerank",
            benchmark_stage=StageName.RERANK_BENCHMARK.value,
            model=request.model,
            input_path=candidate_bundle.data_path,
            gguf_root=GGUF_ROOT,
            budget_seconds=request.budget_seconds,
            dry_run=request.dry_run,
            force=request.force,
            kaggle_account=request.kaggle_account,
        )
        if resolution.profile is None:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (missing_pairs, f"profile={resolution.action}"),
                incomplete=True,
            )
        remote_dir = WORK_DIR / "kaggle-rerank-scores" / spec.slug
        result = run_kaggle_stage(
            stage=StageName.RERANK,
            model=request.model,
            input_path=candidate_bundle.data_path,
            output_dir=remote_dir,
            gguf_root=GGUF_ROOT,
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
                (missing_pairs, f"target={workspace.rerank_dir(spec.slug)}", *actions),
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
                force=request.force,
            )
        _cleanup_completed_stage_artifact(result.artifact_path, remote_dir)
        return RerankStageResult(
            bundle.root,
            identity.sha256,
            subset.sha256,
            actions,
        )
