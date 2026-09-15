from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.cache.jsonl_records import ValidatedSubset, merge_records
from seed_pipeline.config.paths import (
    COMPOSE_FILE,
    GGUF_ROOT,
    WORK_DIR,
    rerank_score_cache_path,
)
from seed_pipeline.evaluation.local_rerank_benchmark import (
    ComposeRerankServer,
    LocalRerankBenchmarkResult,
    load_local_rerank_profile,
    run_local_rerank_benchmark,
)
from seed_pipeline.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
)
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.evaluation.rerank_log import open_rerank_log
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
from seed_pipeline.integrations.kaggle.artifacts import sha256_file
from seed_pipeline.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
)
from seed_pipeline.runtime.catalog import ModelKind, ModelSpec, require_model
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.compose import LlamaCppComposeManager, reranker_environment


@dataclass(frozen=True)
class RerankRequest:
    run_root: Path
    model: str
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    benchmark: bool = False
    kaggle_account: str | None = None
    max_runs: int | None = None
    recover_kernel: str | None = None


@dataclass(frozen=True)
class RerankStageResult:
    artifact_dir: Path | None
    variant_sha256: str
    subset_sha256: str | None
    actions: tuple[str, ...]
    incomplete: bool = False
    benchmark_report: Path | None = None
    benchmark_levels: int = 0
    quota: tuple[str, ...] = ()


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


def _local_subset(
    cache_path: Path,
    candidates: Path,
    model: str,
    *,
    model_sha256: str,
    request_contract_sha256: str,
) -> ValidatedSubset:
    with kaggle_cache_lock(cache_path):
        return RerankScoreCache(
            cache_path,
            model_sha256=model_sha256,
            request_contract_sha256=request_contract_sha256,
        ).validate_subset(candidates, model)


def _finalize_variant(
    request: RerankRequest,
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
    cache_path: Path,
    identity: RerankVariantIdentity,
    subset: ValidatedSubset,
    actions: tuple[str, ...],
) -> RerankStageResult:
    # Different model jobs may finish concurrently; serialize the run registry update.
    with kaggle_cache_lock(request.run_root / "run.json"):
        bundle = finalize_run_rerank_bundle(
            workspace=workspace,
            candidate_bundle=candidate_bundle,
            cache_path=cache_path,
            identity=identity,
            force=request.force,
        )
    return RerankStageResult(bundle.root, identity.sha256, subset.sha256, actions)


def _recovered_scores(root: Path, model: str) -> Path:
    """The verified rerank score file inside a downloaded kernel output."""
    manifests = [
        path
        for path in Path(root).rglob("manifest.json")
        if (path.parent / "rerank_scores.jsonl").is_file()
    ]
    if len(manifests) != 1:
        raise ValueError(
            "Expected one rerank_scores.jsonl with a manifest.json in the kernel "
            f"output, found {len(manifests)}"
        )
    manifest_path = manifests[0]
    data_path = manifest_path.parent / "rerank_scores.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "rerank_scores"
        or manifest.get("data_filename") != data_path.name
    ):
        raise ValueError(
            f"Kernel output is not a rerank score artifact: {manifest_path}"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("model") != model:
        raise ValueError(
            f"Kernel output holds scores of another model than {model}: {manifest_path}"
        )
    if manifest.get("data_sha256") != sha256_file(data_path):
        raise ValueError(
            f"Kernel output checksum does not match its manifest: {data_path}"
        )
    return data_path


class LocalRerankBackend:
    def __init__(
        self,
        reranker_factory: Callable[[ModelSpec, float], Reranker] | None = None,
        benchmark_runner: Callable[[ModelSpec, Path, float], LocalRerankBenchmarkResult]
        | None = None,
    ):
        self.reranker_factory = reranker_factory or self._default_reranker
        self.benchmark_runner = benchmark_runner or self._default_benchmark

    @staticmethod
    def _default_reranker(spec: ModelSpec, timeout: float) -> Reranker:
        # A stored local benchmark profile sets the llama-reranker level; without one
        # compose defaults (or the root .env) apply.
        profile = load_local_rerank_profile(spec)
        manager = LlamaCppComposeManager(COMPOSE_FILE)
        endpoint = manager.ensure(
            "reranker",
            spec,
            GGUF_ROOT,
            runtime=profile.selected if profile is not None else None,
        )
        return LlamaCppReranker(spec, LlamaCppClient(endpoint, timeout=timeout))

    @staticmethod
    def _default_benchmark(
        spec: ModelSpec, candidate_data_path: Path, timeout: float
    ) -> LocalRerankBenchmarkResult:
        manager = LlamaCppComposeManager(COMPOSE_FILE)
        return run_local_rerank_benchmark(
            spec=spec,
            candidate_data_path=candidate_data_path,
            server=ComposeRerankServer(manager, spec, GGUF_ROOT),
            client_factory=lambda endpoint: LlamaCppClient(endpoint, timeout=timeout),
        )

    def _benchmark(
        self,
        request: RerankRequest,
        spec: ModelSpec,
        candidate_data_path: Path,
        variant_sha256: str,
    ) -> RerankStageResult:
        result = self.benchmark_runner(
            spec, candidate_data_path, request.request_timeout_seconds
        )
        selected = result.profile.selected
        environment = " ".join(
            f"{name}={value}"
            for name, value in sorted(reranker_environment(selected).items())
        )
        return RerankStageResult(
            None,
            variant_sha256,
            None,
            (
                f"selected=server_slots={selected.server_slots} "
                f"ubatch={selected.physical_batch_size} threads={selected.threads}",
                "latency_p95_seconds="
                f"{result.selected_measurement.latency_p95_seconds}",
                f"env={environment}",
            ),
            benchmark_report=result.path,
            benchmark_levels=len(result.measurements),
        )

    def run(self, request: RerankRequest) -> RerankStageResult:
        if request.recover_kernel is not None:
            raise ValueError("--recover-kernel requires --backend kaggle")
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        if request.benchmark and request.dry_run:
            raise ValueError("--benchmark cannot be combined with --dry-run")
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        if request.benchmark:
            return self._benchmark(
                request, spec, candidate_bundle.data_path, identity.sha256
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
        # only when the first query with an unscored candidate is found.
        reranker: Reranker | None = None
        processed = 0
        for record in CandidateArtifactReader.from_data_path(
            candidate_bundle.data_path
        ):
            row = {"query_id": record["query_id"], "query": record["query"]}
            candidates = [_candidate(item) for item in record["candidates"]]
            unscored = [
                candidate
                for candidate in candidates
                if cache.key_for(request.model, row, candidate) not in cache.records
            ]
            if not unscored:
                continue
            if reranker is None:
                reranker = self.reranker_factory(spec, request.request_timeout_seconds)
            # One /v1/rerank request per query, like the Kaggle worker and the backend.
            for scored in reranker.rerank(row["query"], unscored):
                cache.set(
                    request.model,
                    row,
                    scored,
                    scored.rerank_score or 0.0,
                    protocol=spec.reranker_protocol,
                )
            processed += len(unscored)
        subset = cache.validate_subset(candidate_bundle.data_path, request.model)
        if not subset.is_complete:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (f"scored={processed}",),
                incomplete=True,
            )
        return _finalize_variant(
            request,
            workspace,
            candidate_bundle,
            cache_path,
            identity,
            subset,
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
        if request.benchmark:
            raise ValueError(
                "--benchmark measures the CPU reranker with --backend local; Kaggle "
                "benchmarks run automatically when no runtime profile matches"
            )
        spec = require_model(request.model)
        # Kaggle staging/runtime resources are model-scoped under WORK_DIR,
        # so serialize the same model across runs while allowing variants
        # for different models to proceed concurrently.
        lock_target = WORK_DIR / "kaggle-rerank-jobs" / spec.slug
        with kaggle_job_lock(lock_target):
            if request.recover_kernel is not None:
                return KaggleRerankBackend._recover_kernel(request)
            return KaggleRerankBackend._run_kaggle_unlocked(request)

    @staticmethod
    def _recover_kernel(request: RerankRequest) -> RerankStageResult:
        """Merge the scores a finished kernel left in its output into the local cache.

        Works across job identities: every record carries the model and request
        contract digests and the query and document hashes, and the local cache
        rejects records that do not match the current catalog entry.
        """
        from seed_pipeline.integrations.kaggle.kernel_service import KernelService
        from seed_pipeline.integrations.kaggle.models import (
            KernelPresence,
            KernelStatus,
        )
        from seed_pipeline.integrations.kaggle.service import (
            resolve_profile_execution_contexts,
        )
        from seed_pipeline.integrations.kaggle.workspace import (
            managed_staging_directory,
        )

        reference = str(request.recover_kernel)
        owner, separator, slug = reference.partition("/")
        if not separator or not owner or not slug:
            raise ValueError("--recover-kernel must be OWNER/KERNEL-SLUG")
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        context = next(
            (
                item
                for item in resolve_profile_execution_contexts()
                if item.profile is not None
                and item.profile.username.casefold() == owner.casefold()
            ),
            None,
        )
        if context is None:
            raise ValueError(
                f"No Kaggle account profile in seed-pipeline/.env has username {owner}"
            )
        kernels = KernelService(context.runner, context.owners.execution)
        remote = kernels.inspect_state(reference)
        if remote.presence is not KernelPresence.EXISTS or remote.status not in {
            KernelStatus.COMPLETE,
            KernelStatus.ERROR,
        }:
            raise ValueError(
                f"Kernel {reference} has no finished output to recover "
                f"(presence={remote.presence.value}, status={remote.status})"
            )
        log = open_rerank_log(request.model, echo=True)
        cache_path = rerank_score_cache_path(request.model)
        with managed_staging_directory(
            WORK_DIR / "kaggle-rerank-recovery", prefix=f"{spec.slug}-"
        ) as staging:
            kernels.download_output(reference, staging)
            remote_scores = RerankScoreCache(
                _recovered_scores(staging, request.model),
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )
            with kaggle_cache_lock(cache_path):
                _merge_remote_rerank_scores(
                    cache_path,
                    remote_scores,
                    model_sha256=spec.sha256,
                    request_contract_sha256=contract,
                )
        subset = _local_subset(
            cache_path,
            candidate_bundle.data_path,
            request.model,
            model_sha256=spec.sha256,
            request_contract_sha256=contract,
        )
        recovered = f"recovered_pairs={len(remote_scores.records)}"
        log(
            f"recovered kernel={reference} {recovered} "
            f"local cache pairs={subset.complete}/{subset.total}"
        )
        actions = (f"kernel={reference}", recovered, f"missing_pairs={subset.missing}")
        if not subset.is_complete:
            return RerankStageResult(
                None, identity.sha256, None, actions, incomplete=True
            )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root, identity.sha256, bundle.manifest.data_sha256, actions
            )
        return _finalize_variant(
            request, workspace, candidate_bundle, cache_path, identity, subset, actions
        )

    @staticmethod
    def _run_kaggle_unlocked(request: RerankRequest) -> RerankStageResult:
        from seed_pipeline.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )
        from seed_pipeline.integrations.kaggle.models import (
            CloudArtifact,
            PipelineResult,
            StageJob,
            StageName,
        )
        from seed_pipeline.integrations.kaggle.service import (
            active_kernel_profile,
            resolve_session_contexts,
            run_kaggle_stage,
            runtime_manifest_sha256,
        )
        from seed_pipeline.integrations.kaggle.sessions import (
            QuotaExhausted,
            ReservedAccount,
            reserve_session_account,
            run_account_sessions,
        )

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
        candidates = candidate_bundle.data_path
        log = open_rerank_log(request.model, echo=True)

        def local_subset() -> ValidatedSubset:
            return _local_subset(
                cache_path,
                candidates,
                request.model,
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )

        if request.force and not request.dry_run:
            with kaggle_cache_lock(cache_path):
                cache = RerankScoreCache(
                    cache_path,
                    model_sha256=spec.sha256,
                    request_contract_sha256=contract,
                )
                cache.replace_keys(
                    cache.expected_keys_from_candidates(candidates, request.model)
                )
            log("force: removed this run's pairs from the local score cache")
        subset = local_subset()
        missing_pairs = f"missing_pairs={subset.missing}"
        target = f"target={workspace.rerank_dir(spec.slug)}"
        log(
            f"kaggle start run={request.run_root.name} "
            f"account={request.kaggle_account or 'default'} {missing_pairs}"
        )
        if subset.is_complete and not request.dry_run:
            log("local score cache is complete; no Kaggle session needed")
            return _finalize_variant(
                request,
                workspace,
                candidate_bundle,
                cache_path,
                identity,
                subset,
                (missing_pairs, "reuse=local score cache"),
            )

        contexts = resolve_session_contexts(request.kaggle_account)
        first = contexts[0]

        def benchmark_runner(**benchmark_arguments: Any) -> PipelineResult:
            with reserve_session_account(
                contexts, requested_budget_seconds=request.budget_seconds, log=log
            ) as reserved:
                return run_kaggle_stage(
                    stage=StageName.RERANK_BENCHMARK,
                    model=request.model,
                    output_dir=WORK_DIR / "kaggle-runtime-benchmarks" / spec.slug,
                    budget_seconds=reserved.budget_seconds,
                    kaggle_account=reserved.profile,
                    **benchmark_arguments,
                )

        try:
            resolution = ensure_runtime_profile(
                workload="rerank",
                benchmark_stage=StageName.RERANK_BENCHMARK.value,
                model=request.model,
                input_path=candidates,
                gguf_root=GGUF_ROOT,
                budget_seconds=request.budget_seconds,
                dry_run=request.dry_run,
                force=request.force,
                kaggle_account=first.profile.name if first.profile else None,
                runtime_sha256=(
                    None
                    if request.dry_run
                    else runtime_manifest_sha256(first.runner, first.owners)
                ),
                benchmark_runner=benchmark_runner,
            )
        except QuotaExhausted as exhausted:
            for row in exhausted.table:
                log(row)
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (missing_pairs, target, "stop=quota-exhausted"),
                incomplete=True,
                quota=exhausted.table,
            )
        if resolution.profile is None:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (missing_pairs, f"profile={resolution.action}"),
                incomplete=True,
            )
        runtime_profile = resolution.profile.selected
        remote_dir = WORK_DIR / "kaggle-rerank-scores" / spec.slug

        def merge_artifact(_job: StageJob, artifact: CloudArtifact) -> None:
            remote = RerankScoreCache(
                artifact.data_path,
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
            merged = local_subset()
            log(
                f"merged artifact pairs={artifact.completion.complete}/"
                f"{artifact.completion.total}; local cache pairs="
                f"{merged.complete}/{merged.total}"
            )

        checkpoint_source = RerankCacheCheckpoint(cache_path)

        def run_session(reserved: ReservedAccount, index: int) -> PipelineResult:
            return run_kaggle_stage(
                stage=StageName.RERANK,
                model=request.model,
                input_path=candidates,
                output_dir=remote_dir,
                gguf_root=GGUF_ROOT,
                force=request.force and index == 1,
                resume_remote=not request.force,
                check_only=request.dry_run,
                max_runs=1,
                budget_seconds=reserved.budget_seconds,
                runtime_profile=runtime_profile,
                kaggle_account=reserved.profile,
                artifact_sink=merge_artifact,
                local_checkpoint=checkpoint_source,
            )

        preferred = (
            None
            if request.dry_run or request.force
            else active_kernel_profile(
                contexts,
                stage=StageName.RERANK,
                model=request.model,
                input_path=candidates,
                output_dir=remote_dir,
                gguf_root=GGUF_ROOT,
                runtime_profile=runtime_profile,
            )
        )
        outcome = run_account_sessions(
            contexts,
            run_session,
            requested_budget_seconds=request.budget_seconds,
            max_sessions=request.max_runs,
            check_only=request.dry_run,
            log=log,
            preferred_profile=preferred,
        )
        subset = local_subset()
        actions = (
            f"missing_pairs={subset.missing}",
            target,
            f"sessions={outcome.sessions}",
            f"stop={outcome.stop_reason.value}",
            *(
                f"{action.verb.value} {action.resource_kind} "
                f"{action.reference}: {action.reason}"
                for action in (outcome.result.actions if outcome.result else ())
            ),
        )
        if request.dry_run or not subset.is_complete:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                actions,
                incomplete=True,
                quota=outcome.quota_table,
            )
        result = _finalize_variant(
            request, workspace, candidate_bundle, cache_path, identity, subset, actions
        )
        if outcome.result is not None and outcome.result.artifact_path is not None:
            _cleanup_completed_stage_artifact(outcome.result.artifact_path, remote_dir)
        log(f"kaggle complete variant={result.artifact_dir}")
        return result
