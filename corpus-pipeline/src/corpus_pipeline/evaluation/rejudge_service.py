from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from corpus_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.config.paths import RAG_FINAL_SECTIONS_PATH
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    atomic_promote,
    load_manifest,
    sha256_file,
    write_json,
)
from corpus_pipeline.evaluation.metrics_service import (
    MetricsRequest,
    MetricsResult,
    run_metrics,
)
from corpus_pipeline.evaluation.rejudging import RejudgingResult, rejudge_rows
from corpus_pipeline.evaluation.rerank_artifacts import load_registered_rerank_bundle
from corpus_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from corpus_pipeline.evaluation.run_workspace import (
    RunRecord,
    RunWorkspace,
    load_run_record,
)
from corpus_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class RejudgeRequest:
    evaluation_path: Path
    dense_run_root: Path
    dense_artifact_root: Path
    hybrid_run_root: Path
    hybrid_artifact_root: Path
    apply: bool = False
    top_k: int = 30
    window_size: int = 3


@dataclass(frozen=True)
class RejudgeResult:
    old_evaluation_sha256: str
    new_evaluation_sha256: str
    summary: dict[str, int]
    reports: tuple[Path, ...]
    applied: bool


@dataclass(frozen=True)
class _RunInputs:
    metadata_root: Path
    artifact_root: Path
    record: RunRecord
    candidate_bundle: ArtifactBundle
    candidate_rows: list[dict[str, Any]]
    manifest_paths: tuple[Path, ...]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_objects(path))


def _known_section_ids() -> set[str]:
    return {
        str(row.get("id"))
        for row in iter_jsonl_objects(RAG_FINAL_SECTIONS_PATH)
        if row.get("id")
    }


def _candidate_bundle(
    metadata_root: Path, artifact_root: Path, record: RunRecord
) -> _RunInputs:
    candidate_root = artifact_root / "candidates"
    bundle = load_bundle(
        candidate_root,
        expected_type="retrieval_candidates",
        require_complete=True,
    )
    snapshot_path = metadata_root / "candidates-manifest.json"
    snapshot = load_manifest(snapshot_path)
    if snapshot.data_sha256 != bundle.manifest.data_sha256:
        raise ArtifactContractError(f"candidate snapshot mismatch: {metadata_root}")
    old_eval = record.identity.evaluation_sha256
    if bundle.manifest.identity.get("evaluation_sha256") != old_eval:
        raise ArtifactContractError(f"candidate evaluation mismatch: {candidate_root}")
    rows = list(CandidateArtifactReader.from_data_path(bundle.data_path))
    manifest_paths = (snapshot_path, bundle.manifest_path)
    return _RunInputs(
        metadata_root, artifact_root, record, bundle, rows, manifest_paths
    )


def _load_run_inputs(
    metadata_root: Path, artifact_root: Path, expected_evaluation_sha256: str
) -> _RunInputs:
    record = load_run_record(Path(metadata_root) / "run.json")
    if record.identity.evaluation_sha256 != expected_evaluation_sha256:
        label = "dense" if record.identity.retriever == "dense" else "hybrid"
        raise ArtifactContractError(f"{label} run evaluation hash is stale")
    return _candidate_bundle(Path(metadata_root), Path(artifact_root), record)


def _validate_query_rows(
    evaluation_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> None:
    expected = [(str(row["query_id"]), str(row["query"])) for row in evaluation_rows]
    actual = [(str(row["query_id"]), str(row["query"])) for row in candidate_rows]
    if actual != expected:
        raise ArtifactContractError(
            "Candidate query inputs differ from evaluation; cannot reuse artifacts"
        )


def _validate_rerank_variants(inputs: _RunInputs) -> None:
    workspace = RunWorkspace(
        inputs.metadata_root, inputs.record.identity, inputs.artifact_root
    )
    for variant_sha256, variant_record in inputs.record.rerank_variants.items():
        bundle = load_registered_rerank_bundle(
            workspace, variant_sha256, variant_record
        )
        spec = require_model(variant_record.model)
        if spec.rerank_contract is None:
            raise ArtifactContractError(
                f"Reranker has no contract: {variant_record.model}"
            )
        cache = RerankScoreCache(
            bundle.data_path,
            model_sha256=spec.sha256,
            request_contract_sha256=spec.rerank_contract.sha256,
            rewrite_legacy=False,
        )
        subset = cache.validate_subset(
            inputs.candidate_bundle.data_path, variant_record.model
        )
        if not subset.is_complete:
            raise ArtifactContractError(
                f"Rerank variant is incomplete: {variant_sha256}"
            )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _update_run_record(path: Path, new_evaluation_sha256: str) -> None:
    record = load_run_record(path)
    identity = replace(record.identity, evaluation_sha256=new_evaluation_sha256)
    RunWorkspace(path.parent, identity).write_record(replace(record, identity=identity))


def _update_manifest(path: Path, old_sha256: str, new_sha256: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = payload.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("evaluation_sha256") != old_sha256
    ):
        raise ArtifactContractError(f"manifest evaluation mismatch: {path}")
    identity["evaluation_sha256"] = new_sha256
    payload["identity"] = identity
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def _report_targets(request: RejudgeRequest) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                Path(request.dense_run_root) / "reports",
                Path(request.dense_artifact_root) / "reports",
                Path(request.hybrid_run_root) / "reports",
                Path(request.hybrid_artifact_root) / "reports",
            )
        )
    )


def _run_metrics(
    request: RejudgeRequest,
    metrics_runner: Callable[[MetricsRequest], MetricsResult],
) -> tuple[Path, ...]:
    dense = metrics_runner(
        MetricsRequest(
            Path(request.dense_run_root),
            top_k=request.top_k,
            window_size=request.window_size,
            artifact_root=Path(request.dense_artifact_root),
        )
    )
    hybrid = metrics_runner(
        MetricsRequest(
            Path(request.hybrid_run_root),
            top_k=request.top_k,
            window_size=request.window_size,
            artifact_root=Path(request.hybrid_artifact_root),
        )
    )
    for result, metadata_root in (
        (dense, Path(request.dense_run_root)),
        (hybrid, Path(request.hybrid_run_root)),
    ):
        source = Path(result.baseline.report_path)
        if not source.is_file():
            raise ArtifactContractError(f"baseline report is missing: {source}")
        summary = metadata_root / "reports" / "baseline.md"
        summary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, summary)
    reports = [
        dense.baseline.report_path,
        *[item.report_path for item in dense.reranked],
    ]
    reports.extend(
        [hybrid.baseline.report_path, *[item.report_path for item in hybrid.reranked]]
    )
    return tuple(reports)


def run_rejudging(
    request: RejudgeRequest,
    *,
    metrics_runner: Callable[[MetricsRequest], MetricsResult] = run_metrics,
    known_section_ids: set[str] | None = None,
) -> RejudgeResult:
    evaluation_path = Path(request.evaluation_path)
    old_sha256 = sha256_file(evaluation_path)
    dense = _load_run_inputs(
        request.dense_run_root, request.dense_artifact_root, old_sha256
    )
    hybrid = _load_run_inputs(
        request.hybrid_run_root, request.hybrid_artifact_root, old_sha256
    )
    if dense.record.identity.evaluation_sha256 != old_sha256:
        raise ArtifactContractError("dense run evaluation hash is stale")
    if hybrid.record.identity.evaluation_sha256 != old_sha256:
        raise ArtifactContractError("hybrid run evaluation hash is stale")
    evaluation_rows = _load_rows(evaluation_path)
    _validate_query_rows(evaluation_rows, dense.candidate_rows)
    _validate_query_rows(evaluation_rows, hybrid.candidate_rows)
    _validate_rerank_variants(hybrid)
    candidates = dense.candidate_rows + hybrid.candidate_rows
    rejudged: RejudgingResult = rejudge_rows(
        evaluation_rows,
        candidates,
        known_section_ids if known_section_ids is not None else _known_section_ids(),
    )

    temporary_root = Path(tempfile.mkdtemp(prefix="rejudge-"))
    staged_evaluation = temporary_root / evaluation_path.name
    _write_jsonl(staged_evaluation, rejudged.rows)
    new_sha256 = sha256_file(staged_evaluation)
    result = RejudgeResult(
        old_sha256,
        new_sha256,
        rejudged.summary,
        (),
        False,
    )
    if not request.apply:
        shutil.rmtree(temporary_root, ignore_errors=True)
        return result

    rollback_root = Path(tempfile.mkdtemp(prefix="rejudge-rollback-"))
    replaced_files = [
        evaluation_path,
        dense.metadata_root / "run.json",
        hybrid.metadata_root / "run.json",
        *dense.manifest_paths,
        *hybrid.manifest_paths,
    ]
    backups: dict[Path, Path] = {}
    report_targets = _report_targets(request)
    report_backups: dict[Path, Path] = {}
    try:
        for index, path in enumerate(dict.fromkeys(replaced_files)):
            backup = rollback_root / "files" / str(index)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            backups[path] = backup
        for index, target in enumerate(report_targets):
            if target.exists():
                backup = rollback_root / "reports" / str(index)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(backup))
                report_backups[target] = backup

        atomic_promote(staged_evaluation, evaluation_path)
        _update_run_record(dense.metadata_root / "run.json", new_sha256)
        _update_run_record(hybrid.metadata_root / "run.json", new_sha256)
        for path in (*dense.manifest_paths, *hybrid.manifest_paths):
            _update_manifest(path, old_sha256, new_sha256)
        reports = _run_metrics(request, metrics_runner)
        shutil.rmtree(rollback_root, ignore_errors=True)
        shutil.rmtree(temporary_root, ignore_errors=True)
        return replace(result, reports=reports, applied=True)
    except Exception:
        for target in report_targets:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        for target, backup in report_backups.items():
            if backup.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup), str(target))
        for target, backup in backups.items():
            atomic_promote(backup, target)
        shutil.rmtree(rollback_root, ignore_errors=True)
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
