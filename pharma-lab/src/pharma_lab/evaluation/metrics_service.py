from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pharma_lab.artifacts.bundle import load_bundle
from pharma_lab.artifacts.jsonl import iter_jsonl_objects
from pharma_lab.config.defaults import DEFAULT_TOP_K
from pharma_lab.config.paths import DATA_DIR
from pharma_lab.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
)
from pharma_lab.evaluation.metrics_artifacts import (
    MetricsArtifactResult,
    publish_metrics_artifact,
)
from pharma_lab.evaluation.relevance_judgments import (
    LoadedJudgments,
    judgments_path,
    load_judgments,
)
from pharma_lab.evaluation.rerank_artifacts import load_registered_rerank_bundle
from pharma_lab.evaluation.rerank_score_cache import (
    RerankScoreCache,
    RerankScoreCacheError,
)
from pharma_lab.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from pharma_lab.evaluation.retrieval_metrics import (
    accumulate_metrics,
    new_metric_bucket,
    score_ranked_payloads,
)
from pharma_lab.evaluation.retrieval_types import RetrievalCandidate
from pharma_lab.evaluation.run_workspace import (
    RerankVariantRecord,
    RunWorkspace,
    load_run_record,
    resolve_evaluation_reference,
)
from pharma_lab.evaluation.variant_identity import MetricsArtifactIdentity
from pharma_lab.runtime.catalog import require_model

DISPLAY_HIT_KS = (3, 5, 10, 30)
BREAKDOWN_DIMENSIONS = ("eval_group", "difficulty")
EVAL_GROUP_DISPLAY_LABELS = {
    "leaflet": "brand_product_qa",
    "chunk_risk": "chunk_level_retrieval",
}


def validate_metrics_cutoff(top_k: int, candidate_k: int) -> None:
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if top_k > candidate_k:
        raise ValueError(
            f"--top-k={top_k} exceeds run candidate-k={candidate_k}; "
            "rerun retrieval with a deeper candidate artifact"
        )


def markdown_metric_table(metrics: dict) -> str:
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total Queries | {metrics['count']} |",
    ]
    if metrics["count"] == 0:
        return "\n".join(lines)
    for k in DISPLAY_HIT_KS:
        key = f"hit@{k}"
        if key in metrics:
            lines.append(f"| Hit@{k} | {metrics[key] / metrics['count'] * 100:.2f}% |")
    lines.append(f"| MRR | {metrics['mrr'] / metrics['count']:.4f} |")
    multi_count = int(metrics.get("multi_count", 0))
    if multi_count:
        lines.extend(
            [
                "",
                "## Multi-required queries",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Total Multi-required Queries | {multi_count} |",
            ]
        )
        for k in DISPLAY_HIT_KS:
            recall_key = f"multi_section_recall@{k}"
            all_hit_key = f"multi_all_hit@{k}"
            if recall_key in metrics:
                lines.append(
                    f"| Multi-section Recall@{k} | "
                    f"{metrics[recall_key] / multi_count * 100:.2f}% |"
                )
            if all_hit_key in metrics:
                lines.append(
                    f"| Multi-all-hit@{k} | "
                    f"{metrics[all_hit_key] / multi_count * 100:.2f}% |"
                )
    return "\n".join(lines)


def accumulate_breakdown_metrics(breakdowns: dict, hits: dict, row: dict) -> None:
    for dimension in BREAKDOWN_DIMENSIONS:
        value = str(row.get(dimension) or "unknown")
        accumulate_metrics(breakdowns[dimension][value], hits, row)


def markdown_breakdown_tables(breakdowns: dict) -> str:
    sections = []
    for dimension in BREAKDOWN_DIMENSIONS:
        lines = [
            f"## Breakdown by {dimension}",
            "",
            f"| {dimension} | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for value, metrics in sorted(breakdowns[dimension].items()):
            count = metrics["count"]
            display_value = (
                EVAL_GROUP_DISPLAY_LABELS.get(value, value)
                if dimension == "eval_group"
                else value
            )
            cells = [display_value, str(int(count))]
            cells.extend(
                f"{metrics.get(f'hit@{k}', 0.0) / count * 100:.2f}%"
                for k in DISPLAY_HIT_KS
            )
            cells.append(f"{metrics.get('mrr', 0.0) / count:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


@dataclass(frozen=True)
class MetricsRequest:
    run_root: Path
    top_k: int = DEFAULT_TOP_K
    window_size: int = 3
    model: str | None = None
    force: bool = False


@dataclass(frozen=True)
class MetricsResult:
    baseline: MetricsArtifactResult
    reranked: tuple[MetricsArtifactResult, ...]


@dataclass(frozen=True)
class RerankMetricInput:
    variant_sha256: str
    model: str
    cache: RerankScoreCache


@dataclass(frozen=True)
class MetricInputs:
    evaluation_rows: dict[str, dict]
    candidates: CandidateArtifactReader
    candidate_data_sha256: str
    evaluation_sha256: str
    rerank_variants: tuple[RerankMetricInput, ...]
    run_root: Path
    judgments: LoadedJudgments


def select_rerank_variants(
    variants: dict[str, RerankVariantRecord], *, model: str | None
) -> dict[str, RerankVariantRecord]:
    if model is None:
        return dict(sorted(variants.items()))
    slug = require_model(model).slug
    if slug not in variants:
        available = ", ".join(sorted(item.model for item in variants.values()))
        raise ValueError(f"Unknown reranker model {model}; available: {available}")
    return {slug: variants[slug]}


def load_and_validate_metric_inputs(request: MetricsRequest) -> MetricInputs:
    record = load_run_record(request.run_root / "run.json")
    validate_metrics_cutoff(request.top_k, record.identity.candidate_k)
    workspace = RunWorkspace(request.run_root, record.identity, record.origin)
    if record.candidates_dir is None:
        raise ArtifactContractError(
            f"Run {request.run_root} has no candidates; run pharma-lab retrieve first"
        )
    candidate_bundle = load_bundle(
        workspace.resolve(record.candidates_dir),
        expected_type="retrieval_candidates",
        require_complete=True,
    )
    evaluation = resolve_evaluation_reference(record.identity.evaluation_path, DATA_DIR)
    if sha256_file(evaluation) != record.identity.evaluation_sha256:
        raise ArtifactContractError(
            "Run evaluation input changed after retrieval; create a new --run"
        )
    # Judgments change only what counts as a hit, so the run's candidates stay valid.
    judgments = load_judgments(
        judgments_path(evaluation),
        evaluation_sha256=record.identity.evaluation_sha256,
    )
    rows = {str(row["query_id"]): row for row in iter_jsonl_objects(evaluation)}
    rerank_inputs: list[RerankMetricInput] = []
    for variant in select_rerank_variants(
        record.rerank_variants, model=request.model
    ).values():
        score_bundle = load_registered_rerank_bundle(workspace, variant)
        spec = require_model(variant.model)
        if spec.rerank_contract is None:
            raise RerankScoreCacheError(
                f"Reranker {variant.model} has no scoring contract"
            )
        score_cache = RerankScoreCache(
            score_bundle.data_path,
            model_sha256=spec.sha256,
            request_contract_sha256=spec.rerank_contract.sha256,
        )
        subset = score_cache.validate_subset(candidate_bundle.data_path, variant.model)
        if not subset.is_complete:
            raise RerankScoreCacheError(
                f"Rerank score cache is missing {subset.missing} records"
            )
        rerank_inputs.append(
            RerankMetricInput(variant.variant_sha256, variant.model, score_cache)
        )
    return MetricInputs(
        rows,
        CandidateArtifactReader.from_data_path(candidate_bundle.data_path),
        candidate_bundle.manifest.data_sha256,
        record.identity.evaluation_sha256,
        tuple(rerank_inputs),
        request.run_root,
        judgments,
    )


def _records(
    inputs: MetricInputs,
    top_k: int,
    window_size: int,
    rerank_input: RerankMetricInput | None = None,
):
    metrics = new_metric_bucket()
    breakdowns = {
        dimension: defaultdict(new_metric_bucket) for dimension in BREAKDOWN_DIMENSIONS
    }
    output = []
    for record in inputs.candidates:
        row = inputs.evaluation_rows[record["query_id"]]
        candidates = [
            RetrievalCandidate(
                chunk_id=item["chunk_id"],
                score=float(item["retrieval_score"]),
                rank=int(item["retrieval_rank"]),
                source=str(item["source"]),
                payload=dict(item.get("payload") or {}),
                document_text=str(item["document_text"]),
                document_hash=str(item["document_hash"]),
            )
            for item in record["candidates"]
        ]
        if rerank_input is not None:
            scored = [
                (rerank_input.cache.require_score(rerank_input.model, row, c), c)
                for c in candidates
            ]
            candidates = [
                c.with_rerank_score(score, rank)
                for rank, (score, c) in enumerate(
                    sorted(scored, key=lambda x: -x[0]), 1
                )
            ]
        payloads = [c.payload for c in candidates]
        hits = score_ranked_payloads(
            payloads,
            row,
            top_k=top_k,
            window_size=window_size,
            judgments=inputs.judgments.for_query(str(record["query_id"])),
        )
        accumulate_metrics(metrics, hits, row)
        accumulate_breakdown_metrics(breakdowns, hits, row)
        output.append({"query_id": record["query_id"], **hits})
    return (
        dict(metrics),
        {
            dimension: {
                value: dict(bucket) for value, bucket in dimension_buckets.items()
            }
            for dimension, dimension_buckets in breakdowns.items()
        },
        output,
    )


def run_metrics(request: MetricsRequest) -> MetricsResult:
    inputs = load_and_validate_metric_inputs(request)
    baseline, baseline_breakdowns, baseline_rows = _records(
        inputs, request.top_k, request.window_size
    )
    baseline_identity = MetricsArtifactIdentity.create(
        evaluation_sha256=inputs.evaluation_sha256,
        candidate_data_sha256=inputs.candidate_data_sha256,
        top_k=request.top_k,
        window_size=request.window_size,
        judgments_sha256=inputs.judgments.sha256,
    )
    baseline_artifact = publish_metrics_artifact(
        inputs.run_root,
        baseline_identity,
        baseline,
        baseline_breakdowns,
        baseline_rows,
        top_k=request.top_k,
        window_size=request.window_size,
        force=request.force,
    )
    reranked_artifacts: list[MetricsArtifactResult] = []
    for rerank_input in inputs.rerank_variants:
        reranked, breakdowns, rows = _records(
            inputs, request.top_k, request.window_size, rerank_input
        )
        identity = MetricsArtifactIdentity.create(
            evaluation_sha256=inputs.evaluation_sha256,
            candidate_data_sha256=inputs.candidate_data_sha256,
            top_k=request.top_k,
            window_size=request.window_size,
            rerank_variant_sha256=rerank_input.variant_sha256,
            judgments_sha256=inputs.judgments.sha256,
        )
        reranked_artifacts.append(
            publish_metrics_artifact(
                inputs.run_root,
                identity,
                reranked,
                breakdowns,
                rows,
                top_k=request.top_k,
                window_size=request.window_size,
                model=rerank_input.model,
                variant_sha256=rerank_input.variant_sha256,
                force=request.force,
            )
        )
    return MetricsResult(baseline_artifact, tuple(reranked_artifacts))
