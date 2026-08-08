from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.artifacts.bundle import load_bundle
from corpus_pipeline.artifacts.jsonl import iter_jsonl_objects
from corpus_pipeline.config.defaults import DEFAULT_TOP_K
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
)
from corpus_pipeline.evaluation.rerank_score_cache import (
    RerankScoreCache,
    RerankScoreCacheError,
    prompt_contract_hash,
)
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from corpus_pipeline.evaluation.retrieval_metrics import (
    accumulate_metrics,
    new_metric_bucket,
    score_ranked_payloads,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.evaluation.run_workspace import RunWorkspace, load_run_record
from corpus_pipeline.runtime.catalog import require_model

DISPLAY_HIT_KS = (3, 5, 10, 30)


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


@dataclass(frozen=True)
class MetricsRequest:
    run_root: Path
    top_k: int = DEFAULT_TOP_K
    window_size: int = 3
    candidates_dir: Path | None = None
    rerank_scores_dir: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class MetricsResult:
    baseline_report: Path
    baseline_results: Path
    reranked_report: Path | None
    reranked_results: Path | None


@dataclass(frozen=True)
class MetricInputs:
    evaluation_rows: dict[str, dict]
    candidates: CandidateArtifactReader
    rerank_scores: RerankScoreCache | None
    output_dir: Path
    reranker: str | None = None


def load_and_validate_metric_inputs(request: MetricsRequest) -> MetricInputs:
    record = load_run_record(request.run_root / "run.json")
    validate_metrics_cutoff(request.top_k, record.identity.candidate_k)
    candidates_dir = request.candidates_dir or Path(
        record.candidates_dir or request.run_root / "candidates"
    )
    candidate_bundle = load_bundle(
        candidates_dir, expected_type="retrieval_candidates", require_complete=True
    )
    evaluation = Path(record.identity.evaluation_path)
    if sha256_file(evaluation) != record.identity.evaluation_sha256:
        raise ArtifactContractError(
            "Run evaluation input changed after retrieval; create a new --run"
        )
    rows = {}
    for row in iter_jsonl_objects(evaluation):
        rows[str(row["query_id"])] = row
    score_cache = None
    reranker = None
    score_dir = request.rerank_scores_dir or (
        Path(record.rerank_scores_dir) if record.rerank_scores_dir else None
    )
    if score_dir:
        score_path = Path(score_dir)
        if score_path.is_dir():
            score_bundle = load_bundle(
                score_path, expected_type="rerank_score_cache", require_complete=True
            )
            score_path = score_bundle.data_path
            reranker = str(score_bundle.manifest.identity.get("reranker") or "")
        else:
            reranker = str(record.reranker or "")
        if not reranker:
            raise RerankScoreCacheError(
                "Rerank score cache requires the reranker model in run.json"
            )
        spec = require_model(reranker)
        contract = prompt_contract_hash(protocol=spec.reranker_protocol or "")
        score_cache = RerankScoreCache(
            score_path,
            model_sha256=spec.sha256,
            request_contract_sha256=contract,
        )
        subset = score_cache.validate_subset(candidate_bundle.data_path, reranker)
        if not subset.is_complete:
            raise RerankScoreCacheError(
                f"Rerank score cache is missing {subset.missing} records"
            )
    return MetricInputs(
        rows,
        CandidateArtifactReader.from_data_path(candidate_bundle.data_path),
        score_cache,
        request.output_dir or request.run_root / "reports",
        reranker,
    )


def _records(
    inputs: MetricInputs, top_k: int, window_size: int, reranked: bool = False
):
    metrics = new_metric_bucket()
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
        if reranked:
            scored = [
                (inputs.rerank_scores.require_score(inputs.reranker or "", row, c), c)
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
            payloads, row, top_k=top_k, window_size=window_size
        )
        accumulate_metrics(metrics, hits, row)
        output.append({"query_id": record["query_id"], **hits})
    return dict(metrics), output


def _write(output_dir: Path, name: str, metrics: dict, results: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / f"{name}.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = output_dir / f"{name}.md"
    report.write_text(
        f"# {name.title()} Retrieval Metrics\n\n{markdown_metric_table(metrics)}\n",
        encoding="utf-8",
    )
    return report, jsonl


def run_metrics(request: MetricsRequest) -> MetricsResult:
    inputs = load_and_validate_metric_inputs(request)
    baseline, baseline_rows = _records(inputs, request.top_k, request.window_size)
    baseline_report, baseline_results = _write(
        inputs.output_dir, "baseline", baseline, baseline_rows
    )
    if inputs.rerank_scores is None:
        result = MetricsResult(baseline_report, baseline_results, None, None)
        RunWorkspace(
            request.run_root, load_run_record(request.run_root / "run.json").identity
        ).record_reports(inputs.output_dir)
        return result
    reranked, reranked_rows = _records(inputs, request.top_k, request.window_size, True)
    reranked_report, reranked_results = _write(
        inputs.output_dir, "reranked", reranked, reranked_rows
    )
    result = MetricsResult(
        baseline_report, baseline_results, reranked_report, reranked_results
    )
    RunWorkspace(
        request.run_root, load_run_record(request.run_root / "run.json").identity
    ).record_reports(inputs.output_dir)
    return result
