from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharma_lab.evaluation.build_section_retrieval_eval import (
    DEFAULT_TARGET_ROWS,
    build_jsonl,
)
from pharma_lab.evaluation.patient_query_generation import (
    DEFAULT_TARGET_COUNT,
    build_patient_queries,
)


@dataclass(frozen=True)
class EvaluationBuildRequest:
    sections_path: Path
    chunks_path: Path
    output_dir: Path
    patient_query_count: int = DEFAULT_TARGET_COUNT
    evaluation_row_count: int = DEFAULT_TARGET_ROWS


@dataclass(frozen=True)
class EvaluationBuildResult:
    patient_queries_path: Path
    evaluation_path: Path
    patient_query_count: int
    evaluation_row_count: int


def build_evaluation_dataset(
    request: EvaluationBuildRequest,
) -> EvaluationBuildResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    patient_path = request.output_dir / "patient_queries.json"
    evaluation_path = request.output_dir / "section_retrieval_eval.jsonl"
    patient_rows = build_patient_queries(
        request.sections_path,
        request.chunks_path,
        patient_path,
        request.patient_query_count,
    )
    evaluation_rows = build_jsonl(
        sections_path=request.sections_path,
        chunks_path=request.chunks_path,
        output_path=evaluation_path,
        target_rows=request.evaluation_row_count,
        patient_queries_path=patient_path,
    )
    return EvaluationBuildResult(
        patient_queries_path=patient_path,
        evaluation_path=evaluation_path,
        patient_query_count=len(patient_rows),
        evaluation_row_count=len(evaluation_rows),
    )
