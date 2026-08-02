from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_pipeline.corpus.validation.validate_final_rag import (
    ValidationReport,
    validate_final_rag,
    write_json_report,
    write_markdown_report,
)


@dataclass(frozen=True)
class ValidationRequest:
    final_dir: Path
    output_json: Path
    output_markdown: Path | None


@dataclass(frozen=True)
class ValidationResult:
    output_json: Path
    output_markdown: Path | None
    ok: bool
    errors: tuple[str, ...]
    metrics: dict[str, Any]


def run_validation(request: ValidationRequest) -> ValidationResult:
    report: ValidationReport = validate_final_rag(
        final_dir=request.final_dir,
        include_deep_audit=True,
    )
    write_json_report(request.output_json, report)
    if request.output_markdown is not None:
        write_markdown_report(request.output_markdown, report)
    return ValidationResult(
        output_json=request.output_json,
        output_markdown=request.output_markdown,
        ok=report.ok,
        errors=tuple(report.errors),
        metrics=dict(report.metrics),
    )
