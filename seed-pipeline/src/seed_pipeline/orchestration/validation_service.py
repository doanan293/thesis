from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.artifacts.contract import validate_contract_directory
from seed_pipeline.corpus.validation.validate_final_rag import (
    ValidationReport,
    validate_unified_chunks,
    write_json_report,
    write_markdown_report,
)


@dataclass(frozen=True)
class ValidationRequest:
    final_dir: Path
    output_json: Path | None = None
    output_markdown: Path | None = None


@dataclass(frozen=True)
class ValidationResult:
    output_json: Path | None
    output_markdown: Path | None
    ok: bool
    errors: tuple[str, ...]
    metrics: dict[str, Any]


def _destination_error(final_dir: Path, destination: Path | None) -> str | None:
    if destination is None:
        return None
    if destination.resolve().is_relative_to(final_dir.resolve()):
        return f"Validation report must be outside published bundle: {destination}"
    return None


def run_validation(request: ValidationRequest) -> ValidationResult:
    destination_errors = [
        error
        for destination in (request.output_json, request.output_markdown)
        if (error := _destination_error(request.final_dir, destination)) is not None
    ]
    if destination_errors:
        report = ValidationReport(ok=False, errors=destination_errors)
    else:
        try:
            manifest = validate_contract_directory(request.final_dir)
            unified_report = validate_unified_chunks(request.final_dir / "chunks.jsonl")
        except (OSError, RuntimeError, ValueError) as exc:
            report = ValidationReport(ok=False, errors=[str(exc)])
        else:
            report = ValidationReport(
                ok=unified_report.ok,
                errors=list(unified_report.errors),
                warnings=list(unified_report.warnings),
                metrics={
                    "schema_version": manifest["schema_version"],
                    "build_id": manifest["build_id"],
                    "section_count": manifest["section_count"],
                    "chunk_count": manifest["chunk_count"],
                },
            )

    written_json = None
    written_markdown = None
    if not destination_errors and request.output_json is not None:
        write_json_report(request.output_json, report)
        written_json = request.output_json
    if not destination_errors and request.output_markdown is not None:
        write_markdown_report(request.output_markdown, report)
        written_markdown = request.output_markdown
    return ValidationResult(
        output_json=written_json,
        output_markdown=written_markdown,
        ok=report.ok,
        errors=tuple(report.errors),
        metrics=dict(report.metrics),
    )
