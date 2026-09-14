from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    write_json,
)
from seed_pipeline.evaluation.run_workspace import RunConflictError, replace_directory
from seed_pipeline.evaluation.variant_identity import MetricsArtifactIdentity
from seed_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class MetricsArtifactResult:
    artifact_dir: Path
    report_path: Path
    results_path: Path
    metrics_sha256: str
    model: str | None = None
    variant_sha256: str | None = None


def report_dir(
    run_root: Path, *, top_k: int, window_size: int, model: str | None = None
) -> Path:
    name = f"top{top_k}-window{window_size}"
    if model is None:
        return Path(run_root) / "reports" / "baseline" / name
    return Path(run_root) / "reports" / "rerank" / require_model(model).slug / name


def _load_metrics_bundle(path: Path) -> ArtifactBundle:
    return load_bundle(path, expected_type="metrics_report", require_complete=True)


def publish_metrics_artifact(
    run_root: Path,
    identity: MetricsArtifactIdentity,
    metrics: dict,
    breakdowns: dict,
    results: list[dict],
    *,
    top_k: int,
    window_size: int,
    model: str | None = None,
    variant_sha256: str | None = None,
    force: bool = False,
) -> MetricsArtifactResult:
    from seed_pipeline.evaluation.metrics_service import (
        markdown_breakdown_tables,
        markdown_metric_table,
    )

    target = report_dir(run_root, top_k=top_k, window_size=window_size, model=model)
    if target.exists() and not force:
        existing = _load_metrics_bundle(target)
        existing_sha256 = existing.manifest.identity.get("metrics_sha256")
        if existing_sha256 != identity.sha256:
            raise RunConflictError(
                f"Metrics report {target} already exists for different inputs "
                f"(metrics_sha256 {existing_sha256} != {identity.sha256}); use --force"
            )
        return MetricsArtifactResult(
            target,
            target / "report.md",
            existing.data_path,
            identity.sha256,
            model,
            variant_sha256,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        results_path = temporary / "metrics.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
        (temporary / "report.md").write_text(
            "# Retrieval Metrics\n\n"
            + markdown_metric_table(metrics)
            + "\n\n"
            + markdown_breakdown_tables(breakdowns)
            + "\n",
            encoding="utf-8",
        )
        manifest = ArtifactManifest.create(
            artifact_type="metrics_report",
            data_path=results_path,
            record_count=len(results),
            identity={
                **identity.payload,
                "metrics_sha256": identity.sha256,
                "model": model,
                "variant_sha256": variant_sha256,
            },
        )
        payload = manifest.to_dict()
        payload.update(total=len(results), complete=len(results), missing=0)
        write_json(temporary / "manifest.json", payload)
        _load_metrics_bundle(temporary)
        replace_directory(temporary, target)
        temporary = Path()
    finally:
        if str(temporary) != ".":
            shutil.rmtree(temporary, ignore_errors=True)
    promoted = _load_metrics_bundle(target)
    if promoted.manifest.identity.get("metrics_sha256") != identity.sha256:
        raise ArtifactContractError(f"Metrics identity mismatch at {target}")
    return MetricsArtifactResult(
        target,
        target / "report.md",
        promoted.data_path,
        identity.sha256,
        model,
        variant_sha256,
    )
