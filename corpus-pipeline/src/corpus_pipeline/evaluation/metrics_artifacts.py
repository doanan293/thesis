from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    write_json,
)
from corpus_pipeline.evaluation.run_workspace import RerankVariantRecord
from corpus_pipeline.evaluation.variant_identity import MetricsArtifactIdentity
from corpus_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class MetricsArtifactResult:
    artifact_dir: Path
    report_path: Path
    results_path: Path
    metrics_sha256: str
    model: str | None = None
    variant_sha256: str | None = None


def select_rerank_variants(
    variants: dict[str, RerankVariantRecord],
    *,
    model: str | None = None,
    variant: str | None = None,
) -> dict[str, RerankVariantRecord]:
    if model is not None and variant is not None:
        raise ValueError("--model and --variant are mutually exclusive")
    if variant is not None:
        matches = {
            digest: item
            for digest, item in variants.items()
            if digest.startswith(variant)
        }
        if not matches:
            raise ValueError(
                f"Unknown rerank variant {variant}; available: {', '.join(sorted(variants))}"
            )
        if len(matches) != 1:
            raise ValueError(f"Rerank variant prefix {variant} is ambiguous")
        return matches
    selected = {
        digest: item
        for digest, item in variants.items()
        if model is None or item.model == model
    }
    if model is not None and not selected:
        available = sorted({item.model for item in variants.values()})
        raise ValueError(
            f"Unknown reranker model {model}; available: {', '.join(available)}"
        )
    return dict(sorted(selected.items()))


def _report_parent(
    run_root: Path,
    model: str | None,
    variant_sha256: str | None,
) -> Path:
    if model is None:
        return Path(run_root) / "reports" / "baseline"
    model_slug = require_model(model).slug
    variant_prefix = (variant_sha256 or "")[:12]
    return Path(run_root) / "reports" / "rerank" / model_slug / variant_prefix


def _report_dir(parent: Path, metrics_sha256: str) -> Path:
    for length in range(12, 65, 4):
        candidate = parent / metrics_sha256[:length]
        if not candidate.exists():
            return candidate
        try:
            manifest = json.loads(
                (candidate / "manifest.json").read_text(encoding="utf-8")
            )
            existing = manifest.get("identity") or {}
            if existing.get("metrics_sha256") == metrics_sha256:
                return candidate
            if not isinstance(existing, dict) or "metrics_sha256" not in existing:
                raise ArtifactContractError(
                    f"Existing metrics artifact is invalid: {candidate}"
                )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise ArtifactContractError(
                f"Existing metrics artifact is invalid: {candidate}"
            ) from exc
    raise ArtifactContractError(
        f"Unable to allocate metrics artifact path for {metrics_sha256}"
    )


def _load_metrics_bundle(
    path: Path, identity: MetricsArtifactIdentity
) -> ArtifactBundle:
    bundle = load_bundle(
        path,
        expected_type="metrics_report",
        require_complete=True,
    )
    if bundle.manifest.identity.get("metrics_sha256") != identity.sha256:
        raise ArtifactContractError(f"Metrics identity mismatch at {path}")
    return bundle


def publish_metrics_artifact(
    run_root: Path,
    identity: MetricsArtifactIdentity,
    metrics: dict,
    breakdowns: dict,
    results: list[dict],
    model: str | None = None,
    variant_sha256: str | None = None,
) -> MetricsArtifactResult:
    from corpus_pipeline.evaluation.metrics_service import (
        markdown_breakdown_tables,
        markdown_metric_table,
    )

    parent = _report_parent(run_root, model, variant_sha256)
    target = _report_dir(parent, identity.sha256)
    if target.exists():
        bundle = _load_metrics_bundle(target, identity)
        return MetricsArtifactResult(
            target,
            target / "report.md",
            bundle.data_path,
            identity.sha256,
            model,
            variant_sha256,
        )
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity.sha256[:12]}.", dir=parent))
    try:
        results_path = temporary / "metrics.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
        report_path = temporary / "report.md"
        report_path.write_text(
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
        bundle = _load_metrics_bundle(temporary, identity)
        os.replace(temporary, target)
        temporary = Path()
        promoted = _load_metrics_bundle(target, identity)
        return MetricsArtifactResult(
            target,
            target / "report.md",
            promoted.data_path,
            identity.sha256,
            model,
            variant_sha256,
        )
    finally:
        if str(temporary) not in {"", "."}:
            shutil.rmtree(temporary, ignore_errors=True)
