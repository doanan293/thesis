from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from corpus_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    sha256_file,
    write_json,
)
from corpus_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from corpus_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunWorkspace,
    load_run_record,
)
from corpus_pipeline.evaluation.variant_identity import RerankVariantIdentity
from corpus_pipeline.runtime.catalog import require_model


def _model_slug(model: str) -> str:
    return require_model(model).slug


def variant_artifact_dir(
    workspace: RunWorkspace,
    model_slug: str,
    variant_sha256: str,
) -> Path:
    parent = workspace.root / "rerank" / model_slug
    for length in range(12, 65, 4):
        candidate = parent / variant_sha256[:length]
        if not candidate.exists():
            return candidate
        try:
            manifest = json.loads(
                (candidate / "manifest.json").read_text(encoding="utf-8")
            )
            existing_manifest_identity = manifest.get("identity")
            if (
                not isinstance(existing_manifest_identity, dict)
                or "variant_sha256" not in existing_manifest_identity
            ):
                raise ArtifactContractError(
                    f"Existing rerank variant is invalid: {candidate}"
                )
            existing_identity = existing_manifest_identity["variant_sha256"]
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise ArtifactContractError(
                f"Existing rerank variant is invalid: {candidate}"
            ) from exc
        if existing_identity == variant_sha256:
            return candidate
    raise ArtifactContractError(
        f"Unable to allocate rerank variant path for {variant_sha256}"
    )


def _load_complete_variant(path: Path, variant_sha256: str) -> ArtifactBundle:
    bundle = load_bundle(
        path,
        expected_type="rerank_score_cache",
        require_complete=True,
    )
    if bundle.manifest.identity.get("variant_sha256") != variant_sha256:
        raise ArtifactContractError(f"Rerank variant identity mismatch at {path}")
    return bundle


def finalize_run_rerank_bundle(
    *,
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
    cache_path: Path,
    identity: RerankVariantIdentity,
) -> ArtifactBundle:
    model = str(identity.payload["reranker"])
    spec = require_model(model)
    if str(identity.payload["model_sha256"]) != spec.sha256:
        raise ArtifactContractError(
            f"Rerank model digest does not match catalog for {model}"
        )
    contract = str(identity.payload["request_contract_sha256"])
    protocol = str(identity.payload["protocol"])
    cache = RerankScoreCache(
        cache_path,
        model_sha256=spec.sha256,
        request_contract_sha256=contract,
        rewrite_legacy=False,
    )
    records = cache.subset_records(candidate_bundle.data_path, model)
    target = variant_artifact_dir(
        workspace,
        _model_slug(model),
        identity.sha256,
    )
    if target.exists():
        bundle = _load_complete_variant(target, identity.sha256)
        workspace.register_rerank_variant(
            identity.sha256,
            RerankVariantRecord(
                model,
                target.relative_to(workspace.root).as_posix(),
            ),
        )
        return bundle

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        data_path = temporary / "rerank_scores.jsonl"
        with data_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        data_sha256 = sha256_file(data_path)
        manifest = ArtifactManifest.create(
            artifact_type="rerank_score_cache",
            data_path=data_path,
            record_count=len(records),
            identity={
                **identity.payload,
                "variant_sha256": identity.sha256,
                "protocol": protocol,
                "pair_count": len(records),
            },
        )
        payload = manifest.to_dict()
        payload.update(total=len(records), complete=len(records), missing=0)
        write_json(temporary / "manifest.json", payload)
        bundle = _load_complete_variant(temporary, identity.sha256)
        os.replace(temporary, target)
        temporary = Path()
        promoted = _load_complete_variant(target, identity.sha256)
        workspace.register_rerank_variant(
            identity.sha256,
            RerankVariantRecord(
                model,
                target.relative_to(workspace.root).as_posix(),
            ),
        )
        del bundle, data_sha256
        return promoted
    finally:
        if str(temporary) not in {"", "."}:
            shutil.rmtree(temporary, ignore_errors=True)


def load_registered_rerank_bundle(
    workspace: RunWorkspace,
    variant_sha256: str,
    record: RerankVariantRecord,
) -> ArtifactBundle:
    if record.status != "complete":
        raise ArtifactContractError(f"Rerank variant {variant_sha256} is not complete")
    bundle = _load_complete_variant(
        workspace.resolve_relative_path(record.artifact_dir), variant_sha256
    )
    if bundle.manifest.identity.get("reranker") != record.model:
        raise ArtifactContractError(
            f"Rerank variant model mismatch for {variant_sha256}"
        )
    return bundle


def migrate_legacy_rerank(
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
) -> tuple[str, RerankVariantRecord] | None:
    record = load_run_record(workspace.root / "run.json")
    legacy = record.legacy_rerank
    if legacy is None:
        return None
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256,
        legacy.model,
    )
    bundle = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=Path(legacy.cache_path),
        identity=identity,
    )
    del bundle
    registered = load_run_record(workspace.root / "run.json").rerank_variants[
        identity.sha256
    ]
    return identity.sha256, registered
