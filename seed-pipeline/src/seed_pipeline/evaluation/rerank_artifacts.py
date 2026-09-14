from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    write_json,
)
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunConflictError,
    RunWorkspace,
    replace_directory,
)
from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity
from seed_pipeline.runtime.catalog import require_model


def _load_complete_variant(path: Path) -> ArtifactBundle:
    return load_bundle(path, expected_type="rerank_score_cache", require_complete=True)


def finalize_run_rerank_bundle(
    *,
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
    cache_path: Path,
    identity: RerankVariantIdentity,
    force: bool = False,
) -> ArtifactBundle:
    model = str(identity.payload["reranker"])
    spec = require_model(model)
    if str(identity.payload["model_sha256"]) != spec.sha256:
        raise ArtifactContractError(
            f"Rerank model digest does not match catalog for {model}"
        )
    target = workspace.rerank_dir(spec.slug)
    record = RerankVariantRecord(
        model, identity.sha256, target.relative_to(workspace.root).as_posix()
    )
    if target.exists() and not force:
        existing = _load_complete_variant(target)
        existing_sha256 = existing.manifest.identity.get("variant_sha256")
        if existing_sha256 != identity.sha256:
            raise RunConflictError(
                f"Rerank variant {target} already exists for different inputs "
                f"(variant_sha256 {existing_sha256} != {identity.sha256}); use --force"
            )
        workspace.register_rerank_variant(spec.slug, record)
        return existing
    cache = RerankScoreCache(
        cache_path,
        model_sha256=spec.sha256,
        request_contract_sha256=str(identity.payload["request_contract_sha256"]),
    )
    records = cache.subset_records(candidate_bundle.data_path, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        data_path = temporary / "rerank_scores.jsonl"
        with data_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
        manifest = ArtifactManifest.create(
            artifact_type="rerank_score_cache",
            data_path=data_path,
            record_count=len(records),
            identity={
                **identity.payload,
                "variant_sha256": identity.sha256,
                "pair_count": len(records),
            },
        )
        payload = manifest.to_dict()
        payload.update(total=len(records), complete=len(records), missing=0)
        write_json(temporary / "manifest.json", payload)
        _load_complete_variant(temporary)
        replace_directory(temporary, target)
        temporary = Path()
    finally:
        if str(temporary) != ".":
            shutil.rmtree(temporary, ignore_errors=True)
    promoted = _load_complete_variant(target)
    if promoted.manifest.identity.get("variant_sha256") != identity.sha256:
        raise ArtifactContractError(f"Rerank variant identity mismatch at {target}")
    workspace.register_rerank_variant(spec.slug, record)
    return promoted


def load_registered_rerank_bundle(
    workspace: RunWorkspace, record: RerankVariantRecord
) -> ArtifactBundle:
    bundle = _load_complete_variant(workspace.resolve(record.artifact_dir))
    identity = bundle.manifest.identity
    if identity.get("variant_sha256") != record.variant_sha256:
        raise ArtifactContractError(
            f"Rerank variant at {bundle.root} does not match run.json"
        )
    if identity.get("reranker") != record.model:
        raise ArtifactContractError(f"Rerank variant model mismatch at {bundle.root}")
    return bundle
