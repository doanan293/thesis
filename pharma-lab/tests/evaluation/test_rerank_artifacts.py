from pathlib import Path

import pytest

from pharma_lab.artifacts.bundle import ArtifactBundle
from pharma_lab.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
)
from pharma_lab.evaluation.rerank_score_cache import RerankScoreCache
from pharma_lab.evaluation.run_workspace import (
    RunConflictError,
    RunWorkspace,
    load_run_record,
)
from pharma_lab.evaluation.variant_identity import RerankVariantIdentity

MODEL = "qwen3-reranker:0.6b-fp16"
SLUG = "qwen3_reranker_0_6b_fp16"


def workspace_for(run: Path) -> RunWorkspace:
    record = load_run_record(run / "run.json")
    return RunWorkspace(run, record.identity, record.origin)


def test_variant_is_published_under_the_model_slug_and_reused(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )

    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    second = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )

    assert first.root == second.root == complete_run / "rerank" / SLUG
    assert first.completion.is_complete
    record = workspace.variant_records()[SLUG]
    assert (record.model, record.variant_sha256, record.artifact_dir) == (
        MODEL,
        identity.sha256,
        f"rerank/{SLUG}",
    )
    assert load_registered_rerank_bundle(workspace, record).root == first.root


def test_a_different_variant_conflicts_unless_forced(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )
    finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    changed = RerankVariantIdentity.from_values(
        candidate_data_sha256=candidate_bundle.manifest.data_sha256,
        model=MODEL,
        model_sha256=str(identity.payload["model_sha256"]),
        protocol=str(identity.payload["protocol"]),
        request_contract_sha256="c" * 64,
    )

    with pytest.raises(RunConflictError, match=f"rerank/{SLUG}"):
        finalize_run_rerank_bundle(
            workspace=workspace,
            candidate_bundle=candidate_bundle,
            cache_path=complete_rerank_cache.path,
            identity=changed,
        )


def test_force_rebuilds_a_variant_with_the_same_identity(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )
    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    (first.root / "stale.txt").write_text("stale", encoding="utf-8")

    second = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
        force=True,
    )

    assert second.root == first.root
    assert not (second.root / "stale.txt").exists()
