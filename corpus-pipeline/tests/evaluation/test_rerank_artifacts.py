import json

import pytest

from corpus_pipeline.artifacts.bundle import load_bundle
from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError
from corpus_pipeline.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
    migrate_legacy_rerank,
    variant_artifact_dir,
)
from corpus_pipeline.evaluation.run_workspace import RunWorkspace, load_run_record
from corpus_pipeline.evaluation.variant_identity import RerankVariantIdentity


def test_finalize_run_bundle_selects_exact_subset_and_is_idempotent(
    complete_run, candidate_bundle, complete_rerank_cache
):
    workspace = RunWorkspace(
        complete_run,
        load_run_record(complete_run / "run.json").identity,
    )
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256,
        "qwen3-reranker:0.6b-fp16",
    )

    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=load_bundle(
            candidate_bundle.data_path.parent,
            expected_type="retrieval_candidates",
            require_complete=True,
        ),
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    second = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=load_bundle(
            candidate_bundle.data_path.parent,
            expected_type="retrieval_candidates",
            require_complete=True,
        ),
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )

    assert first.root == second.root
    assert first.manifest.identity["variant_sha256"] == identity.sha256
    assert first.completion.is_complete


def test_variant_path_rejects_invalid_existing_target(complete_run):
    workspace = RunWorkspace(
        complete_run,
        load_run_record(complete_run / "run.json").identity,
    )
    target = complete_run / "rerank" / "model" / ("a" * 12)
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="Existing rerank variant"):
        variant_artifact_dir(workspace, "model", "a" * 64)


def test_variant_path_extends_prefix_for_different_identity(complete_run):
    workspace = RunWorkspace(
        complete_run,
        load_run_record(complete_run / "run.json").identity,
    )
    target = complete_run / "rerank" / "model" / ("a" * 12)
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(
        '{"schema_version":1,"artifact_type":"rerank_score_cache",'
        '"created_at":"now","data_filename":"scores.jsonl",'
        '"data_sha256":"hash","record_count":0,"identity":'
        '{"variant_sha256":"different"}}',
        encoding="utf-8",
    )

    assert variant_artifact_dir(workspace, "model", "a" * 64).name == "a" * 16


def test_legacy_migration_registers_bundle_without_rewriting_cache(
    complete_run, candidate_bundle, complete_rerank_cache
):
    run_path = complete_run / "run.json"
    current = load_run_record(run_path)
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": current.identity.__dict__,
                "status": "reranked",
                "candidates_dir": "candidates",
                "rerank_scores_dir": str(complete_rerank_cache.path),
                "reranker": "qwen3-reranker:0.6b-fp16",
            }
        ),
        encoding="utf-8",
    )
    before = complete_rerank_cache.path.read_bytes()

    migrated = migrate_legacy_rerank(
        RunWorkspace(complete_run, current.identity), candidate_bundle
    )

    assert migrated is not None
    assert complete_rerank_cache.path.read_bytes() == before
    assert load_run_record(run_path).schema_version == 2
    assert migrated[0] in load_run_record(run_path).rerank_variants


def test_incomplete_legacy_cache_does_not_write_registry(
    complete_run, candidate_bundle, tmp_path
):
    run_path = complete_run / "run.json"
    current = load_run_record(run_path)
    cache_path = tmp_path / "incomplete.jsonl"
    cache_path.write_text("", encoding="utf-8")
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": current.identity.__dict__,
                "status": "retrieved",
                "candidates_dir": "candidates",
                "rerank_scores_dir": str(cache_path),
                "reranker": "qwen3-reranker:0.6b-fp16",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing 1 records"):
        migrate_legacy_rerank(
            RunWorkspace(complete_run, current.identity), candidate_bundle
        )

    assert load_run_record(run_path).schema_version == 1
    assert not (complete_run / "rerank").exists()


def test_split_workspace_finalizes_and_reuses_variant_under_artifact_root(
    complete_run, candidate_bundle, complete_rerank_cache, tmp_path
):
    current = load_run_record(complete_run / "run.json")
    heavy = tmp_path / "heavy" / "retrieval_eval" / "run"
    workspace = RunWorkspace(complete_run, current.identity, artifact_root=heavy)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256,
        "qwen3-reranker:0.6b-fp16",
    )

    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    record = load_run_record(complete_run / "run.json").rerank_variants[
        identity.sha256
    ]
    second = load_registered_rerank_bundle(workspace, identity.sha256, record)

    assert first.root.is_relative_to(heavy / "rerank")
    assert second.root == first.root
    assert record.artifact_dir == first.root.relative_to(heavy).as_posix()
    assert not (complete_run / "rerank").exists()
