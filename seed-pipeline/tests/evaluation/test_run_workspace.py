import json
from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.evaluation.run_workspace import (
    RUN_SCHEMA_VERSION,
    RerankVariantRecord,
    RunConflictError,
    RunIdentity,
    RunWorkspace,
    evaluation_reference,
    load_run_record,
    resolve_evaluation_reference,
)


def identity(*, candidate_k: int = 30, rrf_k: int = 2) -> RunIdentity:
    return RunIdentity(
        evaluation_path="evaluation/gold/section_retrieval_eval.jsonl",
        evaluation_sha256="e" * 64,
        collection_name="chunks_current",
        embedding_model="qwen3-embedding:4b-fp16",
        query_embeddings_sha256="q" * 64,
        retriever="hybrid",
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        limit=None,
        prefetch_k=50,
    )


def test_new_run_writes_schema_three_with_origin(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(
        tmp_path / "run", identity(), origin="imported"
    )

    payload = json.loads(workspace.record_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RUN_SCHEMA_VERSION == 3
    assert payload["origin"] == "imported"
    assert load_run_record(workspace.record_path).identity == identity()


def test_reopening_a_run_with_the_same_identity_keeps_its_files(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    marker = workspace.candidates_dir / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    RunWorkspace.open_or_create(tmp_path / "run", identity())

    assert marker.read_text(encoding="utf-8") == "keep"


def test_a_different_identity_names_the_run_and_the_fields(tmp_path: Path) -> None:
    RunWorkspace.open_or_create(tmp_path / "run", identity())

    with pytest.raises(RunConflictError, match=r"run .*candidate_k, rrf_k"):
        RunWorkspace.open_or_create(
            tmp_path / "run", identity(candidate_k=10, rrf_k=60)
        )


def test_force_replaces_the_whole_run_tree(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    stale = workspace.reports_dir / "old.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    RunWorkspace.open_or_create(tmp_path / "run", identity(rrf_k=60), force=True)

    assert not stale.exists()
    assert load_run_record(tmp_path / "run" / "run.json").identity.rrf_k == 60


def test_record_candidates_stores_a_relative_directory_only(complete_run: Path) -> None:
    record = load_run_record(complete_run / "run.json")

    assert record.candidates_dir == "candidates"
    assert not (complete_run / "candidates-manifest.json").exists()


def test_registering_a_variant_keys_it_by_model_slug(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    variant = RerankVariantRecord(
        "qwen3-reranker:0.6b-fp16", "v" * 64, "rerank/qwen3_reranker_0_6b_fp16"
    )

    workspace.register_rerank_variant("qwen3_reranker_0_6b_fp16", variant)

    assert workspace.variant_records() == {"qwen3_reranker_0_6b_fp16": variant}


def test_older_run_schemas_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 2, "identity": {}}), encoding="utf-8")

    with pytest.raises(RunConflictError, match="Unsupported run schema 2"):
        load_run_record(path)


def test_evaluation_paths_inside_data_are_stored_relative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    gold = data / "evaluation" / "gold" / "section_retrieval_eval.jsonl"

    reference = evaluation_reference(gold, data)

    assert reference == "evaluation/gold/section_retrieval_eval.jsonl"
    assert resolve_evaluation_reference(reference, data) == gold
    outside = tmp_path / "elsewhere.jsonl"
    assert (
        resolve_evaluation_reference(evaluation_reference(outside, data), data)
        == outside.resolve()
    )


def test_unregister_rerank_variant_keeps_the_other_variants(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    kept = RerankVariantRecord("model-a", "a" * 64, "rerank/model_a")
    removed = RerankVariantRecord("model-b", "b" * 64, "rerank/model_b")
    workspace.register_rerank_variant("model_a", kept)
    workspace.register_rerank_variant("model_b", removed)

    assert workspace.unregister_rerank_variant("model_b") == removed
    assert workspace.variant_records() == {"model_a": kept}
    with pytest.raises(RunConflictError, match="model_b"):
        workspace.unregister_rerank_variant("model_b")


def test_run_json_written_before_sampling_loads_as_unsampled(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    payload = json.loads(workspace.record_path.read_text(encoding="utf-8"))
    del payload["identity"]["sample"], payload["identity"]["sample_seed"]
    workspace.record_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run_record(workspace.record_path).identity

    assert loaded == identity()
    assert (loaded.sample, loaded.sample_seed) == (None, None)


def test_a_different_sample_seed_is_an_identity_conflict(tmp_path: Path) -> None:
    sampled = replace(identity(), sample=1000, sample_seed=0)
    RunWorkspace.open_or_create(tmp_path / "run", sampled)

    with pytest.raises(RunConflictError, match="sample_seed"):
        RunWorkspace.open_or_create(tmp_path / "run", replace(sampled, sample_seed=1))
