import json
from types import SimpleNamespace

from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunIdentity,
    RunWorkspace,
    load_run_record,
)


def identity() -> RunIdentity:
    return RunIdentity(
        evaluation_path="evaluation.jsonl",
        evaluation_sha256="evaluation",
        collection_name="collection",
        embedding_model="embedding",
        query_embeddings_sha256="queries",
        retriever="hybrid",
        candidate_k=30,
        rrf_k=2,
        limit=None,
        prefetch_k=50,
    )


def test_load_v1_preserves_legacy_reference_without_writing(tmp_path):
    path = tmp_path / "run.json"
    payload = {
        "schema_version": 1,
        "identity": identity().__dict__,
        "status": "reranked",
        "candidates_dir": "candidates",
        "rerank_scores_dir": "/cache/old.jsonl",
        "reranker": "old-model",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    record = load_run_record(path)

    assert record.legacy_rerank is not None
    assert record.legacy_rerank.model == "old-model"
    assert record.rerank_variants == {}
    assert path.read_bytes() == before


def test_register_variant_preserves_existing_variants_and_uses_relative_path(tmp_path):
    workspace = RunWorkspace.open_or_create(tmp_path, identity())
    first = RerankVariantRecord("model-a", "rerank/model-a/aaaa", "complete")
    second = RerankVariantRecord("model-b", "rerank/model-b/bbbb", "complete")

    workspace.register_rerank_variant("a" * 64, first)
    workspace.register_rerank_variant("b" * 64, second)

    record = load_run_record(tmp_path / "run.json")
    assert record.schema_version == 2
    assert set(record.rerank_variants) == {"a" * 64, "b" * 64}
    assert record.rerank_variants["a" * 64].artifact_dir == "rerank/model-a/aaaa"


def test_record_candidates_stores_relative_path_when_inside_run(tmp_path):
    workspace = RunWorkspace.open_or_create(tmp_path, identity())
    candidate_path = tmp_path / "candidates" / "candidates.jsonl"
    candidate_path.parent.mkdir()
    candidate_path.write_text("{}\n", encoding="utf-8")
    manifest_path = candidate_path.with_name("manifest.json")
    manifest_path.write_text("{}\n", encoding="utf-8")

    workspace.record_candidates(
        SimpleNamespace(data_path=candidate_path, manifest_path=manifest_path)
    )

    assert load_run_record(tmp_path / "run.json").candidates_dir == "candidates"


def test_split_workspace_keeps_registry_and_manifest_outside_heavy_root(tmp_path):
    metadata = tmp_path / "retrieval_eval" / "run-a"
    heavy = tmp_path / "heavy" / "retrieval_eval" / "run-a"
    workspace = RunWorkspace.open_or_create(metadata, identity(), artifact_root=heavy)
    artifact = SimpleNamespace(
        data_path=heavy / "candidates" / "candidates.jsonl",
        manifest_path=heavy / "candidates" / "manifest.json",
    )
    artifact.data_path.parent.mkdir(parents=True)
    artifact.data_path.write_text("{}\n", encoding="utf-8")
    artifact.manifest_path.write_text('{"complete":1}\n', encoding="utf-8")

    workspace.record_candidates(artifact)

    assert (metadata / "run.json").is_file()
    assert (
        metadata / "candidates-manifest.json"
    ).read_bytes() == artifact.manifest_path.read_bytes()
    assert workspace.candidates_dir == heavy / "candidates"
    assert load_run_record(metadata / "run.json").candidates_dir == "candidates"


def test_reopen_split_workspace_preserves_heavy_artifact_root(tmp_path):
    metadata = tmp_path / "retrieval_eval" / "run-a"
    heavy = tmp_path / "heavy" / "retrieval_eval" / "run-a"
    RunWorkspace.open_or_create(metadata, identity(), artifact_root=heavy)

    reopened = RunWorkspace.open_or_create(metadata, identity(), artifact_root=heavy)

    assert reopened.artifact_base == heavy
    assert reopened.candidates_dir == heavy / "candidates"
