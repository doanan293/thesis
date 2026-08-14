import json
from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.integrations.kaggle.stages import (
    CorpusEmbedStage,
    QueryEmbedStage,
    RerankStage,
)


def _owners():
    return SimpleNamespace(
        execution="owner", runtime="owner", corpus="owner", checkpoint="owner"
    )


def _request(stage: StageName, model: str, input_path: Path):
    return SimpleNamespace(
        stage=stage,
        model=model,
        input_path=input_path,
        output_dir=input_path.parent / "output",
        gguf_root=input_path.parent / "gguf",
        owners=_owners(),
    )


def test_corpus_embed_builds_version_2_single_file_bundle(tmp_path):
    source = tmp_path / "chunks.jsonl"
    source.write_text(json.dumps({"chunk_id": "c1"}) + "\n", encoding="utf-8")

    job = CorpusEmbedStage().build_job(
        _request(StageName.CORPUS_EMBED, "qwen3-embedding:4b-fp16", source)
    )

    assert job.contract_version == 2
    assert job.identity.payload["input_sha256"] == job.input_bundle.sha256
    assert set(job.input_bundle.descriptors()) == {"input"}
    assert "input_path" not in job.worker_config


def test_query_embed_builds_version_2_single_file_bundle(tmp_path):
    source = tmp_path / "evaluation.jsonl"
    source.write_text(json.dumps({"query_id": "q1"}) + "\n", encoding="utf-8")

    job = QueryEmbedStage().build_job(
        _request(StageName.QUERY_EMBED, "qwen3-embedding:4b-fp16", source)
    )

    assert job.contract_version == 2
    assert job.identity.payload["input_sha256"] == job.input_bundle.sha256
    assert set(job.input_bundle.descriptors()) == {"input"}
    assert "input_path" not in job.worker_config


def test_rerank_builds_version_2_two_file_bundle(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [{"chunk_id": "c1", "document_text": "text"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1}) + "\n", encoding="utf-8"
    )

    job = RerankStage().build_job(
        _request(StageName.RERANK, "qwen3-reranker:0.6b-fp16", candidates)
    )

    assert job.contract_version == 2
    assert set(job.input_bundle.descriptors()) == {
        "candidates",
        "candidate_manifest",
    }
    assert job.identity.payload["input_sha256"] == job.input_bundle.sha256
    assert job.input_bundle.sha256 != job.input_bundle.file("candidates").sha256
    assert "candidate_path" not in job.worker_config
    assert "candidate_manifest_path" not in job.worker_config
