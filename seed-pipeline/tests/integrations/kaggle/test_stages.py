import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import stage_request

from seed_pipeline.integrations.kaggle.models import StageName, StageRequest
from seed_pipeline.integrations.kaggle.stages import (
    CorpusEmbedStage,
    QueryEmbedStage,
    RerankStage,
    get_stage_adapter,
)
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate


def _request(stage: StageName, model: str, input_path: Path) -> StageRequest:
    return stage_request(
        stage,
        model,
        input_path,
        runtime_profile=RuntimeCandidate(
            server_slots=4,
            concurrency=2,
            request_batch_size=1,
            context_per_slot=4096,
            logical_batch_size=4096,
            physical_batch_size=2048,
        ),
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


def test_rerank_builds_version_3_two_file_bundle(tmp_path):
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

    assert job.contract_version == 3
    runtime_parameters = job.identity.payload["runtime_parameters"]
    assert isinstance(runtime_parameters, Mapping)
    assert "request_contract_sha256" in runtime_parameters
    assert "prompt_contract_sha256" not in runtime_parameters


def test_production_stage_requires_runtime_profile(tmp_path):
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
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    request = replace(
        _request(StageName.RERANK, "qwen3-reranker:0.6b-fp16", candidates),
        runtime_profile=None,
    )

    with pytest.raises(ValueError, match="runtime profile is required"):
        RerankStage().build_job(request)


def test_rerank_benchmark_isolated_from_production_artifact(tmp_path):
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
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")

    job = get_stage_adapter(StageName.RERANK_BENCHMARK).build_job(
        _request(StageName.RERANK_BENCHMARK, "qwen3-reranker:0.6b-fp16", candidates)
    )

    assert job.data_filename == "benchmark_results.jsonl"
    assert job.worker_module.endswith("workers.benchmark")
    assert job.local_cache_path.name == "benchmark_results.jsonl"
    assert "benchmark_levels" in job.worker_config
    assert set(job.input_bundle.descriptors()) == {
        "candidates",
        "candidate_manifest",
    }
    assert job.identity.payload["input_sha256"] == job.input_bundle.sha256
    assert job.input_bundle.sha256 != job.input_bundle.file("candidates").sha256
    assert "candidate_path" not in job.worker_config
    assert "candidate_manifest_path" not in job.worker_config
