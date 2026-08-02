import json
from pathlib import Path
from typing import Any, cast

import pytest

from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration
from corpus_pipeline.integrations.kaggle.models import StageName, StageRequest
from corpus_pipeline.integrations.kaggle.stages import get_stage_adapter

OWNERS = OwnerConfiguration("run", "runtime", "corpus", "checkpoint")


def request(
    tmp_path: Path, stage: StageName, model: str, input_path: Path
) -> StageRequest:
    return StageRequest(
        stage=stage,
        model=model,
        input_path=input_path,
        output_dir=tmp_path / "runs",
        gguf_root=tmp_path / "gguf",
        owners=OWNERS,
    )


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("stage", "model", "filename"),
    [
        (
            StageName.CORPUS_EMBED,
            "qwen3-embedding:0.6b-fp16",
            "vector_embeddings.jsonl",
        ),
        (StageName.QUERY_EMBED, "qwen3-embedding:0.6b-fp16", "query_embeddings.jsonl"),
        (StageName.RERANK, "qwen3-reranker:0.6b-fp16", "rerank_scores.jsonl"),
    ],
)
def test_stage_builds_canonical_job(
    tmp_path: Path, stage: StageName, model: str, filename: str
):
    input_path = tmp_path / f"{stage.value}.jsonl"
    if stage is StageName.RERANK:
        write_jsonl(
            input_path,
            [
                {
                    "query_id": "q1",
                    "candidates": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                }
            ],
        )
        input_path.with_name("manifest.json").write_text(
            json.dumps({"pair_count": 2}), encoding="utf-8"
        )
    else:
        write_jsonl(
            input_path,
            [
                {"query_id": "q1", "query": "hello"},
                {"query_id": "q2", "query": "world"},
            ],
        )
    job = get_stage_adapter(stage).build_job(
        request(tmp_path, stage, model, input_path)
    )
    assert job.data_filename == filename
    assert job.identity.payload["stage"] == stage.value
    assert job.expected_total == 2
    assert not job.output_dir.exists()
    if stage in {StageName.CORPUS_EMBED, StageName.QUERY_EMBED}:
        runtime_parameters = cast(
            dict[str, Any], job.identity.payload["runtime_parameters"]
        )
        assert job.worker_config["batch_size"] == runtime_parameters["batch_size"]


def test_query_embed_rejects_reranker(tmp_path: Path):
    input_path = tmp_path / "eval.jsonl"
    write_jsonl(input_path, [{"query_id": "q1", "query": "hello"}])
    with pytest.raises(ValueError, match="embedding model"):
        get_stage_adapter(StageName.QUERY_EMBED).build_job(
            request(
                tmp_path, StageName.QUERY_EMBED, "qwen3-reranker:0.6b-fp16", input_path
            )
        )


def test_rerank_requires_candidate_manifest(tmp_path: Path):
    input_path = tmp_path / "candidates.jsonl"
    write_jsonl(input_path, [{"query_id": "q1", "candidates": []}])
    with pytest.raises(ValueError, match="manifest"):
        get_stage_adapter(StageName.RERANK).build_job(
            request(tmp_path, StageName.RERANK, "qwen3-reranker:0.6b-fp16", input_path)
        )
