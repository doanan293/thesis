import json
from pathlib import Path

from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.integrations.kaggle.artifacts import load_cloud_artifact
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName
from corpus_pipeline.integrations.kaggle.workers.query_embed import (
    run_query_embed_worker,
)


def test_query_worker_embeds_only_missing_records(tmp_path: Path):
    input_path = tmp_path / "eval.jsonl"
    input_path.write_text(
        '{"query_id":"q1","query":"one"}\n{"query_id":"q2","query":"two"}\n',
        encoding="utf-8",
    )
    identity = JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={},
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "query_embeddings.jsonl").write_text(
        json.dumps(
            {
                "model": "fake",
                "query_id": "q1",
                "query_hash": query_hash("one"),
                "query": "one",
                "embedding": [1.0, 2.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observed = []

    def embed_batch(texts, _model):
        observed.extend(texts)
        return [[float(len(texts)), 3.0] for _ in texts]

    result = run_query_embed_worker(
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "model": "fake",
            "job_sha256": identity.sha256,
            "identity": identity.payload,
        },
        embed_batch=embed_batch,
    )
    artifact = load_cloud_artifact(result.data_path, result.manifest_path, identity)
    assert artifact.completion.is_complete
    assert observed == ["two"]
    assert len(result.data_path.read_text().splitlines()) == 2


def test_query_worker_replaces_records_without_current_contract(tmp_path: Path):
    input_path = tmp_path / "eval.jsonl"
    input_path.write_text('{"query_id":"q1","query":"one"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "query_embeddings.jsonl").write_text(
        '{"query_id":"q1","query":"one","embedding":[1.0]}\n',
        encoding="utf-8",
    )
    observed = []
    identity = JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={},
    )
    run_query_embed_worker(
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "model": "fake",
            "job_sha256": identity.sha256,
            "identity": identity.payload,
        },
        embed_batch=lambda texts, _model: observed.extend(texts) or [[2.0]],
    )
    assert observed == ["one"]
