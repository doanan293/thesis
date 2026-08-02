import json
from pathlib import Path

from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName
from corpus_pipeline.integrations.kaggle.workers.rerank import run_rerank_worker


def test_rerank_worker_resumes_missing_pairs(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "hello",
                "candidates": [
                    {"chunk_id": "c1", "document_text": "one"},
                    {"chunk_id": "c2", "document_text": "two"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "rerank_scores.jsonl").write_text(
        json.dumps({"query_id": "q1", "chunk_id": "c1", "score": 0.5}) + "\n",
        encoding="utf-8",
    )
    identity = JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={},
    )
    result = run_rerank_worker(
        {
            "candidate_path": str(candidates),
            "output_dir": str(output),
            "model": "fake",
            "job_sha256": identity.sha256,
            "identity": identity.payload,
        },
        score_pair=lambda query, text, model: 0.9,
    )
    assert result.completion.is_complete
    records = [json.loads(line) for line in result.data_path.read_text().splitlines()]
    assert {record["chunk_id"] for record in records} == {"c1", "c2"}
