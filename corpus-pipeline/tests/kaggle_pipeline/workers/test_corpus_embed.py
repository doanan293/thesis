import json
from pathlib import Path

from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName
from corpus_pipeline.integrations.kaggle.workers.corpus_embed import (
    run_corpus_embed_worker,
)


def test_corpus_worker_exports_common_manifest(tmp_path: Path):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    output = tmp_path / "output"
    identity = JobIdentity.create(
        stage=StageName.CORPUS_EMBED,
        contract_version=1,
        model="fake",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={},
    )

    def executor(config):
        Path(config["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(config["output_path"]).write_text(
            json.dumps({"chunk_id": "c1", "embedding": [0.1]}) + "\n", encoding="utf-8"
        )
        return 0

    result = run_corpus_embed_worker(
        {
            "input_path": str(corpus),
            "output_dir": str(output),
            "output_path": str(output / "vector_embeddings.jsonl"),
            "model": "fake",
            "job_sha256": identity.sha256,
            "identity": identity.payload,
        },
        command_executor=executor,
    )
    assert result.completion.is_complete
    assert result.manifest_path.is_file()
