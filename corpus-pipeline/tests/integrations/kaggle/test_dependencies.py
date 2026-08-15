import json
from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle import dependencies
from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.integrations.kaggle.stages import RerankStage
from corpus_pipeline.runtime.catalog import require_model


def _job(tmp_path: Path):
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
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    request = SimpleNamespace(
        stage=StageName.RERANK,
        model="qwen3-reranker:0.6b-fp16",
        input_path=candidates,
        output_dir=tmp_path / "output",
        gguf_root=tmp_path / "gguf",
        owners=SimpleNamespace(
            execution="owner", runtime="owner", corpus="owner", checkpoint="owner"
        ),
        runtime_profile=require_model(
            "qwen3-reranker:0.6b-fp16"
        ).rerank_search_space.candidates[0],
    )
    return RerankStage().build_job(request), candidates, manifest


def test_input_dataset_materializes_every_bundle_file_and_manifest(
    tmp_path, monkeypatch
):
    job, candidates, manifest = _job(tmp_path)
    monkeypatch.setattr(
        dependencies,
        "resolve_publishable_artifact",
        lambda *_args: SimpleNamespace(
            dataset_slug="model",
            sha256="model-sha",
        ),
    )
    desired = dependencies.default_desired_datasets(
        job,
        SimpleNamespace(
            execution="owner", runtime="owner", corpus="owner", checkpoint="owner"
        ),
        tmp_path,
    )
    input_dataset = next(item for item in desired if item.resource_kind == "input")
    staged = input_dataset.materialize(tmp_path / "staged")
    payload = json.loads(
        (staged / "dependency_manifest.json").read_text(encoding="utf-8")
    )

    assert input_dataset.fingerprint == job.input_bundle.sha256
    assert input_dataset.manifest_filename == "dependency_manifest.json"
    assert payload == {
        "schema_version": 2,
        "resource_kind": "input",
        "fingerprint": job.input_bundle.sha256,
        "files": job.input_bundle.descriptors(),
    }
    assert (staged / "candidates.jsonl").read_bytes() == candidates.read_bytes()
    assert (staged / "manifest.json").read_bytes() == manifest.read_bytes()
