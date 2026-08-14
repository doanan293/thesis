import json
from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle.dependencies import input_dataset_slug
from corpus_pipeline.integrations.kaggle.kernels import PipelineKernelService
from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.integrations.kaggle.stages import RerankStage


def test_prepare_bundle_serializes_mounted_input_descriptors(tmp_path):
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
        SimpleNamespace(
            stage=StageName.RERANK,
            model="qwen3-reranker:0.6b-fp16",
            input_path=candidates,
            output_dir=tmp_path / "output",
            gguf_root=tmp_path / "gguf",
            owners=SimpleNamespace(
                execution="owner",
                runtime="owner",
                corpus="owner",
                checkpoint="owner",
            ),
        )
    )

    bundle = PipelineKernelService(
        service=None, owner="owner", source_root=Path("src")
    ).prepare_bundle(
        job,
        root=tmp_path / "bundle",
        dataset_references=["owner/runtime"],
        checkpoint_reference=None,
        total_budget_seconds=60,
    )
    config = json.loads((bundle / "stage_config.json").read_text(encoding="utf-8"))
    mounted_root = Path("/kaggle/input") / input_dataset_slug(job)

    assert config["input_files"] == {
        key: {
            "path": str(mounted_root / descriptor["filename"]),
            "filename": descriptor["filename"],
            "sha256": descriptor["sha256"],
        }
        for key, descriptor in job.input_bundle.descriptors().items()
    }
    assert "candidate_path" not in config
    assert "candidate_manifest_path" not in config
