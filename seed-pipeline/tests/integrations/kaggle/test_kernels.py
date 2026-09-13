import json
from pathlib import Path

from tests.integrations.kaggle.factories import (
    UnusedKernelService,
    rerank_runtime_profile,
    stage_request,
)

from seed_pipeline.integrations.kaggle.api import kernel_output_command
from seed_pipeline.integrations.kaggle.dependencies import input_dataset_slug
from seed_pipeline.integrations.kaggle.kernels import PipelineKernelService
from seed_pipeline.integrations.kaggle.models import (
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    StageName,
)
from seed_pipeline.integrations.kaggle.stages import RerankStage


def test_kernel_output_downloads_diagnostic_sidecars(tmp_path):
    command = kernel_output_command("owner/kernel", tmp_path)

    assert command[-1] == r".*(jsonl|json|zip|md|log)$"


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
        stage_request(
            StageName.RERANK,
            "qwen3-reranker:0.6b-fp16",
            candidates,
            runtime_profile=rerank_runtime_profile(),
        )
    )

    bundle = PipelineKernelService(
        service=UnusedKernelService(), owner="owner", source_root=Path("src")
    ).prepare_bundle(
        job,
        root=tmp_path / "bundle",
        dataset_references=["owner/runtime"],
        checkpoint_reference="secondary-user/checkpoint-slug",
        total_budget_seconds=60,
    )
    config = json.loads((bundle / "stage_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
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
    assert "secondary-user/checkpoint-slug" in metadata["dataset_sources"]
    assert "owner/checkpoint-slug" not in metadata["dataset_sources"]


def test_kernel_references_prefer_sixteen_characters_and_keep_legacy(tmp_path):
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
    job = RerankStage().build_job(
        stage_request(
            StageName.RERANK,
            "qwen3-reranker:0.6b-fp16",
            candidates,
            runtime_profile=rerank_runtime_profile(),
        )
    )

    service = PipelineKernelService(
        service=UnusedKernelService(), owner="owner", source_root=Path("src")
    )

    assert service.references(job) == (
        f"owner/rerank-{job.identity.sha256[:16]}",
        f"owner/rerank-{job.identity.sha256[:8]}",
    )
    assert service.reference(job) == service.references(job)[0]


def test_discover_falls_back_when_missing_preferred_slug_is_denied(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {"query_id": "q1", "query": "query", "candidates": [{"chunk_id": "c1"}]}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    job = RerankStage().build_job(
        stage_request(
            StageName.RERANK,
            "qwen3-reranker:0.6b-fp16",
            candidates,
            runtime_profile=rerank_runtime_profile(),
        )
    )

    class FakeKernelService(UnusedKernelService):
        def __init__(self):
            self.checked = []

        def inspect_state(self, reference: str) -> KernelRemoteState:
            self.checked.append(reference)
            if reference.endswith(job.identity.sha256[:16]):
                return KernelRemoteState(
                    reference,
                    KernelPresence.UNKNOWN,
                    detail="Permission 'kernels.get' was denied",
                )
            return KernelRemoteState(
                reference, KernelPresence.EXISTS, KernelStatus.RUNNING
            )

        def confirm_missing(self, reference: str) -> bool | None:
            assert reference.endswith(job.identity.sha256[:16])
            return True

    fake = FakeKernelService()
    service = PipelineKernelService(fake, owner="owner", source_root=Path("src"))

    state = service.discover(job)

    assert state.presence is KernelPresence.EXISTS
    assert state.reference.endswith(job.identity.sha256[:8])
    assert fake.checked == list(service.references(job))
