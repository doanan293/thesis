from __future__ import annotations

import base64
import io
import json
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

from corpus_pipeline.integrations.kaggle.api import kernel_metadata
from corpus_pipeline.integrations.kaggle.dependencies import kaggle_input_root
from corpus_pipeline.integrations.kaggle.errors import KaggleRemoteStateError
from corpus_pipeline.integrations.kaggle.kernel_service import KernelService
from corpus_pipeline.integrations.kaggle.models import (
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    StageJob,
)


class PipelineKernelService:
    def __init__(self, service: KernelService, *, owner: str, source_root: Path):
        self.service = service
        self.owner = owner
        self.source_root = Path(source_root)

    def prepare_bundle(
        self,
        job: StageJob,
        *,
        root: Path,
        dataset_references: Sequence[str],
        checkpoint_reference: str | None,
        total_budget_seconds: int,
    ) -> Path:
        bundle = Path(root) / f"{job.stage.value}-{job.identity.sha256[:16]}"
        shutil.rmtree(bundle, ignore_errors=True)
        bundle.mkdir(parents=True)
        references = list(dataset_references)
        if checkpoint_reference:
            references.append(checkpoint_reference)
        metadata = kernel_metadata(
            self.owner,
            bundle.name,
            bundle.name,
            references,
            code_file="runner.py",
            enable_internet=False,
        )
        (bundle / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        config = dict(job.worker_config) | {
            "output_dir": "/kaggle/working/artifact",
            "total_budget_seconds": total_budget_seconds,
            "identity": job.identity.payload,
            "job_sha256": job.identity.sha256,
        }
        if checkpoint_reference:
            checkpoint_filename = {
                "corpus-embed": "vector_embeddings.jsonl",
                "query-embed": "query_embeddings.journal.jsonl",
                "rerank": "rerank_scores.journal.jsonl",
            }.get(job.stage.value)
            if checkpoint_filename is not None:
                config["checkpoint_filename"] = checkpoint_filename
        mounted_root = kaggle_input_root(job)
        config["input_files"] = {
            key: {
                "path": str(mounted_root / descriptor["filename"]),
                "filename": descriptor["filename"],
                "sha256": descriptor["sha256"],
            }
            for key, descriptor in job.input_bundle.descriptors().items()
        }
        (bundle / "stage_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        source_b64 = self._build_source_b64()
        (bundle / "source_bundle.zip").write_bytes(base64.b64decode(source_b64))
        runner = f"""import base64, json, os, runpy, sys\nfrom pathlib import Path\nSOURCE_B64 = {source_b64!r}\nsource = Path('/tmp/source_bundle.zip')\nsource.write_bytes(base64.b64decode(SOURCE_B64))\nsys.path.insert(0, str(source))\nconfig = Path('/tmp/stage_config.json')\nconfig.write_text({json.dumps(json.dumps(config))}, encoding='utf-8')\nos.environ['KAGGLE_PIPELINE_CONFIG'] = str(config)\nrunpy.run_module({job.worker_module!r}, run_name='__main__')\n"""
        (bundle / "runner.py").write_text(runner, encoding="utf-8")
        return bundle

    def _build_source_b64(self) -> str:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.source_root.rglob("*.py")):
                archive.write(path, arcname=str(path.relative_to(self.source_root)))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def references(self, job: StageJob) -> tuple[str, ...]:
        base = f"{self.owner}/{job.stage.value}-"
        preferred = base + job.identity.sha256[:16]
        legacy = base + job.identity.sha256[:8]
        return (preferred, legacy) if preferred != legacy else (preferred,)

    def reference(self, job: StageJob) -> str:
        return self.references(job)[0]

    def discover(self, job: StageJob) -> KernelRemoteState:
        for reference in self.references(job):
            state = self.service.inspect_state(reference)
            if state.presence is KernelPresence.UNKNOWN:
                if (
                    "kernels.get" in state.detail.casefold()
                    and self.service.confirm_missing(reference)
                ):
                    continue
                raise KaggleRemoteStateError(
                    f"Cannot inspect Kaggle kernel {reference}: {state.detail}"
                )
            if state.presence is KernelPresence.EXISTS:
                return state
        return KernelRemoteState(self.reference(job), KernelPresence.ABSENT)

    def push(self, bundle: Path, *, timeout_seconds: int) -> None:
        self.service.push(bundle, timeout_seconds=timeout_seconds)

    def wait_for_terminal(
        self, reference: str, *, timeout_seconds: int
    ) -> KernelStatus:
        return self.service.wait_for_terminal(
            reference, timeout_seconds=timeout_seconds
        )

    def download_output(self, reference: str, destination: Path) -> None:
        self.service.download_output(reference, destination)

    def run(self, job: StageJob, bundle: Path, *, timeout_seconds: int) -> Path:
        self.service.push(bundle, timeout_seconds=timeout_seconds)
        reference = self.reference(job)
        self.service.poll(reference, timeout_seconds=timeout_seconds)
        destination = Path(bundle) / "downloaded-output"
        self.service.download_output(reference, destination)
        return destination
