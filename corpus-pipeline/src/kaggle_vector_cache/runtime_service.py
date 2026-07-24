from __future__ import annotations

import json
from pathlib import Path

from kaggle_vector_cache.dataset_service import DatasetService
from kaggle_vector_cache.kaggle_api import (
    KaggleCommandRunner,
    dataset_metadata,
    kernel_metadata,
    kernel_output_all_command,
)
from kaggle_vector_cache.kernel_service import KernelService
from kaggle_vector_cache.llama_cpp_runtime import (
    LLAMA_CPP_BUILDER_KERNEL_SLUG,
    LLAMA_CPP_DATASET_SLUG,
    build_kernel_source,
    validate_runtime_artifact,
)
from kaggle_vector_cache.manifests import write_json
from kaggle_vector_cache.models import require_model


class RuntimeService:
    def __init__(
        self,
        runner: KaggleCommandRunner,
        datasets: DatasetService,
        kernels: KernelService,
    ):
        self.runner = runner
        self.datasets = datasets
        self.kernels = kernels

    def prepare_llama_cpp_builder_bundle(
        self,
        bundle_root: Path,
        model_dataset_slug: str | None = None,
    ) -> Path:
        if model_dataset_slug is None:
            model_dataset_slug = require_model("bge-m3:567m-fp16").gguf_dataset_slug
        bundle_dir = Path(bundle_root) / LLAMA_CPP_BUILDER_KERNEL_SLUG
        bundle_dir.mkdir(parents=True, exist_ok=False)
        meta = kernel_metadata(
            owner=self.datasets.owner,
            kernel_slug=LLAMA_CPP_BUILDER_KERNEL_SLUG,
            title="LLaMA C++ Runtime Builder",
            dataset_sources=[],
            code_file="main.py",
            enable_internet=True,
        )
        (bundle_dir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        (bundle_dir / "main.py").write_text(
            build_kernel_source(model_dataset_slug),
            encoding="utf-8",
        )
        return bundle_dir

    def build_llama_cpp_runtime(
        self,
        bundle_root: Path,
        incoming: Path,
        *,
        force: bool = False,
    ) -> str:
        reference = f"{self.datasets.owner}/{LLAMA_CPP_DATASET_SLUG}"
        if not force and self.datasets.optional_status(reference) == "READY":
            return reference
        bundle = self.prepare_llama_cpp_builder_bundle(bundle_root)
        kernel_reference = f"{self.datasets.owner}/{LLAMA_CPP_BUILDER_KERNEL_SLUG}"
        self.kernels.push_kernel(bundle, timeout_seconds=14_400)
        self.kernels.poll_kernel(kernel_reference)
        self.runner.run(kernel_output_all_command(kernel_reference, incoming))
        manifests = list(Path(incoming).rglob("runtime_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError(f"Expected one runtime manifest, found {len(manifests)}")
        runtime_root = manifests[0].parent
        validate_runtime_artifact(runtime_root)
        write_json(
            runtime_root / "dataset-metadata.json",
            dataset_metadata(
                self.datasets.owner,
                LLAMA_CPP_DATASET_SLUG,
                "Vector Cache llama.cpp CUDA T4",
                public=True,
            ),
        )
        self.datasets.ensure_dataset(
            slug=LLAMA_CPP_DATASET_SLUG,
            title="Vector Cache llama.cpp CUDA T4",
            path=runtime_root,
            public=True,
        )
        self.datasets.wait_for_dataset_ready(reference)
        return reference
