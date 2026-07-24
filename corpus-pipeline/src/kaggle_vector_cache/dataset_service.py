from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from kaggle_vector_cache.canonical_gguf import (
    CanonicalModelArtifact,
    stage_model_dataset,
)
from kaggle_vector_cache.kaggle_api import (
    KaggleCommandRunner,
    dataset_create_command,
    dataset_file_download_command,
    dataset_metadata,
    dataset_status_command,
    dataset_version_command,
)
from kaggle_vector_cache.manifests import (
    ManifestError,
    build_corpus_manifest,
    load_json,
    write_json,
)
from kaggle_vector_cache.parsers import parse_dataset_status

CORPUS_DATASET_SLUG = "corpus-pipeline-rag-final"


class DatasetService:
    def __init__(self, runner: KaggleCommandRunner, owner: str):
        self.runner = runner
        self.owner = owner

    def _write_dataset_metadata(
        self,
        path: Path,
        slug: str,
        title: str,
        public: bool = False,
    ) -> None:
        write_json(
            Path(path) / "dataset-metadata.json",
            dataset_metadata(
                owner=self.owner,
                slug=slug,
                title=title,
                public=public,
            ),
        )

    def status(self, reference: str) -> str:
        output = self.runner.run(
            dataset_status_command(reference),
            capture_output=True,
        )
        return parse_dataset_status(output)

    def optional_status(self, reference: str) -> str | None:
        try:
            output = self.runner.run(
                dataset_status_command(reference),
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = str(exc.stdout or exc.output or "")
            if "not found" in detail.casefold() or "404" in detail:
                return None
            raise RuntimeError(
                f"Kaggle dataset status failed for {reference}: {detail}"
            ) from exc
        if not output.strip():
            return None
        return parse_dataset_status(output)

    def require_ready(self, reference: str, publish_command: str) -> None:
        try:
            status = self.status(reference)
        except Exception as exc:
            raise RuntimeError(
                f"Required Kaggle dataset is unavailable: {reference}; "
                f"publish it with: {publish_command}"
            ) from exc
        if status != "READY":
            raise RuntimeError(
                f"Required Kaggle dataset is not READY: {reference} ({status}); "
                f"publish it with: {publish_command}"
            )

    def fetch_json(
        self,
        reference: str,
        filename: str,
        destination: Path,
    ) -> dict:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.runner.run(dataset_file_download_command(reference, filename, destination))
        matches = [path for path in destination.rglob(filename) if path.is_file()]
        if len(matches) != 1:
            raise ManifestError(
                f"Expected one {filename} from {reference}, found {len(matches)}"
            )
        return load_json(matches[0])

    def stage_corpus_dataset(
        self,
        corpus_path: Path,
        staging_root: Path,
    ) -> Path:
        staging_root = Path(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        target = staging_root / CORPUS_DATASET_SLUG
        target.mkdir(exist_ok=False)
        staged_corpus = target / "chunks.jsonl"
        shutil.copy2(corpus_path, staged_corpus)
        with staged_corpus.open(encoding="utf-8") as handle:
            row_count = sum(1 for line in handle if line.strip())
        write_json(
            target / "manifest.json",
            build_corpus_manifest(staged_corpus, row_count),
        )
        self._write_dataset_metadata(
            target,
            CORPUS_DATASET_SLUG,
            "Corpus Pipeline RAG Final",
            public=True,
        )
        return target

    def publish_corpus(self, corpus_path: Path, staging_root: Path) -> str:
        staged = self.stage_corpus_dataset(corpus_path, staging_root)
        self.ensure_dataset(
            slug=CORPUS_DATASET_SLUG,
            title="Corpus Pipeline RAG Final",
            path=staged,
            public=True,
        )
        return f"{self.owner}/{CORPUS_DATASET_SLUG}"

    def ensure_dataset(
        self,
        slug: str,
        title: str,
        path: Path,
        public: bool = False,
    ) -> None:
        self._write_dataset_metadata(path, slug, title, public)
        reference = f"{self.owner}/{slug}"
        status = self.optional_status(reference)
        if status == "PENDING":
            self.wait_for_dataset_ready(reference)
            status = "READY"
        if status == "READY":
            self.runner.run(dataset_version_command(path, f"Update {slug}"))
            return
        if status is None:
            self.runner.run(dataset_create_command(path, public=public))
            return
        raise RuntimeError(
            f"Cannot publish Kaggle dataset {reference} in status {status}"
        )

    def wait_for_dataset_ready(
        self,
        reference: str,
        max_attempts: int = 300,
    ) -> None:
        for _ in range(max_attempts):
            if self.status(reference) == "READY":
                return
            time.sleep(2)
        raise TimeoutError(f"Dataset {reference} was not ready in time")

    def publish_canonical_gguf(
        self,
        artifact: CanonicalModelArtifact,
        staging_root: Path,
        public: bool = False,
    ) -> str:
        staging_dir = stage_model_dataset(
            artifact=artifact,
            staging_root=staging_root,
            owner=self.owner,
        )
        self.ensure_dataset(
            slug=artifact.dataset_slug,
            title=f"Vector Cache GGUF {artifact.model}"[:50],
            path=staging_dir,
            public=public,
        )
        return f"{self.owner}/{artifact.dataset_slug}"
