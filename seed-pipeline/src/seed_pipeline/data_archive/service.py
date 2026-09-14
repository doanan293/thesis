from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from seed_pipeline.data_archive.manifest import (
    ARCHIVE_MANIFEST_NAME,
    read_archive_manifest,
    write_archive_manifest,
)
from seed_pipeline.data_archive.pack import (
    PART_SIZE_BYTES,
    git_head,
    list_archive_files,
    pack_archive,
)
from seed_pipeline.data_archive.unpack import part_matches, unpack_archive
from seed_pipeline.integrations.kaggle.dataset_service import DatasetService
from seed_pipeline.integrations.kaggle.job_lock import kaggle_job_lock

DATASET_SLUG = "seed-pipeline-data"
DATASET_TITLE = "Seed pipeline data"


@dataclass(frozen=True)
class PushResult:
    reference: str
    version: int | None
    file_count: int
    part_count: int


@dataclass(frozen=True)
class PullResult:
    reference: str
    file_count: int
    written: int
    unchanged: int
    downloaded_parts: int


def archive_dir(project_root: Path) -> Path:
    return Path(project_root) / "data" / "work" / "archive"


def push_data(
    *,
    project_root: Path,
    dataset: DatasetService,
    message: str,
    now: datetime,
    part_size: int = PART_SIZE_BYTES,
) -> PushResult:
    staging = archive_dir(project_root)
    with kaggle_job_lock(staging):
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        manifest = pack_archive(
            Path(project_root) / "data",
            list_archive_files(project_root),
            staging,
            part_size=part_size,
            created_at=now.isoformat(),
            git_commit=git_head(project_root),
        )
        write_archive_manifest(staging / ARCHIVE_MANIFEST_NAME, manifest)
        prepared = dataset.ensure_dataset(
            DATASET_SLUG, DATASET_TITLE, staging, message=message
        )
    return PushResult(
        prepared.reference,
        prepared.expected_version,
        len(manifest.files),
        len(manifest.parts),
    )


def pull_data(
    *, project_root: Path, dataset: DatasetService, force: bool
) -> PullResult:
    staging = archive_dir(project_root)
    reference = f"{dataset.owner}/{DATASET_SLUG}"
    with kaggle_job_lock(staging):
        dataset.require_ready(reference, "`seed data push`")
        manifest = read_archive_manifest(
            dataset.download_file(reference, ARCHIVE_MANIFEST_NAME, staging)
        )
        downloaded = 0
        for part in manifest.parts:
            if not part_matches(staging / part.name, part):
                dataset.download_file(reference, part.name, staging)
                downloaded += 1
        result = unpack_archive(
            staging, manifest, Path(project_root) / "data", force=force
        )
    return PullResult(
        reference, len(manifest.files), result.written, result.unchanged, downloaded
    )
