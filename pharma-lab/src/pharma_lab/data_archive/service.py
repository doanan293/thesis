from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pharma_lab.data_archive.manifest import (
    ARCHIVE_MANIFEST_NAME,
    ArchiveError,
    read_archive_manifest,
    write_archive_manifest,
)
from pharma_lab.data_archive.pack import (
    PART_SIZE_BYTES,
    git_head,
    list_archive_files,
    pack_archive,
)
from pharma_lab.data_archive.unpack import part_matches, unpack_archive
from pharma_lab.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetService,
)
from pharma_lab.integrations.kaggle.job_lock import kaggle_job_lock

# Kept from the project's former name: renaming the Kaggle dataset would mean re-uploading it.
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


def _archived_files_missing_locally(
    dataset: DatasetService, staging: Path, local_files: list[str]
) -> list[str]:
    reference = f"{dataset.owner}/{DATASET_SLUG}"
    state = dataset.inspect_state(reference, active_owner=dataset.owner)
    if state.presence is not DatasetPresence.EXISTS:
        return []
    remote = read_archive_manifest(
        dataset.download_file(reference, ARCHIVE_MANIFEST_NAME, staging)
    )
    local = set(local_files)
    return sorted(item.path for item in remote.files if item.path not in local)


def push_data(
    *,
    project_root: Path,
    dataset: DatasetService,
    message: str,
    now: datetime,
    part_size: int = PART_SIZE_BYTES,
    allow_removal: bool = False,
) -> PushResult:
    staging = archive_dir(project_root)
    with kaggle_job_lock(staging):
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        local_files = list_archive_files(project_root)
        if not allow_removal:
            # A push replaces the whole archive, so a trimmed data/ would silently
            # drop results that exist only on Kaggle.
            missing = _archived_files_missing_locally(dataset, staging, local_files)
            if missing:
                listed = "\n".join(f"  {path}" for path in missing)
                raise ArchiveError(
                    f"{len(missing)} archived files are missing from data/; run "
                    "`pharma-lab data pull` first, or pass --allow-removal to drop "
                    f"them from the archive:\n{listed}"
                )
            shutil.rmtree(staging)
            staging.mkdir(parents=True)
        manifest = pack_archive(
            Path(project_root) / "data",
            local_files,
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
        dataset.require_ready(reference, "`pharma-lab data push`")
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
