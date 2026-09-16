# Seed Data Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `seed data push` and `seed data pull`, which store every Git-ignored file under `seed-pipeline/data/` (except `data/work/`) in a private Kaggle dataset as checksummed zstd tar parts and restore them byte for byte.

**Architecture:** A new package `seed_pipeline.data_archive` has three focused modules: `manifest.py` (archive manifest schema and path safety), `pack.py` (list ignored files with Git, stream them into a tar, compress with zstd and split into parts while hashing) and `unpack.py` (verify parts, stream them back through zstd and tar, write each file atomically after checking its sha256). `service.py` connects them to the existing `DatasetService`, and `cli/commands/data.py` exposes the two commands. Tests run against a temporary Git repository and a fake Kaggle runner; nothing touches the real Kaggle account.

**Tech Stack:** Python 3.12, `tarfile`, `zstandard` 0.25 (already a dependency), Kaggle CLI 2.2.3 through `DatasetService`, Typer, pytest.

**Spec:** `seed-pipeline/docs/superpowers/specs/2026-09-14-seed-data-layout-design.md` §4.1, §6.8, §9, §12.

**Depends on:** plan `2026-09-14-seed-data-layout-code.md` (uses `paths.PROJECT_ROOT`, the readable `kaggle_job_lock` of its Task 2 and the root `tests/conftest.py` that isolates locks).

## Global Constraints

- Dataset: private `<owner>/seed-pipeline-data`, title `Seed pipeline data`; `<owner>` is `context.owners.execution` of `resolve_execution_context(--kaggle-account)` (the profile username, default profile `KAGGLE_ACCOUNT_DEFAULT`). Never pass `--public`.
- Archive contents: `git ls-files --others --ignored --exclude-standard -z -- data`, minus `data/work/`; member and manifest paths are relative to `data/`.
- Staging directory: `data/work/archive/`, containing `archive-manifest.json` and parts `seed-pipeline-data.tar.zst.part-0001`, `-0002`, …; each part at most `PART_SIZE_BYTES = 1900 * 1024 * 1024` (≤ 1.9 GiB, under Kaggle's ~2 GB per-file limit).
- Manifest schema `seed-data-archive-v1`: `created_at`, `git_commit`, `file_count`, `files` (`path`, `size`, `sha256`), `parts` (`name`, `size`, `sha256`).
- `seed data pull` stops when an existing file differs from the manifest unless `--force`; identical files are left untouched.
- No test or implementation step runs a real Kaggle command. The first real `seed data push` happens in the data migration plan, after the user confirms.
- No `noqa`, `type: ignore`, `pyrefly: ignore` or relaxed rules. Test folders have no `__init__.py`. Run `uv run ruff format src tests` before each gate.
- **Seed gate** (in `seed-pipeline/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
- **Commit** from the repository root after each task: `git add` only the task's files, check `git diff --cached --name-status`, commit with a conventional message ending with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N4DN578zgtYyAbaKdYXRX4
  ```

## File Structure

| File | Responsibility |
| --- | --- |
| `src/seed_pipeline/integrations/kaggle/dataset_service.py` | Version message; unzip a single file the Kaggle API served as `<name>.zip` |
| `src/seed_pipeline/data_archive/__init__.py` | Package marker (empty) |
| `src/seed_pipeline/data_archive/manifest.py` | `ArchiveManifest`, `ArchiveFile`, `ArchivePart`, `ArchiveError`, path safety, read/write |
| `src/seed_pipeline/data_archive/pack.py` | `list_archive_files`, `git_head`, `PartWriter`, `pack_archive` |
| `src/seed_pipeline/data_archive/unpack.py` | `part_matches`, `verify_parts`, `unpack_archive`, `UnpackResult` |
| `src/seed_pipeline/data_archive/service.py` | `push_data`, `pull_data`, `PushResult`, `PullResult` |
| `src/seed_pipeline/cli/commands/data.py`, `src/seed_pipeline/cli/app.py` | `seed data push`, `seed data pull` |
| `tests/data_archive/archive_project.py` | Temporary Git project helpers shared by the archive tests |
| `tests/data_archive/test_pack.py`, `test_unpack.py`, `test_service.py`, `tests/cli/test_data_command.py`, `tests/integrations/kaggle/test_dataset_service.py` | Tests |
| `docs/guides/cli-reference.md`, `data/README.md` | Documentation |

---

### Task 1: Dataset service version message and zipped single-file downloads

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/dataset_service.py`
- Test: `seed-pipeline/tests/integrations/kaggle/test_dataset_service.py`

**Interfaces:**
- Produces: `DatasetService.ensure_dataset(slug, title, path, *, public=False, active_owner=None, message=None) -> PreparedDataset` (a new version uses `message or title`); `DatasetService.download_file(reference, filename, destination) -> Path` also accepts a downloaded `<filename>.zip` that contains `filename`.

- [ ] **Step 1: Write the failing tests**

Add `import zipfile` and `from pathlib import Path` to the imports of `tests/integrations/kaggle/test_dataset_service.py` and append:

```python
def test_version_dataset_uses_the_given_message(tmp_path, monkeypatch):
    runner = RecordingRunner()
    service = DatasetService(runner, "owner")
    monkeypatch.setattr(
        service,
        "inspect_state",
        lambda *_args, **_kwargs: DatasetRemoteState(
            DatasetPresence.EXISTS, current_version=3
        ),
    )

    service.ensure_dataset("slug", "Title", tmp_path, message="Refresh data")

    command = runner.calls[-1][0]
    assert command[command.index("-m") + 1] == "Refresh data"


class ZippingRunner(RecordingRunner):
    """Serves each requested file the way the Kaggle API serves large files."""

    def run(self, args, capture_output=False, *, live_output=False):
        super().run(args, capture_output, live_output=live_output)
        name = args[args.index("-f") + 1]
        destination = Path(args[args.index("-p") + 1])
        with zipfile.ZipFile(destination / f"{name}.zip", "w") as archive:
            archive.writestr(name, b"payload")
        return ""


def test_download_file_unpacks_a_zipped_single_file(tmp_path):
    service = DatasetService(ZippingRunner(), "owner")

    path = service.download_file("owner/slug", "data.part-0001", tmp_path)

    assert path == tmp_path / "data.part-0001"
    assert path.read_bytes() == b"payload"
    assert not (tmp_path / "data.part-0001.zip").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle/test_dataset_service.py`
Expected: FAIL with `TypeError: DatasetService.ensure_dataset() got an unexpected keyword argument 'message'` and `KaggleRemoteStateError: Expected one data.part-0001 ... found 0`.

- [ ] **Step 3: Implement**

In `dataset_service.py` add `import zipfile` and replace `download_file` with:

<!-- fmt: off -->
```python
    def download_file(self, reference: str, filename: str, destination: Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.runner.run(dataset_file_download_command(reference, filename, destination))
        compressed = destination / f"{filename}.zip"
        if compressed.is_file():
            # The Kaggle API names a single-file download after its URL, which can be
            # a zip of the file.
            with zipfile.ZipFile(compressed) as archive:
                if filename not in archive.namelist():
                    raise KaggleRemoteStateError(
                        f"{compressed.name} from {reference} does not contain {filename}"
                    )
                archive.extract(filename, destination)
            compressed.unlink()
        matches = list(destination.rglob(filename))
        if len(matches) != 1:
            raise KaggleRemoteStateError(
                f"Expected one {filename} in downloaded dataset {reference}, found {len(matches)}"
            )
        return matches[0]
```
<!-- fmt: on -->

In `ensure_dataset` add the keyword parameter `message: str | None = None,` after `active_owner`, and change the version call to:

<!-- fmt: off -->
```python
        self.runner.run(
            dataset_version_command(Path(path), message=message or title),
            live_output=True,
        )
```
<!-- fmt: on -->

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/dataset_service.py seed-pipeline/tests/integrations/kaggle/test_dataset_service.py
git commit -m "feat(kaggle): pass a dataset version message and unzip single-file downloads"
```

---

### Task 2: Archive manifest and packing

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/data_archive/__init__.py` (empty), `seed-pipeline/src/seed_pipeline/data_archive/manifest.py`, `seed-pipeline/src/seed_pipeline/data_archive/pack.py`
- Create: `seed-pipeline/tests/data_archive/archive_project.py`, `seed-pipeline/tests/data_archive/test_pack.py`

**Interfaces:**
- Produces (`manifest`): `ARCHIVE_SCHEMA = "seed-data-archive-v1"`, `ARCHIVE_MANIFEST_NAME = "archive-manifest.json"`, `PART_PREFIX = "seed-pipeline-data.tar.zst.part-"`, `ArchiveError(RuntimeError)`, `ArchiveFile(path, size, sha256)`, `ArchivePart(name, size, sha256)`, `ArchiveManifest(created_at, git_commit, files, parts)` with `to_dict()`, `safe_data_path(value) -> str`, `read_archive_manifest(path) -> ArchiveManifest`, `write_archive_manifest(path, manifest) -> None`.
- Produces (`pack`): `PART_SIZE_BYTES`, `list_archive_files(project_root) -> list[str]`, `git_head(project_root) -> str`, `PartWriter(directory, *, part_size)` (an `io.RawIOBase`) with `write`, `flush`, `close`, `parts`, `pack_archive(data_dir, files, output_dir, *, part_size, created_at, git_commit) -> ArchiveManifest`.
- Produces (test helper `tests/data_archive/archive_project.py`): `FILES: dict[str, bytes]`, `make_project(root: Path, *, with_data: bool = True) -> Path`.

- [ ] **Step 1: Write the shared test helper**

Create `seed-pipeline/tests/data_archive/archive_project.py`:

```python
"""A throwaway Git project whose data/ mixes tracked, ignored and work files."""

import hashlib
import subprocess
from pathlib import Path


def noise(seed: str, size: int) -> bytes:
    """Deterministic bytes that zstd cannot shrink, so small parts still split."""
    blocks = (
        hashlib.sha256(f"{seed}:{index}".encode()).digest()
        for index in range(size // 32 + 1)
    )
    return b"".join(blocks)[:size]


FILES = {
    "cache/nested/scores.bin": noise("scores", 3000),
    "cache/text_embeddings/model.jsonl": noise("embeddings", 6000),
}


def git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def make_project(root: Path, *, with_data: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    (root / ".gitignore").write_text("data/cache/\ndata/work/\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "README.md").write_text("tracked\n", encoding="utf-8")
    git(root, "add", ".gitignore", "data/README.md")
    git(root, "commit", "-q", "-m", "init")
    if with_data:
        for relative, payload in FILES.items():
            path = root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        lock = root / "data" / "work" / "locks" / "probe.job.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("lock", encoding="utf-8")
    return root
```

- [ ] **Step 2: Write the failing tests**

Create `seed-pipeline/tests/data_archive/test_pack.py`:

```python
import hashlib
from pathlib import Path

import pytest

from seed_pipeline.data_archive.manifest import (
    PART_PREFIX,
    ArchiveError,
    ArchiveFile,
    ArchiveManifest,
    ArchivePart,
    read_archive_manifest,
    write_archive_manifest,
)
from seed_pipeline.data_archive.pack import git_head, list_archive_files, pack_archive
from tests.data_archive.archive_project import FILES, make_project


def test_only_ignored_files_outside_work_are_archived(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")

    assert list_archive_files(root) == sorted(FILES)


def test_pack_splits_the_stream_into_numbered_parts(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project")

    manifest = pack_archive(
        root / "data",
        list_archive_files(root),
        tmp_path / "archive",
        part_size=1024,
        created_at="2026-09-14T00:00:00+00:00",
        git_commit=git_head(root),
    )

    assert len(manifest.parts) > 1
    assert [part.name for part in manifest.parts] == [
        f"{PART_PREFIX}{index:04d}" for index in range(1, len(manifest.parts) + 1)
    ]
    for part in manifest.parts:
        path = tmp_path / "archive" / part.name
        assert part.size == path.stat().st_size <= 1024
        assert part.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.files == tuple(
        ArchiveFile(relative, len(payload), hashlib.sha256(payload).hexdigest())
        for relative, payload in sorted(FILES.items())
    )
    assert len(manifest.git_commit) == 40


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = ArchiveManifest(
        "2026-09-14T00:00:00+00:00",
        "c" * 40,
        (ArchiveFile("cache/a.jsonl", 3, "a" * 64),),
        (ArchivePart(f"{PART_PREFIX}0001", 10, "b" * 64),),
    )
    path = tmp_path / "archive-manifest.json"

    write_archive_manifest(path, manifest)

    assert read_archive_manifest(path) == manifest


@pytest.mark.parametrize(
    "unsafe", ["../escape.txt", "/etc/passwd", "work/locks/probe.job.lock", "./x"]
)
def test_manifest_rejects_unsafe_paths(tmp_path: Path, unsafe: str) -> None:
    path = tmp_path / "archive-manifest.json"
    write_archive_manifest(
        path,
        ArchiveManifest(
            "2026-09-14T00:00:00+00:00",
            "c" * 40,
            (ArchiveFile(unsafe, 1, "a" * 64),),
            (),
        ),
    )

    with pytest.raises(ArchiveError, match="Unsafe archive path"):
        read_archive_manifest(path)
```

The helper is imported as `tests.data_archive.archive_project`, the same way the Kaggle tests import `tests.integrations.kaggle.factories`; ruff sorts it after the `seed_pipeline` imports.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/data_archive/test_pack.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.data_archive'`.

- [ ] **Step 4: Create `manifest.py`**

```python
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_SCHEMA = "seed-data-archive-v1"
ARCHIVE_MANIFEST_NAME = "archive-manifest.json"
PART_PREFIX = "seed-pipeline-data.tar.zst.part-"
_PART_NAME = re.compile(rf"{re.escape(PART_PREFIX)}\d{{4}}")


class ArchiveError(RuntimeError):
    """Raised when the data archive is incomplete, corrupt or unsafe to restore."""


@dataclass(frozen=True)
class ArchiveFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArchivePart:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArchiveManifest:
    created_at: str
    git_commit: str
    files: tuple[ArchiveFile, ...]
    parts: tuple[ArchivePart, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARCHIVE_SCHEMA,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "file_count": len(self.files),
            "files": [asdict(item) for item in self.files],
            "parts": [asdict(item) for item in self.parts],
        }


def safe_data_path(value: str) -> str:
    """A normalized relative path inside data/ that is not scratch space."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or path.parts[0] == "work"
    ):
        raise ArchiveError(f"Unsafe archive path: {value!r}")
    return value


def read_archive_manifest(path: Path) -> ArchiveManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARCHIVE_SCHEMA:
        raise ArchiveError(f"{path} is not a {ARCHIVE_SCHEMA} manifest")
    files = tuple(
        ArchiveFile(
            safe_data_path(str(item["path"])), int(item["size"]), str(item["sha256"])
        )
        for item in payload["files"]
    )
    parts = tuple(
        ArchivePart(str(item["name"]), int(item["size"]), str(item["sha256"]))
        for item in payload["parts"]
    )
    if int(payload["file_count"]) != len(files):
        raise ArchiveError(f"{path} file_count does not match its file list")
    if len({item.path for item in files}) != len(files):
        raise ArchiveError(f"{path} lists a file more than once")
    for part in parts:
        if _PART_NAME.fullmatch(part.name) is None:
            raise ArchiveError(f"Unexpected archive part name: {part.name!r}")
    return ArchiveManifest(
        str(payload["created_at"]), str(payload["git_commit"]), files, parts
    )


def write_archive_manifest(path: Path, manifest: ArchiveManifest) -> None:
    Path(path).write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 5: Create `pack.py`**

```python
from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from collections.abc import Buffer, Sequence
from pathlib import Path
from typing import BinaryIO

import zstandard

from seed_pipeline.data_archive.manifest import (
    PART_PREFIX,
    ArchiveError,
    ArchiveFile,
    ArchiveManifest,
    ArchivePart,
    safe_data_path,
)

PART_SIZE_BYTES = 1900 * 1024 * 1024
ZSTD_LEVEL = 10


def list_archive_files(project_root: Path) -> list[str]:
    """Git-ignored files under data/, relative to data/, without data/work/."""
    output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            "data",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
    ).stdout
    files: list[str] = []
    for item in output.split(b"\0"):
        if not item:
            continue
        relative = item.decode("utf-8").removeprefix("data/")
        if not relative.startswith("work/"):
            files.append(safe_data_path(relative))
    return sorted(files)


def git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ArchiveError(
            f"Cannot read the Git commit of {project_root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


class PartWriter(io.RawIOBase):
    """Binary sink that splits a stream into numbered part files while hashing them."""

    def __init__(self, directory: Path, *, part_size: int) -> None:
        super().__init__()
        if part_size < 1:
            raise ValueError("part_size must be >= 1")
        self._directory = Path(directory)
        self._part_size = part_size
        self._handle: BinaryIO | None = None
        self._name = ""
        self._digest = hashlib.sha256()
        self._written = 0
        self.parts: list[ArchivePart] = []

    def writable(self) -> bool:
        return True

    def write(self, data: Buffer, /) -> int:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            handle = self._current_handle()
            take = min(len(view) - offset, self._part_size - self._written)
            chunk = view[offset : offset + take]
            handle.write(chunk)
            self._digest.update(chunk)
            self._written += take
            offset += take
        return len(view)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        self._finish_part()
        super().close()

    def _current_handle(self) -> BinaryIO:
        if self._handle is not None and self._written < self._part_size:
            return self._handle
        self._finish_part()
        self._name = f"{PART_PREFIX}{len(self.parts) + 1:04d}"
        handle = (self._directory / self._name).open("wb")
        self._handle = handle
        return handle

    def _finish_part(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self.parts.append(
            ArchivePart(self._name, self._written, self._digest.hexdigest())
        )
        self._handle = None
        self._digest = hashlib.sha256()
        self._written = 0


class _HashingReader(io.RawIOBase):
    def __init__(self, handle: BinaryIO) -> None:
        super().__init__()
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        view = memoryview(buffer).cast("B")
        data = self._handle.read(len(view))
        view[: len(data)] = data
        self._digest.update(data)
        self.size += len(data)
        return len(data)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def pack_archive(
    data_dir: Path,
    files: Sequence[str],
    output_dir: Path,
    *,
    part_size: int,
    created_at: str,
    git_commit: str,
) -> ArchiveManifest:
    if not files:
        raise ArchiveError(f"No Git-ignored files under {data_dir} to archive")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = PartWriter(output_dir, part_size=part_size)
    records: list[ArchiveFile] = []
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
    with (
        compressor.stream_writer(writer) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for relative in files:
            path = Path(data_dir) / relative
            info = tarfile.TarInfo(relative)
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open("rb") as handle:
                reader = _HashingReader(handle)
                archive.addfile(info, io.BufferedReader(reader))
            if reader.size != info.size:
                raise ArchiveError(f"{relative} changed while it was being archived")
            records.append(ArchiveFile(relative, info.size, reader.hexdigest()))
    return ArchiveManifest(created_at, git_commit, tuple(records), tuple(writer.parts))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/data_archive/test_pack.py`
Expected: PASS.

- [ ] **Step 7: Run the seed gate and commit**

Run the **Seed gate**. Expected: PASS.

```bash
git add seed-pipeline/src/seed_pipeline/data_archive seed-pipeline/tests/data_archive
git commit -m "feat(seed): pack ignored data into checksummed zstd tar parts"
```

---

### Task 3: Restoring an archive

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/data_archive/unpack.py`
- Create: `seed-pipeline/tests/data_archive/test_unpack.py`

**Interfaces:**
- Consumes: Task 2 (`pack_archive`, `list_archive_files`, `git_head`, manifest types), `seed_pipeline.evaluation.artifact_contracts.sha256_file`.
- Produces: `UnpackResult(written: int, unchanged: int)`; `part_matches(path, part) -> bool`; `verify_parts(archive_dir, manifest) -> None`; `unpack_archive(archive_dir, manifest, data_dir, *, force) -> UnpackResult`.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/data_archive/test_unpack.py`:

```python
from pathlib import Path

import pytest

from seed_pipeline.data_archive.manifest import ArchiveError, ArchiveManifest
from seed_pipeline.data_archive.pack import git_head, list_archive_files, pack_archive
from seed_pipeline.data_archive.unpack import UnpackResult, unpack_archive
from tests.data_archive.archive_project import FILES, make_project


def packed(tmp_path: Path) -> tuple[Path, ArchiveManifest]:
    root = make_project(tmp_path / "project")
    archive = tmp_path / "archive"
    manifest = pack_archive(
        root / "data",
        list_archive_files(root),
        archive,
        part_size=1024,
        created_at="2026-09-14T00:00:00+00:00",
        git_commit=git_head(root),
    )
    return archive, manifest


def test_unpack_restores_every_file_and_skips_identical_ones(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    target = tmp_path / "restore" / "data"

    first = unpack_archive(archive, manifest, target, force=False)
    second = unpack_archive(archive, manifest, target, force=False)

    assert first == UnpackResult(written=2, unchanged=0)
    assert second == UnpackResult(written=0, unchanged=2)
    for relative, payload in FILES.items():
        assert (target / relative).read_bytes() == payload


def test_a_corrupt_part_is_rejected_before_writing(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    broken = archive / manifest.parts[1].name
    data = bytearray(broken.read_bytes())
    data[0] ^= 0xFF
    broken.write_bytes(bytes(data))
    target = tmp_path / "restore" / "data"

    with pytest.raises(ArchiveError, match=manifest.parts[1].name):
        unpack_archive(archive, manifest, target, force=False)
    assert not target.exists()


def test_a_different_existing_file_stops_unless_forced(tmp_path: Path) -> None:
    archive, manifest = packed(tmp_path)
    target = tmp_path / "restore" / "data"
    changed = target / "cache" / "nested" / "scores.bin"
    changed.parent.mkdir(parents=True)
    changed.write_bytes(b"local edit")

    with pytest.raises(ArchiveError, match="--force"):
        unpack_archive(archive, manifest, target, force=False)
    assert changed.read_bytes() == b"local edit"

    result = unpack_archive(archive, manifest, target, force=True)

    assert result == UnpackResult(written=2, unchanged=0)
    assert changed.read_bytes() == FILES["cache/nested/scores.bin"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/data_archive/test_unpack.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.data_archive.unpack'`.

- [ ] **Step 3: Create `unpack.py`**

```python
from __future__ import annotations

import hashlib
import io
import os
import tarfile
from collections.abc import Buffer, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

import zstandard

from seed_pipeline.data_archive.manifest import (
    ArchiveError,
    ArchiveFile,
    ArchiveManifest,
    ArchivePart,
)
from seed_pipeline.evaluation.artifact_contracts import sha256_file


@dataclass(frozen=True)
class UnpackResult:
    written: int
    unchanged: int


def part_matches(path: Path, part: ArchivePart) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == part.size
        and sha256_file(path) == part.sha256
    )


def verify_parts(archive_dir: Path, manifest: ArchiveManifest) -> None:
    for part in manifest.parts:
        if not part_matches(Path(archive_dir) / part.name, part):
            raise ArchiveError(
                f"Archive part {part.name} is missing or does not match its sha256"
            )


def _file_matches(path: Path, record: ArchiveFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == record.size
        and sha256_file(path) == record.sha256
    )


class _PartReader(io.RawIOBase):
    """Reads the part files one after another as a single stream."""

    def __init__(self, paths: Sequence[Path]) -> None:
        super().__init__()
        self._pending = list(paths)
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        view = memoryview(buffer).cast("B")
        while True:
            if self._handle is None:
                if not self._pending:
                    return 0
                self._handle = self._pending.pop(0).open("rb")
            data = self._handle.read(len(view))
            if data:
                view[: len(data)] = data
                return len(data)
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _write_verified(target: Path, payload: IO[bytes], record: ArchiveFile) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pulling")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as handle:
        while block := payload.read(1024 * 1024):
            handle.write(block)
            digest.update(block)
            size += len(block)
    if size != record.size or digest.hexdigest() != record.sha256:
        temporary.unlink()
        raise ArchiveError(f"{record.path} does not match the archive manifest")
    os.replace(temporary, target)


def unpack_archive(
    archive_dir: Path, manifest: ArchiveManifest, data_dir: Path, *, force: bool
) -> UnpackResult:
    verify_parts(archive_dir, manifest)
    data_dir = Path(data_dir)
    expected = {item.path: item for item in manifest.files}
    unchanged: set[str] = set()
    conflicts: list[str] = []
    for item in manifest.files:
        target = data_dir / item.path
        if not target.exists():
            continue
        if _file_matches(target, item):
            unchanged.add(item.path)
        else:
            conflicts.append(item.path)
    if conflicts and not force:
        raise ArchiveError(
            f"{len(conflicts)} file(s) in {data_dir} differ from the archive "
            f"(first: {', '.join(conflicts[:5])}); use --force to overwrite them"
        )
    seen: set[str] = set()
    written = 0
    parts = [Path(archive_dir) / part.name for part in manifest.parts]
    with (
        io.BufferedReader(_PartReader(parts)) as source,
        zstandard.ZstdDecompressor().stream_reader(source) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            record = expected.get(member.name)
            if record is None or member.name in seen or not member.isfile():
                raise ArchiveError(f"Unexpected archive member: {member.name!r}")
            seen.add(member.name)
            if member.name in unchanged:
                continue
            payload = archive.extractfile(member)
            if payload is None:
                raise ArchiveError(f"Cannot read archive member: {member.name}")
            _write_verified(data_dir / record.path, payload, record)
            written += 1
    missing = set(expected) - seen
    if missing:
        raise ArchiveError(
            f"The archive lacks {len(missing)} file(s) that its manifest lists"
        )
    return UnpackResult(written, len(unchanged))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/data_archive`
Expected: PASS.

- [ ] **Step 5: Run the seed gate and commit**

Run the **Seed gate**. Expected: PASS.

```bash
git add seed-pipeline/src/seed_pipeline/data_archive/unpack.py seed-pipeline/tests/data_archive/test_unpack.py
git commit -m "feat(seed): restore the data archive with per-part and per-file checksums"
```

---

### Task 4: `seed data push` and `seed data pull`

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/data_archive/service.py`, `seed-pipeline/src/seed_pipeline/cli/commands/data.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/app.py`, `seed-pipeline/docs/guides/cli-reference.md`, `seed-pipeline/data/README.md`
- Create: `seed-pipeline/tests/data_archive/test_service.py`, `seed-pipeline/tests/cli/test_data_command.py`

**Interfaces:**
- Consumes: Tasks 1–3; `DatasetService`; `kaggle_job_lock(target)`; `resolve_execution_context(kaggle_account)` (`.runner`, `.owners.execution`); `paths.PROJECT_ROOT`.
- Produces: `DATASET_SLUG = "seed-pipeline-data"`, `DATASET_TITLE = "Seed pipeline data"`, `archive_dir(project_root) -> Path`, `PushResult(reference, version, file_count, part_count)`, `PullResult(reference, file_count, written, unchanged, downloaded_parts)`, `push_data(*, project_root, dataset, message, now, part_size=PART_SIZE_BYTES) -> PushResult`, `pull_data(*, project_root, dataset, force) -> PullResult`; CLI `seed data push [--kaggle-account] [--message]`, `seed data pull [--kaggle-account] [--force]`.

- [ ] **Step 1: Write the failing service tests**

Create `seed-pipeline/tests/data_archive/test_service.py`:

```python
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from seed_pipeline.data_archive.manifest import ARCHIVE_MANIFEST_NAME
from seed_pipeline.data_archive.service import pull_data, push_data
from seed_pipeline.integrations.kaggle.dataset_service import DatasetService
from seed_pipeline.integrations.kaggle.errors import KaggleCommandError
from tests.data_archive.archive_project import FILES, make_project

NOW = datetime(2026, 9, 14, tzinfo=UTC)


class FakeKaggle:
    """A private dataset kept in a local folder, driven by Kaggle CLI arguments."""

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.version = 0
        self.calls: list[list[str]] = []

    def run(
        self,
        args: list[str],
        /,
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        del capture_output, live_output
        self.calls.append(args)
        action = args[1:3]
        if action == ["datasets", "status"]:
            if self.version == 0:
                raise KaggleCommandError(
                    operation="status",
                    target=args[3],
                    returncode=1,
                    stdout="",
                    stderr="404 Not Found",
                )
            return json.dumps(
                {"status": "READY", "current_version_number": self.version}
            )
        if action in (["datasets", "create"], ["datasets", "version"]):
            shutil.rmtree(self.remote, ignore_errors=True)
            shutil.copytree(Path(args[args.index("-p") + 1]), self.remote)
            self.version += 1
            return ""
        if action == ["datasets", "download"]:
            name = args[args.index("-f") + 1]
            destination = Path(args[args.index("-p") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.remote / name, destination / name)
            return ""
        raise AssertionError(f"unexpected Kaggle command: {args}")


def test_push_then_pull_restores_the_ignored_files(tmp_path: Path) -> None:
    kaggle = FakeKaggle(tmp_path / "remote")
    dataset = DatasetService(kaggle, "owner")
    source = make_project(tmp_path / "source")

    pushed = push_data(
        project_root=source, dataset=dataset, message="First", now=NOW, part_size=1024
    )

    assert (pushed.reference, pushed.version, pushed.file_count) == (
        "owner/seed-pipeline-data",
        1,
        len(FILES),
    )
    create = next(call for call in kaggle.calls if call[1:3] == ["datasets", "create"])
    assert "--public" not in create
    assert create[create.index("--dir-mode") + 1] == "skip"
    assert (kaggle.remote / ARCHIVE_MANIFEST_NAME).is_file()

    clone = make_project(tmp_path / "clone", with_data=False)
    pulled = pull_data(project_root=clone, dataset=dataset, force=False)

    assert (pulled.written, pulled.unchanged) == (len(FILES), 0)
    assert pulled.downloaded_parts == pushed.part_count
    for relative, payload in FILES.items():
        assert (clone / "data" / relative).read_bytes() == payload

    again = pull_data(project_root=clone, dataset=dataset, force=False)

    assert (again.written, again.unchanged, again.downloaded_parts) == (
        0,
        len(FILES),
        0,
    )


def test_a_second_push_creates_a_version_with_the_message(tmp_path: Path) -> None:
    kaggle = FakeKaggle(tmp_path / "remote")
    dataset = DatasetService(kaggle, "owner")
    source = make_project(tmp_path / "source")
    push_data(project_root=source, dataset=dataset, message="First", now=NOW)

    second = push_data(project_root=source, dataset=dataset, message="Refresh", now=NOW)

    version = kaggle.calls[-1]
    assert version[1:3] == ["datasets", "version"]
    assert version[version.index("-m") + 1] == "Refresh"
    assert second.version == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/data_archive/test_service.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.data_archive.service'`.

- [ ] **Step 3: Create `service.py`**

```python
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
```

- [ ] **Step 4: Run the service tests to verify they pass**

Run: `uv run pytest -q tests/data_archive/test_service.py`
Expected: PASS.

- [ ] **Step 5: Write the failing CLI tests**

Create `seed-pipeline/tests/cli/test_data_command.py`:

```python
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.data as data_command
from seed_pipeline.cli.app import app
from seed_pipeline.data_archive.service import PullResult, PushResult

runner = CliRunner()


@pytest.fixture
def accounts(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    seen: list[str | None] = []

    def resolve(kaggle_account: str | None) -> SimpleNamespace:
        seen.append(kaggle_account)
        return SimpleNamespace(
            runner=object(), owners=SimpleNamespace(execution="owner")
        )

    monkeypatch.setattr(data_command, "resolve_execution_context", resolve)
    return seen


def test_push_forwards_account_and_message(
    monkeypatch: pytest.MonkeyPatch, accounts: list[str | None]
) -> None:
    captured: dict = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return PushResult("owner/seed-pipeline-data", 2, 10, 1)

    monkeypatch.setattr(data_command, "push_data", fake_push)

    result = runner.invoke(
        app,
        ["--json", "data", "push", "--kaggle-account", "acc2", "--message", "Refresh"],
    )

    assert result.exit_code == 0, result.output
    assert accounts == ["acc2"]
    assert captured["message"] == "Refresh"
    assert captured["dataset"].owner == "owner"
    assert json.loads(result.stdout)["details"]["dataset"] == "owner/seed-pipeline-data"


def test_pull_forwards_force(
    monkeypatch: pytest.MonkeyPatch, accounts: list[str | None]
) -> None:
    captured: dict = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return PullResult("owner/seed-pipeline-data", 10, 3, 7, 1)

    monkeypatch.setattr(data_command, "pull_data", fake_pull)

    result = runner.invoke(app, ["--json", "data", "pull", "--force"])

    assert result.exit_code == 0, result.output
    assert accounts == [None]
    assert captured["force"] is True
    assert json.loads(result.stdout)["details"]["written"] == 3
```

Run: `uv run pytest -q tests/cli/test_data_command.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.cli.commands.data'`.

- [ ] **Step 6: Create the CLI**

Create `cli/commands/data.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import PROJECT_ROOT
from seed_pipeline.data_archive.service import pull_data, push_data
from seed_pipeline.integrations.kaggle.dataset_service import DatasetService
from seed_pipeline.integrations.kaggle.service import resolve_execution_context

data_app = typer.Typer(
    help="Store Git-ignored data/ files in a private Kaggle dataset and restore them.",
    no_args_is_help=True,
)


def _dataset(kaggle_account: str | None) -> DatasetService:
    context = resolve_execution_context(kaggle_account)
    return DatasetService(context.runner, context.owners.execution)


@data_app.command("push")
def push(
    ctx: typer.Context,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
    message: Annotated[str, typer.Option("--message")] = "Update seed-pipeline data",
) -> None:
    def run() -> CommandResult:
        result = push_data(
            project_root=PROJECT_ROOT,
            dataset=_dataset(kaggle_account),
            message=message,
            now=datetime.now(UTC),
        )
        return CommandResult(
            "data push",
            CommandStatus.COMPLETE,
            None,
            {
                "dataset": result.reference,
                "version": result.version,
                "files": result.file_count,
                "parts": result.part_count,
            },
        )

    run_handler(state_from_context(ctx), run)


@data_app.command("pull")
def pull(
    ctx: typer.Context,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    def run() -> CommandResult:
        result = pull_data(
            project_root=PROJECT_ROOT, dataset=_dataset(kaggle_account), force=force
        )
        return CommandResult(
            "data pull",
            CommandStatus.COMPLETE,
            PROJECT_ROOT / "data",
            {
                "dataset": result.reference,
                "files": result.file_count,
                "written": result.written,
                "unchanged": result.unchanged,
                "downloaded_parts": result.downloaded_parts,
            },
        )

    run_handler(state_from_context(ctx), run)
```

In `cli/app.py` add `from seed_pipeline.cli.commands.data import data_app` (sorted with the other command imports) and `app.add_typer(data_app, name="data")` after the `bundle` registration.

- [ ] **Step 7: Document the commands**

- `docs/guides/cli-reference.md`: add a `seed data` section with these two code lines and one sentence each on what they archive, where parts are staged and when `--force` is needed:

```bash
uv run seed data push --kaggle-account acc1 --message "Rebuild evaluation runs"
uv run seed data pull --kaggle-account acc1
```

- `data/README.md`: add the same two commands under the paragraph that says ignored data lives in the Kaggle archive, plus: "The first push after a fresh setup needs the owner's confirmation; pull refuses to overwrite files that differ from the archive unless `--force` is given."

- [ ] **Step 8: Run the seed gate**

Run the **Seed gate**. Expected: PASS, including `tests/test_docs.py::test_documented_seed_commands_parse` for the two new documented commands.

- [ ] **Step 9: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/data_archive/service.py seed-pipeline/src/seed_pipeline/cli/commands/data.py seed-pipeline/src/seed_pipeline/cli/app.py seed-pipeline/tests/data_archive/test_service.py seed-pipeline/tests/cli/test_data_command.py seed-pipeline/docs/guides/cli-reference.md seed-pipeline/data/README.md
git commit -m "feat(seed): add seed data push and pull for the private Kaggle archive"
```
