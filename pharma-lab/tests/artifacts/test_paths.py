import re
from pathlib import Path

import pytest

from pharma_lab.artifacts.paths import ArtifactPaths, retain_failed_workspace


def test_build_workspace_has_a_fixed_readable_name(tmp_path: Path) -> None:
    paths = ArtifactPaths.create(tmp_path / "build")

    assert paths.root == tmp_path / "build" / "in-progress"
    assert all(directory.is_dir() for directory in paths.directories())
    assert not any(
        re.search(r"[0-9a-f]{12,}", part)
        for part in paths.root.relative_to(tmp_path).parts
    )


def test_an_unfinished_workspace_blocks_a_new_build(tmp_path: Path) -> None:
    ArtifactPaths.create(tmp_path / "build")

    with pytest.raises(FileExistsError, match="in-progress"):
        ArtifactPaths.create(tmp_path / "build")


def test_failed_workspace_is_kept_as_latest(tmp_path: Path) -> None:
    paths = ArtifactPaths.create(tmp_path / "build")

    latest = retain_failed_workspace(paths)

    assert latest == tmp_path / "build" / "failed" / "latest"
    assert not paths.root.exists()
