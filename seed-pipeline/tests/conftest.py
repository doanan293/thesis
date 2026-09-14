from pathlib import Path

import pytest

from seed_pipeline.integrations.kaggle import job_lock


@pytest.fixture(autouse=True)
def isolated_lock_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Locks taken by tests never land in the real data/work/locks."""
    root = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(job_lock, "LOCK_DIR", root)
    return root
