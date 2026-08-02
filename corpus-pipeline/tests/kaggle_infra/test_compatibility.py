from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.api import KaggleCommandRunner
from corpus_pipeline.integrations.kaggle.workspace import JobRunLock


def test_canonical_runner_requires_kaggle_command():
    with pytest.raises(ValueError, match="must start with 'kaggle'"):
        KaggleCommandRunner().run(["not-kaggle", "status"])


def test_generic_job_lock_rejects_second_holder(tmp_path: Path):
    path = tmp_path / "job.lock"
    with JobRunLock(path, "rerank:abc"):
        with pytest.raises(RuntimeError, match="already active"):
            with JobRunLock(path, "rerank:abc"):
                pass
