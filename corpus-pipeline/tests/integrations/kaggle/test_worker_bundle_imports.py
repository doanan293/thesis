from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

from corpus_pipeline.config.paths import PROJECT_ROOT
from corpus_pipeline.integrations.kaggle.kernels import PipelineKernelService


def _source_bundle(tmp_path: Path) -> Path:
    service = PipelineKernelService(
        service=None,  # type: ignore[arg-type]
        owner="test-owner",
        source_root=PROJECT_ROOT / "src",
    )
    bundle = tmp_path / "source_bundle.zip"
    bundle.write_bytes(base64.b64decode(service._build_source_b64()))
    return bundle


@pytest.mark.parametrize(
    "worker_module",
    [
        "corpus_pipeline.integrations.kaggle.workers.query_embed",
        "corpus_pipeline.integrations.kaggle.workers.corpus_embed",
        "corpus_pipeline.integrations.kaggle.workers.rerank",
    ],
)
def test_worker_imports_from_source_bundle_without_local_workspace(
    tmp_path: Path, worker_module: str
):
    bundle = _source_bundle(tmp_path)
    code = (
        "import importlib, sys; "
        f"sys.path.insert(0, {str(bundle)!r}); "
        f"module = importlib.import_module({worker_module!r}); "
        f"assert str(module.__file__).startswith({str(bundle)!r}); "
        "assert 'corpus_pipeline.config.paths' not in sys.modules"
    )
    environment = dict(os.environ)
    environment.pop("CORPUS_PIPELINE_ROOT", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
