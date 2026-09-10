import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle.workers.query_embed import (
    run_query_embed_worker,
)
from corpus_pipeline.runtime.client import LlamaCppRequestError, LlamaCppResponseError


def _config(tmp_path: Path) -> dict:
    input_path = tmp_path / "queries.jsonl"
    input_path.write_text(
        '{"query_id":"q1","query":"one"}\n{"query_id":"q2","query":"two"}\n',
        encoding="utf-8",
    )
    return {
        "input_files": {
            "input": {
                "path": str(input_path),
                "filename": input_path.name,
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
        },
        "output_dir": str(tmp_path / "out"),
        "model": "embed-model",
        "vector_dimension": 2,
        "job_sha256": "a" * 64,
        "identity": {"model": "embed-model"},
        "runtime_overrides": {
            "request_batch_size": 1,
            "concurrency": 1,
        },
    }


@contextmanager
def _fake_servers(_config, **_kwargs):
    yield [SimpleNamespace(base_url="http://fake")]


class _SucceedOnceThenDisconnect:
    calls = 0

    def __init__(self, _url):
        pass

    def embed(self, texts, _model, _dimension):
        type(self).calls += 1
        if type(self).calls == 1:
            return [[1.0, 2.0] for _ in texts]
        raise LlamaCppRequestError("connection refused", retryable=True)


def test_query_embed_seals_partial_artifact_after_server_failure(tmp_path):
    _SucceedOnceThenDisconnect.calls = 0
    artifact = run_query_embed_worker(
        _config(tmp_path),
        server_manager=_fake_servers,
        client_factory=_SucceedOnceThenDisconnect,
    )

    assert artifact.completion.complete == 1
    assert artifact.completion.missing == 1
    assert artifact.checkpoint_path is not None
    manifest = json.loads((artifact.data_path.parent / "manifest.json").read_text())
    assert manifest["runtime"]["termination"]["recoverable"] is True


class _AlwaysInvalidClient:
    def __init__(self, _url):
        pass

    def embed(self, *_args):
        raise LlamaCppResponseError("invalid response")


def test_query_embed_keeps_fatal_server_error_fatal(tmp_path):
    try:
        run_query_embed_worker(
            _config(tmp_path),
            server_manager=_fake_servers,
            client_factory=_AlwaysInvalidClient,
        )
    except LlamaCppResponseError:
        pass
    else:
        raise AssertionError("fatal response error should be propagated")
