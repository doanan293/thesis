import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.workers.rerank import run_rerank_worker
from corpus_pipeline.runtime.client import (
    CompletionPromptTiming,
    CompletionRerankResult,
    LlamaCppRequestError,
    LlamaCppResponseError,
)


def _write_candidates(
    path: Path,
    query_count: int,
    candidates_per_query: int,
    candidate_counts: list[int] | None = None,
) -> None:
    counts = candidate_counts or [candidates_per_query] * query_count
    with path.open("w", encoding="utf-8") as handle:
        for query_index, candidate_count in enumerate(counts):
            row = {
                "query_id": f"q-{query_index}",
                "query": f"query {query_index}",
                "candidates": [
                    {
                        "chunk_id": f"c-{query_index}-{candidate_index}",
                        "document_text": f"document {query_index}-{candidate_index}",
                    }
                    for candidate_index in range(candidate_count)
                ],
            }
            handle.write(json.dumps(row) + "\n")


def _config(
    tmp_path: Path,
    query_count: int,
    candidates_per_query: int,
    candidate_counts: list[int] | None = None,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates = tmp_path / "candidates.jsonl"
    manifest = tmp_path / "candidate-manifest.json"
    _write_candidates(candidates, query_count, candidates_per_query, candidate_counts)
    manifest.write_text("{}\n", encoding="utf-8")
    return {
        "model": "qwen3-reranker:0.6b-fp16",
        "protocol": "completion_logprobs",
        "output_dir": str(tmp_path / "output"),
        "batch_size": 32,
        "identity": {"stage": "rerank", "model_sha256": "a" * 64},
        "job_sha256": "b" * 64,
        "input_files": {
            "candidates": {
                "path": str(candidates),
                "filename": candidates.name,
                "sha256": sha256_file(candidates),
            },
            "candidate_manifest": {
                "path": str(manifest),
                "filename": manifest.name,
                "sha256": sha256_file(manifest),
            },
        },
    }


def test_rerank_worker_emits_input_milestone_and_artifact_progress(tmp_path):
    messages: list[str] = []
    ticks = iter((0.0, 10.0, 20.0))

    artifact = run_rerank_worker(
        _config(tmp_path, query_count=501, candidates_per_query=2),
        score_pair=lambda _query, _document, _model: 0.5,
        emit=messages.append,
        clock=lambda: next(ticks),
    )

    assert artifact.completion.complete == 1002
    assert messages[0] == (
        "input queries total=501 reusable=0 missing=501 "
        "pairs total=1002 reusable=0 missing=1002"
    )
    assert any(
        message.startswith(
            "milestone=500 processed=501/501 queries pairs=1002/1002 "
            "progress=100.00% elapsed=00:00:10 "
        )
        and "recent_rate=100.20 average_rate=100.20 pairs/s" in message
        for message in messages
    )
    assert messages[-1] == "artifact queries=501/501 pairs=1002/1002 complete=true"
    joined = "\n".join(messages)
    assert "query 0" not in joined
    assert "document 0-0" not in joined


def test_rerank_worker_does_not_replay_resumed_milestones(tmp_path):
    initial = _config(tmp_path / "initial", query_count=500, candidates_per_query=1)
    run_rerank_worker(
        initial,
        score_pair=lambda _query, _document, _model: 0.5,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )
    checkpoint = Path(initial["output_dir"]) / "rerank_scores.journal.jsonl"

    resumed = _config(tmp_path / "resumed", query_count=501, candidates_per_query=1)
    resumed["checkpoint_filename"] = str(checkpoint)
    messages: list[str] = []
    run_rerank_worker(
        resumed,
        score_pair=lambda _query, _document, _model: 0.5,
        emit=messages.append,
        clock=lambda: 0.0,
    )

    assert messages[0] == (
        "input queries total=501 reusable=500 missing=1 "
        "pairs total=501 reusable=500 missing=1"
    )
    assert not any("milestone=500 " in message for message in messages)
    assert messages[-1] == "artifact queries=501/501 pairs=501/501 complete=true"


def test_rerank_worker_counts_queries_with_uneven_candidate_totals(tmp_path):
    messages: list[str] = []
    run_rerank_worker(
        _config(
            tmp_path,
            query_count=3,
            candidates_per_query=1,
            candidate_counts=[1, 3, 2],
        ),
        score_pair=lambda _query, _document, _model: 0.5,
        emit=messages.append,
        clock=lambda: 0.0,
    )

    assert messages[-1] == "artifact queries=3/3 pairs=6/6 complete=true"


def test_rerank_worker_emits_artifact_for_fully_reusable_checkpoint(tmp_path):
    config = _config(tmp_path, query_count=2, candidates_per_query=1)
    run_rerank_worker(
        config,
        score_pair=lambda _query, _document, _model: 0.5,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )
    messages: list[str] = []

    run_rerank_worker(
        config,
        score_pair=lambda _query, _document, _model: 0.5,
        emit=messages.append,
        clock=lambda: 0.0,
    )

    assert messages[0] == (
        "input queries total=2 reusable=2 missing=0 pairs total=2 reusable=2 missing=0"
    )
    assert messages[-1] == "artifact queries=2/2 pairs=2/2 complete=true"


@contextmanager
def _fake_server_manager(_config, **_kwargs):
    yield [SimpleNamespace(base_url="http://127.0.0.1:11434")]


class _SucceedOnceThenDisconnect:
    def __init__(self):
        self.calls = 0

    def rerank_native(self, _query, documents, _model):
        self.calls += 1
        if self.calls == 1:
            return [0.75 for _document in documents]
        raise LlamaCppRequestError("connection refused", retryable=True)


def test_rerank_worker_seals_partial_artifact_after_server_failure(tmp_path):
    config = _config(tmp_path, query_count=2, candidates_per_query=1)
    config["model"] = "bge-reranker-v2-m3:f16"
    config["protocol"] = "native_rerank"
    config["runtime_overrides"] = {
        "server_slots": 1,
        "concurrency": 1,
        "request_batch_size": 1,
        "context_per_slot": 4096,
        "logical_batch_size": 4096,
        "physical_batch_size": 2048,
    }
    client = _SucceedOnceThenDisconnect()

    artifact = run_rerank_worker(
        config,
        server_manager=_fake_server_manager,
        client_factory=lambda _url: client,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert artifact.completion.complete == 1
    assert artifact.completion.missing == 1
    assert artifact.checkpoint_path is not None
    assert manifest["runtime"]["termination"]["recoverable"] is True
    assert manifest["runtime"]["termination"]["category"] == (
        "model_server_unavailable"
    )


class _AlwaysInvalidClient:
    def rerank_native(self, _query, _documents, _model):
        raise LlamaCppResponseError("bad scores")


def test_rerank_worker_keeps_response_errors_fatal(tmp_path):
    config = _config(tmp_path, query_count=1, candidates_per_query=1)
    config["model"] = "bge-reranker-v2-m3:f16"
    config["protocol"] = "native_rerank"
    config["runtime_overrides"] = {
        "server_slots": 1,
        "concurrency": 1,
        "request_batch_size": 1,
        "context_per_slot": 4096,
        "logical_batch_size": 4096,
        "physical_batch_size": 2048,
    }

    with pytest.raises(LlamaCppResponseError, match="bad scores"):
        run_rerank_worker(
            config,
            server_manager=_fake_server_manager,
            client_factory=lambda _url: _AlwaysInvalidClient(),
        )

    assert not (Path(config["output_dir"]) / "manifest.json").exists()


class _QwenCacheAwareClient:
    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    async def rerank_completion_results_async(
        self, prompts, _model, *, concurrency, contract, cache_prompt
    ):
        assert concurrency == 1
        assert contract is not None
        self.calls.extend((prompt, cache_prompt) for prompt in prompts)
        return [
            CompletionRerankResult(
                score=0.5,
                timing=CompletionPromptTiming(cached_tokens=12, evaluated_tokens=3),
            )
            for _prompt in prompts
        ]


def test_completion_worker_enables_slot_prompt_cache_and_records_timings(tmp_path):
    config = _config(tmp_path, query_count=2, candidates_per_query=2)
    config["runtime_overrides"] = {
        "server_slots": 1,
        "concurrency": 1,
        "request_batch_size": 1,
        "context_per_slot": 4096,
        "logical_batch_size": 4096,
        "physical_batch_size": 2048,
    }
    client = _QwenCacheAwareClient()

    artifact = run_rerank_worker(
        config,
        server_manager=_fake_server_manager,
        client_factory=lambda _url: client,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )

    assert len(client.calls) == 4
    assert all(cache_prompt is True for _prompt, cache_prompt in client.calls)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    operation = manifest["runtime"]["operations"]
    assert operation["prompt_tokens_cached"] == 48
    assert operation["prompt_tokens_evaluated"] == 12
    assert operation["prompt_cache_hit_ratio"] == pytest.approx(0.8)
