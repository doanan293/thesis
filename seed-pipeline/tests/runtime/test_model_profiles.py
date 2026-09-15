import pytest

from seed_pipeline.runtime.catalog import RERANKER_MODELS, require_model
from seed_pipeline.runtime.model_profiles import (
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
)


def test_rerank_contract_rejects_other_protocols():
    with pytest.raises(ValueError, match="unsupported rerank protocol"):
        RerankContract(protocol="completion_logprobs")


def test_embedding_profile_separates_query_and_corpus_workloads():
    profile = require_model("qwen3-embedding:4b-fp16").embedding_runtime

    assert isinstance(profile, EmbeddingRuntimeProfile)
    assert isinstance(profile.query, EmbeddingWorkloadProfile)
    assert isinstance(profile.corpus, EmbeddingWorkloadProfile)
    assert profile.query is not profile.corpus
    assert profile.query.production_batch_size > 0
    assert profile.corpus.production_batch_size > 0


def test_embedding_model_exposes_separate_runtime_search_spaces():
    search_space = require_model("qwen3-embedding:4b-fp16").embedding_search_space

    assert search_space is not None
    assert search_space.query != search_space.corpus
    assert all(
        candidate.request_batch_size > 0 for candidate in search_space.query.candidates
    )


@pytest.mark.parametrize(
    ("model", "server_slots", "ubatch_sizes", "concurrency"),
    [
        ("qwen3-reranker:0.6b-fp16", 64, (8192, 16384, 32768), 4),
        ("qwen3-reranker:4b-fp16", 32, (8192, 16384), 3),
        ("qwen3-reranker:8b-fp16", 16, (4096, 8192), 2),
        ("bge-reranker-v2-m3:f16", 64, (8192, 16384, 32768), 4),
    ],
)
def test_kaggle_rerank_search_space_sweeps_the_ubatch(
    model: str, server_slots: int, ubatch_sizes: tuple[int, ...], concurrency: int
):
    space = require_model(model).rerank_search_space

    assert space is not None
    assert tuple(item.physical_batch_size for item in space.candidates) == ubatch_sizes
    for item in space.candidates:
        assert (
            item.server_slots,
            item.request_batch_size,
            item.concurrency,
            item.threads,
        ) == (server_slots, 30, concurrency, None)
        assert item.context_per_slot == item.logical_batch_size
        assert item.logical_batch_size == item.physical_batch_size


@pytest.mark.parametrize("model", sorted(RERANKER_MODELS))
def test_local_rerank_search_space_sweeps_ubatch_and_threads(model: str):
    space = require_model(model).local_rerank_search_space

    assert space is not None
    assert [(item.physical_batch_size, item.threads) for item in space.candidates] == [
        (4096, 8),
        (4096, 12),
        (8192, 8),
        (8192, 12),
        (16384, 8),
        (16384, 12),
    ]
    for item in space.candidates:
        assert (item.server_slots, item.request_batch_size, item.concurrency) == (
            16,
            15,
            1,
        )
        assert item.context_per_slot == item.logical_batch_size
        assert item.logical_batch_size == item.physical_batch_size
