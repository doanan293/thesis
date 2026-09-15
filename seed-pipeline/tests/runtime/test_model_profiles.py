from dataclasses import replace

import pytest

from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.model_profiles import (
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    RerankRuntimeProfile,
)


def test_rerank_contract_rejects_other_protocols():
    with pytest.raises(ValueError, match="unsupported rerank protocol"):
        RerankContract(protocol="completion_logprobs")


def test_runtime_changes_do_not_change_rerank_contract_hash():
    spec = require_model("qwen3-reranker:0.6b-fp16")
    assert spec.rerank_runtime is not None
    assert spec.rerank_contract is not None
    tuned = replace(
        spec, rerank_runtime=replace(spec.rerank_runtime, concurrency_per_gpu=2)
    )

    assert tuned.rerank_runtime != spec.rerank_runtime
    assert tuned.rerank_contract is not None
    assert tuned.rerank_contract.sha256 == spec.rerank_contract.sha256


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


def test_reranker_exposes_runtime_search_space():
    spec = require_model("qwen3-reranker:0.6b-fp16")

    assert spec.rerank_search_space is not None
    assert len(spec.rerank_search_space.candidates) >= 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RerankRuntimeProfile(
            server_slots_per_gpu=1,
            concurrency_per_gpu=0,
            context_per_slot=4096,
            logical_batch_size=4096,
            physical_batch_size=2048,
            benchmark_concurrency=(1,),
        ),
        lambda: RerankRuntimeProfile(
            server_slots_per_gpu=1,
            concurrency_per_gpu=2,
            context_per_slot=4096,
            logical_batch_size=4096,
            physical_batch_size=2048,
            benchmark_concurrency=(1,),
        ),
    ],
)
def test_rerank_runtime_rejects_invalid_concurrency(factory):
    with pytest.raises(ValueError):
        factory()
