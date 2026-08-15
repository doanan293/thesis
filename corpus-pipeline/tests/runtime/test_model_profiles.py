from dataclasses import replace

import pytest

from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.runtime.model_profiles import (
    CompletionScoring,
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    RerankRuntimeProfile,
    build_qwen3_yes_no_prompt,
)


def test_qwen_prompt_contains_canonical_instruction_and_turns():
    prompt = build_qwen3_yes_no_prompt("thuốc gì", "tài liệu")

    assert "<|im_start|>system" in prompt
    assert "<Instruct>: Given a Vietnamese medical retrieval query" in prompt
    assert "<Query>: thuốc gì" in prompt
    assert "<Document>: tài liệu" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen_contract_hash_changes_for_semantic_fields():
    spec = require_model("qwen3-reranker:0.6b-fp16")
    contract = spec.rerank_contract

    assert isinstance(contract, RerankContract)
    assert replace(contract, instruction=contract.instruction + " changed").sha256 != contract.sha256
    assert replace(
        contract,
        scoring=replace(contract.scoring, n_predict=2),
    ).sha256 != contract.sha256


def test_runtime_changes_do_not_change_rerank_contract_hash():
    spec = require_model("qwen3-reranker:0.6b-fp16")
    tuned = replace(spec, rerank_runtime=replace(spec.rerank_runtime, concurrency_per_gpu=2))

    assert tuned.rerank_runtime != spec.rerank_runtime
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
    spec = require_model("qwen3-embedding:4b-fp16")

    assert spec.embedding_search_space.query != spec.embedding_search_space.corpus
    assert all(
        candidate.request_batch_size > 0
        for candidate in spec.embedding_search_space.query.candidates
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


def test_native_contract_rejects_completion_settings():
    scoring = CompletionScoring(
        positive_token="yes",
        negative_token="no",
        n_predict=1,
        temperature=1.0,
        samplers=("temperature",),
        n_probs=2,
        min_keep=2,
        post_sampling_probs=True,
        logit_bias=100.0,
    )

    with pytest.raises(ValueError, match="native_rerank"):
        RerankContract(
            protocol="native_rerank",
            template_id=None,
            template_version=None,
            instruction="",
            scoring=scoring,
        )
