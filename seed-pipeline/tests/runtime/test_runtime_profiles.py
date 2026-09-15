import json
from dataclasses import replace

import pytest

from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileIdentity,
    RuntimeProfileStore,
    RuntimeSearchSpace,
    rerank_concurrency,
    reranker_candidate,
    reranker_context_size,
)
from seed_pipeline.runtime.server_policy import inference_cache_policy


def candidate(concurrency: int = 2) -> RuntimeCandidate:
    return RuntimeCandidate(
        server_slots=4,
        concurrency=concurrency,
        request_batch_size=1,
        context_per_slot=4096,
        logical_batch_size=4096,
        physical_batch_size=2048,
    )


def identity(runtime_sha256: str = "a" * 64) -> RuntimeProfileIdentity:
    space = RuntimeSearchSpace((candidate(2), candidate(4)))
    return RuntimeProfileIdentity.create(
        workload="rerank",
        model="qwen3-reranker:0.6b-fp16",
        model_sha256="b" * 64,
        runtime_sha256=runtime_sha256,
        inference_cache_policy_sha256=inference_cache_policy(
            require_model("qwen3-reranker:0.6b-fp16")
        ).sha256,
        machine_shape="NvidiaTeslaT4",
        topology="replicated_2x1",
        search_space=space,
    )


def test_profile_identity_excludes_input_but_changes_with_runtime():
    assert identity().sha256 == identity().sha256
    assert identity().sha256 != identity("c" * 64).sha256


def test_runtime_profile_identity_changes_with_inference_cache_policy():
    common = {
        "workload": "rerank",
        "model": "bge-reranker-v2-m3:f16",
        "model_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "machine_shape": "NvidiaTeslaT4",
        "topology": "replicated_2x1",
        "search_space": RuntimeSearchSpace((candidate(),)),
    }

    before = RuntimeProfileIdentity.create(
        **common, inference_cache_policy_sha256="c" * 64
    )
    after = RuntimeProfileIdentity.create(
        **common, inference_cache_policy_sha256="d" * 64
    )

    assert before.sha256 != after.sha256
    assert after.payload["inference_cache_policy_sha256"] == "d" * 64


def test_profile_store_round_trips_and_rejects_corruption(tmp_path):
    store = RuntimeProfileStore(tmp_path)
    profile = RuntimeProfile.create(
        identity(),
        candidate(4),
        sample_count=512,
        measurements=({"candidate": candidate(4).to_dict(), "status": "ok"},),
        benchmark_job_sha256="d" * 64,
    )
    path = store.save(profile, model_slug="qwen3_reranker_0_6b_fp16")
    assert path.parent.parent == tmp_path
    assert path.name == "qwen3_reranker_0_6b_fp16.json"

    assert (
        store.load(profile.identity, model_slug="qwen3_reranker_0_6b_fp16") == profile
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected"]["concurrency"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.load(profile.identity, model_slug="qwen3_reranker_0_6b_fp16") is None


def test_search_space_rejects_empty_or_duplicate_candidates():
    with pytest.raises(ValueError):
        RuntimeSearchSpace(())
    with pytest.raises(ValueError):
        RuntimeSearchSpace((candidate(), candidate()))


def test_candidate_threads_are_optional_in_the_payload():
    assert "threads" not in candidate().to_dict()
    local = replace(candidate(), threads=12)

    assert local.to_dict()["threads"] == 12
    assert RuntimeCandidate.from_dict(local.to_dict()) == local
    assert RuntimeCandidate.from_dict(candidate().to_dict()) == candidate()


@pytest.mark.parametrize(
    ("server_slots", "request_batch_size", "expected"),
    [(64, 30, 4), (32, 30, 3), (16, 30, 2), (16, 15, 3)],
)
def test_rerank_concurrency_keeps_every_slot_busy(
    server_slots: int, request_batch_size: int, expected: int
):
    assert rerank_concurrency(server_slots, request_batch_size) == expected


def test_reranker_candidate_holds_one_prompt_per_slot_beside_the_batch():
    level = reranker_candidate(
        server_slots=8, ubatch=4096, request_batch_size=30, concurrency=2
    )

    assert (
        level.context_per_slot,
        level.logical_batch_size,
        level.physical_batch_size,
    ) == (2048, 4096, 4096)


def test_reranker_context_holds_one_batch_and_the_prompt_of_every_slot():
    assert (
        reranker_context_size(server_slots=8, ubatch=4096, context_per_slot=2048)
        == 20480
    )


def test_reranker_candidate_rejects_ubatch_below_the_longest_prompt():
    with pytest.raises(ValueError, match="2048"):
        reranker_candidate(
            server_slots=4, ubatch=1024, request_batch_size=30, concurrency=1
        )
