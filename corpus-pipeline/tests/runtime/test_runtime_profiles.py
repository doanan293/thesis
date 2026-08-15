import json

import pytest

from corpus_pipeline.runtime.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileIdentity,
    RuntimeProfileStore,
    RuntimeSearchSpace,
)


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
        machine_shape="NvidiaTeslaT4",
        topology="replicated_2x1",
        search_space=space,
    )


def test_profile_identity_excludes_input_but_changes_with_runtime():
    assert identity().sha256 == identity().sha256
    assert identity().sha256 != identity("c" * 64).sha256


def test_profile_store_round_trips_and_rejects_corruption(tmp_path):
    store = RuntimeProfileStore(tmp_path)
    profile = RuntimeProfile.create(
        identity(),
        candidate(4),
        sample_count=512,
        measurements=(
            {"candidate": candidate(4).to_dict(), "status": "ok"},
        ),
        benchmark_job_sha256="d" * 64,
    )
    path = store.save(profile, model_slug="qwen3_reranker_0_6b_fp16")

    assert store.load(
        profile.identity, model_slug="qwen3_reranker_0_6b_fp16"
    ) == profile

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected"]["concurrency"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.load(
        profile.identity, model_slug="qwen3_reranker_0_6b_fp16"
    ) is None


def test_search_space_rejects_empty_or_duplicate_candidates():
    with pytest.raises(ValueError):
        RuntimeSearchSpace(())
    with pytest.raises(ValueError):
        RuntimeSearchSpace((candidate(), candidate()))
