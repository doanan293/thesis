from collections.abc import Mapping
from pathlib import Path

from pharma_lab.integrations.kaggle.artifacts import load_cloud_artifact
from pharma_lab.integrations.kaggle.models import (
    JobIdentity,
    StageName,
    reuse_payload_of,
)
from pharma_lab.integrations.kaggle.workers.runtime import artifact_from_output
from pharma_lab.runtime.catalog import require_model
from pharma_lab.runtime.runtime_profiles import RuntimeCandidate

MODEL = "qwen3-reranker:0.6b-fp16"


def _levels() -> tuple[RuntimeCandidate, RuntimeCandidate]:
    space = require_model(MODEL).rerank_search_space
    assert space is not None
    return space.candidates[0], space.candidates[1]


def _identity(
    profile: RuntimeCandidate,
    *,
    contract: str = "c" * 64,
    input_sha256: str = "b" * 64,
) -> JobIdentity:
    return JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=3,
        model=MODEL,
        model_sha256="a" * 64,
        input_sha256=input_sha256,
        runtime_parameters={
            "protocol": "native_rerank",
            "request_contract_sha256": contract,
            "runtime_profile": profile.to_dict(),
        },
    )


def test_runtime_profile_changes_the_job_but_not_the_reuse_identity():
    first, second = _levels()
    one, other = _identity(first), _identity(second)

    assert one.sha256 != other.sha256
    assert one.reuse_sha256 == other.reuse_sha256
    parameters = one.reuse_payload["runtime_parameters"]
    assert isinstance(parameters, Mapping)
    assert "runtime_profile" not in parameters
    assert "input_sha256" not in one.reuse_payload


def test_reuse_identity_still_tracks_the_scoring_contract():
    first, _second = _levels()
    base = _identity(first)

    assert _identity(first, contract="d" * 64).reuse_sha256 != base.reuse_sha256
    assert _identity(first, input_sha256="e" * 64).reuse_sha256 == base.reuse_sha256


def test_reuse_payload_without_runtime_parameters_only_drops_the_input():
    assert reuse_payload_of({"stage": "rerank", "input_sha256": "b" * 64}) == {
        "stage": "rerank"
    }


def test_artifact_from_another_runtime_profile_is_reusable(tmp_path: Path):
    first, second = _levels()
    data = tmp_path / "rerank_scores.jsonl"
    data.write_text('{"score": 0.5}\n', encoding="utf-8")
    artifact = artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=_identity(first),
        total=1,
        complete=1,
    )

    loaded = load_cloud_artifact(
        data, artifact.manifest_path, _identity(second), allow_reuse=True
    )

    assert loaded.strict_identity_match is False
    assert loaded.completion.complete == 1
