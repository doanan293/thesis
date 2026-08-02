from pathlib import Path

from corpus_pipeline.integrations.kaggle.config import OwnerConfiguration
from corpus_pipeline.integrations.kaggle.models import (
    Completion,
    JobIdentity,
    StageName,
    StageRequest,
)


def identity(parameters):
    return JobIdentity.create(
        stage=StageName.QUERY_EMBED,
        contract_version=1,
        model="qwen3-embedding:0.6b-fp16",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters=parameters,
    )


def test_job_identity_is_stable_for_same_payload():
    assert identity({"batch_size": 32}).sha256 == identity({"batch_size": 32}).sha256


def test_result_affecting_parameter_changes_identity():
    assert (
        identity({"protocol": "native_rerank"}).sha256
        != identity({"protocol": "completion_logprobs"}).sha256
    )


def test_completion_requires_consistent_counts():
    assert Completion(total=3, complete=2, missing=1).is_complete is False


def test_stage_request_operational_options_are_not_identity_inputs(tmp_path: Path):
    request = StageRequest(
        stage=StageName.QUERY_EMBED,
        model="qwen3-embedding:0.6b-fp16",
        input_path=tmp_path / "eval.jsonl",
        output_dir=tmp_path / "output",
        gguf_root=tmp_path / "gguf",
        owners=OwnerConfiguration("run", "runtime", "corpus", "checkpoint"),
        max_runs=1,
        total_budget_seconds=10,
    )
    same_identity = JobIdentity.create(
        stage=request.stage,
        contract_version=1,
        model=request.model,
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={"batch_size": 32},
    )
    assert same_identity.payload["stage"] == "query-embed"
