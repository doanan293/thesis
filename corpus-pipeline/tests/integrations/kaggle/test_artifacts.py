import json
from pathlib import Path

from corpus_pipeline.integrations.kaggle.artifacts import (
    load_cloud_artifact,
    promote_complete_artifact,
    sha256_file,
)
from corpus_pipeline.integrations.kaggle.models import JobIdentity, StageName
from corpus_pipeline.integrations.kaggle.workers.runtime import artifact_from_output


def _identity() -> JobIdentity:
    return JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=3,
        model="qwen3-reranker:0.6b-fp16",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        runtime_parameters={"request_contract_sha256": "c" * 64},
    )


def test_load_and_promote_known_diagnostics_without_validating_contents(tmp_path: Path):
    data = tmp_path / "scores.jsonl"
    manifest = tmp_path / "manifest.json"
    telemetry = tmp_path / "telemetry.json"
    report = tmp_path / "benchmark_report.md"
    log = tmp_path / "server-0.log"
    data.write_text('{"score": 0.5}\n', encoding="utf-8")
    telemetry.write_text("not-json-but-diagnostic\n", encoding="utf-8")
    report.write_text("# report\n", encoding="utf-8")
    log.write_text("server diagnostics\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "rerank_scores",
                "created_at": "now",
                "data_filename": data.name,
                "data_sha256": sha256_file(data),
                "record_count": 1,
                "identity": {"job_sha256": _identity().sha256},
                "total": 1,
                "complete": 1,
                "missing": 0,
            }
        ),
        encoding="utf-8",
    )

    artifact = load_cloud_artifact(data, manifest, _identity())
    destination = tmp_path / "promoted" / "scores.jsonl"
    promote_complete_artifact(artifact, destination)

    assert {path.name for path in artifact.diagnostic_paths} == {
        "telemetry.json",
        "benchmark_report.md",
        "server-0.log",
    }
    assert (destination.parent / "telemetry.json").is_file()
    assert (destination.parent / "benchmark_report.md").is_file()
    assert (destination.parent / "server-0.log").is_file()


def test_artifact_manifest_can_store_runtime_summary(tmp_path: Path):
    data = tmp_path / "scores.jsonl"
    data.write_text('{"score": 0.5}\n', encoding="utf-8")

    artifact = artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=_identity(),
        total=1,
        complete=1,
        runtime_summary={"gpu_sampling_status": "unavailable"},
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"] == {"gpu_sampling_status": "unavailable"}
