import json
import shutil
from pathlib import Path

import pytest

from corpus_pipeline.evaluation import metrics_service
from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError
from corpus_pipeline.evaluation.metrics_artifacts import (
    publish_metrics_artifact,
    select_rerank_variants,
)
from corpus_pipeline.evaluation.metrics_service import (
    MetricsRequest,
    load_and_validate_metric_inputs,
    run_metrics,
)
from corpus_pipeline.evaluation.rerank_artifacts import finalize_run_rerank_bundle
from corpus_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunWorkspace,
    load_run_record,
)
from corpus_pipeline.evaluation.variant_identity import (
    MetricsArtifactIdentity,
    RerankVariantIdentity,
)


def _split_artifact_root(complete_run: Path, tmp_path: Path) -> Path:
    heavy_root = tmp_path / "heavy-run"
    shutil.copytree(complete_run / "candidates", heavy_root / "candidates")
    return heavy_root


def _rewrite_run_json(run_root: Path, **updates) -> None:
    path = run_root / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metrics_rebases_missing_legacy_candidate_path_to_verified_heavy_bundle(
    complete_run: Path, tmp_path: Path
):
    heavy_root = _split_artifact_root(complete_run, tmp_path)
    _rewrite_run_json(
        complete_run,
        candidates_dir=str(tmp_path / "legacy" / "candidates"),
    )

    inputs = load_and_validate_metric_inputs(
        MetricsRequest(complete_run, top_k=1, artifact_root=heavy_root)
    )

    assert next(iter(inputs.candidates))["query_id"] == "query-1"


def test_metrics_rejects_legacy_candidate_fallback_when_snapshot_differs(
    complete_run: Path, tmp_path: Path
):
    heavy_root = _split_artifact_root(complete_run, tmp_path)
    snapshot = complete_run / "candidates-manifest.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["data_sha256"] = "0" * 64
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    _rewrite_run_json(
        complete_run,
        candidates_dir=str(tmp_path / "legacy" / "candidates"),
    )

    with pytest.raises(ArtifactContractError, match="candidate snapshot mismatch"):
        load_and_validate_metric_inputs(
            MetricsRequest(complete_run, top_k=1, artifact_root=heavy_root)
        )


def test_metrics_rebases_missing_legacy_evaluation_path_by_verified_hash(
    complete_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    heavy_root = _split_artifact_root(complete_run, tmp_path)
    current_evaluation_dir = tmp_path / "processed" / "evaluation"
    current_evaluation_dir.mkdir(parents=True)
    evaluation_path = Path(
        load_run_record(complete_run / "run.json").identity.evaluation_path
    )
    shutil.copy2(evaluation_path, current_evaluation_dir / evaluation_path.name)
    monkeypatch.setattr(
        metrics_service, "PROCESSED_EVALUATION_DIR", current_evaluation_dir
    )
    _rewrite_run_json(
        complete_run,
        identity={
            **json.loads((complete_run / "run.json").read_text(encoding="utf-8"))[
                "identity"
            ],
            "evaluation_path": str(tmp_path / "legacy" / evaluation_path.name),
        },
    )

    inputs = load_and_validate_metric_inputs(
        MetricsRequest(complete_run, top_k=1, artifact_root=heavy_root)
    )

    assert "query-1" in inputs.evaluation_rows


def test_metrics_rejects_legacy_evaluation_fallback_when_hash_differs(
    complete_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    heavy_root = _split_artifact_root(complete_run, tmp_path)
    current_evaluation_dir = tmp_path / "processed" / "evaluation"
    current_evaluation_dir.mkdir(parents=True)
    evaluation_path = Path(
        load_run_record(complete_run / "run.json").identity.evaluation_path
    )
    fallback = current_evaluation_dir / evaluation_path.name
    fallback.write_text("different evaluation\n", encoding="utf-8")
    monkeypatch.setattr(
        metrics_service, "PROCESSED_EVALUATION_DIR", current_evaluation_dir
    )
    _rewrite_run_json(
        complete_run,
        identity={
            **json.loads((complete_run / "run.json").read_text(encoding="utf-8"))[
                "identity"
            ],
            "evaluation_path": str(tmp_path / "legacy" / evaluation_path.name),
        },
    )

    with pytest.raises(ArtifactContractError, match="changed after retrieval"):
        load_and_validate_metric_inputs(
            MetricsRequest(complete_run, top_k=1, artifact_root=heavy_root)
        )


def test_select_variants_defaults_to_all_and_model_keeps_all_revisions():
    variants = {
        "a" * 64: RerankVariantRecord("model-a", "rerank/a/one"),
        "b" * 64: RerankVariantRecord("model-a", "rerank/a/two"),
        "c" * 64: RerankVariantRecord("model-b", "rerank/b/one"),
    }

    assert list(select_rerank_variants(variants)) == sorted(variants)
    assert list(select_rerank_variants(variants, model="model-a")) == [
        "a" * 64,
        "b" * 64,
    ]


def test_select_exact_variant_rejects_ambiguous_prefix():
    variants = {
        "abcd" + "0" * 60: RerankVariantRecord("model-a", "rerank/a/one"),
        "abcd" + "1" * 60: RerankVariantRecord("model-a", "rerank/a/two"),
    }

    with pytest.raises(ValueError, match="ambiguous"):
        select_rerank_variants(variants, variant="abcd")


def test_metrics_publishes_baseline_and_variant_without_overwriting(
    complete_run, candidate_bundle, complete_rerank_cache
):
    workspace = RunWorkspace(
        complete_run,
        load_run_record(complete_run / "run.json").identity,
    )
    variant = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256,
        "qwen3-reranker:0.6b-fp16",
    )
    finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=variant,
    )

    first = run_metrics(MetricsRequest(complete_run, top_k=1, window_size=3))
    second = run_metrics(MetricsRequest(complete_run, top_k=1, window_size=4))

    assert len(first.reranked) == 1
    assert first.baseline.artifact_dir != second.baseline.artifact_dir
    assert first.reranked[0].artifact_dir != second.reranked[0].artifact_dir
    assert first.baseline.report_path.is_file()
    assert first.reranked[0].report_path.is_file()
    assert first.baseline.report_path != first.reranked[0].report_path
    assert "rerank" in first.reranked[0].report_path.parts


def test_metrics_payload_and_markdown_use_separate_roots(tmp_path):
    heavy_root = tmp_path / "heavy" / "retrieval_eval" / "run-a"
    summary_root = tmp_path / "retrieval_eval" / "run-a"
    identity = MetricsArtifactIdentity.create(
        evaluation_sha256="evaluation",
        candidate_data_sha256="candidates",
        top_k=1,
        window_size=3,
    )

    result = publish_metrics_artifact(
        heavy_root,
        summary_root,
        identity,
        {"count": 0, "mrr": 0.0},
        {"eval_group": {}, "difficulty": {}},
        [],
    )

    assert result.results_path.is_relative_to(heavy_root)
    assert result.report_path.is_relative_to(summary_root)
    assert result.report_path.is_file()
    assert (result.artifact_dir / "manifest.json").is_file()
