import re
from pathlib import Path

import pytest

from seed_pipeline.artifacts.bundle import ArtifactBundle
from seed_pipeline.evaluation.metrics_artifacts import (
    MetricsArtifactResult,
    publish_metrics_artifact,
    report_dir,
)
from seed_pipeline.evaluation.metrics_service import (
    MetricsRequest,
    run_metrics,
    select_rerank_variants,
)
from seed_pipeline.evaluation.rerank_artifacts import finalize_run_rerank_bundle
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunConflictError,
    RunWorkspace,
    load_run_record,
)
from seed_pipeline.evaluation.variant_identity import (
    MetricsArtifactIdentity,
    RerankVariantIdentity,
)

MODEL = "qwen3-reranker:0.6b-fp16"
SLUG = "qwen3_reranker_0_6b_fp16"


def register_variant(
    run: Path, candidate_bundle: ArtifactBundle, cache: RerankScoreCache
) -> None:
    record = load_run_record(run / "run.json")
    finalize_run_rerank_bundle(
        workspace=RunWorkspace(run, record.identity, record.origin),
        candidate_bundle=candidate_bundle,
        cache_path=cache.path,
        identity=RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, MODEL
        ),
    )


def metrics_identity(evaluation_sha256: str) -> MetricsArtifactIdentity:
    return MetricsArtifactIdentity.create(
        evaluation_sha256=evaluation_sha256,
        candidate_data_sha256="candidates",
        top_k=1,
        window_size=3,
    )


def publish_empty(
    root: Path, identity: MetricsArtifactIdentity, *, force: bool = False
) -> MetricsArtifactResult:
    return publish_metrics_artifact(
        root,
        identity,
        {"count": 0, "mrr": 0.0},
        {"eval_group": {}, "difficulty": {}},
        [],
        top_k=1,
        window_size=3,
        force=force,
    )


def test_report_directories_are_named_after_cutoff_and_model(tmp_path: Path) -> None:
    assert report_dir(tmp_path, top_k=30, window_size=3) == (
        tmp_path / "reports" / "baseline" / "top30-window3"
    )
    assert report_dir(tmp_path, top_k=10, window_size=3, model=MODEL) == (
        tmp_path / "reports" / "rerank" / SLUG / "top10-window3"
    )


def test_metrics_publish_baseline_and_rerank_reports_in_the_run(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    register_variant(complete_run, candidate_bundle, complete_rerank_cache)

    result = run_metrics(MetricsRequest(complete_run, top_k=1))

    assert result.baseline.artifact_dir == report_dir(
        complete_run, top_k=1, window_size=3
    )
    assert [item.artifact_dir for item in result.reranked] == [
        report_dir(complete_run, top_k=1, window_size=3, model=MODEL)
    ]
    for artifact in (result.baseline, *result.reranked):
        assert {path.name for path in artifact.artifact_dir.iterdir()} == {
            "manifest.json",
            "metrics.jsonl",
            "report.md",
        }
    assert (
        run_metrics(MetricsRequest(complete_run, top_k=1)).baseline == result.baseline
    )
    produced = [
        path.relative_to(complete_run).as_posix() for path in complete_run.rglob("*")
    ]
    assert [path for path in produced if re.search(r"[0-9a-f]{12,}", path)] == []


def test_a_report_for_different_inputs_conflicts_unless_forced(tmp_path: Path) -> None:
    first = publish_empty(tmp_path, metrics_identity("a"))

    with pytest.raises(RunConflictError, match="top1-window3"):
        publish_empty(tmp_path, metrics_identity("b"))
    replaced = publish_empty(tmp_path, metrics_identity("b"), force=True)

    assert replaced.artifact_dir == first.artifact_dir
    assert replaced.metrics_sha256 == metrics_identity("b").sha256


def test_select_variants_returns_all_or_the_named_model() -> None:
    qwen = RerankVariantRecord(MODEL, "a" * 64, f"rerank/{SLUG}")
    bge = RerankVariantRecord(
        "bge-reranker-v2-m3:f16", "b" * 64, "rerank/bge_reranker_v2_m3_f16"
    )
    variants = {SLUG: qwen, "bge_reranker_v2_m3_f16": bge}

    assert list(select_rerank_variants(variants, model=None)) == [
        "bge_reranker_v2_m3_f16",
        SLUG,
    ]
    assert select_rerank_variants(variants, model=MODEL) == {SLUG: qwen}
    with pytest.raises(ValueError, match="available: qwen3-reranker"):
        select_rerank_variants({SLUG: qwen}, model="bge-reranker-v2-m3:f16")


def test_metrics_leave_reports_of_unregistered_rerankers_alone(complete_run: Path):
    report = (
        complete_run
        / "reports"
        / "rerank"
        / "bge_reranker_v2_gemma_f16"
        / "top1-window3"
        / "report.md"
    )
    report.parent.mkdir(parents=True)
    report.write_text("final Gemma numbers\n", encoding="utf-8")

    result = run_metrics(MetricsRequest(complete_run, top_k=1))

    assert result.reranked == ()
    assert report.read_text(encoding="utf-8") == "final Gemma numbers\n"
