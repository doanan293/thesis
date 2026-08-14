import pytest

from corpus_pipeline.evaluation.metrics_artifacts import select_rerank_variants
from corpus_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics
from corpus_pipeline.evaluation.rerank_artifacts import finalize_run_rerank_bundle
from corpus_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunWorkspace,
    load_run_record,
)
from corpus_pipeline.evaluation.variant_identity import RerankVariantIdentity


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
