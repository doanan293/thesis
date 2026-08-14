from corpus_pipeline.evaluation.variant_identity import (
    MetricsArtifactIdentity,
    RerankVariantIdentity,
)


def test_rerank_identity_changes_for_model_revision():
    first = RerankVariantIdentity.from_values(
        candidate_data_sha256="candidate",
        model="reranker",
        model_sha256="model-v1",
        protocol="native_rerank",
        request_contract_sha256="prompt",
    )
    second = RerankVariantIdentity.from_values(
        candidate_data_sha256="candidate",
        model="reranker",
        model_sha256="model-v2",
        protocol="native_rerank",
        request_contract_sha256="prompt",
    )

    assert first.sha256 != second.sha256
    assert first.payload["contract_version"] == 1


def test_metrics_identity_includes_parameters_and_rerank_variant():
    baseline = MetricsArtifactIdentity.create(
        evaluation_sha256="evaluation",
        candidate_data_sha256="candidate",
        top_k=30,
        window_size=3,
    )
    reranked = MetricsArtifactIdentity.create(
        evaluation_sha256="evaluation",
        candidate_data_sha256="candidate",
        top_k=30,
        window_size=3,
        rerank_variant_sha256="variant",
    )
    changed_cutoff = MetricsArtifactIdentity.create(
        evaluation_sha256="evaluation",
        candidate_data_sha256="candidate",
        top_k=10,
        window_size=3,
    )

    assert len({baseline.sha256, reranked.sha256, changed_cutoff.sha256}) == 3
