import pytest

from pharma_lab.runtime.catalog import MODEL_CATALOG, RERANKER_MODELS, require_model
from pharma_lab.runtime.model_profiles import native_rerank_contract

# Every bge-reranker-v2-m3 score in data/cache/rerank_scores was sealed with this digest.
SCORED_NATIVE_CONTRACT_SHA256 = (
    "95b81f733a6695906ec4b9c0a30ab9588dc1f43bd9e64e45800101c87ee0eb48"
)


@pytest.mark.parametrize("model", sorted(RERANKER_MODELS))
def test_every_reranker_is_called_through_v1_rerank(model: str) -> None:
    spec = require_model(model)

    assert spec.reranker_protocol == "native_rerank"
    assert spec.rerank_contract == native_rerank_contract()


def test_native_contract_digest_matches_existing_scores() -> None:
    assert native_rerank_contract().sha256 == SCORED_NATIVE_CONTRACT_SHA256


def test_completion_reranker_is_not_in_the_catalog() -> None:
    assert "bge-reranker-v2-gemma:f16" not in MODEL_CATALOG
