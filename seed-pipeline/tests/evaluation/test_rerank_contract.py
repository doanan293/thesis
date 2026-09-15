import pytest

from seed_pipeline.evaluation.rerankers import build_qwen_rerank_prompt
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.model_profiles import qwen3_rerank_contract


def test_local_and_kaggle_qwen_build_the_same_prompt():
    contract = qwen3_rerank_contract()

    assert contract.build_prompt("thuốc gì", "tài liệu") == build_qwen_rerank_prompt(
        "thuốc gì", "tài liệu"
    )


def test_completion_payload_is_derived_from_contract():
    contract = qwen3_rerank_contract()
    assert contract.scoring is not None
    client = LlamaCppClient("http://127.0.0.1:1")

    payload = client.completion_payload("prompt", contract, yes_id=10, no_id=11)

    assert payload["n_predict"] == contract.scoring.n_predict
    assert payload["samplers"] == list(contract.scoring.samplers)
    assert payload["logit_bias"] == [[10, 100.0], [11, 100.0]]


@pytest.mark.parametrize(
    "model",
    (
        "bge-reranker-v2-m3:f16",
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker:4b-fp16",
        "qwen3-reranker:8b-fp16",
    ),
)
def test_native_rerank_contract_has_no_completion_prompt(model):
    contract = require_model(model).rerank_contract

    assert contract is not None
    assert contract.protocol == "native_rerank"
    assert contract.scoring is None
    assert contract.template_id is None
