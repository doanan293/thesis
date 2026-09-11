from corpus_pipeline.evaluation.rerankers import build_qwen_rerank_prompt
from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.runtime.client import LlamaCppClient


def test_local_and_kaggle_qwen_build_the_same_prompt():
    contract = require_model("qwen3-reranker:0.6b-fp16").rerank_contract

    assert contract is not None
    assert contract.build_prompt("thuốc gì", "tài liệu") == build_qwen_rerank_prompt(
        "thuốc gì", "tài liệu"
    )


def test_completion_payload_is_derived_from_contract():
    contract = require_model("qwen3-reranker:0.6b-fp16").rerank_contract
    assert contract is not None and contract.scoring is not None
    client = LlamaCppClient("http://127.0.0.1:1")

    payload = client.completion_payload("prompt", contract, yes_id=10, no_id=11)

    assert payload["n_predict"] == contract.scoring.n_predict
    assert payload["samplers"] == list(contract.scoring.samplers)
    assert payload["logit_bias"] == [[10, 100.0], [11, 100.0]]


def test_native_rerank_contract_has_no_completion_prompt():
    contract = require_model("bge-reranker-v2-m3:f16").rerank_contract

    assert contract is not None
    assert contract.protocol == "native_rerank"
    assert contract.scoring is None
    assert contract.template_id is None
