from __future__ import annotations

import hashlib

from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.runtime.model_profiles import (
    RerankContract,
    native_rerank_contract,
    qwen3_rerank_contract,
)


def document_hash(document_text: str) -> str:
    return hashlib.sha256(str(document_text).encode("utf-8")).hexdigest()


def prompt_contract_hash(
    *,
    protocol: str,
    instruction: str = "",
    template_version: str = "rerank-prompt-v1",
) -> str:
    if protocol == "completion_logprobs":
        contract = qwen3_rerank_contract()
        if instruction and instruction != contract.instruction:
            contract = RerankContract(
                protocol=contract.protocol,
                template_id=contract.template_id,
                template_version=template_version,
                instruction=instruction,
                scoring=contract.scoring,
            )
        return contract.sha256
    if protocol == "native_rerank":
        return native_rerank_contract().sha256
    return canonical_sha256(
        {
            "protocol": protocol,
            "instruction": instruction,
            "template_version": template_version,
        }
    )
