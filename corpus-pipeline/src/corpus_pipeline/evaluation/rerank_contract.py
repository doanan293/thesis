from __future__ import annotations

import hashlib

from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256


def document_hash(document_text: str) -> str:
    return hashlib.sha256(str(document_text).encode("utf-8")).hexdigest()


def prompt_contract_hash(
    *,
    protocol: str,
    instruction: str = "",
    template_version: str = "rerank-prompt-v1",
) -> str:
    return canonical_sha256(
        {
            "protocol": protocol,
            "instruction": instruction,
            "template_version": template_version,
        }
    )
