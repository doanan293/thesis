from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corpus_pipeline.evaluation.artifact_contracts import canonical_sha256
from corpus_pipeline.evaluation.rerank_score_cache import prompt_contract_hash
from corpus_pipeline.runtime.catalog import require_model

RERANK_VARIANT_CONTRACT_VERSION = 1
METRICS_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class RerankVariantIdentity:
    payload: dict[str, Any]
    sha256: str

    @classmethod
    def from_values(
        cls,
        *,
        candidate_data_sha256: str,
        model: str,
        model_sha256: str,
        protocol: str,
        request_contract_sha256: str,
    ) -> RerankVariantIdentity:
        payload = {
            "contract_version": RERANK_VARIANT_CONTRACT_VERSION,
            "candidate_data_sha256": str(candidate_data_sha256),
            "reranker": str(model),
            "model_sha256": str(model_sha256),
            "protocol": str(protocol),
            "request_contract_sha256": str(request_contract_sha256),
        }
        return cls(payload, canonical_sha256(payload))

    @classmethod
    def create(cls, candidate_data_sha256: str, model: str) -> RerankVariantIdentity:
        spec = require_model(model)
        protocol = spec.reranker_protocol or ""
        return cls.from_values(
            candidate_data_sha256=candidate_data_sha256,
            model=model,
            model_sha256=spec.sha256,
            protocol=protocol,
            request_contract_sha256=prompt_contract_hash(protocol=protocol),
        )


@dataclass(frozen=True)
class MetricsArtifactIdentity:
    payload: dict[str, Any]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        evaluation_sha256: str,
        candidate_data_sha256: str,
        top_k: int,
        window_size: int,
        rerank_variant_sha256: str | None = None,
    ) -> MetricsArtifactIdentity:
        payload: dict[str, Any] = {
            "contract_version": METRICS_CONTRACT_VERSION,
            "evaluation_sha256": str(evaluation_sha256),
            "candidate_data_sha256": str(candidate_data_sha256),
            "top_k": int(top_k),
            "window_size": int(window_size),
        }
        if rerank_variant_sha256 is not None:
            payload["rerank_variant_sha256"] = str(rerank_variant_sha256)
        return cls(payload, canonical_sha256(payload))
