from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pharma_lab.evaluation.artifact_contracts import canonical_sha256
from pharma_lab.runtime.catalog import require_model

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
            "candidate_data_sha256": candidate_data_sha256,
            "reranker": model,
            "model_sha256": model_sha256,
            "protocol": protocol,
            "request_contract_sha256": request_contract_sha256,
        }
        return cls(payload, canonical_sha256(payload))

    @classmethod
    def create(cls, candidate_data_sha256: str, model: str) -> RerankVariantIdentity:
        spec = require_model(model)
        protocol = spec.reranker_protocol or ""
        contract = spec.rerank_contract
        if contract is None:
            raise ValueError(f"Reranker {model} has no scoring contract")
        return cls.from_values(
            candidate_data_sha256=candidate_data_sha256,
            model=model,
            model_sha256=spec.sha256,
            protocol=protocol,
            request_contract_sha256=contract.sha256,
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
        judgments_sha256: str | None = None,
    ) -> MetricsArtifactIdentity:
        payload: dict[str, Any] = {
            "contract_version": METRICS_CONTRACT_VERSION,
            "evaluation_sha256": evaluation_sha256,
            "candidate_data_sha256": candidate_data_sha256,
            "top_k": top_k,
            "window_size": window_size,
        }
        if rerank_variant_sha256 is not None:
            payload["rerank_variant_sha256"] = rerank_variant_sha256
        if judgments_sha256 is not None:
            payload["judgments_sha256"] = judgments_sha256
        return cls(payload, canonical_sha256(payload))
