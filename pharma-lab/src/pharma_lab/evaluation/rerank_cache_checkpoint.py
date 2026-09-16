"""The local rerank score cache as a checkpoint source for Kaggle sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharma_lab.artifacts.manifest import Completion
from pharma_lab.evaluation.rerank_score_cache import RerankScoreCache
from pharma_lab.integrations.kaggle.checkpoints import CheckpointState
from pharma_lab.integrations.kaggle.job_lock import kaggle_cache_lock
from pharma_lab.integrations.kaggle.models import StageJob
from pharma_lab.integrations.kaggle.workers.rerank import write_rerank_checkpoint
from pharma_lab.runtime.catalog import require_model


@dataclass(frozen=True)
class RerankCacheCheckpoint:
    """Offers the scores in `data/cache/rerank_scores/<model>.jsonl` as a checkpoint.

    The cache is the source of truth: it survives account switches, runtime profile
    changes and job identity changes, so it can seed every session.
    """

    cache_path: Path

    def inspect(self, job: StageJob, *, download_root: Path) -> CheckpointState:
        spec = require_model(job.model)
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {job.model} has no scoring contract")
        candidates = job.input_bundle.file("candidates").source_path
        with kaggle_cache_lock(self.cache_path):
            records = RerankScoreCache(
                self.cache_path,
                model_sha256=spec.sha256,
                request_contract_sha256=spec.rerank_contract.sha256,
            ).available_records(candidates, job.model)
        if not records:
            empty = Completion(job.expected_total, 0, job.expected_total)
            return CheckpointState(None, None, empty)
        artifact = write_rerank_checkpoint(
            records,
            identity=job.identity,
            total=job.expected_total,
            output_dir=download_root,
        )
        return CheckpointState(None, artifact, artifact.completion)
