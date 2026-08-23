import re
from types import SimpleNamespace

import pytest

from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointService
from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.runtime.catalog import MODEL_CATALOG


def _job(stage: StageName, model: str):
    return SimpleNamespace(
        stage=stage,
        model=model,
        identity=SimpleNamespace(reuse_sha256="12345678" + "a" * 56),
    )


def test_rerank_checkpoint_reference_respects_kaggle_slug_limit():
    service = CheckpointService(SimpleNamespace(), "owner")

    reference = service.reference(
        _job(StageName.RERANK, "qwen3-reranker:0.6b-fp16")
    )
    owner, slug = reference.split("/", 1)

    assert owner == "owner"
    assert slug == "re-eval-rerank-qwen3-reranker-12345678-checkpoint"
    assert 6 <= len(slug) <= 50
    assert re.fullmatch(r"[a-z0-9-]+", slug)


PRODUCTION_STAGES = tuple(
    stage for stage in StageName if not stage.value.endswith("-benchmark")
)


@pytest.mark.parametrize("stage", PRODUCTION_STAGES)
@pytest.mark.parametrize("model", tuple(MODEL_CATALOG))
def test_production_checkpoint_references_are_bounded_and_deterministic(
    stage, model
):
    service = CheckpointService(SimpleNamespace(), "owner")
    job = _job(stage, model)

    first = service.reference(job)
    second = service.reference(job)
    slug = first.split("/", 1)[1]

    assert first == second
    assert 6 <= len(slug) <= 50
    assert slug.startswith(f"re-eval-{stage.value}-")
    assert slug.endswith("-12345678-checkpoint")
    assert re.fullmatch(r"[a-z0-9-]+", slug)
