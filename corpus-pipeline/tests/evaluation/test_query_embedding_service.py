from types import SimpleNamespace

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.evaluation.query_embedding_service import (
    KaggleQueryEmbeddingBackend,
    QueryEmbeddingRequest,
)
from corpus_pipeline.integrations.kaggle import auto_profile
from corpus_pipeline.integrations.kaggle import service as kaggle_service
from corpus_pipeline.runtime.catalog import require_model

MODEL = "qwen3-embedding:4b-fp16"


def test_kaggle_query_embedding_propagates_selected_account(tmp_path, monkeypatch):
    seen = []
    search_space = require_model(MODEL).embedding_search_space
    assert search_space is not None
    selected = search_space.query.candidates[0]
    input_path = tmp_path / "evaluation.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **kwargs: (
            seen.append(("profile", kwargs["kaggle_account"]))
            or SimpleNamespace(
                profile=SimpleNamespace(selected=selected), action="reuse"
            )
        ),
    )
    monkeypatch.setattr(
        kaggle_service,
        "run_kaggle_stage",
        lambda **kwargs: (
            seen.append(("stage", kwargs["kaggle_account"]))
            or SimpleNamespace(
                artifact_path=None,
                completion=Completion(1, 0, 1),
                actions=(),
            )
        ),
    )

    result = KaggleQueryEmbeddingBackend().run(
        QueryEmbeddingRequest(
            evaluation_path=input_path,
            model=MODEL,
            output_dir=tmp_path / "cache.jsonl",
            force=False,
            dry_run=False,
            budget_seconds=60,
            request_timeout_seconds=5.0,
            kaggle_account="acc2",
        )
    )

    assert result.incomplete
    assert seen == [("profile", "acc2"), ("stage", "acc2")]
