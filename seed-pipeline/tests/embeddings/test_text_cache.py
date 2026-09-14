import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.embeddings.service import (
    KaggleTextEmbeddingBackend,
    LocalTextEmbeddingBackend,
    TextEmbeddingRequest,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import (
    EmbeddingInput,
    TextEmbeddingError,
    embed_missing,
    read_embedding_inputs,
    text_sha256,
    write_embedding_inputs,
)
from seed_pipeline.integrations.kaggle import auto_profile
from seed_pipeline.integrations.kaggle import service as kaggle_service
from seed_pipeline.runtime.catalog import require_model

MODEL = "qwen3-embedding:4b-fp16"
DIMS = require_model(MODEL).vector_dimension or 0


def _input(text: str) -> EmbeddingInput:
    return EmbeddingInput(embedding_text_sha256=text_sha256(text), embedding_text=text)


def _vector(seed: float) -> list[float]:
    return [seed] * DIMS


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self, texts: list[str], model: str, expected_dimension: int
    ) -> list[list[float]]:
        assert model == MODEL
        assert expected_dimension == DIMS
        self.calls.append(list(texts))
        return [_vector(float(len(text))) for text in texts]


def test_inputs_file_is_unique_sorted_and_verified(tmp_path: Path) -> None:
    path = tmp_path / "embedding_inputs.jsonl"

    count = write_embedding_inputs([_input("b"), _input("a"), _input("b")], path)

    assert count == 2
    rows = read_embedding_inputs(path)
    assert [row.embedding_text_sha256 for row in rows] == sorted(
        [text_sha256("a"), text_sha256("b")]
    )
    tampered = json.loads(path.read_text("utf-8").splitlines()[0])
    tampered["embedding_text"] = "khác"
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(TextEmbeddingError, match="sha256 mismatch"):
        read_embedding_inputs(path)


def test_cache_persists_vectors_per_model(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = open_text_cache(path, MODEL)
    cache.set(text_sha256("a"), _vector(0.5))

    reloaded = open_text_cache(path, MODEL)

    assert reloaded.get(text_sha256("a")) == _vector(0.5)
    assert reloaded.missing([_input("a"), _input("b")]) == [_input("b")]
    assert len(reloaded.records_for([text_sha256("a")])) == 1
    with pytest.raises(TextEmbeddingError, match="dimension"):
        cache.set(text_sha256("b"), [0.1])


def test_records_for_reads_this_models_full_records_from_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.jsonl"
    other = "embeddinggemma:300m"
    open_text_cache(path, other).set(text_sha256("a"), [0.5] * 768)
    cache = open_text_cache(path, MODEL)
    cache.set(text_sha256("a"), _vector(0.25))

    (record,) = cache.records_for([text_sha256("b"), text_sha256("a")])

    assert (record["model"], record["embedding"]) == (MODEL, _vector(0.25))
    assert open_text_cache(path, other).get(text_sha256("a")) == [0.5] * 768


def test_embed_missing_only_embeds_new_texts(tmp_path: Path) -> None:
    cache = open_text_cache(tmp_path / "cache.jsonl", MODEL)
    cache.set(text_sha256("đã có"), _vector(1.0))
    client = RecordingClient()
    inputs = [_input("đã có"), _input("mới một"), _input("mới hai")]

    first = embed_missing(inputs, cache, [client], batch_size=1)
    second = embed_missing(inputs, cache, [client], batch_size=1)

    assert (first.total, first.cached, first.embedded, first.stopped_early) == (
        3,
        1,
        2,
        False,
    )
    assert sorted(text for call in client.calls for text in call) == [
        "mới hai",
        "mới một",
    ]
    assert (second.cached, second.embedded) == (3, 0)
    assert cache.get(text_sha256("mới một")) == _vector(float(len("mới một")))


def test_local_backend_does_not_start_a_server_when_cache_is_complete(
    tmp_path: Path,
) -> None:
    inputs_path = tmp_path / "embedding_inputs.jsonl"
    write_embedding_inputs([_input("a")], inputs_path)
    cache_path = tmp_path / "cache.jsonl"
    open_text_cache(cache_path, MODEL).set(text_sha256("a"), _vector(0.1))
    backend = LocalTextEmbeddingBackend(
        compose_file=tmp_path / "missing-compose.yml", gguf_root=tmp_path
    )

    result = backend.run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=MODEL,
            cache_path=cache_path,
            force=False,
            dry_run=False,
            budget_seconds=60,
            request_timeout_seconds=5.0,
        )
    )

    assert result == type(result)(cache_path, ("cached",), False)


def test_kaggle_backend_propagates_account_and_merges_remote_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs_path = tmp_path / "embedding_inputs.jsonl"
    write_embedding_inputs([_input("a")], inputs_path)
    remote_path = tmp_path / "remote" / "text_embeddings.jsonl"
    open_text_cache(remote_path, MODEL).set(text_sha256("a"), _vector(0.25))
    selected = require_model(MODEL).embedding_search_space
    assert selected is not None
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **kwargs: (
            seen.append(("profile", kwargs["kaggle_account"]))
            or SimpleNamespace(
                profile=SimpleNamespace(selected=selected.corpus.candidates[0]),
                action="reuse",
            )
        ),
    )
    monkeypatch.setattr(
        kaggle_service,
        "run_kaggle_stage",
        lambda **kwargs: (
            seen.append(("stage", kwargs["kaggle_account"]))
            or SimpleNamespace(
                artifact_path=remote_path, completion=Completion(1, 1, 0), actions=()
            )
        ),
    )
    cache_path = tmp_path / "cache.jsonl"

    result = KaggleTextEmbeddingBackend().run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=MODEL,
            cache_path=cache_path,
            force=False,
            dry_run=False,
            budget_seconds=60,
            request_timeout_seconds=5.0,
            kaggle_account="acc2",
        )
    )

    assert not result.incomplete
    assert seen == [("profile", "acc2"), ("stage", "acc2")]
    assert open_text_cache(cache_path, MODEL).get(text_sha256("a")) == _vector(0.25)
