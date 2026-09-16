from pathlib import Path

import pytest
from pharma_agent.domain.corpus.bundle import (
    BundleEmbeddingFile,
    KnowledgeBundle,
    model_slug,
    read_bundle,
    read_bundle_embeddings,
)

from pharma_lab.bundle.embed import (
    BundleEmbedRequest,
    collect_embedding_inputs,
    embed_bundle,
)
from pharma_lab.bundle.export import ExportRequest, export_bundle
from pharma_lab.embeddings.service import (
    TextEmbeddingRequest,
    TextEmbeddingResult,
    open_text_cache,
)
from pharma_lab.embeddings.text_cache import (
    TextEmbeddingError,
    read_embedding_inputs,
    text_sha256,
)
from pharma_lab.runtime.catalog import require_model

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
MODEL = "qwen3-embedding:4b-fp16"
OTHER_MODEL = "embeddinggemma:300m"
DIMS = require_model(MODEL).vector_dimension or 0


def fake_vector(text: str, dims: int = DIMS) -> list[float]:
    return [float(len(text) % 7)] * dims


class CacheFillingBackend:
    def __init__(self, *, incomplete: bool = False, skip: int = 0) -> None:
        self.incomplete = incomplete
        self.skip = skip
        self.requests: list[TextEmbeddingRequest] = []

    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.requests.append(request)
        if self.incomplete:
            return TextEmbeddingResult(None, ("profile=pending",), incomplete=True)
        dims = require_model(request.model).vector_dimension or 0
        cache = open_text_cache(request.cache_path, request.model)
        for item in read_embedding_inputs(request.inputs_path)[self.skip :]:
            if cache.get(item.embedding_text_sha256) is None:
                cache.set(
                    item.embedding_text_sha256, fake_vector(item.embedding_text, dims)
                )
        return TextEmbeddingResult(request.cache_path, ("fake",))


def _bundle_dir(tmp_path: Path) -> Path:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return tmp_path / "bundle"


def _request(
    tmp_path: Path, bundle_dir: Path, model: str = MODEL
) -> BundleEmbedRequest:
    return BundleEmbedRequest(
        bundle_dir=bundle_dir,
        model=model,
        cache_path=tmp_path / f"{model_slug(model)}.jsonl",
        work_dir=tmp_path / "work",
    )


def test_collect_embedding_inputs_is_unique_and_hashed(tmp_path: Path) -> None:
    bundle: KnowledgeBundle = read_bundle(_bundle_dir(tmp_path))

    inputs = collect_embedding_inputs(bundle)

    assert len(inputs) == 6
    assert [item.embedding_text_sha256 for item in inputs] == sorted(
        item.embedding_text_sha256 for item in inputs
    )
    assert all(
        item.embedding_text_sha256 == text_sha256(item.embedding_text)
        for item in inputs
    )


def test_embed_bundle_writes_embeddings_file_and_manifest(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    backend = CacheFillingBackend()

    result = embed_bundle(_request(tmp_path, bundle_dir), backend)

    assert not result.incomplete
    assert (result.inputs, result.vectors) == (6, 6)
    expected_file = f"embeddings/{model_slug(MODEL)}.jsonl"
    assert result.embeddings_file == expected_file
    assert (bundle_dir / expected_file).is_file()
    assert result.manifest is not None
    assert result.manifest.embeddings == [
        BundleEmbeddingFile(model=MODEL, dims=DIMS, file=expected_file)
    ]
    reloaded = read_bundle(bundle_dir)
    assert reloaded.manifest == result.manifest
    vectors = read_bundle_embeddings(bundle_dir, reloaded.manifest, MODEL, DIMS)
    assert vectors == {
        item.embedding_text_sha256: fake_vector(item.embedding_text)
        for item in collect_embedding_inputs(reloaded)
    }
    assert backend.requests[0].inputs_path == (
        tmp_path / "work" / require_model(MODEL).slug / "embedding_inputs.jsonl"
    )


def test_embedding_another_model_leaves_existing_embedding_files_untouched(
    tmp_path: Path,
) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    embed_bundle(_request(tmp_path, bundle_dir), CacheFillingBackend())
    first_file = bundle_dir / f"embeddings/{model_slug(MODEL)}.jsonl"
    before = (first_file.read_bytes(), first_file.stat().st_mtime_ns)

    result = embed_bundle(
        _request(tmp_path, bundle_dir, OTHER_MODEL), CacheFillingBackend()
    )

    assert result.manifest is not None
    assert [entry.model for entry in result.manifest.embeddings] == [
        OTHER_MODEL,
        MODEL,
    ]
    assert (first_file.read_bytes(), first_file.stat().st_mtime_ns) == before
    assert read_bundle(bundle_dir).manifest == result.manifest


def test_incomplete_embedding_leaves_the_bundle_untouched(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir(tmp_path)

    result = embed_bundle(
        _request(tmp_path, bundle_dir), CacheFillingBackend(incomplete=True)
    )

    assert result.incomplete
    assert result.manifest is None
    assert not (bundle_dir / "embeddings").exists()
    assert read_bundle(bundle_dir).manifest.embeddings == []


def test_a_vector_missing_from_the_cache_fails_without_touching_the_bundle(
    tmp_path: Path,
) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    before = {path.name: path.read_bytes() for path in bundle_dir.iterdir()}

    with pytest.raises(TextEmbeddingError, match="is missing"):
        embed_bundle(_request(tmp_path, bundle_dir), CacheFillingBackend(skip=1))

    assert {path.name: path.read_bytes() for path in bundle_dir.iterdir()} == before


BACKEND_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "tests"
    / "fixtures"
    / "knowledge_bundle_small"
)


def test_shared_backend_fixture_bundle_is_read_and_fully_embedded() -> None:
    bundle = read_bundle(BACKEND_FIXTURE)

    inputs = collect_embedding_inputs(bundle)

    assert inputs
    vectors = read_bundle_embeddings(
        BACKEND_FIXTURE, bundle.manifest, "fake-embedding-4d", 4
    )
    assert {item.embedding_text_sha256 for item in inputs} <= set(vectors)
