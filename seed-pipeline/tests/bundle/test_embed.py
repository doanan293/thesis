from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleEmbeddingFile,
    KnowledgeBundle,
    model_slug,
    read_bundle,
)

from seed_pipeline.bundle.embed import (
    BundleEmbedRequest,
    collect_embedding_inputs,
    embed_bundle,
)
from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.embeddings.service import (
    TextEmbeddingRequest,
    TextEmbeddingResult,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import read_embedding_inputs, text_sha256
from seed_pipeline.runtime.catalog import require_model

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
MODEL = "qwen3-embedding:4b-fp16"
DIMS = require_model(MODEL).vector_dimension or 0


def fake_vector(text: str) -> list[float]:
    return [float(len(text) % 7)] * DIMS


class CacheFillingBackend:
    def __init__(self, *, incomplete: bool = False) -> None:
        self.incomplete = incomplete
        self.requests: list[TextEmbeddingRequest] = []

    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.requests.append(request)
        if self.incomplete:
            return TextEmbeddingResult(None, ("profile=pending",), incomplete=True)
        cache = open_text_cache(request.cache_path, request.model)
        for item in read_embedding_inputs(request.inputs_path):
            if cache.get(item.embedding_text_sha256) is None:
                cache.set(item.embedding_text_sha256, fake_vector(item.embedding_text))
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


def _request(tmp_path: Path, bundle_dir: Path) -> BundleEmbedRequest:
    return BundleEmbedRequest(
        bundle_dir=bundle_dir,
        model=MODEL,
        cache_path=tmp_path / "cache.jsonl",
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
    for item in collect_embedding_inputs(reloaded):
        assert reloaded.embeddings[MODEL][item.embedding_text_sha256] == fake_vector(
            item.embedding_text
        )
    assert backend.requests[0].inputs_path == (
        tmp_path / "work" / require_model(MODEL).slug / "embedding_inputs.jsonl"
    )


def test_incomplete_embedding_leaves_the_bundle_untouched(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir(tmp_path)

    result = embed_bundle(
        _request(tmp_path, bundle_dir), CacheFillingBackend(incomplete=True)
    )

    assert result.incomplete
    assert result.manifest is None
    assert not (bundle_dir / "embeddings").exists()
    assert read_bundle(bundle_dir).embeddings == {}


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
    assert {item.embedding_text_sha256 for item in inputs} <= set(
        bundle.embeddings["fake-embedding-4d"]
    )
