from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.evaluation import retrieval_service
from corpus_pipeline.evaluation.retrieval_service import RetrieveRequest
from corpus_pipeline.evaluation.retrievers import QdrantBm25Retriever


class AliasOnlyQdrantClient:
    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name="versioned-physical-collection")]
        )

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name="thesis_chunks_embeddinggemma_300m")
            ]
        )


def test_build_retriever_accepts_alias_backed_collection(monkeypatch):
    client = AliasOnlyQdrantClient()
    monkeypatch.setattr(retrieval_service, "_qdrant_client", lambda _url: client)
    request = RetrieveRequest(
        evaluation_path=Path("evaluation.jsonl"),
        query_embeddings_dir=None,
        run_root=Path("run"),
        embedding_model="embeddinggemma:300m",
        qdrant_url="http://localhost:6333",
        retriever="bm25",
        candidate_k=30,
        rrf_k=60,
        limit=None,
        force=False,
    )

    retriever = retrieval_service.build_retriever_for_request(
        request, rows=[], query_cache=None
    )

    assert isinstance(retriever, QdrantBm25Retriever)
    assert retriever.collection_name == "thesis_chunks_embeddinggemma_300m"
