from types import SimpleNamespace

from corpus_pipeline.evaluation.retrievers import QdrantHybridRetriever


class CapturingQdrantClient:
    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(points=[])


def test_hybrid_retriever_forwards_explicit_rrf_k():
    client = CapturingQdrantClient()
    retriever = QdrantHybridRetriever(
        client,
        "collection",
        embed_query=lambda _query: [0.1, 0.2],
        rrf_k=2,
        prefetch_k=50,
    )

    assert retriever.search("query", limit=30) == []
    assert client.query_kwargs["query"].rrf.k == 2
    assert client.query_kwargs["query"].rrf.weights is None
