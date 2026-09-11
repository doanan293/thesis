from types import SimpleNamespace
from unittest.mock import create_autospec

from qdrant_client import QdrantClient

from corpus_pipeline.evaluation.retrievers import QdrantHybridRetriever


def test_hybrid_retriever_forwards_explicit_rrf_k():
    client = create_autospec(QdrantClient, instance=True)
    client.query_points.return_value = SimpleNamespace(points=[])
    retriever = QdrantHybridRetriever(
        client,
        "collection",
        embed_query=lambda _query: [0.1, 0.2],
        rrf_k=2,
        prefetch_k=50,
    )

    assert retriever.search("query", limit=30) == []
    query = client.query_points.call_args.kwargs["query"]
    assert query.rrf.k == 2
    assert query.rrf.weights is None
