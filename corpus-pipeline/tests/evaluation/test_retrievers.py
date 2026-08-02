from types import SimpleNamespace

from corpus_pipeline.evaluation.retrievers import DenseQdrantRetriever


class FakeClient:
    def __init__(self):
        self.calls = 0

    def query_batch_points(self, *, collection_name, requests):
        self.calls += 1
        return [
            SimpleNamespace(
                points=[SimpleNamespace(payload={"chunk_id": str(i)}, score=1.0)]
            )
            for i, _ in enumerate(requests)
        ]


def test_dense_retriever_batches_qdrant_requests():
    client = FakeClient()
    retriever = DenseQdrantRetriever(client, "chunks", lambda row: [float(row["n"])])
    result = retriever.search_batch([{"n": 1}, {"n": 2}, {"n": 3}], 5)
    assert client.calls == 1
    assert [items[0].resolved_chunk_id for items in result] == ["0", "1", "2"]
