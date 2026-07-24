from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from rag_metadata.qdrant_payload_contract import (
    QDRANT_INTEGER_INDEX_FIELDS,
    QDRANT_RUNTIME_INDEX_FIELDS,
)

DENSE_VECTOR_NAME = "dense_vector"
BM25_SPARSE_VECTOR_NAME = "bm25_sparse_vector"
BM25_MODEL_NAME = "Qdrant/bm25"


PAYLOAD_INDEXES = {
    field_name: PayloadSchemaType.INTEGER
    if field_name in QDRANT_INTEGER_INDEX_FIELDS
    else PayloadSchemaType.KEYWORD
    for field_name in QDRANT_RUNTIME_INDEX_FIELDS
}


class QdrantClientHelper:
    def __init__(
        self,
        host="localhost",
        port=6333,
        collection_name="thesis_chunks",
        vector_size=1024,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size

    def collection_exists(self) -> bool:
        collections = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in collections

    def init_collection(self, vector_size=None):
        size = vector_size or self.vector_size
        if not self.collection_exists():
            # 1. Create collection with Cosine similarity
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=size, distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    BM25_SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
                },
            )

            # 2. Create payload indexes for retained runtime lookup/filter fields.
            for field_name, field_schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.init_collection(vector_size=self.vector_size)

    def insert_chunk(self, chunk_id: int, text: str, vector: list, payload: dict):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector={DENSE_VECTOR_NAME: vector},
                    payload=payload,
                )
            ],
        )

    def insert_chunks_batch(self, points):
        self.client.upsert(collection_name=self.collection_name, points=points)
