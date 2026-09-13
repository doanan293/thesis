"""Qdrant for integration tests: one container per session, collections wiped per test."""

from collections.abc import AsyncGenerator, Iterator

import pytest
from qdrant_client import AsyncQdrantClient

# Same server version as docker-compose.yml; BM25 inference from models.Document needs >= 1.15.3.
QDRANT_IMAGE = "qdrant/qdrant:v1.19.1"


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    from testcontainers.community.qdrant import QdrantContainer

    with QdrantContainer(QDRANT_IMAGE) as container:
        host = container.get_container_host_ip()
        yield f"http://{host}:{container.get_exposed_port(6333)}"


@pytest.fixture
async def qdrant_client(qdrant_url: str) -> AsyncGenerator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=qdrant_url)
    for collection in (await client.get_collections()).collections:
        await client.delete_collection(collection.name)
    yield client
    await client.close()
