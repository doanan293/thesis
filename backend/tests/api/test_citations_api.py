from pharma_agent.domain.conversation.models import CitationBlock, CitedChunk
from pharma_agent.domain.retrieval.models import HydrateStrategy
from tests.api.harness import OWNER, build_harness
from tests.citations import build_citation, chunk_id

MESSAGE_ID = "d" * 32


async def test_citation_detail_endpoint() -> None:
    harness = build_harness()
    harness.citations.blocks[(OWNER.hex, MESSAGE_ID, 1)] = CitationBlock(
        citation=build_citation(
            1, chunk=1, strategy=HydrateStrategy.CHUNK_WINDOW, block=(1, 2)
        ),
        chunks=[
            CitedChunk(
                chunk_version_id=chunk_id(n),
                text=f"| Tuổi | Liều |\n| --- | --- |\n| {n} | x |",
                start_page=None,
                end_page=None,
            )
            for n in (1, 2)
        ],
        is_current=True,
    )

    async with harness.client() as client:
        found = await client.get(f"/api/v1/messages/{MESSAGE_ID}/citations/1")
        missing = await client.get(f"/api/v1/messages/{MESSAGE_ID}/citations/2")
        zero = await client.get(f"/api/v1/messages/{MESSAGE_ID}/citations/0")
        bad_id = await client.get("/api/v1/messages/nope/citations/1")

    assert found.status_code == 200, found.text
    body = found.json()
    assert (body["strategy"], body["is_current"], body["document_title"]) == (
        "chunk_window",
        True,
        "Paracetamol",
    )
    assert [(chunk["id"], chunk["matched"]) for chunk in body["chunks"]] == [
        (str(chunk_id(1)), True),
        (str(chunk_id(2)), False),
    ]
    assert body["chunks"][0]["start_page"] is None
    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/problem+json"
    assert missing.json()["code"] == "CITATION_NOT_FOUND"
    assert missing.json()["type"] == "urn:pharma-agent:problem:citation-not-found"
    assert zero.status_code == 422 and zero.json()["code"] == "VALIDATION_ERROR"
    assert bad_id.status_code == 422
