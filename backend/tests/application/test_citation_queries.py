import pytest

from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.errors import CitationNotFound
from pharma_agent.domain.conversation.models import CitationBlock, CitedChunk
from pharma_agent.domain.shared.clock import FixedClock
from tests.citations import SOURCE_TITLE, build_citation, chunk_id
from tests.domain.factories import NOW
from tests.memory_repository import (
    InMemoryCitationReader,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)

OWNER, STRANGER = "a" * 32, "b" * 32
MESSAGE_ID = "d" * 32


async def test_get_citation_marks_the_matched_chunk_in_block_order() -> None:
    reader = InMemoryCitationReader()
    reader.blocks[(OWNER, MESSAGE_ID, 1)] = CitationBlock(
        citation=build_citation(1, chunk=2, block=(1, 2, 3)),
        chunks=[
            CitedChunk(
                chunk_version_id=chunk_id(n),
                text=f"đoạn {n}",
                start_page=810 + n,
                end_page=810 + n,
            )
            for n in (1, 2, 3)
        ],
        is_current=False,
    )
    queries = ConversationQueries(
        InMemoryConversationRepository(),
        FixedClock(NOW),
        feedback=InMemoryFeedbackRepository(),
        citations=reader,
    )

    detail = await queries.get_citation(OWNER, MESSAGE_ID, 1)

    assert detail.model_dump(mode="json") == {
        "index": 1,
        "source": SOURCE_TITLE,
        "document_title": "Paracetamol",
        "section": "Liều lượng và cách dùng",
        "start_page": 812,
        "end_page": 813,
        "strategy": "full_section",
        "is_current": False,
        "chunks": [
            {
                "id": str(chunk_id(n)),
                "text": f"đoạn {n}",
                "matched": n == 2,
                "start_page": 810 + n,
                "end_page": 810 + n,
            }
            for n in (1, 2, 3)
        ],
    }
    for user_id, index in ((STRANGER, 1), (OWNER, 2)):
        with pytest.raises(CitationNotFound):
            await queries.get_citation(user_id, MESSAGE_ID, index)
