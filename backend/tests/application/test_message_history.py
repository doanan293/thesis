import uuid
from collections.abc import Collection, Sequence

from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.feedback.models import Feedback, Rating
from pharma_agent.domain.shared.clock import FixedClock
from pharma_agent.domain.shared.ids import new_id
from tests.citations import CURRENT_RELEASE_ID, OLD_RELEASE_ID, build_citation
from tests.domain.factories import NOW, make_run
from tests.memory_repository import (
    InMemoryCitationReader,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)

OWNER, STRANGER = "a" * 32, "b" * 32


class CountingFeedback(InMemoryFeedbackRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[list[str]] = []

    async def for_messages(
        self, user_id: str, message_ids: Sequence[str]
    ) -> dict[str, Feedback]:
        self.lookups.append(list(message_ids))
        return await super().for_messages(user_id, message_ids)


class CountingCitations(InMemoryCitationReader):
    def __init__(self, *current_release_ids: uuid.UUID) -> None:
        super().__init__(*current_release_ids)
        self.lookups: list[set[uuid.UUID]] = []

    async def current_release_ids(
        self, release_ids: Collection[uuid.UUID]
    ) -> set[uuid.UUID]:
        self.lookups.append(set(release_ids))
        return await super().current_release_ids(release_ids)


async def test_list_messages_returns_ui_messages_with_feedback_and_is_current() -> None:
    repo, feedback = InMemoryConversationRepository(), CountingFeedback()
    citations = CountingCitations(CURRENT_RELEASE_ID)
    queries = ConversationQueries(
        repo, FixedClock(NOW), feedback=feedback, citations=citations
    )
    conversation = Conversation.start(
        user_id=OWNER, first_message="Paracetamol?", now=NOW
    )
    await repo.create(conversation)
    run = make_run("Paracetamol?")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=NOW)
    run.complete()
    user_msg, assistant_msg = build_turn_messages(
        user_message_id=new_id(),
        assistant_message_id=new_id(),
        conversation_id=conversation.conversation_id,
        run=run,
        answer_text="Người lớn 0,5–1 g [1], trẻ em theo cân nặng [2].",
        citations=[
            build_citation(1, chunk=1),
            build_citation(2, chunk=2, release_id=OLD_RELEASE_ID),
        ],
        phases=["answering"],
        now=NOW,
    )
    conversation.record_turn(assistant_msg.created_at)
    await repo.append_turn(conversation, user_msg, assistant_msg, [])
    await feedback.save(
        Feedback.create(
            user_id=OWNER,
            message_id=assistant_msg.message_id,
            rating=Rating.DOWN,
            note="thiếu liều trẻ em",
            now=NOW,
        )
    )
    await feedback.save(
        Feedback.create(
            user_id=STRANGER,
            message_id=assistant_msg.message_id,
            rating=Rating.UP,
            note="",
            now=NOW,
        )
    )

    page = await queries.list_messages(OWNER, conversation.conversation_id, limit=30)

    assert page.next_cursor is None
    assert [message.id for message in page.items] == [
        user_msg.message_id,
        assistant_msg.message_id,
    ]
    user, assistant = (message.model_dump(mode="json") for message in page.items)
    assert user["parts"] == [{"type": "text", "text": "Paracetamol?"}]
    assert user["metadata"]["feedback"] is None
    assert [part["type"] for part in assistant["parts"]] == [
        "text",
        "source-document",
        "source-document",
    ]
    assert [
        part["providerMetadata"]["pharma"]["isCurrent"]
        for part in assistant["parts"][1:]
    ] == [True, False]
    assert assistant["metadata"]["feedback"] == {
        "rating": "down",
        "note": "thiếu liều trẻ em",
    }
    assert assistant["metadata"]["runId"] == run.run_id
    assert feedback.lookups == [[assistant_msg.message_id]]
    assert citations.lookups == [{CURRENT_RELEASE_ID, OLD_RELEASE_ID}]
