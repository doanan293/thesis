import pytest

from pharma_agent.application.errors import InvalidInput
from pharma_agent.application.feedback.service import FeedbackService, MessageNotFound
from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.feedback.models import Rating
from pharma_agent.domain.shared.clock import FixedClock
from pharma_agent.domain.shared.ids import new_id
from tests.domain.factories import NOW, make_run
from tests.memory_repository import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)

OWNER, STRANGER = "a" * 32, "b" * 32


class RecordingSink:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail = fail

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        if self.fail:
            raise RuntimeError("langfuse down")
        self.calls.append((run_id, rating, note))


async def seeded() -> tuple[InMemoryConversationRepository, str, str]:
    repo = InMemoryConversationRepository()
    conversation = Conversation.start(user_id=OWNER, first_message="hi", now=NOW)
    await repo.create(conversation)
    run = make_run("q")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=NOW)
    run.complete()
    user_msg, assistant_msg = build_turn_messages(
        user_message_id=new_id(),
        assistant_message_id=new_id(),
        conversation_id=conversation.conversation_id,
        run=run,
        answer_text="a",
        citations=[],
        phases=[],
        now=NOW,
    )
    conversation.record_turn(NOW)
    await repo.append_turn(conversation, user_msg, assistant_msg, [])
    return repo, user_msg.message_id, assistant_msg.message_id


async def test_submit_stores_feedback_and_scores_the_run() -> None:
    conversations, _, assistant_id = await seeded()
    feedback_repo, sink = InMemoryFeedbackRepository(), RecordingSink()
    service = FeedbackService(conversations, feedback_repo, sink, FixedClock(NOW))

    view = await service.submit(
        user_id=OWNER, message_id=assistant_id, rating="up", note=" tốt "
    )

    assert (
        view.rating is Rating.UP
        and view.note == "tốt"
        and view.message_id == assistant_id
    )
    stored = await feedback_repo.get(OWNER, assistant_id)
    assert stored is not None and stored.rating is Rating.UP
    assert sink.calls == [("run-1", "up", "tốt")]


async def test_submit_rejects_foreign_unknown_and_user_messages() -> None:
    conversations, user_id, assistant_id = await seeded()
    service = FeedbackService(
        conversations, InMemoryFeedbackRepository(), RecordingSink(), FixedClock(NOW)
    )
    for owner, message_id in (
        (STRANGER, assistant_id),
        (OWNER, "f" * 32),
        (OWNER, user_id),
    ):
        with pytest.raises(MessageNotFound):
            await service.submit(
                user_id=owner, message_id=message_id, rating="up", note=""
            )
    with pytest.raises(InvalidInput, match="rating"):
        await service.submit(
            user_id=OWNER, message_id=assistant_id, rating="meh", note=""
        )
    with pytest.raises(InvalidInput, match="note"):
        await service.submit(
            user_id=OWNER, message_id=assistant_id, rating="up", note="x" * 3000
        )


async def test_sink_failure_does_not_lose_feedback() -> None:
    conversations, _, assistant_id = await seeded()
    feedback_repo = InMemoryFeedbackRepository()
    service = FeedbackService(
        conversations, feedback_repo, RecordingSink(fail=True), FixedClock(NOW)
    )
    view = await service.submit(
        user_id=OWNER, message_id=assistant_id, rating="down", note=""
    )
    assert (
        view.rating is Rating.DOWN
        and await feedback_repo.get(OWNER, assistant_id) is not None
    )
