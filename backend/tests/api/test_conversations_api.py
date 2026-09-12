from datetime import UTC, datetime, timedelta

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from tests.api.harness import OWNER, build_harness
from tests.domain.factories import make_run


async def test_conversation_endpoints() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    mine = Conversation.start(user_id=OWNER.hex, first_message="Paracetamol", now=now)
    foreign = Conversation.start(
        user_id="b" * 32, first_message="Không phải của tôi", now=now
    )
    await harness.repo.create(mine)
    await harness.repo.create(foreign)
    run = make_run("q")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=now)
    run.complete()
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=mine.conversation_id,
        run=run,
        answer_text="a",
        citations=[],
        phases=[],
        now=now + timedelta(seconds=1),
    )
    mine.record_turn(now + timedelta(seconds=1))
    await harness.repo.append_turn(mine, user_msg, assistant_msg, [])

    async with harness.client() as client:
        listed = await client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert [item["title"] for item in listed.json()] == ["Paracetamol"]

        detail = await client.get(f"/api/v1/conversations/{mine.conversation_id}")
        assert detail.json()["turn_count"] == 1

        messages = (
            await client.get(f"/api/v1/conversations/{mine.conversation_id}/messages")
        ).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]

        renamed = await client.patch(
            f"/api/v1/conversations/{mine.conversation_id}",
            json={"title": "Thuốc hạ sốt"},
        )
        assert renamed.status_code == 200 and renamed.json()["title"] == "Thuốc hạ sốt"
        blank = await client.patch(
            f"/api/v1/conversations/{mine.conversation_id}", json={"title": ""}
        )
        assert blank.status_code == 422

        foreign_get = await client.get(
            f"/api/v1/conversations/{foreign.conversation_id}"
        )
        assert foreign_get.status_code == 404
        foreign_delete = await client.delete(
            f"/api/v1/conversations/{foreign.conversation_id}"
        )
        assert foreign_delete.status_code == 404
        deleted = await client.delete(f"/api/v1/conversations/{mine.conversation_id}")
        assert deleted.status_code == 204
        assert (await client.get("/api/v1/conversations")).json() == []
