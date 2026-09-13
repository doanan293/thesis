from datetime import UTC, datetime, timedelta

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from tests.api.harness import OWNER, Harness, build_harness
from tests.domain.factories import make_run


async def add_turn(harness: Harness, conversation: Conversation, at: datetime) -> None:
    run = make_run("q")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=at)
    run.complete()
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=conversation.conversation_id,
        run=run,
        answer_text="a",
        citations=[],
        phases=[],
        now=at,
    )
    conversation.record_turn(at)
    await harness.repo.append_turn(conversation, user_msg, assistant_msg, [])


async def test_conversation_endpoints() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    mine = Conversation.start(user_id=OWNER.hex, first_message="Paracetamol", now=now)
    foreign = Conversation.start(
        user_id="b" * 32, first_message="Không phải của tôi", now=now
    )
    await harness.repo.create(mine)
    await harness.repo.create(foreign)
    await add_turn(harness, mine, now + timedelta(seconds=1))

    async with harness.client() as client:
        listed = await client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert [item["title"] for item in listed.json()["items"]] == ["Paracetamol"]
        assert listed.json()["next_cursor"] is None

        detail = await client.get(f"/api/v1/conversations/{mine.conversation_id}")
        assert detail.json()["turn_count"] == 1

        messages = (
            await client.get(f"/api/v1/conversations/{mine.conversation_id}/messages")
        ).json()
        assert [m["role"] for m in messages["items"]] == ["user", "assistant"]

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
        assert (await client.get("/api/v1/conversations")).json() == {
            "items": [],
            "next_cursor": None,
        }


async def test_lists_page_with_cursors() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    created: list[Conversation] = []
    for index in range(3):
        conversation = Conversation.start(
            user_id=OWNER.hex, first_message=f"c{index}", now=now
        )
        await harness.repo.create(conversation)
        await add_turn(harness, conversation, now)
        created.append(conversation)
    for _ in range(2):
        await add_turn(harness, created[0], now)
    empty = Conversation.start(user_id=OWNER.hex, first_message="empty", now=now)
    await harness.repo.create(empty)
    target = created[0].conversation_id

    async with harness.client() as client:
        first = (await client.get("/api/v1/conversations", params={"limit": 2})).json()
        assert len(first["items"]) == 2 and first["next_cursor"]
        second = (
            await client.get(
                "/api/v1/conversations",
                params={"limit": 2, "cursor": first["next_cursor"]},
            )
        ).json()
        assert second["next_cursor"] is None
        listed = [item["id"] for item in first["items"] + second["items"]]
        assert listed == sorted((c.conversation_id for c in created), reverse=True)
        assert empty.conversation_id not in listed

        url = f"/api/v1/conversations/{target}/messages"
        newest = (await client.get(url, params={"limit": 4})).json()
        older = (
            await client.get(url, params={"limit": 4, "cursor": newest["next_cursor"]})
        ).json()
        assert [len(newest["items"]), len(older["items"])] == [4, 2]
        assert older["next_cursor"] is None
        timeline = older["items"] + newest["items"]
        assert len({m["id"] for m in timeline}) == 6
        stamps = [m["created_at"] for m in timeline]
        assert stamps == sorted(stamps)


async def test_list_parameters_are_validated() -> None:
    harness = build_harness()
    conversation = Conversation.start(
        user_id=OWNER.hex, first_message="x", now=datetime.now(UTC)
    )
    await harness.repo.create(conversation)
    async with harness.client() as client:
        bad = await client.get("/api/v1/conversations", params={"cursor": "nope"})
        bad_messages = await client.get(
            f"/api/v1/conversations/{conversation.conversation_id}/messages",
            params={"cursor": "e30"},  # base64url of "{}"
        )
        too_many = await client.get("/api/v1/conversations", params={"limit": 101})
        too_long = await client.get(
            "/api/v1/conversations", params={"cursor": "a" * 513}
        )
    for response in (bad, bad_messages):
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_CURSOR"
    for response in (too_many, too_long):
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
