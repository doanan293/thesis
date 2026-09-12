from tests.api.harness import OWNER, build_harness
from tests.api.test_chat_api import script_turn


async def test_feedback_on_own_assistant_message() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    async with harness.client() as client:
        chat = (
            await client.post("/api/v1/chat", json={"message": "Paracetamol?"})
        ).json()
        message_id = chat["message_id"]

        created = await client.post(
            f"/api/v1/messages/{message_id}/feedback",
            json={"rating": "down", "note": "thiếu liều trẻ em"},
        )
        assert created.status_code == 201, created.text
        assert created.json()["rating"] == "down"
        assert harness.sink.calls == [(chat["run_id"], "down", "thiếu liều trẻ em")]

        again = await client.post(
            f"/api/v1/messages/{message_id}/feedback", json={"rating": "up"}
        )
        assert again.status_code == 201
        stored = await harness.feedback_repo.get(OWNER.hex, message_id)
        assert stored is not None and stored.rating.value == "up" and stored.note == ""

        missing = await client.post(
            f"/api/v1/messages/{'f' * 32}/feedback", json={"rating": "up"}
        )
        assert missing.status_code == 404
        assert missing.json()["code"] == "MESSAGE_NOT_FOUND"

        invalid = await client.post(
            f"/api/v1/messages/{message_id}/feedback", json={"rating": "meh"}
        )
        assert invalid.status_code == 422
        bad_id = await client.post(
            "/api/v1/messages/nope/feedback", json={"rating": "up"}
        )
        assert bad_id.status_code == 422
