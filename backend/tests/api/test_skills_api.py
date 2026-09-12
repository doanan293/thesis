from pharma_agent.application.skill.service import MAX_SKILL_BYTES
from pharma_agent.domain.skill.models import Skill
from tests.api.harness import build_harness

VALID = (
    "---\nname: Ghi chú của tôi\ndescription: Dùng khi hỏi về thuốc bổ sung.\n---\n\n"
    "## Tìm kiếm\n- Tìm mục liều.\n"
).encode()
SYSTEM = Skill(
    skill_id="drug-monograph",
    name="Tra cứu chuyên luận",
    description="Dùng khi hỏi liều.",
    search_guidance="Tìm mục liều.",
    version="v1",
)


async def test_skill_lifecycle() -> None:
    harness = build_harness()
    await harness.skill_repo.upsert_system([SYSTEM])
    async with harness.client() as client:
        uploaded = await client.post(
            "/api/v1/skills", files={"file": ("SKILL.md", VALID, "text/markdown")}
        )
        assert uploaded.status_code == 201, uploaded.text
        skill = uploaded.json()
        assert skill["is_system"] is False and skill["enabled"] is True
        assert skill["id"].startswith("ghi-chu-cua-toi-")

        listed = (await client.get("/api/v1/skills")).json()
        assert [item["id"] for item in listed] == ["drug-monograph", skill["id"]]
        assert listed[0]["is_system"] is True

        disabled = await client.patch(
            f"/api/v1/skills/{skill['id']}", json={"enabled": False}
        )
        assert disabled.status_code == 200 and disabled.json()["enabled"] is False

        system_patch = await client.patch(
            "/api/v1/skills/drug-monograph", json={"enabled": False}
        )
        assert system_patch.status_code == 404
        assert system_patch.json()["code"] == "SKILL_NOT_FOUND"
        assert (await client.delete("/api/v1/skills/drug-monograph")).status_code == 404

        deleted = await client.delete(f"/api/v1/skills/{skill['id']}")
        assert deleted.status_code == 204
        remaining = (await client.get("/api/v1/skills")).json()
        assert [item["id"] for item in remaining] == ["drug-monograph"]


async def test_upload_errors() -> None:
    harness = build_harness()
    async with harness.client() as client:
        wrong_name = await client.post(
            "/api/v1/skills", files={"file": ("notes.md", VALID, "text/markdown")}
        )
        assert wrong_name.status_code == 422
        assert wrong_name.json()["code"] == "INVALID_INPUT"

        too_big = await client.post(
            "/api/v1/skills",
            files={
                "file": ("SKILL.md", b"x" * (MAX_SKILL_BYTES + 10), "text/markdown")
            },
        )
        assert too_big.status_code == 413
        assert too_big.json()["code"] == "PAYLOAD_TOO_LARGE"

        assert (await client.post("/api/v1/skills")).status_code == 422
        bad_id = await client.patch("/api/v1/skills/Bad_Id", json={"enabled": True})
        assert bad_id.status_code == 422
