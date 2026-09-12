from pharma_agent.application.skill.service import MAX_SKILL_BYTES
from pharma_agent.domain.skill.models import Skill
from tests.api.harness import build_harness

VALID_TEXT = (
    "---\n"
    "name: ghi-chu-thuoc-bo\n"
    "description: Ghi chú thuốc bổ. Dùng khi hỏi về vitamin.\n"
    "---\n\n"
    "# Ghi chú thuốc bổ\n\n"
    "Tìm mục liều.\n"
)
VALID = VALID_TEXT.encode()
SYSTEM = Skill.from_markdown(
    "---\nname: drug-monograph\ndescription: Tra cứu chuyên luận. Dùng khi hỏi liều.\n"
    "---\n\n# Tra cứu chuyên luận thuốc\n"
)


async def test_skill_lifecycle() -> None:
    harness = build_harness()
    await harness.skill_repo.replace_system([SYSTEM])
    async with harness.client() as client:
        uploaded = await client.post(
            "/api/v1/skills", files={"file": ("SKILL.md", VALID, "text/markdown")}
        )
        assert uploaded.status_code == 201, uploaded.text
        skill = uploaded.json()
        assert (
            skill["name"],
            skill["title"],
            skill["is_system"],
            skill["enabled"],
        ) == (
            "ghi-chu-thuoc-bo",
            "Ghi chú thuốc bổ",
            False,
            True,
        )

        listed = (await client.get("/api/v1/skills")).json()
        assert [(item["name"], item["is_system"]) for item in listed] == [
            ("drug-monograph", True),
            ("ghi-chu-thuoc-bo", False),
        ]

        duplicate = await client.post(
            "/api/v1/skills", files={"file": ("SKILL.md", VALID, "text/markdown")}
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["code"] == "SKILL_NAME_TAKEN"

        disabled = await client.patch(
            "/api/v1/skills/ghi-chu-thuoc-bo", json={"enabled": False}
        )
        assert disabled.status_code == 200 and disabled.json()["enabled"] is False

        system_patch = await client.patch(
            "/api/v1/skills/drug-monograph", json={"enabled": False}
        )
        assert system_patch.status_code == 404
        assert system_patch.json()["code"] == "SKILL_NOT_FOUND"
        assert (await client.delete("/api/v1/skills/drug-monograph")).status_code == 404

        deleted = await client.delete("/api/v1/skills/ghi-chu-thuoc-bo")
        assert deleted.status_code == 204
        remaining = (await client.get("/api/v1/skills")).json()
        assert [item["name"] for item in remaining] == ["drug-monograph"]


async def test_upload_errors() -> None:
    harness = build_harness()
    async with harness.client() as client:
        wrong_file = await client.post(
            "/api/v1/skills", files={"file": ("notes.md", VALID, "text/markdown")}
        )
        assert wrong_file.status_code == 422
        assert wrong_file.json()["code"] == "INVALID_INPUT"

        free_text_name = VALID_TEXT.replace(
            "name: ghi-chu-thuoc-bo", "name: Ghi chú thuốc bổ"
        ).encode()
        invalid_name = await client.post(
            "/api/v1/skills",
            files={"file": ("SKILL.md", free_text_name, "text/markdown")},
        )
        assert invalid_name.status_code == 422
        assert "lowercase" in invalid_name.json()["message"]

        too_big = await client.post(
            "/api/v1/skills",
            files={
                "file": ("SKILL.md", b"x" * (MAX_SKILL_BYTES + 10), "text/markdown")
            },
        )
        assert too_big.status_code == 413
        assert too_big.json()["code"] == "PAYLOAD_TOO_LARGE"

        assert (await client.post("/api/v1/skills")).status_code == 422
        long_name = await client.patch(
            f"/api/v1/skills/{'a' * 65}", json={"enabled": True}
        )
        assert long_name.status_code == 422
