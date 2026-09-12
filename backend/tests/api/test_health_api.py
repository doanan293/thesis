from tests.api.harness import build_harness


async def _down() -> bool:
    return False


async def _up() -> bool:
    return True


async def test_health_ok_and_degraded() -> None:
    ok = build_harness(authenticated=False, health={"postgres": _up})
    async with ok.client() as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "agent": True,
            "checks": {"postgres": True},
        }

    degraded = build_harness(
        authenticated=False, agent=False, health={"postgres": _up, "qdrant": _down}
    )
    async with degraded.client() as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "agent": False,
            "checks": {"postgres": True, "qdrant": False},
        }
