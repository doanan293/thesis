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
            "reasons": {},
        }

    degraded = build_harness(
        authenticated=False,
        agent=False,
        health={"postgres": _up, "corpus": _down},
        health_reasons={"corpus": "CORPUS_NOT_READY"},
    )
    async with degraded.client() as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "agent": False,
            "checks": {"postgres": True, "corpus": False},
            "reasons": {"corpus": "CORPUS_NOT_READY"},
        }


async def test_reasons_list_only_failed_checks() -> None:
    harness = build_harness(
        authenticated=False,
        health={"postgres": _down, "corpus": _up},
        health_reasons={"corpus": "CORPUS_NOT_READY"},
    )
    async with harness.client() as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["reasons"] == {}
