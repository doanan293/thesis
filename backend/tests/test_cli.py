import json
from types import SimpleNamespace

from typer.testing import CliRunner

from pharma_agent import cli
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
    SkillSelection,
)
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import LlmRole
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps


def fake_application() -> SimpleNamespace:
    llm = FakeLlm()
    llm.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    llm.script(
        LlmRole.REPHRASE,
        RephraseResult(
            standalone_query="Liều paracetamol",
            audience=Audience.GENERAL_PUBLIC,
            language=Language.VI,
            intent=Intent.PHARMA_QUESTION,
        ),
    )
    llm.script(LlmRole.SKILL_SELECTOR, SkillSelection(skill_names=["drug-monograph"]))
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    deps = build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)]))
    runner = ChatTurnRunner(build_chat_graph(), deps, BudgetLimits())

    async def aclose() -> None:
        return None

    return SimpleNamespace(runner=runner, aclose=aclose)


def test_ask_streams_answer_and_citations(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_application", lambda settings: fake_application())
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    result = CliRunner().invoke(cli.app, ["ask", "Paracetamol uống bao nhiêu?"])
    assert result.exit_code == 0, result.output
    assert "500 mg" in result.output
    assert "[1] Paracetamol > Liều dùng" in result.output
    assert "status: completed" in result.output


def test_ask_json_output(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_application", lambda settings: fake_application())
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    result = CliRunner().invoke(
        cli.app, ["ask", "Paracetamol uống bao nhiêu?", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "completed" and payload["citations"][0]["index"] == 1
    assert payload["trace"]["usage"]["llm_calls"] == 5


def test_migrate_upgrades_to_head(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "alembic.command.upgrade",
        lambda config, revision: calls.append((config, revision)),
    )
    result = CliRunner().invoke(cli.app, ["migrate"])
    assert result.exit_code == 0, result.output
    config, revision = calls[0]
    assert revision == "head"
    assert config.get_main_option("script_location").endswith(":migrations")


def test_serve_runs_uvicorn_factory(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: calls.update(app=app, **kwargs)
    )
    monkeypatch.setenv("PHARMA_API__PORT", "9001")
    result = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert (
        calls["app"] == "pharma_agent.api.app:create_app" and calls["factory"] is True
    )
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)


def test_cleanup_checkpoints_uses_settings_or_option(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    async def fake_cleanup(conninfo: str, *, retention_days: int) -> int:
        calls.append((conninfo, retention_days))
        return 3

    monkeypatch.setattr(cli, "delete_expired_checkpoints", fake_cleanup)
    monkeypatch.setenv("PHARMA_CHECKPOINTS__RETENTION_DAYS", "14")
    monkeypatch.setenv(
        "PHARMA_POSTGRES__DSN", "postgresql+psycopg://u:p@db.example:5432/app"
    )

    default = CliRunner().invoke(cli.app, ["cleanup-checkpoints"])
    assert default.exit_code == 0, default.output
    assert "deleted 3 checkpoints older than 14 days" in default.output

    explicit = CliRunner().invoke(cli.app, ["cleanup-checkpoints", "--days", "2"])
    assert explicit.exit_code == 0, explicit.output
    assert calls == [
        ("postgresql://u:p@db.example:5432/app", 14),
        ("postgresql://u:p@db.example:5432/app", 2),
    ]
