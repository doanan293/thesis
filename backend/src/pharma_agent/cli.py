"""Developer CLI: `pharma-agent ask "..."` runs one turn end-to-end; `pharma-agent check` verifies connectivity."""

import asyncio
import json
import sys
from typing import Any

import typer

from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.retrieval.models import Hit, HydrateStrategy
from pharma_agent.infrastructure.composition import build_application
from pharma_agent.infrastructure.settings import Settings

app = typer.Typer(help="Pharma agent developer CLI", no_args_is_help=True)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Câu hỏi về thuốc"),
    json_output: bool = typer.Option(
        False, "--json", help="In kết quả dạng JSON thay vì stream"
    ),
) -> None:
    """Chạy một lượt hỏi đáp với cấu hình trong .env."""
    settings = Settings()
    asyncio.run(_ask(settings, question, json_output))


async def _ask(settings: Settings, question: str, json_output: bool) -> None:
    application = build_application(settings)
    try:
        await _run_turn(application, question, json_output)
    finally:
        await application.aclose()


async def _run_turn(application: Any, question: str, json_output: bool) -> None:
    execution = application.runner.start(user_id="cli", message=question)
    evidence: list[dict[str, Any]] = []
    async for event in execution.events():
        if json_output:
            if event.type is EventType.EVIDENCE:
                evidence = event.data["items"]
            continue
        _render(event, evidence)
    outcome = execution.outcome
    assert outcome is not None
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "status": outcome.run.status.value,
                    "answer": outcome.answer_text,
                    "citations": [c.model_dump() for c in outcome.citations],
                    "evidence": evidence,
                    "trace": outcome.run.to_trace(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo("")
    for citation in outcome.citations:
        pages = (
            f"trang {citation.start_page}"
            if citation.start_page == citation.end_page
            else f"trang {citation.start_page}-{citation.end_page}"
        )
        typer.echo(
            f"[{citation.index}] {citation.title} > {citation.section} ({pages})"
        )
    usage = outcome.run.usage
    typer.echo(
        f"status: {outcome.run.status.value} | llm calls: {usage.llm_calls} | tokens: {usage.total_tokens} | search rounds: {usage.search_rounds}"
    )


def _render(event: ProgressEvent, evidence: list[dict[str, Any]]) -> None:
    if event.type is EventType.PHASE:
        typer.echo(f"… {event.data['phase']}", err=True)
    elif event.type is EventType.SKILLS_SELECTED:
        names = ", ".join(s["name"] for s in event.data["skills"])
        typer.echo(f"… skills: {names}", err=True)
    elif event.type is EventType.EVIDENCE:
        evidence.extend(event.data["items"])
        typer.echo(f"… evidence: {len(event.data['items'])} nguồn", err=True)
    elif event.type is EventType.TOKEN:
        sys.stdout.write(event.data["text"])
        sys.stdout.flush()
    elif event.type is EventType.ERROR:
        typer.echo(f"!! {event.data['code']}: {event.data['message']}", err=True)


@app.command()
def check() -> None:
    """Kiểm tra kết nối Qdrant, embedding, reranker và cấu hình LLM."""
    settings = Settings()
    failures = asyncio.run(_check(settings))
    for name, ok, detail in failures:
        typer.echo(f"{'OK ' if ok else 'ERR'} {name}: {detail}")
    if any(not ok for _, ok, _ in failures):
        raise typer.Exit(code=1)


async def _check(settings: Settings) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    results.append(
        (
            "llm",
            settings.llm.configured,
            "api key configured"
            if settings.llm.configured
            else "missing PHARMA_LLM__DEFAULT__API_KEY",
        )
    )
    if not settings.llm.configured:
        return results
    application = build_application(settings)
    try:
        try:
            await application.retriever.verify_collection(
                settings.retrieval.embedding.dimension
            )
            results.append(
                (
                    "qdrant",
                    True,
                    f"{settings.retrieval.collection_alias} dimension {settings.retrieval.embedding.dimension}",
                )
            )
        except Exception as exc:
            results.append(("qdrant", False, str(exc)))
        try:
            vectors = await application.embedder.embed(["kiểm tra"])
            results.append(
                (
                    "embedding",
                    True,
                    f"{settings.retrieval.embedding.model} -> {len(vectors[0])} dims",
                )
            )
        except Exception as exc:
            results.append(("embedding", False, str(exc)))
        try:
            ranked = await application.reranker.rerank(
                "liều paracetamol", [_PROBE_HIT], top_n=1
            )
            results.append(
                (
                    "rerank",
                    True,
                    f"{settings.retrieval.rerank.protocol} score {ranked[0].rerank_score}",
                )
            )
        except Exception as exc:
            results.append(("rerank", False, str(exc)))
    finally:
        await application.aclose()
    return results


_PROBE_HIT = Hit(
    chunk_id="probe",
    section_id="probe",
    chunk_index=0,
    hydrate_strategy=HydrateStrategy.SEARCH_ONLY,
    source="probe",
    title="Paracetamol",
    section="Liều dùng",
    start_page=1,
    end_page=1,
    context_header="Paracetamol > Liều dùng",
    chunk_text="Người lớn 500 mg",
    embedding_text="Paracetamol > Liều dùng\n\nNgười lớn 500 mg",
)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default from settings)"),
    port: int | None = typer.Option(None, help="Port (default from settings)"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Chạy HTTP API (FastAPI + SSE)."""
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "pharma_agent.api.app:create_app",
        factory=True,
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )


@app.command()
def migrate(revision: str = typer.Argument("head", help="Alembic revision")) -> None:
    """Áp dụng migration Postgres."""
    from alembic import command

    from pharma_agent.infrastructure.persistence.postgres.alembic_config import (
        alembic_config,
    )

    command.upgrade(alembic_config(), revision)
