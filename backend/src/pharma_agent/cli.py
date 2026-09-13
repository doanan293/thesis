"""Developer CLI: `pharma-agent ask "..."` runs one turn end-to-end; `pharma-agent check` verifies connectivity."""

import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import typer

from pharma_agent.application.corpus.import_bundle import ImportReport
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle
from pharma_agent.domain.corpus.models import ReleaseSummary
from pharma_agent.domain.retrieval.models import Hit, HydrateStrategy, page_label
from pharma_agent.domain.shared.errors import DomainError
from pharma_agent.infrastructure.auth.session_cleanup import (
    delete_expired_access_tokens,
)
from pharma_agent.infrastructure.composition import (
    build_application,
    build_retrieval_service,
)
from pharma_agent.infrastructure.corpus_factory import (
    CorpusServices,
    open_corpus_services,
)
from pharma_agent.infrastructure.langgraph.cleanup import delete_expired_checkpoints
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
                    "citations": [c.model_dump(mode="json") for c in outcome.citations],
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
        label = page_label(citation.start_page, citation.end_page)
        pages = f" ({label})" if label else ""
        typer.echo(f"[{citation.index}] {citation.title} > {citation.section}{pages}")
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
    elif event.type is EventType.DONE and event.data.get("persisted") is False:
        typer.echo("!! Không lưu được lượt hội thoại này.", err=True)


@app.command()
def check() -> None:
    """Kiểm tra cấu hình LLM, corpus (Qdrant + release hiện hành), embedding và reranker."""
    settings = Settings()
    failures = asyncio.run(_check(settings))
    for name, ok, detail in failures:
        typer.echo(f"{'OK ' if ok else 'ERR'} {name}: {detail}")
    if any(not ok for _, ok, _ in failures):
        raise typer.Exit(code=1)


async def _check(settings: Settings) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = [
        (
            "llm",
            settings.llm.configured,
            "api key configured"
            if settings.llm.configured
            else "missing PHARMA_LLM__DEFAULT__API_KEY",
        )
    ]
    retrieval = build_retrieval_service(settings)
    embedding = settings.retrieval.embedding
    try:
        try:
            await retrieval.retriever.verify_corpus(
                embedding_model=embedding.model, dimension=embedding.dimension
            )
            results.append(
                (
                    "corpus",
                    True,
                    f"{settings.retrieval.qdrant_collection} ({embedding.model}, {embedding.dimension} dims), "
                    f"current release for {', '.join(settings.retrieval.collections)}",
                )
            )
        except Exception as exc:
            results.append(("corpus", False, f"CORPUS_NOT_READY: {exc}"))
        try:
            vectors = await retrieval.embedder.embed(["kiểm tra"])
            results.append(
                ("embedding", True, f"{embedding.model} -> {len(vectors[0])} dims")
            )
        except Exception as exc:
            results.append(("embedding", False, str(exc)))
        try:
            ranked = await retrieval.reranker.rerank(
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
        await retrieval.aclose()
    return results


_PROBE_ID = uuid.UUID(int=0)
_PROBE_HIT = Hit(
    chunk_version_id=_PROBE_ID,
    release_id=_PROBE_ID,
    collection_id=_PROBE_ID,
    document_key="probe",
    section_key="probe",
    section_revision_id=_PROBE_ID,
    ordinal=1,
    hydrate_strategy=HydrateStrategy.SEARCH_ONLY,
    source="probe",
    title="Paracetamol",
    section="Liều dùng",
    start_page=None,
    end_page=None,
    context_header="Paracetamol > Liều dùng",
    chunk_text="Người lớn 500 mg",
    embedding_text="Paracetamol > Liều dùng\n\nNgười lớn 500 mg",
    kind="prose",
    table_key=None,
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


@app.command("export-openapi")
def export_openapi(
    output: Path = typer.Option(
        ...,
        "--output",
        dir_okay=False,
        help="File to write, for example ../frontend/openapi.json",
    ),
) -> None:
    """Ghi tài liệu OpenAPI của HTTP API ra file, không cần chạy server."""
    from pharma_agent.api.app import create_app
    from pharma_agent.api.openapi import openapi_export_settings, render_openapi

    document = render_openapi(create_app(openapi_export_settings()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8", newline="\n")
    typer.echo(f"wrote {output}")


@app.command("cleanup-checkpoints")
def cleanup_checkpoints(
    days: int | None = typer.Option(
        None, min=1, help="Số ngày lưu giữ (mặc định lấy từ settings)"
    ),
) -> None:
    """Xóa checkpoint LangGraph cũ hơn số ngày lưu giữ."""
    settings = Settings()
    retention = days if days is not None else settings.checkpoints.retention_days
    deleted = asyncio.run(
        delete_expired_checkpoints(settings.postgres.conninfo, retention_days=retention)
    )
    typer.echo(f"deleted {deleted} checkpoints older than {retention} days")


@app.command("cleanup-sessions")
def cleanup_sessions() -> None:
    """Xóa phiên đăng nhập bằng cookie đã hết hạn."""
    settings = Settings()
    lifetime = settings.auth.session_lifetime_seconds
    deleted = asyncio.run(
        delete_expired_access_tokens(
            settings.postgres.conninfo, lifetime_seconds=lifetime
        )
    )
    typer.echo(f"deleted {deleted} sessions older than {lifetime} seconds")


corpus_app = typer.Typer(
    help="Corpus: import knowledge bundle, quản lý release", no_args_is_help=True
)
app.add_typer(corpus_app, name="corpus")

CorpusAction = Callable[[CorpusServices], Awaitable[None]]


def _run_corpus(action: CorpusAction, settings: Settings | None = None) -> None:
    try:
        asyncio.run(_with_corpus_services(settings or Settings(), action))
    except DomainError as exc:
        typer.echo(f"!! {exc.code}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _with_corpus_services(settings: Settings, action: CorpusAction) -> None:
    async with open_corpus_services(settings) as services:
        await action(services)


def _echo_import(report: ImportReport) -> None:
    release = report.release
    published = " published" if report.published else ""
    typer.echo(
        f"{report.outcome.value}: {report.collection_key} release {release.number} "
        f"{release.id} [{release.status.value}]{published}"
    )
    stats = release.stats
    if stats is not None:
        typer.echo(
            f"chunks {stats.chunks} | embeddings cached {stats.embeddings_cached} "
            f"bundle {stats.embeddings_from_bundle} computed {stats.embeddings_computed} "
            f"| points new {stats.points_upserted} updated {stats.points_updated}"
        )


def _release_line(summary: ReleaseSummary) -> str:
    release = summary.release
    marker = "*" if summary.current else " "
    published = release.published_at.isoformat() if release.published_at else "-"
    return (
        f"{marker} {summary.collection_key} #{release.number} {release.id} "
        f"{release.status.value} chunks={summary.chunk_count} published={published}"
    )


@corpus_app.command("import")
def corpus_import(
    bundle_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Thư mục knowledge bundle",
    ),
    collection: str = typer.Option(
        ..., "--collection", help="Key của collection, phải trùng manifest"
    ),
    publish: bool = typer.Option(
        False, "--publish", help="Đặt release mới làm release hiện hành"
    ),
) -> None:
    """Import knowledge bundle thành một release mới."""
    try:
        bundle = read_bundle(bundle_dir)
    except BundleValidationError as exc:
        typer.echo(f"!! {exc.code}: {bundle_dir}", err=True)
        for problem in exc.problems:
            typer.echo(f"   - {problem}", err=True)
        raise typer.Exit(code=2) from exc
    if bundle.manifest.collection.key != collection:
        typer.echo(
            f"!! bundle collection is {bundle.manifest.collection.key}, not {collection}",
            err=True,
        )
        raise typer.Exit(code=2)

    async def action(services: CorpusServices) -> None:
        _echo_import(await services.importer(bundle, publish=publish))

    _run_corpus(action)


@corpus_app.command("releases")
def corpus_releases(
    collection: str | None = typer.Option(
        None, "--collection", help="Lọc theo collection"
    ),
) -> None:
    """Liệt kê release, trạng thái, số chunk và release hiện hành (*)."""

    async def action(services: CorpusServices) -> None:
        summaries = await services.releases.list_releases(collection)
        if not summaries:
            typer.echo("no releases")
        for summary in summaries:
            typer.echo(_release_line(summary))

    _run_corpus(action)


@corpus_app.command("publish")
def corpus_publish(
    release_id: uuid.UUID = typer.Argument(
        ..., help="Id của release ở trạng thái ready"
    ),
) -> None:
    """Trỏ release hiện hành của collection tới release này."""

    async def action(services: CorpusServices) -> None:
        release = await services.releases.publish(release_id)
        typer.echo(f"published release {release.number} {release.id}")

    _run_corpus(action)


@corpus_app.command("rollback")
def corpus_rollback(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
) -> None:
    """Trỏ về release được publish liền trước."""

    async def action(services: CorpusServices) -> None:
        release = await services.releases.rollback(collection)
        typer.echo(f"rolled back {collection} to release {release.number} {release.id}")

    _run_corpus(action)


@corpus_app.command("gc")
def corpus_gc(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
    keep: int | None = typer.Option(
        None,
        "--keep",
        min=0,
        help="Số release gần nhất được giữ (mặc định từ settings)",
    ),
) -> None:
    """Retire release cũ, dọn point Qdrant và chunk version không còn dùng."""
    settings = Settings()
    retained = keep if keep is not None else settings.corpus.gc_keep

    async def action(services: CorpusServices) -> None:
        report = await services.releases.gc(collection, retained)
        typer.echo(
            f"retired {len(report.retired)} releases (keep {retained}) | points updated "
            f"{report.points_updated} deleted {report.points_deleted} | chunk versions "
            f"deleted {report.purge.chunk_versions_deleted} kept "
            f"{report.purge.chunk_versions_kept} | section revisions deleted "
            f"{report.purge.section_revisions_deleted}"
        )

    _run_corpus(action, settings)


@corpus_app.command("reindex")
def corpus_reindex(
    collection: str = typer.Option(..., "--collection", help="Key của collection"),
) -> None:
    """Dựng lại point Qdrant của các release chưa retire từ Postgres và embedding cache."""

    async def action(services: CorpusServices) -> None:
        report = await services.releases.reindex(collection)
        typer.echo(f"points upserted {report.points_upserted}")
        for release_id, count in report.release_points.items():
            typer.echo(f"release {release_id}: {count} points")
        for release_id in report.skipped:
            typer.echo(f"skipped release {release_id} (other embedding model)")

    _run_corpus(action)
