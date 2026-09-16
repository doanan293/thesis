"""Resumable, concurrent E2E run over the golden set (spec §6.3)."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.infrastructure.composition import build_application
from pharma_agent.infrastructure.settings import Settings

from pharma_lab.e2e.configs import E2EConfig, pipeline_for, settings_for
from pharma_lab.e2e.executor import BackendTurnExecutor
from pharma_lab.e2e.golden import Category, GoldenItem, load_golden
from pharma_lab.e2e.records import AnswerRecord, JsonlStore
from pharma_lab.e2e.run_identity import git_commit, identity_from_settings, open_run
from pharma_lab.evaluation.artifact_contracts import sha256_file
from pharma_lab.evaluation.backend_retrieval import current_release_id

ANSWERS_FILE = "answers.jsonl"


class TurnExecutor(Protocol):
    async def execute(self, item: GoldenItem) -> AnswerRecord: ...


@dataclass(frozen=True)
class RunSummary:
    total: int
    ran: int
    errors: int


async def run_items(
    items: Sequence[GoldenItem],
    executor: TurnExecutor,
    store: JsonlStore[AnswerRecord],
    *,
    config: str,
    concurrency: int,
    retry_errors: bool,
) -> RunSummary:
    """Answer every item without a final record; errors rerun only on request."""
    if concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    latest = store.latest()
    pending = [
        item
        for item in items
        if item.item_id not in latest
        or (retry_errors and latest[item.item_id].retryable)
    ]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def answer(item: GoldenItem) -> None:
        async with semaphore:
            try:
                record = await executor.execute(item)
            except Exception as error:  # a failed item must not stop the run
                record = AnswerRecord.failed(item.item_id, config, error)
        async with lock:
            store.append(record)

    await asyncio.gather(*(answer(item) for item in pending))
    final = store.latest()
    errors = sum(
        1 for item in items if item.item_id in final and final[item.item_id].retryable
    )
    return RunSummary(total=len(items), ran=len(pending), errors=errors)


def spread(items: Sequence[GoldenItem], limit: int | None) -> list[GoldenItem]:
    """`limit` items taken evenly across the (id-sorted) golden set."""
    if limit is None or limit >= len(items):
        return list(items)
    if limit < 1:
        raise ValueError("--limit must be >= 1")
    step = math.ceil(len(items) / limit)
    return list(items[::step])[:limit]


@dataclass(frozen=True)
class E2ERunRequest:
    run_root: Path
    config: E2EConfig
    golden_path: Path
    backend_env_file: Path
    concurrency: int = 4
    retry_errors: bool = False
    limit: int | None = None
    # Evaluation may wait longer than the production turn deadline; recorded in run.json.
    deadline_seconds: float | None = None


def with_deadline(settings: Settings, seconds: float | None) -> Settings:
    if seconds is None:
        return settings
    if seconds <= 0:
        raise ValueError("--deadline-seconds must be positive")
    budget = settings.budget.model_copy(update={"deadline_seconds": seconds})
    return settings.model_copy(update={"budget": budget})


def config_dir(run_root: Path, config: E2EConfig | str) -> Path:
    return Path(run_root) / str(config)


def run_e2e(request: E2ERunRequest) -> RunSummary:
    items = spread(load_golden(request.golden_path), request.limit)
    settings = settings_for(
        Settings(_env_file=request.backend_env_file), request.config
    )
    settings = with_deadline(settings, request.deadline_seconds)
    pipeline = pipeline_for(request.config)
    directory = config_dir(request.run_root, request.config)
    probe = next(
        item.question for item in items if item.category is Category.ANSWERABLE
    )
    app = build_application(settings)
    with asyncio.Runner() as runner:
        try:
            release_id = runner.run(current_release_id(app.retrieval.service, probe))
            open_run(
                directory,
                identity_from_settings(
                    settings,
                    golden_sha256=sha256_file(request.golden_path),
                    config=str(request.config),
                    pipeline=asdict(pipeline),
                    release_id=release_id,
                ),
                commit=git_commit(Path(__file__).resolve().parent),
            )
            executor = BackendTurnExecutor(
                graph=build_chat_graph(),
                llm=app.deps.llm,
                retrieval=app.retrieval.service,
                limits=settings.budget,
                pipeline=pipeline,
                config=str(request.config),
            )
            return runner.run(
                run_items(
                    items,
                    executor,
                    JsonlStore(directory / ANSWERS_FILE, AnswerRecord),
                    config=str(request.config),
                    concurrency=request.concurrency,
                    retry_errors=request.retry_errors,
                )
            )
        finally:
            runner.run(app.aclose())
