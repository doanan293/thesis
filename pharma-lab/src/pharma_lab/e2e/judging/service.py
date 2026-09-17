"""Judge every answer of one run/config (spec §7)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pharma_agent.infrastructure.settings import Settings

from pharma_lab.e2e.configs import E2EConfig
from pharma_lab.e2e.golden import Category, ExpectedBehavior, GoldenItem, load_golden
from pharma_lab.e2e.harness import ANSWERS_FILE, config_dir
from pharma_lab.e2e.judging.code_metrics import context_blocks
from pharma_lab.e2e.judging.ragas_scorer import RagasScores, build_ragas_scorer
from pharma_lab.e2e.judging.structured import (
    JUDGE_MODEL,
    JUDGE_REASONING,
    StructuredJudge,
    judge_llm,
)
from pharma_lab.e2e.records import AnswerRecord, JsonlStore, Judgement
from pharma_lab.evaluation.artifact_contracts import write_json

JUDGMENTS_FILE = "judgments.jsonl"
JUDGE_FILE = "judge.json"


class Scorer(Protocol):
    async def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None,
    ) -> RagasScores: ...


async def judge_record(
    item: GoldenItem, record: AnswerRecord, judge: StructuredJudge, ragas: Scorer
) -> Judgement:
    base = Judgement(
        item_id=item.item_id,
        config=record.config,
        category=item.category.value,
    )
    try:
        injection_followed = (
            await judge.injection(item, record.answer_text)
            if item.category is Category.INJECTION
            else None
        )
        declined = (
            await judge.abstention(item, record.answer_text)
            if item.category is Category.UNANSWERABLE
            and record.answer_mode != ExpectedBehavior.ABSTAIN.value
            else None
        )
        # Behaviour and citation scores need no LLM; `e2e report` computes them.
        update: dict[str, object] = {
            "injection_followed": injection_followed,
            "declined": declined,
        }
        if item.expected_behavior is ExpectedBehavior.GROUNDED:
            if record.answer_mode == ExpectedBehavior.GROUNDED.value:
                verdicts = await judge.key_facts(item, record.answer_text)
                update |= {
                    "key_fact_verdicts": verdicts,
                    "key_fact_recall": verdicts.count("supported") / len(verdicts),
                    "contradiction": "contradicted" in verdicts,
                    "citation_support": await judge.citation_support(
                        record.answer_text, record.context_text
                    ),
                }
                scores = await ragas.score(
                    question=item.question,
                    answer=record.answer_text,
                    contexts=list(context_blocks(record.context_text).values()),
                    reference=item.reference.answer or None,
                )
                update |= {
                    "faithfulness": scores.faithfulness,
                    "factual_correctness": scores.factual_correctness,
                    "answer_relevancy": scores.answer_relevancy,
                }
            else:
                # No answer was given: nothing of the reference is covered.
                update |= {"key_fact_recall": 0.0, "contradiction": False}
        return base.model_copy(update=update)
    except Exception as error:  # one bad item must not stop the judging run
        return base.model_copy(
            update={"error": f"{type(error).__name__}: {error}"[:500]}
        )


@dataclass(frozen=True)
class JudgeRequest:
    run_root: Path
    config: E2EConfig
    golden_path: Path
    backend_env_file: Path
    concurrency: int = 4
    force: bool = False
    judge_model: str = JUDGE_MODEL
    # Items to judge again even when they already have a judgement.
    only: frozenset[str] = frozenset()


def pin_judge(directory: Path, *, model: str, force: bool) -> None:
    """All judgments of one configuration come from one judge model."""
    path = directory / JUDGE_FILE
    current = {"model": model, "reasoning_effort": JUDGE_REASONING}
    if path.is_file() and not force:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != current:
            raise ValueError(
                f"{directory} was judged with {stored}; pass --force to judge "
                f"everything again with {current}"
            )
        return
    write_json(path, current)


@dataclass(frozen=True)
class JudgeSummary:
    total: int
    judged: int
    errors: int


async def judge_records(
    items: dict[str, GoldenItem],
    answers: JsonlStore[AnswerRecord],
    judgments: JsonlStore[Judgement],
    judge: StructuredJudge,
    ragas: Scorer,
    *,
    concurrency: int,
    force: bool,
    only: frozenset[str] = frozenset(),
) -> JudgeSummary:
    records = answers.latest()
    retryable = sorted(k for k, record in records.items() if record.retryable)
    if retryable:
        raise ValueError(
            f"{len(retryable)} answers failed (e.g. {retryable[0]}); "
            "re-run `pharma-lab e2e run --retry-errors` first"
        )
    unknown = sorted(set(records) - set(items))
    if unknown:
        raise ValueError(f"answers for items outside the golden set: {unknown[:3]}")
    done = judgments.latest()
    pending = [
        record
        for key, record in sorted(records.items())
        if force or key in only or key not in done or done[key].error is not None
    ]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(record: AnswerRecord) -> None:
        async with semaphore:
            judgement = await judge_record(items[record.item_id], record, judge, ragas)
        async with lock:
            judgments.append(judgement)

    await asyncio.gather(*(one(record) for record in pending))
    final = judgments.latest()
    return JudgeSummary(
        total=len(records),
        judged=len(pending),
        errors=sum(1 for judgement in final.values() if judgement.error is not None),
    )


def run_judge(request: JudgeRequest) -> JudgeSummary:
    if request.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    directory = config_dir(request.run_root, request.config)
    answers_path = directory / ANSWERS_FILE
    if not answers_path.is_file():
        raise ValueError(f"no answers to judge: {answers_path}")
    settings = Settings(_env_file=request.backend_env_file)
    items = {item.item_id: item for item in load_golden(request.golden_path)}
    pin_judge(directory, model=request.judge_model, force=request.force)
    return asyncio.run(
        judge_records(
            items,
            JsonlStore(answers_path, AnswerRecord),
            JsonlStore(directory / JUDGMENTS_FILE, Judgement),
            StructuredJudge(judge_llm(settings, request.judge_model)),
            build_ragas_scorer(settings, request.judge_model),
            concurrency=request.concurrency,
            force=request.force,
            only=request.only,
        )
    )
