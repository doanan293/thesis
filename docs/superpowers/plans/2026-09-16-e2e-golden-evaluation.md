# E2E Golden Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the whole pharma agent end to end on a frozen 500-item golden set, comparing the full agent with one-step RAG and three ablations, scored by RAGAS plus a structured gpt-5-mini judge, calibrated on 100 blind items.

**Architecture:** The backend gains `PipelineOptions` (rephrase, judge/refine), which only code can pass; production always runs the full graph. A new package `pharma_lab.e2e` samples and validates the golden set, runs the backend chat graph in-process per item with resumable JSONL records, judges the answers, exports and scores calibration, and writes CSV/LaTeX reports.

**Tech Stack:** Python 3.12, uv workspace, LangGraph (backend), pydantic, typer, RAGAS 0.4.3, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md`

## Global Constraints

- One uv workspace and one `uv.lock`. RAGAS lives in pharma-lab's dev group: `ragas==0.4.3`, `langchain-community==0.4.1`, `scipy==1.18.1`. The root `pyproject.toml` has `[tool.uv] override-dependencies = ["jiter>=0.16.0,<1"]`.
- pharma-lab dependencies are exact pins (`tests/test_dependency_pins.py`).
- No setting or environment variable for `PipelineOptions`. `build_application` always builds the default (full) pipeline.
- Judge model is `gpt-5-mini` with `reasoning_effort="medium"`. The endpoint (base_url, api_key) comes from `backend/.env` (`PHARMA_LLM__DEFAULT__*`).
- Harness disables Langfuse.
- Golden set: 420 answerable (70 × 6 eval groups), 50 multi_turn, 10 unanswerable, 10 out_of_scope, 10 injection. Schema `e2e-golden-v1`.
- Configs: `full`, `one-step`, `no-judge-refine`, `no-rephrase`, `no-rerank`.
- Lint and type findings are fixed in code, never suppressed. pytest runs with `filterwarnings = ["error"]`.
- Stage explicit paths only; never `git add -A`; never `--no-verify`. Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Long runs (full E2E runs, judging) go in tmux, one session per job.
- Gates per project: `uv run ruff check . && uv run ruff format --check . && uv run pyrefly check --min-severity warn && uv run pytest -q`.

## File Structure

Backend (`backend/src/pharma_agent/`):
- `domain/agent/run.py`: add `AgentRun.skip_judge`.
- `application/chat/context.py`: add `PipelineOptions`, `TurnContext.pipeline`.
- `application/chat/nodes.py`: rephrase and judge nodes honour options; answer node returns `context_text`.
- `application/chat/state.py`: `ChatTurnState.context_text`.
- `application/chat/runner.py`: `ChatTurnRunner(pipeline=...)`, `TurnOutcome.context_text`.

pharma-lab (`pharma-lab/src/pharma_lab/`):
- `config/paths.py`: E2E paths.
- `e2e/golden.py`: golden item models, loading, validation, manifest.
- `e2e/corpus_text.py`: section and title texts from the bundle.
- `e2e/sampling.py`: answerable and multi-turn source selection, authoring batches.
- `e2e/configs.py`: E2E configs and settings derivation.
- `e2e/records.py`: `AnswerRecord`, `Judgement`, `JsonlStore`.
- `e2e/run_identity.py`: identity of one run/config directory.
- `e2e/recording_llm.py`: per-role usage counter around `LlmPort`.
- `e2e/executor.py`: one golden item through the backend graph.
- `e2e/harness.py`: resumable concurrent run.
- `e2e/judging/code_metrics.py`: citation and behaviour metrics.
- `e2e/judging/structured.py`: key-fact, citation-support and injection judges.
- `e2e/judging/ragas_scorer.py`: RAGAS metrics.
- `e2e/judging/service.py`: judge a run/config.
- `e2e/calibration.py`: blind export, agreement statistics.
- `e2e/report.py`: tables, CSV, LaTeX, error analysis.
- `cli/commands/e2e.py`: `pharma-lab e2e ...`.

Tests under `pharma-lab/tests/e2e/`, backend tests next to existing ones.

---

### Task 1: Backend — `AgentRun.skip_judge`

**Files:**
- Modify: `backend/src/pharma_agent/domain/agent/run.py`
- Test: `backend/tests/domain/test_agent_run.py`

**Interfaces:**
- Produces: `AgentRun.skip_judge(self, *, now: datetime) -> None`

- [ ] **Step 1: Write the failing tests**

```python
def test_skipped_judge_answers_from_one_search_round() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(None, now=NOW, skipped=True)
    run.record_search(Q1, search_result("c1"), now=NOW)
    run.skip_judge(now=NOW)
    assert run.allowed_steps() == {Step.ANSWER}
    assert run.actions.entries[-1].outcome == "skipped"
    assert run.decide_answer() == AnswerPlan(mode=AnswerMode.GROUNDED, partial=False)


def test_skipped_judge_without_evidence_abstains() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_search(Q1, search_result(), now=NOW)
    run.skip_judge(now=NOW)
    assert run.decide_answer() == AnswerPlan(mode=AnswerMode.ABSTAIN)
```

- [ ] **Step 2:** `cd backend && uv run pytest tests/domain/test_agent_run.py -q` → FAIL (`skip_judge` missing).
- [ ] **Step 3: Implement** after `record_judge`:

```python
    def skip_judge(self, *, now: datetime) -> None:
        """Answer from the evidence found so far without asking the judge."""
        self.last_judge = JudgeOutcome.ANSWER
        self._log(ActionKind.JUDGE, now, outcome="skipped")
```

- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** commit `feat(agent): let a run answer without judging`.

### Task 2: Backend — `PipelineOptions` and `context_text`

**Files:**
- Modify: `backend/src/pharma_agent/application/chat/{context,nodes,state,runner}.py`
- Test: `backend/tests/application/test_chat_graph.py`, `backend/tests/infrastructure/test_composition.py`

**Interfaces:**
- Consumes: `AgentRun.skip_judge`.
- Produces:
  - `PipelineOptions(rephrase: bool = True, judge_refine: bool = True)` (frozen dataclass) in `pharma_agent.application.chat.context`;
  - `TurnContext.pipeline: PipelineOptions = PipelineOptions()`;
  - `ChatTurnRunner(graph, deps, limits, tracer=None, pipeline: PipelineOptions | None = None)` with public attribute `pipeline`;
  - `ChatTurnState.context_text: str = ""`; `TurnOutcome.context_text: str = ""`.

- [ ] **Step 1: Write failing tests** in `test_chat_graph.py`. Extend `run_turn` with `pipeline: PipelineOptions | None = None` passed to `ChatTurnRunner`. Add:

```python
async def test_one_step_pipeline_calls_guard_and_answer_only() -> None:
    llm = FakeLlm()
    llm.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    retriever = FakeRetriever([make_hit("c1", fusion=0.9)])

    events, outcome = await run_turn(
        llm, retriever, pipeline=PipelineOptions(rephrase=False, judge_refine=False)
    )

    assert outcome.run.status is RunStatus.COMPLETED
    assert retriever.calls[0][0].text == QUESTION
    assert llm.calls_for(LlmRole.REPHRASE) == []
    assert llm.calls_for(LlmRole.JUDGE) == []
    assert events[-1].data["usage"]["llm_calls"] == 2
    assert len(retriever.calls) == 1


async def test_no_rephrase_keeps_the_judge() -> None:
    llm = FakeLlm()
    llm.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    retriever = FakeRetriever([make_hit("c1", fusion=0.9)])

    _, outcome = await run_turn(
        llm, retriever, pipeline=PipelineOptions(rephrase=False)
    )

    assert retriever.calls[0][0].text == QUESTION
    assert len(llm.calls_for(LlmRole.JUDGE)) == 1
    assert outcome.run.status is RunStatus.COMPLETED


async def test_no_judge_refine_keeps_the_rephrase() -> None:
    llm = passing_llm()
    retriever = FakeRetriever([make_hit("c1", fusion=0.9)])

    _, outcome = await run_turn(
        llm, retriever, pipeline=PipelineOptions(judge_refine=False)
    )

    assert retriever.calls[0][0].text == "Liều paracetamol cho người lớn"
    assert llm.calls_for(LlmRole.JUDGE) == []
    assert outcome.run.status is RunStatus.COMPLETED
```

In `test_grounded_answer_in_one_round` add `assert outcome.context_text.startswith("[1] Paracetamol > Liều dùng")`. In `test_composition.py::test_build_application_wires_real_adapters` add `assert app.runner.pipeline == PipelineOptions()`.

- [ ] **Step 2:** run the two test files → FAIL.
- [ ] **Step 3: Implement.**
  - `context.py`: add the frozen dataclass and the `pipeline` field (last, with default).
  - `rephrase_node`: `if not runtime.context.pipeline.rephrase or not run.can_afford_rephrase(): run.record_rephrase(None, now=now, skipped=True); return {"run": run}`.
  - `judge_node`: at the top, `if not runtime.context.pipeline.judge_refine: run.skip_judge(now=now); return {"run": run, "last_gaps": []}`.
  - `answer_node`: add `"context_text": context_text` to the returned update.
  - `state.py`: `context_text: str = ""`.
  - `runner.py`: `ChatTurnExecution` takes `pipeline` and builds `TurnContext(deps=..., conversation=..., pipeline=self._pipeline)`. `_finish` takes `context_text: str` and stores it in `TurnOutcome`; failure paths pass `""`. `ChatTurnRunner.__init__` stores `self.pipeline = pipeline or PipelineOptions()` and passes it to executions.
- [ ] **Step 4:** full backend gates → PASS.
- [ ] **Step 5:** commit `feat(chat): add pipeline options for evaluation ablations`.

### Task 3: Workspace — RAGAS in pharma-lab

**Files:**
- Modify: `pyproject.toml`, `pharma-lab/pyproject.toml`, `uv.lock`
- Test: `pharma-lab/tests/e2e/__init__.py`, `pharma-lab/tests/e2e/test_ragas_dependency.py`

- [ ] **Step 1: Failing test**

```python
from importlib.metadata import version


def test_ragas_imports_cleanly_with_the_workspace_openai() -> None:
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        FactualCorrectness,
        Faithfulness,
    )

    assert version("ragas") == "0.4.3"
    assert version("openai").startswith("3.")
    assert all(
        callable(item)
        for item in (
            llm_factory,
            AnswerRelevancy,
            FactualCorrectness,
            Faithfulness,
            OpenAIEmbeddings,
        )
    )
```

- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3:** add the override in the root `pyproject.toml` under `[tool.uv]` with a comment naming the instructor/jiter cap; add the three pins to pharma-lab's dev group (alphabetical); `uv lock && uv sync`.
- [ ] **Step 4:** pharma-lab and backend gates → PASS; check `git diff uv.lock` only adds packages plus `rich` 14.3.4.
- [ ] **Step 5:** commit `build(lab): add RAGAS with a jiter override`.

### Task 4: Golden set model, corpus text and `e2e golden build`

**Files:**
- Modify: `pharma-lab/src/pharma_lab/config/paths.py`, `pharma-lab/.gitignore`, `pharma-lab/tests/test_repository_data_policy.py`, `pharma-lab/src/pharma_lab/cli/app.py`
- Create: `pharma-lab/src/pharma_lab/e2e/__init__.py`, `e2e/golden.py`, `e2e/corpus_text.py`, `cli/commands/e2e.py`
- Test: `pharma-lab/tests/e2e/test_golden.py`

**Interfaces:**
- Produces:
  - paths: `E2E_DIR = EVALUATION_DIR / "e2e"`, `GOLDEN_E2E_PATH = E2E_DIR / "golden_e2e.jsonl"`, `E2E_AUTHORING_DIR = E2E_DIR / "authoring"`, `E2E_RUNS_DIR = E2E_DIR / "runs"`, `e2e_run_dir(run: str) -> Path`.
  - `corpus_text.CorpusText` (frozen dataclass: `sections: dict[str, str]`, `section_documents: dict[str, str]`, `titles: dict[str, str]`) with `load_corpus_text(bundle_dir: Path) -> CorpusText`, `normalize_space(text: str) -> str`, `contains(self, term: str) -> bool` (case-insensitive over all section texts and titles).
  - `golden.Category`, `golden.ExpectedBehavior` (StrEnum), `golden.Turn`, `golden.KeyFact`, `golden.Reference`, `golden.GoldenItem` (pydantic, `extra="forbid"`), `GoldenItem.question -> str` (last user turn), `GoldenItem.history -> list[tuple[str, str]]` (user/assistant pairs before it).
  - `golden.QUOTAS: dict[Category, int]`, `golden.ANSWERABLE_PER_GROUP = 70`, `golden.EVAL_GROUPS` (six names).
  - `golden.load_golden(path: Path) -> list[GoldenItem]`, `golden.validate_items(items, corpus, *, complete: bool) -> list[str]`, `golden.build_golden(sources: Sequence[Path], output: Path, corpus: CorpusText, *, bundle_manifest_sha256: str, gold_sha256: str) -> GoldenManifest`.

Validation rules (one test each):
1. `evidence_quote` (normalised) is a substring of the normalised text of `key_fact.section_id`.
2. `key_fact.section_id` is in `gold_section_ids`.
3. `grounded` items have ≥1 key fact and ≥1 gold section; `abstain`/`redirect`/`blocked` items have none.
4. Category → behaviour: answerable and multi_turn are `grounded`; unanswerable `abstain`; out_of_scope `redirect`; injection `blocked`.
5. multi_turn items have ≥3 turns alternating user/assistant ending with user; other categories exactly one user turn.
6. unanswerable items have non-empty `absent_terms` and `corpus.contains(term)` is false for each.
7. `item_id` unique and prefixed per category (`e2e-ans-`, `e2e-mt-`, `e2e-una-`, `e2e-oos-`, `e2e-inj-`).
8. With `complete=True`: category counts equal `QUOTAS` and answerable items have exactly 70 per eval group.

`build_golden` reads every source JSONL (authored batches), validates with `complete=True`, sorts by `item_id`, writes the output atomically plus `golden_e2e.manifest.json` (`schema`, `sha256`, `counts`, `gold_sha256`, `bundle_manifest_sha256`, `created_at`) and raises `ValueError` listing all problems.

CLI: `pharma-lab e2e golden build [--sources DIR] [--output FILE] [--bundle DIR]`, default sources `E2E_AUTHORING_DIR/*.authored.jsonl`.

`.gitignore`: `data/evaluation/e2e/**`, re-include directories, `golden_e2e.manifest.json`, `runs/**/run.json` and `runs/**/reports/**`. Policy test rows: golden jsonl ignored, manifest tracked, `answers.jsonl` ignored, `reports/main.csv` tracked.

- [ ] Steps: failing tests (fixture bundle built in `tmp_path` with `write_bundle` from two sections) → FAIL → implement → PASS → gates → commit `feat(lab): validate and freeze the E2E golden set`.

### Task 5: Source sampling and `e2e golden sample`

**Files:**
- Create: `pharma-lab/src/pharma_lab/e2e/sampling.py`
- Modify: `cli/commands/e2e.py`
- Test: `pharma-lab/tests/e2e/test_sampling.py`

**Interfaces:**
- Consumes: `load_query_rows`, `sample_quotas` from `pharma_lab.evaluation.backend_retrieval`; `CorpusText`.
- Produces:
  - `sample_answerable(rows, *, per_group: int, seed: int, corpus: CorpusText) -> list[dict]`: rows whose expected sections all exist; per eval group, quotas over the `(difficulty, answer_mode)` strata via `sample_quotas`; `random.Random(seed)`; file order.
  - `sample_multi_turn(rows, *, count: int, seed: int, corpus: CorpusText, exclude: set[str]) -> list[tuple[dict, dict]]`: pairs of `single` rows from different sections of the same document, each document used once, rows in `exclude` skipped.
  - `write_authoring_batches(output_dir, answerable, pairs, corpus, *, batch_size: int = 35) -> list[Path]`: `answerable-NN.todo.jsonl` and `multi-turn-NN.todo.jsonl`; each line has `slot_id` (the future `item_id`), `category`, `source_rows`, and `sections` (id → text) for the rows' expected sections.
- CLI: `pharma-lab e2e golden sample [--evaluation FILE] [--bundle DIR] [--seed 0]`, refuses to overwrite existing `.todo.jsonl` files unless `--force`.

- [ ] Steps: tests for deterministic sampling, per-group counts, strata proportionality, same-document pairs, exclusion → implement → gates → commit `feat(lab): sample sources for the E2E golden set`.

### Task 6: Author and freeze the golden set (operational)

- [ ] Run `uv run pharma-lab e2e golden sample`.
- [ ] Author each `.todo.jsonl` into `<name>.authored.jsonl` (one `GoldenItem` per slot): answerable items keep the gold query as the only turn; multi-turn items use the first row's query, a short reference answer for it, then a follow-up rewritten with a pronoun or ellipsis whose answer is in the second row's section; key facts quote the section text verbatim.
- [ ] Author `special.authored.jsonl` with 10 unanswerable (drugs absent from the corpus, with `absent_terms`), 10 out-of-scope and 10 injection items.
- [ ] Run `uv run pharma-lab e2e golden build` until it reports no problems.
- [ ] Commit the manifest and `.gitignore`-tracked files, then run `uv run pharma-lab data push --kaggle-account acc1 --message "Add the E2E golden set"` in tmux.

### Task 7: Configs, records, identity and usage recording

**Files:**
- Create: `e2e/configs.py`, `e2e/records.py`, `e2e/run_identity.py`, `e2e/recording_llm.py`
- Test: `tests/e2e/test_configs.py`, `test_records.py`, `test_run_identity.py`, `test_recording_llm.py`

**Interfaces:**
- `configs.E2EConfig` (StrEnum: `FULL="full"`, `ONE_STEP="one-step"`, `NO_JUDGE_REFINE="no-judge-refine"`, `NO_REPHRASE="no-rephrase"`, `NO_RERANK="no-rerank"`); `pipeline_for(config) -> PipelineOptions`; `settings_for(base: Settings, config) -> Settings` (Langfuse keys cleared; rerank protocol `none` for `NO_RERANK`).
- `records.AnswerRecord` (pydantic): `item_id`, `config`, `status`, `answer_mode: str | None`, `answer_text`, `context_text`, `citations: list[CitedSection]` (`index`, `chunk_version_id`, `chunk_id`, `section_id`), `standalone_query`, `search_queries: list[list[str]]`, `judge_outcomes: list[str]`, `llm_calls`, `usage_by_role: dict[str, TokenUsage]`, `latency_seconds`, `step_seconds: dict[str, float]`, `error: str | None`, `retryable: bool`.
- `records.Judgement` (pydantic): `item_id`, `config`, `category`, `behaviour_correct`, `injection_followed: bool | None`, `key_fact_verdicts: list[str] | None`, `key_fact_recall`, `contradiction: bool | None`, `citation_precision`, `citation_recall`, `citation_support`, `faithfulness`, `factual_correctness`, `answer_relevancy` (floats or `None`), `error: str | None`.
- `records.JsonlStore[T]`: `JsonlStore(path, model)`, `latest() -> dict[str, T]` (last line per `item_id` wins), `append(record) -> None` (append + flush + fsync).
- `run_identity.RunIdentity` (frozen dataclass): `golden_sha256`, `config`, `pipeline`, `release_id`, `retrieval` (dict), `rerank` (dict without secrets), `role_models` (role → `{model, reasoning_effort}`), `budget` (dict); `open_run(directory, identity, *, git_commit) -> None` writes `run.json` on first use and raises `ValueError("... start a new --run")` when any identity field differs (git commit is informational).
- `recording_llm.RecordingLlm(inner: LlmPort)`: implements `LlmPort`; `usage_by_role() -> dict[str, TokenUsage]` summing `LlmUsage` for structured calls and the final stream delta.

- [ ] Steps: tests → implement → gates → commit `feat(lab): add E2E configs, records and run identity`.

### Task 8: Executor, harness and `e2e run`

**Files:**
- Create: `e2e/executor.py`, `e2e/harness.py`
- Modify: `cli/commands/e2e.py`
- Test: `tests/e2e/test_executor.py`, `tests/e2e/test_harness.py`

**Interfaces:**
- `executor.conversation_for(item: GoldenItem) -> ConversationContext` (turns from history with `status="completed"`).
- `executor.answer_record(item, config, outcome: TurnOutcome, usage, latency) -> AnswerRecord`: citations mapped through `{e.hit.chunk_version_id: e.hit for e in outcome.run.evidence.active()}` to `gold_chunk_label(hit.section_key, hit.ordinal)` and `hit.section_key`; queries and judge outcomes from `outcome.run.actions.entries`; step seconds from consecutive action timestamps; `retryable` when run status is `error` or `timeout`.
- `executor.BackendTurnExecutor(graph, llm, retrieval, limits, pipeline, config)` with `async execute(item) -> AnswerRecord`: new `RecordingLlm` and `TurnDeps` per item, `ChatTurnRunner(..., pipeline=...)`, drain `events()`.
- `harness.TurnExecutor` Protocol (`async execute(item) -> AnswerRecord`).
- `harness.run_items(items, executor, store, *, config, concurrency, retry_errors) -> RunSummary` (`done`, `ran`, `errors`): skip items whose latest record exists and is not retryable (retryable ones only rerun with `retry_errors`); an exception becomes a retryable error record.
- `harness.run_e2e(request: E2ERunRequest) -> RunSummary`: load golden, `Settings(_env_file=backend_env_file)`, `settings_for`, `build_application(settings)`, probe release with `current_release_id(app.retrieval.service, first answerable question)`, `open_run`, then `run_items` with `BackendTurnExecutor(build_chat_graph(), app.deps.llm, app.retrieval.service, settings.budget, pipeline_for(config), config)`; `app.aclose()` at the end.
- CLI: `pharma-lab e2e run --run NAME --config CONFIG [--golden FILE] [--limit N] [--concurrency 4] [--retry-errors] [--backend-env-file FILE]`; status `incomplete` when errors remain.

Tests use a fake `TurnOutcome` built from backend domain objects (`AgentRun.start`, `record_search` with a `SearchResult` of `RetrievedItem`s) and a fake executor for resume, retry and concurrency.

- [ ] Steps: tests → implement → gates → commit `feat(lab): run the agent end to end on the golden set`.

### Task 9: Code metrics and structured judges

**Files:**
- Create: `e2e/judging/__init__.py`, `e2e/judging/code_metrics.py`, `e2e/judging/structured.py`
- Test: `tests/e2e/test_code_metrics.py`, `tests/e2e/test_structured_judge.py`

**Interfaces:**
- `code_metrics.behaviour_correct(item, record) -> bool`: `answer_mode == expected_behavior`, and for injection also accepts `grounded` (safe answer), leaving compliance to the injection judge.
- `code_metrics.citation_scores(item, record) -> tuple[float | None, float | None]`: precision = cited sections in gold / cited; recall = 1.0 if any cited section in gold else 0.0; `(None, None)` without gold sections; `(0.0, 0.0)` for grounded-expected items that cite nothing.
- `code_metrics.context_blocks(context_text) -> dict[int, str]` (split on lines starting `[n] `), `code_metrics.cited_sentences(answer_text) -> dict[int, list[str]]`.
- `structured` schemas: `FactCheck(index: int, verdict: Literal["supported", "missing", "contradicted"])`, `KeyFactJudgement(facts: list[FactCheck])`, `CitationCheck(citation: int, supported: bool)`, `CitationSupportJudgement(checks: list[CitationCheck])`, `InjectionJudgement(followed_injection: bool, reason: str)`.
- `structured.StructuredJudge(llm: LlmPort)` with `async key_facts(item, answer) -> list[str]`, `async citation_support(answer, context_text) -> float | None`, `async injection(item, answer) -> bool`; Vietnamese prompts; role `LlmRole.JUDGE`; a missing index in the model output counts as `missing` / unsupported.
- `structured.judge_llm(settings: Settings) -> OpenAiLlmAdapter` with `LlmSettings(default=LlmEndpoint(base_url, api_key, model="gpt-5-mini", reasoning_effort="medium"))`.

- [ ] Steps: tests with a fake `LlmPort` → implement → gates → commit `feat(lab): add code metrics and structured judges`.

### Task 10: RAGAS scorer, judge service and `e2e judge`

**Files:**
- Create: `e2e/judging/ragas_scorer.py`, `e2e/judging/service.py`
- Modify: `cli/commands/e2e.py`
- Test: `tests/e2e/test_ragas_scorer.py`, `tests/e2e/test_judge_service.py`

**Interfaces:**
- `ragas_scorer.RagasScores(faithfulness, factual_correctness, answer_relevancy)` (floats or `None`).
- `ragas_scorer.RagasScorer(faithfulness, factual, relevancy)` taking objects with `ascore`; `async score(*, question, answer, contexts: list[str], reference: str | None) -> RagasScores` (factual only with a reference).
- `ragas_scorer.build_ragas_scorer(settings: Settings) -> RagasScorer`: `AsyncOpenAI` from the backend LLM endpoint, `llm_factory("gpt-5-mini", client=..., reasoning_effort="medium")`; `OpenAIEmbeddings(client=AsyncOpenAI(base_url=embedding.base_url, api_key=embedding.api_key), model=embedding.model)`.
- `service.judge_record(item, record, judge, ragas) -> Judgement`: grounded answers get RAGAS scores (contexts = context blocks), key facts, citation support; injection items get the injection judge; exceptions become `error`.
- `service.run_judge(request: JudgeRequest) -> JudgeSummary`: reads `answers.jsonl`, skips items already judged without error (unless `--force`), concurrency semaphore, appends to `judgments.jsonl`, refuses when `answers.jsonl` has retryable records.
- CLI: `pharma-lab e2e judge --run NAME --config CONFIG [--concurrency 4] [--force]`.

Test `build_ragas_scorer` constructs real RAGAS objects without network, and `RagasScorer` with fake metrics.

- [ ] Steps: tests → implement → gates → commit `feat(lab): judge E2E answers with RAGAS and structured checks`.

### Task 11: Calibration

**Files:**
- Create: `e2e/calibration.py`
- Modify: `cli/commands/e2e.py`
- Test: `tests/e2e/test_calibration.py`

**Interfaces:**
- `export_calibration(run_root, *, seed=0, size=100) -> Path`: 50 items each from `full` and `one-step` (35 answerable/multi_turn + 15 special per config, sampled with `random.Random(seed)`), shuffled; writes `calibration/items.jsonl` (`blind_id`, `category`, `turns`, `reference`, `context_text`, `answer_text`, `citations` indexes only) and `calibration/key.json` (`blind_id` → config, item_id). No judge scores in items.
- Grades file `calibration/grades.jsonl`: `blind_id`, `grader`, `key_fact_verdicts`, `faithfulness`, `citation_checks: dict[int, bool]`, `injection_followed`.
- `cohen_kappa(a: Sequence[str|bool], b) -> float`, `spearman(a, b) -> float` (`scipy.stats.spearmanr`), `bootstrap_interval(pairs, statistic, *, resamples, seed) -> tuple[float, float]`.
- `score_calibration(run_root) -> CalibrationReport` with rows (`metric`, `n`, `value`, `ci_low`, `ci_high`, `reliable`) for key-fact κ, citation-support κ, injection κ, key-fact-recall ρ, faithfulness ρ; `reliable` = κ ≥ 0.6 (ρ rows use the same threshold); writes `calibration/agreement.json`.
- CLI: `pharma-lab e2e calibration export --run NAME`, `pharma-lab e2e calibration score --run NAME`.

- [ ] Steps: tests with known values (perfect agreement κ=1, chance-level example) → implement → gates → commit `feat(lab): add blind judge calibration`.

### Task 12: Report

**Files:**
- Create: `e2e/report.py`
- Modify: `cli/commands/e2e.py`
- Test: `tests/e2e/test_report.py`

**Interfaces:**
- `METRICS`: ordered metric names from `Judgement` plus `tokens_per_turn`, `llm_calls`, `latency_p50`, `latency_p95`.
- `mean_interval(values, *, resamples=10_000, seed=0) -> tuple[float, float, float]`.
- `write_report(run_root) -> list[Path]`: reads every config's answers and judgments, writes `reports/main.csv|tex` (config × metric, mean [CI]), `reports/ablation.csv|tex` (Δ vs `full` using `paired_bootstrap` on shared items), `reports/by_group.csv` (config × category/eval_group × metric), `reports/calibration.csv|tex` if `agreement.json` exists, and `reports/errors.md` (20 lowest `full` items by key-fact recall then faithfulness, with a retrieval/judge/answer label: retrieval when no cited or retrieved gold section, judge when gold was retrieved but not in the final context, otherwise answer).
- CLI: `pharma-lab e2e report --run NAME`.

- [ ] Steps: tests with two configs of tiny records → implement → gates → commit `feat(lab): write E2E evaluation reports`.

### Task 13: Docs, smoke run and full runs

- [ ] Document all `pharma-lab e2e` commands in `pharma-lab/docs/guides/evaluation.md` and `cli-reference.md` (update `tests/test_docs.py` expectations if it checks command lists).
- [ ] Smoke run (needs compose stack and LLM endpoint): `uv run pharma-lab e2e run --run e2e-smoke --config full --limit 20`, then `e2e judge`.
- [ ] Full runs in tmux, one session per config: `uv run pharma-lab e2e run --run e2e-v1 --config CONFIG`, then `e2e judge`, then calibration export, grading, `calibration score`, `report`.
- [ ] Commit run manifests and reports; push `dev` and the feature branch.
