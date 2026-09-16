# Remove Product Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Pharma AI Agent product Skills capability from runtime, schema, contracts, frontend, tests, and documentation without changing repository-local development skills.

**Architecture:** Collapse the graph to `guard -> rephrase -> search <-> (judge -> refine) -> answer`, remove the unreferenced vertical slice, rewrite clean database history, regenerate the web contract, then purge feature documentation.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Pydantic, SQLAlchemy/Alembic, pytest, React Router 8, React 19, TypeScript 7, Vitest, Playwright, orval, Markdown.

**Spec:** `docs/superpowers/specs/2026-09-16-remove-product-skills-design.md`

## Global Constraints

- Preserve `.agents/skills/`, `.claude/skills/`, `skills-lock.json`, and development-process skill references.
- Leave no feature flag, deprecated alias, compatibility type, empty package, dead adapter, or drop migration.
- Existing databases must be recreated from the rewritten migration chain.
- Old trace/UI-message compatibility is not required.
- Regenerate OpenAPI and orval outputs; do not hand-edit generated files.
- Delete this design and plan in the final documentation cleanup.

---

### Task 1: Remove selection from the agent core

**Files:**
- Modify: `backend/src/pharma_agent/application/chat/{context,graph,nodes,routing}.py`
- Modify: `backend/src/pharma_agent/application/progress.py`
- Modify: `backend/src/pharma_agent/domain/agent/{actions,prompts,run,schemas}.py`
- Modify: `backend/src/pharma_agent/domain/llm/models.py`
- Modify: `backend/src/pharma_agent/infrastructure/settings.py`
- Modify: `backend/tests/{fakes,memory_repository}.py`
- Test: `backend/tests/application/test_chat_graph.py`
- Test: `backend/tests/domain/{test_agent_run,test_prompts}.py`

**Interfaces:**
- Consumes: `TurnDeps(llm, guardrail, retrieval, skills, clock)`.
- Produces: `TurnDeps(llm, guardrail, retrieval, clock)` and direct `rephrase -> search` routing.

- [ ] **Step 1: Rewrite tests for the target graph and budget**

Use these grounded-path assertions and remove all selector scripting:

```python
assert phases(events) == [
    Phase.GUARDING,
    Phase.UNDERSTANDING,
    Phase.SEARCHING,
    Phase.READING,
    Phase.ANSWERING,
]
assert done.data["usage"]["llm_calls"] == 4
assert [role for role, _ in llm.calls] == [
    LlmRole.GUARDRAIL,
    LlmRole.REPHRASE,
    LlmRole.JUDGE,
    LlmRole.ANSWER,
]
```

Replace the optional-step assertion with:

```python
run = make_run(max_llm_calls=4)
assert run.can_afford_rephrase() is True
run.charge(LlmUsage())
run.charge(LlmUsage())
assert run.can_afford_rephrase() is False
```

Remove skill fixtures from prompt tests; assert judge still contains evidence and refine still contains used queries, gaps, and term hints.

- [ ] **Step 2: Confirm tests fail against the old core**

Run: `uv run --directory backend pytest tests/application/test_chat_graph.py tests/domain/test_agent_run.py tests/domain/test_prompts.py -q`

Expected: failures mention the selection phase/fifth call, missing `can_afford_rephrase`, or old feature imports.

- [ ] **Step 3: Implement the reduced graph and state**

```python
@dataclass(frozen=True)
class TurnDeps:
    llm: LlmPort
    guardrail: GuardrailService
    retrieval: RetrievalService
    clock: Clock
```

```python
def route_after_rephrase(
    state: ChatTurnState,
) -> Literal["search", "answer", "fallback"]:
    if _failed(state):
        return FALLBACK
    return "answer" if Step.ANSWER in state.run.allowed_steps() else "search"
```

Delete `resolve_skills_node`, `route_after_resolve_skills`, its graph node/edge,
selection phase/event/action/error, `SelectedSkill`, `AgentRun.skills`,
`record_skills`, trace field, `SkillSelection`, `LlmRole.SKILL_SELECTOR`,
`_skill_block`, selector prompt/builder, and selector settings.

Replace the optional enum/method with:

```python
RESERVED_CALLS = 3


def can_afford_rephrase(self) -> bool:
    return self.usage.llm_calls + RESERVED_CALLS <= self.limits.max_llm_calls
```

- [ ] **Step 4: Reduce fakes and verify**

Delete `FakeSkillCatalog`, product skill markdown, factory, imports/exports, and
the `catalog` argument. Construct `TurnDeps` with only llm, guardrail, retrieval,
and clock. Run `rg -n 'MemorySkillRepository' backend/tests`; remove the class
and imports when the only matches are its definition and feature-specific tests
deleted in Task 2.

Run: `uv run --directory backend pytest tests/application/test_chat_graph.py tests/domain/test_agent_run.py tests/domain/test_prompts.py -q`

Run: `uv run --directory backend ruff check src/pharma_agent/application src/pharma_agent/domain tests/application tests/domain tests/fakes.py`

- [ ] **Step 5: Commit**

```bash
git add backend/src/pharma_agent/application backend/src/pharma_agent/domain backend/src/pharma_agent/infrastructure/settings.py backend/tests
git commit -m "refactor(agent): remove product skill selection"
```

### Task 2: Remove backend API and runtime feature

**Files:**
- Delete: `backend/skills/`
- Delete: `backend/src/pharma_agent/{domain,application}/skill/`
- Delete: `backend/src/pharma_agent/infrastructure/skills/`
- Delete: `backend/src/pharma_agent/infrastructure/persistence/postgres/skill_repository.py`
- Delete: `backend/src/pharma_agent/api/routers/skills.py`
- Delete: dedicated skill tests under `backend/tests/{api,application,domain,infrastructure}`
- Modify: `backend/src/pharma_agent/infrastructure/{composition,container}.py`
- Modify: `backend/src/pharma_agent/api/{app,deps,errors,schemas,ui_stream}.py`
- Modify: `backend/src/pharma_agent/application/conversation/ui_message.py`
- Modify: `backend/src/pharma_agent/cli.py`
- Modify: related API, stream, container, CLI, E2E, and contract tests/fixtures

**Interfaces:**
- Consumes: Task 1 core.
- Produces: no `/api/v1/skills`; data parts contain phase/evidence/conversation; no `data-skills`.

- [ ] **Step 1: Add API absence tests**

Remove four Skills operation IDs and add:

```python
def test_product_skills_are_absent_from_openapi() -> None:
    doc = document()
    assert "/api/v1/skills" not in doc["paths"]
    assert "/api/v1/skills/{name}" not in doc["paths"]
    assert not {
        "SkillView",
        "SkillRef",
        "SkillsData",
        "EnableSkillRequest",
        "Body_upload_skill",
    } & set(doc["components"]["schemas"])
```

Remove `SkillRef`/`SkillsData` from UI schema expectations, remove the selected
event/chunk from stream tests, set usage to four calls, and assert the container
has feedback but no Skills service.

- [ ] **Step 2: Confirm old surface fails absence tests**

Run: `uv run --directory backend pytest tests/api/test_openapi.py tests/api/test_ui_stream.py tests/infrastructure/test_container.py -q`

- [ ] **Step 3: Remove runtime and HTTP wiring**

Build dependencies as:

```python
deps = TurnDeps(
    llm=llm_port,
    guardrail=GuardrailService(llm_port),
    retrieval=retrieval.service,
    clock=SystemClock(),
)
```

Remove catalog parameters/construction, `Container.skills`, repository/service,
startup sync, router inclusion, dependency getter, error mappings, and schemas.
Delete all dedicated source/test paths listed above.

- [ ] **Step 4: Remove stream, CLI, and fixture contracts**

Delete `SkillRef`, `SkillsData`, `PharmaDataParts.skills`, the SSE encoder arm,
and CLI branch. Remove selector scripts and `data-skills` frames from chat/API/
E2E helpers and all SSE fixtures; decrement applicable LLM-call counts.

- [ ] **Step 5: Verify and commit**

```bash
uv run --directory backend pytest -q
uv run --directory backend ruff check .
uv run --directory backend pyrefly check
git add -A backend/skills backend/src backend/tests
git commit -m "refactor(backend): remove product skills surface"
```

### Task 3: Rewrite schema history and dependencies

**Files:**
- Delete: migrations `0002_skills.py`, `0004_agent_skills_spec.py`
- Modify: migrations `0003_feedback.py`, `0005_corpus_schema.py`
- Modify: PostgreSQL `tables.py`, `metadata.py`, and schema tests
- Modify: `backend/pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `backend/.env.example`

**Interfaces:**
- Consumes: Task 2 persistence removal.
- Produces: chain `0001 -> 0003 -> 0005 -> ... -> 0009`; no table/dependency/hook.

- [ ] **Step 1: Require absence in metadata tests**

Assert the exact retained public table set and migration-file absence:

```python
assert public == {
    "access_tokens",
    "conversations",
    "feedback",
    "message_citations",
    "messages",
    "oauth_account",
    "retrieval_hits",
    "retrieval_runs",
    "user",
}
assert "skills" not in target_metadata.tables
names = {path.name for path in versions_path.glob("*.py")}
assert "0002_skills.py" not in names
assert "0004_agent_skills_spec.py" not in names
```

- [ ] **Step 2: Confirm schema tests fail**

Run: `uv run --directory backend pytest tests/infrastructure/test_persistence_metadata.py tests/api/test_e2e_postgres.py -q`

- [ ] **Step 3: Delete ORM schema and reconnect revisions**

Delete pattern/table/metadata imports and both migrations. Set:

```python
# 0003_feedback.py
down_revision: str | None = "0001"

# 0005_corpus_schema.py
down_revision: str | None = "0003"
```

- [ ] **Step 4: Remove dependency, hook, env examples; refresh lock**

Remove `skills-ref`, `validate-agent-skills`, and selector comments/examples.
Run: `uv lock`

Expected: `skills-ref` leaves `uv.lock`; `skills-lock.json` is untouched.

- [ ] **Step 5: Verify clean upgrade and commit**

```bash
uv run --directory backend pytest -m integration tests/infrastructure/test_persistence_metadata.py tests/api/test_e2e_postgres.py -q
git add -A .pre-commit-config.yaml backend/pyproject.toml backend/.env.example backend/src/pharma_agent/infrastructure/persistence backend/tests/infrastructure backend/tests/api/test_e2e_postgres.py uv.lock
git commit -m "refactor(database): erase product skills schema"
```

### Task 4: Regenerate contract and remove frontend feature

**Files:**
- Regenerate: `frontend/openapi.json`, `frontend/app/api/gen/`
- Delete: `frontend/app/features/skills/`, Skills route/locales/E2E file
- Modify: routes, sidebar, i18n, landing, chat message schema/view/component and tests
- Modify: auth/query-client tests and `frontend/tests/chat` fixtures/handler

**Interfaces:**
- Consumes: Tasks 2–3 OpenAPI/stream.
- Produces: no generated symbol, route, nav, landing claim, chat part, or locale namespace.

- [ ] **Step 1: Rewrite frontend tests for absence**

```tsx
await expect.element(page.getByRole("link", { name: "Cài đặt" })).toHaveAttribute("href", "/settings")
expect(page.getByRole("link", { name: "Kỹ năng" }).query()).toBeNull()
```

Remove `SkillsData` schema tests, the `data-skills` message part, badge assertion,
`/skills` auth cases, and Skills query-key test. Assert `"skills" in view` is false.

- [ ] **Step 2: Regenerate contract**

```bash
npm --prefix frontend run api:openapi
npm --prefix frontend run api:generate
```

Expected: clean mode deletes generated endpoints/types; handwritten imports fail until Step 3.

- [ ] **Step 3: Remove feature route, navigation, and copy**

Delete feature/route/E2E/locale files. Use:

```tsx
const NAV_ITEMS = [
  { to: "/chat", label: "nav.chat", Icon: MessageSquareIcon },
  { to: "/settings", label: "nav.settings", Icon: SettingsIcon },
] as const
```

Remove the namespace, nav/error/shell strings. Replace landing key with:

```tsx
const FEATURES = ["citations", "streaming", "grounded"] as const
```

Vietnamese: `Bám sát tài liệu` — `Agent chỉ trả lời phần có bằng chứng và nói rõ khi tài liệu chưa đủ.` English: `Grounded answers` — `The agent answers only what the evidence supports and says when the sources are insufficient.`

- [ ] **Step 4: Remove the chat stream part**

Set phases to guarding, understanding, searching, reading, answering. Delete
the skill data schema/import/view field/branch, selection phase key, badge list,
unused import, and fixture/handler branches.

- [ ] **Step 5: Verify and commit**

```bash
npm --prefix frontend run format:write
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend run test:unit
npm --prefix frontend run test:browser
npm --prefix frontend run api:check
npm --prefix frontend run e2e -- tests/e2e/auth.e2e.ts
git add -A frontend
git commit -m "refactor(frontend): remove product skills experience"
```

### Task 5: Rewrite README and thesis report

**Files:**
- Modify: `backend/README.md`, `frontend/README.md`, `report/report.md`
- Inspect without changing unless a product marker is found: `README.md`, `seed-pipeline/README.md`

**Interfaces:**
- Consumes: final code facts.
- Produces: current documentation with no product feature claim.

- [ ] **Step 1: Inventory current-document matches**

Run: `rg -n -i 'resolve_skills|skill_selector|selecting_skills|data-skills|SKILL\.md|/skills|\bskills?\b' README.md backend/README.md frontend/README.md seed-pipeline/README.md report/report.md`

- [ ] **Step 2: Rewrite README facts**

Remove the stream row, endpoint/startup-sync/directory instructions, upload
feature, and old E2E command. List only current phases/routes. Use
`npx playwright test tests/e2e/auth.e2e.ts` as the frontend example.

- [ ] **Step 3: Rewrite report architecture/evaluation**

Delete the selection node/row, feature row, and ablation factor. Reduce LLM-call
calculations by one and verify against tests. Use:

```text
guard -> rephrase -> search -> judge --(đủ evidence)--> answer[grounded]
                         ^       |
                         |       +--(thiếu evidence)--> refine
                         +-------------------------------+
```

- [ ] **Step 4: Audit and commit**

Run the Step 1 command again; expect no product match. Then:

```bash
git diff --check
git add README.md backend/README.md frontend/README.md seed-pipeline/README.md report/report.md
git commit -m "docs: remove product skills from current documentation"
```

### Task 6: Purge historical product documentation

**Files:**
- Modify: affected backend/frontend/seed-pipeline design specs
- Delete or rewrite: affected historical implementation plans
- Delete last: this plan and its design spec

**Interfaces:**
- Consumes: final architecture.
- Produces: retained docs contain development workflow only, not product history.

- [ ] **Step 1: Build authoritative inventory**

Run: `rg -l -i 'resolve_skills|skill_selector|selecting_skills|data-skills|skills_selected|SKILL\.md|backend/skills|/api/v1/skills|route\("skills"|\bproduct skills\b' backend/docs frontend/docs seed-pipeline/docs docs/superpowers | sort`

- [ ] **Step 2: Rewrite retained specs**

Make them agree: no selection node/role/table/upload/API/UI; graph routes from
rephrase to search; stream parts are phase/evidence/conversation; private routes
are chat/settings. In seed-data layout docs, remove the product skill file and
remove `backend/skills` from policy paths.

- [ ] **Step 3: Delete obsolete implementation-history files**

Delete plans where feature code/tasks are interwoven, including the backend
core/extras/platform/web-api/web-chat plans and frontend foundation/chat/
skills-settings plans. Retain a file whose only match is the standard
`REQUIRED SUB-SKILL` header; edit isolated product references instead of
deleting unrelated documents.

- [ ] **Step 4: Audit retained docs**

Run the Step 1 expression against `backend/docs frontend/docs seed-pipeline/docs`.
Expected: no match; generic development workflow wording is allowed.

- [ ] **Step 5: Delete temporary artifacts and commit**

```bash
git rm docs/superpowers/specs/2026-09-16-remove-product-skills-design.md docs/superpowers/plans/2026-09-16-remove-product-skills.md
git add -A backend/docs frontend/docs seed-pipeline/docs docs/superpowers
git commit -m "docs: purge product skills history"
```

Keep this task in session context until commit finishes.

### Task 7: Whole-repository acceptance audit

**Files:**
- Modify only files proven incomplete by checks below

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: verified removal with development skills preserved.

- [ ] **Step 1: Prove protected tooling remains**

```bash
test -d .agents/skills
test -e .claude/skills/kaggle-cli
test -e .claude/skills/latex-document-skill
test -f skills-lock.json
git diff d586837^ -- .agents .claude skills-lock.json
```

- [ ] **Step 2: Run marker audits**

```bash
rg -n -i --hidden --glob '!.git/**' --glob '!.venv/**' --glob '!node_modules/**' --glob '!.agents/**' --glob '!.claude/**' --glob '!skills-lock.json' 'resolve_skills|skill_selector|selecting_skills|data-skills|skills_selected|SKILL\.md|backend/skills|/api/v1/skills|route\("skills"|\bproduct skills\b' .
rg -n -i --hidden --glob '!.git/**' --glob '!.venv/**' --glob '!node_modules/**' --glob '!.agents/**' --glob '!.claude/**' --glob '!skills-lock.json' '\bskills?\b' .
```

Expected: narrow search is empty. Classify broad matches; retain only human
ability or development workflow wording.

- [ ] **Step 3: Run backend verification**

```bash
uv run --directory backend pytest -q
uv run --directory backend pytest -m integration -q
uv run --directory backend ruff check .
uv run --directory backend ruff format --check .
uv run --directory backend pyrefly check
```

- [ ] **Step 4: Run frontend verification**

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend run format
npm --prefix frontend run test
npm --prefix frontend run api:check
npm --prefix frontend run build
```

- [ ] **Step 5: Run hooks and inspect diff**

```bash
uv run pre-commit run --all-files
git diff --check
git status --short
git log --oneline d586837..HEAD
git diff --stat d586837^..HEAD
```

Expected: checks pass, worktree is clean, protected tooling is unchanged.

- [ ] **Step 6: Commit verification fixes if needed**

If checks changed tracked files, run `git add -u` and commit with
`fix: complete product skills removal audit`; otherwise create no empty commit.
Re-run the failed check and both marker audits.

- [ ] **Step 7: Record operational handoff**

Use this exact statement:

```text
Product Skills were removed from runtime, database history, API/stream contracts,
frontend, tests, and documentation. Existing databases must be recreated before
starting this revision; there is no compatibility migration for the former table.
```

Report passed commands and confirm protected development skill paths remain.
