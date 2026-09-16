# Remove Product Skills — Design

Date: 2026-09-16

## 1. Objective

Remove the Skills capability from the Pharma AI Agent as though it had never
been part of the product. The final repository must contain no product runtime,
API, database schema, frontend, tests, generated contract, README, report, or
historical architecture documentation that describes or depends on product
Skills.

This is intentionally a clean-history change. Existing databases are not
upgrade-compatible and must be reset after it lands.

## 2. Scope boundary

The removal covers only the Pharma AI Agent product feature: system and
user-uploaded `SKILL.md` files, skill selection during a chat turn, skill
instructions in prompts, skill persistence and management APIs, and all related
frontend behavior.

Repository-local development tooling is outside the feature and must remain:

- `.agents/skills/`
- `.claude/skills/`
- `skills-lock.json`
- references to development-process skills such as `superpowers:...`
- the Kaggle CLI and LaTeX document skills
- uses of the word “skills” that describe human abilities rather than the
  Pharma AI Agent feature

## 3. Target architecture

### 3.1 Agent flow

The chat graph becomes:

```text
guard -> rephrase -> search <-> (judge -> refine) -> answer
```

The edge after `rephrase` routes directly to retrieval. There is no selection
node, catalog dependency, selected-skill state, or optional LLM call between
rephrasing and search.

Judge, refine, and answer prompts use the question, conversation context,
retrieval evidence, gaps, and term hints already required by their respective
steps. They receive no external skill instructions.

### 3.2 Backend boundaries

Delete the complete Skills vertical slice:

- system product skills under `backend/skills/`
- domain models, parser, resolver, files loader, and ports
- application service for list/upload/enable/disable/delete and startup sync
- filesystem catalog and PostgreSQL repository
- Skills API router and request/response schemas

Remove the feature from shared product code:

- `TurnDeps`, the application composition root, and the runtime container
- graph nodes and routing
- agent run state, action log, trace, budgets, errors, and progress events
- LLM roles, default role settings, structured-output schemas, and prompts
- CLI progress output
- conversation/UI-message schemas and SSE encoding

No empty adapter, deprecated alias, feature flag, compatibility type, or
placeholder package remains.

### 3.3 Frontend boundaries

Delete the Skills route and feature directory, including upload, enable,
disable, delete, validation, error mapping, translations, tests, and E2E flow.
Remove its navigation item and any landing-page claim that users can customize
the agent through `SKILL.md`.

Chat messages no longer parse, retain, or render `data-skills`. The message view
and UI show only phases, evidence, answer text, citations, notices, and feedback
that remain in the new contract.

Regenerate the API client from the cleaned backend OpenAPI document so generated
endpoints, mocks, Zod schemas, and TypeScript models contain no product Skills
symbols.

## 4. Data and migration history

Delete the Skills migrations instead of adding a later drop migration. Reconnect
the remaining Alembic chain so it is linear:

- the migration currently following the first Skills migration points to
  `0001`
- the migration currently following the Agent Skills migration points to the
  preceding retained migration
- all later revision identifiers and ordering remain stable unless migration
  validation proves a rename is necessary

Remove the Skills ORM table, indexes, constraints, relationships, metadata
expectations, and repository tests. A fresh upgrade to `head` must never create
a `skills` table.

No production-data migration is provided. All environments using the old
history must recreate their database from the rewritten migration chain.

## 5. API and streaming contract

The following contract surface disappears:

- `GET /api/v1/skills`
- `POST /api/v1/skills`
- `PATCH /api/v1/skills/{name}`
- `DELETE /api/v1/skills/{name}`
- upload bodies and product types such as `SkillView`, `SkillRef`, and
  `SkillsData`
- the `data-skills` UI Message Stream part

Requests to the removed backend endpoints receive the application's normal 404
response. The removed frontend `/skills` URL follows the standard not-found
route. There is no redirect or deprecation response.

Old stored UI-message or trace payloads are not supported. Newly persisted
messages and traces contain no skill field or event.

## 6. Budget and failure behavior

Removing the selector saves one optional LLM call on turns where it previously
ran. Budget reservation and call-count tests must be recalculated from the new
graph rather than retaining a phantom allowance for Skills.

Remove all skill-specific failure paths, including parser, catalog, sync,
selection, upload, conflict, size-limit, and not-found errors. The remaining
nodes keep their current guarded failure and fallback behavior. A chat turn can
no longer fail or degrade because of a skill catalog or selector.

## 7. Documentation cleanup

Documentation is rewritten semantically rather than by blindly deleting the
word “skill”:

- `report/report.md` describes the new graph, feature set, LLM-call budget, and
  ablation study without Skills.
- root, backend, frontend, and relevant seed-pipeline README files describe only
  capabilities that remain.
- product Skills-only design or implementation documents are deleted.
- combined specs and plans under `backend/docs`, `frontend/docs`, and
  `seed-pipeline/docs` are edited so diagrams, file trees, contracts, task lists,
  examples, and test matrices consistently describe the target architecture.

Development-process skill instructions in those documents remain because they
belong to repository tooling, not to the product feature.

## 8. Testing and acceptance criteria

### 8.1 Backend

- Graph tests prove the transition from rephrase to search and the absence of a
  selector call.
- Prompt tests prove judge, refine, and answer prompts contain no skill block.
- Run, budget, action, progress, trace, UI-message, SSE, container, and API tests
  reflect the reduced state and contract.
- A clean Alembic upgrade reaches `head`, and metadata/schema tests prove no
  `skills` table exists.
- The generated OpenAPI document has no Skills endpoint, tag, body, or schema.

### 8.2 Frontend

- Type checking, linting, unit tests, browser tests, and applicable E2E tests
  pass without the feature.
- Sidebar, routing, landing content, i18n configuration, message rendering, and
  auth next-path tests contain no Skills behavior.
- Generated client code matches the cleaned OpenAPI source.

### 8.3 Repository-wide audit

Run a case-insensitive repository search for product Skills terminology and
classify every remaining match. Accepted matches are limited to the explicit
development-tooling boundary in section 2 and unrelated human-skill language.
There must be no product match in source, tests, fixtures, migrations, generated
files, README files, the thesis report, design specs, or implementation plans.

Review the final diff to confirm that `.agents/skills/`, `.claude/skills/`, and
`skills-lock.json` were not removed or rewritten as part of this change.

## 9. Completion condition

The work is complete only when a new reader can inspect the current code,
database history, generated contracts, tests, and documentation without finding
evidence that the Pharma AI Agent supports or previously supported product
Skills. The sole operational caveat is an explicit database reset requirement
caused by the approved clean-history migration rewrite.
