# Pharma Agent Platform Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the agent core into a usable service: Postgres persistence with Alembic, conversations with rolling-summary memory, retrieval audit, a Postgres LangGraph checkpointer, fastapi-users auth (email/password + Google), and a FastAPI HTTP API with SSE streaming.

**Architecture:** Domain gains a `Conversation` aggregate, `Message`, retrieval-audit value objects and repository ports. The application layer adds `ChatService` (wraps the Plan 1 runner, persists each turn in one transaction, emits `done` with ids), `SummarizeConversation` and `ConversationQueries`. Infrastructure adds SQLAlchemy 2 async models on psycopg 3, Alembic migrations, a psycopg-pool `AsyncPostgresSaver`, fastapi-users wiring and an async `open_container` that owns every resource. `api/` is thin FastAPI routers over application services.

**Tech Stack:** FastAPI 0.141, sse-starlette 3.4, uvicorn, SQLAlchemy 2.0 async + psycopg 3 (`postgresql+psycopg://`), Alembic 1.19, langgraph-checkpoint-postgres 3.1 + psycopg-pool, fastapi-users 15 (+ fastapi-users-db-sqlalchemy 7, httpx-oauth 0.17), testcontainers Postgres.

**Spec:** `backend/docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md` (sections 8, 9, 10, 12). Builds on Plan 1: `backend/docs/superpowers/plans/2026-09-11-pharma-agent-core.md`.

## Global Constraints

- Tooling standard (repo `README.md`): Python 3.12, shared root `ruff.toml`, strict `pyrefly.toml`, no `# noqa`, no `# type: ignore`, no rule toggles. Every task ends with `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q` all green (pyrefly must print `0 diagnostics`).
- Domain stays framework-free (`tests/architecture/test_layering.py`); `api/` never imports `pharma_agent.domain`.
- One Postgres driver: psycopg 3. SQLAlchemy URL `postgresql+psycopg://thesis:thesis@localhost:5433/thesis` (repo `.env` exposes Postgres on 5433); the checkpointer uses the same DSN without `+psycopg`.
- Domain ids are `uuid4().hex` strings; Postgres stores them as `uuid`.
- Integration tests needing Docker carry `@pytest.mark.integration` and share one session-scoped Postgres container.
- Answers and prompts never contain a medical disclaimer.
- Commit messages end with the session attribution trailer.

## Deviations from the spec (decided here)

- Conversation summary progress is tracked with `turn_count` and `summarized_turns` instead of `summary_covers_message_id`; it expresses "summarize every 2 new turns" without extra queries.
- Conversation title is set from the user's first message (first 80 characters) when the conversation is created, so the stream can return `conversation_id` in its first event. The spec's "title from standalone query" needs a round trip after rephrase for no user-visible gain.
- Skills stay filesystem-only until Plan 3; `feedback` and `skills` tables arrive with Plan 3.

---

## File Structure

```text
backend/
  alembic.ini                                        # Task 5 (points at the packaged migrations)
  src/pharma_agent/
    domain/conversation/
      models.py        (+ Conversation, Message)                        # Task 2
      turns.py         build_turn_messages                              # Task 2
      ports.py         ConversationRepository                           # Task 4
      prompts.py       summarizer prompt                                # Task 4
    domain/retrieval/audit.py   RetrievalRunRecord, audit_from_run      # Task 3
    application/errors.py                                               # Task 7
    application/chat/service.py         ChatService, ChatTurnResult     # Task 7
    application/conversation/queries.py ConversationQueries + views     # Task 8
    application/memory/summarize.py     SummarizeConversation           # Task 8
    infrastructure/settings.py          (+ postgres, auth, api, memory) # Task 1
    infrastructure/persistence/postgres/database.py                     # Task 5
    infrastructure/persistence/postgres/tables.py                       # Task 5
    infrastructure/persistence/postgres/migrations/env.py, script.py.mako, versions/0001_initial.py  # Task 5
    infrastructure/persistence/postgres/conversation_repository.py      # Task 6
    infrastructure/langgraph/checkpointer.py                            # Task 9
    infrastructure/container.py         open_container                  # Task 9
    infrastructure/auth/users.py        fastapi-users wiring            # Task 10
    api/app.py, api/deps.py, api/schemas.py, api/sse.py                 # Task 11
    api/routers/health.py, chat.py, conversations.py                    # Task 11
    cli.py             (+ serve, migrate)                               # Task 12
  tests/
    postgres.py                         shared container fixture       # Task 1
    domain/test_conversation.py, test_audit.py, test_summary_prompt.py  # Tasks 2-4
    memory_repository.py                InMemoryConversationRepository  # Task 7
    application/test_chat_service.py, test_summarize.py, test_queries.py # Tasks 7-8
    infrastructure/test_migrations.py, test_conversation_repository.py  # Tasks 5-6
    infrastructure/test_checkpointer.py                                 # Task 9
    api/test_auth.py, test_chat_api.py, test_conversations_api.py       # Tasks 10-11
```

---

### Task 1: Dependencies, platform settings and the shared Postgres fixture

**Files:**
- Modify: `backend/pyproject.toml` (dependencies)
- Modify: `backend/src/pharma_agent/infrastructure/settings.py`
- Modify: `backend/.env.example`
- Create: `backend/tests/postgres.py`, `backend/tests/conftest.py`
- Test: `backend/tests/infrastructure/test_settings.py` (extend), `backend/tests/infrastructure/test_postgres_fixture.py`

**Interfaces:**
- Produces: `PostgresSettings(dsn, pool_size, echo)` with `.conninfo`; `AuthSettings(jwt_secret, jwt_lifetime_seconds, google_client_id, google_client_secret, frontend_url)` with `.google_enabled` and `.require_jwt_secret()`; `ApiSettings(host, port, cors_origins)`; `MemorySettings(context_turns, context_chars, summary_every_turns, summary_max_chars)`; `Settings.postgres/auth/api/memory`; pytest fixture `postgres_dsn` (session scope, SQLAlchemy URL).

- [ ] **Step 1: Add dependencies**

In `backend/pyproject.toml` `dependencies`, add (keep alphabetical order):

```toml
    "alembic>=1.19,<2",
    "fastapi>=0.141,<1",
    "fastapi-users[oauth,sqlalchemy]>=15.0,<16",
    "langgraph-checkpoint-postgres>=3.1,<4",
    "psycopg[binary,pool]>=3.3,<4",
    "sqlalchemy[asyncio]>=2.0.52,<2.1",
    "sse-starlette>=3.4,<4",
    "uvicorn>=0.52,<1",
```

Run: `cd /home/andv/personal/thesis/backend && uv lock && uv sync`
Expected: resolves without conflicts.

- [ ] **Step 2: Write the failing settings tests**

Append to `backend/tests/infrastructure/test_settings.py`:

```python
def test_platform_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PHARMA_AUTH__JWT_SECRET", "PHARMA_POSTGRES__DSN"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.postgres.dsn == "postgresql+psycopg://thesis:thesis@localhost:5433/thesis"
    assert settings.postgres.conninfo == "postgresql://thesis:thesis@localhost:5433/thesis"
    assert settings.memory.summary_every_turns == 2
    assert settings.memory.context_turns == 4
    assert settings.auth.google_enabled is False
    with pytest.raises(ValueError, match="PHARMA_AUTH__JWT_SECRET"):
        settings.auth.require_jwt_secret()


def test_auth_secret_and_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_AUTH__JWT_SECRET", "x" * 40)
    monkeypatch.setenv("PHARMA_AUTH__GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("PHARMA_AUTH__GOOGLE_CLIENT_SECRET", "secret")
    settings = Settings(_env_file=None)
    assert settings.auth.require_jwt_secret() == "x" * 40
    assert settings.auth.google_enabled is True


def test_short_jwt_secret_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_AUTH__JWT_SECRET", "short")
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None)
```

Run: `uv run pytest tests/infrastructure/test_settings.py -q`
Expected: FAIL (`Settings` has no `postgres`).

- [ ] **Step 3: Implement settings**

In `backend/src/pharma_agent/infrastructure/settings.py` add `SecretStr, field_validator` to the pydantic import and add these models above `class Settings`:

```python
class PostgresSettings(BaseModel):
    dsn: str = "postgresql+psycopg://thesis:thesis@localhost:5433/thesis"
    pool_size: int = 10
    echo: bool = False

    @property
    def conninfo(self) -> str:
        """libpq connection string for psycopg (the checkpointer pool)."""
        return self.dsn.replace("postgresql+psycopg://", "postgresql://", 1)


class AuthSettings(BaseModel):
    jwt_secret: SecretStr | None = None
    jwt_lifetime_seconds: int = 7 * 24 * 3600
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    frontend_url: str = "http://localhost:3000"

    @field_validator("jwt_secret")
    @classmethod
    def _secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("jwt_secret must be at least 32 characters")
        return value

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    def require_jwt_secret(self) -> str:
        if self.jwt_secret is None:
            raise ValueError("set PHARMA_AUTH__JWT_SECRET (at least 32 characters)")
        return self.jwt_secret.get_secret_value()


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class MemorySettings(BaseModel):
    context_turns: int = 4
    context_chars: int = 4000
    summary_every_turns: int = 2
    summary_max_chars: int = 1500
```

Add to `class Settings`:

```python
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
```

Append to `backend/.env.example`:

```dotenv

# Postgres (docker-compose exposes it on POSTGRES_PORT=5433)
PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5433/thesis

# Auth (fastapi-users). Generate with: python -c "import secrets; print(secrets.token_urlsafe(48))"
PHARMA_AUTH__JWT_SECRET=
# PHARMA_AUTH__GOOGLE_CLIENT_ID=
# PHARMA_AUTH__GOOGLE_CLIENT_SECRET=
PHARMA_AUTH__FRONTEND_URL=http://localhost:3000

# HTTP API
PHARMA_API__HOST=127.0.0.1
PHARMA_API__PORT=8000
```

Run: `uv run pytest tests/infrastructure/test_settings.py -q`
Expected: PASS.

- [ ] **Step 4: Shared Postgres fixture**

`backend/tests/postgres.py`:

```python
"""Session-scoped Postgres for integration tests (one container per test session)."""

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        "postgres:17-alpine",
        username="thesis",
        password="thesis",
        dbname="thesis",
        driver="psycopg",
    ) as container:
        yield container.get_connection_url()
```

`backend/tests/conftest.py`:

```python
pytest_plugins = ["tests.postgres"]
```

`backend/tests/infrastructure/test_postgres_fixture.py`:

```python
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.integration


async def test_postgres_fixture_is_reachable(postgres_dsn: str) -> None:
    assert postgres_dsn.startswith("postgresql+psycopg://")
    engine = create_async_engine(postgres_dsn)
    async with engine.connect() as connection:
        assert (await connection.execute(text("select 1"))).scalar_one() == 1
    await engine.dispose()
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_postgres_fixture.py`
Expected: 1 passed (pulls `postgres:17-alpine` on first run).

- [ ] **Step 5: Full checks and commit**

Run the Global Constraints check command. Expected: all green.

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/pharma_agent/infrastructure/settings.py backend/.env.example backend/tests/postgres.py backend/tests/conftest.py backend/tests/infrastructure/test_settings.py backend/tests/infrastructure/test_postgres_fixture.py
git commit -m "feat(infra): add platform settings, dependencies and Postgres test fixture"
```

---

### Task 2: Conversation aggregate, Message and turn builder

**Files:**
- Modify: `backend/src/pharma_agent/domain/conversation/models.py`
- Create: `backend/src/pharma_agent/domain/conversation/turns.py`
- Test: `backend/tests/domain/test_conversation.py`

**Interfaces:**
- Consumes: `AgentRun`, `RunStatus`, `Citation`, `new_id`, `DomainError`.
- Produces: `MAX_TITLE_CHARS = 80`; `InvalidTitle(DomainError)`; `Conversation` with `start(user_id, first_message, now, conversation_id=None)`, `record_turn(now)`, `needs_summary(every)`, `apply_summary(summary, covered_turns, now)`, `rename(title, now)`; `Message(message_id, conversation_id, role, content, status, citations, phases, usage, run_id, created_at)` with `to_turn_part()`; `build_turn_messages(conversation_id, run, answer_text, citations, phases, now) -> tuple[Message, Message]`; `pair_turns(messages) -> list[Turn]`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/domain/test_conversation.py`:

```python
from datetime import timedelta

import pytest

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan, RunStatus
from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    InvalidTitle,
    MessageRole,
)
from pharma_agent.domain.conversation.turns import build_turn_messages, pair_turns
from tests.domain.factories import NOW, make_run


def test_start_sets_title_from_first_message() -> None:
    conversation = Conversation.start(
        user_id="u1", first_message="  Paracetamol   uống bao nhiêu?  " + "x" * 200, now=NOW
    )
    assert conversation.title.startswith("Paracetamol uống bao nhiêu?")
    assert len(conversation.title) <= 80
    assert conversation.turn_count == 0 and conversation.summary == ""
    assert conversation.created_at == conversation.updated_at == NOW
    assert len(conversation.conversation_id) == 32


def test_summary_is_due_every_n_turns() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.record_turn(NOW + timedelta(seconds=1))
    assert conversation.needs_summary(every=2) is False
    conversation.record_turn(NOW + timedelta(seconds=2))
    assert conversation.needs_summary(every=2) is True
    conversation.apply_summary("  tóm tắt  ", covered_turns=2, now=NOW + timedelta(seconds=3))
    assert conversation.summary == "tóm tắt" and conversation.summarized_turns == 2
    assert conversation.needs_summary(every=2) is False
    assert conversation.updated_at == NOW + timedelta(seconds=3)


def test_rename_validates_title() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.rename("  Thuốc hạ sốt  ", now=NOW)
    assert conversation.title == "Thuốc hạ sốt"
    with pytest.raises(InvalidTitle):
        conversation.rename("   ", now=NOW)
    with pytest.raises(InvalidTitle):
        conversation.rename("x" * 81, now=NOW)


def test_apply_summary_cannot_cover_future_turns() -> None:
    conversation = Conversation.start(user_id="u1", first_message="hi", now=NOW)
    conversation.record_turn(NOW)
    with pytest.raises(ValueError, match="covered_turns"):
        conversation.apply_summary("s", covered_turns=2, now=NOW)


def test_build_turn_messages_and_pair_turns() -> None:
    run = make_run("Liều paracetamol?")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=NOW)
    run.complete()
    citation = Citation(
        index=1, chunk_id="c1", section_id="s1", title="Paracetamol", section="Liều", start_page=1, end_page=2
    )
    user_msg, assistant_msg = build_turn_messages(
        conversation_id="conv1",
        run=run,
        answer_text="500 mg [1]",
        citations=[citation],
        phases=["guarding", "answering"],
        now=NOW,
    )
    assert user_msg.role is MessageRole.USER and user_msg.content == "Liều paracetamol?"
    assert user_msg.status == RunStatus.COMPLETED.value
    assert assistant_msg.role is MessageRole.ASSISTANT and assistant_msg.citations == [citation]
    assert assistant_msg.run_id == run.run_id and assistant_msg.phases == ["guarding", "answering"]
    assert assistant_msg.usage["llm_calls"] == 0
    assert assistant_msg.created_at > user_msg.created_at

    turns = pair_turns([user_msg, assistant_msg, user_msg])
    assert len(turns) == 1
    assert turns[0].user_text == "Liều paracetamol?" and turns[0].assistant_text == "500 mg [1]"
    assert turns[0].status == "completed"
```

Run: `uv run pytest tests/domain/test_conversation.py -q`
Expected: FAIL with `ImportError` (`Conversation` missing).

- [ ] **Step 2: Implement the aggregate**

Append to `backend/src/pharma_agent/domain/conversation/models.py` (add `from datetime import datetime`, `from typing import Any`, and imports of `DomainError`, `new_id`):

```python
MAX_TITLE_CHARS = 80


class InvalidTitle(DomainError):
    code = "INVALID_TITLE"


def _clean_title(text: str) -> str:
    return " ".join(text.split())


class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    summary: str = ""
    turn_count: int = 0
    summarized_turns: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def start(
        cls,
        *,
        user_id: str,
        first_message: str,
        now: datetime,
        conversation_id: str | None = None,
    ) -> "Conversation":
        title = _clean_title(first_message)[:MAX_TITLE_CHARS].rstrip() or "Cuộc trò chuyện mới"
        return cls(
            conversation_id=conversation_id or new_id(),
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def record_turn(self, now: datetime) -> None:
        self.turn_count += 1
        self.updated_at = now

    def needs_summary(self, every: int) -> bool:
        return self.turn_count - self.summarized_turns >= every

    def apply_summary(self, summary: str, *, covered_turns: int, now: datetime) -> None:
        if covered_turns > self.turn_count or covered_turns < self.summarized_turns:
            raise ValueError(
                f"covered_turns={covered_turns} must be between "
                f"{self.summarized_turns} and {self.turn_count}"
            )
        self.summary = summary.strip()
        self.summarized_turns = covered_turns
        self.updated_at = now

    def rename(self, title: str, *, now: datetime) -> None:
        cleaned = _clean_title(title)
        if not cleaned or len(cleaned) > MAX_TITLE_CHARS:
            raise InvalidTitle(f"title must be 1-{MAX_TITLE_CHARS} characters")
        self.title = cleaned
        self.updated_at = now


class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: MessageRole
    content: str
    status: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    created_at: datetime
```

- [ ] **Step 3: Implement the turn builder**

`backend/src/pharma_agent/domain/conversation/turns.py`:

```python
from collections.abc import Sequence
from datetime import datetime, timedelta

from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation, Message, MessageRole, Turn
from pharma_agent.domain.shared.ids import new_id

# The assistant message must sort after the user message even when both are
# created within the same clock tick.
_ASSISTANT_OFFSET = timedelta(microseconds=1)


def build_turn_messages(
    *,
    conversation_id: str,
    run: AgentRun,
    answer_text: str,
    citations: Sequence[Citation],
    phases: Sequence[str],
    now: datetime,
) -> tuple[Message, Message]:
    status = run.status.value
    user_message = Message(
        message_id=new_id(),
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=run.original_query,
        status=status,
        run_id=run.run_id,
        created_at=now,
    )
    assistant_message = Message(
        message_id=new_id(),
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content=answer_text,
        status=status,
        citations=list(citations),
        phases=list(phases),
        usage=run.usage.model_dump(),
        run_id=run.run_id,
        created_at=now + _ASSISTANT_OFFSET,
    )
    return user_message, assistant_message


def pair_turns(messages: Sequence[Message]) -> list[Turn]:
    """Pair consecutive user/assistant messages (oldest first) into turns."""
    turns: list[Turn] = []
    pending: Message | None = None
    for message in messages:
        if message.role is MessageRole.USER:
            pending = message
        elif pending is not None:
            turns.append(
                Turn(
                    user_text=pending.content,
                    assistant_text=message.content,
                    status=message.status,
                )
            )
            pending = None
    return turns
```

Run: `uv run pytest tests/domain/test_conversation.py -q`
Expected: 5 passed.

- [ ] **Step 4: Full checks and commit**

```bash
git add backend/src/pharma_agent/domain/conversation backend/tests/domain/test_conversation.py
git commit -m "feat(domain): add Conversation aggregate, Message and turn builder"
```

---

### Task 3: Retrieval audit records derived from a run

**Files:**
- Create: `backend/src/pharma_agent/domain/retrieval/audit.py`
- Test: `backend/tests/domain/test_audit.py`

**Interfaces:**
- Consumes: `AgentRun`, `ActionKind`, `Citation`, `Evidence`.
- Produces: `RetrievalHitRecord(rank, chunk_id, section_id, table_id, fusion_score, rerank_score, hydrate_strategy, cited, snippet)`, `RetrievalRunRecord(round, query_text, hits)`, `audit_from_run(run, citations, snippet_chars=300) -> list[RetrievalRunRecord]`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/domain/test_audit.py`:

```python
from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.retrieval.models import Query, QueryOrigin, RetrievedItem
from pharma_agent.domain.retrieval.service import SearchResult
from tests.domain.factories import NOW, make_hit, make_run


def result_for(query: str, *hits: tuple[str, float]) -> SearchResult:
    return SearchResult(
        items=[
            RetrievedItem(
                hit=make_hit(chunk_id, rerank=score).model_copy(update={"matched_queries": [query]})
            )
            for chunk_id, score in hits
        ]
    )


def test_one_record_per_query_per_round_with_ranked_hits_and_citations() -> None:
    run = make_run()
    q1 = Query(text="paracetamol liều", origin=QueryOrigin.INITIAL)
    run.record_search([q1], result_for(q1.text, ("c1", 0.4), ("c2", 0.9)), now=NOW)
    q2 = Query(text="paracetamol quá liều", origin=QueryOrigin.REFINED)
    q3 = Query(text="paracetamol trẻ em", origin=QueryOrigin.REFINED)
    run.record_search([q2, q3], result_for(q2.text, ("c3", 0.7)), now=NOW)

    citations = [
        Citation(index=1, chunk_id="c2", section_id="sec-1", title="t", section="s", start_page=1, end_page=1)
    ]
    records = audit_from_run(run, citations, snippet_chars=5)

    assert [(r.round, r.query_text) for r in records] == [
        (1, "paracetamol liều"),
        (2, "paracetamol quá liều"),
        (2, "paracetamol trẻ em"),
    ]
    first = records[0]
    assert [(h.rank, h.chunk_id, h.cited) for h in first.hits] == [(1, "c2", True), (2, "c1", False)]
    assert first.hits[0].rerank_score == 0.9 and first.hits[0].snippet == "parac"
    assert [h.chunk_id for h in records[1].hits] == ["c3"]
    assert records[2].hits == []


def test_failed_search_still_produces_a_record() -> None:
    run = make_run()
    q1 = Query(text="x", origin=QueryOrigin.INITIAL)
    run.record_search([q1], SearchResult(error="qdrant down"), now=NOW)
    records = audit_from_run(run, [])
    assert len(records) == 1 and records[0].hits == []
```

Run: `uv run pytest tests/domain/test_audit.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement**

`backend/src/pharma_agent/domain/retrieval/audit.py`:

```python
"""Retrieval audit derived from a finished AgentRun (one record per query per search round)."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.actions import ActionKind
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation


class RetrievalHitRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    section_id: str
    table_id: str
    fusion_score: float
    rerank_score: float | None
    hydrate_strategy: str
    cited: bool
    snippet: str


class RetrievalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    query_text: str
    hits: list[RetrievalHitRecord] = Field(default_factory=list)


def audit_from_run(
    run: AgentRun, citations: Sequence[Citation], *, snippet_chars: int = 300
) -> list[RetrievalRunRecord]:
    cited_chunks = {citation.chunk_id for citation in citations}
    records: list[RetrievalRunRecord] = []
    search_actions = [a for a in run.actions.entries if a.kind is ActionKind.SEARCH]
    for round_number, action in enumerate(search_actions, start=1):
        queries = action.payload.get("queries", [])
        for query_text in queries if isinstance(queries, list) else []:
            matching = sorted(
                (e for e in run.evidence.items if query_text in e.hit.matched_queries),
                key=lambda e: e.score,
                reverse=True,
            )
            hits = [
                RetrievalHitRecord(
                    rank=rank,
                    chunk_id=evidence.hit.chunk_id,
                    section_id=evidence.hit.section_id,
                    table_id=evidence.hit.table_id,
                    fusion_score=evidence.hit.fusion_score,
                    rerank_score=evidence.hit.rerank_score,
                    hydrate_strategy=evidence.hit.hydrate_strategy.value,
                    cited=evidence.hit.chunk_id in cited_chunks,
                    snippet=" ".join(evidence.hit.chunk_text.split())[:snippet_chars],
                )
                for rank, evidence in enumerate(matching, start=1)
            ]
            records.append(
                RetrievalRunRecord(round=round_number, query_text=str(query_text), hits=hits)
            )
    return records
```

Note: `query_text` comes from an untyped action payload (`dict[str, Any]`), so the `str()` guards the JSON boundary; if pyrefly reports it as unnecessary, the payload value type is already `str` and the call must be removed.

Run: `uv run pytest tests/domain/test_audit.py -q`
Expected: 2 passed. Evidence from one round that is merged again in a later round keeps both queries in `matched_queries`, so a chunk can appear under several records; that is intended.

- [ ] **Step 3: Full checks and commit**

```bash
git add backend/src/pharma_agent/domain/retrieval/audit.py backend/tests/domain/test_audit.py
git commit -m "feat(domain): derive retrieval audit records from an agent run"
```

---

### Task 4: Conversation repository port and summarizer prompt

**Files:**
- Create: `backend/src/pharma_agent/domain/conversation/ports.py`
- Create: `backend/src/pharma_agent/domain/conversation/prompts.py`
- Test: `backend/tests/domain/test_summary_prompt.py`

**Interfaces:**
- Consumes: `Conversation`, `Message`, `Turn`, `RetrievalRunRecord`, `ChatMessage`, `system`, `user`, `DISCLAIMER_PHRASES`.
- Produces: `ConversationRepository` protocol (`create`, `get`, `list_for_user`, `update_title`, `update_summary`, `delete`, `append_turn`, `recent_turns`, `turns_since`, `messages`); `summary_messages(previous_summary, turns, max_chars) -> list[ChatMessage]`.
- Concurrency rule: no method writes `turn_count` from an in-memory value. `append_turn` increments it in SQL, and `update_summary` / `update_title` only touch their own columns, so a summary finishing while a new turn is being written cannot lose a turn.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/test_summary_prompt.py`:

```python
from pharma_agent.domain.agent.prompts import DISCLAIMER_PHRASES
from pharma_agent.domain.conversation.models import Turn
from pharma_agent.domain.conversation.prompts import summary_messages


def test_summary_prompt_contains_previous_summary_turns_and_limit() -> None:
    messages = summary_messages(
        "Người dùng hỏi về amoxicillin.",
        [Turn(user_text="Uống lúc no hay đói?", assistant_text="Uống lúc nào cũng được [1].", status="completed")],
        max_chars=1500,
    )
    text = "\n".join(m.content for m in messages)
    assert "Người dùng hỏi về amoxicillin." in text
    assert "Uống lúc no hay đói?" in text and "Uống lúc nào cũng được [1]." in text
    assert "1500" in text
    assert not any(phrase in text.lower() for phrase in DISCLAIMER_PHRASES)


def test_summary_prompt_without_previous_summary() -> None:
    messages = summary_messages("", [Turn(user_text="a", assistant_text="b", status="completed")], max_chars=500)
    assert "(chưa có)" in messages[1].content
```

Run: `uv run pytest tests/domain/test_summary_prompt.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement the port and prompt**

`backend/src/pharma_agent/domain/conversation/ports.py`:

```python
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from pharma_agent.domain.conversation.models import Conversation, Message, Turn
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord


class ConversationRepository(Protocol):
    async def create(self, conversation: Conversation) -> None: ...

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        """Return the conversation only when it belongs to user_id."""
        ...

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        """Most recently updated first."""
        ...

    async def update_title(self, conversation: Conversation) -> None:
        """Persist only title and updated_at."""
        ...

    async def update_summary(self, conversation: Conversation) -> None:
        """Persist only summary, summarized_turns and updated_at."""
        ...

    async def delete(self, user_id: str, conversation_id: str) -> bool: ...

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        """Write both messages and the audit, and increment turn_count, in one transaction."""
        ...

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        """The last `limit` turns, oldest first."""
        ...

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        """All turns after the first `skip` turns, oldest first."""
        ...

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        """Newest `limit` messages strictly before `before`, returned oldest first."""
        ...
```

`backend/src/pharma_agent/domain/conversation/prompts.py`:

```python
from collections.abc import Sequence

from pharma_agent.domain.conversation.models import Turn
from pharma_agent.domain.llm.models import ChatMessage, system, user

SUMMARY_SYSTEM = """Bạn tóm tắt hội thoại giữa người dùng và trợ lý tra cứu thuốc để dùng làm ngữ cảnh cho các câu hỏi sau.
Trả về JSON {{"summary": "..."}}.

Quy tắc:
- Giữ lại: tên thuốc/hoạt chất/biệt dược đã nhắc, đối tượng (trẻ em, thai kỳ, bệnh nền), triệu chứng, các con số liều đã trao đổi, và điều người dùng còn muốn biết.
- Bỏ: lời chào, câu lặp lại, số trích dẫn [n].
- Viết tiếng Việt, câu ngắn, tối đa {max_chars} ký tự.
- Gộp tóm tắt cũ với các lượt mới thành một bản duy nhất; thông tin mới hơn thay thế thông tin cũ mâu thuẫn."""


def summary_messages(
    previous_summary: str, turns: Sequence[Turn], *, max_chars: int
) -> list[ChatMessage]:
    rendered = "\n\n".join(
        f"Người dùng: {turn.user_text}\nTrợ lý: {turn.assistant_text}" for turn in turns
    )
    content = (
        f"Tóm tắt cũ:\n{previous_summary.strip() or '(chưa có)'}\n\n"
        f"Các lượt mới:\n{rendered}"
    )
    return [system(SUMMARY_SYSTEM.format(max_chars=max_chars)), user(content)]
```

Run: `uv run pytest tests/domain/test_summary_prompt.py -q`
Expected: 2 passed.

- [ ] **Step 3: Full checks and commit**

```bash
git add backend/src/pharma_agent/domain/conversation/ports.py backend/src/pharma_agent/domain/conversation/prompts.py backend/tests/domain/test_summary_prompt.py
git commit -m "feat(domain): add conversation repository port and summarizer prompt"
```

---

### Task 5: SQLAlchemy tables, Database and packaged Alembic migrations

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/persistence/__init__.py`, `.../persistence/postgres/__init__.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/database.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/{env.py,script.py.mako,versions/0001_initial.py}` (+ empty `__init__.py` in `migrations/` and `versions/`)
- Create: `backend/alembic.ini`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/alembic_config.py`
- Modify: `backend/tests/postgres.py` (database helpers, `migrated_dsn`)
- Test: `backend/tests/infrastructure/test_migrations.py`

**Interfaces:**
- Produces: `Base` (naming convention), `UserTable`, `OAuthAccountTable`, `ConversationTable`, `MessageTable`, `RetrievalRunTable`, `RetrievalHitTable`; `Database(dsn, pool_size, echo)` with `.engine`, `.sessions`, `ping()`, `dispose()`; `alembic_config(dsn: str | None) -> alembic.config.Config`; fixtures `fresh_database_dsn` (function scope) and `migrated_dsn` (session scope).

- [ ] **Step 1: Write the failing migration tests**

Extend `backend/tests/postgres.py`:

```python
"""Postgres for integration tests: one container per session, databases created on demand."""

from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        "postgres:17-alpine",
        username="thesis",
        password="thesis",
        dbname="thesis",
        driver="psycopg",
    ) as container:
        yield container.get_connection_url()


def create_database(server_dsn: str, name: str) -> str:
    url = make_url(server_dsn)
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as connection:
        connection.execute(f'CREATE DATABASE "{name}"')
    return url.set(database=name).render_as_string(hide_password=False)


@pytest.fixture
def fresh_database_dsn(postgres_dsn: str) -> str:
    return create_database(postgres_dsn, f"t_{uuid4().hex[:12]}")


@pytest.fixture(scope="session")
def migrated_dsn(postgres_dsn: str) -> str:
    from alembic import command

    from pharma_agent.infrastructure.persistence.postgres.alembic_config import alembic_config

    dsn = create_database(postgres_dsn, "app")
    command.upgrade(alembic_config(dsn), "head")
    return dsn
```

`backend/tests/infrastructure/test_migrations.py`:

```python
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from pharma_agent.infrastructure.persistence.postgres.alembic_config import alembic_config
from pharma_agent.infrastructure.persistence.postgres.tables import Base

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {"user", "oauth_account", "conversations", "messages", "retrieval_runs", "retrieval_hits"}


def _diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(connection, opts={"compare_type": True})
    return list(compare_metadata(context, Base.metadata))


def _tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


async def test_upgrade_head_matches_the_models(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        assert EXPECTED_TABLES <= await connection.run_sync(_tables)
        assert await connection.run_sync(_diff) == []
    await engine.dispose()


async def test_downgrade_then_upgrade_round_trips(fresh_database_dsn: str) -> None:
    import asyncio

    config = alembic_config(fresh_database_dsn)
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")
    engine = create_async_engine(fresh_database_dsn)
    async with engine.connect() as connection:
        assert not (EXPECTED_TABLES & await connection.run_sync(_tables))
    await engine.dispose()
    await asyncio.to_thread(command.upgrade, config, "head")
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_migrations.py`
Expected: FAIL (`ModuleNotFoundError: pharma_agent.infrastructure.persistence`).

- [ ] **Step 2: Tables**

`backend/src/pharma_agent/infrastructure/persistence/__init__.py` and `.../postgres/__init__.py`: empty.

`backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`:

```python
import uuid
from datetime import datetime
from typing import Any

from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseOAuthAccountTableUUID,
    SQLAlchemyBaseUserTableUUID,
)
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class OAuthAccountTable(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """fastapi-users OAuth accounts (table name `oauth_account`)."""


class UserTable(SQLAlchemyBaseUserTableUUID, Base):
    """fastapi-users users (table name `user`) plus profile columns."""

    display_name: Mapped[str] = mapped_column(String(120), nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    oauth_accounts: Mapped[list[OAuthAccountTable]] = relationship(lazy="joined")


class ConversationTable(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    summarized_turns: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MessageTable(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    phases: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RetrievalRunTable(Base):
    __tablename__ = "retrieval_runs"
    __table_args__ = (Index("ix_retrieval_runs_message", "message_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    corpus_version: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    retriever_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RetrievalHitTable(Base):
    __tablename__ = "retrieval_hits"

    retrieval_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("retrieval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    section_id: Mapped[str] = mapped_column(Text, nullable=False)
    table_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    fusion_score: Mapped[float] = mapped_column(Float, nullable=False)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hydrate_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    cited: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    snippet: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
```

- [ ] **Step 3: Database and Alembic config**

`backend/src/pharma_agent/infrastructure/persistence/postgres/database.py`:

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


class Database:
    def __init__(self, dsn: str, *, pool_size: int = 10, echo: bool = False) -> None:
        self.engine: AsyncEngine = create_async_engine(
            dsn, pool_size=pool_size, echo=echo, pool_pre_ping=True
        )
        self.sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    async def dispose(self) -> None:
        await self.engine.dispose()
```

`backend/src/pharma_agent/infrastructure/persistence/postgres/alembic_config.py`:

```python
from alembic.config import Config

SCRIPT_LOCATION = "pharma_agent.infrastructure.persistence.postgres:migrations"


def alembic_config(dsn: str | None = None) -> Config:
    """Alembic config that works from an installed package; `dsn` overrides settings."""
    config = Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    config.attributes["dsn"] = dsn
    return config
```

`backend/alembic.ini` (for running `uv run alembic ...` from `backend/`):

```ini
[alembic]
script_location = pharma_agent.infrastructure.persistence.postgres:migrations
path_separator = os

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 4: Migration environment and initial revision**

`.../postgres/migrations/__init__.py` and `.../migrations/versions/__init__.py`: empty.

`.../postgres/migrations/script.py.mako`:

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

`.../postgres/migrations/env.py`:

```python
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from pharma_agent.infrastructure.persistence.postgres.tables import Base
from pharma_agent.infrastructure.settings import Settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _dsn() -> str:
    dsn = config.attributes.get("dsn")
    return dsn if isinstance(dsn, str) and dsn else Settings().postgres.dsn


def run_migrations_offline() -> None:
    context.configure(
        url=_dsn(), target_metadata=target_metadata, literal_binds=True, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_dsn())
    async with engine.connect() as connection:
        await connection.run_sync(_run_with_connection)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

`.../postgres/migrations/versions/0001_initial.py` — generate it, then review against the tables above:

Run (from `backend/`, Docker needed):
```bash
uv run python - <<'PY'
from testcontainers.postgres import PostgresContainer
from alembic import command
from pharma_agent.infrastructure.persistence.postgres.alembic_config import alembic_config
with PostgresContainer("postgres:17-alpine", driver="psycopg") as pg:
    command.revision(alembic_config(pg.get_connection_url()), message="initial", autogenerate=True, rev_id="0001")
PY
```

Then edit the generated file: rename to `0001_initial.py`; import `fastapi_users_db_sqlalchemy.generics` if the autogenerate emitted `GUID()`; make sure `upgrade()` creates `user`, `oauth_account`, `conversations`, `messages`, `retrieval_runs`, `retrieval_hits` with the constraint names from the naming convention and `downgrade()` drops them in reverse order. Keep the file free of `# noqa` (run `uv run ruff format` on it).

Run: `uv run pytest -q -m integration tests/infrastructure/test_migrations.py`
Expected: 2 passed (`compare_metadata` returns `[]`).

- [ ] **Step 5: Full checks (unit + integration) and commit**

Run the Global Constraints check command, then `uv run pytest -q -m integration`.

```bash
git add backend/alembic.ini backend/src/pharma_agent/infrastructure/persistence backend/tests/postgres.py backend/tests/infrastructure/test_migrations.py
git commit -m "feat(infra): add Postgres tables, database session factory and Alembic migrations"
```

---

### Task 6: Postgres conversation repository with retrieval audit

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py`
- Test: `backend/tests/infrastructure/test_conversation_repository.py`

**Interfaces:**
- Consumes: `ConversationRepository`, `Conversation`, `Message`, `Turn`, `Citation`, `RetrievalRunRecord`, `pair_turns`, tables, `Database`.
- Produces: `AuditContext(corpus_version, embedding_model, retriever_config)`; `PostgresConversationRepository(sessions, audit_context)` implementing the port; `ConversationRowMissing(LookupError)`.

- [ ] **Step 1: Write the failing integration tests**

`backend/tests/infrastructure/test_conversation_repository.py`:

```python
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import func, select, text

from pharma_agent.domain.conversation.models import Citation, Conversation, Message, MessageRole
from pharma_agent.domain.retrieval.audit import RetrievalHitRecord, RetrievalRunRecord
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    ConversationRowMissing,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.tables import (
    RetrievalHitTable,
    RetrievalRunTable,
    UserTable,
)
from tests.domain.factories import NOW

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncIterator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(
            text('TRUNCATE "user", conversations, messages, retrieval_runs, retrieval_hits CASCADE')
        )
    yield db
    await db.dispose()


async def make_user(database: Database, email: str = "a@example.com") -> str:
    user_id = uuid.uuid4()
    async with database.sessions.begin() as session:
        session.add(UserTable(id=user_id, email=email, hashed_password="x"))
    return user_id.hex


def repository(database: Database) -> PostgresConversationRepository:
    return PostgresConversationRepository(
        database.sessions,
        AuditContext(
            corpus_version="thesis_chunks_qwen3_embedding_4b_fp16",
            embedding_model="qwen3-embedding:4b-fp16",
            retriever_config={"mode": "hybrid", "rrf_k": 2},
        ),
    )


def turn(conversation_id: str, index: int, status: str = "completed") -> tuple[Message, Message]:
    at = NOW + timedelta(minutes=index)
    citation = Citation(index=1, chunk_id="c1", section_id="s1", title="T", section="S", start_page=1, end_page=1)
    return (
        Message(message_id=uuid.uuid4().hex, conversation_id=conversation_id, role=MessageRole.USER,
                content=f"q{index}", status=status, run_id="r" * 32, created_at=at),
        Message(message_id=uuid.uuid4().hex, conversation_id=conversation_id, role=MessageRole.ASSISTANT,
                content=f"a{index}", status=status, citations=[citation], phases=["answering"],
                usage={"llm_calls": 5}, run_id="r" * 32, created_at=at + timedelta(microseconds=1)),
    )


AUDIT = [
    RetrievalRunRecord(
        round=1,
        query_text="paracetamol liều",
        hits=[
            RetrievalHitRecord(rank=1, chunk_id="c1", section_id="s1", table_id="", fusion_score=0.5,
                               rerank_score=0.9, hydrate_strategy="chunk_window", cited=True, snippet="x"),
        ],
    )
]


async def test_create_get_is_scoped_to_owner(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    stranger = await make_user(database, "b@example.com")
    conversation = Conversation.start(user_id=owner, first_message="Paracetamol?", now=NOW)
    await repo.create(conversation)

    loaded = await repo.get(owner, conversation.conversation_id)
    assert loaded is not None and loaded.title == "Paracetamol?" and loaded.created_at == NOW
    assert await repo.get(stranger, conversation.conversation_id) is None
    assert await repo.get(owner, "not-a-uuid") is None
    assert await repo.delete(stranger, conversation.conversation_id) is False


async def test_append_turn_is_atomic_and_increments_turn_count(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await repo.create(conversation)

    for index in range(3):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(conversation, user_msg, assistant_msg, AUDIT if index == 0 else [])

    loaded = await repo.get(owner, conversation.conversation_id)
    assert loaded is not None and loaded.turn_count == 3
    assert [t.user_text for t in await repo.recent_turns(conversation.conversation_id, 2)] == ["q1", "q2"]
    assert [t.user_text for t in await repo.turns_since(conversation.conversation_id, 1)] == ["q1", "q2"]
    messages = await repo.messages(conversation.conversation_id, limit=3)
    assert [m.content for m in messages] == ["a1", "q2", "a2"]
    assert messages[0].citations[0].chunk_id == "c1" and messages[0].usage == {"llm_calls": 5}
    older = await repo.messages(conversation.conversation_id, limit=10, before=messages[0].created_at)
    assert [m.content for m in older] == ["q0", "a0", "q1"]

    async with database.sessions() as session:
        run = (await session.execute(select(RetrievalRunTable))).scalar_one()
        assert run.corpus_version == "thesis_chunks_qwen3_embedding_4b_fp16" and run.round == 1
        hit = (await session.execute(select(RetrievalHitTable))).scalar_one()
        assert hit.cited is True and hit.rerank_score == 0.9


async def test_append_turn_rolls_back_when_conversation_is_missing(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    ghost = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    user_msg, assistant_msg = turn(ghost.conversation_id, 0)
    with pytest.raises(ConversationRowMissing):
        await repo.append_turn(ghost, user_msg, assistant_msg, AUDIT)
    async with database.sessions() as session:
        assert (await session.execute(select(func.count()).select_from(RetrievalRunTable))).scalar_one() == 0


async def test_update_summary_does_not_overwrite_turn_count(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await repo.create(conversation)
    stale = conversation.model_copy(deep=True)

    for index in range(2):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])

    stale.turn_count = 2
    stale.apply_summary("tóm tắt", covered_turns=2, now=NOW + timedelta(hours=1))
    await repo.update_summary(stale)
    stale.rename("Tên mới", now=NOW + timedelta(hours=1))
    await repo.update_title(stale)

    loaded = await repo.get(owner, conversation.conversation_id)
    assert loaded is not None
    assert (loaded.turn_count, loaded.summarized_turns, loaded.summary, loaded.title) == (2, 2, "tóm tắt", "Tên mới")


async def test_list_for_user_orders_by_recent_activity_and_deletes_cascade(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    first = Conversation.start(user_id=owner, first_message="first", now=NOW)
    second = Conversation.start(user_id=owner, first_message="second", now=NOW + timedelta(minutes=5))
    await repo.create(first)
    await repo.create(second)
    user_msg, assistant_msg = turn(first.conversation_id, 10)
    first.record_turn(assistant_msg.created_at)
    await repo.append_turn(first, user_msg, assistant_msg, AUDIT)

    listed = await repo.list_for_user(owner, limit=10)
    assert [c.title for c in listed] == ["first", "second"]
    assert [c.title for c in await repo.list_for_user(owner, limit=10, before=listed[0].updated_at)] == ["second"]

    assert await repo.delete(owner, first.conversation_id) is True
    async with database.sessions() as session:
        assert (await session.execute(select(func.count()).select_from(RetrievalHitTable))).scalar_one() == 0
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_conversation_repository.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement the repository**

`backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py`:

```python
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    Message,
    MessageRole,
    Turn,
)
from pharma_agent.domain.conversation.turns import pair_turns
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord
from pharma_agent.infrastructure.persistence.postgres.tables import (
    ConversationTable,
    MessageTable,
    RetrievalHitTable,
    RetrievalRunTable,
)


class ConversationRowMissing(LookupError):
    """append_turn was called for a conversation that is not in the database."""


@dataclass(frozen=True)
class AuditContext:
    corpus_version: str
    embedding_model: str
    retriever_config: dict[str, Any] = field(default_factory=dict)


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _conversation(row: ConversationTable) -> Conversation:
    return Conversation(
        conversation_id=row.id.hex,
        user_id=row.user_id.hex,
        title=row.title,
        summary=row.summary,
        turn_count=row.turn_count,
        summarized_turns=row.summarized_turns,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _message(row: MessageTable) -> Message:
    return Message(
        message_id=row.id.hex,
        conversation_id=row.conversation_id.hex,
        role=MessageRole(row.role),
        content=row.content,
        status=row.status,
        citations=[Citation.model_validate(item) for item in row.citations],
        phases=list(row.phases),
        usage=dict(row.usage),
        run_id=row.run_id,
        created_at=row.created_at,
    )


def _message_row(message: Message) -> MessageTable:
    return MessageTable(
        id=uuid.UUID(hex=message.message_id),
        conversation_id=uuid.UUID(hex=message.conversation_id),
        role=message.role.value,
        content=message.content,
        status=message.status,
        citations=[citation.model_dump() for citation in message.citations],
        phases=list(message.phases),
        usage=dict(message.usage),
        run_id=message.run_id,
        created_at=message.created_at,
    )


class PostgresConversationRepository:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], audit_context: AuditContext
    ) -> None:
        self._sessions = sessions
        self._audit_context = audit_context

    async def create(self, conversation: Conversation) -> None:
        async with self._sessions.begin() as session:
            session.add(
                ConversationTable(
                    id=uuid.UUID(hex=conversation.conversation_id),
                    user_id=uuid.UUID(hex=conversation.user_id),
                    title=conversation.title,
                    summary=conversation.summary,
                    turn_count=conversation.turn_count,
                    summarized_turns=conversation.summarized_turns,
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                )
            )

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        owner, key = _uuid(user_id), _uuid(conversation_id)
        if owner is None or key is None:
            return None
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(ConversationTable).where(
                        ConversationTable.id == key, ConversationTable.user_id == owner
                    )
                )
            ).scalar_one_or_none()
        return _conversation(row) if row is not None else None

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        query = select(ConversationTable).where(ConversationTable.user_id == owner)
        if before is not None:
            query = query.where(ConversationTable.updated_at < before)
        query = query.order_by(ConversationTable.updated_at.desc()).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_conversation(row) for row in rows]

    async def update_title(self, conversation: Conversation) -> None:
        await self._update(
            conversation, title=conversation.title, updated_at=conversation.updated_at
        )

    async def update_summary(self, conversation: Conversation) -> None:
        await self._update(
            conversation,
            summary=conversation.summary,
            summarized_turns=conversation.summarized_turns,
            updated_at=conversation.updated_at,
        )

    async def delete(self, user_id: str, conversation_id: str) -> bool:
        owner, key = _uuid(user_id), _uuid(conversation_id)
        if owner is None or key is None:
            return False
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(ConversationTable)
                .where(ConversationTable.id == key, ConversationTable.user_id == owner)
                .returning(ConversationTable.id)
            )
            return result.first() is not None

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        key = uuid.UUID(hex=conversation.conversation_id)
        assistant_id = uuid.UUID(hex=assistant_message.message_id)
        async with self._sessions.begin() as session:
            updated = await session.execute(
                update(ConversationTable)
                .where(ConversationTable.id == key)
                .values(
                    turn_count=ConversationTable.turn_count + 1,
                    updated_at=conversation.updated_at,
                )
                .returning(ConversationTable.id)
            )
            if updated.first() is None:
                raise ConversationRowMissing(conversation.conversation_id)
            session.add_all([_message_row(user_message), _message_row(assistant_message)])
            await session.flush()
            for record in audit:
                run_row = RetrievalRunTable(
                    message_id=assistant_id,
                    conversation_id=key,
                    run_id=assistant_message.run_id or "",
                    round=record.round,
                    query_text=record.query_text,
                    corpus_version=self._audit_context.corpus_version,
                    embedding_model=self._audit_context.embedding_model,
                    retriever_config=dict(self._audit_context.retriever_config),
                    created_at=assistant_message.created_at,
                )
                session.add(run_row)
                await session.flush()
                session.add_all(
                    RetrievalHitTable(retrieval_run_id=run_row.id, **hit.model_dump())
                    for hit in record.hits
                )

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        messages = await self.messages(conversation_id, limit=limit * 2)
        return pair_turns(messages)[-limit:]

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        key = _uuid(conversation_id)
        if key is None:
            return []
        query = (
            select(MessageTable)
            .where(MessageTable.conversation_id == key)
            .order_by(MessageTable.created_at)
            .offset(skip * 2)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return pair_turns([_message(row) for row in rows])

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        key = _uuid(conversation_id)
        if key is None:
            return []
        query = select(MessageTable).where(MessageTable.conversation_id == key)
        if before is not None:
            query = query.where(MessageTable.created_at < before)
        query = query.order_by(MessageTable.created_at.desc()).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_message(row) for row in reversed(rows)]

    async def _update(self, conversation: Conversation, **values: object) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ConversationTable)
                .where(ConversationTable.id == uuid.UUID(hex=conversation.conversation_id))
                .values(**values)
            )
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_conversation_repository.py`
Expected: 5 passed.

- [ ] **Step 3: Full checks and commit**

```bash
git add backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py backend/tests/infrastructure/test_conversation_repository.py
git commit -m "feat(infra): add Postgres conversation repository with retrieval audit"
```

---

### Task 7: ChatService — run a turn inside a conversation and persist it

**Files:**
- Create: `backend/src/pharma_agent/application/errors.py`
- Modify: `backend/src/pharma_agent/application/progress.py` (add `EventType.CONVERSATION`)
- Create: `backend/src/pharma_agent/application/chat/service.py`
- Create: `backend/tests/memory_repository.py`
- Test: `backend/tests/application/test_chat_service.py`

**Interfaces:**
- Consumes: `ChatTurnRunner`, `ChatTurnExecution`, `ProgressEvent`, `ConversationRepository`, `Conversation`, `ConversationContext`, `context_for_rephrase`, `build_turn_messages`, `audit_from_run`, `Clock`.
- Produces: `ApplicationError`, `ConversationNotFound`, `InvalidInput`, `AgentUnavailable`; `EventType.CONVERSATION` (`data = {conversation_id, title, created}`); `MemoryPolicy(context_turns, context_chars)`; `ChatTurnResult(conversation_id, message_id, run_id, status, content, citations, phases, usage, persisted)`; `ChatSession` (`conversation_id`, `events()`, `result`); `ChatService(runner, conversations, clock, policy)` with `open_turn(user_id, message, conversation_id)` and `ask(...)`; test double `InMemoryConversationRepository`.
- Event contract after this task: `conversation` → (runner events) → `done` (adds `message_id`, may be `null`) → optional `error{code: "PERSIST_FAILED"}`.

- [ ] **Step 1: In-memory repository test double**

`backend/tests/memory_repository.py`:

```python
from collections.abc import Sequence
from datetime import datetime

from pharma_agent.domain.conversation.models import Conversation, Message, Turn
from pharma_agent.domain.conversation.turns import pair_turns
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord


class InMemoryConversationRepository:
    """Behaves like the Postgres repository, including owner scoping and SQL-side turn counting."""

    def __init__(self) -> None:
        self.rows: dict[str, Conversation] = {}
        self.message_log: dict[str, list[Message]] = {}
        self.audit: dict[str, list[RetrievalRunRecord]] = {}
        self.fail_append = False

    async def create(self, conversation: Conversation) -> None:
        self.rows[conversation.conversation_id] = conversation.model_copy(deep=True)
        self.message_log[conversation.conversation_id] = []

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        row = self.rows.get(conversation_id)
        return row.model_copy(deep=True) if row is not None and row.user_id == user_id else None

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        rows = [
            row.model_copy(deep=True)
            for row in self.rows.values()
            if row.user_id == user_id and (before is None or row.updated_at < before)
        ]
        return sorted(rows, key=lambda row: row.updated_at, reverse=True)[:limit]

    async def update_title(self, conversation: Conversation) -> None:
        row = self.rows[conversation.conversation_id]
        row.title, row.updated_at = conversation.title, conversation.updated_at

    async def update_summary(self, conversation: Conversation) -> None:
        row = self.rows[conversation.conversation_id]
        row.summary = conversation.summary
        row.summarized_turns = conversation.summarized_turns
        row.updated_at = conversation.updated_at

    async def delete(self, user_id: str, conversation_id: str) -> bool:
        row = self.rows.get(conversation_id)
        if row is None or row.user_id != user_id:
            return False
        del self.rows[conversation_id]
        self.message_log.pop(conversation_id, None)
        return True

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        if self.fail_append:
            raise RuntimeError("database unavailable")
        row = self.rows.get(conversation.conversation_id)
        if row is None:
            raise LookupError(conversation.conversation_id)
        row.turn_count += 1
        row.updated_at = conversation.updated_at
        self.message_log[conversation.conversation_id].extend([user_message, assistant_message])
        self.audit[assistant_message.message_id] = list(audit)

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        return pair_turns(self.message_log.get(conversation_id, []))[-limit:]

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        return pair_turns(self.message_log.get(conversation_id, [])[skip * 2 :])

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        items = [
            message
            for message in self.message_log.get(conversation_id, [])
            if before is None or message.created_at < before
        ]
        return items[-limit:]
```

- [ ] **Step 2: Write the failing service tests**

`backend/tests/application/test_chat_service.py`:

```python
import pytest

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.errors import ConversationNotFound
from pharma_agent.application.progress import EventType, ProgressEvent
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
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_hit
from tests.fakes import NOW, FakeLlm, FakeRetriever, build_deps
from tests.memory_repository import InMemoryConversationRepository

OWNER = "a" * 32
STRANGER = "b" * 32


def scripted_turn(llm: FakeLlm, standalone: str) -> None:
    llm.script(LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok"))
    llm.script(
        LlmRole.REPHRASE,
        RephraseResult(standalone_query=standalone, audience=Audience.GENERAL_PUBLIC, language=Language.VI, intent=Intent.PHARMA_QUESTION),
    )
    llm.script(LlmRole.SKILL_SELECTOR, SkillSelection(skill_ids=[]))
    llm.script(LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ"))


def service_with(llm: FakeLlm, repo: InMemoryConversationRepository, rounds: int = 2) -> ChatService:
    retriever = FakeRetriever(*[[make_hit(f"c{i}", fusion=0.9)] for i in range(rounds)])
    runner = ChatTurnRunner(build_chat_graph(), build_deps(llm, retriever), BudgetLimits())
    return ChatService(runner, repo, FixedClock(NOW), MemoryPolicy(context_turns=4, context_chars=4000))


async def collect(events) -> list[ProgressEvent]:
    return [event async for event in events]


async def test_new_conversation_turn_is_persisted_with_audit() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    service = service_with(llm, repo)

    session = await service.open_turn(user_id=OWNER, message="Paracetamol uống bao nhiêu?", conversation_id=None)
    events = await collect(session.events())

    first = events[0]
    assert first.type is EventType.CONVERSATION
    assert first.data == {"conversation_id": session.conversation_id, "title": "Paracetamol uống bao nhiêu?", "created": True}
    done = next(e for e in events if e.type is EventType.DONE)
    assert done.data["conversation_id"] == session.conversation_id
    assert done.data["status"] == "completed" and done.data["message_id"]
    assert events[-1] is done

    stored = repo.rows[session.conversation_id]
    assert stored.turn_count == 1 and stored.user_id == OWNER
    user_msg, assistant_msg = repo.message_log[session.conversation_id]
    assert user_msg.content == "Paracetamol uống bao nhiêu?"
    assert assistant_msg.message_id == done.data["message_id"] and assistant_msg.citations
    assert "answering" in assistant_msg.phases
    assert repo.audit[assistant_msg.message_id][0].query_text == "Liều paracetamol cho người lớn"
    assert session.result is not None and session.result.persisted is True


async def test_follow_up_turn_sends_previous_turn_to_rephrase() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    scripted_turn(llm, "Paracetamol có dùng cho trẻ em không")
    service = service_with(llm, repo)

    first = await service.ask(user_id=OWNER, message="Paracetamol uống bao nhiêu?", conversation_id=None)
    second_session = await service.open_turn(user_id=OWNER, message="Còn trẻ em thì sao?", conversation_id=first.conversation_id)
    events = await collect(second_session.events())

    assert events[0].data["created"] is False
    rephrase_prompt = llm.calls_for(LlmRole.REPHRASE)[1][1].content
    assert "Paracetamol uống bao nhiêu?" in rephrase_prompt and "Còn trẻ em thì sao?" in rephrase_prompt
    assert repo.rows[first.conversation_id].turn_count == 2


async def test_unknown_or_foreign_conversation_is_not_found() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "x")
    service = service_with(llm, repo)
    owned = await service.ask(user_id=OWNER, message="hi", conversation_id=None)

    with pytest.raises(ConversationNotFound):
        await service.open_turn(user_id=OWNER, message="hi", conversation_id="f" * 32)
    with pytest.raises(ConversationNotFound):
        await service.open_turn(user_id=STRANGER, message="hi", conversation_id=owned.conversation_id)


async def test_persist_failure_reports_error_after_done() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol")
    service = service_with(llm, repo)
    session = await service.open_turn(user_id=OWNER, message="Paracetamol?", conversation_id=None)
    repo.fail_append = True

    events = await collect(session.events())

    assert [e.type for e in events[-2:]] == [EventType.DONE, EventType.ERROR]
    assert events[-2].data["message_id"] is None
    assert events[-1].data["code"] == "PERSIST_FAILED"
    assert session.result is not None and session.result.persisted is False
    assert session.result.content  # the answer is still delivered
```

Run: `uv run pytest -q tests/application/test_chat_service.py`
Expected: FAIL (`ModuleNotFoundError: pharma_agent.application.chat.service`).

- [ ] **Step 3: Errors and the conversation event**

`backend/src/pharma_agent/application/errors.py`:

```python
class ApplicationError(Exception):
    code = "APPLICATION_ERROR"


class ConversationNotFound(ApplicationError):
    code = "CONVERSATION_NOT_FOUND"


class InvalidInput(ApplicationError):
    code = "INVALID_INPUT"


class AgentUnavailable(ApplicationError):
    code = "AGENT_UNAVAILABLE"
```

In `backend/src/pharma_agent/application/progress.py` add to `EventType` (first member):

```python
    CONVERSATION = "conversation"
```

- [ ] **Step 4: Implement the service**

`backend/src/pharma_agent/application/chat/service.py`:

```python
"""Chat use case: run one agent turn inside a conversation and persist it atomically."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from pharma_agent.application.chat.runner import ChatTurnExecution, ChatTurnRunner
from pharma_agent.application.errors import ConversationNotFound
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.conversation.context import context_for_rephrase
from pharma_agent.domain.conversation.models import Citation, Conversation, ConversationContext
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.retrieval.audit import audit_from_run
from pharma_agent.domain.shared.clock import Clock

logger = logging.getLogger(__name__)

PERSIST_FAILED_MESSAGE = "Không lưu được lượt hội thoại này; câu trả lời vẫn hiển thị bình thường."


class MemoryPolicy(BaseModel):
    context_turns: int = 4
    context_chars: int = 4000


class ChatTurnResult(BaseModel):
    conversation_id: str
    message_id: str | None
    run_id: str
    status: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    persisted: bool


class ChatSession:
    def __init__(
        self,
        execution: ChatTurnExecution,
        conversation: Conversation,
        *,
        created: bool,
        conversations: ConversationRepository,
        clock: Clock,
    ) -> None:
        self._execution = execution
        self._conversation = conversation
        self._created = created
        self._conversations = conversations
        self._clock = clock
        self.result: ChatTurnResult | None = None

    @property
    def conversation_id(self) -> str:
        return self._conversation.conversation_id

    async def events(self) -> AsyncGenerator[ProgressEvent]:
        yield ProgressEvent(
            type=EventType.CONVERSATION,
            data={
                "conversation_id": self.conversation_id,
                "title": self._conversation.title,
                "created": self._created,
            },
        )
        phases: list[str] = []
        async for event in self._execution.events():
            if event.type is EventType.PHASE:
                phases.append(str(event.data["phase"]))
            if event.type is not EventType.DONE:
                yield event
                continue
            result = await self._persist(phases)
            yield ProgressEvent(
                type=EventType.DONE,
                data={
                    **event.data,
                    "conversation_id": self.conversation_id,
                    "message_id": result.message_id,
                },
            )
            if not result.persisted:
                yield ProgressEvent.error("PERSIST_FAILED", PERSIST_FAILED_MESSAGE)

    async def _persist(self, phases: list[str]) -> ChatTurnResult:
        outcome = self._execution.outcome
        if outcome is None:
            raise RuntimeError("runner emitted done without an outcome")
        user_message, assistant_message = build_turn_messages(
            conversation_id=self.conversation_id,
            run=outcome.run,
            answer_text=outcome.answer_text,
            citations=outcome.citations,
            phases=phases,
            now=self._clock.now(),
        )
        self._conversation.record_turn(assistant_message.created_at)
        persisted = True
        try:
            await self._conversations.append_turn(
                self._conversation,
                user_message,
                assistant_message,
                audit_from_run(outcome.run, outcome.citations),
            )
        except Exception:
            logger.exception("failed to persist turn for conversation %s", self.conversation_id)
            persisted = False
        self.result = ChatTurnResult(
            conversation_id=self.conversation_id,
            message_id=assistant_message.message_id if persisted else None,
            run_id=outcome.run.run_id,
            status=outcome.run.status.value,
            content=outcome.answer_text,
            citations=list(outcome.citations),
            phases=phases,
            usage=outcome.run.usage.model_dump(),
            persisted=persisted,
        )
        return self.result


class ChatService:
    def __init__(
        self,
        runner: ChatTurnRunner,
        conversations: ConversationRepository,
        clock: Clock,
        policy: MemoryPolicy,
    ) -> None:
        self._runner = runner
        self._conversations = conversations
        self._clock = clock
        self._policy = policy

    async def open_turn(
        self, *, user_id: str, message: str, conversation_id: str | None
    ) -> ChatSession:
        if conversation_id is None:
            conversation = Conversation.start(
                user_id=user_id, first_message=message, now=self._clock.now()
            )
            await self._conversations.create(conversation)
            context = ConversationContext()
            created = True
        else:
            found = await self._conversations.get(user_id, conversation_id)
            if found is None:
                raise ConversationNotFound(conversation_id)
            conversation = found
            turns = await self._conversations.recent_turns(
                conversation_id, self._policy.context_turns
            )
            context = context_for_rephrase(
                conversation.summary,
                turns,
                max_turns=self._policy.context_turns,
                max_chars=self._policy.context_chars,
            )
            created = False
        execution = self._runner.start(
            user_id=user_id,
            message=message,
            conversation=context,
            conversation_id=conversation.conversation_id,
        )
        return ChatSession(
            execution,
            conversation,
            created=created,
            conversations=self._conversations,
            clock=self._clock,
        )

    async def ask(
        self, *, user_id: str, message: str, conversation_id: str | None
    ) -> ChatTurnResult:
        session = await self.open_turn(
            user_id=user_id, message=message, conversation_id=conversation_id
        )
        async for _ in session.events():
            pass
        if session.result is None:
            raise RuntimeError("chat turn finished without a result")
        return session.result
```

Run: `uv run pytest -q tests/application/test_chat_service.py`
Expected: 4 passed.

- [ ] **Step 5: Full checks and commit**

```bash
git add backend/src/pharma_agent/application backend/tests/memory_repository.py backend/tests/application/test_chat_service.py
git commit -m "feat(application): add ChatService that persists each conversation turn"
```

---

### Task 8: Rolling summary and conversation queries

**Files:**
- Create: `backend/src/pharma_agent/application/memory/__init__.py`, `.../memory/summarize.py`
- Create: `backend/src/pharma_agent/application/conversation/__init__.py`, `.../conversation/queries.py`
- Test: `backend/tests/application/test_summarize.py`, `backend/tests/application/test_queries.py`

**Interfaces:**
- Consumes: `LlmPort`, `LlmRole.SUMMARIZER`, `ConversationSummary`, `summary_messages`, `EXCLUDED_STATUSES`, `ConversationRepository`, `InvalidTitle`, application errors.
- Produces: `SummarizeConversation(llm, conversations, clock, every, max_chars)` with `run_if_needed(user_id, conversation_id) -> bool`; `ConversationView`, `MessageView`; `ConversationQueries(conversations, clock)` with `list_conversations`, `get_conversation`, `list_messages`, `rename`, `delete`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/application/test_summarize.py`:

```python
from datetime import timedelta

from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.conversation.models import ConversationSummary, Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_run
from tests.fakes import NOW, FakeLlm
from tests.memory_repository import InMemoryConversationRepository

OWNER = "a" * 32


async def seeded(repo: InMemoryConversationRepository, statuses: list[str]) -> Conversation:
    conversation = Conversation.start(user_id=OWNER, first_message="hi", now=NOW)
    await repo.create(conversation)
    for index, status in enumerate(statuses):
        run = make_run(f"câu hỏi {index}")
        user_msg, assistant_msg = build_turn_messages(
            conversation_id=conversation.conversation_id, run=run, answer_text=f"trả lời {index}",
            citations=[], phases=[], now=NOW + timedelta(minutes=index),
        )
        user_msg = user_msg.model_copy(update={"status": status})
        assistant_msg = assistant_msg.model_copy(update={"status": status})
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])
    return conversation


def summarizer(llm: FakeLlm, repo: InMemoryConversationRepository) -> SummarizeConversation:
    return SummarizeConversation(llm, repo, FixedClock(NOW + timedelta(hours=1)), every=2, max_chars=20)


async def test_not_due_does_not_call_llm() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed"])
    assert await summarizer(llm, repo).run_if_needed(user_id=OWNER, conversation_id=conversation.conversation_id) is False
    assert llm.calls == []


async def test_due_summary_uses_usable_turns_and_is_truncated() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed", "blocked"])
    llm.script(LlmRole.SUMMARIZER, ConversationSummary(summary="Người dùng hỏi về paracetamol và liều dùng."))

    assert await summarizer(llm, repo).run_if_needed(user_id=OWNER, conversation_id=conversation.conversation_id) is True

    prompt = llm.calls_for(LlmRole.SUMMARIZER)[0][1].content
    assert "câu hỏi 0" in prompt and "câu hỏi 1" not in prompt
    stored = repo.rows[conversation.conversation_id]
    assert stored.summarized_turns == 2 and stored.summary == "Người dùng hỏi về pa"
    assert stored.turn_count == 2


async def test_only_excluded_turns_advance_without_llm() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["error", "timeout"])
    assert await summarizer(llm, repo).run_if_needed(user_id=OWNER, conversation_id=conversation.conversation_id) is False
    assert llm.calls == [] and repo.rows[conversation.conversation_id].summarized_turns == 2


async def test_llm_failure_keeps_previous_summary() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed", "completed"])
    llm.script(LlmRole.SUMMARIZER, LlmError("down"))
    assert await summarizer(llm, repo).run_if_needed(user_id=OWNER, conversation_id=conversation.conversation_id) is False
    assert repo.rows[conversation.conversation_id].summarized_turns == 0


async def test_missing_conversation_is_ignored() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    assert await summarizer(llm, repo).run_if_needed(user_id=OWNER, conversation_id="f" * 32) is False
```

`backend/tests/application/test_queries.py`:

```python
from datetime import timedelta

import pytest

from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.errors import ConversationNotFound, InvalidInput
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_run
from tests.fakes import NOW
from tests.memory_repository import InMemoryConversationRepository

OWNER, STRANGER = "a" * 32, "b" * 32


async def test_list_get_messages_rename_delete() -> None:
    repo = InMemoryConversationRepository()
    queries = ConversationQueries(repo, FixedClock(NOW + timedelta(days=1)))
    older = Conversation.start(user_id=OWNER, first_message="older", now=NOW)
    newer = Conversation.start(user_id=OWNER, first_message="newer", now=NOW + timedelta(minutes=1))
    await repo.create(older)
    await repo.create(newer)
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=older.conversation_id, run=make_run("q"), answer_text="a", citations=[], phases=["answering"], now=NOW
    )
    older.record_turn(NOW + timedelta(minutes=2))
    await repo.append_turn(older, user_msg, assistant_msg, [])

    listed = await queries.list_conversations(OWNER, limit=10)
    assert [c.title for c in listed] == ["older", "newer"] and listed[0].turn_count == 1

    messages = await queries.list_messages(OWNER, older.conversation_id, limit=10)
    assert [(m.role, m.content) for m in messages] == [("user", "q"), ("assistant", "a")]
    assert messages[1].phases == ["answering"]

    renamed = await queries.rename(OWNER, older.conversation_id, "  Thuốc hạ sốt ")
    assert renamed.title == "Thuốc hạ sốt" and renamed.updated_at == NOW + timedelta(days=1)

    with pytest.raises(InvalidInput):
        await queries.rename(OWNER, older.conversation_id, " ")
    for call in (
        queries.get_conversation(STRANGER, older.conversation_id),
        queries.list_messages(STRANGER, older.conversation_id, limit=10),
        queries.rename(STRANGER, older.conversation_id, "x"),
        queries.delete(STRANGER, older.conversation_id),
    ):
        with pytest.raises(ConversationNotFound):
            await call

    await queries.delete(OWNER, older.conversation_id)
    assert [c.title for c in await queries.list_conversations(OWNER, limit=10)] == ["newer"]
```

Run: `uv run pytest -q tests/application/test_summarize.py tests/application/test_queries.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement the summarizer**

`backend/src/pharma_agent/application/memory/__init__.py`: empty.

`backend/src/pharma_agent/application/memory/summarize.py`:

```python
import logging

from pharma_agent.domain.conversation.context import EXCLUDED_STATUSES
from pharma_agent.domain.conversation.models import ConversationSummary
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.conversation.prompts import summary_messages
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.llm.port import LlmError, LlmPort
from pharma_agent.domain.shared.clock import Clock

logger = logging.getLogger(__name__)


class SummarizeConversation:
    """Background job: fold new turns into the rolling summary every `every` turns."""

    def __init__(
        self,
        llm: LlmPort,
        conversations: ConversationRepository,
        clock: Clock,
        *,
        every: int,
        max_chars: int,
    ) -> None:
        self._llm = llm
        self._conversations = conversations
        self._clock = clock
        self._every = every
        self._max_chars = max_chars

    async def run_if_needed(self, *, user_id: str, conversation_id: str) -> bool:
        conversation = await self._conversations.get(user_id, conversation_id)
        if conversation is None or not conversation.needs_summary(self._every):
            return False
        turns = await self._conversations.turns_since(
            conversation_id, conversation.summarized_turns
        )
        covered = conversation.summarized_turns + len(turns)
        usable = [turn for turn in turns if turn.status not in EXCLUDED_STATUSES]
        if not usable:
            conversation.apply_summary(
                conversation.summary, covered_turns=covered, now=self._clock.now()
            )
            await self._conversations.update_summary(conversation)
            return False
        try:
            result, _ = await self._llm.structured(
                LlmRole.SUMMARIZER,
                summary_messages(conversation.summary, usable, max_chars=self._max_chars),
                ConversationSummary,
            )
        except LlmError:
            logger.warning("summary failed for conversation %s", conversation_id, exc_info=True)
            return False
        conversation.apply_summary(
            result.summary[: self._max_chars], covered_turns=covered, now=self._clock.now()
        )
        await self._conversations.update_summary(conversation)
        return True
```

- [ ] **Step 3: Implement the queries**

`backend/src/pharma_agent/application/conversation/__init__.py`: empty.

`backend/src/pharma_agent/application/conversation/queries.py`:

```python
from datetime import datetime

from pydantic import BaseModel, Field

from pharma_agent.application.errors import ConversationNotFound, InvalidInput
from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    InvalidTitle,
    Message,
)
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.shared.clock import Clock


class ConversationView(BaseModel):
    id: str
    title: str
    turn_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, conversation: Conversation) -> "ConversationView":
        return cls(
            id=conversation.conversation_id,
            title=conversation.title,
            turn_count=conversation.turn_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class MessageView(BaseModel):
    id: str
    role: str
    content: str
    status: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def of(cls, message: Message) -> "MessageView":
        return cls(
            id=message.message_id,
            role=message.role.value,
            content=message.content,
            status=message.status,
            citations=list(message.citations),
            phases=list(message.phases),
            created_at=message.created_at,
        )


class ConversationQueries:
    def __init__(self, conversations: ConversationRepository, clock: Clock) -> None:
        self._conversations = conversations
        self._clock = clock

    async def list_conversations(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[ConversationView]:
        rows = await self._conversations.list_for_user(user_id, limit=limit, before=before)
        return [ConversationView.of(row) for row in rows]

    async def get_conversation(self, user_id: str, conversation_id: str) -> ConversationView:
        return ConversationView.of(await self._owned(user_id, conversation_id))

    async def list_messages(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int,
        before: datetime | None = None,
    ) -> list[MessageView]:
        await self._owned(user_id, conversation_id)
        rows = await self._conversations.messages(conversation_id, limit=limit, before=before)
        return [MessageView.of(row) for row in rows]

    async def rename(self, user_id: str, conversation_id: str, title: str) -> ConversationView:
        conversation = await self._owned(user_id, conversation_id)
        try:
            conversation.rename(title, now=self._clock.now())
        except InvalidTitle as exc:
            raise InvalidInput(str(exc)) from exc
        await self._conversations.update_title(conversation)
        return ConversationView.of(conversation)

    async def delete(self, user_id: str, conversation_id: str) -> None:
        if not await self._conversations.delete(user_id, conversation_id):
            raise ConversationNotFound(conversation_id)

    async def _owned(self, user_id: str, conversation_id: str) -> Conversation:
        conversation = await self._conversations.get(user_id, conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)
        return conversation
```

Run: `uv run pytest -q tests/application/test_summarize.py tests/application/test_queries.py`
Expected: 6 passed.

- [ ] **Step 4: Full checks and commit**

```bash
git add backend/src/pharma_agent/application/memory backend/src/pharma_agent/application/conversation backend/tests/application/test_summarize.py backend/tests/application/test_queries.py
git commit -m "feat(application): add rolling conversation summary and conversation queries"
```

---

### Task 9: Postgres checkpointer and the resource container

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/langgraph/__init__.py`, `.../langgraph/checkpointer.py`
- Modify: `backend/src/pharma_agent/infrastructure/composition.py` (`build_application(settings, *, checkpointer=None)`)
- Create: `backend/src/pharma_agent/infrastructure/container.py`
- Test: `backend/tests/infrastructure/test_checkpointer.py`, `backend/tests/infrastructure/test_container.py`

**Interfaces:**
- Consumes: `checkpoint_serializer`, `build_chat_graph`, `Database`, `PostgresConversationRepository`, `AuditContext`, `ChatService`, `MemoryPolicy`, `SummarizeConversation`, `ConversationQueries`, `Settings`.
- Produces: `open_postgres_checkpointer(conninfo, max_size=10)` async context manager yielding `AsyncPostgresSaver`; `Container(settings, sessions, queries, chat, summarizer, health_checks)`; `open_container(settings)` async context manager; `ContainerFactory` type alias.

- [ ] **Step 1: Write the failing tests**

`backend/tests/infrastructure/test_checkpointer.py`:

```python
import warnings

import pytest
from sqlalchemy.engine import make_url

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.agent.schemas import JudgeDecision, JudgeOutcome
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.langgraph.checkpointer import open_postgres_checkpointer
from tests.application.test_chat_graph import QUESTION, passing_llm
from tests.domain.factories import make_hit
from tests.fakes import FakeRetriever, build_deps

pytestmark = pytest.mark.integration


async def test_graph_state_is_checkpointed_in_postgres(fresh_database_dsn: str) -> None:
    conninfo = make_url(fresh_database_dsn).set(drivername="postgresql").render_as_string(hide_password=False)
    llm = passing_llm()
    llm.script(LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ"))
    async with open_postgres_checkpointer(conninfo, max_size=2) as saver:
        graph = build_chat_graph(checkpointer=saver)
        runner = ChatTurnRunner(graph, build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])), BudgetLimits())
        execution = runner.start(user_id="u1", message=QUESTION)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            async for _ in execution.events():
                pass
            snapshot = await graph.aget_state({"configurable": {"thread_id": execution.run.run_id}})
    run = snapshot.values["run"] if isinstance(snapshot.values, dict) else snapshot.values.run
    assert isinstance(run, AgentRun) and run.status.value == "completed"
```

`backend/tests/infrastructure/test_container.py`:

```python
import pytest

from pharma_agent.infrastructure.container import open_container
from pharma_agent.infrastructure.settings import Settings

pytestmark = pytest.mark.integration


async def test_container_without_llm_serves_queries_only(migrated_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is None and container.summarizer is None
        assert await container.health_checks["postgres"]() is True
        assert await container.queries.list_conversations("a" * 32, limit=5) == []


async def test_container_with_llm_builds_chat(migrated_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is not None and container.summarizer is not None
        assert set(container.health_checks) == {"postgres", "qdrant"}
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_checkpointer.py tests/infrastructure/test_container.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Checkpointer**

`backend/src/pharma_agent/infrastructure/langgraph/__init__.py`: empty.

`backend/src/pharma_agent/infrastructure/langgraph/checkpointer.py`:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from pharma_agent.application.chat.checkpoint import checkpoint_serializer


@asynccontextmanager
async def open_postgres_checkpointer(
    conninfo: str, *, max_size: int = 10
) -> AsyncIterator[AsyncPostgresSaver]:
    """Pooled AsyncPostgresSaver with the strict msgpack allowlist; creates its tables on start."""
    async with AsyncConnectionPool(
        conninfo,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        saver = AsyncPostgresSaver(pool, serde=checkpoint_serializer())
        await saver.setup()
        yield saver
```

- [ ] **Step 3: Let composition accept a checkpointer**

In `backend/src/pharma_agent/infrastructure/composition.py`:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
```

Change the signature and graph construction:

```python
def build_application(
    settings: Settings, *, checkpointer: BaseCheckpointSaver | None = None
) -> Application:
    ...
    runner = ChatTurnRunner(build_chat_graph(checkpointer=checkpointer), deps, settings.budget)
```

- [ ] **Step 4: Container**

`backend/src/pharma_agent/infrastructure/container.py`:

```python
"""Owns every long-lived resource of the HTTP service and wires application services."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.composition import Application, build_application
from pharma_agent.infrastructure.langgraph.checkpointer import open_postgres_checkpointer
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.settings import Settings

HealthCheck = Callable[[], Awaitable[bool]]


@dataclass
class Container:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession] | None
    queries: ConversationQueries
    chat: ChatService | None = None
    summarizer: SummarizeConversation | None = None
    health_checks: dict[str, HealthCheck] = field(default_factory=dict)


ContainerFactory = Callable[[Settings], AbstractAsyncContextManager[Container]]


def _qdrant_check(agent: Application, dimension: int) -> HealthCheck:
    async def check() -> bool:
        try:
            await agent.retriever.verify_collection(dimension)
        except RetrievalError:
            return False
        return True

    return check


@asynccontextmanager
async def open_container(settings: Settings) -> AsyncIterator[Container]:
    database = Database(
        settings.postgres.dsn, pool_size=settings.postgres.pool_size, echo=settings.postgres.echo
    )
    clock = SystemClock()
    repository = PostgresConversationRepository(
        database.sessions,
        AuditContext(
            corpus_version=settings.retrieval.collection_alias,
            embedding_model=settings.retrieval.embedding.model,
            retriever_config=settings.retrieval.model_dump(
                mode="json", exclude={"embedding": {"api_key"}}
            ),
        ),
    )
    container = Container(
        settings=settings,
        sessions=database.sessions,
        queries=ConversationQueries(repository, clock),
        health_checks={"postgres": database.ping},
    )
    try:
        async with AsyncExitStack() as stack:
            if settings.llm.configured:
                checkpointer = await stack.enter_async_context(
                    open_postgres_checkpointer(settings.postgres.conninfo)
                )
                agent = build_application(settings, checkpointer=checkpointer)
                stack.push_async_callback(agent.aclose)
                container.chat = ChatService(
                    agent.runner,
                    repository,
                    clock,
                    MemoryPolicy(
                        context_turns=settings.memory.context_turns,
                        context_chars=settings.memory.context_chars,
                    ),
                )
                container.summarizer = SummarizeConversation(
                    agent.deps.llm,
                    repository,
                    clock,
                    every=settings.memory.summary_every_turns,
                    max_chars=settings.memory.summary_max_chars,
                )
                container.health_checks["qdrant"] = _qdrant_check(
                    agent, settings.retrieval.embedding.dimension
                )
            yield container
    finally:
        await database.dispose()
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_checkpointer.py tests/infrastructure/test_container.py`
Expected: 3 passed.

- [ ] **Step 5: Full checks and commit**

```bash
git add backend/src/pharma_agent/infrastructure/langgraph backend/src/pharma_agent/infrastructure/composition.py backend/src/pharma_agent/infrastructure/container.py backend/tests/infrastructure/test_checkpointer.py backend/tests/infrastructure/test_container.py
git commit -m "feat(infra): add Postgres checkpointer and service container"
```

---

### Task 10: Authentication with fastapi-users (email/password + Google)

**Files:**
- Create: `backend/src/pharma_agent/infrastructure/auth/__init__.py`, `.../auth/users.py`
- Test: `backend/tests/infrastructure/test_auth_wiring.py`

**Interfaces:**
- Consumes: `AuthSettings`, `UserTable`, `OAuthAccountTable`.
- Produces: `UserRead`, `UserCreate`, `UserUpdate`; `UserManager`; `SessionFactoryResolver = Callable[[Request], async_sessionmaker[AsyncSession]]`; `Auth(users, backend, current_active_user, google, secret, frontend_url)`; `build_auth(settings, resolve_sessions) -> Auth`; `include_auth_routes(router, auth)` mounting `/auth/jwt`, `/auth/register`, `/users`, and `/auth/google` when configured.

- [ ] **Step 1: Write the failing unit test**

`backend/tests/infrastructure/test_auth_wiring.py`:

```python
import pytest
from fastapi import APIRouter, HTTPException, Request
from pydantic import SecretStr

from pharma_agent.infrastructure.auth.users import build_auth, include_auth_routes
from pharma_agent.infrastructure.settings import AuthSettings

SECRET = "s" * 40


def no_sessions(request: Request):
    raise HTTPException(status_code=503)


def paths(router: APIRouter) -> set[str]:
    return {getattr(route, "path", "") for route in router.routes}


def test_routes_without_google() -> None:
    auth = build_auth(AuthSettings(jwt_secret=SecretStr(SECRET)), no_sessions)
    router = APIRouter()
    include_auth_routes(router, auth)
    assert {"/auth/jwt/login", "/auth/jwt/logout", "/auth/register", "/users/me"} <= paths(router)
    assert not any(path.startswith("/auth/google") for path in paths(router))
    assert auth.google is None


def test_routes_with_google() -> None:
    auth = build_auth(
        AuthSettings(
            jwt_secret=SecretStr(SECRET),
            google_client_id="id",
            google_client_secret=SecretStr("secret"),
            frontend_url="https://app.example",
        ),
        no_sessions,
    )
    router = APIRouter()
    include_auth_routes(router, auth)
    assert {"/auth/google/authorize", "/auth/google/callback"} <= paths(router)


def test_missing_secret_fails_fast() -> None:
    with pytest.raises(ValueError, match="PHARMA_AUTH__JWT_SECRET"):
        build_auth(AuthSettings(), no_sessions)
```

Run: `uv run pytest -q tests/infrastructure/test_auth_wiring.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement**

`backend/src/pharma_agent/infrastructure/auth/__init__.py`: empty.

`backend/src/pharma_agent/infrastructure/auth/users.py`:

```python
"""fastapi-users wiring: JWT bearer auth, registration, current user, Google OAuth."""

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.infrastructure.persistence.postgres.tables import OAuthAccountTable, UserTable
from pharma_agent.infrastructure.settings import AuthSettings

LOGIN_URL = "/api/v1/auth/jwt/login"

SessionFactoryResolver = Callable[[Request], async_sessionmaker[AsyncSession]]
UserDatabase = SQLAlchemyUserDatabase[UserTable, uuid.UUID]


class UserRead(schemas.BaseUser[uuid.UUID]):
    display_name: str = ""


class UserCreate(schemas.BaseUserCreate):
    display_name: str = ""


class UserUpdate(schemas.BaseUserUpdate):
    display_name: str | None = None


class UserManager(UUIDIDMixin, BaseUserManager[UserTable, uuid.UUID]):
    def __init__(self, user_db: UserDatabase, secret: str) -> None:
        super().__init__(user_db)
        self.reset_password_token_secret = secret
        self.verification_token_secret = secret


@dataclass(frozen=True)
class Auth:
    users: FastAPIUsers[UserTable, uuid.UUID]
    backend: AuthenticationBackend[UserTable, uuid.UUID]
    current_active_user: Callable[..., Any]
    google: GoogleOAuth2 | None
    secret: str
    frontend_url: str


def build_auth(settings: AuthSettings, resolve_sessions: SessionFactoryResolver) -> Auth:
    secret = settings.require_jwt_secret()

    async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
        async with resolve_sessions(request)() as session:
            yield session

    async def get_user_db(
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> AsyncIterator[UserDatabase]:
        yield SQLAlchemyUserDatabase(session, UserTable, OAuthAccountTable)

    async def get_user_manager(
        user_db: Annotated[UserDatabase, Depends(get_user_db)],
    ) -> AsyncIterator[UserManager]:
        yield UserManager(user_db, secret)

    def get_jwt_strategy() -> JWTStrategy[UserTable, uuid.UUID]:
        return JWTStrategy(secret=secret, lifetime_seconds=settings.jwt_lifetime_seconds)

    backend = AuthenticationBackend(
        name="jwt",
        transport=BearerTransport(tokenUrl=LOGIN_URL),
        get_strategy=get_jwt_strategy,
    )
    users = FastAPIUsers[UserTable, uuid.UUID](get_user_manager, [backend])
    google = None
    if settings.google_enabled and settings.google_client_id and settings.google_client_secret:
        google = GoogleOAuth2(
            settings.google_client_id, settings.google_client_secret.get_secret_value()
        )
    return Auth(
        users=users,
        backend=backend,
        current_active_user=users.current_user(active=True),
        google=google,
        secret=secret,
        frontend_url=settings.frontend_url,
    )


def include_auth_routes(router: APIRouter, auth: Auth) -> None:
    router.include_router(
        auth.users.get_auth_router(auth.backend), prefix="/auth/jwt", tags=["auth"]
    )
    router.include_router(
        auth.users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
    )
    router.include_router(
        auth.users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"]
    )
    if auth.google is not None:
        router.include_router(
            auth.users.get_oauth_router(
                auth.google,
                auth.backend,
                auth.secret,
                redirect_url=f"{auth.frontend_url.rstrip('/')}/auth/google/callback",
                associate_by_email=True,
                is_verified_by_default=True,
            ),
            prefix="/auth/google",
            tags=["auth"],
        )
```

Run: `uv run pytest -q tests/infrastructure/test_auth_wiring.py`
Expected: 3 passed. If pyrefly rejects a fastapi-users generic (for example the `FastAPIUsers[...]` specialisation), fix the annotation to what the library declares rather than suppressing.

- [ ] **Step 3: Full checks and commit**

```bash
git add backend/src/pharma_agent/infrastructure/auth backend/tests/infrastructure/test_auth_wiring.py
git commit -m "feat(infra): wire fastapi-users JWT auth with optional Google OAuth"
```

---

### Task 11: FastAPI application, SSE chat, conversations and health

**Files:**
- Create: `backend/src/pharma_agent/api/__init__.py`, `api/routers/__init__.py`
- Create: `backend/src/pharma_agent/api/app.py`, `api/deps.py`, `api/errors.py`, `api/schemas.py`, `api/sse.py`
- Create: `backend/src/pharma_agent/api/routers/health.py`, `routers/chat.py`, `routers/conversations.py`
- Create: `backend/tests/api/__init__.py`, `backend/tests/api/harness.py`
- Test: `backend/tests/api/test_chat_api.py`, `backend/tests/api/test_conversations_api.py`, `backend/tests/api/test_health_api.py`

**Interfaces:**
- Consumes: `Container`, `ContainerFactory`, `open_container`, `build_auth`, `include_auth_routes`, `UserTable`, `ChatService`, `ChatTurnResult`, `ConversationQueries`, `ConversationView`, `MessageView`, `SummarizeConversation`, `ProgressEvent`, application errors.
- Produces: `create_app(settings=None, *, container_factory=open_container) -> FastAPI`; routes under `/api/v1`: `POST /chat`, `POST /chat/stream` (SSE), `GET /conversations`, `GET|PATCH|DELETE /conversations/{id}`, `GET /conversations/{id}/messages`, `GET /health`; error body `{"code", "message"}`.

- [ ] **Step 1: Test harness**

`backend/tests/api/__init__.py`: empty.

`backend/tests/api/harness.py`:

```python
"""Build the real FastAPI app over in-memory services and a fake authenticated user."""

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi.testclient import TestClient

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable
from pharma_agent.infrastructure.settings import Settings
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps
from tests.memory_repository import InMemoryConversationRepository

OWNER = uuid.UUID(hex="a" * 32)


@dataclass
class Harness:
    client: TestClient
    llm: FakeLlm
    repo: InMemoryConversationRepository
    container: Container


def settings() -> Settings:
    return Settings(_env_file=None, auth={"jwt_secret": "s" * 40})


def build_harness(
    *,
    agent: bool = True,
    authenticated: bool = True,
    health: dict[str, Callable[[], Awaitable[bool]]] | None = None,
) -> Harness:
    llm = FakeLlm()
    repo = InMemoryConversationRepository()
    clock = SystemClock()
    retriever = FakeRetriever(*[[make_hit(f"c{i}", fusion=0.9)] for i in range(10)])
    runner = ChatTurnRunner(build_chat_graph(), build_deps(llm, retriever), BudgetLimits())
    container = Container(
        settings=settings(),
        sessions=None,
        queries=ConversationQueries(repo, clock),
        chat=ChatService(runner, repo, clock, MemoryPolicy()) if agent else None,
        summarizer=SummarizeConversation(llm, repo, clock, every=2, max_chars=500) if agent else None,
        health_checks=health if health is not None else {"postgres": _ok},
    )

    @asynccontextmanager
    async def factory(_: Settings) -> AsyncIterator[Container]:
        yield container

    app = create_app(settings(), container_factory=factory)
    if authenticated:
        auth = app.state.auth
        app.dependency_overrides[auth.current_active_user] = lambda: UserTable(
            id=OWNER, email="owner@example.com", hashed_password="x", is_active=True
        )
    return Harness(client=TestClient(app), llm=llm, repo=repo, container=container)


async def _ok() -> bool:
    return True


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data += line.removeprefix("data:").strip()
        if name and data:
            events.append((name, json.loads(data)))
    return events
```

- [ ] **Step 2: Write the failing API tests**

`backend/tests/api/test_chat_api.py`:

```python
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
    SkillSelection,
)
from pharma_agent.domain.conversation.models import ConversationSummary
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import LlmRole
from tests.api.harness import OWNER, build_harness, parse_sse
from tests.fakes import FakeLlm


def script_turn(llm: FakeLlm) -> None:
    llm.script(LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok"))
    llm.script(
        LlmRole.REPHRASE,
        RephraseResult(standalone_query="Liều paracetamol", audience=Audience.GENERAL_PUBLIC, language=Language.VI, intent=Intent.PHARMA_QUESTION),
    )
    llm.script(LlmRole.SKILL_SELECTOR, SkillSelection(skill_ids=[]))
    llm.script(LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ"))


def test_stream_emits_conversation_first_and_done_last() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    with harness.client as client:
        response = client.post("/api/v1/chat/stream", json={"message": "Paracetamol uống bao nhiêu?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)

    names = [name for name, _ in events]
    assert names[0] == "conversation" and names[-1] == "done"
    assert {"phase", "evidence", "token", "citations"} <= set(names)
    done = events[-1][1]
    conversation_id = str(events[0][1]["conversation_id"])
    assert done["status"] == "completed" and done["message_id"] and done["conversation_id"] == conversation_id
    assert harness.repo.rows[conversation_id].user_id == OWNER.hex


def test_non_stream_chat_and_background_summary() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    script_turn(harness.llm)
    harness.llm.script(LlmRole.SUMMARIZER, ConversationSummary(summary="Hỏi về liều paracetamol."))
    with harness.client as client:
        first = client.post("/api/v1/chat", json={"message": "Paracetamol?"})
        assert first.status_code == 200
        body = first.json()
        assert body["persisted"] is True and body["citations"]
        second = client.post("/api/v1/chat", json={"message": "Còn trẻ em?", "conversation_id": body["conversation_id"]})
        assert second.status_code == 200
    stored = harness.repo.rows[body["conversation_id"]]
    assert stored.turn_count == 2 and stored.summary == "Hỏi về liều paracetamol."


def test_chat_errors() -> None:
    harness = build_harness()
    with harness.client as client:
        missing = client.post("/api/v1/chat", json={"message": "hi", "conversation_id": "f" * 32})
        assert missing.status_code == 404 and missing.json()["code"] == "CONVERSATION_NOT_FOUND"
        assert client.post("/api/v1/chat", json={"message": "hi", "conversation_id": "nope"}).status_code == 422
        assert client.post("/api/v1/chat", json={"message": ""}).status_code == 422

    unavailable = build_harness(agent=False)
    with unavailable.client as client:
        response = client.post("/api/v1/chat/stream", json={"message": "hi"})
        assert response.status_code == 503 and response.json()["code"] == "AGENT_UNAVAILABLE"

    anonymous = build_harness(authenticated=False)
    with anonymous.client as client:
        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 401
```

`backend/tests/api/test_conversations_api.py`:

```python
from datetime import UTC, datetime, timedelta

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from tests.api.harness import OWNER, build_harness
from tests.domain.factories import make_run


def test_conversation_endpoints() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    mine = Conversation.start(user_id=OWNER.hex, first_message="Paracetamol", now=now)
    foreign = Conversation.start(user_id="b" * 32, first_message="Không phải của tôi", now=now)
    with harness.client as client:
        import asyncio

        async def seed() -> None:
            await harness.repo.create(mine)
            await harness.repo.create(foreign)
            run = make_run("q")
            run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=now)
            run.complete()
            user_msg, assistant_msg = build_turn_messages(
                conversation_id=mine.conversation_id, run=run, answer_text="a", citations=[], phases=[], now=now + timedelta(seconds=1)
            )
            mine.record_turn(now + timedelta(seconds=1))
            await harness.repo.append_turn(mine, user_msg, assistant_msg, [])

        asyncio.run(seed())

        listed = client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert [item["title"] for item in listed.json()] == ["Paracetamol"]

        detail = client.get(f"/api/v1/conversations/{mine.conversation_id}")
        assert detail.json()["turn_count"] == 1

        messages = client.get(f"/api/v1/conversations/{mine.conversation_id}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]

        renamed = client.patch(f"/api/v1/conversations/{mine.conversation_id}", json={"title": "Thuốc hạ sốt"})
        assert renamed.status_code == 200 and renamed.json()["title"] == "Thuốc hạ sốt"
        assert client.patch(f"/api/v1/conversations/{mine.conversation_id}", json={"title": ""}).status_code == 422

        assert client.get(f"/api/v1/conversations/{foreign.conversation_id}").status_code == 404
        assert client.delete(f"/api/v1/conversations/{foreign.conversation_id}").status_code == 404
        assert client.delete(f"/api/v1/conversations/{mine.conversation_id}").status_code == 204
        assert client.get("/api/v1/conversations").json() == []
```

`backend/tests/api/test_health_api.py`:

```python
from tests.api.harness import build_harness


async def _down() -> bool:
    return False


async def _up() -> bool:
    return True


def test_health_ok_and_degraded() -> None:
    ok = build_harness(authenticated=False, health={"postgres": _up})
    with ok.client as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent": True, "checks": {"postgres": True}}

    degraded = build_harness(authenticated=False, agent=False, health={"postgres": _up, "qdrant": _down})
    with degraded.client as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json() == {"status": "degraded", "agent": False, "checks": {"postgres": True, "qdrant": False}}
```

Note: `test_conversation_endpoints` seeds the in-memory repository with `asyncio.run` before any request; the repository has no event-loop affinity, so this is safe.

Run: `uv run pytest -q tests/api`
Expected: FAIL (`ModuleNotFoundError: pharma_agent.api.app`).

- [ ] **Step 3: Schemas, SSE mapping, errors and dependencies**

`backend/src/pharma_agent/api/__init__.py`, `api/routers/__init__.py`: empty.

`backend/src/pharma_agent/api/schemas.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field

CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    agent: bool
    checks: dict[str, bool]


class ErrorResponse(BaseModel):
    code: str
    message: str
```

`backend/src/pharma_agent/api/sse.py`:

```python
import json

from sse_starlette import ServerSentEvent

from pharma_agent.application.progress import ProgressEvent


def to_server_sent_event(event: ProgressEvent) -> ServerSentEvent:
    return ServerSentEvent(
        data=json.dumps(event.data, ensure_ascii=False, default=str),
        event=event.type.value,
    )
```

`backend/src/pharma_agent/api/errors.py`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pharma_agent.application.errors import (
    AgentUnavailable,
    ApplicationError,
    ConversationNotFound,
    InvalidInput,
)

STATUS_BY_ERROR: dict[type[ApplicationError], int] = {
    ConversationNotFound: 404,
    InvalidInput: 422,
    AgentUnavailable: 503,
}


async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApplicationError):
        raise exc
    status = next(
        (code for kind, code in STATUS_BY_ERROR.items() if isinstance(exc, kind)), 400
    )
    return JSONResponse(
        status_code=status, content={"code": exc.code, "message": str(exc) or exc.code}
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApplicationError, application_error_handler)
```

`backend/src/pharma_agent/api/deps.py`:

```python
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService
from pharma_agent.application.errors import AgentUnavailable
from pharma_agent.infrastructure.auth.users import Auth
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable

UserIdDependency = Callable[..., Awaitable[str]]


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, Container):
        raise HTTPException(status_code=503, detail="service is starting")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    sessions = get_container(request).sessions
    if sessions is None:
        raise HTTPException(status_code=503, detail="database is not configured")
    return sessions


def require_chat(container: Container) -> ChatService:
    if container.chat is None:
        raise AgentUnavailable("the agent is not configured (set PHARMA_LLM__DEFAULT__API_KEY)")
    return container.chat


def user_id_dependency(auth: Auth) -> UserIdDependency:
    async def current_user_id(
        user: Annotated[UserTable, Depends(auth.current_active_user)],
    ) -> str:
        return user.id.hex

    return current_user_id
```

- [ ] **Step 4: Routers**

`backend/src/pharma_agent/api/routers/health.py`:

```python
import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pharma_agent.api.deps import ContainerDep
from pharma_agent.api.schemas import HealthResponse


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthResponse)
    async def health(container: ContainerDep) -> JSONResponse:
        names = list(container.health_checks)
        results = await asyncio.gather(*(container.health_checks[name]() for name in names))
        checks = dict(zip(names, results, strict=True))
        healthy = all(checks.values())
        body = HealthResponse(
            status="ok" if healthy else "degraded",
            agent=container.chat is not None,
            checks=checks,
        )
        return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())

    return router
```

`backend/src/pharma_agent/api/routers/chat.py`:

```python
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.background import BackgroundTask

from pharma_agent.api.deps import ContainerDep, UserIdDependency, require_chat
from pharma_agent.api.schemas import ChatRequest
from pharma_agent.api.sse import to_server_sent_event
from pharma_agent.application.chat.service import ChatTurnResult
from pharma_agent.application.memory.summarize import SummarizeConversation

logger = logging.getLogger(__name__)


async def summarize_quietly(
    summarizer: SummarizeConversation | None, user_id: str, conversation_id: str
) -> None:
    if summarizer is None:
        return
    try:
        await summarizer.run_if_needed(user_id=user_id, conversation_id=conversation_id)
    except Exception:
        logger.exception("background summary failed for %s", conversation_id)


def build_chat_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(prefix="/chat", tags=["chat"])
    UserId = Annotated[str, Depends(current_user_id)]

    @router.post("", response_model=ChatTurnResult)
    async def chat(
        body: ChatRequest,
        user_id: UserId,
        container: ContainerDep,
        background: BackgroundTasks,
    ) -> ChatTurnResult:
        service = require_chat(container)
        result = await service.ask(
            user_id=user_id, message=body.message, conversation_id=body.conversation_id
        )
        background.add_task(
            summarize_quietly, container.summarizer, user_id, result.conversation_id
        )
        return result

    @router.post("/stream")
    async def chat_stream(
        body: ChatRequest, user_id: UserId, container: ContainerDep
    ) -> EventSourceResponse:
        service = require_chat(container)
        session = await service.open_turn(
            user_id=user_id, message=body.message, conversation_id=body.conversation_id
        )

        async def stream() -> AsyncIterator[ServerSentEvent]:
            async for event in session.events():
                yield to_server_sent_event(event)

        return EventSourceResponse(
            stream(),
            ping=15,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(
                summarize_quietly, container.summarizer, user_id, session.conversation_id
            ),
        )

    return router
```

`backend/src/pharma_agent/api/routers/conversations.py`:

```python
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response

from pharma_agent.api.deps import ContainerDep, UserIdDependency
from pharma_agent.api.schemas import CONVERSATION_ID_PATTERN, RenameConversationRequest
from pharma_agent.application.conversation.queries import ConversationView, MessageView

ConversationId = Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN)]


def build_conversations_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(prefix="/conversations", tags=["conversations"])
    UserId = Annotated[str, Depends(current_user_id)]

    @router.get("", response_model=list[ConversationView])
    async def list_conversations(
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        before: datetime | None = None,
    ) -> list[ConversationView]:
        return await container.queries.list_conversations(user_id, limit=limit, before=before)

    @router.get("/{conversation_id}", response_model=ConversationView)
    async def get_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> ConversationView:
        return await container.queries.get_conversation(user_id, conversation_id)

    @router.get("/{conversation_id}/messages", response_model=list[MessageView])
    async def list_messages(
        conversation_id: ConversationId,
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        before: datetime | None = None,
    ) -> list[MessageView]:
        return await container.queries.list_messages(
            user_id, conversation_id, limit=limit, before=before
        )

    @router.patch("/{conversation_id}", response_model=ConversationView)
    async def rename_conversation(
        conversation_id: ConversationId,
        body: RenameConversationRequest,
        user_id: UserId,
        container: ContainerDep,
    ) -> ConversationView:
        return await container.queries.rename(user_id, conversation_id, body.title)

    @router.delete("/{conversation_id}", status_code=204)
    async def delete_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> Response:
        await container.queries.delete(user_id, conversation_id)
        return Response(status_code=204)

    return router
```

- [ ] **Step 5: Application factory**

`backend/src/pharma_agent/api/app.py`:

```python
"""FastAPI application factory: `uvicorn pharma_agent.api.app:create_app --factory`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pharma_agent.api.deps import session_factory, user_id_dependency
from pharma_agent.api.errors import install_error_handlers
from pharma_agent.api.routers.chat import build_chat_router
from pharma_agent.api.routers.conversations import build_conversations_router
from pharma_agent.api.routers.health import build_health_router
from pharma_agent.infrastructure.auth.users import build_auth, include_auth_routes
from pharma_agent.infrastructure.container import ContainerFactory, open_container
from pharma_agent.infrastructure.settings import Settings


def create_app(
    settings: Settings | None = None, *, container_factory: ContainerFactory = open_container
) -> FastAPI:
    resolved = settings if settings is not None else Settings()
    auth = build_auth(resolved.auth, session_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with container_factory(resolved) as container:
            app.state.container = container
            yield

    app = FastAPI(title="Pharma Agent API", version="0.1.0", lifespan=lifespan)
    app.state.auth = auth
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix="/api/v1")
    include_auth_routes(api, auth)
    current_user_id = user_id_dependency(auth)
    api.include_router(build_health_router())
    api.include_router(build_chat_router(current_user_id))
    api.include_router(build_conversations_router(current_user_id))
    app.include_router(api)
    return app
```

Run: `uv run pytest -q tests/api`
Expected: 5 passed. Layering test still passes (`api/` imports application and infrastructure only).

- [ ] **Step 6: Full checks and commit**

```bash
git add backend/src/pharma_agent/api backend/tests/api
git commit -m "feat(api): add FastAPI app with SSE chat, conversations and health endpoints"
```

---

### Task 12: CLI serve/migrate, end-to-end integration and docs

**Files:**
- Modify: `backend/src/pharma_agent/cli.py` (commands `serve`, `migrate`)
- Modify: `backend/README.md`
- Test: `backend/tests/test_cli.py` (extend), `backend/tests/api/test_e2e_postgres.py`

**Interfaces:**
- Consumes: `create_app`, `alembic_config`, `Container`, `PostgresConversationRepository`, `open_postgres_checkpointer`, harness helpers.
- Produces: `pharma-agent serve [--host --port --reload]`, `pharma-agent migrate [REVISION]`.

- [ ] **Step 1: Write the failing CLI tests**

Append to `backend/tests/test_cli.py`:

```python
def test_migrate_upgrades_to_head(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("alembic.command.upgrade", lambda config, revision: calls.append((config, revision)))
    result = CliRunner().invoke(cli.app, ["migrate"])
    assert result.exit_code == 0, result.output
    config, revision = calls[0]
    assert revision == "head"
    assert config.get_main_option("script_location").endswith(":migrations")


def test_serve_runs_uvicorn_factory(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.update(app=app, **kwargs))
    monkeypatch.setenv("PHARMA_API__PORT", "9001")
    result = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert calls["app"] == "pharma_agent.api.app:create_app" and calls["factory"] is True
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)
```

Run: `uv run pytest -q tests/test_cli.py`
Expected: FAIL (`No such command 'migrate'`).

- [ ] **Step 2: Implement the commands**

Add to `backend/src/pharma_agent/cli.py`:

```python
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

    from pharma_agent.infrastructure.persistence.postgres.alembic_config import alembic_config

    command.upgrade(alembic_config(), revision)
```

Run: `uv run pytest -q tests/test_cli.py`
Expected: all CLI tests pass.

- [ ] **Step 3: End-to-end test over real Postgres**

`backend/tests/api/test_e2e_postgres.py`:

```python
"""HTTP → auth → ChatService → Postgres repository, with the fake LLM/retriever of Plan 1."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.tables import MessageTable, RetrievalRunTable
from pharma_agent.infrastructure.settings import Settings
from tests.api.harness import parse_sse
from tests.api.test_chat_api import script_turn
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps

pytestmark = pytest.mark.integration


def test_register_login_stream_and_persist(migrated_dsn: str) -> None:
    settings = Settings(_env_file=None, auth={"jwt_secret": "s" * 40}, postgres={"dsn": migrated_dsn})
    llm = FakeLlm()
    script_turn(llm)
    database_holder: dict[str, Database] = {}

    @asynccontextmanager
    async def factory(resolved: Settings) -> AsyncIterator[Container]:
        database = Database(resolved.postgres.dsn, pool_size=2)
        database_holder["db"] = database
        repo = PostgresConversationRepository(
            database.sessions, AuditContext(corpus_version="test", embedding_model="test")
        )
        runner = ChatTurnRunner(
            build_chat_graph(), build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])), BudgetLimits()
        )
        clock = SystemClock()
        try:
            yield Container(
                settings=resolved,
                sessions=database.sessions,
                queries=ConversationQueries(repo, clock),
                chat=ChatService(runner, repo, clock, MemoryPolicy()),
                health_checks={"postgres": database.ping},
            )
        finally:
            await database.dispose()

    app = create_app(settings, container_factory=factory)
    email, password = f"{uuid4().hex[:10]}@example.com", "S3cure-password!"
    with TestClient(app) as client:
        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 401
        registered = client.post("/api/v1/auth/register", json={"email": email, "password": password, "display_name": "An"})
        assert registered.status_code == 201, registered.text
        token = client.post("/api/v1/auth/jwt/login", data={"username": email, "password": password}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/v1/users/me", headers=headers).json()["display_name"] == "An"

        response = client.post("/api/v1/chat/stream", json={"message": "Paracetamol uống bao nhiêu?"}, headers=headers)
        events = parse_sse(response.text)
        conversation_id = str(events[0][1]["conversation_id"])
        assert events[-1][0] == "done" and events[-1][1]["message_id"]

        listed = client.get("/api/v1/conversations", headers=headers).json()
        assert [item["id"] for item in listed] == [conversation_id]
        messages = client.get(f"/api/v1/conversations/{conversation_id}/messages", headers=headers).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]

        async def count_rows() -> tuple[int, int]:
            async with database_holder["db"].sessions() as session:
                message_count = (await session.execute(select(func.count()).select_from(MessageTable))).scalar_one()
                audit_count = (await session.execute(select(func.count()).select_from(RetrievalRunTable))).scalar_one()
            return message_count, audit_count

        message_count, audit_count = client.portal.call(count_rows)
        assert message_count >= 2 and audit_count >= 1
```

Run: `uv run pytest -q -m integration tests/api/test_e2e_postgres.py`
Expected: 1 passed. `client.portal` runs the coroutine on the TestClient's event loop, where the engine's connections live.

- [ ] **Step 4: README**

Add to `backend/README.md` after "Chạy lần đầu":

```markdown
## Chạy HTTP API

```bash
docker compose up -d postgres qdrant llama-embedding llama-reranker   # từ repo root
cd backend
uv run pharma-agent migrate
uv run pharma-agent serve            # http://127.0.0.1:8000/docs
```

Luồng dùng thử bằng curl:

```bash
curl -s -X POST localhost:8000/api/v1/auth/register -H 'content-type: application/json' \
  -d '{"email":"an@example.com","password":"S3cure-password!"}'
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/jwt/login \
  -d 'username=an@example.com&password=S3cure-password!' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -N -X POST localhost:8000/api/v1/chat/stream -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message":"Paracetamol người lớn uống bao nhiêu?"}'
```

SSE trả lần lượt `conversation`, `phase`, `skills_selected`, `evidence`, `token`, `citations`, `done`
(và `error` với mã `PERSIST_FAILED` nếu không lưu được lượt hội thoại).
```

- [ ] **Step 5: Final verification and commit**

Run the Global Constraints check command, then `uv run pytest -q -m integration` (all integration tests), then from the repo root `uv run --project backend pre-commit run --all-files`.
Expected: everything green.

```bash
git add backend/src/pharma_agent/cli.py backend/README.md backend/tests/test_cli.py backend/tests/api/test_e2e_postgres.py
git commit -m "feat(backend): add serve/migrate commands, Postgres end-to-end test and API docs"
```

---

## Out of scope (Plan 3)

Skill upload/enable/delete API backed by a Postgres `skills` table, feedback endpoint with Langfuse scores, Langfuse `CallbackHandler` on the graph, checkpoint cleanup job.
