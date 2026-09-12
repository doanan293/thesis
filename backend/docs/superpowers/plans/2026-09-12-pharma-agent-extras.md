# Pharma Agent Extras Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the spec: user-uploaded skills stored in Postgres next to the system skills, message feedback that also lands as a Langfuse score, Langfuse tracing of every chat turn (graph callbacks + OpenAI generations under one trace), and a checkpoint cleanup job.

**Architecture:** Domain gains `Feedback`, a `FeedbackRepository` port, a `SkillRepository` port (superset of `SkillCatalog`) and a slug helper. Application gains `SkillService`, `FeedbackService`, a `TurnTracer` port that the runner uses to wrap each graph run, and a `ScoreSink` port. Infrastructure adds `PostgresSkillRepository`, `PostgresFeedbackRepository`, `LangfuseTracing` (implements both ports; `NullTracing` when Langfuse is off), system-skill sync at startup, `checkpoint_cleanup`, and migration `0002`. API adds `/skills` and `/messages/{id}/feedback`.

**Tech Stack:** as Plan 2 plus `langchain>=1.4,<2` (required by `langfuse.langchain.CallbackHandler`), OpenTelemetry `InMemorySpanExporter` for tracing tests.

**Spec:** `backend/docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md` (sections 8.3, 8.5 feedback row, 11). Builds on Plans 1 and 2.

## Global Constraints

- Same tooling standard as Plan 2: shared `ruff.toml`, strict `pyrefly.toml` (`0 diagnostics` at `--min-severity warn`, `unused-ignore = true`), `filterwarnings = ["error"]`, no new suppressions. Every task ends with `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q` green; tasks touching Postgres also run `uv run pytest -q -m integration`.
- Layering unchanged: domain framework-free; `api/` never imports `pharma_agent.domain`.
- Skill ids are globally unique (`skill_id` unique column). User skills get a slug from the name plus a 6-hex suffix, so they never collide with system ids and `get_by_ids` needs no user filter.
- Langfuse trace id of a turn is deterministic: `Langfuse.create_trace_id(seed=run_id)`. Feedback scores use the same seed, so nothing extra is stored.
- Commit messages end with the session attribution trailer.

---

## File Structure

```text
backend/
  src/pharma_agent/
    domain/skill/slug.py                     slugify (Vietnamese-safe)                  # Task 1
    domain/skill/ports.py                    (+ SkillRepository)                          # Task 1
    domain/feedback/{__init__,models,ports}.py                                            # Task 3
    application/skill/{__init__,service.py}  SkillService                                 # Task 2
    application/feedback/{__init__,service.py} FeedbackService                            # Task 3
    application/tracing.py                   TurnTracer, TraceHandle, ScoreSink, NullTracing  # Task 4
    application/chat/runner.py               (+ tracer)                                   # Task 4
    application/chat/service.py              (+ user_id/conversation_id on trace)         # Task 4
    infrastructure/persistence/postgres/tables.py            (+ SkillTable, FeedbackTable) # Task 1, 3
    infrastructure/persistence/postgres/migrations/versions/0002_skills_feedback.py       # Task 1, 3
    infrastructure/persistence/postgres/skill_repository.py                              # Task 1
    infrastructure/persistence/postgres/feedback_repository.py                           # Task 3
    infrastructure/observability/{__init__,langfuse_tracing.py}                          # Task 5
    infrastructure/langgraph/cleanup.py      delete_expired_checkpoints                   # Task 6
    infrastructure/container.py              (+ skills, feedback, tracing, cleanup)       # Task 5, 6
    api/routers/skills.py, api/routers/feedback.py, api/schemas.py, api/app.py           # Task 7
    cli.py                                   (+ cleanup-checkpoints)                      # Task 6
  tests/
    domain/test_slug.py, test_feedback.py; application/test_skill_service.py, test_feedback_service.py, test_tracing.py
    infrastructure/test_skill_repository.py, test_feedback_repository.py, test_langfuse_tracing.py, test_checkpoint_cleanup.py
    api/test_skills_api.py, test_feedback_api.py; memory_repository.py (+ InMemorySkillRepository, InMemoryFeedbackRepository)
```

---

### Task 1: Skill repository port, slug helper, Postgres skills table and repository

**Files:**
- Create: `backend/src/pharma_agent/domain/skill/slug.py`
- Modify: `backend/src/pharma_agent/domain/skill/ports.py`
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0002_skills_feedback.py` (skills part; Task 3 extends it before it is committed... no: this task commits `0002` with the `skills` table only, Task 3 adds `0003_feedback`)
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/skill_repository.py`
- Test: `backend/tests/domain/test_slug.py`, `backend/tests/infrastructure/test_skill_repository.py`

**Interfaces:**
- Produces: `slugify(text) -> str` (lowercase ASCII kebab, Vietnamese diacritics and đ folded, empty → `"skill"`); `SkillRepository(SkillCatalog)` protocol adding `upsert_system(skills)`, `create(skill)`, `list_for_user(user_id)`, `get_owned(user_id, skill_id)`, `set_enabled(user_id, skill_id, enabled) -> Skill | None`, `delete_owned(user_id, skill_id) -> bool`; `SkillTable`; `PostgresSkillRepository(sessions)`.

- [ ] **Step 1: Failing tests**

`backend/tests/domain/test_slug.py`:

```python
from pharma_agent.domain.skill.slug import slugify


def test_slugify_folds_vietnamese_and_punctuation() -> None:
    assert slugify("Tương tác thuốc & rượu (bia)") == "tuong-tac-thuoc-ruou-bia"
    assert slugify("  Liều  Đặc Biệt  ") == "lieu-dac-biet"
    assert slugify("!!!") == "skill"
    assert len(slugify("x" * 200)) <= 60
```

`backend/tests/infrastructure/test_skill_repository.py`:

```python
import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text

from pharma_agent.domain.skill.models import Skill
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.skill_repository import (
    PostgresSkillRepository,
)
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(text('TRUNCATE "user", skills CASCADE'))
    yield db
    await db.dispose()


async def make_user(database: Database, email: str) -> str:
    user_id = uuid.uuid4()
    async with database.sessions.begin() as session:
        session.add(UserTable(id=user_id, email=email, hashed_password="x"))
    return user_id.hex


def skill(skill_id: str, owner: str | None = None, version: str = "v1") -> Skill:
    return Skill(
        skill_id=skill_id,
        owner_user_id=owner,
        name=skill_id.title(),
        description=f"d {version}",
        search_guidance="s",
        answer_guidance="a",
        version=version,
    )


async def test_system_upsert_is_idempotent_and_updates_changed_versions(
    database: Database,
) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.upsert_system([skill("drug-monograph"), skill("plain-language")])
    await repo.upsert_system(
        [skill("drug-monograph", version="v2"), skill("plain-language")]
    )
    catalog = await repo.list_catalog(None, limit=10)
    assert sorted(m.skill_id for m in catalog) == ["drug-monograph", "plain-language"]
    loaded = await repo.get_by_ids(["drug-monograph"])
    assert (
        loaded[0].version == "v2"
        and loaded[0].description == "d v2"
        and loaded[0].is_system
    )


async def test_catalog_mixes_system_and_own_enabled_skills(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    owner = await make_user(database, "a@example.com")
    other = await make_user(database, "b@example.com")
    await repo.upsert_system([skill("drug-monograph")])
    await repo.create(skill("my-notes-ab12cd", owner))
    await repo.create(skill("theirs-ab12cd", other))
    disabled = await repo.set_enabled(owner, "my-notes-ab12cd", False)
    assert disabled is not None and disabled.enabled is False
    assert [m.skill_id for m in await repo.list_catalog(owner, limit=10)] == [
        "drug-monograph"
    ]
    await repo.set_enabled(owner, "my-notes-ab12cd", True)
    assert sorted(m.skill_id for m in await repo.list_catalog(owner, limit=10)) == [
        "drug-monograph",
        "my-notes-ab12cd",
    ]
    assert [s.skill_id for s in await repo.list_for_user(owner)] == ["my-notes-ab12cd"]
    assert await repo.get_owned(other, "my-notes-ab12cd") is None
    assert await repo.set_enabled(other, "my-notes-ab12cd", False) is None
    assert await repo.delete_owned(other, "my-notes-ab12cd") is False
    assert await repo.delete_owned(owner, "my-notes-ab12cd") is True
    assert (
        await repo.get_by_ids(["my-notes-ab12cd", "drug-monograph"])
        and len(await repo.get_by_ids(["my-notes-ab12cd"])) == 0
    )


async def test_limit_applies_to_catalog(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.upsert_system([skill(f"s-{i}") for i in range(5)])
    assert len(await repo.list_catalog(None, limit=3)) == 3
```

Run: `uv run pytest -q tests/domain/test_slug.py; uv run pytest -q -m integration tests/infrastructure/test_skill_repository.py`
Expected: both FAIL with import errors.

- [ ] **Step 2: Slug and port**

`backend/src/pharma_agent/domain/skill/slug.py`:

```python
import re
import unicodedata

MAX_SLUG_CHARS = 60
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase ASCII kebab-case; Vietnamese letters are folded (đ → d, ệ → e)."""
    folded = text.replace("đ", "d").replace("Đ", "D")
    ascii_text = (
        unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    )
    slug = (
        _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")[:MAX_SLUG_CHARS].strip("-")
    )
    return slug or "skill"
```

Append to `backend/src/pharma_agent/domain/skill/ports.py`:

```python
class SkillRepository(SkillCatalog, Protocol):
    async def upsert_system(self, skills: Sequence[Skill]) -> None:
        """Insert or update system skills (owner None) by skill_id."""
        ...

    async def create(self, skill: Skill) -> None: ...

    async def list_for_user(self, user_id: str) -> list[Skill]:
        """All skills owned by the user, enabled or not, ordered by name."""
        ...

    async def get_owned(self, user_id: str, skill_id: str) -> Skill | None: ...

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> Skill | None: ...

    async def delete_owned(self, user_id: str, skill_id: str) -> bool: ...
```

- [ ] **Step 3: Table, migration, repository**

Append to `tables.py`:

```python
class SkillTable(Base):
    __tablename__ = "skills"
    __table_args__ = (Index("ix_skills_owner", "owner_user_id"),)

    skill_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    search_guidance: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=""
    )
    answer_guidance: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=""
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

Generate `0002_skills.py` with autogenerate (same command as Plan 2 Task 5, `rev_id="0002"`, message `skills`), review, format.

`skill_repository.py`:

```python
import uuid
from collections.abc import Sequence

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.skill.models import Skill, SkillMetadata
from pharma_agent.infrastructure.persistence.postgres.tables import SkillTable


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _skill(row: SkillTable) -> Skill:
    return Skill(
        skill_id=row.skill_id,
        owner_user_id=row.owner_user_id.hex if row.owner_user_id else None,
        name=row.name,
        description=row.description,
        search_guidance=row.search_guidance,
        answer_guidance=row.answer_guidance,
        version=row.version,
        enabled=row.enabled,
    )


def _values(skill: Skill) -> dict[str, object]:
    return {
        "skill_id": skill.skill_id,
        "owner_user_id": uuid.UUID(hex=skill.owner_user_id)
        if skill.owner_user_id
        else None,
        "name": skill.name,
        "description": skill.description,
        "search_guidance": skill.search_guidance,
        "answer_guidance": skill.answer_guidance,
        "version": skill.version,
        "enabled": skill.enabled,
    }


class PostgresSkillRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def upsert_system(self, skills: Sequence[Skill]) -> None:
        if not skills:
            return
        statement = insert(SkillTable).values([_values(s) for s in skills])
        statement = statement.on_conflict_do_update(
            index_elements=[SkillTable.skill_id],
            set_={
                "name": statement.excluded.name,
                "description": statement.excluded.description,
                "search_guidance": statement.excluded.search_guidance,
                "answer_guidance": statement.excluded.answer_guidance,
                "version": statement.excluded.version,
                "updated_at": func.now(),
            },
            where=SkillTable.version != statement.excluded.version,
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def create(self, skill: Skill) -> None:
        async with self._sessions.begin() as session:
            session.add(SkillTable(**_values(skill)))

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        owner = _uuid(user_id) if user_id else None
        visible = SkillTable.owner_user_id.is_(None)
        if owner is not None:
            visible = or_(visible, SkillTable.owner_user_id == owner)
        query = (
            select(SkillTable)
            .where(visible, SkillTable.enabled.is_(True))
            .order_by(SkillTable.owner_user_id.is_not(None), SkillTable.skill_id)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_skill(row).metadata() for row in rows]

    async def get_by_ids(self, skill_ids: Sequence[str]) -> list[Skill]:
        if not skill_ids:
            return []
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(SkillTable).where(
                            SkillTable.skill_id.in_(list(skill_ids))
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [_skill(row) for row in rows]

    async def list_for_user(self, user_id: str) -> list[Skill]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(SkillTable)
                        .where(SkillTable.owner_user_id == owner)
                        .order_by(SkillTable.name)
                    )
                )
                .scalars()
                .all()
            )
        return [_skill(row) for row in rows]

    async def get_owned(self, user_id: str, skill_id: str) -> Skill | None:
        owner = _uuid(user_id)
        if owner is None:
            return None
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(SkillTable).where(
                        SkillTable.skill_id == skill_id,
                        SkillTable.owner_user_id == owner,
                    )
                )
            ).scalar_one_or_none()
        return _skill(row) if row is not None else None

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> Skill | None:
        owner = _uuid(user_id)
        if owner is None:
            return None
        async with self._sessions.begin() as session:
            row = (
                await session.execute(
                    update(SkillTable)
                    .where(
                        SkillTable.skill_id == skill_id,
                        SkillTable.owner_user_id == owner,
                    )
                    .values(enabled=enabled, updated_at=func.now())
                    .returning(SkillTable)
                )
            ).scalar_one_or_none()
            return _skill(row) if row is not None else None

    async def delete_owned(self, user_id: str, skill_id: str) -> bool:
        owner = _uuid(user_id)
        if owner is None:
            return False
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(SkillTable)
                .where(
                    SkillTable.skill_id == skill_id, SkillTable.owner_user_id == owner
                )
                .returning(SkillTable.skill_id)
            )
            return result.first() is not None
```

(Import `func` from sqlalchemy.) The catalog orders system skills first so the `limit + 1` overflow check in `resolve_skills_node` never hides system skills behind a user's many uploads.

- [ ] **Step 4: Checks and commit**

`git commit -m "feat(skills): add Postgres skill repository, slug helper and skills table"`

---

### Task 2: SkillService and in-memory skill repository

**Files:**
- Create: `backend/src/pharma_agent/application/skill/__init__.py`, `.../skill/service.py`
- Modify: `backend/tests/memory_repository.py` (+ `InMemorySkillRepository`)
- Test: `backend/tests/application/test_skill_service.py`

**Interfaces:**
- Produces: `SkillView(id, name, description, enabled, is_system, version)`; `SkillService(skills, clock)` with `sync_system(root: Path)`, `list_for_user(user_id)`, `upload(user_id, filename, content: bytes)`, `set_enabled(user_id, skill_id, enabled)`, `delete(user_id, skill_id)`; errors `InvalidInput` (bad file name, too large, parse error) and `SkillNotFound(ApplicationError)`; `MAX_SKILL_BYTES = 64 * 1024`.

- [ ] **Step 1: Failing tests**

Add to `backend/tests/memory_repository.py`:

```python
class InMemorySkillRepository:
    def __init__(self, *skills: Skill) -> None:
        self.rows: dict[str, Skill] = {
            s.skill_id: s.model_copy(deep=True) for s in skills
        }

    async def upsert_system(self, skills: Sequence[Skill]) -> None:
        for skill in skills:
            self.rows[skill.skill_id] = skill.model_copy(deep=True)

    async def create(self, skill: Skill) -> None:
        if skill.skill_id in self.rows:
            raise ValueError(f"duplicate skill_id {skill.skill_id}")
        self.rows[skill.skill_id] = skill.model_copy(deep=True)

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        visible = [
            s
            for s in self.rows.values()
            if s.enabled and (s.owner_user_id is None or s.owner_user_id == user_id)
        ]
        visible.sort(key=lambda s: (s.owner_user_id is not None, s.skill_id))
        return [s.metadata() for s in visible[:limit]]

    async def get_by_ids(self, skill_ids: Sequence[str]) -> list[Skill]:
        return [self.rows[i].model_copy(deep=True) for i in skill_ids if i in self.rows]

    async def list_for_user(self, user_id: str) -> list[Skill]:
        return sorted(
            (
                s.model_copy(deep=True)
                for s in self.rows.values()
                if s.owner_user_id == user_id
            ),
            key=lambda s: s.name,
        )

    async def get_owned(self, user_id: str, skill_id: str) -> Skill | None:
        skill = self.rows.get(skill_id)
        return (
            skill.model_copy(deep=True)
            if skill and skill.owner_user_id == user_id
            else None
        )

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> Skill | None:
        skill = self.rows.get(skill_id)
        if skill is None or skill.owner_user_id != user_id:
            return None
        skill.enabled = enabled
        return skill.model_copy(deep=True)

    async def delete_owned(self, user_id: str, skill_id: str) -> bool:
        skill = self.rows.get(skill_id)
        if skill is None or skill.owner_user_id != user_id:
            return False
        del self.rows[skill_id]
        return True
```

`backend/tests/application/test_skill_service.py`:

```python
from pathlib import Path

import pytest

from pharma_agent.application.errors import InvalidInput
from pharma_agent.application.skill.service import (
    MAX_SKILL_BYTES,
    SkillNotFound,
    SkillService,
)
from pharma_agent.domain.shared.clock import FixedClock
from tests.fakes import NOW
from tests.memory_repository import InMemorySkillRepository

OWNER, STRANGER = "a" * 32, "b" * 32
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
VALID = b"""---
name: Ghi ch\xc3\xba c\xe1\xbb\xa7a t\xc3\xb4i
description: D\xc3\xb9ng khi h\xe1\xbb\x8fi v\xe1\xbb\x81 thu\xe1\xbb\x91c b\xe1\xbb\x95 sung.
---

## T\xc3\xacm ki\xe1\xba\xbfm
- T\xc3\xacm m\xe1\xbb\xa5c li\xe1\xbb\x81u.
"""


def service(
    repo: InMemorySkillRepository | None = None,
) -> tuple[SkillService, InMemorySkillRepository]:
    repo = repo or InMemorySkillRepository()
    return SkillService(repo, FixedClock(NOW)), repo


async def test_sync_system_loads_repo_skills() -> None:
    svc, repo = service()
    await svc.sync_system(SKILLS_DIR)
    assert len(repo.rows) == 5 and all(s.is_system for s in repo.rows.values())


async def test_upload_creates_slugged_unique_skill_and_lists_it() -> None:
    svc, repo = service()
    first = await svc.upload(OWNER, "SKILL.md", VALID)
    second = await svc.upload(OWNER, "skill.md", VALID)
    assert (
        first.id.startswith("ghi-chu-cua-toi-")
        and len(first.id) == len("ghi-chu-cua-toi-") + 6
    )
    assert first.id != second.id and first.is_system is False and first.enabled is True
    listed = await svc.list_for_user(OWNER)
    assert [s.id for s in listed] == sorted([first.id, second.id]) or len(listed) == 2
    assert await svc.list_for_user(STRANGER) == []


async def test_upload_validation() -> None:
    svc, _ = service()
    with pytest.raises(InvalidInput, match="SKILL.md"):
        await svc.upload(OWNER, "notes.txt", VALID)
    with pytest.raises(InvalidInput, match="64"):
        await svc.upload(OWNER, "SKILL.md", b"x" * (MAX_SKILL_BYTES + 1))
    with pytest.raises(InvalidInput, match="frontmatter"):
        await svc.upload(OWNER, "SKILL.md", b"no frontmatter")
    with pytest.raises(InvalidInput, match="UTF-8"):
        await svc.upload(OWNER, "SKILL.md", b"\xff\xfe")


async def test_enable_disable_delete_are_owner_scoped() -> None:
    svc, _ = service()
    created = await svc.upload(OWNER, "SKILL.md", VALID)
    disabled = await svc.set_enabled(OWNER, created.id, False)
    assert disabled.enabled is False
    with pytest.raises(SkillNotFound):
        await svc.set_enabled(STRANGER, created.id, True)
    with pytest.raises(SkillNotFound):
        await svc.delete(STRANGER, created.id)
    await svc.delete(OWNER, created.id)
    with pytest.raises(SkillNotFound):
        await svc.delete(OWNER, created.id)
```

- [ ] **Step 2: Implement**

`backend/src/pharma_agent/application/skill/service.py`:

```python
import secrets
from pathlib import Path

from pydantic import BaseModel

from pharma_agent.application.errors import ApplicationError, InvalidInput
from pharma_agent.domain.shared.clock import Clock
from pharma_agent.domain.skill.models import Skill
from pharma_agent.domain.skill.parser import SkillParseError, parse_skill_markdown
from pharma_agent.domain.skill.ports import SkillRepository
from pharma_agent.domain.skill.slug import slugify
from pharma_agent.infrastructure.skills.filesystem_catalog import (
    load_system_skills,
)  # NO: application must not import infrastructure

MAX_SKILL_BYTES = 64 * 1024
```

Correction: `load_system_skills` lives in infrastructure. Move that function into the domain: create `backend/src/pharma_agent/domain/skill/files.py` with `load_system_skills(root: Path) -> list[Skill]` (verbatim from `infrastructure/skills/filesystem_catalog.py`, which now imports it from there). Then the service file is:

```python
import secrets
from pathlib import Path

from pydantic import BaseModel

from pharma_agent.application.errors import ApplicationError, InvalidInput
from pharma_agent.domain.skill.files import load_system_skills
from pharma_agent.domain.skill.models import Skill
from pharma_agent.domain.skill.parser import SkillParseError, parse_skill_markdown
from pharma_agent.domain.skill.ports import SkillRepository
from pharma_agent.domain.skill.slug import slugify

MAX_SKILL_BYTES = 64 * 1024


class SkillNotFound(ApplicationError):
    code = "SKILL_NOT_FOUND"


class SkillView(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    is_system: bool
    version: str

    @classmethod
    def of(cls, skill: Skill) -> "SkillView":
        return cls(
            id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            enabled=skill.enabled,
            is_system=skill.is_system,
            version=skill.version,
        )


class SkillService:
    def __init__(self, skills: SkillRepository) -> None:
        self._skills = skills

    async def sync_system(self, root: Path) -> int:
        skills = load_system_skills(root)
        await self._skills.upsert_system(skills)
        return len(skills)

    async def list_for_user(self, user_id: str) -> list[SkillView]:
        return [SkillView.of(s) for s in await self._skills.list_for_user(user_id)]

    async def upload(self, user_id: str, filename: str, content: bytes) -> SkillView:
        if Path(filename).name.lower() != "skill.md":
            raise InvalidInput("the uploaded file must be named SKILL.md")
        if len(content) > MAX_SKILL_BYTES:
            raise InvalidInput(f"SKILL.md must be at most {MAX_SKILL_BYTES // 1024} KB")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInput("SKILL.md must be UTF-8 encoded") from exc
        try:
            parsed = parse_skill_markdown(text)
        except SkillParseError as exc:
            raise InvalidInput(str(exc)) from exc
        skill = Skill(
            skill_id=f"{slugify(parsed.name)[:50]}-{secrets.token_hex(3)}",
            owner_user_id=user_id,
            name=parsed.name,
            description=parsed.description,
            search_guidance=parsed.search_guidance,
            answer_guidance=parsed.answer_guidance,
            version=parsed.version,
        )
        await self._skills.create(skill)
        return SkillView.of(skill)

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> SkillView:
        skill = await self._skills.set_enabled(user_id, skill_id, enabled)
        if skill is None:
            raise SkillNotFound(skill_id)
        return SkillView.of(skill)

    async def delete(self, user_id: str, skill_id: str) -> None:
        if not await self._skills.delete_owned(user_id, skill_id):
            raise SkillNotFound(skill_id)
```

(The service takes no clock; drop `FixedClock` from the test helper accordingly: `SkillService(repo)`.) `SkillParseError` message for missing frontmatter contains "frontmatter" already.

- [ ] **Step 3: Checks and commit**

`git commit -m "feat(skills): add SkillService for system sync and user uploads"`

---

### Task 3: Feedback domain, Postgres feedback repository, FeedbackService

**Files:**
- Create: `backend/src/pharma_agent/domain/feedback/{__init__,models,ports}.py`
- Modify: `tables.py` (+ `FeedbackTable`), migration `0003_feedback.py`
- Create: `.../postgres/feedback_repository.py`
- Modify: `domain/conversation/ports.py` (+ `get_message(user_id, message_id) -> Message | None` on `ConversationRepository`), Postgres and in-memory repositories implement it
- Create: `backend/src/pharma_agent/application/feedback/{__init__,service.py}`, `application/tracing.py` (`ScoreSink` protocol only for now; Task 4 adds the tracer)
- Test: `tests/domain/test_feedback.py`, `tests/infrastructure/test_feedback_repository.py`, `tests/application/test_feedback_service.py`

**Interfaces:**
- Domain: `Rating(StrEnum) UP/DOWN`, `Feedback(feedback_id, user_id, message_id, rating, note, created_at)` with `Feedback.create(user_id, message_id, rating, note, now)` (note stripped, max 2000 chars → `InvalidNote`), `FeedbackRepository.save(feedback)` (upsert per user+message), `.get(user_id, message_id)`.
- Application: `ScoreSink.record_feedback(run_id, rating, note) -> None` (Protocol; `NullScoreSink`), `FeedbackView`, `FeedbackService(conversations, feedback, sink, clock).submit(user_id, message_id, rating, note) -> FeedbackView` (404 `MessageNotFound(ApplicationError)` when the message is not the user's assistant message; sink errors are logged, never raised).
- Table: `feedback(feedback_id uuid pk, user_id uuid fk, message_id uuid fk cascade, rating varchar(8), note text, created_at)`, unique `(user_id, message_id)`.

Tests mirror Task 1/2 patterns: domain validation (`InvalidNote`, rating enum), repository upsert overwrites the note, service rejects foreign/unknown/user-role messages, calls the sink with the assistant message's `run_id`, and swallows sink exceptions.

- [ ] Steps: failing tests → implement → `git commit -m "feat(feedback): add feedback domain, repository and service with score sink"`

---

### Task 4: TurnTracer port in the runner and chat service

**Files:**
- Modify: `application/tracing.py` (+ `TraceHandle`, `TurnTracer`, `NullTracing` implementing both `TurnTracer` and `ScoreSink`)
- Modify: `application/chat/runner.py` (`ChatTurnRunner(graph, deps, limits, tracer=NullTracing())`; `_produce` wraps `astream` in `with self._tracer.trace_turn(run) as handle:` and passes `config={"configurable": {...}, "callbacks": handle.callbacks}`; on completion `handle.finish(status=..., output=answer_text)`)
- Test: `tests/application/test_tracing.py` with a `RecordingTracer` asserting the runner opens one trace per turn with `run_id`, `user_id`, `conversation_id`, that callbacks reach LangGraph (use a `BaseCallbackHandler` subclass counting `on_chain_start`), and that `finish` receives the final status.

`application/tracing.py`:

```python
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler

from pharma_agent.domain.agent.run import AgentRun


class TraceHandle(Protocol):
    @property
    def callbacks(self) -> Sequence[BaseCallbackHandler]: ...

    def finish(self, *, status: str, output: str) -> None: ...


class TurnTracer(Protocol):
    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]: ...


class ScoreSink(Protocol):
    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None: ...


class _NullHandle:
    callbacks: Sequence[BaseCallbackHandler] = ()

    def finish(self, *, status: str, output: str) -> None:
        return None


class NullTracing:
    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]:
        return nullcontext(_NullHandle())

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        return None
```

Add `langchain>=1.4,<2` to dependencies (needed by the Langfuse handler in Task 5; `langchain_core` is already transitively present).

- [ ] Steps: failing tests → implement → `git commit -m "feat(application): add TurnTracer port and wire it through the chat runner"`

---

### Task 5: Langfuse tracing adapter, system-skill sync and container wiring

**Files:**
- Create: `infrastructure/observability/__init__.py`, `infrastructure/observability/langfuse_tracing.py`
- Modify: `infrastructure/container.py` (`Container` gains `skills: SkillService`, `feedback: FeedbackService | None`, `tracing: TurnTracer & ScoreSink`; `open_container` syncs system skills from `settings.skills_dir` into Postgres, uses `PostgresSkillRepository` as the agent's `SkillCatalog` instead of the filesystem catalog, and shuts Langfuse down on exit)
- Modify: `infrastructure/composition.py` (`build_application(settings, *, checkpointer=None, skills: SkillCatalog | None = None, tracer: TurnTracer | None = None)`)
- Test: `tests/infrastructure/test_langfuse_tracing.py`, extend `tests/infrastructure/test_container.py`

`langfuse_tracing.py`:

```python
"""Langfuse tracing: one trace per chat turn (deterministic id from run_id), LangGraph
callbacks nested inside it, feedback recorded as a score on the same trace."""

import logging
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from pharma_agent.application.tracing import TraceHandle
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.infrastructure.settings import LangfuseSettings

logger = logging.getLogger(__name__)
FEEDBACK_SCORE_NAME = "user_feedback"


class _Handle:
    def __init__(self, span, handler: CallbackHandler) -> None:  # span: LangfuseAgent
        self._span = span
        self._callbacks: tuple[BaseCallbackHandler, ...] = (handler,)

    @property
    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return self._callbacks

    def finish(self, *, status: str, output: str) -> None:
        self._span.update(output=output, metadata={"status": status})


class LangfuseTracing:
    def __init__(self, client: Langfuse) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: LangfuseSettings) -> "LangfuseTracing":
        return cls(
            Langfuse(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                base_url=settings.host,
            )
        )

    def trace_id_for(self, run_id: str) -> str:
        return self._client.create_trace_id(seed=run_id)

    @contextmanager
    def _trace(self, run: AgentRun) -> Iterator[TraceHandle]:
        with ExitStack() as stack:
            stack.enter_context(
                propagate_attributes(
                    user_id=run.user_id,
                    session_id=run.conversation_id,
                    trace_name="chat_turn",
                    metadata={"run_id": run.run_id},
                )
            )
            span = stack.enter_context(
                self._client.start_as_current_observation(
                    trace_context={"trace_id": self.trace_id_for(run.run_id)},
                    name="chat_turn",
                    as_type="agent",
                    input={"query": run.original_query},
                )
            )
            yield _Handle(span, CallbackHandler())

    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]:
        return self._trace(run)

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        self._client.create_score(
            name=FEEDBACK_SCORE_NAME,
            value=1.0 if rating == "up" else 0.0,
            trace_id=self.trace_id_for(run_id),
            comment=note or None,
            data_type="NUMERIC",
        )

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            logger.exception("langfuse shutdown failed")
```

Test with `Langfuse(public_key="pk", secret_key="sk", base_url="http://langfuse.test", span_exporter=InMemorySpanExporter(), httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))`: run a scripted turn through `ChatTurnRunner(..., tracer=LangfuseTracing(client))`, `client.flush()`, then assert the exporter holds a `chat_turn` span whose trace id equals `create_trace_id(seed=run_id)` and LangGraph node spans (names like `guard`) share it; `record_feedback` + `flush()` must produce an ingestion POST whose body contains `user_feedback`. If the OpenAI wrapper is exercised, generations join the same trace automatically.

- [ ] Steps: failing tests → implement → `git commit -m "feat(observability): trace chat turns in Langfuse and store skills in Postgres"`

---

### Task 6: Checkpoint cleanup job and CLI command

**Files:**
- Create: `infrastructure/langgraph/cleanup.py`
- Modify: `infrastructure/container.py` (run cleanup once at startup in a background task when the checkpointer is configured), `cli.py` (`cleanup-checkpoints --days 7`), `settings.py` (`CheckpointSettings(retention_days=7)` under `Settings.checkpoints`)
- Test: `tests/infrastructure/test_checkpoint_cleanup.py` (integration): run one real turn through `open_postgres_checkpointer`, then rewrite its rows' `checkpoint->>'ts'` to 10 days ago with SQL, run `delete_expired_checkpoints(conninfo, retention_days=7)`, assert the thread's rows in `checkpoints`, `checkpoint_blobs` and `checkpoint_writes` are gone while a fresh thread's rows remain.

```python
async def delete_expired_checkpoints(conninfo: str, *, retention_days: int) -> int:
    """Delete checkpoints (and their blobs/writes) whose `ts` is older than retention_days."""
    async with await AsyncConnection.connect(conninfo, autocommit=True) as connection:
        cursor = await connection.execute(
            """
            WITH expired AS (
                DELETE FROM checkpoints
                WHERE (checkpoint->>'ts')::timestamptz < now() - make_interval(days => %s)
                RETURNING thread_id, checkpoint_ns, checkpoint_id
            ),
            writes AS (
                DELETE FROM checkpoint_writes w USING expired e
                WHERE w.thread_id = e.thread_id AND w.checkpoint_ns = e.checkpoint_ns
                  AND w.checkpoint_id = e.checkpoint_id
            ),
            blobs AS (
                DELETE FROM checkpoint_blobs b
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints c
                    WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns
                )
            )
            SELECT count(*) FROM expired
            """,
            (retention_days,),
        )
        row = await cursor.fetchone()
    return int(row[0]) if row else 0
```

- [ ] Steps: failing test → implement → `git commit -m "feat(infra): add checkpoint retention cleanup"`

---

### Task 7: API routes for skills and feedback, README

**Files:**
- Create: `api/routers/skills.py`, `api/routers/feedback.py`; modify `api/schemas.py` (`FeedbackRequest(rating: Literal["up","down"], note: str = ""`), `EnableSkillRequest(enabled: bool)`), `api/app.py`, `api/errors.py` (`SkillNotFound`, `MessageNotFound` → 404), `tests/api/harness.py` (container gets `skills`, `feedback`, `tracing`)
- Test: `tests/api/test_skills_api.py`, `tests/api/test_feedback_api.py`; extend `tests/api/test_e2e_postgres.py` with upload → chat → feedback over real Postgres.

Routes (all authenticated): `GET /api/v1/skills` → `list[SkillView]` (system + own; system entries come from `list_catalog(None)`), `POST /api/v1/skills` multipart field `file` → 201 `SkillView`, `PATCH /api/v1/skills/{id}` `{enabled}` → `SkillView`, `DELETE /api/v1/skills/{id}` → 204, `POST /api/v1/messages/{id}/feedback` → 201 `FeedbackView`. Upload uses `fastapi.UploadFile`; 413 when `content-length` exceeds `MAX_SKILL_BYTES` (`InvalidInput` otherwise → 422).

- [ ] Steps: failing tests → implement → README table rows → `git commit -m "feat(api): add skill upload and message feedback endpoints"`

---

## Done when

`uv run pytest -q`, `uv run pytest -q -m integration`, and `uv run --project backend pre-commit run --all-files` are green; the spec's sections 8.3 (skills), feedback, and 11 (Langfuse) are implemented; README documents Langfuse env vars and the new endpoints.

## As built (differences from the plan above)

- Migrations are split: `0002_skills.py` creates `skills`, `0003_feedback.py` creates `feedback` with the unique constraint `uq_feedback_user_message`.
- `load_system_skills` moved from `infrastructure/skills/filesystem_catalog.py` to `domain/skill/files.py`, so `SkillService` does not import infrastructure. `SkillService` takes only the repository (no clock).
- `SkillRepository` also has `list_system()`, and `SkillService.list_visible(user_id)` backs `GET /skills` (system skills first, then the user's own, enabled or not).
- Oversized uploads raise `PayloadTooLarge(InvalidInput)` and map to HTTP 413; other upload problems stay 422.
- `ConversationRepository.get_message(user_id, message_id)` was added for `FeedbackService`.
- `application/tracing.py` defines `Tracing(TurnTracer, ScoreSink)` so the container holds one object for both roles.
- Langfuse keeps one resource manager per public key for the whole process and `shutdown()` closes it; the tracing tests give every client a unique public key.
- Checkpoint cleanup deletes orphaned blobs in a second statement of the same transaction, because CTEs share one snapshot. The service runs it once in the background at startup.
