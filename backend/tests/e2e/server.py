"""Backend for Playwright E2E runs: the real app over real Postgres and Qdrant, scripted LLM.

Run from backend/:  uv run python -m tests.e2e.server --port 8001

On start it drops and recreates the E2E database (its name must contain "e2e"), runs the
migrations, deletes the fake-embedding Qdrant collection, imports and publishes the small
fixture bundle under the alias `e2e_chunks_current`, then serves the production container
with `ScenarioLlm` and `FakeEmbedder` injected (spec A §9). Environment: E2E_POSTGRES_DSN,
E2E_QDRANT_URL.
"""

import argparse
import asyncio
import os
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
import uvicorn
from alembic import command
from fastapi import FastAPI
from psycopg import sql
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from qdrant_client import AsyncQdrantClient
from sqlalchemy.engine import make_url

from pharma_agent.api.app import create_app
from pharma_agent.infrastructure.container import (
    Container,
    ContainerFactory,
    open_container,
)
from pharma_agent.infrastructure.corpus_factory import open_corpus_services
from pharma_agent.infrastructure.persistence.postgres.alembic_config import (
    alembic_config,
)
from pharma_agent.infrastructure.retrieval.qdrant_index import physical_collection_name
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import COLLECTION_KEY, small_bundle
from tests.e2e.scenario_llm import ScenarioLlm
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, FakeEmbedder

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POSTGRES_DSN = "postgresql+psycopg://thesis:thesis@localhost:5433/pharma_e2e"
DEFAULT_QDRANT_URL = "http://localhost:6333"
E2E_QDRANT_ALIAS = "e2e_chunks_current"
E2E_JWT_SECRET = "e2e-jwt-secret-for-tests-only-0123456789"
E2E_CSRF_SECRET = "e2e-csrf-secret-for-tests-only-0123456789"


@dataclass(frozen=True)
class E2EConfig:
    postgres_dsn: str = DEFAULT_POSTGRES_DSN
    qdrant_url: str = DEFAULT_QDRANT_URL
    deadline_seconds: float = 5.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "E2EConfig":
        return cls(
            postgres_dsn=environ.get("E2E_POSTGRES_DSN", DEFAULT_POSTGRES_DSN),
            qdrant_url=environ.get("E2E_QDRANT_URL", DEFAULT_QDRANT_URL),
        )


class IsolatedSettings(Settings):
    """Settings from constructor arguments only: no PHARMA_* variables and no .env file."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def e2e_settings(config: E2EConfig) -> Settings:
    return IsolatedSettings(
        postgres={"dsn": config.postgres_dsn, "pool_size": 5},
        qdrant={"url": config.qdrant_url},
        retrieval={
            # Drives both the alias the import writes and the alias search reads.
            "qdrant_collection": E2E_QDRANT_ALIAS,
            "collections": [COLLECTION_KEY],
            "embedding": {
                "model": FAKE_EMBEDDING_MODEL,
                "dimension": FAKE_EMBEDDING_DIMENSION,
            },
            "rerank": {"protocol": "none"},
        },
        budget={"deadline_seconds": config.deadline_seconds},
        # Playwright talks plain HTTP and WebKit rejects Secure cookies on http://localhost.
        auth={
            "jwt_secret": E2E_JWT_SECRET,
            "csrf_secret": E2E_CSRF_SECRET,
            "cookie_secure": False,
        },
        skills_dir=BACKEND_DIR / "skills",
    )


def reset_database(dsn: str) -> None:
    url = make_url(dsn)
    name = url.database
    if not name or "e2e" not in name:
        raise ValueError(
            f"refusing to reset database {name!r}: the E2E database name must contain 'e2e'"
        )
    admin = url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(name)
            )
        )
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


async def reset_qdrant(settings: Settings) -> None:
    """Delete the fake-embedding collection; its aliases go with it."""
    collection = physical_collection_name(settings.retrieval.embedding.model)
    client = AsyncQdrantClient(url=settings.qdrant.url)
    try:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
    finally:
        await client.close()


async def prepare_backing_services(settings: Settings) -> None:
    await asyncio.to_thread(reset_database, settings.postgres.dsn)
    await asyncio.to_thread(
        command.upgrade, alembic_config(settings.postgres.dsn), "head"
    )
    await reset_qdrant(settings)
    async with open_corpus_services(settings, embedder=FakeEmbedder()) as services:
        await services.importer(small_bundle(), publish=True)


def e2e_container_factory() -> ContainerFactory:
    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncGenerator[Container]:
        await prepare_backing_services(settings)
        async with open_container(
            settings, llm=ScenarioLlm(), embedder=FakeEmbedder()
        ) as container:
            yield container

    return factory


def build_e2e_app(config: E2EConfig) -> FastAPI:
    return create_app(e2e_settings(config), container_factory=e2e_container_factory())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pharma agent backend for Playwright E2E runs"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    uvicorn.run(
        build_e2e_app(E2EConfig.from_env(os.environ)), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
