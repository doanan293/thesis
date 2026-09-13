"""Corpus schema: collections, documents, sections, immutable section revisions and chunk
versions, releases and the embedding cache (spec 2026-09-13 corpus platform §6.2).

Revision ID: 0005
Create Date: 2026-09-13 12:00:00
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "corpus"


def _created_at(name: str = "created_at") -> sa.Column[datetime]:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


def _optional_at(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=True)


def upgrade() -> None:
    op.execute(sa.schema.CreateSchema(SCHEMA))
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "visibility", sa.String(length=16), server_default="private", nullable=False
        ),
        sa.Column("current_release_id", sa.Uuid(), nullable=True),
        _created_at(),
        _created_at("updated_at"),
        sa.CheckConstraint(
            "visibility IN ('private', 'public')",
            name=op.f("ck_collections_visibility"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["user.id"],
            name=op.f("fk_collections_owner_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_collections")),
        sa.UniqueConstraint("key", name="uq_collections_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=300), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "attributes",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["corpus.collections.id"],
            name=op.f("fk_documents_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("collection_id", "key", name="uq_documents_collection_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "sections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=400), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column(
            "context_path",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("retrieval_mode", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["corpus.documents.id"],
            name=op.f("fk_sections_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sections")),
        sa.UniqueConstraint("document_id", "key", name="uq_sections_document_key"),
        schema=SCHEMA,
    )
    op.create_table(
        "section_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=False),
        sa.Column("blocks", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("start_page", sa.Integer(), nullable=True),
        sa.Column("end_page", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["corpus.sections.id"],
            name=op.f("fk_section_revisions_section_id_sections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_section_revisions")),
        schema=SCHEMA,
    )
    op.create_table(
        "chunk_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("section_revision_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("context_header", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("embedding_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("start_page", sa.Integer(), nullable=True),
        sa.Column("end_page", sa.Integer(), nullable=True),
        sa.Column("table_key", sa.String(length=200), nullable=True),
        sa.Column(
            "term_annotations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("colloquial", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["section_revision_id"],
            ["corpus.section_revisions.id"],
            name=op.f("fk_chunk_versions_section_revision_id_section_revisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunk_versions")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_chunk_versions_embedding_text_sha256",
        "chunk_versions",
        ["embedding_text_sha256"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_table(
        "releases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("bundle_digest", sa.String(length=64), nullable=False),
        sa.Column("chunker_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        _created_at(),
        _optional_at("ready_at"),
        _optional_at("published_at"),
        _optional_at("retired_at"),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'retired')",
            name=op.f("ck_releases_status"),
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["corpus.collections.id"],
            name=op.f("fk_releases_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_releases")),
        sa.UniqueConstraint(
            "collection_id", "number", name="uq_releases_collection_number"
        ),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        op.f("fk_collections_current_release_id_releases"),
        "collections",
        "releases",
        ["current_release_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_table(
        "release_chunks",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_version_id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=False),
        sa.Column("section_revision_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("hydrate_strategy", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_version_id"],
            ["corpus.chunk_versions.id"],
            name=op.f("fk_release_chunks_chunk_version_id_chunk_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_release_chunks_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["corpus.sections.id"],
            name=op.f("fk_release_chunks_section_id_sections"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_revision_id"],
            ["corpus.section_revisions.id"],
            name=op.f("fk_release_chunks_section_revision_id_section_revisions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "release_id", "chunk_version_id", name=op.f("pk_release_chunks")
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_release_chunks_release_revision_ordinal",
        "release_chunks",
        ["release_id", "section_revision_id", "ordinal"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_release_chunks_chunk_version_id",
        "release_chunks",
        ["chunk_version_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_table(
        "glossary_entries",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_glossary_entries_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("release_id", "term", name=op.f("pk_glossary_entries")),
        schema=SCHEMA,
    )
    op.create_table(
        "colloquial_mappings",
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=300), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_colloquial_mappings_release_id_releases"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "release_id", "position", name=op.f("pk_colloquial_mappings")
        ),
        schema=SCHEMA,
    )
    op.create_table(
        "embedding_cache",
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("embedding_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("dims", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint(
            "model", "embedding_text_sha256", name=op.f("pk_embedding_cache")
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_collections_current_release_id_releases"),
        "collections",
        schema=SCHEMA,
        type_="foreignkey",
    )
    op.drop_table("embedding_cache", schema=SCHEMA)
    op.drop_table("colloquial_mappings", schema=SCHEMA)
    op.drop_table("glossary_entries", schema=SCHEMA)
    op.drop_table("release_chunks", schema=SCHEMA)
    op.drop_table("releases", schema=SCHEMA)
    op.drop_table("chunk_versions", schema=SCHEMA)
    op.drop_table("section_revisions", schema=SCHEMA)
    op.drop_table("sections", schema=SCHEMA)
    op.drop_table("documents", schema=SCHEMA)
    op.drop_table("collections", schema=SCHEMA)
    op.execute(sa.schema.DropSchema(SCHEMA))
