"""message_citations: citations reference immutable corpus chunk versions (spec A §5.1).

The development database is reset, so `messages.citations` is dropped without a backfill.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-13 14:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "message_citations",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("chunk_version_id", sa.Uuid(), nullable=False),
        sa.Column("release_id", sa.Uuid(), nullable=False),
        sa.Column("strategy", sa.String(length=16), nullable=False),
        sa.Column(
            "block_chunk_version_ids", postgresql.ARRAY(sa.Uuid()), nullable=False
        ),
        sa.CheckConstraint(
            "strategy IN ('search_only', 'chunk_window', 'full_section')",
            name=op.f("ck_message_citations_strategy"),
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_message_citations_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_version_id"],
            ["corpus.chunk_versions.id"],
            name=op.f("fk_message_citations_chunk_version_id_chunk_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["release_id"],
            ["corpus.releases.id"],
            name=op.f("fk_message_citations_release_id_releases"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "message_id", "index", name=op.f("pk_message_citations")
        ),
    )
    op.create_index(
        "ix_message_citations_chunk_version_id",
        "message_citations",
        ["chunk_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_message_citations_release_id",
        "message_citations",
        ["release_id"],
        unique=False,
    )
    op.drop_column("messages", "citations")


def downgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.drop_index("ix_message_citations_release_id", table_name="message_citations")
    op.drop_index(
        "ix_message_citations_chunk_version_id", table_name="message_citations"
    )
    op.drop_table("message_citations")
