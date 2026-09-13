"""Retrieval audit references corpus releases and chunk versions (spec C §9).

Existing audit rows and stored citations point at position-based chunk ids that no longer
exist, so they are discarded (development data only). There is deliberately no foreign key
into schema `corpus`: audit must never block `pharma-agent corpus gc`.

Revision ID: 0006
Create Date: 2026-09-13 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _discard_audit_and_citations() -> None:
    op.execute("DELETE FROM retrieval_runs")  # cascades to retrieval_hits
    op.execute("UPDATE messages SET citations = '[]'::jsonb")


def upgrade() -> None:
    _discard_audit_and_citations()
    op.drop_column("retrieval_runs", "corpus_version")
    op.add_column(
        "retrieval_runs",
        sa.Column(
            "release_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.drop_column("retrieval_hits", "chunk_id")
    op.drop_column("retrieval_hits", "section_id")
    op.drop_column("retrieval_hits", "table_id")
    op.add_column(
        "retrieval_hits", sa.Column("chunk_version_id", sa.Uuid(), nullable=False)
    )
    op.add_column("retrieval_hits", sa.Column("section_key", sa.Text(), nullable=False))
    op.add_column("retrieval_hits", sa.Column("table_key", sa.Text(), nullable=True))


def downgrade() -> None:
    _discard_audit_and_citations()
    op.drop_column("retrieval_hits", "table_key")
    op.drop_column("retrieval_hits", "section_key")
    op.drop_column("retrieval_hits", "chunk_version_id")
    op.add_column(
        "retrieval_hits",
        sa.Column("table_id", sa.Text(), server_default="", nullable=False),
    )
    op.add_column("retrieval_hits", sa.Column("section_id", sa.Text(), nullable=False))
    op.add_column("retrieval_hits", sa.Column("chunk_id", sa.Text(), nullable=False))
    op.drop_column("retrieval_runs", "release_ids")
    op.add_column(
        "retrieval_runs",
        sa.Column("corpus_version", sa.String(length=200), nullable=False),
    )
