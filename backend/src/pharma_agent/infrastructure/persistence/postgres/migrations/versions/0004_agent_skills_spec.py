"""Skills follow the Agent Skills specification: `name` is the identifier, unique per owner.

Rows stored with the previous layout used free-text names that are invalid under the
specification, so the table is recreated; system skills are re-synced from the repository
when the service starts.

Revision ID: 0004
Create Date: 2026-09-12 09:30:00
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column[datetime]]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    ]


def _owner_fk() -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["owner_user_id"],
        ["user.id"],
        name=op.f("fk_skills_owner_user_id_user"),
        ondelete="CASCADE",
    )


def upgrade() -> None:
    op.drop_index("ix_skills_owner", table_name="skills")
    op.drop_table("skills")
    op.create_table(
        "skills",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        *_timestamps(),
        _owner_fk(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_skills")),
        sa.UniqueConstraint(
            "owner_user_id",
            "name",
            name="uq_skills_owner_name",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "name ~ '^[a-z0-9]+(-[a-z0-9]+)*$'", name=op.f("ck_skills_name_format")
        ),
    )
    op.create_index("ix_skills_owner", "skills", ["owner_user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_skills_owner", table_name="skills")
    op.drop_table("skills")
    op.create_table(
        "skills",
        sa.Column("skill_id", sa.String(length=80), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("search_guidance", sa.Text(), server_default="", nullable=False),
        sa.Column("answer_guidance", sa.Text(), server_default="", nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        *_timestamps(),
        _owner_fk(),
        sa.PrimaryKeyConstraint("skill_id", name=op.f("pk_skills")),
    )
    op.create_index("ix_skills_owner", "skills", ["owner_user_id"], unique=False)
