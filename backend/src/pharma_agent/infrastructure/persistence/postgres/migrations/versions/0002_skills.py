"""skills

Revision ID: 0002
Create Date: 2026-09-12 07:51:16.724764
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["user.id"],
            name=op.f("fk_skills_owner_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("skill_id", name=op.f("pk_skills")),
    )
    op.create_index("ix_skills_owner", "skills", ["owner_user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_skills_owner", table_name="skills")
    op.drop_table("skills")
