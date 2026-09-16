"""feedback

Revision ID: 0003
Create Date: 2026-09-12 07:56:20.463408
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("feedback_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("note", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_feedback_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name=op.f("fk_feedback_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("feedback_id", name=op.f("pk_feedback")),
        sa.UniqueConstraint("user_id", "message_id", name="uq_feedback_user_message"),
    )


def downgrade() -> None:
    op.drop_table("feedback")
