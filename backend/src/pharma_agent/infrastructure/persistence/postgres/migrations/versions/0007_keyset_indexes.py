"""Keyset pagination indexes: the sort keys end with the id tie-breaker.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13 12:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_conversations_user_updated", table_name="conversations")
    op.create_index(
        "ix_conversations_user_updated_id",
        "conversations",
        ["user_id", "updated_at", "id"],
        unique=False,
    )
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.create_index(
        "ix_messages_conversation_created_id",
        "messages",
        ["conversation_id", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.drop_index("ix_conversations_user_updated_id", table_name="conversations")
    op.create_index(
        "ix_conversations_user_updated",
        "conversations",
        ["user_id", "updated_at"],
        unique=False,
    )
