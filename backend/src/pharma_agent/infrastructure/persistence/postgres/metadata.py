"""Alembic's view of the database: every table module on one MetaData, and what it manages.

Importing `tables` and `corpus_tables` here registers all tables on `Base.metadata`, so
`migrations/env.py` and the migration test see the complete schema.
"""

from pharma_agent.infrastructure.persistence.postgres import corpus_tables, tables

target_metadata = tables.Base.metadata

# Schemas Alembic owns besides the default one (reported to include_name as None).
MANAGED_SCHEMAS = frozenset({corpus_tables.CORPUS_SCHEMA})


def include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Alembic `include_name` hook.

    Only the default schema and `corpus` are compared, and tables owned by the LangGraph
    checkpointer are skipped.
    """
    if type_ == "schema":
        return name is None or name in MANAGED_SCHEMAS
    return not (type_ == "table" and name in tables.EXTERNALLY_MANAGED_TABLES)
