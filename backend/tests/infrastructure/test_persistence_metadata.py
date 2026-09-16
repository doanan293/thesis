from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CORPUS_SCHEMA,
)
from pharma_agent.infrastructure.persistence.postgres.metadata import (
    include_name,
    target_metadata,
)

CORPUS_TABLES = {
    "collections",
    "documents",
    "sections",
    "section_revisions",
    "chunk_versions",
    "releases",
    "release_chunks",
    "glossary_entries",
    "colloquial_mappings",
    "embedding_cache",
}


def test_target_metadata_holds_app_and_corpus_tables() -> None:
    corpus = {
        table.name
        for table in target_metadata.tables.values()
        if table.schema == CORPUS_SCHEMA
    }
    public = {
        table.name for table in target_metadata.tables.values() if table.schema is None
    }
    assert corpus == CORPUS_TABLES
    assert public == {
        "access_tokens",
        "conversations",
        "feedback",
        "message_citations",
        "messages",
        "oauth_account",
        "retrieval_hits",
        "retrieval_runs",
        "user",
    }
    assert "skills" not in target_metadata.tables


def test_include_name_manages_default_and_corpus_schemas_only() -> None:
    assert include_name(None, "schema", {}) is True
    assert include_name("corpus", "schema", {}) is True
    assert include_name("langfuse", "schema", {}) is False
    assert include_name("checkpoints", "table", {}) is False
    assert include_name("conversations", "table", {}) is True


def test_current_release_fk_is_deferrable_and_breaks_the_cycle() -> None:
    collections = target_metadata.tables["corpus.collections"]
    (fk,) = [
        fk
        for fk in collections.foreign_key_constraints
        if fk.column_keys == ["current_release_id"]
    ]
    assert fk.deferrable is True and fk.initially == "DEFERRED" and fk.use_alter
    assert fk.name == "fk_collections_current_release_id_releases"


def test_chunk_versions_are_protected_by_restrict() -> None:
    release_chunks = target_metadata.tables["corpus.release_chunks"]
    chunk_versions = target_metadata.tables["corpus.chunk_versions"]
    ondelete = {
        tuple(fk.column_keys): fk.ondelete
        for fk in [
            *release_chunks.foreign_key_constraints,
            *chunk_versions.foreign_key_constraints,
        ]
    }
    assert ondelete[("chunk_version_id",)] == "RESTRICT"
    assert ondelete[("section_revision_id",)] == "RESTRICT"
    assert ondelete[("release_id",)] == "CASCADE"
