from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    AliasDescription,
    CollectionDescription,
    CollectionsAliasesResponse,
    CollectionsResponse,
)

from corpus_pipeline.vector_store.qdrant_client_helper import QdrantClientHelper


def helper_with_client(
    *,
    collections: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    points_count: int = 0,
) -> tuple[QdrantClientHelper, MagicMock]:
    client = create_autospec(QdrantClient, instance=True)
    client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name=name) for name in collections]
    )
    client.get_aliases.return_value = CollectionsAliasesResponse(
        aliases=[
            AliasDescription(alias_name=name, collection_name="previous")
            for name in aliases
        ]
    )
    client.get_collection.return_value = SimpleNamespace(points_count=points_count)
    helper = QdrantClientHelper(
        collection_name="versioned", vector_size=3, client=client
    )
    return helper, client


def test_point_count_reads_collection_info():
    helper, client = helper_with_client(points_count=12)

    assert helper.point_count() == 12
    client.get_collection.assert_called_once_with(helper.collection_name)


def test_switch_alias_replaces_existing_alias_with_1_18_models():
    helper, client = helper_with_client(aliases=("stable",))

    helper.switch_alias("stable")

    operations = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    delete, create = operations
    assert delete.delete_alias.alias_name == "stable"
    assert create.create_alias.alias_name == "stable"
    assert create.create_alias.collection_name == helper.collection_name


def test_switch_alias_requires_permission_to_delete_legacy_collection():
    helper, client = helper_with_client(collections=("stable",))

    with pytest.raises(RuntimeError, match="legacy collection"):
        helper.switch_alias("stable")

    client.delete_collection.assert_not_called()


def test_switch_alias_deletes_authorized_legacy_collection_before_creation():
    helper, client = helper_with_client(collections=("stable",))

    helper.switch_alias("stable", delete_legacy_collection=True)

    client.delete_collection.assert_called_once_with("stable")
    operations = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    assert operations[0].create_alias.alias_name == "stable"
