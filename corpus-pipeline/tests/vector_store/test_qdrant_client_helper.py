from types import SimpleNamespace

import pytest

from corpus_pipeline.vector_store.qdrant_client_helper import QdrantClientHelper


class FakeQdrantTransport:
    def __init__(self, *, collections=(), aliases=(), points_count=0):
        self.collections = set(collections)
        self.aliases = set(aliases)
        self.points_count = points_count
        self.deleted_collections = []
        self.alias_updates = []
        self.get_collection_names = []

    def get_collections(self):
        self.get_collection_names.append("all")
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[SimpleNamespace(alias_name=name) for name in self.aliases]
        )

    def get_collection(self, name):
        self.get_collection_names.append(name)
        return SimpleNamespace(points_count=self.points_count)

    def delete_collection(self, name):
        self.deleted_collections.append(name)
        self.collections.discard(name)

    def update_collection_aliases(self, *, change_aliases_operations):
        self.alias_updates.append(change_aliases_operations)


def helper_with_fake_transport(**kwargs):
    transport = FakeQdrantTransport(**kwargs)
    helper = QdrantClientHelper.__new__(QdrantClientHelper)
    helper.client = transport
    helper.collection_name = "versioned"
    helper.vector_size = 3
    return helper, transport


def test_point_count_reads_collection_info():
    helper, transport = helper_with_fake_transport(points_count=12)

    assert helper.point_count() == 12
    assert transport.get_collection_names == [helper.collection_name]


def test_switch_alias_replaces_existing_alias_with_1_18_models():
    helper, transport = helper_with_fake_transport(aliases={"stable"})

    helper.switch_alias("stable")

    delete, create = transport.alias_updates[0]
    assert delete.delete_alias.alias_name == "stable"
    assert create.create_alias.alias_name == "stable"
    assert create.create_alias.collection_name == helper.collection_name


def test_switch_alias_requires_permission_to_delete_legacy_collection():
    helper, transport = helper_with_fake_transport(collections={"stable"})

    with pytest.raises(RuntimeError, match="legacy collection"):
        helper.switch_alias("stable")

    assert transport.deleted_collections == []


def test_switch_alias_deletes_authorized_legacy_collection_before_creation():
    helper, transport = helper_with_fake_transport(collections={"stable"})

    helper.switch_alias("stable", delete_legacy_collection=True)

    assert transport.deleted_collections == ["stable"]
    assert transport.alias_updates[0][0].create_alias.alias_name == "stable"
