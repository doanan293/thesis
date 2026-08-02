from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetInventory,
    DatasetPresence,
    DatasetService,
)
from corpus_pipeline.integrations.kaggle.errors import KaggleCommandError

OWNER = "owner"
REFERENCE = f"{OWNER}/cache"


class SequenceRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def run(self, args, **kwargs):
        del kwargs
        self.commands.append(args)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def command_error(detail: str) -> KaggleCommandError:
    return KaggleCommandError(
        operation="dataset status",
        target=REFERENCE,
        returncode=1,
        stdout="",
        stderr=detail,
    )


def test_non_authoritative_empty_listing_is_unknown():
    runner = SequenceRunner(
        [
            command_error("403 Forbidden"),
            "- username: owner\n",
            "unexpected listing output\n",
        ]
    )
    state = DatasetService(runner, OWNER).inspect_state(REFERENCE, active_owner=OWNER)
    assert state.presence is DatasetPresence.UNKNOWN
    assert "authoritative CSV header" in state.detail


def test_owned_csv_empty_listing_is_absent():
    runner = SequenceRunner(
        [
            command_error("403 Forbidden"),
            "- username: owner\n",
            "No datasets found\n",
        ]
    )
    state = DatasetService(runner, OWNER).inspect_state(REFERENCE, active_owner=OWNER)
    assert state.presence is DatasetPresence.ABSENT


def test_inventory_fetches_each_resource_once():
    runner = SequenceRunner(
        [
            '{"status":"READY"}',
            '{"status":"READY"}',
        ]
    )
    service = DatasetService(runner, OWNER)
    manifests = iter([{"fingerprint": "abc"}, {"fingerprint": "def"}])
    service.fetch_json = lambda reference, filename: next(manifests)
    inventory = DatasetInventory.load(
        service,
        {
            "owner/runtime": "dependency_manifest.json",
            "owner/model": "dependency_manifest.json",
        },
        active_owner=OWNER,
    )
    assert inventory.get("owner/runtime").manifest == {"fingerprint": "abc"}
    assert len(runner.commands) == 2


def test_inventory_treats_not_found_manifest_as_missing():
    runner = SequenceRunner(
        [
            '{"status":"READY"}',
            command_error("404 Client Error: Not Found"),
        ]
    )
    inventory = DatasetInventory.load(
        DatasetService(runner, OWNER),
        {REFERENCE: "dependency_manifest.json"},
        active_owner=OWNER,
    )

    state = inventory.get(REFERENCE)
    assert state.presence is DatasetPresence.EXISTS
    assert state.status == "READY"
    assert state.manifest is None
    assert "dependency_manifest.json" in state.detail
