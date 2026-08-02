from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from corpus_pipeline.integrations.kaggle.api import (
    KaggleCommandRunner,
    config_view_command,
    dataset_create_command,
    dataset_file_download_command,
    dataset_list_mine_command,
    dataset_metadata,
    dataset_status_command,
    dataset_version_command,
)
from corpus_pipeline.integrations.kaggle.errors import (
    KaggleCommandError,
    KaggleRemoteStateError,
)
from corpus_pipeline.integrations.kaggle.parsers import (
    parse_dataset_references,
    parse_dataset_status,
    parse_kaggle_username,
)


class DatasetPresence(Enum):
    ABSENT = "absent"
    EXISTS = "exists"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DatasetRemoteState:
    presence: DatasetPresence
    status: str | None = None
    current_version: int | None = None
    detail: str = ""
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedDataset:
    reference: str
    changed: bool


def _command_detail(error: BaseException) -> str:
    return str(error).strip()


def _is_not_found(error: KaggleCommandError) -> bool:
    detail = f"{error.stdout}\n{error.stderr}".casefold()
    return "404" in detail or "not found" in detail


class DatasetService:
    def __init__(self, runner: KaggleCommandRunner, owner: str):
        self.runner = runner
        self.owner = owner

    def status(self, reference: str) -> str:
        return parse_dataset_status(
            self.runner.run(dataset_status_command(reference), capture_output=True)
        )

    def inspect_state(
        self, reference: str, *, active_owner: str | None = None
    ) -> DatasetRemoteState:
        try:
            output = self.runner.run(
                dataset_status_command(reference), capture_output=True
            )
        except (subprocess.CalledProcessError, KaggleCommandError) as exc:
            detail = _command_detail(exc)
            folded = detail.casefold()
            if "not found" in folded or "404" in folded:
                return DatasetRemoteState(DatasetPresence.ABSENT, detail=detail)
            if "403" in folded and active_owner is not None:
                return self._inspect_owned_forbidden(
                    reference, active_owner=active_owner, status_detail=detail
                )
            return DatasetRemoteState(DatasetPresence.UNKNOWN, detail=detail)
        if not output.strip():
            return DatasetRemoteState(
                DatasetPresence.UNKNOWN,
                detail="Kaggle dataset status returned an empty response",
            )
        try:
            return DatasetRemoteState(
                DatasetPresence.EXISTS, status=parse_dataset_status(output)
            )
        except (RuntimeError, ValueError) as exc:
            return DatasetRemoteState(DatasetPresence.UNKNOWN, detail=str(exc))

    def _inspect_owned_forbidden(
        self, reference: str, *, active_owner: str, status_detail: str
    ) -> DatasetRemoteState:
        reference_owner, separator, slug = reference.partition("/")
        if not separator or reference_owner.casefold() != active_owner.casefold():
            return DatasetRemoteState(
                DatasetPresence.UNKNOWN,
                detail=f"cannot verify optional dataset {reference}: owner differs",
            )
        try:
            authenticated_owner = parse_kaggle_username(
                self.runner.run(config_view_command(), capture_output=True)
            )
        except (subprocess.CalledProcessError, KaggleCommandError, RuntimeError) as exc:
            return DatasetRemoteState(
                DatasetPresence.UNKNOWN,
                detail=f"cannot verify authenticated user: {_command_detail(exc)}",
            )
        if authenticated_owner.casefold() != active_owner.casefold():
            return DatasetRemoteState(
                DatasetPresence.UNKNOWN,
                detail=f"authenticated user {authenticated_owner!r} does not match {active_owner!r}",
            )
        for page in range(1, 4):
            try:
                output = self.runner.run(
                    dataset_list_mine_command(slug, page=page), capture_output=True
                )
            except (subprocess.CalledProcessError, KaggleCommandError) as exc:
                return DatasetRemoteState(
                    DatasetPresence.UNKNOWN,
                    detail=f"owned dataset listing failed: {_command_detail(exc)}",
                )
            lines = output.splitlines()
            has_header = any(
                line.split(",", 1)[0].strip().casefold() == "ref" for line in lines
            )
            if not has_header:
                if any(
                    line.strip().casefold() == "no datasets found" for line in lines
                ):
                    return DatasetRemoteState(
                        DatasetPresence.ABSENT,
                        detail="owned dataset listing contained no rows",
                    )
                return DatasetRemoteState(
                    DatasetPresence.UNKNOWN,
                    detail=f"non-authoritative dataset listing for {reference}: expected authoritative CSV header",
                )
            if reference.casefold() in {
                item.casefold() for item in parse_dataset_references(output)
            }:
                return DatasetRemoteState(
                    DatasetPresence.EXISTS, status=None, detail=status_detail
                )
        return DatasetRemoteState(
            DatasetPresence.ABSENT, detail="owned dataset listing contained no rows"
        )

    def fetch_json(
        self, reference: str, filename: str, destination: Path | None = None
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="kaggle-dataset-") as raw:
            target = Path(destination) if destination is not None else Path(raw)
            target.mkdir(parents=True, exist_ok=True)
            self.runner.run(dataset_file_download_command(reference, filename, target))
            candidates = list(target.rglob(filename))
            if len(candidates) != 1:
                raise KaggleRemoteStateError(
                    f"Expected one {filename} in downloaded dataset {reference}, found {len(candidates)}"
                )
            try:
                payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise KaggleRemoteStateError(
                    f"Invalid {filename} in dataset {reference}"
                ) from exc
            if not isinstance(payload, dict):
                raise KaggleRemoteStateError(
                    f"Dataset file {filename} must contain an object"
                )
            return payload

    def download_file(self, reference: str, filename: str, destination: Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        self.runner.run(dataset_file_download_command(reference, filename, destination))
        matches = list(destination.rglob(filename))
        if len(matches) != 1:
            raise KaggleRemoteStateError(
                f"Expected one {filename} in downloaded dataset {reference}, found {len(matches)}"
            )
        return matches[0]

    def ensure_dataset(
        self,
        slug: str,
        title: str,
        path: Path,
        *,
        public: bool = False,
        active_owner: str | None = None,
    ) -> PreparedDataset:
        reference = f"{self.owner}/{slug}"
        state = self.inspect_state(reference, active_owner=active_owner or self.owner)
        if state.presence is DatasetPresence.UNKNOWN:
            raise KaggleRemoteStateError(
                f"Cannot publish {reference}: remote state is unknown: {state.detail}"
            )
        metadata_path = Path(path) / "dataset-metadata.json"
        metadata_path.write_text(
            json.dumps(dataset_metadata(self.owner, slug, title, public), indent=2),
            encoding="utf-8",
        )
        if state.presence is DatasetPresence.ABSENT:
            self.runner.run(dataset_create_command(Path(path), public=public))
            return PreparedDataset(reference, True)
        self.runner.run(dataset_version_command(Path(path), message=title))
        return PreparedDataset(reference, True)

    def wait_for_dataset_ready(
        self, reference: str, *, max_attempts: int = 300
    ) -> None:
        for _ in range(max_attempts):
            try:
                if self.status(reference) == "READY":
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"Dataset {reference} was not ready in time")

    def require_ready(self, reference: str, guidance: str) -> None:
        state = self.inspect_state(reference, active_owner=self.owner)
        if state.presence is DatasetPresence.ABSENT:
            raise KaggleRemoteStateError(
                f"Required dataset is missing: {reference}; run {guidance}"
            )
        if state.presence is DatasetPresence.UNKNOWN:
            raise KaggleRemoteStateError(
                f"Required dataset state is unknown: {reference}: {state.detail}"
            )
        if state.status != "READY":
            raise KaggleRemoteStateError(
                f"Required dataset is not READY: {reference} ({state.status})"
            )


@dataclass
class DatasetInventory:
    service: DatasetService
    active_owner: str
    states: dict[str, DatasetRemoteState]

    @classmethod
    def load(
        cls, service: DatasetService, resources: dict[str, str], *, active_owner: str
    ) -> DatasetInventory:
        states: dict[str, DatasetRemoteState] = {}
        for reference, manifest_filename in resources.items():
            state = service.inspect_state(reference, active_owner=active_owner)
            if state.presence is DatasetPresence.EXISTS and state.status == "READY":
                try:
                    manifest = service.fetch_json(reference, manifest_filename)
                except KaggleCommandError as exc:
                    if not _is_not_found(exc):
                        raise
                    state = replace(
                        state,
                        detail=f"missing reconciliation manifest {manifest_filename}",
                        manifest=None,
                    )
                else:
                    state = replace(state, manifest=manifest)
            states[reference] = state
        return cls(service, active_owner, states)

    def get(self, reference: str) -> DatasetRemoteState:
        return self.states[reference]

    def remember(self, reference: str, state: DatasetRemoteState) -> None:
        self.states[reference] = state
