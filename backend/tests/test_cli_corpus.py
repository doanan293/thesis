import asyncio
import shutil
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from pharma_agent import cli
from pharma_agent.domain.corpus.models import build_snapshot
from pharma_agent.infrastructure.corpus_factory import CorpusServices
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    FIXTURE_DIR,
    small_bundle,
    with_section_text,
)
from tests.corpus_memory import (
    CorpusAdapters,
    SteppingClock,
    build_importer,
    build_release_service,
)
from tests.fakes import FakeEmbedder


@dataclass
class FakeCorpus:
    adapters: CorpusAdapters = field(default_factory=CorpusAdapters)
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    clock: SteppingClock = field(default_factory=SteppingClock)
    opened: int = 0

    @asynccontextmanager
    async def open(self, settings: Settings) -> AsyncGenerator[CorpusServices]:
        self.opened += 1
        yield CorpusServices(
            importer=build_importer(self.adapters, self.embedder, clock=self.clock),
            releases=build_release_service(
                self.adapters, self.embedder, clock=self.clock
            ),
            index=self.adapters.index,
            embedder=self.embedder,
        )


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch) -> FakeCorpus:
    fake = FakeCorpus()
    monkeypatch.setattr(cli, "open_corpus_services", fake.open)
    return fake


def invoke(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["corpus", *args])


def test_import_publishes_then_reports_no_change(corpus: FakeCorpus) -> None:
    args = ("import", str(FIXTURE_DIR), "--collection", "formulary", "--publish")

    first = invoke(*args)
    assert first.exit_code == 0, first.output
    assert "imported: formulary release 1 " in first.output
    assert "[ready] published" in first.output
    assert "chunks " in first.output and "points new " in first.output

    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert "no_change: formulary release 1 " in again.output
    assert corpus.opened == 2


def test_import_rejects_wrong_collection_and_invalid_bundle(
    corpus: FakeCorpus, tmp_path: Path
) -> None:
    mismatch = invoke("import", str(FIXTURE_DIR), "--collection", "other")
    assert mismatch.exit_code == 2
    assert "bundle collection is formulary, not other" in mismatch.output

    broken = tmp_path / "bundle"
    shutil.copytree(FIXTURE_DIR, broken)
    with (broken / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    invalid = invoke("import", str(broken), "--collection", "formulary")
    assert invalid.exit_code == 2
    assert "BUNDLE_INVALID" in invalid.output
    assert corpus.opened == 0


def test_release_commands(corpus: FakeCorpus, monkeypatch: pytest.MonkeyPatch) -> None:
    imported = invoke(
        "import", str(FIXTURE_DIR), "--collection", "formulary", "--publish"
    )
    assert imported.exit_code == 0, imported.output
    edited = with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú từ CLI.")
    second = asyncio.run(
        build_importer(corpus.adapters, corpus.embedder, clock=corpus.clock)(
            edited, publish=True
        )
    ).release

    listing = invoke("releases")
    assert listing.exit_code == 0, listing.output
    lines = listing.output.splitlines()
    assert lines[0].startswith(f"* formulary #2 {second.id} ready chunks=")
    assert lines[1].startswith("  formulary #1 ")
    assert "no releases" in invoke("releases", "--collection", "missing").output

    rolled = invoke("rollback", "--collection", "formulary")
    assert rolled.exit_code == 0, rolled.output
    assert "rolled back formulary to release 1 " in rolled.output
    republished = invoke("publish", str(second.id))
    assert republished.exit_code == 0, republished.output
    assert f"published release 2 {second.id}" in republished.output
    missing = invoke("publish", str(uuid.uuid4()))
    assert missing.exit_code == 1 and "RELEASE_NOT_FOUND" in missing.output

    monkeypatch.setenv("PHARMA_CORPUS__GC_KEEP", "5")
    kept = invoke("gc", "--collection", "formulary")
    assert kept.exit_code == 0, kept.output
    assert "retired 0 releases (keep 5)" in kept.output
    collected = invoke("gc", "--collection", "formulary", "--keep", "1")
    assert collected.exit_code == 0, collected.output
    assert "retired 1 releases (keep 1)" in collected.output

    corpus.adapters.index.points.clear()
    reindexed = invoke("reindex", "--collection", "formulary")
    assert reindexed.exit_code == 0, reindexed.output
    assert f"points upserted {len(build_snapshot(edited).chunks)}" in reindexed.output
    assert f"release {second.id}: " in reindexed.output

    unknown = invoke("rollback", "--collection", "missing")
    assert unknown.exit_code == 1 and "COLLECTION_NOT_FOUND" in unknown.output
