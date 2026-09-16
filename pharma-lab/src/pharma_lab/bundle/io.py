"""Write knowledge bundles through a validated staging directory."""

import shutil
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleManifest,
    KnowledgeBundle,
    read_bundle,
    write_bundle,
)


def write_validated_bundle(bundle: KnowledgeBundle, output_dir: Path) -> BundleManifest:
    """Write to `.<name>.next`, validate with `read_bundle`, then replace `output_dir`."""
    output_dir = Path(output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.next")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest = write_bundle(bundle, staging)
        read_bundle(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
