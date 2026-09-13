from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DigestCache:
    _entries: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def sha256(self, path: Path) -> str:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        cached = self._entries.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        self._entries[key] = value
        return value


process_digest_cache = DigestCache()
