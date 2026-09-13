# Data layout

`data/heavy/` is the local payload boundary. It is ignored by Git and contains
raw binaries, generated corpus/evaluation files, caches, build workspaces, and
retrieval candidates/rerank/per-query artifacts. Archive it as one unit when
moving data to Drive:

```bash
zip -r seed-pipeline-heavy.zip data/heavy
```

Restore by unpacking the archive at the project root so that the directory is
again `data/heavy/`.

Tracked data is intentionally small:

- `resources/`: maintained mappings, glossary, curation inputs, URL lists, and
  syllable data.
- `manifests/`: source and corpus provenance snapshots.
- `retrieval_eval/`: run metadata, candidate-manifest snapshots, and Markdown
  summaries. Large run payloads with the same run name are under
  `heavy/retrieval_eval/`.

`heavy/runtime_kaggle_profiles/` is generated for a particular runtime and is
included in the heavy archive; it can be regenerated when the profile is
missing or invalid.
