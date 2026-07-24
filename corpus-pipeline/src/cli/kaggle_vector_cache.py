from __future__ import annotations

import argparse
import os
from pathlib import Path

from kaggle_vector_cache.canonical_gguf import resolve_publishable_artifacts
from kaggle_vector_cache.kaggle_api import KaggleCommandRunner
from kaggle_vector_cache.models import MODEL_CATALOG
from kaggle_vector_cache.orchestrator import (
    KaggleVectorCacheOrchestrator,
)
from kaggle_vector_cache.temp_workspace import unwind_on_sigterm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
DEFAULT_ENV_PATH = PROJECT_ROOT.parent / ".env"


def load_kaggle_env(path: Path = DEFAULT_ENV_PATH) -> None:
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def add_remote_options(parser):
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    parser.add_argument("--runtime-owner")
    parser.add_argument("--corpus-owner")
    parser.add_argument("--checkpoint-owner")
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("/tmp"),
        help="Root for auto-cleaned Kaggle transfer workspaces.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=15)


def add_run_options(parser):
    add_remote_options(parser)
    parser.add_argument("--total-budget-seconds", type=int, default=21_600)
    parser.add_argument("--export-reserve-seconds", type=int, default=900)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--retune", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    load_kaggle_env()
    parser = argparse.ArgumentParser(
        description="Build resumable embedding caches with llama.cpp on Kaggle."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("llama-cpp-runtime")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    build = runtime_commands.add_parser("build")
    add_remote_options(build)
    build.add_argument("--build-timeout-seconds", type=int, default=3600)
    build.add_argument("--force", action="store_true")
    corpus = commands.add_parser("corpus")
    corpus_commands = corpus.add_subparsers(
        dest="corpus_command",
        required=True,
    )
    corpus_publish = corpus_commands.add_parser("publish")
    add_remote_options(corpus_publish)
    corpus_publish.add_argument(
        "--corpus-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/rag-final/chunks.jsonl",
    )
    models = commands.add_parser("models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    publish = model_commands.add_parser("publish")
    add_remote_options(publish)
    publish.add_argument("--gguf-root", type=Path, default=DEFAULT_GGUF_ROOT)
    run = commands.add_parser("run")
    add_run_options(run)
    run.add_argument("--model", choices=sorted(MODEL_CATALOG), required=True)
    run.add_argument("--repeat-until-complete", action="store_true")
    benchmark = commands.add_parser("benchmark")
    add_remote_options(benchmark)
    benchmark.add_argument("--model", choices=sorted(MODEL_CATALOG), required=True)
    benchmark.add_argument("--total-budget-seconds", type=int, default=600)
    run_all = commands.add_parser("run-all")
    add_run_options(run_all)
    status = commands.add_parser("status")
    add_remote_options(status)
    status.add_argument("--model", choices=sorted(MODEL_CATALOG))
    download = commands.add_parser("download")
    add_remote_options(download)
    download.add_argument("--model", choices=sorted(MODEL_CATALOG), required=True)
    download.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "vector_embeddings",
    )
    return parser


def make_orchestrator(args) -> KaggleVectorCacheOrchestrator:
    if not args.owner:
        raise ValueError("Kaggle owner is required via --owner or KAGGLE_USERNAME")
    return KaggleVectorCacheOrchestrator(
        owner=args.owner,
        runtime_owner=args.runtime_owner or args.owner,
        corpus_owner=args.corpus_owner or args.owner,
        checkpoint_owner=args.checkpoint_owner,
        project_root=PROJECT_ROOT,
        temp_root=args.temp_root,
        output_dir=getattr(args, "output_dir", None),
        command_runner=KaggleCommandRunner(dry_run=args.dry_run),
        poll_interval_seconds=args.poll_interval,
    )


def run_model(orchestrator, model, args, repeat):
    max_runs = args.max_runs if repeat else 1
    result = orchestrator.run_until_complete(
        model,
        max_runs=max_runs,
        total_budget_seconds=args.total_budget_seconds,
        export_reserve_seconds=args.export_reserve_seconds,
        retune=args.retune,
    )
    print(
        f"{model}: complete={result.complete}/{result.total}, missing={result.missing}"
    )
    return 0 if result.is_complete else 2


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with unwind_on_sigterm():
            orchestrator = make_orchestrator(args)
            if args.command == "corpus":
                print(orchestrator.publish_corpus(args.corpus_path))
                return 0
            if args.command == "llama-cpp-runtime":
                print(orchestrator.build_llama_cpp_runtime(force=args.force))
                return 0
            if args.command == "models":
                for artifact in resolve_publishable_artifacts(args.gguf_root):
                    print(orchestrator.publish_canonical_gguf(artifact))
                return 0
            if args.command == "run":
                return run_model(
                    orchestrator, args.model, args, args.repeat_until_complete
                )
            if args.command == "benchmark":
                result = orchestrator.benchmark(
                    args.model,
                    args.total_budget_seconds,
                )
                print(
                    f"{args.model}: complete={result.complete}/{result.total}, "
                    f"missing={result.missing}"
                )
                return 0
            if args.command == "run-all":
                for model in MODEL_CATALOG:
                    code = run_model(orchestrator, model, args, True)
                    if code:
                        return code
                return 0
            if args.command == "status":
                models = [args.model] if args.model else list(MODEL_CATALOG)
                for model in models:
                    result = orchestrator.inspect_cloud(model)
                    print(
                        f"{model}: {result.complete}/{result.total}, "
                        f"missing={result.missing}, "
                        f"kernel={result.kernel_status or 'NOT_FOUND'}"
                    )
                return 0
            cache_path, manifest_path = orchestrator.download_checkpoint(
                args.model,
                args.output_dir,
            )
            print(cache_path)
            print(manifest_path)
            return 0
    except KeyboardInterrupt:
        print("error: interrupted")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
