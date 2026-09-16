from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pharma_lab.config.environment import PROJECT_ENV_FILE, load_project_env
from pharma_lab.config.paths import GGUF_ROOT, PROJECT_ROOT
from pharma_lab.integrations.kaggle.api import (
    KaggleCommandRunner,
    config_view_command,
)
from pharma_lab.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceService,
    LocalCheckpointSource,
    ProfileCheckpointService,
)
from pharma_lab.integrations.kaggle.checkpoints import CheckpointService
from pharma_lab.integrations.kaggle.config import (
    KaggleAccountProfile,
    OwnerConfiguration,
    discover_account_profiles,
    profile_owner_configuration,
    profile_runner_environment,
    resolve_account_profile,
    resolve_owner_configuration,
)
from pharma_lab.integrations.kaggle.dataset_service import DatasetService
from pharma_lab.integrations.kaggle.dependencies import (
    DependencyService,
    default_desired_datasets,
)
from pharma_lab.integrations.kaggle.kernel_service import KernelService
from pharma_lab.integrations.kaggle.kernels import PipelineKernelService
from pharma_lab.integrations.kaggle.models import (
    KernelPresence,
    KernelStatus,
    PipelineResult,
    StageName,
    StageRequest,
)
from pharma_lab.integrations.kaggle.orchestrator import (
    ArtifactSink,
    KagglePipelineOrchestrator,
)
from pharma_lab.integrations.kaggle.parsers import parse_kaggle_username
from pharma_lab.integrations.kaggle.stages import get_stage_adapter
from pharma_lab.integrations.kaggle.workspace import unwind_on_sigterm
from pharma_lab.runtime.runtime_profiles import RuntimeCandidate, canonical_sha256

DEFAULT_ENV_PATH = PROJECT_ENV_FILE
AUTO_ACCOUNT = "auto"


@dataclass(frozen=True)
class KaggleExecutionContext:
    profile: KaggleAccountProfile | None
    owners: OwnerConfiguration
    runner: KaggleCommandRunner


def load_kaggle_env(path: Path = DEFAULT_ENV_PATH) -> None:
    load_project_env(path)


def resolve_execution_context(
    kaggle_account: str | None,
    *,
    env_file: Path = DEFAULT_ENV_PATH,
) -> KaggleExecutionContext:
    load_kaggle_env(env_file)
    profile = resolve_account_profile(kaggle_account, os.environ)
    if profile is None:
        runner = KaggleCommandRunner()
        return KaggleExecutionContext(None, owner_configuration(runner), runner)
    runner = KaggleCommandRunner(
        environment=profile_runner_environment(profile, os.environ)
    )
    owners = profile_owner_configuration(profile, os.environ)
    return KaggleExecutionContext(profile, owners, runner)


def resolve_profile_execution_contexts(
    *, env_file: Path = DEFAULT_ENV_PATH
) -> tuple[KaggleExecutionContext, ...]:
    load_kaggle_env(env_file)
    contexts: list[KaggleExecutionContext] = []
    for name in discover_account_profiles(os.environ):
        profile = resolve_account_profile(name, os.environ)
        assert profile is not None
        runner = KaggleCommandRunner(
            environment=profile_runner_environment(profile, os.environ)
        )
        contexts.append(
            KaggleExecutionContext(
                profile,
                profile_owner_configuration(profile, os.environ),
                runner,
            )
        )
    return tuple(contexts)


def resolve_session_contexts(
    kaggle_account: str | None,
    *,
    env_file: Path = DEFAULT_ENV_PATH,
) -> tuple[KaggleExecutionContext, ...]:
    """Accounts a quota-aware session may use: every profile for `auto`, else one."""
    if kaggle_account is not None and kaggle_account.strip().casefold() == AUTO_ACCOUNT:
        contexts = resolve_profile_execution_contexts(env_file=env_file)
    else:
        contexts = (resolve_execution_context(kaggle_account, env_file=env_file),)
    if not contexts or any(context.profile is None for context in contexts):
        raise ValueError(
            "Kaggle sessions need account profiles: set KAGGLE_ACCOUNT_DEFAULT, "
            "KAGGLE_SHARED_OWNER and KAGGLE_ACC<N>_USERNAME/KAGGLE_ACC<N>_API_TOKEN "
            "in pharma-lab/.env"
        )
    return contexts


def active_kernel_profile(
    contexts: Sequence[KaggleExecutionContext],
    *,
    stage: StageName,
    model: str,
    input_path: Path,
    output_dir: Path,
    gguf_root: Path = GGUF_ROOT,
    runtime_profile: RuntimeCandidate | None = None,
) -> str | None:
    """Profile whose account already queues or runs this job's kernel, if any.

    A command restarted after a crash or reboot attaches to that kernel instead of
    starting a duplicate on the account with the most quota.
    """
    profiled = [context for context in contexts if context.profile is not None]
    if not profiled:
        return None
    job = get_stage_adapter(stage).build_job(
        StageRequest(
            stage,
            model,
            Path(input_path),
            Path(output_dir),
            Path(gguf_root),
            profiled[0].owners,
            runtime_profile=runtime_profile,
        )
    )
    for context in profiled:
        assert context.profile is not None
        kernels = PipelineKernelService(
            KernelService(context.runner, context.owners.execution),
            owner=context.owners.execution,
            source_root=PROJECT_ROOT / "src",
        )
        state = kernels.discover(job)
        if state.presence is KernelPresence.EXISTS and state.status in {
            KernelStatus.QUEUED,
            KernelStatus.RUNNING,
        }:
            return context.profile.name
    return None


def owner_configuration(
    runner: KaggleCommandRunner,
    *,
    owner: str | None = None,
    runtime_owner: str | None = None,
    corpus_owner: str | None = None,
    checkpoint_owner: str | None = None,
) -> OwnerConfiguration:
    authenticated = None
    if not os.environ.get("KAGGLE_USERNAME") and not owner:
        try:
            authenticated = parse_kaggle_username(
                runner.run(config_view_command(), capture_output=True)
            )
        except Exception:
            authenticated = None
    values = type(
        "OwnerOptions",
        (),
        {
            "owner": owner,
            "runtime_owner": runtime_owner,
            "corpus_owner": corpus_owner,
            "checkpoint_owner": checkpoint_owner,
        },
    )()
    return resolve_owner_configuration(values, os.environ, authenticated)


def make_orchestrator(
    owners: OwnerConfiguration,
    runner: KaggleCommandRunner,
    *,
    target_profile: str | None = None,
    checkpoint_contexts: tuple[KaggleExecutionContext, ...] = (),
    temp_root: Path = Path("/tmp"),
    local_checkpoint: LocalCheckpointSource | None = None,
    artifact_sink: ArtifactSink | None = None,
) -> KagglePipelineOrchestrator:
    datasets = DatasetService(runner, owners.checkpoint)
    kernels = PipelineKernelService(
        KernelService(runner, owners.execution),
        owner=owners.execution,
        source_root=PROJECT_ROOT / "src",
    )
    checkpoints = CheckpointService(datasets, owners.checkpoint)
    inheritance = None
    if target_profile is not None:
        candidates = []
        target_checkpoint = checkpoints
        for context in checkpoint_contexts:
            if context.profile is None:
                continue
            if context.profile.name == target_profile:
                target_checkpoint = checkpoints
                candidate_checkpoint = checkpoints
            else:
                candidate_checkpoint = CheckpointService(
                    DatasetService(context.runner, context.owners.checkpoint),
                    context.owners.checkpoint,
                )
            candidates.append(
                ProfileCheckpointService(context.profile.name, candidate_checkpoint)
            )
        inheritance = CheckpointInheritanceService(
            target_profile=target_profile,
            target=target_checkpoint,
            candidates=tuple(candidates),
            temp_root=temp_root,
            local=local_checkpoint,
        )
    dependencies = DependencyService(
        DatasetService(runner, owners.execution),
        default_desired_datasets,
        publishers=tuple(
            DatasetService(context.runner, context.owners.execution)
            for context in checkpoint_contexts
            if context.profile is not None
        ),
    )
    return KagglePipelineOrchestrator(
        get_stage_adapter,
        dependencies,
        checkpoints,
        kernels,
        temp_root=temp_root,
        checkpoint_inheritance=inheritance,
        artifact_sink=artifact_sink,
    )


def runtime_manifest_sha256(
    runner: KaggleCommandRunner | None = None,
    owners: OwnerConfiguration | None = None,
    *,
    kaggle_account: str | None = None,
) -> str:
    """Return the fingerprint of the immutable Kaggle runtime manifest."""
    if runner is None or owners is None:
        context = resolve_execution_context(kaggle_account)
        active_runner = runner or context.runner
        active_owners = owners or context.owners
    else:
        active_runner = runner
        active_owners = owners
    from pharma_lab.integrations.kaggle.orchestrator import (
        runtime_dataset_reference,
    )

    manifest = DatasetService(active_runner, active_owners.runtime).fetch_json(
        runtime_dataset_reference(active_owners), "runtime_manifest.json"
    )
    return canonical_sha256(manifest)


def run_kaggle_stage(
    *,
    stage: StageName,
    model: str,
    input_path: Path,
    output_dir: Path,
    gguf_root: Path = GGUF_ROOT,
    force: bool = False,
    check_only: bool = False,
    max_runs: int = 10,
    budget_seconds: int = 21_600,
    benchmark_items: int | None = None,
    runtime_profile: RuntimeCandidate | None = None,
    env_file: Path = DEFAULT_ENV_PATH,
    kaggle_account: str | None = None,
    resume_remote: bool = True,
    artifact_sink: ArtifactSink | None = None,
    local_checkpoint: LocalCheckpointSource | None = None,
) -> PipelineResult:
    context = resolve_execution_context(kaggle_account, env_file=env_file)
    if local_checkpoint is not None and context.profile is None:
        raise ValueError("a local checkpoint source needs a Kaggle account profile")
    runner = context.runner
    owners = context.owners
    request = StageRequest(
        stage,
        model,
        Path(input_path),
        Path(output_dir),
        Path(gguf_root),
        owners,
        force,
        check_only,
        max_runs,
        budget_seconds,
        benchmark_items,
        runtime_profile,
        resume_remote,
    )
    contexts = (
        resolve_profile_execution_contexts(env_file=env_file)
        if context.profile is not None
        else ()
    )
    with unwind_on_sigterm():
        return make_orchestrator(
            owners,
            runner,
            target_profile=context.profile.name if context.profile else None,
            checkpoint_contexts=contexts,
            local_checkpoint=local_checkpoint,
            artifact_sink=artifact_sink,
        ).run(request)
