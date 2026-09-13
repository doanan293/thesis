from __future__ import annotations

import typer

from seed_pipeline.cli.commands.build import build
from seed_pipeline.cli.commands.doctor import doctor
from seed_pipeline.cli.commands.embed import embed_app
from seed_pipeline.cli.commands.evaluation import evaluation_app
from seed_pipeline.cli.commands.metrics import metrics
from seed_pipeline.cli.commands.rerank import rerank
from seed_pipeline.cli.commands.retrieve import retrieve
from seed_pipeline.cli.commands.source import source_app
from seed_pipeline.cli.commands.validate import validate
from seed_pipeline.cli.commands.vectors import vectors_app
from seed_pipeline.cli.runtime import CliState
from seed_pipeline.config.environment import load_project_env

app = typer.Typer(
    name="seed",
    help="Build the seed corpus, export knowledge bundles, embed and evaluate retrieval.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _callback(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show full exception tracebacks."
    ),
) -> None:
    """Operate the seed pipeline by explicit stage."""
    load_project_env()
    ctx.obj = CliState(json_output=json_output, debug=debug)


app.command("doctor")(doctor)
app.command("build")(build)
app.command("validate")(validate)
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(embed_app, name="embed")
app.add_typer(vectors_app, name="vectors")
app.add_typer(source_app, name="source")
app.command("retrieve")(retrieve)
app.command("rerank")(rerank)
app.command("metrics")(metrics)


def main() -> None:
    app()
