from __future__ import annotations

import typer

from pharma_lab.cli.commands.build import build
from pharma_lab.cli.commands.bundle import bundle_app
from pharma_lab.cli.commands.data import data_app
from pharma_lab.cli.commands.doctor import doctor
from pharma_lab.cli.commands.e2e import e2e_app
from pharma_lab.cli.commands.embed import embed_app
from pharma_lab.cli.commands.evaluation import evaluation_app
from pharma_lab.cli.commands.metrics import metrics_app
from pharma_lab.cli.commands.rerank import rerank
from pharma_lab.cli.commands.retrieve import retrieve
from pharma_lab.cli.commands.source import source_app
from pharma_lab.cli.commands.validate import validate
from pharma_lab.cli.runtime import CliState
from pharma_lab.config.environment import load_project_env

app = typer.Typer(
    name="pharma-lab",
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
    """Operate pharma-lab by explicit stage."""
    load_project_env()
    ctx.obj = CliState(json_output=json_output, debug=debug)


app.command("doctor")(doctor)
app.command("build")(build)
app.command("validate")(validate)
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(embed_app, name="embed")
app.add_typer(source_app, name="source")
app.add_typer(bundle_app, name="bundle")
app.add_typer(data_app, name="data")
app.command("retrieve")(retrieve)
app.command("rerank")(rerank)
app.add_typer(metrics_app, name="metrics")
app.add_typer(e2e_app, name="e2e")


def main() -> None:
    app()
