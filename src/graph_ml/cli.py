"""Command-line interface for the graph ML pipeline."""
from __future__ import annotations

import json
import logging

import click

from graph_ml.config import PipelineConfig, setup_logging
from graph_ml.engine import train_and_evaluate
from graph_ml.inference import predict_all, predict_nodes

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Production-ready node classification pipeline (GCN / GraphSAGE / GAT)."""


@cli.command()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def train(config_path: str) -> None:
    """Train the GNN and evaluate on the held-out validation/test splits."""
    config = PipelineConfig.from_yaml(config_path)
    setup_logging(config.log_level)
    metrics = train_and_evaluate(config)
    click.echo(json.dumps({
        "best_val_f1_macro": metrics["best_val_f1_macro"],
        "test_metrics": metrics["test_metrics"],
    }, indent=2))


@cli.command()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
@click.option("--node-id", "node_ids", multiple=True, help="Predict for specific node id(s) (repeatable). Omit to predict for all nodes.")
def predict(config_path: str, node_ids: tuple[str, ...]) -> None:
    """Predict labels for nodes in the graph."""
    config = PipelineConfig.from_yaml(config_path)
    setup_logging(config.log_level)

    if node_ids:
        # node ids in the CSV may not be strings; try to coerce numerics for convenience
        coerced = []
        for n in node_ids:
            try:
                coerced.append(int(n))
            except ValueError:
                coerced.append(n)
        results = predict_nodes(config, coerced)
    else:
        results = predict_all(config)

    click.echo(json.dumps({str(k): v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    cli()
