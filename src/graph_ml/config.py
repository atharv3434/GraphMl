"""Configuration loading for the graph ML pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    nodes_csv: str
    edges_csv: str
    node_id_column: str = "node_id"
    label_column: str = "label"
    split_column: str | None = "split"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    random_state: int = 42
    directed: bool = False


@dataclass
class ModelConfig:
    type: str = "graphsage"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4


@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    early_stopping_patience: int = 20
    device: str = "auto"
    seed: int = 42


@dataclass
class OutputConfig:
    model_dir: str = "checkpoints"
    model_name: str = "gnn_model.pt"
    metrics_path: str = "checkpoints/metrics.json"


@dataclass
class PipelineConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            output=OutputConfig(**raw.get("output", {})),
            log_level=raw.get("logging", {}).get("level", "INFO"),
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
