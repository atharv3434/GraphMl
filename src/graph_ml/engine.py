"""Training and evaluation loop for transductive node classification."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

from graph_ml.config import PipelineConfig
from graph_ml.data.loader import load_graph_data
from graph_ml.models.factory import build_model

logger = logging.getLogger(__name__)


def _resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def _evaluate_mask(model, data: Data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[mask].argmax(dim=1).cpu().numpy()
        targets = data.y[mask].cpu().numpy()
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train_and_evaluate(config: PipelineConfig) -> dict[str, Any]:
    """Train a GNN on the transductive node classification task; evaluate on val/test."""
    torch.manual_seed(config.training.seed)
    device = _resolve_device(config.training.device)
    logger.info(f"Using device: {device}")

    data, node_id_to_idx, label_classes = load_graph_data(config.data)
    data = data.to(device)

    num_features = data.x.shape[1]
    num_classes = len(label_classes)
    model = build_model(config.model, num_features, num_classes).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(config.training.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = _evaluate_mask(model, data, data.val_mask)
        history.append({"epoch": epoch + 1, "train_loss": float(loss.item()), **val_metrics})

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.training.epochs} | "
                f"loss={loss.item():.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1_macro={val_metrics['f1_macro']:.4f}"
            )

        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping triggered after epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate_mask(model, data, data.test_mask)
    val_metrics_final = _evaluate_mask(model, data, data.val_mask)
    logger.info(f"Test metrics: {test_metrics}")

    metrics = {
        "best_val_f1_macro": best_val_f1,
        "final_val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "history": history,
        "model_type": config.model.type,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": num_features,
        "num_classes": num_classes,
        "label_classes": [str(c) for c in label_classes],
    }

    _persist(model, metrics, config, num_features, num_classes)
    return metrics


def _persist(model, metrics: dict[str, Any], config: PipelineConfig, num_features: int, num_classes: int) -> None:
    model_dir = Path(config.output.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / config.output.model_name
    torch.save({
        "state_dict": model.state_dict(),
        "num_features": num_features,
        "num_classes": num_classes,
        "model_config": config.model.__dict__,
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    metrics_path = Path(config.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
