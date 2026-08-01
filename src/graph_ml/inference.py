"""Load a trained GNN and predict labels for nodes in the graph."""
from __future__ import annotations

import logging

import torch

from graph_ml.config import ModelConfig, PipelineConfig
from graph_ml.data.loader import load_graph_data
from graph_ml.models.factory import build_model

logger = logging.getLogger(__name__)


def load_model_and_data(config: PipelineConfig):
    """Load the persisted model checkpoint plus the full graph data. Returns (model, data, node_id_to_idx, label_classes)."""
    checkpoint = torch.load(
        f"{config.output.model_dir}/{config.output.model_name}", map_location="cpu"
    )
    data, node_id_to_idx, label_classes = load_graph_data(config.data)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = build_model(model_config, checkpoint["num_features"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, data, node_id_to_idx, label_classes


def predict_all(config: PipelineConfig) -> dict:
    """Predict labels for every node in the graph. Returns {node_id: predicted_label}."""
    model, data, node_id_to_idx, label_classes = load_model_and_data(config)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).numpy()

    idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}
    return {
        idx_to_node_id[i]: label_classes[preds[i]]
        for i in range(len(preds))
    }


def predict_nodes(config: PipelineConfig, node_ids: list) -> dict:
    """Predict labels for a specific list of node ids."""
    model, data, node_id_to_idx, label_classes = load_model_and_data(config)

    missing = [n for n in node_ids if n not in node_id_to_idx]
    if missing:
        raise ValueError(f"Unknown node id(s): {missing}")

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1).numpy()

    return {node_id: label_classes[preds[node_id_to_idx[node_id]]] for node_id in node_ids}
