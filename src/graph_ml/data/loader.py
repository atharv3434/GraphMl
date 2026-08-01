"""Load nodes/edges CSVs into a PyTorch Geometric `Data` object with train/val/test masks."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from graph_ml.config import DataConfig

logger = logging.getLogger(__name__)


def _make_masks(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random train/val/test split by label, returned as boolean masks."""
    n = len(labels)
    indices = np.arange(n)

    train_idx, rest_idx = train_test_split(
        indices, train_size=train_ratio, random_state=random_state, stratify=labels
    )
    # val_ratio is expressed relative to the full dataset; rescale relative to `rest`.
    rest_labels = labels[rest_idx]
    val_fraction_of_rest = val_ratio / (1 - train_ratio)
    val_fraction_of_rest = min(max(val_fraction_of_rest, 0.0), 1.0)

    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_fraction_of_rest, random_state=random_state, stratify=rest_labels
    )

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_graph_data(config: DataConfig) -> tuple[Data, dict[str, int], list]:
    """Build a PyG `Data` object plus the node-id <-> index mapping and label class names.

    Returns (data, node_id_to_idx, label_classes) where label_classes[i] gives the
    original label value for encoded class i.
    """
    nodes_df = pd.read_csv(config.nodes_csv)
    edges_df = pd.read_csv(config.edges_csv)

    required_node_cols = [config.node_id_column, config.label_column]
    missing = [c for c in required_node_cols if c not in nodes_df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {config.nodes_csv}. "
            f"Available columns: {list(nodes_df.columns)}"
        )

    if not {"source", "target"}.issubset(edges_df.columns):
        raise ValueError(
            f"Edges CSV must have 'source' and 'target' columns. "
            f"Available columns: {list(edges_df.columns)}"
        )

    nodes_df = nodes_df.dropna(subset=[config.label_column]).reset_index(drop=True)

    node_ids = nodes_df[config.node_id_column].tolist()
    node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    feature_cols = [
        c for c in nodes_df.columns
        if c not in (config.node_id_column, config.label_column, config.split_column)
    ]
    if not feature_cols:
        raise ValueError(
            f"No feature columns found in {config.nodes_csv} — need at least one "
            f"column besides id/label/split."
        )

    x = torch.tensor(nodes_df[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

    label_classes = sorted(nodes_df[config.label_column].unique().tolist(), key=str)
    label_to_idx = {label: i for i, label in enumerate(label_classes)}
    y = torch.tensor(
        [label_to_idx[label] for label in nodes_df[config.label_column]], dtype=torch.long
    )

    # Build edge_index, dropping edges referencing unknown node ids.
    valid_mask = edges_df["source"].isin(node_id_to_idx) & edges_df["target"].isin(node_id_to_idx)
    dropped = (~valid_mask).sum()
    if dropped:
        logger.warning(f"Dropping {dropped} edges referencing unknown node ids")
    edges_df = edges_df[valid_mask]

    src = edges_df["source"].map(node_id_to_idx).to_numpy()
    dst = edges_df["target"].map(node_id_to_idx).to_numpy()

    if config.directed:
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long
        )

    if config.split_column and config.split_column in nodes_df.columns:
        split = nodes_df[config.split_column]
        train_mask = torch.tensor((split == "train").to_numpy())
        val_mask = torch.tensor((split == "val").to_numpy())
        test_mask = torch.tensor((split == "test").to_numpy())
        logger.info("Using explicit split column from nodes CSV")
    else:
        train_mask_np, val_mask_np, test_mask_np = _make_masks(
            y.numpy(), config.train_ratio, config.val_ratio, config.random_state
        )
        train_mask = torch.tensor(train_mask_np)
        val_mask = torch.tensor(val_mask_np)
        test_mask = torch.tensor(test_mask_np)
        logger.info(
            f"Generated random stratified split: train={train_mask.sum()} "
            f"val={val_mask.sum()} test={test_mask.sum()}"
        )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    logger.info(
        f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges, "
        f"{x.shape[1]} features, {len(label_classes)} classes"
    )
    return data, node_id_to_idx, label_classes
