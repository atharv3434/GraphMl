"""Tests for the graph ML pipeline: data loading/splitting, models, and end-to-end training."""
import numpy as np
import pandas as pd
import pytest
import torch

from graph_ml.config import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    TrainingConfig,
)
from graph_ml.data.loader import load_graph_data
from graph_ml.engine import train_and_evaluate
from graph_ml.inference import predict_all, predict_nodes
from graph_ml.models.factory import build_model


def _make_synthetic_graph(tmp_path, n_nodes=60, n_features=8, n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    communities = rng.integers(0, n_classes, size=n_nodes)
    community_means = rng.normal(size=(n_classes, n_features)) * 2
    features = community_means[communities] + rng.normal(size=(n_nodes, n_features)) * 0.5

    nodes_rows = []
    for i in range(n_nodes):
        row = {"node_id": i, "label": f"c{communities[i]}"}
        for f in range(n_features):
            row[f"feature_{f}"] = features[i, f]
        nodes_rows.append(row)
    nodes_df = pd.DataFrame(nodes_rows)

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = 0.15 if communities[i] == communities[j] else 0.01
            if rng.random() < p:
                edges.append((i, j))
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    return str(nodes_csv), str(edges_csv)


@pytest.fixture
def data_paths(tmp_path):
    return _make_synthetic_graph(tmp_path)


@pytest.fixture
def config(tmp_path, data_paths):
    nodes_csv, edges_csv = data_paths
    checkpoint_dir = tmp_path / "checkpoints"

    return PipelineConfig(
        data=DataConfig(
            nodes_csv=nodes_csv, edges_csv=edges_csv,
            split_column=None, train_ratio=0.6, val_ratio=0.2, random_state=42,
        ),
        model=ModelConfig(type="graphsage", hidden_dim=16, num_layers=2, dropout=0.3),
        training=TrainingConfig(epochs=30, lr=0.01, early_stopping_patience=30, device="cpu", seed=42),
        output=OutputConfig(
            model_dir=str(checkpoint_dir),
            model_name="model.pt",
            metrics_path=str(checkpoint_dir / "metrics.json"),
        ),
    )


class TestDataLoading:
    def test_builds_correct_shapes(self, data_paths):
        nodes_csv, edges_csv = data_paths
        cfg = DataConfig(nodes_csv=nodes_csv, edges_csv=edges_csv, split_column=None)
        data, node_id_to_idx, label_classes = load_graph_data(cfg)

        assert data.x.shape[0] == len(node_id_to_idx)
        assert data.y.shape[0] == data.x.shape[0]
        assert data.edge_index.shape[0] == 2
        assert len(label_classes) >= 2

    def test_masks_are_disjoint_and_cover_all_nodes(self, data_paths):
        nodes_csv, edges_csv = data_paths
        cfg = DataConfig(
            nodes_csv=nodes_csv, edges_csv=edges_csv,
            split_column=None, train_ratio=0.6, val_ratio=0.2,
        )
        data, _, _ = load_graph_data(cfg)

        overlap = (data.train_mask & data.val_mask).sum() + (data.train_mask & data.test_mask).sum()
        assert overlap == 0
        assert (data.train_mask | data.val_mask | data.test_mask).sum() == data.num_nodes

    def test_undirected_edges_are_symmetric(self, data_paths):
        nodes_csv, edges_csv = data_paths
        cfg = DataConfig(nodes_csv=nodes_csv, edges_csv=edges_csv, split_column=None, directed=False)
        data, _, _ = load_graph_data(cfg)
        assert data.edge_index.shape[1] % 2 == 0

    def test_missing_column_raises(self, tmp_path):
        bad_nodes = tmp_path / "bad_nodes.csv"
        pd.DataFrame({"wrong_col": [1, 2]}).to_csv(bad_nodes, index=False)
        edges_csv = tmp_path / "edges.csv"
        pd.DataFrame({"source": [0], "target": [1]}).to_csv(edges_csv, index=False)

        cfg = DataConfig(nodes_csv=str(bad_nodes), edges_csv=str(edges_csv))
        with pytest.raises(ValueError):
            load_graph_data(cfg)


class TestModelFactory:
    @pytest.mark.parametrize("model_type", ["gcn", "graphsage", "gat"])
    def test_builds_and_forward_pass_works(self, model_type):
        config = ModelConfig(type=model_type, hidden_dim=8, num_layers=2, dropout=0.1, heads=2)
        model = build_model(config, in_channels=5, out_channels=3)

        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 20))
        out = model(x, edge_index)
        assert out.shape == (10, 3)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            build_model(ModelConfig(type="not_a_model"), in_channels=5, out_channels=3)


class TestEndToEnd:
    def test_train_produces_metrics_and_checkpoint(self, config):
        metrics = train_and_evaluate(config)

        assert "test_metrics" in metrics
        assert 0.0 <= metrics["test_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= metrics["best_val_f1_macro"] <= 1.0

        import os
        model_path = os.path.join(config.output.model_dir, config.output.model_name)
        assert os.path.exists(model_path)
        assert os.path.exists(config.output.metrics_path)

    def test_predict_all_covers_every_node(self, config):
        train_and_evaluate(config)
        predictions = predict_all(config)

        import pandas as pd
        nodes_df = pd.read_csv(config.data.nodes_csv)
        assert len(predictions) == len(nodes_df)

    def test_predict_specific_nodes(self, config):
        train_and_evaluate(config)
        predictions = predict_nodes(config, [0, 1, 2])
        assert set(predictions.keys()) == {0, 1, 2}

    def test_predict_unknown_node_raises(self, config):
        train_and_evaluate(config)
        with pytest.raises(ValueError):
            predict_nodes(config, [999999])
