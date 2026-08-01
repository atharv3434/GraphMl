# Graph ML Pipeline

A production-ready node classification pipeline built on PyTorch Geometric:
GCN, GraphSAGE, or GAT, config-driven, with proper train/val/test splitting,
early stopping, and a CLI for training and inference.

## Features

- **Three GNN architectures** — GCN, GraphSAGE, GAT — swap via one config line
- **Config-driven** — all data paths, model architecture, and training hyperparameters live in `configs/default.yaml`
- **Stratified train/val/test splitting** by label, or use your own explicit split column
- **Early stopping** on validation macro-F1, with best-checkpoint restoration
- **Transductive node classification** — the standard setup for citation networks, social graphs, fraud graphs, etc.
- **CLI** for training (`graph-ml train`) and inference (`graph-ml predict`), including predicting for specific node ids
- **Tests** covering data loading/splitting, all three model architectures, and a full end-to-end train→predict run
- **Dockerfile** for containerized deployment

## Project Structure

```
graph-ml-pipeline/
├── configs/
│   └── default.yaml          # all data/model/training configuration
├── data/raw/                  # nodes.csv, edges.csv
├── checkpoints/               # saved model + metrics.json
├── scripts/
│   └── make_sample_data.py   # generates a synthetic graph with real community structure
├── src/graph_ml/
│   ├── config.py              # config dataclasses + YAML loading
│   ├── cli.py                  # `graph-ml` command line entrypoint
│   ├── engine.py                # training + evaluation loop
│   ├── inference.py            # load trained model, predict labels
│   ├── data/loader.py           # CSV -> PyG Data object, splitting
│   └── models/factory.py       # GCN / GraphSAGE / GAT definitions
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Quick Start

```bash
# 1. Install (editable, with dev deps)
pip install -e ".[dev]"

# 2. Generate a synthetic graph with real community structure (a stochastic
#    block model — nodes cluster into communities via both edges and features)
python scripts/make_sample_data.py

# 3. Train + evaluate
graph-ml train --config configs/default.yaml

# 4. Predict labels — for everyone, or specific nodes
graph-ml predict --config configs/default.yaml
graph-ml predict --config configs/default.yaml --node-id 0 --node-id 1
```

## Using Your Own Graph Data

Drop your data into `data/raw/`:

```
data/raw/
├── nodes.csv   # columns: node_id, feature_1..feature_N, label[, split]
└── edges.csv   # columns: source, target
```

- Every non-id/label/split column in `nodes.csv` is treated as a numeric feature.
- If you already have a fixed split (common for benchmark datasets), add a
  `split` column with values `train`/`val`/`test` and the pipeline will use it
  instead of generating a random stratified split.
- Set `data.directed: true` if your edges shouldn't be treated as bidirectional.

## Configuration Reference

```yaml
model:
  type: "graphsage"     # gcn | graphsage | gat
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5
  heads: 4              # only used by GAT

training:
  epochs: 200
  lr: 0.01
  early_stopping_patience: 20
```

## Testing

```bash
pytest tests/ -v --cov=graph_ml
```

Tests cover all three architectures with tiny synthetic graphs, so the full
suite runs quickly on CPU without needing real data.

## Running with Docker

```bash
docker build -t graph-ml .
docker run -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints graph-ml train
docker run -v $(pwd)/checkpoints:/app/checkpoints graph-ml predict --node-id 0
```

## Extending

- **Link prediction / edge-level tasks**: swap the classification head for a
  dot-product or MLP edge decoder over node embedding pairs
- **Inductive setting** (predicting on entirely new graphs/nodes not seen during
  training): GraphSAGE already supports this natively since it aggregates from
  neighbor features rather than a fixed node embedding table — just call
  `model(new_x, new_edge_index)` on a new graph
- **Heterogeneous graphs** (multiple node/edge types): migrate to PyG's
  `HeteroData` and `HeteroConv` wrappers around the existing conv layers
- **Scaling to large graphs**: replace full-batch training with `NeighborLoader`
  mini-batching from `torch_geometric.loader`
