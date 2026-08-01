"""Generate a synthetic graph with real community structure for node classification.

Nodes are assigned to one of a few "communities" (the label to predict). Edges
are far more likely within a community than across communities (a stochastic
block model), and node features are noisy signals of community membership —
so a GNN genuinely has structural + feature signal to learn from, unlike pure
random noise.
"""
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]

N_NODES = 500
N_COMMUNITIES = 4
N_FEATURES = 16
P_IN = 0.05     # edge probability within the same community
P_OUT = 0.002   # edge probability across communities


def generate() -> None:
    communities = np.random.randint(0, N_COMMUNITIES, size=N_NODES)

    # Node features: a community-specific mean vector + noise
    community_means = np.random.randn(N_COMMUNITIES, N_FEATURES) * 2
    features = community_means[communities] + np.random.randn(N_NODES, N_FEATURES) * 0.8

    nodes_rows = []
    for i in range(N_NODES):
        row = {"node_id": i, "label": f"community_{communities[i]}"}
        for f in range(N_FEATURES):
            row[f"feature_{f}"] = features[i, f]
        nodes_rows.append(row)
    nodes_df = pd.DataFrame(nodes_rows)

    edges = []
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            p = P_IN if communities[i] == communities[j] else P_OUT
            if np.random.random() < p:
                edges.append((i, j))
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)
    edges_df.to_csv(out_dir / "edges.csv", index=False)

    print(f"Wrote {len(nodes_df)} nodes to {out_dir / 'nodes.csv'}")
    print(f"Wrote {len(edges_df)} edges to {out_dir / 'edges.csv'}")
    print(nodes_df["label"].value_counts())


if __name__ == "__main__":
    generate()
