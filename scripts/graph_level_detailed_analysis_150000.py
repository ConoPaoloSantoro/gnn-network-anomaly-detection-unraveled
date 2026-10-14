from pathlib import Path
import pickle
import numpy as np
import pandas as pd


ROOT = Path("/user/csantoro/gnn_thesis/unraveled_graphs_150000_all")
TIMELINE_PATH = Path("/user/csantoro/gnn_thesis/graph_timeline_150000_with_split.csv")

DETAIL_OUT = Path("/user/csantoro/gnn_thesis/graph_level_detailed_150000.csv")
SUMMARY_OUT = Path("/user/csantoro/gnn_thesis/graph_level_summary_150000.csv")

REPRESENTATIONS = ["tdg_graph", "etdg_graph"]


def graph_folder_from_type(graph_type):
    if graph_type == "benign":
        return "full_benign"
    if graph_type == "mixed":
        return "mixed"
    raise ValueError(f"Unknown graph_type: {graph_type}")


def safe_density_undirected(num_nodes, num_edges_undirected):
    if num_nodes <= 1:
        return 0.0
    return num_edges_undirected / (num_nodes * (num_nodes - 1) / 2)


def safe_avg_degree(num_nodes, num_edges_undirected):
    if num_nodes == 0:
        return 0.0
    return (2 * num_edges_undirected) / num_nodes


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


df = pd.read_csv(TIMELINE_PATH)

rows = []
missing_files = []

for _, row in df.iterrows():
    day_folder = row["day_folder"]
    capture = row["capture"]
    graph_id = row["graph_id"]
    graph_type = row["graph_type"]
    split = row["split"]

    graph_subfolder = graph_folder_from_type(graph_type)

    for representation in REPRESENTATIONS:
        graph_path = (
            ROOT
            / day_folder
            / capture
            / representation
            / graph_subfolder
            / f"{graph_id}.pkl"
        )

        if not graph_path.exists():
            missing_files.append(str(graph_path))
            continue

        g = load_graph(graph_path)

        adj = g["adj"]
        node_labels = g["node_labels"]
        edge_list = g["edge_list"]
        node_features = g["node_features"]

        num_nodes = int(adj.shape[0])

        # The adjacency matrix is symmetric, so this is the undirected edge count.
        num_edges_undirected = int(np.sum(adj) / 2)

        # The edge_list usually stores both directions, so this is useful for checking.
        if isinstance(edge_list, tuple) and len(edge_list) >= 2:
            edge_list_size = int(len(edge_list[0]))
        else:
            edge_list_size = np.nan

        num_attack_nodes = int(np.sum(node_labels == 1))
        num_benign_nodes = int(np.sum(node_labels == 0))

        attack_node_ratio = (
            num_attack_nodes / num_nodes if num_nodes > 0 else 0.0
        )

        density = safe_density_undirected(num_nodes, num_edges_undirected)
        avg_degree = safe_avg_degree(num_nodes, num_edges_undirected)

        if isinstance(node_features, np.ndarray):
            if node_features.ndim == 2:
                node_feature_dim = int(node_features.shape[1])
            elif node_features.size == 0:
                node_feature_dim = 0
            else:
                node_feature_dim = 1
        else:
            node_feature_dim = 0

        rows.append({
            "split": split,
            "day_folder": day_folder,
            "capture": capture,
            "representation": representation,
            "graph_id": graph_id,
            "graph_type": graph_type,
            "num_nodes": num_nodes,
            "num_edges_undirected": num_edges_undirected,
            "edge_list_size": edge_list_size,
            "density": density,
            "avg_degree": avg_degree,
            "num_benign_nodes": num_benign_nodes,
            "num_attack_nodes": num_attack_nodes,
            "attack_node_ratio": attack_node_ratio,
            "node_feature_dim": node_feature_dim,
            "graph_path": str(graph_path),
        })


detail = pd.DataFrame(rows)
detail.to_csv(DETAIL_OUT, index=False)

numeric_cols = [
    "num_nodes",
    "num_edges_undirected",
    "edge_list_size",
    "density",
    "avg_degree",
    "num_benign_nodes",
    "num_attack_nodes",
    "attack_node_ratio",
    "node_feature_dim",
]

summary = (
    detail
    .groupby(["split", "representation", "graph_type"])[numeric_cols]
    .agg(["count", "mean", "std", "var", "min", "median", "max"])
)

summary.columns = ["_".join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()
summary.to_csv(SUMMARY_OUT, index=False)

print("Saved:", DETAIL_OUT)
print("Saved:", SUMMARY_OUT)

print("\n=== Loaded graphs ===")
print(len(detail))

print("\n=== Missing files ===")
print(len(missing_files))
if missing_files:
    print("First missing files:")
    for p in missing_files[:20]:
        print(p)

print("\n=== Counts by split / representation / graph_type ===")
print(
    detail
    .groupby(["split", "representation", "graph_type"])
    .size()
    .unstack(fill_value=0)
)

print("\n=== Summary preview ===")
print(summary.head(20).to_string(index=False))
