from pathlib import Path
import pickle
import numpy as np
import pandas as pd


ROOT = Path("/user/csantoro/gnn_thesis/unraveled_graphs_150000_all")
TIMELINE_PATH = Path("/user/csantoro/gnn_thesis/graph_timeline_150000_with_split.csv")

OUT_DETAIL = Path("/user/csantoro/gnn_thesis/node_feature_detailed_150000.csv")
OUT_SUMMARY = Path("/user/csantoro/gnn_thesis/node_feature_summary_150000.csv")


FEATURE_NAMES = [
    "graph_degree_centrality",
    "graph_betweenness_centrality",
    "graph_closeness_centrality",
    "cc_degree_centrality",
    "cc_betweenness_centrality",
    "cc_closeness_centrality",
    "second_order_centrality",
    "edge_betweenness_centrality",
    "eccentricity",
    "barycenter",
    "radius",
    "center",
    "component_size",
    "degree",
    "degree_two_hops",
]


def graph_folder_from_type(graph_type):
    if graph_type == "benign":
        return "full_benign"
    if graph_type == "mixed":
        return "mixed"
    raise ValueError(f"Unknown graph_type: {graph_type}")


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def node_group(split, graph_type, label):
    """
    label: 0 = benign node, 1 = attack node
    """
    if split == "excluded":
        return "excluded"

    if graph_type == "benign":
        return f"{split}_benign_graph_nodes"

    if graph_type == "mixed" and label == 0:
        return f"{split}_mixed_benign_nodes"

    if graph_type == "mixed" and label == 1:
        return f"{split}_mixed_attack_nodes"

    return "unknown"


timeline = pd.read_csv(TIMELINE_PATH)

# We analyse e-TDG only, because TDG has empty node_features.
rows = []
missing_files = []
processed_graphs = 0

for _, row in timeline.iterrows():
    split = row["split"]
    day_folder = row["day_folder"]
    capture = row["capture"]
    graph_id = row["graph_id"]
    graph_type = row["graph_type"]

    graph_path = (
        ROOT
        / day_folder
        / capture
        / "etdg_graph"
        / graph_folder_from_type(graph_type)
        / f"{graph_id}.pkl"
    )

    if not graph_path.exists():
        missing_files.append(str(graph_path))
        continue

    g = load_graph(graph_path)

    node_features = g["node_features"]
    node_labels = g["node_labels"]

    if not isinstance(node_features, np.ndarray) or node_features.size == 0:
        continue

    if node_features.ndim != 2:
        continue

    if node_features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected feature dimension in {graph_path}: "
            f"{node_features.shape[1]} instead of {len(FEATURE_NAMES)}"
        )

    processed_graphs += 1

    for node_idx in range(node_features.shape[0]):
        label = int(node_labels[node_idx])
        group = node_group(split, graph_type, label)

        for feature_idx, feature_name in enumerate(FEATURE_NAMES):
            rows.append({
                "split": split,
                "day_folder": day_folder,
                "capture": capture,
                "graph_type": graph_type,
                "graph_id": graph_id,
                "node_idx": node_idx,
                "node_label": label,
                "node_group": group,
                "feature_idx": feature_idx,
                "feature_name": feature_name,
                "value": float(node_features[node_idx, feature_idx]),
            })


detail = pd.DataFrame(rows)
detail.to_csv(OUT_DETAIL, index=False)

summary = (
    detail
    .groupby(["split", "graph_type", "node_group", "feature_idx", "feature_name"])
    .agg(
        count=("value", "count"),
        mean=("value", "mean"),
        std=("value", "std"),
        var=("value", "var"),
        min=("value", "min"),
        median=("value", "median"),
        max=("value", "max"),
    )
    .reset_index()
)

summary.to_csv(OUT_SUMMARY, index=False)

print("Saved:", OUT_DETAIL)
print("Saved:", OUT_SUMMARY)

print("\nProcessed e-TDG graphs:", processed_graphs)
print("Missing files:", len(missing_files))

print("\n=== Node groups ===")
print(detail.groupby(["split", "graph_type", "node_group"])["node_idx"].count())

print("\n=== Summary preview ===")
print(summary.head(40).to_string(index=False))
