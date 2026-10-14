from pathlib import Path
from collections import defaultdict, Counter
import pickle
import numpy as np
import pandas as pd


ROOT = Path("/user/csantoro/gnn_thesis")
DATA_ROOT = ROOT / "data" / "network-flows"
GRAPH_ROOT = ROOT / "unraveled_graphs_150000_all"
TIMELINE_PATH = ROOT / "graph_timeline_150000_with_split.csv"

OUT_GRAPH_STAGE = ROOT / "attack_graph_stage_mapping_150000.csv"
OUT_NODE_STAGE_SUMMARY = ROOT / "attack_node_stage_feature_summary_150000.csv"
OUT_NODE_STAGE_COUNTS = ROOT / "attack_node_stage_counts_150000.csv"

SNAPSHOT_INTERVAL_MS = 150000


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


ATTACK_STAGES = {
    "Reconnaissance",
    "Establish Foothold",
    "Lateral Movement",
    "Data Exfiltration",
    "Cover up",
}


def normalize_stage(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "Unknown"
    return x


def graph_folder_from_type(graph_type):
    if graph_type == "benign":
        return "full_benign"
    if graph_type == "mixed":
        return "mixed"
    raise ValueError(f"Unknown graph_type: {graph_type}")


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "var": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def compute_graph_stage_mapping():
    """
    For each day/capture/graph_id, compute how many flows belong to each Stage.
    The graph_id is reconstructed using 150000 ms windows.
    """
    rows = []
    csv_files = sorted(DATA_ROOT.glob("*/*_Flow_labeled.csv"))

    print("CSV files found:", len(csv_files))

    for file_idx, csv_path in enumerate(csv_files, start=1):
        day_folder = csv_path.parent.name
        capture = csv_path.stem

        print(f"[{file_idx}/{len(csv_files)}] Stage mapping for {csv_path}")

        usecols = [
            "bidirectional_first_seen_ms",
            "Stage",
        ]

        try:
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
                low_memory=False,
                on_bad_lines="skip",
            )
        except ValueError:
            print("  skipped, required columns missing")
            continue

        if df.empty:
            continue

        df["bidirectional_first_seen_ms"] = pd.to_numeric(
            df["bidirectional_first_seen_ms"],
            errors="coerce",
        )
        df = df.dropna(subset=["bidirectional_first_seen_ms"])

        if df.empty:
            continue

        df["Stage"] = df["Stage"].apply(normalize_stage)

        start_time = df["bidirectional_first_seen_ms"].min()

        df["graph_index"] = (
            (df["bidirectional_first_seen_ms"] - start_time)
            // SNAPSHOT_INTERVAL_MS
        ).astype(int)

        grouped = (
            df.groupby(["graph_index", "Stage"])
            .size()
            .reset_index(name="flows")
        )

        for graph_index, g in grouped.groupby("graph_index"):
            stage_counts = {
                row["Stage"]: int(row["flows"])
                for _, row in g.iterrows()
            }

            attack_stage_counts = {
                stage: count
                for stage, count in stage_counts.items()
                if stage in ATTACK_STAGES and count > 0
            }

            total_flows = sum(stage_counts.values())
            benign_flows = stage_counts.get("Benign", 0)
            attack_flows = sum(attack_stage_counts.values())

            if attack_stage_counts:
                dominant_stage = max(
                    attack_stage_counts.items(),
                    key=lambda x: x[1],
                )[0]
                num_attack_stages = len(attack_stage_counts)
                all_attack_stages = "|".join(sorted(attack_stage_counts.keys()))
            else:
                dominant_stage = "Benign"
                num_attack_stages = 0
                all_attack_stages = ""

            rows.append({
                "day_folder": day_folder,
                "capture": capture,
                "graph_id": f"graph_{int(graph_index)}",
                "graph_index": int(graph_index),
                "total_flows": total_flows,
                "benign_flows": benign_flows,
                "attack_flows": attack_flows,
                "dominant_stage": dominant_stage,
                "num_attack_stages": num_attack_stages,
                "all_attack_stages": all_attack_stages,
                "reconnaissance_flows": stage_counts.get("Reconnaissance", 0),
                "establish_foothold_flows": stage_counts.get("Establish Foothold", 0),
                "lateral_movement_flows": stage_counts.get("Lateral Movement", 0),
                "data_exfiltration_flows": stage_counts.get("Data Exfiltration", 0),
                "cover_up_flows": stage_counts.get("Cover up", 0),
            })

    mapping = pd.DataFrame(rows)
    mapping.to_csv(OUT_GRAPH_STAGE, index=False)

    print("Saved:", OUT_GRAPH_STAGE)
    print("Graph-stage rows:", len(mapping))

    print("\n=== Dominant stage counts ===")
    print(mapping["dominant_stage"].value_counts())

    return mapping


def compute_attack_node_stage_feature_summary(mapping):
    timeline = pd.read_csv(TIMELINE_PATH)

    # We only analyse test mixed graphs for the final evaluation.
    timeline = timeline[
        (timeline["split"] == "test")
        & (timeline["graph_type"] == "mixed")
    ].copy()

    merged = timeline.merge(
        mapping,
        on=["day_folder", "capture", "graph_id", "graph_index"],
        how="left",
    )

    # Keep only graphs for which we found attack-stage information.
    merged = merged[
        merged["dominant_stage"].isin(ATTACK_STAGES)
    ].copy()

    feature_values = defaultdict(list)
    count_rows = []

    total_graphs = len(merged)
    print("\nMixed test graphs with APT stage mapping:", total_graphs)

    for i, row in enumerate(merged.itertuples(index=False), start=1):
        if i % 500 == 0:
            print(f"  processed {i}/{total_graphs} graphs")

        graph_path = (
            GRAPH_ROOT
            / row.day_folder
            / row.capture
            / "etdg_graph"
            / "mixed"
            / f"{row.graph_id}.pkl"
        )

        if not graph_path.exists():
            continue

        g = load_graph(graph_path)

        node_features = g["node_features"]
        node_labels = g["node_labels"]

        if not isinstance(node_features, np.ndarray):
            continue

        if node_features.size == 0 or node_features.ndim != 2:
            continue

        if node_features.shape[1] != len(FEATURE_NAMES):
            continue

        attack_mask = node_labels == 1
        attack_features = node_features[attack_mask]

        if attack_features.shape[0] == 0:
            continue

        dominant_stage = row.dominant_stage

        count_rows.append({
            "day_folder": row.day_folder,
            "capture": row.capture,
            "graph_id": row.graph_id,
            "dominant_stage": dominant_stage,
            "num_attack_nodes": int(attack_features.shape[0]),
            "attack_flows": int(row.attack_flows),
            "num_attack_stages": int(row.num_attack_stages),
            "all_attack_stages": row.all_attack_stages,
        })

        for feature_idx, feature_name in enumerate(FEATURE_NAMES):
            key = (dominant_stage, feature_idx, feature_name)
            feature_values[key].extend(attack_features[:, feature_idx].astype(float).tolist())

    summary_rows = []

    for (stage, feature_idx, feature_name), values in feature_values.items():
        st = safe_stats(values)
        summary_rows.append({
            "Stage": stage,
            "feature_idx": feature_idx,
            "feature_name": feature_name,
            **st,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["Stage", "feature_idx"]).reset_index(drop=True)
    summary.to_csv(OUT_NODE_STAGE_SUMMARY, index=False)

    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT_NODE_STAGE_COUNTS, index=False)

    print("Saved:", OUT_NODE_STAGE_SUMMARY)
    print("Saved:", OUT_NODE_STAGE_COUNTS)

    print("\n=== Attack node counts by dominant stage ===")
    if not counts.empty:
        print(counts.groupby("dominant_stage")["num_attack_nodes"].sum())
    else:
        print("No counts produced.")

    print("\n=== Summary preview ===")
    print(summary.head(40).to_string(index=False))

    return summary, counts


def main():
    mapping = compute_graph_stage_mapping()
    compute_attack_node_stage_feature_summary(mapping)


if __name__ == "__main__":
    main()
