import pandas as pd
from pathlib import Path


IN_PATH = Path("/user/csantoro/gnn_thesis/graph_timeline_150000.csv")
OUT_PATH = Path("/user/csantoro/gnn_thesis/graph_timeline_150000_with_split.csv")
SUMMARY_PATH = Path("/user/csantoro/gnn_thesis/split_summary_150000.csv")
DAY_SUMMARY_PATH = Path("/user/csantoro/gnn_thesis/split_day_summary_150000.csv")


TRAIN_DAYS = {
    "Week1_Day3_05282021",
    "Week1_Day4_05292021",
    "Week1_Day5_05302021",
}

VALIDATION_DAYS = {
    "Week2_Day1_05312021",
    "Week2_Day2_06012021",
    "Week2_Day3_06022021",
}

EXCLUDED_DAYS = {
    "Week1_Day1-2_05262021-05272021",
}


df = pd.read_csv(IN_PATH)


def assign_split(day_folder):
    if day_folder in TRAIN_DAYS:
        return "train"
    if day_folder in VALIDATION_DAYS:
        return "validation"
    if day_folder in EXCLUDED_DAYS:
        return "excluded"
    return "test"


df["split"] = df["day_folder"].apply(assign_split)

# Safety checks: train and validation must be fully benign
for split_name in ["train", "validation"]:
    subset = df[df["split"] == split_name]
    mixed_count = (subset["graph_type"] == "mixed").sum()

    if mixed_count > 0:
        raise ValueError(
            f"{split_name} contains {mixed_count} mixed graphs. "
            "The split is not valid."
        )

df.to_csv(OUT_PATH, index=False)

summary = (
    df.groupby(["split", "graph_type"])
    .agg(
        graphs=("graph_id", "count"),
        avg_nodes=("nodes", "mean"),
        std_nodes=("nodes", "std"),
        var_nodes=("nodes", "var"),
        avg_edges=("edges", "mean"),
        std_edges=("edges", "std"),
        var_edges=("edges", "var"),
        avg_flows=("flows", "mean"),
        std_flows=("flows", "std"),
        var_flows=("flows", "var"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

day_summary = (
    df.groupby(["split", "day_folder", "graph_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

day_summary.to_csv(DAY_SUMMARY_PATH, index=False)

print("Saved:", OUT_PATH)
print("Saved:", SUMMARY_PATH)
print("Saved:", DAY_SUMMARY_PATH)

print("\n=== Split counts ===")
print(df.groupby(["split", "graph_type"]).size().unstack(fill_value=0))

print("\n=== Graph-level statistics ===")
print(summary)

print("\n=== Day-level summary ===")
print(day_summary.to_string(index=False))