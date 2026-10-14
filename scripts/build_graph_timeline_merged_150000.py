from pathlib import Path
import json
import re
import pandas as pd


ROOT = Path("/user/csantoro/gnn_thesis/unraveled_graphs_150000_merged_captures")
OUT = Path("/user/csantoro/gnn_thesis/graph_timeline_merged_150000.csv")

REPRESENTATION = "tdg_graph"


def extract_sort_keys(day_name):
    """
    Extract approximate chronological keys from folder names such as:
    Week1_Day3_05282021
    Week2_Day5-7_06042021-06062021
    Week6_Day5_07022021_part2
    """
    week_match = re.search(r"Week(\d+)", day_name)
    day_match = re.search(r"Day0?(\d+)", day_name)

    week = int(week_match.group(1)) if week_match else 999
    day = int(day_match.group(1)) if day_match else 999

    part = 0
    part_match = re.search(r"part(\d+)", day_name)
    if part_match:
        part = int(part_match.group(1))

    return week, day, part


def graph_index(graph_id):
    """
    Convert graph_123 into 123.
    """
    m = re.search(r"graph_(\d+)", graph_id)
    return int(m.group(1)) if m else -1


rows = []

for stats_path in sorted(ROOT.glob("*/stats_all_captures.json")):
    day_folder = stats_path.parent.name
    week, day, part = extract_sort_keys(day_folder)

    with open(stats_path, "r") as f:
        data = json.load(f)

    for capture, capture_data in data.items():
        if REPRESENTATION not in capture_data:
            continue

        graphs = capture_data[REPRESENTATION]

        for gid, info in graphs.items():
            t = float(info.get("type"))

            if t == 1.0:
                graph_type = "benign"
            elif t == 2.0:
                graph_type = "mixed"
            else:
                graph_type = "unknown"

            rows.append({
                "day_folder": day_folder,
                "week": week,
                "day": day,
                "part": part,
                "capture": capture,
                "representation": REPRESENTATION,
                "graph_id": gid,
                "graph_index": graph_index(gid),
                "nodes": info.get("nodes"),
                "edges": info.get("edges"),
                "flows": info.get("flows"),
                "type": t,
                "graph_type": graph_type,
            })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["week", "day", "part", "capture", "graph_index"]
).reset_index(drop=True)

df["global_graph_index"] = range(len(df))

df.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Total graphs:", len(df))
print()
print("Graph type counts:")
print(df["graph_type"].value_counts())
print()
print("Days with mixed graphs:")
summary = df.groupby("day_folder")["graph_type"].value_counts().unstack(fill_value=0)

if "mixed" in summary.columns:
    print(summary[summary["mixed"] > 0])
else:
    print("No mixed graphs found.")
