from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path.cwd()

TIMELINE_PATH = ROOT / "graph_timeline_merged_150000.csv"
GRAPH_LEVEL_PATH = ROOT / "graph_level_detailed_merged_150000.csv"
DATA_ROOT = ROOT / "data" / "network-flows-merged-net101x"

OUT_TIMELINE = ROOT / "graph_timeline_merged_150000_final_split.csv"
OUT_SPLIT_SUMMARY = ROOT / "split_summary_merged_150000_final.csv"
OUT_DAY_SUMMARY = ROOT / "split_day_summary_merged_150000_final.csv"
OUT_STRUCTURAL = ROOT / "structural_graph_stats_merged_150000_final.csv"
OUT_STRUCTURAL_COMBINED_TEST = ROOT / "structural_graph_stats_merged_150000_final_test_combined.csv"
OUT_EXCEL = ROOT / "temporal_split_merged_net101x_150000ms_final_stats.xlsx"

SNAPSHOT_INTERVAL = 150000

TRAIN_DAYS = {
    "Week1_Day3_05282021",
    "Week1_Day4_05292021",
    "Week1_Day5_05302021",
    "Week2_Day1_05312021",
    "Week2_Day2_06012021",
    "Week2_Day3_06022021",
    "Week2_Day4_06032021",
}

SPECIAL_DAY = "Week2_Day5-7_06042021-06062021"

VALIDATION_DAYS = {
    "Week3_Day1_06072021",
}

TEST_WEEK5_DAYS = {
    "Week5_Day1_06212021",
    "Week5_Day2_06222021",
    "Week5_Day3_06232021",
    "Week5_Day4_06242021",
    "Week5_Day5-6_06252021-06262021",
    "Week5_Day6_06272021",
}

TEST_WEEK6_DAYS = {
    "Week6_Day2_06292021",
    "Week6_Day3_06302021",
    "Week6_Day4_07012021",
    "Week6_Day5_07022021_part2",
    "Week6_Day6-7_07032021-07042021",
}

MOVE_DATES_TO_TRAIN = {"2021-06-04", "2021-06-05"}


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Missing required column. Tried: {candidates}. Available columns: {list(df.columns)}")
    return None


def extract_last_int(x):
    nums = re.findall(r"\d+", str(x))
    return int(nums[-1]) if nums else None


def build_special_day_snapshot_split():
    csv_path = DATA_ROOT / SPECIAL_DAY / "merged_net101x_Flow_labeled.csv"
    flows = pd.read_csv(csv_path)

    ts_col = "bidirectional_first_seen_ms"
    if ts_col not in flows.columns:
        raise RuntimeError(f"Column {ts_col} not found in {csv_path}")

    flows = flows.dropna(subset=[ts_col]).copy()
    flows[ts_col] = pd.to_numeric(flows[ts_col], errors="coerce")
    flows = flows.dropna(subset=[ts_col]).copy()

    min_ts = flows[ts_col].min()
    flows["snapshot_id"] = ((flows[ts_col] - min_ts) // SNAPSHOT_INTERVAL).astype(int)

    snap = (
        flows.groupby("snapshot_id")[ts_col]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "window_first_seen_ms", "max": "window_last_seen_ms", "count": "flows"})
    )

    snap["window_start_date"] = pd.to_datetime(
        snap["window_first_seen_ms"], unit="ms", errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    snap["split_special"] = np.where(
        snap["window_start_date"].isin(MOVE_DATES_TO_TRAIN),
        "train",
        "validation",
    )

    print("\n=== Special folder split by snapshot start date ===")
    print(
        snap.groupby(["window_start_date", "split_special"])["snapshot_id"]
        .nunique()
        .reset_index(name="graph_snapshots")
        .to_string(index=False)
    )

    return snap[["snapshot_id", "window_start_date", "split_special"]]


def assign_final_split(timeline):
    day_col = find_col(timeline, ["day_folder"])
    graph_col = find_col(timeline, ["graph_id"])

    timeline = timeline.copy()
    timeline["split"] = "unused"
    timeline["split_day_detail"] = timeline[day_col]

    timeline.loc[timeline[day_col].isin(TRAIN_DAYS), "split"] = "train"
    timeline.loc[timeline[day_col].isin(VALIDATION_DAYS), "split"] = "validation"
    timeline.loc[timeline[day_col].isin(TEST_WEEK5_DAYS), "split"] = "test_week5"
    timeline.loc[timeline[day_col].isin(TEST_WEEK6_DAYS), "split"] = "test_week6"

    snap_map = build_special_day_snapshot_split()

    mask = timeline[day_col].eq(SPECIAL_DAY)
    special = timeline.loc[mask].copy()

    if "snapshot_id" in special.columns:
        special["snapshot_id_for_merge"] = pd.to_numeric(special["snapshot_id"], errors="coerce").astype("Int64")
    elif special[graph_col].astype(str).str.fullmatch(r"\d+").all():
        special["snapshot_id_for_merge"] = special[graph_col].astype(int)
    else:
        special["snapshot_id_for_merge"] = range(len(special))

    snap_map = snap_map.rename(columns={"snapshot_id": "snapshot_id_for_merge"})
    special = special.merge(snap_map, on="snapshot_id_for_merge", how="left")

    if special["split_special"].isna().any():
        missing = special[special["split_special"].isna()].head()
        raise RuntimeError(f"Some special-folder snapshots were not mapped correctly:\n{missing}")

    timeline.loc[mask, "split"] = special["split_special"].values
    timeline.loc[mask, "split_day_detail"] = special["window_start_date"].values

    return timeline


def make_split_summaries(timeline):
    type_col = find_col(timeline, ["graph_type", "label_type", "category"])

    selected = timeline[timeline["split"].isin(["train", "validation", "test_week5", "test_week6"])].copy()

    summary = (
        selected.pivot_table(
            index="split",
            columns=type_col,
            values="graph_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["benign", "mixed"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["total_graph_snapshots"] = summary["benign"] + summary["mixed"]
    summary["tdg_pkl"] = summary["total_graph_snapshots"]
    summary["etdg_pkl"] = summary["total_graph_snapshots"]
    summary["total_pkl"] = summary["total_graph_snapshots"] * 2

    order = ["train", "validation", "test_week5", "test_week6"]
    summary["split"] = pd.Categorical(summary["split"], categories=order, ordered=True)
    summary = summary.sort_values("split")

    day_summary = (
        selected.pivot_table(
            index=["split", "split_day_detail"],
            columns=type_col,
            values="graph_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"split_day_detail": "day_or_block"})
    )

    for col in ["benign", "mixed"]:
        if col not in day_summary.columns:
            day_summary[col] = 0

    day_summary["total_graph_snapshots"] = day_summary["benign"] + day_summary["mixed"]
    day_summary["split"] = pd.Categorical(day_summary["split"], categories=order, ordered=True)
    day_summary = day_summary.sort_values(["split", "day_or_block"])

    return selected, summary, day_summary


def detect_metric_col(df, possible_names, contains_any=None):
    for c in possible_names:
        if c in df.columns:
            return c
    if contains_any:
        lowered = {c.lower(): c for c in df.columns}
        for low, orig in lowered.items():
            if all(term in low for term in contains_any):
                return orig
    return None


def compute_structural_stats(timeline_final):
    if not GRAPH_LEVEL_PATH.exists():
        print("\nWARNING: graph_level_detailed_merged_150000.csv not found. Structural stats skipped.")
        return pd.DataFrame(), pd.DataFrame()

    g = pd.read_csv(GRAPH_LEVEL_PATH)

    print("\n=== graph_level_detailed columns ===")
    print(list(g.columns))

    if "split" in g.columns:
        g = g.rename(columns={"split": "old_split"})

    merge_keys = [c for c in ["day_folder", "graph_id"] if c in g.columns and c in timeline_final.columns]
    if len(merge_keys) < 2:
        raise RuntimeError(
            f"Cannot merge graph-level stats with final split. "
            f"Common keys found: {merge_keys}. "
            f"Graph-level columns: {list(g.columns)}"
        )

    type_col_tl = find_col(timeline_final, ["graph_type", "label_type", "category"])
    split_map = timeline_final[merge_keys + ["split", type_col_tl]].copy()
    split_map = split_map.rename(columns={type_col_tl: "graph_label_final"})

    g = g.merge(split_map, on=merge_keys, how="left")
    g = g[g["split"].isin(["train", "validation", "test_week5", "test_week6"])].copy()

    repr_col = detect_metric_col(
        g,
        ["graph_repre", "graph_repr", "graph_representation", "representation", "repr"],
    )

    if repr_col:
        print("\n=== Representation values in graph-level file ===")
        print(g[repr_col].value_counts(dropna=False).to_string())

        vals = g[repr_col].astype(str).str.lower()
        tdg_mask = vals.eq("tdg") | ((vals.str.contains("tdg")) & (~vals.str.contains("etdg|e-tdg|e_tdg")))
        if tdg_mask.any():
            g_used = g[tdg_mask].copy()
            used_repr = "TDG"
        else:
            g_used = g.copy()
            used_repr = "all"
    else:
        g_used = g.copy()
        used_repr = "all"

    print(f"\n=== Structural statistics computed on representation: {used_repr} ===")

    node_col = detect_metric_col(g_used, ["num_nodes", "n_nodes", "nodes", "node_count"], ["node"])
    edge_col = detect_metric_col(g_used, ["num_edges", "n_edges", "edges", "edge_count"], ["edge"])
    flow_col = detect_metric_col(g_used, ["num_flows", "n_flows", "flows", "flow_count", "edge_list_size"], None)
    degree_col = detect_metric_col(g_used, ["avg_degree", "average_degree", "mean_degree"], ["degree"])

    needed = {"Avg nodes": node_col, "Avg edges": edge_col, "Avg flows": flow_col}
    for label, col in needed.items():
        if col is None:
            raise RuntimeError(f"Cannot find column for {label}. Available columns: {list(g_used.columns)}")

    g_used["graph_key"] = g_used["day_folder"].astype(str) + "|" + g_used["graph_id"].astype(str)

    group_map = {
        "train": "Train",
        "validation": "Validation",
        "test_week5": "Test Week5",
        "test_week6": "Test Week6",
    }

    g_used["Group"] = g_used["split"].map(group_map) + " " + g_used["graph_label_final"].astype(str)

    agg_dict = {
        "Graphs": ("graph_key", "nunique"),
        "Avg nodes": (node_col, "mean"),
        "Avg edges": (edge_col, "mean"),
        "Avg flows": (flow_col, "mean"),
    }

    if degree_col:
        agg_dict["Avg degree"] = (degree_col, "mean")

    structural = g_used.groupby(["split", "Group"]).agg(**agg_dict).reset_index()

    order = {
        "Train benign": 1,
        "Validation benign": 2,
        "Test Week5 benign": 3,
        "Test Week5 mixed": 4,
        "Test Week6 benign": 5,
        "Test Week6 mixed": 6,
    }
    structural["order"] = structural["Group"].map(order).fillna(99)
    structural = structural.sort_values("order").drop(columns=["order"])

    test = g_used[g_used["split"].isin(["test_week5", "test_week6"])].copy()
    test["Group"] = "Test " + test["graph_label_final"].astype(str)

    structural_combined_test = test.groupby("Group").agg(**agg_dict).reset_index()
    structural_combined_test["order"] = structural_combined_test["Group"].map({"Test benign": 1, "Test mixed": 2}).fillna(99)
    structural_combined_test = structural_combined_test.sort_values("order").drop(columns=["order"])

    return structural, structural_combined_test


def write_excel(split_summary, day_summary, structural, structural_combined_test):
    methodology = pd.DataFrame({
        "Item": [
            "Input graphs",
            "Included CSV captures",
            "Excluded capture",
            "Snapshot interval",
            "Counting rule",
            "Final train update",
            "Number format",
        ],
        "Description": [
            "Graph snapshots generated from merged daily CSV files",
            "net1011x, net1012x, net1013x, net1014x, net1015x",
            "netgw",
            "150000 ms",
            "Each graph snapshot is counted once. TDG and e-TDG .pkl files are counted separately only in the .pkl columns.",
            "Snapshots with window start date 2021-06-04 or 2021-06-05 are assigned to training.",
            "Comma is used as thousands separator and dot is used as decimal separator.",
        ]
    })

    checks = pd.DataFrame({
        "Check": [
            "Training benign-only",
            "Validation benign-only",
            "Test Week5 separated",
            "Test Week6 separated",
            "No unused rows in exported split summary",
        ],
        "Status": [
            "OK" if int(split_summary.loc[split_summary["split"].eq("train"), "mixed"].iloc[0]) == 0 else "CHECK",
            "OK" if int(split_summary.loc[split_summary["split"].eq("validation"), "mixed"].iloc[0]) == 0 else "CHECK",
            "OK" if "test_week5" in set(split_summary["split"].astype(str)) else "CHECK",
            "OK" if "test_week6" in set(split_summary["split"].astype(str)) else "CHECK",
            "OK" if "unused" not in set(split_summary["split"].astype(str)) else "CHECK",
        ]
    })

    with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as writer:
        split_summary.to_excel(writer, sheet_name="Split summary", index=False)
        day_summary.to_excel(writer, sheet_name="Day details", index=False)
        structural.to_excel(writer, sheet_name="Structural summary", index=False)
        structural_combined_test.to_excel(writer, sheet_name="Test combined summary", index=False)
        checks.to_excel(writer, sheet_name="Checks", index=False)
        methodology.to_excel(writer, sheet_name="Methodology", index=False)

        workbook = writer.book

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F2A4D",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })
        int_fmt = workbook.add_format({"num_format": "#,##0"})
        dec_fmt = workbook.add_format({"num_format": "#,##0.0"})
        text_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        for sheet_name, df in [
            ("Split summary", split_summary),
            ("Day details", day_summary),
            ("Structural summary", structural),
            ("Test combined summary", structural_combined_test),
            ("Checks", checks),
            ("Methodology", methodology),
        ]:
            ws = writer.sheets[sheet_name]
            for col_num, value in enumerate(df.columns):
                ws.write(0, col_num, value, header_fmt)

            for idx, col in enumerate(df.columns):
                width = max(14, min(36, max(len(str(col)), *(len(str(x)) for x in df[col].head(100))) + 2))
                ws.set_column(idx, idx, width, text_fmt)

                if col in ["benign", "mixed", "total_graph_snapshots", "tdg_pkl", "etdg_pkl", "total_pkl", "Graphs"]:
                    ws.set_column(idx, idx, 18, int_fmt)

                if col.startswith("Avg"):
                    ws.set_column(idx, idx, 16, dec_fmt)

            ws.freeze_panes(1, 0)

    print(f"\nSaved Excel: {OUT_EXCEL}")


def main():
    timeline = pd.read_csv(TIMELINE_PATH)

    print("=== Timeline columns ===")
    print(list(timeline.columns))

    timeline_final = assign_final_split(timeline)
    selected, split_summary, day_summary = make_split_summaries(timeline_final)

    timeline_final.to_csv(OUT_TIMELINE, index=False)
    split_summary.to_csv(OUT_SPLIT_SUMMARY, index=False)
    day_summary.to_csv(OUT_DAY_SUMMARY, index=False)

    structural, structural_combined_test = compute_structural_stats(timeline_final)
    structural.to_csv(OUT_STRUCTURAL, index=False)
    structural_combined_test.to_csv(OUT_STRUCTURAL_COMBINED_TEST, index=False)

    write_excel(split_summary, day_summary, structural, structural_combined_test)

    print("\n=== Final split summary ===")
    print(split_summary.to_string(index=False))

    print("\n=== Final day summary ===")
    print(day_summary.to_string(index=False))

    print("\n=== Structural summary ===")
    print(structural.to_string(index=False))

    print("\n=== Test combined structural summary ===")
    print(structural_combined_test.to_string(index=False))

    print("\nGenerated files:")
    print(OUT_TIMELINE)
    print(OUT_SPLIT_SUMMARY)
    print(OUT_DAY_SUMMARY)
    print(OUT_STRUCTURAL)
    print(OUT_STRUCTURAL_COMBINED_TEST)
    print(OUT_EXCEL)


if __name__ == "__main__":
    main()
