from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


ROOT = Path("/user/csantoro/gnn_thesis")
DATA_ROOT = ROOT / "data" / "network-flows"

DETAIL_OUT = ROOT / "flow_feature_detailed_150000.csv"
FLOW_SUMMARY_OUT = ROOT / "flow_feature_summary_150000.csv"
APT_STAGE_SUMMARY_OUT = ROOT / "apt_stage_feature_summary_150000.csv"
COUNTS_OUT = ROOT / "flow_stage_counts_150000.csv"


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


FEATURE_COLUMNS = [
    "src_port",
    "dst_port",
    "protocol",
    "bidirectional_duration_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_duration_ms",
    "src2dst_packets",
    "src2dst_bytes",
    "dst2src_duration_ms",
    "dst2src_packets",
    "dst2src_bytes",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "src2dst_min_ps",
    "src2dst_mean_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",
    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",
    "bidirectional_min_piat_ms",
    "bidirectional_mean_piat_ms",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "src2dst_min_piat_ms",
    "src2dst_mean_piat_ms",
    "src2dst_stddev_piat_ms",
    "src2dst_max_piat_ms",
    "dst2src_min_piat_ms",
    "dst2src_mean_piat_ms",
    "dst2src_stddev_piat_ms",
    "dst2src_max_piat_ms",
    "bidirectional_syn_packets",
    "bidirectional_cwr_packets",
    "bidirectional_ece_packets",
    "bidirectional_urg_packets",
    "bidirectional_ack_packets",
    "bidirectional_psh_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "src2dst_syn_packets",
    "src2dst_cwr_packets",
    "src2dst_ece_packets",
    "src2dst_urg_packets",
    "src2dst_ack_packets",
    "src2dst_psh_packets",
    "src2dst_rst_packets",
    "src2dst_fin_packets",
    "dst2src_syn_packets",
    "dst2src_cwr_packets",
    "dst2src_ece_packets",
    "dst2src_urg_packets",
    "dst2src_ack_packets",
    "dst2src_psh_packets",
    "dst2src_rst_packets",
    "dst2src_fin_packets",
    "application_is_guessed",
]


META_COLUMNS = [
    "split",
    "day_folder",
    "capture",
    "flow_group",
    "Stage",
    "Activity",
    "Signature",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
]


def assign_split(day_folder):
    if day_folder in TRAIN_DAYS:
        return "train"
    if day_folder in VALIDATION_DAYS:
        return "validation"
    if day_folder in EXCLUDED_DAYS:
        return "excluded"
    return "test"


def normalize_stage(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "Unknown"
    return x


def normalize_signature(x):
    if pd.isna(x):
        return "None"
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return "None"
    return x


def flow_group_from_stage(stage):
    return "benign" if stage == "Benign" else "attack"


def empty_stats():
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": np.inf,
        "max": -np.inf,
    }


def update_stat_dict(stats_dict, key, values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return

    arr = values.to_numpy(dtype=float)
    st = stats_dict[key]
    st["count"] += int(arr.size)
    st["sum"] += float(arr.sum())
    st["sumsq"] += float(np.square(arr).sum())
    st["min"] = min(st["min"], float(arr.min()))
    st["max"] = max(st["max"], float(arr.max()))


def update_feature_stats(stats_dict, chunk, group_cols, feature_cols):
    for group_values, group_df in chunk.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        for feature in feature_cols:
            key = tuple(group_values) + (feature,)
            update_stat_dict(stats_dict, key, group_df[feature])


def stats_to_dataframe(stats_dict, group_cols, out_path):
    rows = []

    for key, st in stats_dict.items():
        *group_values, feature = key
        count = st["count"]

        if count == 0:
            mean = np.nan
            var = np.nan
            std = np.nan
            min_value = np.nan
            max_value = np.nan
        else:
            mean = st["sum"] / count
            if count > 1:
                var = (st["sumsq"] - (st["sum"] ** 2) / count) / (count - 1)
                var = max(var, 0.0)
            else:
                var = 0.0
            std = np.sqrt(var)
            min_value = st["min"]
            max_value = st["max"]

        row = dict(zip(group_cols, group_values))
        row.update({
            "feature_name": feature,
            "count": count,
            "mean": mean,
            "std": std,
            "var": var,
            "min": min_value,
            "max": max_value,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(group_cols + ["feature_name"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def update_counts(counts, chunk):
    group_cols = ["split", "day_folder", "capture", "flow_group", "Stage", "Signature"]
    c = chunk.groupby(group_cols, dropna=False).size().reset_index(name="flows")

    for _, row in c.iterrows():
        key = tuple(row[col] for col in group_cols)
        counts[key] += int(row["flows"])


def counts_to_dataframe(counts, out_path):
    rows = []
    cols = ["split", "day_folder", "capture", "flow_group", "Stage", "Signature"]

    for key, value in counts.items():
        row = dict(zip(cols, key))
        row["flows"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(cols).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    csv_files = sorted(DATA_ROOT.glob("*/*_Flow_labeled.csv"))

    print("CSV files found:", len(csv_files))

    flow_stats = defaultdict(empty_stats)
    apt_stage_stats = defaultdict(empty_stats)
    counts = defaultdict(int)

    first_detail_write = True
    total_rows = 0

    if DETAIL_OUT.exists():
        DETAIL_OUT.unlink()

    for file_idx, csv_path in enumerate(csv_files, start=1):
        day_folder = csv_path.parent.name
        capture = csv_path.stem
        split = assign_split(day_folder)

        print(f"[{file_idx}/{len(csv_files)}] Reading {csv_path}")

        for chunk in pd.read_csv(
            csv_path,
            chunksize=100000,
            low_memory=False,
            on_bad_lines="skip",
        ):
            chunk["split"] = split
            chunk["day_folder"] = day_folder
            chunk["capture"] = capture

            chunk["Stage"] = chunk["Stage"].apply(normalize_stage)
            chunk["Signature"] = chunk["Signature"].apply(normalize_signature)
            chunk["flow_group"] = chunk["Stage"].apply(flow_group_from_stage)

            available_features = [c for c in FEATURE_COLUMNS if c in chunk.columns]

            # Convert selected numeric features.
            for col in available_features:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # Save detailed file with only metadata + selected features.
            detail_cols = []
            for c in META_COLUMNS:
                if c in chunk.columns and c not in detail_cols:
                    detail_cols.append(c)
            for c in available_features:
                if c not in detail_cols:
                    detail_cols.append(c)

            chunk[detail_cols].to_csv(
                DETAIL_OUT,
                mode="a",
                index=False,
                header=first_detail_write,
            )
            first_detail_write = False

            # Summary 1: benign vs attack flow features per split.
            update_feature_stats(
                flow_stats,
                chunk,
                ["split", "flow_group"],
                available_features,
            )

            # Summary 2: features per APT stage.
            update_feature_stats(
                apt_stage_stats,
                chunk,
                ["split", "Stage"],
                available_features,
            )

            # Counts per day/capture/stage/signature.
            update_counts(counts, chunk)

            total_rows += len(chunk)

        print("  total rows processed so far:", total_rows)

    flow_summary = stats_to_dataframe(
        flow_stats,
        ["split", "flow_group"],
        FLOW_SUMMARY_OUT,
    )

    apt_summary = stats_to_dataframe(
        apt_stage_stats,
        ["split", "Stage"],
        APT_STAGE_SUMMARY_OUT,
    )

    counts_df = counts_to_dataframe(counts, COUNTS_OUT)

    print("\nSaved:", DETAIL_OUT)
    print("Saved:", FLOW_SUMMARY_OUT)
    print("Saved:", APT_STAGE_SUMMARY_OUT)
    print("Saved:", COUNTS_OUT)

    print("\nTotal rows processed:", total_rows)

    print("\n=== Flow counts by split and flow_group ===")
    print(
        counts_df.groupby(["split", "flow_group"])["flows"]
        .sum()
        .unstack(fill_value=0)
    )

    print("\n=== Stage counts ===")
    print(
        counts_df.groupby(["split", "Stage"])["flows"]
        .sum()
        .unstack(fill_value=0)
    )

    print("\n=== Flow feature summary preview ===")
    print(flow_summary.head(30).to_string(index=False))

    print("\n=== APT stage feature summary preview ===")
    print(apt_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
