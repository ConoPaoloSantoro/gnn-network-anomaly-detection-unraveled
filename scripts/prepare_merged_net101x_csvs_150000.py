import re
import pandas as pd
from pathlib import Path

RAW_ROOT = Path("data/network-flows")
OUT_ROOT = Path("data/network-flows-merged-net101x")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MANIFEST_OUT = RESULTS_DIR / "merged_net101x_input_manifest_150000.csv"
SUMMARY_OUT = RESULTS_DIR / "merged_net101x_input_summary_150000.csv"

TIME_COL = "bidirectional_first_seen_ms"

# Include only subnet captures net1011x ... net1015x.
# This also includes files with suffix _part.
INCLUDE_PATTERN = re.compile(r"^net101[1-5]x.*_Flow_labeled\.csv$")

# Always exclude gateway captures.
EXCLUDE_PATTERN = re.compile(r"^netgw.*_Flow_labeled\.csv$")


def is_selected_file(path: Path) -> bool:
    name = path.name

    if EXCLUDE_PATTERN.match(name):
        return False

    if INCLUDE_PATTERN.match(name):
        return True

    return False


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception as e1:
        print(f"  [WARN] C engine failed for {path.name}: {e1}")
        print("  [WARN] Retrying with python engine and skipping bad lines")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def stage_counts_to_string(df: pd.DataFrame) -> str:
    if "Stage" not in df.columns:
        return ""

    vc = df["Stage"].fillna("Unknown").astype(str).value_counts()
    return "; ".join(f"{k}:{int(v)}" for k, v in vc.items())


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_rows = []

    day_dirs = sorted([p for p in RAW_ROOT.iterdir() if p.is_dir()])

    for day_dir in day_dirs:
        day_folder = day_dir.name

        selected_files = sorted([
            p for p in day_dir.glob("*_Flow_labeled.csv")
            if is_selected_file(p)
        ])

        if not selected_files:
            print(f"[SKIP] {day_folder}: no selected net101x files")
            continue

        print(f"\n[DAY] {day_folder}")
        print(f"Selected files: {len(selected_files)}")

        dfs = []
        reference_columns = None

        for csv_file in selected_files:
            print(f"  reading {csv_file.name}")

            df = read_csv_safe(csv_file)

            if reference_columns is None:
                reference_columns = list(df.columns)
            else:
                missing_cols = set(reference_columns) - set(df.columns)
                extra_cols = set(df.columns) - set(reference_columns)

                if missing_cols or extra_cols:
                    print(f"  [WARN] column mismatch in {csv_file.name}")
                    print(f"         missing: {sorted(missing_cols)}")
                    print(f"         extra: {sorted(extra_cols)}")

                # Align columns to the first file schema.
                # Extra columns are dropped, missing columns are added as NaN.
                for col in reference_columns:
                    if col not in df.columns:
                        df[col] = pd.NA

                df = df[reference_columns]

            rows_before = len(df)

            if TIME_COL not in df.columns:
                raise RuntimeError(f"Missing {TIME_COL} in {csv_file}")

            df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
            df = df.dropna(subset=[TIME_COL]).copy()
            df[TIME_COL] = df[TIME_COL].astype("int64")

            rows_after = len(df)

            if rows_after == 0:
                print(f"  [WARN] {csv_file.name} has 0 valid timestamp rows after cleaning")
                continue

            manifest_rows.append({
                "day_folder": day_folder,
                "source_file": csv_file.name,
                "source_path": str(csv_file),
                "rows_before_timestamp_cleaning": rows_before,
                "rows_after_timestamp_cleaning": rows_after,
                "min_bidirectional_first_seen_ms": int(df[TIME_COL].min()),
                "max_bidirectional_first_seen_ms": int(df[TIME_COL].max()),
                "stage_counts": stage_counts_to_string(df),
            })

            dfs.append(df)

        if not dfs:
            print(f"[SKIP] {day_folder}: no valid dataframes after timestamp cleaning")
            continue

        merged = pd.concat(dfs, ignore_index=True)

        merged[TIME_COL] = pd.to_numeric(merged[TIME_COL], errors="coerce")
        merged = merged.dropna(subset=[TIME_COL]).copy()
        merged[TIME_COL] = merged[TIME_COL].astype("int64")

        merged = merged.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

        out_day_dir = OUT_ROOT / day_folder
        out_day_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_day_dir / "merged_net101x_Flow_labeled.csv"
        merged.to_csv(out_csv, index=False)

        summary_rows.append({
            "day_folder": day_folder,
            "selected_files": len(selected_files),
            "merged_rows": len(merged),
            "output_csv": str(out_csv),
            "min_bidirectional_first_seen_ms": int(merged[TIME_COL].min()),
            "max_bidirectional_first_seen_ms": int(merged[TIME_COL].max()),
            "stage_counts": stage_counts_to_string(merged),
        })

        print(f"  saved: {out_csv}")
        print(f"  merged rows: {len(merged)}")
        print(f"  min timestamp: {int(merged[TIME_COL].min())}")
        print(f"  max timestamp: {int(merged[TIME_COL].max())}")

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)

    manifest.to_csv(MANIFEST_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nDone.")
    print(f"Saved manifest: {MANIFEST_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved merged CSV root: {OUT_ROOT}")

    print("\n=== Summary preview ===")
    if summary.empty:
        print("No merged CSV generated.")
    else:
        print(summary[[
            "day_folder",
            "selected_files",
            "merged_rows",
            "output_csv",
            "stage_counts",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
