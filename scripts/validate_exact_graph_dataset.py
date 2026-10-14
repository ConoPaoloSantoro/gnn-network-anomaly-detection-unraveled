from pathlib import Path
from collections import Counter
import json
import pickle
import time

import numpy as np
import pandas as pd


PROJECT = Path.home() / "gnn_thesis"
ROOT = PROJECT / "unraveled_graphs_150000_merged_captures_exact_57_72"
TIMELINE = PROJECT / "graph_timeline_merged_150000_final_split.csv"

REPORT_DIR = PROJECT / "validation_exact_57_72"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FAILURES_CSV = REPORT_DIR / "failures.csv"
SUMMARY_JSON = REPORT_DIR / "summary.json"

TYPE_TO_FOLDER = {
    "benign": "full_benign",
    "full_benign": "full_benign",
    "mixed": "mixed",
    "malicious": "full_malicious",
    "full_malicious": "full_malicious",
}


def clean(v):
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


def paths_from_row(row):
    day = clean(row["day_folder"])
    capture = clean(row["capture"])
    graph_id = clean(row["graph_id"])
    cls = TYPE_TO_FOLDER[str(row["graph_type"])]

    base = ROOT / day / capture

    tdg = base / "tdg_graph" / cls / f"{graph_id}.pkl"
    etdg = base / "etdg_graph" / cls / f"{graph_id}.pkl"

    return tdg, etdg


def add_failure(failures, row, reason, details=""):
    failures.append({
        "global_graph_index": clean(row["global_graph_index"]),
        "split": str(row["split"]),
        "graph_type": str(row["graph_type"]),
        "day_folder": clean(row["day_folder"]),
        "capture": clean(row["capture"]),
        "graph_id": clean(row["graph_id"]),
        "reason": reason,
        "details": details,
    })


print("=" * 90)
print("VALIDAZIONE COMPLETA DATASET TDG 57 / e-TDG 72")
print("=" * 90)
print("Root:", ROOT)
print("Timeline:", TIMELINE)

if not ROOT.is_dir():
    raise FileNotFoundError(ROOT)

if not TIMELINE.is_file():
    raise FileNotFoundError(TIMELINE)

timeline = pd.read_csv(TIMELINE)

print("\nSnapshot timeline:", len(timeline))
print("\nSplit / tipo:")
print(pd.crosstab(timeline["split"], timeline["graph_type"]).to_string())

tdg_files = list(ROOT.glob("*/*/tdg_graph/*/*.pkl"))
etdg_files = list(ROOT.glob("*/*/etdg_graph/*/*.pkl"))

print("\nConteggi filesystem:")
print("TDG:", len(tdg_files))
print("e-TDG:", len(etdg_files))

failures = []
checked = 0

feature_dims_tdg = Counter()
feature_dims_etdg = Counter()
split_counts = Counter()

start_time = time.time()

for i, (_, row) in enumerate(timeline.iterrows(), start=1):
    tdg_path, etdg_path = paths_from_row(row)

    if not tdg_path.is_file():
        add_failure(failures, row, "missing_tdg", str(tdg_path))
        continue

    if not etdg_path.is_file():
        add_failure(failures, row, "missing_etdg", str(etdg_path))
        continue

    try:
        with tdg_path.open("rb") as f:
            tdg = pickle.load(f)

        with etdg_path.open("rb") as f:
            etdg = pickle.load(f)
    except Exception as e:
        add_failure(failures, row, "pickle_read_error", repr(e))
        continue

    try:
        x_tdg = np.asarray(tdg["node_features"])
        x_etdg = np.asarray(etdg["node_features"])

        a_tdg = np.asarray(tdg["adj"])
        a_etdg = np.asarray(etdg["adj"])

        y_tdg = np.asarray(tdg["node_labels"])
        y_etdg = np.asarray(etdg["node_labels"])

        n = a_tdg.shape[0]

        feature_dims_tdg[x_tdg.shape[1] if x_tdg.ndim == 2 else -1] += 1
        feature_dims_etdg[x_etdg.shape[1] if x_etdg.ndim == 2 else -1] += 1

        # Dimensioni generali
        if a_tdg.shape != (n, n):
            add_failure(failures, row, "invalid_tdg_adj_shape", str(a_tdg.shape))

        if a_etdg.shape != (n, n):
            add_failure(failures, row, "invalid_etdg_adj_shape", str(a_etdg.shape))

        if x_tdg.shape != (n, 57):
            add_failure(failures, row, "invalid_tdg_feature_shape", str(x_tdg.shape))

        if x_etdg.shape != (n, 72):
            add_failure(failures, row, "invalid_etdg_feature_shape", str(x_etdg.shape))

        if y_tdg.shape != (n,):
            add_failure(failures, row, "invalid_tdg_label_shape", str(y_tdg.shape))

        if y_etdg.shape != (n,):
            add_failure(failures, row, "invalid_etdg_label_shape", str(y_etdg.shape))

        # TDG / e-TDG devono rappresentare lo stesso grafo
        if not np.array_equal(a_tdg, a_etdg):
            add_failure(failures, row, "adjacency_mismatch")

        if not np.array_equal(y_tdg, y_etdg):
            add_failure(failures, row, "label_mismatch")

        if tdg.get("node_id_to_port_ip") != etdg.get("node_id_to_port_ip"):
            add_failure(failures, row, "node_mapping_mismatch")

        # Le prime 57 feature dell'e-TDG devono essere esattamente quelle TDG
        if (
            x_tdg.ndim == 2
            and x_etdg.ndim == 2
            and x_tdg.shape[1] == 57
            and x_etdg.shape[1] == 72
            and not np.array_equal(x_tdg, x_etdg[:, :57])
        ):
            add_failure(failures, row, "tdg_features_not_preserved_in_etdg")

        # Valori numerici validi
        if not np.isfinite(x_tdg).all():
            add_failure(failures, row, "non_finite_tdg_features")

        if not np.isfinite(x_etdg).all():
            add_failure(failures, row, "non_finite_etdg_features")

        # Label solo 0/1
        if not np.isin(y_tdg, [0, 1]).all():
            add_failure(
                failures,
                row,
                "invalid_node_labels",
                str(np.unique(y_tdg)),
            )

        checked += 1
        split_counts[(str(row["split"]), str(row["graph_type"]))] += 1

    except Exception as e:
        add_failure(failures, row, "validation_exception", repr(e))

    if i % 500 == 0 or i == len(timeline):
        elapsed = time.time() - start_time
        print(
            f"[{i}/{len(timeline)}] "
            f"checked={checked} failures={len(failures)} "
            f"elapsed={elapsed/60:.1f} min",
            flush=True,
        )


elapsed = time.time() - start_time

pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)

summary = {
    "timeline_rows": int(len(timeline)),
    "tdg_files_filesystem": int(len(tdg_files)),
    "etdg_files_filesystem": int(len(etdg_files)),
    "pairs_checked": int(checked),
    "failure_count": int(len(failures)),
    "tdg_feature_dimensions": {
        str(k): int(v) for k, v in sorted(feature_dims_tdg.items())
    },
    "etdg_feature_dimensions": {
        str(k): int(v) for k, v in sorted(feature_dims_etdg.items())
    },
    "elapsed_seconds": float(elapsed),
    "success": (
        len(timeline) == 21314
        and len(tdg_files) == 21314
        and len(etdg_files) == 21314
        and checked == 21314
        and len(failures) == 0
        and feature_dims_tdg == Counter({57: 21314})
        and feature_dims_etdg == Counter({72: 21314})
    ),
}

with SUMMARY_JSON.open("w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 90)
print("RISULTATO FINALE")
print("=" * 90)
print(json.dumps(summary, indent=2))
print("\nReport:", SUMMARY_JSON)
print("Eventuali anomalie:", FAILURES_CSV)
