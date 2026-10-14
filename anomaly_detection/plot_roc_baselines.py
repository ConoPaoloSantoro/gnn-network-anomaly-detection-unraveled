from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


RUNS = {
    "GCN": Path(
        "anomaly_detection/results/base_gnn/gcn_etdg/run_20260717_101857"
    ),
    "GraphSAGE": Path(
        "anomaly_detection/results/base_gnn/sage_etdg/run_20260717_122851"
    ),
    "GAT": Path(
        "anomaly_detection/results/base_gnn/gat_etdg/run_20260717_132453"
    ),
}

SPLITS = {
    "test_week5": "Test Week 5",
    "test_week6": "Test Week 6",
}

OUTPUT_DIR = Path("anomaly_detection/results/base_gnn/roc_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split, split_title in SPLITS.items():

    plt.figure(figsize=(7, 6))

    for model_name, run_dir in RUNS.items():

        scores_file = run_dir / f"{split}_node_scores.npz"

        if not scores_file.exists():
            print(f"[MISSING] {scores_file}")
            continue

        data = np.load(scores_file)

        scores = np.asarray(data["scores"]).reshape(-1)
        labels = np.asarray(data["labels"]).reshape(-1).astype(int)

        valid = np.isfinite(scores)
        scores = scores[valid]
        labels = labels[valid]

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        print(
            f"{model_name} | {split_title} | "
            f"nodes={len(labels)} | AUROC={auc:.6f}"
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUROC = {auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier (AUROC = 0.500)",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Node-level ROC Curves - {split_title}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"roc_node_{split}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")

print("ROC curves completed.")
