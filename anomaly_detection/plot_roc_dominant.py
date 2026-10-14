from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


RUNS = {
    "e-TDG": Path(
        "anomaly_detection/results/dominant/dominant_etdg/run_20260725_102158"
    ),
    "TDG": Path(
        "anomaly_detection/results/dominant/dominant_tdg/run_20260725_105741"
    ),
}

SPLITS = {
    "test_week5": "Test Week 5",
    "test_week6": "Test Week 6",
}

OUTPUT_DIR = Path("anomaly_detection/results/dominant/roc_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for split, split_title in SPLITS.items():
    plt.figure(figsize=(7, 6))

    for rep_name, run_dir in RUNS.items():
        scores_file = run_dir / f"{split}_node_scores.npz"

        if not scores_file.exists():
            print(f"[MISSING] {scores_file}")
            continue

        data = np.load(scores_file)

        if "scores" not in data.files or "labels" not in data.files:
            print(
                f"[ERROR] {scores_file} non contiene 'scores' e 'labels'. "
                f"Chiavi trovate: {data.files}"
            )
            continue

        scores = np.asarray(data["scores"]).reshape(-1)
        labels = np.asarray(data["labels"]).reshape(-1).astype(int)

        valid = np.isfinite(scores)
        scores = scores[valid]
        labels = labels[valid]

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        print(
            f"{rep_name:6s} | {split_title:12s} | "
            f"nodes={len(labels):7d} | AUROC={auc:.6f}"
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{rep_name} (AUROC = {auc:.3f})",
        )

        np.savetxt(
            OUTPUT_DIR / f"roc_points_dominant_{rep_name.lower().replace('-', '').replace(' ', '_')}_{split}.csv",
            np.column_stack((fpr, tpr, thresholds)),
            delimiter=",",
            header="fpr,tpr,threshold",
            comments="",
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
    plt.title(f"Node-level ROC Curves - DOMINANT - {split_title}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"roc_node_dominant_{split}.png"
    pdf_path = OUTPUT_DIR / f"roc_node_dominant_{split}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

print("\nDOMINANT ROC curves completed.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
