from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch_geometric
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN
from tqdm import tqdm


EXPECTED_SPLIT_COUNTS = {
    "train": 4411,
    "validation": 1327,
    "test_week5": 3761,
    "test_week6": 4105,
}

TYPE_TO_FOLDER = {
    "benign": "full_benign",
    "full_benign": "full_benign",
    "mixed": "mixed",
    "malicious": "full_malicious",
    "full_malicious": "full_malicious",
}

BENIGN_GRAPH_TYPES = {"benign", "full_benign"}


@dataclass(frozen=True)
class ExperimentConfig:
    representation: str
    feature_dim: int
    epochs: int
    accumulation_steps: int
    hidden_dim: int
    encoder_layers: int
    decoder_layers: int
    dropout: float
    learning_rate: float
    alpha: float
    validation_fpr_target: float
    patience: int
    min_delta: float
    seed: int
    num_workers: int
    preload_train_val: bool
    smoke_test: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOMINANT adaptation for Unraveled TDG/e-TDG snapshots."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "gnn_thesis",
    )
    parser.add_argument(
        "--timeline",
        default="graph_timeline_merged_150000_final_split.csv",
    )
    parser.add_argument(
        "--graph-root",
        default="unraveled_graphs_150000_merged_captures_exact_57_72",
    )
    parser.add_argument(
        "--output-root",
        default="anomaly_detection/results/dominant",
    )
    parser.add_argument(
        "--representation",
        choices=["tdg", "etdg"],
        default="etdg",
    )

    # Reference protocol used by the comparative-study repository.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--validation-fpr-target", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload-train-val", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_timeline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "day_folder",
        "capture",
        "graph_id",
        "graph_type",
        "split",
        "global_graph_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timeline columns: {sorted(missing)}")

    frame = frame[frame["split"].isin(EXPECTED_SPLIT_COUNTS)].copy()
    frame["global_graph_index"] = pd.to_numeric(
        frame["global_graph_index"], errors="raise"
    ).astype(int)
    frame = frame.sort_values(
        "global_graph_index", kind="stable"
    ).reset_index(drop=True)

    counts = frame["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Found: {counts}"
        )

    for split in ("train", "validation"):
        graph_types = set(frame.loc[frame["split"] == split, "graph_type"])
        if not graph_types or not graph_types.issubset(BENIGN_GRAPH_TYPES):
            raise ValueError(f"{split} is not benign-only: {sorted(graph_types)}")

    return frame


def graph_path(row: pd.Series, graph_root: Path, representation: str) -> Path:
    graph_type = str(row["graph_type"])
    if graph_type not in TYPE_TO_FOLDER:
        raise ValueError(f"Unknown graph type: {graph_type}")

    graph_folder = "tdg_graph" if representation == "tdg" else "etdg_graph"
    return (
        graph_root
        / str(row["day_folder"])
        / str(row["capture"])
        / graph_folder
        / TYPE_TO_FOLDER[graph_type]
        / f"{row['graph_id']}.pkl"
    )


def feature_dim_for_representation(representation: str) -> int:
    return 57 if representation == "tdg" else 72


def fit_min_max_scaler(
    train_rows: pd.DataFrame,
    graph_root: Path,
    representation: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    print("\nCalcolo min-max sul solo training benigno...")

    feature_min = np.full(feature_dim, np.inf, dtype=np.float64)
    feature_max = np.full(feature_dim, -np.inf, dtype=np.float64)
    training_nodes = 0

    for _, row in tqdm(
        train_rows.iterrows(),
        total=len(train_rows),
        desc="Feature min-max",
    ):
        path = graph_path(row, graph_root, representation)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != feature_dim:
            raise ValueError(
                f"Expected {representation} features (N, {feature_dim}), "
                f"found {x.shape} in {path}"
            )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite features in {path}")
        if x.shape[0] == 0:
            continue

        feature_min = np.minimum(feature_min, x.min(axis=0))
        feature_max = np.maximum(feature_max, x.max(axis=0))
        training_nodes += int(x.shape[0])

    if training_nodes == 0:
        raise RuntimeError("No training nodes available for min-max fitting")

    constant_mask = (feature_max - feature_min) < 1e-12
    feature_max[constant_mask] = feature_min[constant_mask] + 1.0

    print("Training nodes used for min-max:", training_nodes)
    print("Constant columns:", np.where(constant_mask)[0].tolist())

    return (
        feature_min.astype(np.float32),
        feature_max.astype(np.float32),
        training_nodes,
    )


class UnraveledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        graph_root: Path,
        representation: str,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        preload: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.graph_root = graph_root
        self.representation = representation
        self.feature_dim = feature_dim_for_representation(representation)
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.cache: list[Data] | None = None

        if preload:
            self.cache = []
            for index in tqdm(range(len(self.rows)), desc="Preloading graphs"):
                self.cache.append(self._load(index))

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, idx: int) -> Data:
        row = self.rows.iloc[idx]
        path = graph_path(row, self.graph_root, self.representation)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float32)
        adjacency = np.asarray(graph["adj"])
        labels = np.asarray(graph["node_labels"], dtype=np.int64).reshape(-1)

        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"Invalid feature shape {x.shape} in {path}")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Invalid adjacency shape {adjacency.shape} in {path}")
        if adjacency.shape[0] != x.shape[0] or labels.shape[0] != x.shape[0]:
            raise ValueError(
                f"Node count mismatch in {path}: "
                f"x={x.shape}, adj={adjacency.shape}, labels={labels.shape}"
            )

        x = (x - self.feature_min) / (self.feature_max - self.feature_min)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite normalized features in {path}")

        # The reference loader adds self-loops before deriving edge_index.
        adjacency_binary = (adjacency != 0)
        np.fill_diagonal(adjacency_binary, True)
        source, target = np.nonzero(adjacency_binary)
        edge_index = np.vstack([source, target]).astype(np.int64, copy=False)

        return Data(
            x=torch.from_numpy(x.astype(np.float32, copy=False)),
            edge_index=torch.from_numpy(edge_index),
            node_y=torch.from_numpy(labels),
            row_id=torch.tensor([idx], dtype=torch.long),
            global_id=torch.tensor(
                [int(row["global_graph_index"])], dtype=torch.long
            ),
            num_nodes=int(x.shape[0]),
        )

    def __getitem__(self, idx: int) -> Data:
        if self.cache is not None:
            return self.cache[idx]
        return self._load(idx)


class DotProductDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gcn = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
            act=torch.nn.functional.relu,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gcn(x, edge_index)
        return h @ h.T


class DOMINANT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if encoder_layers < 2:
            raise ValueError("encoder_layers must be at least 2")
        if decoder_layers < 2:
            raise ValueError("decoder_layers must be at least 2")

        self.shared_encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_layers=encoder_layers,
            out_channels=hidden_dim,
            dropout=dropout,
            act=torch.nn.functional.relu,
        )
        self.attr_decoder = GCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=decoder_layers,
            out_channels=in_dim,
            dropout=dropout,
            act=torch.nn.functional.relu,
        )
        self.struct_decoder = DotProductDecoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers - 1,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.shared_encoder(x, edge_index)
        x_hat = self.attr_decoder(embedding, edge_index)
        s_hat = self.struct_decoder(embedding, edge_index)
        return x_hat, s_hat


def dense_structure_target(data: Data) -> torch.Tensor:
    num_nodes = int(data.num_nodes)
    target = torch.zeros(
        (num_nodes, num_nodes),
        dtype=data.x.dtype,
        device=data.x.device,
    )
    if data.edge_index.numel() > 0:
        target[data.edge_index[0], data.edge_index[1]] = 1.0
    return target


def objective_function(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    s: torch.Tensor,
    s_hat: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Same score convention as the reference implementation:
    # alpha * attribute_error + (1-alpha) * structure_error.
    attr_error = torch.sqrt(torch.sum((x - x_hat) ** 2, dim=1))
    struct_error = torch.sqrt(torch.sum((s - s_hat) ** 2, dim=1))
    score = alpha * attr_error + (1.0 - alpha) * struct_error
    return score, attr_error, struct_error


def compute_scores(
    model: DOMINANT,
    data: Data,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_hat, s_hat = model(data.x, data.edge_index)
    s = dense_structure_target(data)
    return objective_function(data.x, x_hat, s, s_hat, alpha)


def smoke_subset(rows: pd.DataFrame, split: str, size: int) -> pd.DataFrame:
    if split in {"train", "validation"}:
        return rows.head(size).reset_index(drop=True)

    benign_mask = rows["graph_type"].isin(BENIGN_GRAPH_TYPES)
    benign = rows[benign_mask].head(size // 2)
    mixed = rows[~benign_mask].head(size - len(benign))
    return (
        pd.concat([benign, mixed], ignore_index=True)
        .sort_values("global_graph_index", kind="stable")
        .reset_index(drop=True)
    )


def build_loader(
    dataset: UnraveledDataset,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate(
    model: DOMINANT,
    loader: DataLoader,
    dataset: UnraveledDataset,
    device: torch.device,
    alpha: float,
    collect_nodes: bool,
) -> tuple[float, pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()

    total_graph_loss = 0.0
    graph_count = 0
    graph_rows: list[dict[str, Any]] = []

    score_parts: list[np.ndarray] = []
    attr_parts: list[np.ndarray] = []
    struct_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    graph_id_parts: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluation"):
        batch = batch.to(device, non_blocking=True)
        scores, attr_error, struct_error = compute_scores(model, batch, alpha)

        total_graph_loss += float(scores.mean().item())
        graph_count += 1

        row_id = int(batch.row_id.item())
        row = dataset.rows.iloc[row_id]
        global_id = int(row["global_graph_index"])
        num_nodes = int(scores.numel())

        graph_rows.append(
            {
                "global_graph_index": global_id,
                "split": str(row["split"]),
                "day_folder": str(row["day_folder"]),
                "capture": str(row["capture"]),
                "graph_id": str(row["graph_id"]),
                "graph_type": str(row["graph_type"]),
                "num_nodes": num_nodes,
                "score_mean": float(scores.mean().item()),
                "attribute_error_mean": float(attr_error.mean().item()),
                "structure_error_mean": float(struct_error.mean().item()),
            }
        )

        if collect_nodes:
            score_parts.append(scores.cpu().numpy().astype(np.float32))
            attr_parts.append(attr_error.cpu().numpy().astype(np.float32))
            struct_parts.append(struct_error.cpu().numpy().astype(np.float32))
            labels = batch.node_y.cpu().numpy().astype(np.int8)
            label_parts.append(labels)
            graph_id_parts.append(
                np.full(num_nodes, global_id, dtype=np.int64)
            )

    graph_frame = pd.DataFrame(graph_rows).sort_values(
        "global_graph_index", kind="stable"
    ).reset_index(drop=True)

    node_payload: dict[str, np.ndarray] = {}
    if collect_nodes:
        node_payload = {
            "scores": np.concatenate(score_parts),
            "attribute_scores": np.concatenate(attr_parts),
            "structure_scores": np.concatenate(struct_parts),
            "labels": np.concatenate(label_parts),
            "global_graph_index": np.concatenate(graph_id_parts),
        }

    return total_graph_loss / max(graph_count, 1), graph_frame, node_payload


def select_threshold_for_fpr(
    benign_scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, float, int]:
    scores = np.asarray(benign_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Validation scores are empty")
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain non-finite values")

    max_false_positives = max(0, math.ceil(target_fpr * scores.size) - 1)
    sorted_scores = np.sort(scores)[::-1]

    if max_false_positives == 0:
        threshold = float(sorted_scores[0])
    else:
        threshold = float(sorted_scores[max_false_positives])

    actual_fp = int(np.sum(scores > threshold))
    actual_fpr = float(actual_fp / scores.size)
    if not actual_fpr < target_fpr:
        raise RuntimeError(
            f"Could not satisfy validation FPR constraint: "
            f"{actual_fpr} >= {target_fpr}"
        )
    return threshold, actual_fpr, actual_fp


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    precision = float(precision_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions, zero_division=0))
    auc = (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )

    return {
        "samples": int(labels.size),
        "benign": int((labels == 0).sum()),
        "anomalous": int((labels == 1).sum()),
        "threshold": float(threshold),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_snapshot_metrics(
    graph_frame: pd.DataFrame,
    node_payload: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    scores = np.asarray(node_payload["scores"])
    labels = np.asarray(node_payload["labels"])
    graph_ids = np.asarray(node_payload["global_graph_index"])

    rows: list[dict[str, Any]] = []
    for _, graph_row in graph_frame.iterrows():
        global_id = int(graph_row["global_graph_index"])
        mask = graph_ids == global_id
        if not np.any(mask):
            raise RuntimeError(f"No node scores found for graph {global_id}")

        metrics = binary_metrics(labels[mask], scores[mask], threshold)
        metrics.update(
            {
                "split": str(graph_row["split"]),
                "global_graph_index": global_id,
                "day_folder": str(graph_row["day_folder"]),
                "capture": str(graph_row["capture"]),
                "graph_id": str(graph_row["graph_id"]),
                "graph_type": str(graph_row["graph_type"]),
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        "global_graph_index", kind="stable"
    ).reset_index(drop=True)


def summarize_snapshot_metrics(snapshot_metrics: pd.DataFrame) -> pd.DataFrame:
    if snapshot_metrics.empty:
        return pd.DataFrame()

    split = str(snapshot_metrics.iloc[0]["split"])
    scenarios = [
        (
            "benign",
            snapshot_metrics["graph_type"].isin(BENIGN_GRAPH_TYPES),
            ["fpr"],
        ),
        (
            "mixed",
            ~snapshot_metrics["graph_type"].isin(BENIGN_GRAPH_TYPES),
            ["fpr", "tpr", "precision", "f1", "auc"],
        ),
    ]

    rows: list[dict[str, Any]] = []
    for scenario, mask, metrics in scenarios:
        subset = snapshot_metrics.loc[mask]
        if subset.empty:
            continue

        for metric in metrics:
            values = pd.to_numeric(
                subset[metric], errors="coerce"
            ).to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "split": split,
                    "graph_type": scenario,
                    "metric": metric.upper(),
                    "mean": (
                        float(np.mean(finite))
                        if finite.size
                        else float("nan")
                    ),
                    "std": (
                        float(np.std(finite, ddof=0))
                        if finite.size
                        else float("nan")
                    ),
                    "snapshots": int(len(subset)),
                    "valid_snapshots": int(finite.size),
                }
            )

    return pd.DataFrame(rows)


def save_environment_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    config: ExperimentConfig,
    script_path: Path,
    timeline_path: Path,
    project_root: Path,
    device: torch.device,
) -> None:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "arguments": serializable_args,
        "experiment_config": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "script_sha256": sha256_file(script_path),
        "timeline_sha256": sha256_file(timeline_path),
        "reference_commits": {
            "DOMINANT_original": git_commit(
                project_root / "external_methods" / "DOMINANT_original"
            )
        },
    }
    save_json(run_dir / "experiment_metadata.json", metadata)


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < args.validation_fpr_target < 1.0:
        raise ValueError("validation_fpr_target must be between 0 and 1")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if args.encoder_layers < 2 or args.decoder_layers < 2:
        raise ValueError("encoder/decoder layers must be >= 2")

    set_seed(args.seed)

    project_root = args.project_root.expanduser().resolve()
    timeline_path = project_root / args.timeline
    graph_root = project_root / args.graph_root
    feature_dim = feature_dim_for_representation(args.representation)
    rep_name = "TDG" if args.representation == "tdg" else "e-TDG"

    output_root = (
        project_root
        / args.output_root
        / f"dominant_{args.representation}"
    )
    prefix = "smoke" if args.smoke_test else "run"
    run_dir = output_root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    if not graph_root.is_dir():
        raise FileNotFoundError(graph_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"DOMINANT - UNRAVELED - {rep_name}")
    print("=" * 80)
    print("Project root:", project_root)
    print("Graph root:", graph_root)
    print("Run directory:", run_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Feature dimension:", feature_dim)
    print("Alpha:", args.alpha)
    print("Structure reconstruction: dense adjacency reconstruction")
    print("Gradient accumulation steps:", args.accumulation_steps)

    timeline = load_timeline(timeline_path)
    rows = {
        split: timeline[timeline["split"] == split]
        .copy()
        .reset_index(drop=True)
        for split in EXPECTED_SPLIT_COUNTS
    }

    if args.smoke_test:
        rows["train"] = smoke_subset(rows["train"], "train", 64)
        rows["validation"] = smoke_subset(rows["validation"], "validation", 32)
        rows["test_week5"] = smoke_subset(rows["test_week5"], "test_week5", 64)
        rows["test_week6"] = smoke_subset(rows["test_week6"], "test_week6", 64)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)

    print("\nGraphs used:")
    for split, split_rows in rows.items():
        print(
            f"{split:12s}: {len(split_rows):5d}",
            split_rows["graph_type"].value_counts().to_dict(),
        )

    feature_min, feature_max, scaler_nodes = fit_min_max_scaler(
        rows["train"],
        graph_root,
        args.representation,
        feature_dim,
    )
    np.savez_compressed(
        run_dir / "feature_scaler.npz",
        min=feature_min,
        max=feature_max,
        training_nodes=scaler_nodes,
    )

    train_dataset = UnraveledDataset(
        rows["train"],
        graph_root,
        args.representation,
        feature_min,
        feature_max,
        preload=args.preload_train_val,
    )
    validation_dataset = UnraveledDataset(
        rows["validation"],
        graph_root,
        args.representation,
        feature_min,
        feature_max,
        preload=args.preload_train_val,
    )

    train_loader = build_loader(
        train_dataset,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    validation_loader = build_loader(
        validation_dataset,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    model = DOMINANT(
        in_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Trainable parameters:", parameter_count)

    config = ExperimentConfig(
        representation=rep_name,
        feature_dim=feature_dim,
        epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        alpha=args.alpha,
        validation_fpr_target=args.validation_fpr_target,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        num_workers=args.num_workers,
        preload_train_val=args.preload_train_val,
        smoke_test=args.smoke_test,
    )
    save_environment_metadata(
        run_dir,
        args,
        config,
        Path(__file__).resolve(),
        timeline_path,
        project_root,
        device,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "best_checkpoint.pt"

    print("\nTraining started")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_graph_loss_sum = 0.0
        train_graphs = 0
        pending_losses: list[torch.Tensor] = []

        for graph_index, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"),
            start=1,
        ):
            batch = batch.to(device, non_blocking=True)
            scores, _, _ = compute_scores(model, batch, args.alpha)
            graph_loss = scores.mean()

            if not torch.isfinite(graph_loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            pending_losses.append(graph_loss)
            train_graph_loss_sum += float(graph_loss.detach().item())
            train_graphs += 1

            if len(pending_losses) == args.accumulation_steps:
                loss = torch.stack(pending_losses).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_losses.clear()

        # Flush the last partial accumulation group instead of dropping it.
        if pending_losses:
            loss = torch.stack(pending_losses).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = train_graph_loss_sum / max(train_graphs, 1)

        validation_loss, _, _ = evaluate(
            model,
            validation_loader,
            validation_dataset,
            device,
            args.alpha,
            collect_nodes=False,
        )

        elapsed = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train={train_loss:.8f} | "
            f"validation={validation_loss:.8f} | time={elapsed:.1f}s"
        )

        if validation_loss < best_validation_loss - args.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "config": asdict(config),
                    "feature_min": feature_min,
                    "feature_max": feature_max,
                },
                checkpoint_path,
            )
            print("Best checkpoint saved")
        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nBest epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["validation_loss"])

    validation_loss, validation_graphs, validation_nodes = evaluate(
        model,
        validation_loader,
        validation_dataset,
        device,
        args.alpha,
        collect_nodes=True,
    )

    node_threshold, validation_fpr, validation_fp = select_threshold_for_fpr(
        validation_nodes["scores"],
        args.validation_fpr_target,
    )

    validation_graphs.to_csv(
        run_dir / "validation_graph_scores.csv", index=False
    )
    np.savez_compressed(
        run_dir / "validation_node_scores.npz", **validation_nodes
    )
    thresholds = {
        "criterion": "validation_fpr_below_target",
        "target_fpr": float(args.validation_fpr_target),
        "actual_validation_fpr": float(validation_fpr),
        "actual_validation_fp": int(validation_fp),
        "validation_nodes": int(validation_nodes["scores"].size),
        "node_score": float(node_threshold),
    }
    save_json(run_dir / "thresholds.json", thresholds)

    print("Node threshold:", node_threshold)
    print("Target validation FPR:", args.validation_fpr_target)
    print("Actual validation FPR:", validation_fpr)

    weekly_metric_parts: list[pd.DataFrame] = []

    for split in ("test_week5", "test_week6"):
        print(f"\nEvaluating {split}")
        test_dataset = UnraveledDataset(
            rows[split],
            graph_root,
            args.representation,
            feature_min,
            feature_max,
            preload=False,
        )
        test_loader = build_loader(
            test_dataset,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )

        _, graph_scores, node_payload = evaluate(
            model,
            test_loader,
            test_dataset,
            device,
            args.alpha,
            collect_nodes=True,
        )

        snapshot_metrics = compute_snapshot_metrics(
            graph_scores,
            node_payload,
            node_threshold,
        )
        snapshot_metrics.to_csv(
            run_dir / f"{split}_snapshot_metrics.csv", index=False
        )

        weekly_metrics = summarize_snapshot_metrics(snapshot_metrics)
        weekly_metric_parts.append(weekly_metrics)

        graph_scores.to_csv(
            run_dir / f"{split}_graph_scores.csv", index=False
        )
        np.savez_compressed(
            run_dir / f"{split}_node_scores.npz", **node_payload
        )

        print(f"\nSnapshot-averaged node metrics - {split}:")
        print(weekly_metrics.to_string(index=False))

        del test_loader
        del test_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_frame = pd.concat(weekly_metric_parts, ignore_index=True)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)

    summary = {
        "method": "DOMINANT",
        "task": "node anomaly detection",
        "representation": rep_name,
        "feature_dim": feature_dim,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "validation_loss_recomputed": float(validation_loss),
        "thresholds": thresholds,
        "parameters": int(parameter_count),
        "alpha": float(args.alpha),
        "structure_reconstruction": "dense adjacency reconstruction",
        "evaluation": (
            "node-level metrics computed per snapshot and reported "
            "as mean and standard deviation"
        ),
        "metrics": metrics_frame.to_dict(orient="records"),
    }
    save_json(run_dir / "summary.json", summary)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(metrics_frame.to_string(index=False))
    print("\nResults saved in:")
    print(run_dir)


if __name__ == "__main__":
    main()
