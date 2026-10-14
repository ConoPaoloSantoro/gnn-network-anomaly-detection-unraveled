from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import add_remaining_self_loops
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

FEATURE_DIMS = {
    "tdg": 57,
    "etdg": 72,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OC-GNN adaptation for exact Unraveled TDG/e-TDG snapshots."
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
        default="anomaly_detection/results/ocgnn",
    )
    parser.add_argument(
        "--representation",
        choices=["tdg", "etdg"],
        required=True,
    )

    # Common experimental protocol from the comparative study.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    # OC-GNN configuration used by the comparative repository.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--beta",
        "--nu",
        dest="beta",
        type=float,
        default=0.1,
        help="OC-GNN hypersphere trade-off parameter (0 < beta <= 1).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Center initialization slack used by the reference implementation.",
    )

    parser.add_argument(
        "--validation-fpr-target",
        type=float,
        default=0.01,
        help="Choose the node threshold on benign validation nodes so empirical FPR is strictly below this value.",
    )
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


def save_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


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
    frame = frame.sort_values("global_graph_index", kind="stable").reset_index(drop=True)

    counts = frame["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unexpected split counts. Expected {EXPECTED_SPLIT_COUNTS}, found {counts}"
        )

    for split in ("train", "validation"):
        values = set(frame.loc[frame["split"] == split, "graph_type"])
        if not values.issubset({"benign", "full_benign"}):
            raise ValueError(f"{split} is not benign-only: {sorted(values)}")

    return frame


def graph_path(row: pd.Series, graph_root: Path, representation: str) -> Path:
    graph_type = str(row["graph_type"])
    if graph_type not in TYPE_TO_FOLDER:
        raise ValueError(f"Unknown graph type: {graph_type}")

    rep_folder = "etdg_graph" if representation == "etdg" else "tdg_graph"
    return (
        graph_root
        / str(row["day_folder"])
        / str(row["capture"])
        / rep_folder
        / TYPE_TO_FOLDER[graph_type]
        / f"{row['graph_id']}.pkl"
    )


def fit_minmax_scaler(
    rows: pd.DataFrame,
    graph_root: Path,
    representation: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    print(f"\nCalcolo min-max {representation.upper()} sul solo training benigno...")
    feature_min = np.full(feature_dim, np.inf, dtype=np.float64)
    feature_max = np.full(feature_dim, -np.inf, dtype=np.float64)
    node_count = 0

    for _, row in tqdm(
        rows.iterrows(), total=len(rows), desc="Feature min-max"
    ):
        path = graph_path(row, graph_root, representation)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != feature_dim:
            raise ValueError(
                f"Expected {representation} features (N,{feature_dim}), found {x.shape} in {path}"
            )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite features in {path}")
        if x.shape[0] == 0:
            continue

        feature_min = np.minimum(feature_min, x.min(axis=0))
        feature_max = np.maximum(feature_max, x.max(axis=0))
        node_count += int(x.shape[0])

    if node_count == 0:
        raise RuntimeError("No training nodes found for feature scaling")

    feature_scale = feature_max - feature_min
    constant_mask = feature_scale < 1e-12
    feature_scale[constant_mask] = 1.0

    print("Training nodes used for min-max:", node_count)
    print("Constant feature columns:", np.where(constant_mask)[0].tolist())

    return (
        feature_min.astype(np.float32),
        feature_scale.astype(np.float32),
        node_count,
    )


class UnraveledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        graph_root: Path,
        representation: str,
        feature_min: np.ndarray,
        feature_scale: np.ndarray,
        feature_dim: int,
        preload: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.graph_root = graph_root
        self.representation = representation
        self.feature_min = feature_min
        self.feature_scale = feature_scale
        self.feature_dim = feature_dim
        self.cache: list[Data] | None = None

        if preload:
            self.cache = [
                self._load(i)
                for i in tqdm(range(len(self.rows)), desc="Preloading graphs")
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, idx: int) -> Data:
        row = self.rows.iloc[idx]
        path = graph_path(row, self.graph_root, self.representation)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        adjacency = np.asarray(graph["adj"])
        labels = np.asarray(graph["node_labels"], dtype=np.int64).reshape(-1)
        x = np.asarray(graph["node_features"], dtype=np.float32)

        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Invalid adjacency {adjacency.shape} in {path}")
        n = int(adjacency.shape[0])
        if labels.shape[0] != n:
            raise ValueError(f"Node-label mismatch in {path}: {labels.shape[0]} vs {n}")
        if x.shape != (n, self.feature_dim):
            raise ValueError(
                f"Expected {self.representation} features ({n},{self.feature_dim}), found {x.shape} in {path}"
            )

        x = (x - self.feature_min.reshape(1, -1)) / self.feature_scale.reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        src, dst = np.nonzero(adjacency)
        edge_index = torch.from_numpy(
            np.vstack([src, dst]).astype(np.int64, copy=False)
        ).long()
        # The reference loader explicitly adds self-loops before creating edge_index.
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=n)

        return Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            node_y=torch.from_numpy(labels),
            row_id=torch.tensor([idx], dtype=torch.long),
            num_nodes=n,
        )

    def __getitem__(self, idx: int) -> Data:
        return self.cache[idx] if self.cache is not None else self._load(idx)


class OCGNN(nn.Module):
    """OC-GNN encoder aligned with the comparative repository: GraphSAGE -> hypersphere."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gnn = GraphSAGE(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
            act="relu",
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn(x, edge_index)


def one_class_loss(
    distances: torch.Tensor,
    radius: float,
    beta: float,
) -> torch.Tensor:
    radius_squared = float(radius) ** 2
    score = distances - radius_squared
    return radius_squared + (1.0 / beta) * torch.mean(torch.relu(score))


@torch.no_grad()
def initialize_center(
    model: OCGNN,
    loader: DataLoader,
    device: torch.device,
    hidden_dim: int,
    eps: float,
) -> torch.Tensor:
    print("\nInitializing one-class center...")
    model.eval()
    center_sum = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
    node_count = 0

    for batch in tqdm(loader, desc="Center"):
        batch = batch.to(device, non_blocking=True)
        embedding = model(batch.x, batch.edge_index)
        center_sum += embedding.sum(dim=0)
        node_count += int(embedding.shape[0])

    if node_count == 0:
        raise RuntimeError("No training nodes found")

    center = center_sum / node_count
    small = torch.abs(center) < eps
    center[small & (center < 0)] = -eps
    center[small & (center > 0)] = eps

    print("Nodes used for center:", node_count)
    print("Center norm:", float(torch.linalg.vector_norm(center).item()))
    return center.detach()


def update_radius(distances: torch.Tensor, beta: float) -> float:
    with torch.no_grad():
        radius = torch.quantile(
            torch.sqrt(torch.clamp(distances, min=0.0)),
            1.0 - beta,
        )
    return float(radius.item())


def make_loader(
    dataset: UnraveledDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_loss(
    model: OCGNN,
    loader: DataLoader,
    center: torch.Tensor,
    radius: float,
    device: torch.device,
    beta: float,
) -> float:
    model.eval()
    total = 0.0
    total_nodes = 0

    for batch in tqdm(loader, desc="Evaluation", leave=False):
        batch = batch.to(device, non_blocking=True)
        z = model(batch.x, batch.edge_index)
        distances = torch.sum((z - center) ** 2, dim=1)
        loss = one_class_loss(distances, radius, beta)
        n = int(distances.numel())
        total += float(loss.item()) * n
        total_nodes += n

    return total / max(total_nodes, 1)


@torch.no_grad()
def collect_validation_scores(
    model: OCGNN,
    loader: DataLoader,
    center: torch.Tensor,
    radius: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    radius_squared = float(radius) ** 2

    for batch in tqdm(loader, desc="Validation scoring"):
        batch = batch.to(device, non_blocking=True)
        z = model(batch.x, batch.edge_index)
        distances = torch.sum((z - center) ** 2, dim=1)
        scores = distances - radius_squared
        scores_parts.append(scores.detach().cpu().numpy().astype(np.float32))
        labels_parts.append(batch.node_y.detach().cpu().numpy().astype(np.int8))

    return np.concatenate(scores_parts), np.concatenate(labels_parts)


def select_threshold_for_fpr(scores: np.ndarray, target_fpr: float) -> tuple[float, float, int]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("No validation scores")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("validation_fpr_target must be in (0,1)")

    n = int(scores.size)
    max_fp = max(0, int(np.ceil(target_fpr * n)) - 1)
    threshold_index = n - max_fp - 1
    threshold = float(np.partition(scores, threshold_index)[threshold_index])
    actual_fp = int(np.sum(scores > threshold))
    actual_fpr = float(actual_fp / n)

    if not actual_fpr < target_fpr:
        raise RuntimeError(
            f"Could not satisfy validation FPR target: {actual_fpr} >= {target_fpr}"
        )
    return threshold, actual_fpr, actual_fp


def snapshot_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    graph_type: str,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

    result: dict[str, float] = {
        "FPR": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    }

    if graph_type in {"mixed", "malicious", "full_malicious"}:
        result.update(
            {
                "TPR": float(recall_score(labels, pred, zero_division=0)),
                "PRECISION": float(precision_score(labels, pred, zero_division=0)),
                "F1": float(f1_score(labels, pred, zero_division=0)),
                "AUC": (
                    float(roc_auc_score(labels, scores))
                    if np.unique(labels).size == 2
                    else float("nan")
                ),
            }
        )
    return result


@torch.no_grad()
def evaluate_test_split(
    model: OCGNN,
    loader: DataLoader,
    dataset: UnraveledDataset,
    center: torch.Tensor,
    radius: float,
    threshold: float,
    device: torch.device,
    split: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()
    radius_squared = float(radius) ** 2
    rows_out: list[dict[str, Any]] = []
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Scoring {split}"):
        batch = batch.to(device, non_blocking=True)
        z = model(batch.x, batch.edge_index)
        distances = torch.sum((z - center) ** 2, dim=1)
        scores = distances - radius_squared

        ptr = batch.ptr.detach().cpu().numpy()
        row_ids = batch.row_id.detach().cpu().numpy().reshape(-1)
        labels_batch = batch.node_y.detach().cpu().numpy().astype(np.int64)
        scores_batch = scores.detach().cpu().numpy().astype(np.float64)

        score_parts.append(scores_batch.astype(np.float32))
        label_parts.append(labels_batch.astype(np.int8))

        for g in range(batch.num_graphs):
            start, end = int(ptr[g]), int(ptr[g + 1])
            row = dataset.rows.iloc[int(row_ids[g])]
            local_labels = labels_batch[start:end]
            local_scores = scores_batch[start:end]
            graph_type = str(row["graph_type"])
            metrics = snapshot_metrics(local_labels, local_scores, threshold, graph_type)

            base = {
                "global_graph_index": int(row["global_graph_index"]),
                "split": split,
                "graph_type": graph_type,
                "day_folder": str(row["day_folder"]),
                "capture": str(row["capture"]),
                "graph_id": str(row["graph_id"]),
                "num_nodes": int(end - start),
            }
            for metric, value in metrics.items():
                rows_out.append({**base, "metric": metric, "value": value})

    frame = pd.DataFrame(rows_out).sort_values(
        ["global_graph_index", "metric"], kind="stable"
    ).reset_index(drop=True)
    payload = {
        "scores": np.concatenate(score_parts),
        "labels": np.concatenate(label_parts),
    }
    return frame, payload


def summarize_snapshot_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []

    for (split, graph_type, metric), group in frame.groupby(
        ["split", "graph_type", "metric"], sort=False
    ):
        values = group["value"].to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        output.append(
            {
                "split": split,
                "graph_type": graph_type,
                "metric": metric,
                "mean": float(np.mean(valid)) if valid.size else float("nan"),
                "std": float(np.std(valid, ddof=0)) if valid.size else float("nan"),
                "snapshots": int(len(group)),
                "valid_snapshots": int(valid.size),
            }
        )

    return pd.DataFrame(output)


def smoke_subset(rows: pd.DataFrame, split: str, size: int) -> pd.DataFrame:
    if split in {"train", "validation"}:
        return rows.head(size).reset_index(drop=True)

    benign_mask = rows["graph_type"].isin(["benign", "full_benign"])
    benign = rows[benign_mask].head(size // 2)
    anomalous = rows[~benign_mask].head(size - len(benign))
    return (
        pd.concat([benign, anomalous], ignore_index=True)
        .sort_values("global_graph_index", kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.beta <= 1.0):
        raise ValueError("beta must satisfy 0 < beta <= 1")
    if not (0.0 < args.validation_fpr_target < 1.0):
        raise ValueError("validation-fpr-target must be in (0,1)")

    set_seed(args.seed)
    project = args.project_root.expanduser().resolve()
    timeline_path = project / args.timeline
    graph_root = project / args.graph_root
    feature_dim = FEATURE_DIMS[args.representation]
    rep_name = "e-TDG" if args.representation == "etdg" else "TDG"

    output_root = project / args.output_root / f"ocgnn_{args.representation}"
    prefix = "smoke" if args.smoke_test else "run"
    run_dir = output_root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"OC-GNN - UNRAVELED - {rep_name}")
    print("=" * 80)
    print("Project root:", project)
    print("Graph root:", graph_root)
    print("Run directory:", run_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Visible GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )
    print("Feature dimension:", feature_dim)
    print("Backbone: GraphSAGE")
    print("Hidden dimension:", args.hidden_dim)
    print("Layers:", args.num_layers)
    print("Beta:", args.beta)

    timeline = load_timeline(timeline_path)
    rows = {
        split: timeline[timeline["split"] == split].copy().reset_index(drop=True)
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

    feature_min, feature_scale, scaler_nodes = fit_minmax_scaler(
        rows["train"], graph_root, args.representation, feature_dim
    )
    np.savez_compressed(
        run_dir / "feature_scaler.npz",
        minimum=feature_min,
        scale=feature_scale,
        training_nodes=scaler_nodes,
        kind="training_minmax",
    )

    train_ds = UnraveledDataset(
        rows["train"],
        graph_root,
        args.representation,
        feature_min,
        feature_scale,
        feature_dim,
        preload=args.preload_train_val,
    )
    val_ds = UnraveledDataset(
        rows["validation"],
        graph_root,
        args.representation,
        feature_min,
        feature_scale,
        feature_dim,
        preload=args.preload_train_val,
    )

    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers, device, args.seed
    )
    train_eval_loader = make_loader(
        train_ds, args.batch_size, False, args.num_workers, device, args.seed
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.num_workers, device, args.seed
    )

    model = OCGNN(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", parameters)

    center = initialize_center(
        model, train_eval_loader, device, args.hidden_dim, args.eps
    )
    np.save(run_dir / "initial_center.npy", center.detach().cpu().numpy())
    radius = 0.0

    config = vars(args).copy()
    config.update(
        {
            "project_root": str(project),
            "timeline_path": str(timeline_path),
            "graph_root": str(graph_root),
            "run_dir": str(run_dir),
            "representation_name": rep_name,
            "input_dim": feature_dim,
            "parameters": parameters,
            "backbone": "GraphSAGE",
            "center_is_fixed_after_initialization": True,
            "radius_update": "(1-beta) quantile of sqrt training distances after each epoch",
            "feature_scaler": "training_minmax",
            "threshold_policy": "empirical validation node FPR strictly below target",
            "evaluation": "node metrics per snapshot, reported as mean and standard deviation",
        }
    )
    save_json(run_dir / "config.json", config)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []

    print("\nTraining started")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        total_nodes = 0
        epoch_distances: list[torch.Tensor] = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z = model(batch.x, batch.edge_index)
            distances = torch.sum((z - center) ** 2, dim=1)
            loss = one_class_loss(distances, radius, args.beta)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            epoch_distances.append(distances.detach().cpu())
            loss.backward()
            optimizer.step()

            n = int(distances.numel())
            total_loss += float(loss.item()) * n
            total_nodes += n

        train_loss = total_loss / max(total_nodes, 1)
        radius = update_radius(torch.cat(epoch_distances), args.beta)

        val_loss = evaluate_loss(
            model, val_loader, center, radius, device, args.beta
        )
        elapsed = time.time() - started

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "radius": radius,
                "elapsed_seconds": elapsed,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"Epoch {epoch:03d} | train={train_loss:.8f} | "
            f"validation={val_loss:.8f} | radius={radius:.8f} | time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "radius": radius,
                    "center": center.detach().cpu(),
                    "config": config,
                },
                checkpoint_path,
            )
            print("Best checkpoint saved")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    center = checkpoint["center"].to(device)
    radius = float(checkpoint["radius"])

    print("\nBest epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["validation_loss"])
    print("Best radius:", radius)

    val_scores, val_labels = collect_validation_scores(
        model, val_loader, center, radius, device
    )
    if np.any(val_labels != 0):
        raise RuntimeError("Validation contains anomalous node labels")

    node_threshold, actual_val_fpr, actual_val_fp = select_threshold_for_fpr(
        val_scores, args.validation_fpr_target
    )
    thresholds = {
        "criterion": "validation_fpr_below_target",
        "target_fpr": float(args.validation_fpr_target),
        "actual_validation_fpr": actual_val_fpr,
        "actual_validation_fp": actual_val_fp,
        "validation_nodes": int(val_scores.size),
        "node_score": node_threshold,
        "ocgnn_radius_boundary": 0.0,
        "radius": radius,
    }
    save_json(run_dir / "thresholds.json", thresholds)
    np.savez_compressed(
        run_dir / "validation_node_scores.npz",
        scores=val_scores.astype(np.float32),
        labels=val_labels.astype(np.int8),
    )

    print("Node threshold:", node_threshold)
    print("Target validation FPR:", args.validation_fpr_target)
    print("Actual validation FPR:", actual_val_fpr)

    metric_frames: list[pd.DataFrame] = []
    for split in ("test_week5", "test_week6"):
        print(f"\nEvaluating {split}")
        test_ds = UnraveledDataset(
            rows[split],
            graph_root,
            args.representation,
            feature_min,
            feature_scale,
            feature_dim,
            preload=False,
        )
        test_loader = make_loader(
            test_ds, args.batch_size, False, args.num_workers, device, args.seed
        )
        snapshot_frame, node_payload = evaluate_test_split(
            model,
            test_loader,
            test_ds,
            center,
            radius,
            node_threshold,
            device,
            split,
        )
        snapshot_frame.to_csv(
            run_dir / f"{split}_snapshot_metrics.csv", index=False
        )
        np.savez_compressed(
            run_dir / f"{split}_node_scores.npz", **node_payload
        )

        summary_frame = summarize_snapshot_metrics(snapshot_frame)
        metric_frames.append(summary_frame)
        print(f"\nSnapshot-averaged node metrics - {split}:")
        print(summary_frame.to_string(index=False))

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    summary = {
        "method": "OC-GNN",
        "task": "node anomaly detection",
        "representation": rep_name,
        "feature_dim": feature_dim,
        "backbone": "GraphSAGE",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "beta": args.beta,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "best_radius": radius,
        "center_norm": float(torch.linalg.vector_norm(center).item()),
        "thresholds": thresholds,
        "parameters": parameters,
        "evaluation": "node-level metrics computed per snapshot and reported as mean and standard deviation",
        "metrics": metrics.to_dict(orient="records"),
    }
    save_json(run_dir / "summary.json", summary)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print("\nResults saved in:")
    print(run_dir)


if __name__ == "__main__":
    main()
