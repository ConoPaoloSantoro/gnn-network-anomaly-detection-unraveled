from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm


GRAPH_TYPE_TO_FOLDER = {
    "benign": "full_benign",
    "full_benign": "full_benign",
    "mixed": "mixed",
    "malicious": "full_malicious",
    "full_malicious": "full_malicious",
}

GRAPH_TYPE_TO_LABEL = {
    "benign": 0,
    "full_benign": 0,
    "mixed": 1,
    "malicious": 1,
    "full_malicious": 1,
}

EXPECTED_SPLIT_COUNTS = {
    "train": 4411,
    "validation": 1327,
    "test_week5": 3761,
    "test_week6": 4105,
}


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    representation: str
    epochs: int
    batch_size: int
    hidden_dim: int
    embedding_dim: int
    num_layers: int
    dropout: float
    gat_heads: int
    learning_rate: float
    weight_decay: float
    alpha: float
    negative_ratio: float
    validation_fpr_target: float
    patience: int
    min_delta: float
    seed: int
    num_workers: int
    preload_train_val: bool
    smoke_test: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static GNN autoencoder baseline for Unraveled TDG snapshots"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "gnn_thesis",
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default="graph_timeline_merged_150000_final_split.csv",
    )
    parser.add_argument(
        "--graph-root",
        type=str,
        default="unraveled_graphs_150000_merged_captures_exact_57_72",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="anomaly_detection/results/base_gnn",
    )

    parser.add_argument(
        "--model",
        choices=["gcn", "sage", "gat"],
        default="gcn",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat-heads", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--validation-fpr-target", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-6)

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


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
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

    frame = frame[
        frame["split"].isin(EXPECTED_SPLIT_COUNTS)
    ].copy()

    frame["global_graph_index"] = pd.to_numeric(
        frame["global_graph_index"], errors="raise"
    ).astype(int)

    frame = frame.sort_values(
        "global_graph_index", kind="stable"
    ).reset_index(drop=True)

    actual_counts = frame["split"].value_counts().to_dict()
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Found: {actual_counts}"
        )

    for split in ("train", "validation"):
        split_types = set(frame.loc[frame["split"] == split, "graph_type"])
        benign_values = {"benign", "full_benign"}
        if not split_types or not split_types.issubset(benign_values):
            raise ValueError(
                f"{split} is not benign-only: {sorted(split_types)}"
            )

    return frame


def graph_path_from_row(row: pd.Series, graph_root: Path) -> Path:
    graph_type = str(row["graph_type"])
    if graph_type not in GRAPH_TYPE_TO_FOLDER:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return (
        graph_root
        / str(row["day_folder"])
        / str(row["capture"])
        / "tdg_graph"
        / GRAPH_TYPE_TO_FOLDER[graph_type]
        / f"{row['graph_id']}.pkl"
    )


def fit_feature_scaler(
    train_rows: pd.DataFrame,
    graph_root: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    print("\nCalcolo min-max sul solo training benigno...")

    feature_min = np.full(57, np.inf, dtype=np.float64)
    feature_max = np.full(57, -np.inf, dtype=np.float64)
    training_nodes = 0

    for _, row in tqdm(
        train_rows.iterrows(),
        total=len(train_rows),
        desc="Feature min-max",
    ):
        path = graph_path_from_row(row, graph_root)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 57:
            raise ValueError(
                f"Expected TDG features (N, 57), "
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

    # Per le feature costanti il denominatore viene reso pari a 1:
    # il valore normalizzato sarà quindi 0.
    feature_max[constant_mask] = feature_min[constant_mask] + 1.0

    print("Training nodes used for min-max:", training_nodes)
    print("Constant columns:", np.where(constant_mask)[0].tolist())

    return (
        feature_min.astype(np.float32),
        feature_max.astype(np.float32),
        training_nodes,
    )


class UnraveledTDGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        graph_root: Path,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        preload: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.graph_root = graph_root
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.cache: list[Data] | None = None

        if preload:
            self.cache = []
            for index in tqdm(
                range(len(self.rows)), desc="Preloading graphs"
            ):
                self.cache.append(self._load(index))

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, row_id: int) -> Data:
        row = self.rows.iloc[row_id]
        path = graph_path_from_row(row, self.graph_root)

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float32)
        adjacency = np.asarray(graph["adj"])
        node_y = np.asarray(graph["node_labels"], dtype=np.int64).reshape(-1)

        if x.ndim != 2 or x.shape[1] != 57:
            raise ValueError(f"Invalid feature shape {x.shape} in {path}")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Invalid adjacency shape {adjacency.shape} in {path}")
        if adjacency.shape[0] != x.shape[0] or node_y.shape[0] != x.shape[0]:
            raise ValueError(
                f"Node count mismatch in {path}: "
                f"x={x.shape}, adj={adjacency.shape}, labels={node_y.shape}"
            )

        x = (
            (x - self.feature_min)
            / (self.feature_max - self.feature_min)
        )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite normalized features in {path}")

        source, target = np.nonzero(adjacency)
        edge_index = np.vstack([source, target]).astype(np.int64, copy=False)

        graph_type = str(row["graph_type"])
        graph_y = GRAPH_TYPE_TO_LABEL[graph_type]

        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            node_y=torch.from_numpy(node_y),
            graph_y=torch.tensor([graph_y], dtype=torch.long),
            row_id=torch.tensor([row_id], dtype=torch.long),
            global_id=torch.tensor(
                [int(row["global_graph_index"])], dtype=torch.long
            ),
            num_nodes=int(x.shape[0]),
        )

    def __getitem__(self, row_id: int) -> Data:
        if self.cache is not None:
            return self.cache[row_id]
        return self._load(row_id)


def make_encoder_layer(
    model_name: str,
    in_dim: int,
    out_dim: int,
    gat_heads: int,
    final_layer: bool,
) -> nn.Module:
    if model_name == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
    if model_name == "sage":
        return SAGEConv(in_dim, out_dim)
    if model_name == "gat":
        if final_layer:
            return GATConv(
                in_dim,
                out_dim,
                heads=1,
                concat=False,
                add_self_loops=True,
            )
        if out_dim % gat_heads != 0:
            raise ValueError(
                "For GAT, hidden_dim must be divisible by gat_heads"
            )
        return GATConv(
            in_dim,
            out_dim // gat_heads,
            heads=gat_heads,
            concat=True,
            add_self_loops=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


class BaseGNNAutoencoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        gat_heads: int,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        encoder_layers: list[nn.Module] = []
        for layer_index in range(num_layers):
            in_dim = input_dim if layer_index == 0 else hidden_dim
            is_final = layer_index == num_layers - 1
            out_dim = embedding_dim if is_final else hidden_dim
            encoder_layers.append(
                make_encoder_layer(
                    model_name,
                    in_dim,
                    out_dim,
                    gat_heads,
                    final_layer=is_final,
                )
            )

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.attr_decoder_1 = GCNConv(
            embedding_dim,
            hidden_dim,
            add_self_loops=True,
            normalize=True,
        )
        self.attr_decoder_2 = GCNConv(
            hidden_dim,
            input_dim,
            add_self_loops=True,
            normalize=True,
        )
        self.structure_projection = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )
        self.dropout = float(dropout)
        self.embedding_dim = int(embedding_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_index, layer in enumerate(self.encoder_layers):
            h = layer(h, edge_index)
            if layer_index < len(self.encoder_layers) - 1:
                h = F.rrelu(h, training=self.training)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode_attributes(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        h = self.attr_decoder_1(z, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.attr_decoder_2(h, edge_index)

    def edge_logits(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.structure_projection(z)
        source, target = edge_index
        return (
            projected[source] * projected[target]
        ).sum(dim=1) / math.sqrt(self.embedding_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index)
        x_hat = self.decode_attributes(z, edge_index)
        return z, x_hat


def unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index

    source, target = edge_index
    mask = source < target
    return edge_index[:, mask]


def compute_node_scores(
    model: BaseGNNAutoencoder,
    batch: Data,
    alpha: float,
    negative_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, x_hat = model(batch.x, batch.edge_index)

    attr_error = torch.mean((batch.x - x_hat) ** 2, dim=1)

    structure_sum = torch.zeros(
        batch.num_nodes, dtype=z.dtype, device=z.device
    )
    structure_count = torch.zeros_like(structure_sum)

    ptr = batch.ptr

    for graph_position in range(batch.num_graphs):
        start = int(ptr[graph_position].item())
        end = int(ptr[graph_position + 1].item())
        num_nodes = end - start

        if num_nodes < 2:
            continue

        source = batch.edge_index[0]
        edge_mask = (source >= start) & (source < end)
        local_edges = batch.edge_index[:, edge_mask] - start

        no_self = local_edges[0] != local_edges[1]
        local_edges = local_edges[:, no_self]
        positive_edges = unique_undirected_edges(local_edges)

        if positive_edges.numel() == 0:
            continue

        num_positive = int(positive_edges.shape[1])
        num_negative = max(1, int(round(num_positive * negative_ratio)))

        negative_edges = negative_sampling(
            edge_index=local_edges,
            num_nodes=num_nodes,
            num_neg_samples=num_negative,
            method="sparse",
            force_undirected=True,
        )
        negative_edges = unique_undirected_edges(negative_edges)

        local_z = z[start:end]

        positive_logits = model.edge_logits(local_z, positive_edges)
        positive_losses = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
            reduction="none",
        )

        all_edges = [positive_edges]
        all_losses = [positive_losses]

        if negative_edges.numel() > 0:
            negative_logits = model.edge_logits(local_z, negative_edges)
            negative_losses = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
                reduction="none",
            )
            all_edges.append(negative_edges)
            all_losses.append(negative_losses)

        score_edges = torch.cat(all_edges, dim=1)
        score_losses = torch.cat(all_losses, dim=0)

        global_source = score_edges[0] + start
        global_target = score_edges[1] + start

        structure_sum.index_add_(0, global_source, score_losses)
        structure_sum.index_add_(0, global_target, score_losses)
        structure_count.index_add_(
            0, global_source, torch.ones_like(score_losses)
        )
        structure_count.index_add_(
            0, global_target, torch.ones_like(score_losses)
        )

    structure_error = structure_sum / structure_count.clamp_min(1.0)
    combined_score = alpha * attr_error + (1.0 - alpha) * structure_error

    return combined_score, attr_error, structure_error


def smoke_subset(
    rows: pd.DataFrame,
    split: str,
    size: int,
) -> pd.DataFrame:
    if split in {"train", "validation"}:
        return rows.head(size).reset_index(drop=True)

    benign_mask = rows["graph_type"].isin(["benign", "full_benign"])
    anomalous_mask = ~benign_mask

    benign = rows[benign_mask].head(size // 2)
    anomalous = rows[anomalous_mask].head(size - len(benign))

    return (
        pd.concat([benign, anomalous], ignore_index=True)
        .sort_values("global_graph_index", kind="stable")
        .reset_index(drop=True)
    )


def build_loader(
    dataset: UnraveledTDGDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
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
def evaluate_split(
    model: BaseGNNAutoencoder,
    loader: DataLoader,
    dataset: UnraveledTDGDataset,
    device: torch.device,
    alpha: float,
    negative_ratio: float,
    eval_seed: int,
    collect_nodes: bool,
) -> tuple[float, pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()
    torch.manual_seed(eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(eval_seed)

    total_score = 0.0
    total_nodes = 0
    graph_rows: list[dict[str, Any]] = []

    node_scores_parts: list[np.ndarray] = []
    node_attr_parts: list[np.ndarray] = []
    node_structure_parts: list[np.ndarray] = []
    node_labels_parts: list[np.ndarray] = []
    node_global_ids_parts: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluation"):
        batch = batch.to(device, non_blocking=True)
        node_scores, attr_error, structure_error = compute_node_scores(
            model, batch, alpha, negative_ratio
        )

        total_score += float(node_scores.sum().item())
        total_nodes += int(node_scores.numel())

        ptr = batch.ptr.detach().cpu().numpy()
        row_ids = batch.row_id.detach().cpu().numpy().reshape(-1)

        if collect_nodes:
            node_scores_cpu = node_scores.detach().cpu().numpy()
            node_attr_cpu = attr_error.detach().cpu().numpy()
            node_structure_cpu = structure_error.detach().cpu().numpy()
            node_labels_cpu = batch.node_y.detach().cpu().numpy()

            node_scores_parts.append(
                node_scores_cpu.astype(np.float32)
            )
            node_attr_parts.append(
                node_attr_cpu.astype(np.float32)
            )
            node_structure_parts.append(
                node_structure_cpu.astype(np.float32)
            )
            node_labels_parts.append(
                node_labels_cpu.astype(np.int8)
            )

        for graph_position in range(batch.num_graphs):
            start = int(ptr[graph_position])
            end = int(ptr[graph_position + 1])
            local_scores = node_scores[start:end]
            local_attr = attr_error[start:end]
            local_structure = structure_error[start:end]
            num_nodes = int(local_scores.numel())

            row_id = int(row_ids[graph_position])
            row = dataset.rows.iloc[row_id]
            global_id = int(row["global_graph_index"])

            graph_rows.append(
                {
                    "row_id": row_id,
                    "global_graph_index": global_id,
                    "split": str(row["split"]),
                    "day_folder": str(row["day_folder"]),
                    "capture": str(row["capture"]),
                    "graph_id": str(row["graph_id"]),
                    "graph_type": str(row["graph_type"]),
                    "num_nodes": num_nodes,
                    "score_mean": float(local_scores.mean().item()),
                    "attribute_error_mean": float(local_attr.mean().item()),
                    "structure_error_mean": float(
                        local_structure.mean().item()
                    ),
                }
            )

            if collect_nodes:
                node_global_ids_parts.append(
                    np.full(num_nodes, global_id, dtype=np.int64)
                )

    graph_frame = pd.DataFrame(graph_rows).sort_values(
        "global_graph_index", kind="stable"
    ).reset_index(drop=True)

    node_payload: dict[str, np.ndarray] = {}
    if collect_nodes:
        node_payload = {
            "scores": np.concatenate(node_scores_parts),
            "attribute_scores": np.concatenate(node_attr_parts),
            "structure_scores": np.concatenate(node_structure_parts),
            "labels": np.concatenate(node_labels_parts),
            "global_graph_index": np.concatenate(node_global_ids_parts),
        }

    return total_score / max(total_nodes, 1), graph_frame, node_payload


def select_threshold_for_fpr(
    benign_scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, float]:
    scores = np.asarray(benign_scores, dtype=np.float64).reshape(-1)

    if scores.size == 0:
        raise ValueError("Validation scores are empty")
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain non-finite values")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")

    # Largest integer number of false positives satisfying FP / N < target_fpr.
    max_false_positives = max(
        0,
        math.ceil(target_fpr * scores.size) - 1,
    )

    sorted_scores = np.sort(scores)[::-1]

    if max_false_positives == 0:
        threshold = float(sorted_scores[0])
    else:
        threshold = float(sorted_scores[max_false_positives])

    actual_fpr = float(np.mean(scores > threshold))

    if not actual_fpr < target_fpr:
        raise RuntimeError(
            f"Could not satisfy validation FPR constraint: "
            f"{actual_fpr} >= {target_fpr}"
        )

    return threshold, actual_fpr


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int64)

    unique_labels = np.unique(labels)
    roc_auc = (
        float(roc_auc_score(labels, scores))
        if unique_labels.size == 2
        else float("nan")
    )
    average_precision = (
        float(average_precision_score(labels, scores))
        if unique_labels.size == 2
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    precision = float(
        precision_score(labels, predictions, zero_division=0)
    )
    f1 = float(f1_score(labels, predictions, zero_division=0))

    return {
        "samples": int(labels.size),
        "benign": int((labels == 0).sum()),
        "anomalous": int((labels == 1).sum()),
        "threshold": float(threshold),
        "auc": roc_auc,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "tpr": tpr,
        "recall": tpr,
        "f1": f1,
        "fpr": fpr,
        "false_positive_rate": fpr,
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
    node_scores = np.asarray(node_payload["scores"])
    node_labels = np.asarray(node_payload["labels"])
    node_graph_ids = np.asarray(node_payload["global_graph_index"])

    rows: list[dict[str, Any]] = []

    for _, graph_row in graph_frame.iterrows():
        global_id = int(graph_row["global_graph_index"])
        mask = node_graph_ids == global_id

        if not np.any(mask):
            raise RuntimeError(
                f"No node scores found for graph {global_id}"
            )

        metrics = binary_metrics(
            node_labels[mask],
            node_scores[mask],
            threshold,
        )

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


def summarize_snapshot_metrics(
    snapshot_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_metrics.empty:
        return pd.DataFrame()

    split = str(snapshot_metrics.iloc[0]["split"])

    scenarios = [
        (
            "benign",
            snapshot_metrics["graph_type"].isin(
                ["benign", "full_benign"]
            ),
            ["fpr"],
        ),
        (
            "mixed",
            snapshot_metrics["graph_type"] == "mixed",
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
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "arguments": vars(args),
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
            "OCGNN_original": git_commit(
                project_root / "external_methods" / "OCGNN_original"
            ),
            "EvolveGCN_original": git_commit(
                project_root / "external_methods" / "EvolveGCN_original"
            ),
            "DOMINANT_original": git_commit(
                project_root / "external_methods" / "DOMINANT_original"
            ),
        },
    }

    serializable_args = {}
    for key, value in vars(args).items():
        serializable_args[key] = str(value) if isinstance(value, Path) else value
    metadata["arguments"] = serializable_args

    save_json(run_dir / "experiment_metadata.json", metadata)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        (run_dir / "pip_freeze.txt").write_text(
            result.stdout, encoding="utf-8"
        )
    except Exception as exc:
        (run_dir / "pip_freeze_error.txt").write_text(
            str(exc), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    if not 0.0 < args.validation_fpr_target < 1.0:
        raise ValueError("validation_fpr_target must be between 0 and 1")

    set_seed(args.seed)

    project_root = args.project_root.expanduser().resolve()
    timeline_path = project_root / args.timeline
    graph_root = project_root / args.graph_root
    output_root = project_root / args.output_root / f"{args.model}_tdg"

    run_prefix = "smoke" if args.smoke_test else "run"
    run_dir = output_root / (
        f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    if not graph_root.is_dir():
        raise FileNotFoundError(graph_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"BASE GNN AUTOENCODER - {args.model.upper()} - TDG")
    print("=" * 80)
    print("Project root:", project_root)
    print("Run directory:", run_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2,
            ),
            "GB",
        )

    timeline = load_timeline(timeline_path)
    rows_by_split = {
        split: timeline[timeline["split"] == split]
        .copy()
        .reset_index(drop=True)
        for split in EXPECTED_SPLIT_COUNTS
    }

    if args.smoke_test:
        rows_by_split["train"] = smoke_subset(
            rows_by_split["train"], "train", 64
        )
        rows_by_split["validation"] = smoke_subset(
            rows_by_split["validation"], "validation", 32
        )
        rows_by_split["test_week5"] = smoke_subset(
            rows_by_split["test_week5"], "test_week5", 64
        )
        rows_by_split["test_week6"] = smoke_subset(
            rows_by_split["test_week6"], "test_week6", 64
        )
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)

    config = ExperimentConfig(
        model=args.model,
        representation="TDG",
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        negative_ratio=args.negative_ratio,
        validation_fpr_target=args.validation_fpr_target,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        num_workers=args.num_workers,
        preload_train_val=args.preload_train_val,
        smoke_test=args.smoke_test,
    )

    print("\nGraphs used:")
    for split, rows in rows_by_split.items():
        print(
            f"{split:12s}: {len(rows):5d}",
            rows["graph_type"].value_counts().to_dict(),
        )

    split_summary = []
    for split, rows in rows_by_split.items():
        for graph_type, count in rows["graph_type"].value_counts().items():
            split_summary.append(
                {
                    "split": split,
                    "graph_type": graph_type,
                    "graphs": int(count),
                }
            )
    pd.DataFrame(split_summary).to_csv(
        run_dir / "split_summary.csv", index=False
    )

    feature_min, feature_max, scaler_nodes = fit_feature_scaler(
        rows_by_split["train"], graph_root
    )
    np.savez_compressed(
        run_dir / "feature_scaler.npz",
        min=feature_min,
        max=feature_max,
        training_nodes=scaler_nodes,
    )

    train_dataset = UnraveledTDGDataset(
        rows_by_split["train"],
        graph_root,
        feature_min,
        feature_max,
        preload=args.preload_train_val,
    )
    validation_dataset = UnraveledTDGDataset(
        rows_by_split["validation"],
        graph_root,
        feature_min,
        feature_max,
        preload=args.preload_train_val,
    )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device,
        args.seed,
    )
    validation_loader = build_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        args.seed,
    )

    model = BaseGNNAutoencoder(
        model_name=args.model,
        input_dim=57,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Trainable parameters:", parameter_count)

    script_path = Path(__file__).resolve()
    save_environment_metadata(
        run_dir,
        args,
        config,
        script_path,
        timeline_path,
        project_root,
        device,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    checkpoint_path = run_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []

    print("\nTraining started")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_total = 0.0
        train_nodes = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            node_scores, _, _ = compute_node_scores(
                model, batch, args.alpha, args.negative_ratio
            )
            loss = node_scores.mean()

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            nodes = int(node_scores.numel())
            train_total += float(loss.item()) * nodes
            train_nodes += nodes

        train_loss = train_total / max(train_nodes, 1)

        validation_loss, _, _ = evaluate_split(
            model=model,
            loader=validation_loader,
            dataset=validation_dataset,
            device=device,
            alpha=args.alpha,
            negative_ratio=args.negative_ratio,
            eval_seed=args.seed + 10_000,
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
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nBest epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["validation_loss"])

    validation_loss, validation_graphs, validation_nodes = evaluate_split(
        model=model,
        loader=validation_loader,
        dataset=validation_dataset,
        device=device,
        alpha=args.alpha,
        negative_ratio=args.negative_ratio,
        eval_seed=args.seed + 20_000,
        collect_nodes=True,
    )

    node_threshold, validation_fpr = select_threshold_for_fpr(
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
        "node_score": float(node_threshold),
    }
    save_json(run_dir / "thresholds.json", thresholds)

    print("Node threshold:", node_threshold)
    print("Target validation FPR:", args.validation_fpr_target)
    print("Actual validation FPR:", validation_fpr)

    weekly_metric_parts: list[pd.DataFrame] = []

    for split_index, split in enumerate(("test_week5", "test_week6")):
        print(f"\nEvaluating {split}")
        test_dataset = UnraveledTDGDataset(
            rows_by_split[split],
            graph_root,
            feature_min,
            feature_max,
            preload=False,
        )
        test_loader = build_loader(
            test_dataset,
            args.batch_size,
            False,
            args.num_workers,
            device,
            args.seed,
        )

        _, graph_scores, node_payload = evaluate_split(
            model=model,
            loader=test_loader,
            dataset=test_dataset,
            device=device,
            alpha=args.alpha,
            negative_ratio=args.negative_ratio,
            eval_seed=args.seed + 30_000 + split_index,
            collect_nodes=True,
        )

        snapshot_metrics = compute_snapshot_metrics(
            graph_scores,
            node_payload,
            node_threshold,
        )

        snapshot_metrics.to_csv(
            run_dir / f"{split}_snapshot_metrics.csv",
            index=False,
        )

        weekly_metrics = summarize_snapshot_metrics(
            snapshot_metrics
        )
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

    metrics_frame = pd.concat(
        weekly_metric_parts,
        ignore_index=True,
    )
    metrics_frame.to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    summary = {
        "method": args.model.upper(),
        "task": "node anomaly detection",
        "representation": "TDG",
        "evaluation": (
            "node-level metrics computed per snapshot and "
            "reported as mean and standard deviation"
        ),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "validation_loss_recomputed": float(validation_loss),
        "thresholds": thresholds,
        "parameters": parameter_count,
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
