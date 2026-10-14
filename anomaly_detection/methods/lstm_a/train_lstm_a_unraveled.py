
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch_geometric.nn import GCNConv
from tqdm import tqdm


EXPECTED_SPLIT_COUNTS = {
    "train": 4411,
    "validation": 1327,
    "test_week5": 3761,
    "test_week6": 4105,
}

GRAPH_TYPE_TO_FOLDER = {
    "benign": "full_benign",
    "full_benign": "full_benign",
    "mixed": "mixed",
}

EVAL_SPLITS = ("validation", "test_week5", "test_week6")
TEST_SPLITS = ("test_week5", "test_week6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSTM-A adaptation for the Unraveled TDG/e-TDG graphs."
    )

    parser.add_argument(
        "--representation",
        choices=["etdg", "tdg"],
        required=True,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "gnn_thesis",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=Path("graph_timeline_merged_150000_final_split.csv"),
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("unraveled_graphs_150000_merged_captures_exact_57_72"),
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--decoder-dropout", type=float, default=0.3)

    # The pub_iot 150k tasker uses max_time=600000 ms. With 150000 ms
    # snapshots this corresponds to four snapshots in a temporal window.
    parser.add_argument("--sequence-length", type=int, default=4)

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Attribute reconstruction weight, matching the 150k LSTM-A reference configuration.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Sampled negative structural pairs per positive/self-loop pair.",
    )
    parser.add_argument(
        "--validation-fpr-target",
        type=float,
        default=0.01,
        help="Choose the node threshold on benign validation scores with empirical FPR strictly below this target.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small subsets and at most two epochs.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _json_default(value: Any):
    """Convert non-standard objects used in experiment metadata to JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
            default=_json_default,
        )


def load_timeline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "day_folder",
        "capture",
        "graph_id",
        "graph_index",
        "global_graph_index",
        "graph_type",
        "split",
        "split_day_detail",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing timeline columns: {sorted(missing)}"
        )

    frame = frame[
        frame["split"].isin(EXPECTED_SPLIT_COUNTS)
    ].copy()

    frame["graph_index"] = pd.to_numeric(
        frame["graph_index"],
        errors="raise",
    ).astype(int)
    frame["global_graph_index"] = pd.to_numeric(
        frame["global_graph_index"],
        errors="raise",
    ).astype(int)

    frame = frame.sort_values(
        ["global_graph_index", "graph_index"],
        kind="stable",
    ).reset_index(drop=True)

    actual = frame["split"].value_counts().to_dict()
    if actual != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected final split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Found: {actual}"
        )

    for split in ("train", "validation"):
        types = set(frame.loc[frame["split"] == split, "graph_type"])
        if types != {"benign"}:
            raise ValueError(
                f"{split} must be benign-only, found: {sorted(types)}"
            )

    return frame


def graph_path_from_row(
    row: pd.Series,
    graph_root: Path,
    representation: str,
) -> Path:
    graph_type = str(row["graph_type"])
    if graph_type not in GRAPH_TYPE_TO_FOLDER:
        raise ValueError(f"Unknown graph type: {graph_type}")

    rep_folder = (
        "etdg_graph" if representation == "etdg" else "tdg_graph"
    )

    return (
        graph_root
        / str(row["day_folder"])
        / str(row["capture"])
        / rep_folder
        / GRAPH_TYPE_TO_FOLDER[graph_type]
        / f"{row['graph_id']}.pkl"
    )


def smoke_subset(
    rows: pd.DataFrame,
    split: str,
    size: int,
) -> pd.DataFrame:
    if split in {"train", "validation"}:
        return rows.head(size).reset_index(drop=True)

    benign = rows[rows["graph_type"] == "benign"].head(size // 2)
    mixed = rows[rows["graph_type"] == "mixed"].head(
        size - len(benign)
    )

    return (
        pd.concat([benign, mixed], ignore_index=True)
        .sort_values("global_graph_index", kind="stable")
        .reset_index(drop=True)
    )


def fit_minmax_scaler(
    train_rows: pd.DataFrame,
    graph_root: Path,
    representation: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    feature_min = np.full(feature_dim, np.inf, dtype=np.float64)
    feature_max = np.full(feature_dim, -np.inf, dtype=np.float64)
    node_count = 0

    rep_name = "e-TDG" if representation == "etdg" else "TDG"
    print(f"\nCalcolo min-max {rep_name} sul solo training benigno...")

    for _, row in tqdm(
        train_rows.iterrows(),
        total=len(train_rows),
        desc="Feature min-max",
    ):
        path = graph_path_from_row(
            row,
            graph_root,
            representation,
        )
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        x = np.asarray(graph["node_features"], dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != feature_dim:
            raise ValueError(
                f"Expected {rep_name} features (N, {feature_dim}), "
                f"found {x.shape} in {path}"
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

    scale = feature_max - feature_min
    constant_mask = scale < 1e-12
    scale[constant_mask] = 1.0

    print("Training nodes used for min-max:", node_count)
    print(
        "Constant feature columns:",
        np.where(constant_mask)[0].tolist(),
    )

    return (
        feature_min.astype(np.float32),
        scale.astype(np.float32),
        node_count,
    )


def canonical_node_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)

    return (str(value),)


@dataclass
class RawGraph:
    x: np.ndarray
    adj: np.ndarray
    labels: np.ndarray
    node_keys: list[tuple[str, ...]]
    row: pd.Series


@dataclass
class WindowData:
    xs: list[torch.Tensor]
    edge_indices: list[torch.Tensor]
    active_masks: list[torch.Tensor]
    target_indices: torch.Tensor
    target_labels: torch.Tensor
    target_x: torch.Tensor
    target_adj_local: np.ndarray
    target_local_to_union: np.ndarray
    target_row: pd.Series


class GraphLoader:
    def __init__(
        self,
        graph_root: Path,
        representation: str,
        feature_min: np.ndarray | None,
        feature_scale: np.ndarray | None,
    ) -> None:
        self.graph_root = graph_root
        self.representation = representation
        self.feature_min = feature_min
        self.feature_scale = feature_scale

    def load(self, row: pd.Series) -> RawGraph:
        path = graph_path_from_row(
            row,
            self.graph_root,
            self.representation,
        )
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            graph = pickle.load(handle)

        adj = np.asarray(graph["adj"])
        labels = np.asarray(graph["node_labels"], dtype=np.int64).reshape(-1)

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Invalid adjacency in {path}: {adj.shape}")

        n = int(adj.shape[0])

        if labels.shape[0] != n:
            raise ValueError(
                f"Label count mismatch in {path}: "
                f"{labels.shape[0]} vs {n}"
            )

        mapping = graph.get("node_id_to_port_ip")
        if not isinstance(mapping, dict):
            raise ValueError(
                f"node_id_to_port_ip missing/invalid in {path}"
            )

        node_keys = []
        for index in range(n):
            if index not in mapping:
                raise ValueError(
                    f"Missing node identity {index} in {path}"
                )
            node_keys.append(canonical_node_key(mapping[index]))

        if len(set(node_keys)) != n:
            raise ValueError(
                f"Duplicate node identities inside snapshot {path}"
            )

        expected_dim = 72 if self.representation == "etdg" else 57
        rep_name = "e-TDG" if self.representation == "etdg" else "TDG"

        x = np.asarray(graph["node_features"], dtype=np.float32)
        if x.ndim != 2 or x.shape != (n, expected_dim):
            raise ValueError(
                f"Expected {rep_name} features ({n}, {expected_dim}), "
                f"found {x.shape} in {path}"
            )
        if self.feature_min is None or self.feature_scale is None:
            raise RuntimeError(f"{rep_name} scaler is not initialized")

        x = (
            x - self.feature_min.reshape(1, -1)
        ) / self.feature_scale.reshape(1, -1)
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        return RawGraph(
            x=x,
            adj=adj,
            labels=labels,
            node_keys=node_keys,
            row=row,
        )


def build_history_indices(
    rows: pd.DataFrame,
    sequence_length: int,
) -> list[list[int]]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    rows = rows.reset_index(drop=True)

    history: list[list[int]] = [[] for _ in range(len(rows))]

    grouped = rows.groupby(
        ["day_folder", "capture"],
        sort=False,
        dropna=False,
    )

    for _, group in grouped:
        local_indices = group.index.to_list()
        local_indices.sort(
            key=lambda idx: int(rows.iloc[idx]["graph_index"])
        )

        position_by_index = {
            idx: pos for pos, idx in enumerate(local_indices)
        }

        for target_idx in local_indices:
            pos = position_by_index[target_idx]
            selected = [target_idx]

            current_graph_index = int(
                rows.iloc[target_idx]["graph_index"]
            )

            scan_pos = pos - 1
            expected = current_graph_index - 1

            while (
                scan_pos >= 0
                and len(selected) < sequence_length
            ):
                candidate_idx = local_indices[scan_pos]
                candidate_graph_index = int(
                    rows.iloc[candidate_idx]["graph_index"]
                )

                if candidate_graph_index != expected:
                    break

                selected.append(candidate_idx)
                expected -= 1
                scan_pos -= 1

            selected.reverse()
            history[target_idx] = selected

    return history


def align_window(
    raw_graphs: list[RawGraph],
) -> WindowData:
    union: dict[tuple[str, ...], int] = {}

    for graph in raw_graphs:
        for key in graph.node_keys:
            if key not in union:
                union[key] = len(union)

    union_n = len(union)
    feature_dim = int(raw_graphs[-1].x.shape[1])

    xs: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    active_masks: list[torch.Tensor] = []
    local_maps: list[np.ndarray] = []

    for graph in raw_graphs:
        local_to_union = np.asarray(
            [union[key] for key in graph.node_keys],
            dtype=np.int64,
        )
        local_maps.append(local_to_union)

        x_union = np.zeros(
            (union_n, feature_dim),
            dtype=np.float32,
        )
        x_union[local_to_union] = graph.x

        active = np.zeros(union_n, dtype=bool)
        active[local_to_union] = True

        src, dst = np.nonzero(graph.adj)
        mapped_src = local_to_union[src]
        mapped_dst = local_to_union[dst]

        edge_index = np.stack(
            [mapped_src, mapped_dst],
            axis=0,
        ).astype(np.int64)

        xs.append(torch.from_numpy(x_union))
        edge_indices.append(torch.from_numpy(edge_index).long())
        active_masks.append(torch.from_numpy(active))

    target = raw_graphs[-1]
    target_map = local_maps[-1]
    target_indices = torch.from_numpy(target_map).long()

    target_x = xs[-1]

    return WindowData(
        xs=xs,
        edge_indices=edge_indices,
        active_masks=active_masks,
        target_indices=target_indices,
        target_labels=torch.from_numpy(
            target.labels.astype(np.int64)
        ),
        target_x=target_x,
        target_adj_local=np.asarray(target.adj),
        target_local_to_union=target_map,
        target_row=target.row,
    )


class LSTMAutoencoder(nn.Module):
    """
    Intended LSTM-A architecture:
      GCN per snapshot -> LSTM across aligned node embeddings -> graph decoder.

    The pub_iot branch currently has the LSTM call commented in
    Sp_GCN_LSTM_A.forward(). Here the recurrent step is deliberately active,
    matching the named temporal method and the original EvolveGCN LSTM-A idea.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        decoder_hidden_dim: int,
        decoder_dropout: float,
    ) -> None:
        super().__init__()

        self.encoder_conv1 = GCNConv(
            input_dim,
            hidden_dim,
            bias=False,
            add_self_loops=True,
            normalize=True,
        )
        self.encoder_conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
            bias=False,
            add_self_loops=True,
            normalize=True,
        )

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
        )

        self.attr_conv1 = GCNConv(
            lstm_hidden_dim,
            decoder_hidden_dim,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )
        self.attr_conv2 = GCNConv(
            decoder_hidden_dim,
            input_dim,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )

        self.struct_conv = GCNConv(
            lstm_hidden_dim,
            decoder_hidden_dim,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )

        self.decoder_dropout = float(decoder_dropout)

    def encode(
        self,
        xs: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        sequence = []

        for x, edge_index in zip(xs, edge_indices):
            h = self.encoder_conv1(x, edge_index)
            h = F.rrelu(h, training=self.training)

            h = self.encoder_conv2(h, edge_index)
            h = F.rrelu(h, training=self.training)

            sequence.append(h)

        stacked = torch.stack(sequence, dim=0)
        out, _ = self.rnn(stacked)
        return out

    def decode_target(
        self,
        z_target: torch.Tensor,
        edge_index_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attr = self.attr_conv1(
            z_target,
            edge_index_target,
        )
        attr = F.relu(attr)
        attr = F.dropout(
            attr,
            p=self.decoder_dropout,
            training=self.training,
        )
        attr = self.attr_conv2(
            attr,
            edge_index_target,
        )

        struct = self.struct_conv(
            z_target,
            edge_index_target,
        )
        struct = F.dropout(
            struct,
            p=self.decoder_dropout,
            training=self.training,
        )

        return attr, struct

    def forward(
        self,
        xs: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_out = self.encode(xs, edge_indices)
        target_z = sequence_out[-1]

        attr_hat, struct_z = self.decode_target(
            target_z,
            edge_indices[-1],
        )
        return target_z, attr_hat, struct_z


def sample_structure_pairs(
    adj_local: np.ndarray,
    local_to_union: np.ndarray,
    negative_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(adj_local.shape[0])

    positive_local: list[tuple[int, int]] = []

    # Undirected canonical positives from the canonical adjacency.
    upper_src, upper_dst = np.nonzero(
        np.triu(adj_local, k=1)
    )

    for u, v in zip(upper_src.tolist(), upper_dst.tolist()):
        positive_local.append((u, v))

    # Include self-loops, as in the adapted structural reconstruction
    # protocol used for the previous reconstruction-based method.
    for node in range(n):
        positive_local.append((node, node))

    existing = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(*np.nonzero(adj_local))
        if int(u) != int(v)
    }

    rng = np.random.default_rng(seed)

    requested_negatives = int(
        round(len(positive_local) * negative_ratio)
    )

    negative_local: list[tuple[int, int]] = []

    if n > 1 and requested_negatives > 0:
        attempts = 0
        max_attempts = max(1000, requested_negatives * 30)
        negative_seen: set[tuple[int, int]] = set()

        while (
            len(negative_local) < requested_negatives
            and attempts < max_attempts
        ):
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            attempts += 1

            if u == v:
                continue

            pair = (min(u, v), max(u, v))
            if pair in existing or pair in negative_seen:
                continue

            negative_seen.add(pair)
            negative_local.append(pair)

    all_pairs = positive_local + negative_local
    targets = np.concatenate(
        [
            np.ones(len(positive_local), dtype=np.float32),
            np.zeros(len(negative_local), dtype=np.float32),
        ],
        axis=0,
    )

    if not all_pairs:
        # This should not occur because self-loops are always included.
        all_pairs = [(0, 0)]
        targets = np.ones(1, dtype=np.float32)

    union_pairs = np.asarray(
        [
            (
                int(local_to_union[u]),
                int(local_to_union[v]),
            )
            for u, v in all_pairs
        ],
        dtype=np.int64,
    )

    return (
        torch.from_numpy(union_pairs).long(),
        torch.from_numpy(targets),
    )


def structural_node_error(
    struct_z: torch.Tensor,
    target_indices: torch.Tensor,
    pairs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    device = struct_z.device

    pairs = pairs.to(device)
    targets = targets.to(device)

    u = pairs[:, 0]
    v = pairs[:, 1]

    predictions = torch.sum(
        struct_z[u] * struct_z[v],
        dim=1,
    )

    errors = (predictions - targets) ** 2

    union_n = int(struct_z.shape[0])

    error_sum = torch.zeros(
        union_n,
        dtype=struct_z.dtype,
        device=device,
    )
    count = torch.zeros(
        union_n,
        dtype=struct_z.dtype,
        device=device,
    )

    error_sum.index_add_(0, u, errors)
    count.index_add_(
        0,
        u,
        torch.ones_like(errors),
    )

    non_self = u != v
    if torch.any(non_self):
        error_sum.index_add_(
            0,
            v[non_self],
            errors[non_self],
        )
        count.index_add_(
            0,
            v[non_self],
            torch.ones_like(errors[non_self]),
        )

    mean_error = error_sum / torch.clamp(count, min=1.0)
    node_error = torch.sqrt(
        torch.clamp(mean_error, min=0.0)
    )

    return node_error[target_indices.to(device)]


def compute_target_scores(
    model: LSTMAutoencoder,
    window: WindowData,
    device: torch.device,
    alpha: float,
    negative_ratio: float,
    structure_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [tensor.to(device) for tensor in window.xs]
    edge_indices = [
        tensor.to(device) for tensor in window.edge_indices
    ]

    _, attr_hat, struct_z = model(xs, edge_indices)

    target_indices = window.target_indices.to(device)
    target_x = window.target_x.to(device)[target_indices]

    attr_error = torch.sqrt(
        torch.sum(
            (target_x - attr_hat[target_indices]) ** 2,
            dim=1,
        )
    )

    pairs, targets = sample_structure_pairs(
        adj_local=window.target_adj_local,
        local_to_union=window.target_local_to_union,
        negative_ratio=negative_ratio,
        seed=structure_seed,
    )

    struct_error = structural_node_error(
        struct_z=struct_z,
        target_indices=target_indices,
        pairs=pairs,
        targets=targets,
    )

    score = alpha * attr_error + (1.0 - alpha) * struct_error

    return score, attr_error, struct_error


def build_windows(
    rows: pd.DataFrame,
    sequence_length: int,
) -> tuple[pd.DataFrame, list[list[int]]]:
    rows = rows.sort_values(
        ["global_graph_index", "graph_index"],
        kind="stable",
    ).reset_index(drop=True)

    history = build_history_indices(
        rows,
        sequence_length,
    )

    return rows, history


def load_window(
    rows: pd.DataFrame,
    history_indices: list[int],
    graph_loader: GraphLoader,
) -> WindowData:
    raw_graphs = [
        graph_loader.load(rows.iloc[index])
        for index in history_indices
    ]
    return align_window(raw_graphs)


def train_epoch(
    model: LSTMAutoencoder,
    optimizer: torch.optim.Optimizer,
    rows: pd.DataFrame,
    histories: list[list[int]],
    graph_loader: GraphLoader,
    device: torch.device,
    alpha: float,
    negative_ratio: float,
    batch_size: int,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    model.train()

    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed + epoch)
    rng.shuffle(indices)

    total_loss = 0.0
    total_attr = 0.0
    total_struct = 0.0
    samples = 0

    for start in tqdm(
        range(0, len(indices), batch_size),
        desc=f"Train epoch {epoch:03d}",
        leave=False,
    ):
        batch_indices = indices[start : start + batch_size]

        optimizer.zero_grad(set_to_none=True)

        for target_idx in batch_indices:
            target_idx = int(target_idx)

            window = load_window(
                rows,
                histories[target_idx],
                graph_loader,
            )

            graph_seed = (
                seed
                + epoch * 1_000_003
                + int(
                    rows.iloc[target_idx]["global_graph_index"]
                )
            )

            score, attr_error, struct_error = compute_target_scores(
                model=model,
                window=window,
                device=device,
                alpha=alpha,
                negative_ratio=negative_ratio,
                structure_seed=graph_seed,
            )

            loss = score.mean()

            (loss / len(batch_indices)).backward()

            total_loss += float(loss.detach().cpu())
            total_attr += float(attr_error.mean().detach().cpu())
            total_struct += float(struct_error.mean().detach().cpu())
            samples += 1

            del window, score, attr_error, struct_error, loss

        optimizer.step()

    return {
        "loss": total_loss / max(samples, 1),
        "attribute_error": total_attr / max(samples, 1),
        "structure_error": total_struct / max(samples, 1),
    }


@torch.no_grad()
def evaluate_loss(
    model: LSTMAutoencoder,
    rows: pd.DataFrame,
    histories: list[list[int]],
    graph_loader: GraphLoader,
    device: torch.device,
    alpha: float,
    negative_ratio: float,
    seed: int,
    desc: str,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_attr = 0.0
    total_struct = 0.0

    for target_idx in tqdm(
        range(len(rows)),
        desc=desc,
        leave=False,
    ):
        window = load_window(
            rows,
            histories[target_idx],
            graph_loader,
        )

        graph_seed = (
            seed
            + int(rows.iloc[target_idx]["global_graph_index"])
        )

        score, attr_error, struct_error = compute_target_scores(
            model=model,
            window=window,
            device=device,
            alpha=alpha,
            negative_ratio=negative_ratio,
            structure_seed=graph_seed,
        )

        total_loss += float(score.mean().cpu())
        total_attr += float(attr_error.mean().cpu())
        total_struct += float(struct_error.mean().cpu())

    n = max(len(rows), 1)
    return {
        "loss": total_loss / n,
        "attribute_error": total_attr / n,
        "structure_error": total_struct / n,
    }


@torch.no_grad()
def evaluate_scores(
    model: LSTMAutoencoder,
    rows: pd.DataFrame,
    histories: list[list[int]],
    graph_loader: GraphLoader,
    device: torch.device,
    alpha: float,
    negative_ratio: float,
    seed: int,
    split: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    model.eval()

    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    graph_ids_all: list[np.ndarray] = []
    graph_rows: list[dict[str, Any]] = []

    for target_idx in tqdm(
        range(len(rows)),
        desc=f"Scoring {split}",
    ):
        row = rows.iloc[target_idx]

        window = load_window(
            rows,
            histories[target_idx],
            graph_loader,
        )

        graph_seed = seed + int(row["global_graph_index"])

        score, _, _ = compute_target_scores(
            model=model,
            window=window,
            device=device,
            alpha=alpha,
            negative_ratio=negative_ratio,
            structure_seed=graph_seed,
        )

        scores = score.detach().cpu().numpy().astype(np.float64)
        labels = (
            window.target_labels
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        global_id = int(row["global_graph_index"])

        labels_all.append(labels)
        scores_all.append(scores)
        graph_ids_all.append(
            np.full(labels.shape[0], global_id, dtype=np.int64)
        )

        top_count = max(
            1,
            int(math.ceil(0.01 * len(scores))),
        )
        top_values = np.partition(
            scores,
            len(scores) - top_count,
        )[-top_count:]

        graph_rows.append({
            "sample_index": target_idx,
            "global_graph_index": global_id,
            "graph_index": int(row["graph_index"]),
            "split": str(row["split"]),
            "split_day_detail": str(row["split_day_detail"]),
            "day_folder": str(row["day_folder"]),
            "capture": str(row["capture"]),
            "graph_id": str(row["graph_id"]),
            "graph_type": str(row["graph_type"]),
            "num_nodes": int(len(scores)),
            "sequence_length_used": int(
                len(histories[target_idx])
            ),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_p95": float(np.quantile(scores, 0.95)),
            "score_top1pct_mean": float(np.mean(top_values)),
        })

    graph_frame = (
        pd.DataFrame(graph_rows)
        .sort_values(
            "global_graph_index",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    payload = {
        "labels": np.concatenate(labels_all),
        "scores": np.concatenate(scores_all),
        "global_graph_index": np.concatenate(graph_ids_all),
    }

    return graph_frame, payload


def select_threshold_for_fpr(
    benign_scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, float, int]:
    scores = np.asarray(benign_scores, dtype=np.float64).reshape(-1)

    if scores.size == 0:
        raise ValueError("Validation scores are empty")
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain non-finite values")

    max_false_positives = max(
        0,
        math.ceil(target_fpr * scores.size) - 1,
    )
    sorted_scores = np.sort(scores)[::-1]

    if max_false_positives == 0:
        threshold = float(sorted_scores[0])
    else:
        threshold = float(sorted_scores[max_false_positives])

    actual_fp = int(np.sum(scores > threshold))
    actual_fpr = float(actual_fp / scores.size)

    if not actual_fpr < target_fpr:
        raise RuntimeError(
            "Could not satisfy validation FPR constraint: "
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
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        float(fp / (fp + tn))
        if (fp + tn) > 0
        else float("nan")
    )
    tpr = (
        float(tp / (tp + fn))
        if (tp + fn) > 0
        else float("nan")
    )
    precision = float(
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )
    f1 = float(
        f1_score(
            labels,
            predictions,
            zero_division=0,
        )
    )
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
            raise RuntimeError(
                f"No node scores found for graph {global_id}"
            )

        metrics = binary_metrics(
            labels[mask],
            scores[mask],
            threshold,
        )
        metrics.update({
            "split": str(graph_row["split"]),
            "global_graph_index": global_id,
            "day_folder": str(graph_row["day_folder"]),
            "capture": str(graph_row["capture"]),
            "graph_id": str(graph_row["graph_id"]),
            "graph_type": str(graph_row["graph_type"]),
        })
        rows.append(metrics)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "global_graph_index",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def summarize_snapshot_metrics(
    snapshot_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_metrics.empty:
        return pd.DataFrame()

    split = str(snapshot_metrics.iloc[0]["split"])

    scenarios = [
        (
            "benign",
            snapshot_metrics["graph_type"] == "benign",
            ["fpr"],
        ),
        (
            "mixed",
            snapshot_metrics["graph_type"] == "mixed",
            ["fpr", "tpr", "precision", "f1", "auc"],
        ),
    ]

    rows: list[dict[str, Any]] = []

    for scenario, mask, metric_names in scenarios:
        subset = snapshot_metrics.loc[mask]

        if subset.empty:
            continue

        for metric_name in metric_names:
            values = pd.to_numeric(
                subset[metric_name],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            finite = values[np.isfinite(values)]

            rows.append({
                "split": split,
                "graph_type": scenario,
                "metric": metric_name.upper(),
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
            })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not 0.0 < args.validation_fpr_target < 1.0:
        raise ValueError(
            "validation_fpr_target must be in (0, 1)"
        )
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0")

    project_root = args.project_root.resolve()
    timeline_path = project_root / args.timeline
    graph_root = project_root / args.graph_root

    output_root = (
        project_root
        / "anomaly_detection/results/lstm_a"
        / f"lstm_a_{args.representation}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"smoke_{timestamp}"
        if args.smoke_test
        else f"run_{timestamp}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 80)
    print(f"LSTM-A - Unraveled - {args.representation.upper()}")
    print("=" * 80)
    print("Project root:", project_root)
    print("Run directory:", run_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "Visible GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    timeline = load_timeline(timeline_path)

    rows_by_split = {
        split: timeline[
            timeline["split"] == split
        ].copy().reset_index(drop=True)
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

    print("\nGraphs used:")
    for split, rows in rows_by_split.items():
        print(
            f"{split:12s}: {len(rows):5d}",
            rows["graph_type"].value_counts().to_dict(),
        )

    input_dim = 72 if args.representation == "etdg" else 57

    (
        feature_min,
        feature_scale,
        scaler_nodes,
    ) = fit_minmax_scaler(
        rows_by_split["train"],
        graph_root,
        args.representation,
        input_dim,
    )

    np.savez(
        run_dir / "feature_scaler.npz",
        minimum=feature_min,
        scale=feature_scale,
        training_nodes=scaler_nodes,
        kind="training_minmax",
        representation=args.representation,
        feature_dim=input_dim,
    )

    alpha = args.alpha

    graph_loader = GraphLoader(
        graph_root=graph_root,
        representation=args.representation,
        feature_min=feature_min,
        feature_scale=feature_scale,
    )

    window_data: dict[str, tuple[pd.DataFrame, list[list[int]]]] = {}

    for split, rows in rows_by_split.items():
        ordered_rows, histories = build_windows(
            rows,
            args.sequence_length,
        )
        window_data[split] = (ordered_rows, histories)

        lengths = np.asarray(
            [len(item) for item in histories],
            dtype=np.int64,
        )

        print(
            f"{split:12s} temporal windows: "
            f"min={lengths.min()} "
            f"mean={lengths.mean():.3f} "
            f"max={lengths.max()}"
        )

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
    ).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Trainable parameters:", parameter_count)
    print("Effective alpha:", alpha)
    print("Temporal sequence length:", args.sequence_length)
    print(
        "RNN note: recurrent LSTM step is ACTIVE. "
        "The current pub_iot branch has that call commented."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update({
        "project_root": str(project_root),
        "timeline_path": str(timeline_path),
        "graph_root": str(graph_root),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "device": str(device),
        "trainable_parameters": parameter_count,
        "input_dim": input_dim,
        "effective_alpha": alpha,
        "feature_scaler": "training_minmax",
        "temporal_window_policy": (
            "causal: previous consecutive snapshots plus current target, "
            "never crossing capture or split boundaries"
        ),
        "published_pub_iot_note": (
            "Sp_GCN_LSTM_A in the currently available pub_iot branch "
            "has the recurrent call commented. This adaptation re-enables "
            "the LSTM so that LSTM-A is genuinely temporal."
        ),
        "structure_reconstruction": (
            "sampled undirected positive edges + self-loops + sampled "
            "non-edges; no dense NxN reconstruction matrix"
        ),
    })
    save_json(run_dir / "config.json", config)

    history_rows = []
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    train_rows, train_histories = window_data["train"]
    val_rows, val_histories = window_data["validation"]

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model=model,
            optimizer=optimizer,
            rows=train_rows,
            histories=train_histories,
            graph_loader=graph_loader,
            device=device,
            alpha=alpha,
            negative_ratio=args.negative_ratio,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
        )

        val_stats = evaluate_loss(
            model=model,
            rows=val_rows,
            histories=val_histories,
            graph_loader=graph_loader,
            device=device,
            alpha=alpha,
            negative_ratio=args.negative_ratio,
            seed=args.seed + 10_000_000,
            desc=f"Validation epoch {epoch:03d}",
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_attribute_error": train_stats[
                "attribute_error"
            ],
            "train_structure_error": train_stats[
                "structure_error"
            ],
            "validation_loss": val_stats["loss"],
            "validation_attribute_error": val_stats[
                "attribute_error"
            ],
            "validation_structure_error": val_stats[
                "structure_error"
            ],
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_stats['loss']:.6f} | "
            f"val={val_stats['loss']:.6f}"
        )

        improved = (
            val_stats["loss"]
            < best_val - args.min_delta
        )

        if improved:
            best_val = float(val_stats["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val,
                    "config": config,
                },
                run_dir / "best_checkpoint.pt",
            )

            print(
                f"  -> Best validation loss: {best_val:.6f}"
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}."
                )
                break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nBest epoch:", best_epoch)
    print("Best validation loss:", best_val)

    # Validation scores establish the node threshold.
    val_graphs, val_payload = evaluate_scores(
        model=model,
        rows=val_rows,
        histories=val_histories,
        graph_loader=graph_loader,
        device=device,
        alpha=alpha,
        negative_ratio=args.negative_ratio,
        seed=args.seed + 20_000_000,
        split="validation",
    )

    if np.any(val_payload["labels"] != 0):
        raise RuntimeError(
            "Validation must be benign-only for threshold selection"
        )

    (
        node_threshold,
        actual_validation_fpr,
        actual_validation_fp,
    ) = select_threshold_for_fpr(
        val_payload["scores"],
        args.validation_fpr_target,
    )

    thresholds = {
        "criterion": "validation_fpr_below_target",
        "target_fpr": float(args.validation_fpr_target),
        "actual_validation_fpr": float(actual_validation_fpr),
        "actual_validation_fp": int(actual_validation_fp),
        "validation_nodes": int(val_payload["scores"].size),
        "node_score": float(node_threshold),
    }
    save_json(run_dir / "thresholds.json", thresholds)

    np.savez_compressed(
        run_dir / "validation_node_scores.npz",
        **val_payload,
    )
    val_graphs.to_csv(
        run_dir / "validation_graph_scores.csv",
        index=False,
    )

    print("\nNode threshold:", node_threshold)
    print("Target validation FPR:", args.validation_fpr_target)
    print("Actual validation FPR:", actual_validation_fpr)

    metric_frames: list[pd.DataFrame] = []

    for split in TEST_SPLITS:
        rows, histories = window_data[split]

        graph_frame, node_payload = evaluate_scores(
            model=model,
            rows=rows,
            histories=histories,
            graph_loader=graph_loader,
            device=device,
            alpha=alpha,
            negative_ratio=args.negative_ratio,
            seed=args.seed + 30_000_000,
            split=split,
        )

        np.savez_compressed(
            run_dir / f"{split}_node_scores.npz",
            **node_payload,
        )
        graph_frame.to_csv(
            run_dir / f"{split}_graph_scores.csv",
            index=False,
        )

        snapshot_metrics = compute_snapshot_metrics(
            graph_frame,
            node_payload,
            node_threshold,
        )
        snapshot_metrics.to_csv(
            run_dir / f"{split}_snapshot_metrics.csv",
            index=False,
        )

        summary_frame = summarize_snapshot_metrics(
            snapshot_metrics
        )
        metric_frames.append(summary_frame)

        print(
            f"\nSnapshot-averaged node metrics - {split}:"
        )
        print(summary_frame.to_string(index=False))

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )
    metrics.to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    summary = {
        "method": "LSTM-A",
        "task": "node anomaly detection",
        "representation": (
            "e-TDG"
            if args.representation == "etdg"
            else "TDG"
        ),
        "feature_dim": input_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "trainable_parameters": parameter_count,
        "effective_alpha": alpha,
        "sequence_length": args.sequence_length,
        "thresholds": thresholds,
        "evaluation": (
            "node-level metrics computed per snapshot and "
            "reported as mean and standard deviation"
        ),
        "metrics": metrics.to_dict(orient="records"),
    }
    save_json(run_dir / "summary.json", summary)

    print("\n" + "=" * 80)
    print("LSTM-A RUN COMPLETED")
    print("=" * 80)
    print(metrics.to_string(index=False))
    print("\nResults saved in:")
    print(run_dir)


if __name__ == "__main__":
    main()
