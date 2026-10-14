from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from tqdm import tqdm

EXPECTED_SPLIT_COUNTS = {
    "train": 4411,
    "validation": 1327,
    "test_week5": 3761,
    "test_week6": 4105,
}
GRAPH_TYPE_TO_FOLDER = {"benign": "full_benign", "full_benign": "full_benign", "mixed": "mixed"}
TEST_SPLITS = ("test_week5", "test_week6")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EGCN-O adaptation for exact Unraveled TDG/e-TDG graphs")
    p.add_argument("--representation", choices=["tdg", "etdg"], required=True)
    p.add_argument("--project-root", type=Path, default=Path.home() / "gnn_thesis")
    p.add_argument("--timeline", type=Path, default=Path("graph_timeline_merged_150000_final_split.csv"))
    p.add_argument("--graph-root", type=Path, default=Path("unraveled_graphs_150000_merged_captures_exact_57_72"))
    p.add_argument("--output-root", type=Path, default=Path("anomaly_detection/results/egcn_o"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--decoder-hidden-dim", type=int, default=64)
    p.add_argument("--decoder-dropout", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--validation-fpr-target", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_json(path: Path, payload: Any) -> None:
    def default(v: Any):
        if isinstance(v, Path): return str(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, torch.device): return str(v)
        raise TypeError(type(v).__name__)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True, default=default)


def load_timeline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"day_folder", "capture", "graph_id", "graph_index", "global_graph_index", "graph_type", "split", "split_day_detail"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timeline columns: {sorted(missing)}")
    frame = frame[frame["split"].isin(EXPECTED_SPLIT_COUNTS)].copy()
    frame["graph_index"] = pd.to_numeric(frame["graph_index"], errors="raise").astype(int)
    frame["global_graph_index"] = pd.to_numeric(frame["global_graph_index"], errors="raise").astype(int)
    frame = frame.sort_values(["global_graph_index", "graph_index"], kind="stable").reset_index(drop=True)
    actual = frame["split"].value_counts().to_dict()
    if actual != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts. Expected {EXPECTED_SPLIT_COUNTS}, found {actual}")
    for split in ("train", "validation"):
        types = set(frame.loc[frame["split"] == split, "graph_type"])
        if types != {"benign"}:
            raise ValueError(f"{split} must be benign-only, found {sorted(types)}")
    return frame


def graph_path_from_row(row: pd.Series, graph_root: Path, representation: str) -> Path:
    graph_type = str(row["graph_type"])
    if graph_type not in GRAPH_TYPE_TO_FOLDER:
        raise ValueError(f"Unknown graph type: {graph_type}")
    rep_folder = "tdg_graph" if representation == "tdg" else "etdg_graph"
    return graph_root / str(row["day_folder"]) / str(row["capture"]) / rep_folder / GRAPH_TYPE_TO_FOLDER[graph_type] / f"{row['graph_id']}.pkl"


def expected_feature_dim(representation: str) -> int:
    return 57 if representation == "tdg" else 72


def smoke_subset(rows: pd.DataFrame, split: str, size: int) -> pd.DataFrame:
    if split in {"train", "validation"}:
        return rows.head(size).reset_index(drop=True)
    benign = rows[rows["graph_type"] == "benign"].head(size // 2)
    mixed = rows[rows["graph_type"] == "mixed"].head(size - len(benign))
    return pd.concat([benign, mixed], ignore_index=True).sort_values("global_graph_index", kind="stable").reset_index(drop=True)


def fit_minmax_scaler(train_rows: pd.DataFrame, graph_root: Path, representation: str) -> tuple[np.ndarray, np.ndarray, int]:
    dim = expected_feature_dim(representation)
    fmin = np.full(dim, np.inf, dtype=np.float64)
    fmax = np.full(dim, -np.inf, dtype=np.float64)
    nodes = 0
    print(f"\nCalcolo min-max {representation.upper()} sul solo training benigno...")
    for _, row in tqdm(train_rows.iterrows(), total=len(train_rows), desc="Feature min-max"):
        path = graph_path_from_row(row, graph_root, representation)
        if not path.is_file(): raise FileNotFoundError(path)
        with path.open("rb") as f: graph = pickle.load(f)
        x = np.asarray(graph["node_features"], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != dim: raise ValueError(f"Expected (N,{dim}), found {x.shape} in {path}")
        if not np.isfinite(x).all(): raise ValueError(f"Non-finite features in {path}")
        if x.shape[0] == 0: continue
        fmin = np.minimum(fmin, x.min(axis=0)); fmax = np.maximum(fmax, x.max(axis=0)); nodes += int(x.shape[0])
    if nodes == 0: raise RuntimeError("No training nodes found for min-max scaling")
    raw = fmax - fmin
    constant = raw < 1e-12
    scale = raw.copy(); scale[constant] = 1.0
    print("Training nodes used for min-max:", nodes)
    print("Constant feature columns:", np.where(constant)[0].tolist())
    return fmin.astype(np.float32), scale.astype(np.float32), nodes


def canonical_node_key(v: Any) -> tuple[str, ...]:
    if isinstance(v, np.ndarray): v = v.tolist()
    if isinstance(v, (tuple, list)): return tuple(str(x) for x in v)
    return (str(v),)


@dataclass
class RawGraph:
    x: np.ndarray
    adj: np.ndarray
    labels: np.ndarray
    node_keys: list[tuple[str, ...]]
    row: pd.Series


@dataclass
class WindowStep:
    x: torch.Tensor
    adj_norm: torch.Tensor
    topk_mask: torch.Tensor
    active_indices: torch.Tensor
    x_local: torch.Tensor
    adj_local: torch.Tensor
    labels_local: torch.Tensor
    row: pd.Series


@dataclass
class WindowData:
    steps: list[WindowStep]


class GraphLoader:
    def __init__(self, graph_root: Path, representation: str, fmin: np.ndarray, fscale: np.ndarray) -> None:
        self.graph_root = graph_root
        self.representation = representation
        self.fmin = fmin
        self.fscale = fscale
        self.dim = expected_feature_dim(representation)

    def load(self, row: pd.Series) -> RawGraph:
        path = graph_path_from_row(row, self.graph_root, self.representation)
        if not path.is_file(): raise FileNotFoundError(path)
        with path.open("rb") as f: graph = pickle.load(f)
        adj = np.asarray(graph["adj"])
        labels = np.asarray(graph["node_labels"], dtype=np.int64).reshape(-1)
        x = np.asarray(graph["node_features"], dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]: raise ValueError(f"Invalid adjacency {adj.shape} in {path}")
        n = int(adj.shape[0])
        if labels.shape[0] != n: raise ValueError(f"Label count mismatch in {path}")
        if x.shape != (n, self.dim): raise ValueError(f"Expected ({n},{self.dim}), found {x.shape} in {path}")
        x = (x - self.fmin.reshape(1, -1)) / self.fscale.reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mapping = graph.get("node_id_to_port_ip")
        if not isinstance(mapping, dict): raise ValueError(f"node_id_to_port_ip missing/invalid in {path}")
        keys = []
        for i in range(n):
            if i not in mapping: raise ValueError(f"Missing node identity {i} in {path}")
            keys.append(canonical_node_key(mapping[i]))
        if len(set(keys)) != n: raise ValueError(f"Duplicate node identities in {path}")
        return RawGraph(x=x, adj=adj, labels=labels, node_keys=keys, row=row)


def build_history_indices(rows: pd.DataFrame, sequence_length: int) -> list[list[int]]:
    if sequence_length < 1: raise ValueError("sequence_length must be >= 1")
    rows = rows.reset_index(drop=True)
    history = [[] for _ in range(len(rows))]
    for _, group in rows.groupby(["day_folder", "capture"], sort=False, dropna=False):
        inds = group.index.to_list(); inds.sort(key=lambda i: int(rows.iloc[i]["graph_index"]))
        pos = {idx: p for p, idx in enumerate(inds)}
        for target in inds:
            p = pos[target]; chosen = [target]; expected = int(rows.iloc[target]["graph_index"]) - 1; scan = p - 1
            while scan >= 0 and len(chosen) < sequence_length:
                cand = inds[scan]; gi = int(rows.iloc[cand]["graph_index"])
                if gi != expected: break
                chosen.append(cand); expected -= 1; scan -= 1
            chosen.reverse(); history[target] = chosen
    return history


def normalized_sparse_adj(adj: np.ndarray, local_to_union: np.ndarray, union_n: int) -> torch.Tensor:
    src, dst = np.nonzero(adj)
    src = local_to_union[src].astype(np.int64); dst = local_to_union[dst].astype(np.int64)
    loops = np.arange(union_n, dtype=np.int64)
    idx = torch.from_numpy(np.vstack([np.concatenate([src, loops]), np.concatenate([dst, loops])])).long()
    vals = torch.ones(idx.shape[1], dtype=torch.float32)
    sp = torch.sparse_coo_tensor(idx, vals, (union_n, union_n)).coalesce()
    idx = sp.indices(); vals = sp.values()
    degree = torch.zeros(union_n, dtype=torch.float32); degree.index_add_(0, idx[0], vals)
    norm_vals = vals * torch.pow(degree[idx[0]] * degree[idx[1]], -0.5)
    norm_vals = torch.nan_to_num(norm_vals, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sparse_coo_tensor(idx, norm_vals, (union_n, union_n)).coalesce()


def align_window(raw_graphs: list[RawGraph]) -> WindowData:
    union: dict[tuple[str, ...], int] = {}
    for g in raw_graphs:
        for key in g.node_keys:
            if key not in union: union[key] = len(union)
    union_n = len(union); dim = int(raw_graphs[-1].x.shape[1]); steps = []
    for g in raw_graphs:
        local_to_union = np.asarray([union[k] for k in g.node_keys], dtype=np.int64)
        x_union = np.zeros((union_n, dim), dtype=np.float32); x_union[local_to_union] = g.x
        mask = np.full((union_n, 1), -np.inf, dtype=np.float32); mask[local_to_union, 0] = 1.0
        steps.append(WindowStep(
            x=torch.from_numpy(x_union),
            adj_norm=normalized_sparse_adj(g.adj, local_to_union, union_n),
            topk_mask=torch.from_numpy(mask),
            active_indices=torch.from_numpy(local_to_union).long(),
            x_local=torch.from_numpy(g.x.astype(np.float32)),
            adj_local=torch.from_numpy(np.asarray(g.adj, dtype=np.float32)),
            labels_local=torch.from_numpy(g.labels.astype(np.int64)),
            row=g.row,
        ))
    return WindowData(steps=steps)


def load_window(rows: pd.DataFrame, history: list[int], loader: GraphLoader) -> WindowData:
    return align_window([loader.load(rows.iloc[i]) for i in history])


class MatGRUGate(nn.Module):
    def __init__(self, rows: int, cols: int, activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.empty(rows, rows))
        self.U = nn.Parameter(torch.empty(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, cols))
        self._reset(self.W)
        self._reset(self.U)

    @staticmethod
    def _reset(t: torch.Tensor) -> None:
        s = 1.0 / math.sqrt(t.size(1))
        nn.init.uniform_(t, -s, s)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.W.matmul(x) + self.U.matmul(hidden) + self.bias
        )


class MatGRUCellO(nn.Module):
    """
    EGCN-O matrix-GRU cell from the pub_iot implementation.

    Unlike EGCN-H, the weight matrix evolves only from its previous
    value; node embeddings and TopK are not used in the recurrent update.
    """

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.update = MatGRUGate(rows, cols, nn.Sigmoid())
        self.reset = MatGRUGate(rows, cols, nn.Sigmoid())
        self.htilda = MatGRUGate(rows, cols, nn.Tanh())

    def forward(self, prev_q: torch.Tensor) -> torch.Tensor:
        z = prev_q
        update = self.update(z, prev_q)
        reset = self.reset(z, prev_q)
        h = self.htilda(z, reset * prev_q)
        return (1.0 - update) * prev_q + update * h


class GRCUO(nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.evolve_weights = MatGRUCellO(in_feats, out_feats)
        self.activation = nn.RReLU()
        self.gcn_init_weights = nn.Parameter(
            torch.empty(in_feats, out_feats)
        )
        s = 1.0 / math.sqrt(out_feats)
        nn.init.uniform_(self.gcn_init_weights, -s, s)

    def forward(
        self,
        adjs: list[torch.Tensor],
        nodes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        w = self.gcn_init_weights
        out: list[torch.Tensor] = []

        for adj, x in zip(adjs, nodes):
            w = self.evolve_weights(w)
            h = self.activation(
                torch.sparse.mm(adj, x.float().matmul(w))
            )

            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError("Non-finite EGCN-O node embeddings")

            out.append(h)

        return out


class EGCNOEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GRCUO(input_dim, hidden_dim),
                GRCUO(hidden_dim, hidden_dim),
            ]
        )

    def forward(
        self,
        adjs: list[torch.Tensor],
        nodes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        h = nodes
        for layer in self.layers:
            h = layer(adjs, h)
        return h


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(); self.weight = nn.Parameter(torch.empty(in_features, out_features)); self.bias = nn.Parameter(torch.empty(out_features)) if bias else None; self.reset_parameters()
    def reset_parameters(self) -> None:
        s = 1.0 / math.sqrt(self.weight.size(1)); nn.init.uniform_(self.weight, -s, s)
        if self.bias is not None: nn.init.uniform_(self.bias, -s, s)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.sparse.mm(adj, x.matmul(self.weight)); return out + self.bias if self.bias is not None else out


class ReconstructionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__(); self.attr1 = GraphConvolution(embedding_dim, hidden_dim); self.attr2 = GraphConvolution(hidden_dim, input_dim); self.struct1 = GraphConvolution(embedding_dim, hidden_dim); self.dropout = float(dropout)
    def forward(self, emb: torch.Tensor, adj: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = F.relu(self.attr1(emb, adj)); a = F.dropout(a, p=self.dropout, training=False); a = self.attr2(a, adj)[active]
        s = self.struct1(emb, adj); s = F.dropout(s, p=self.dropout, training=False); s = s[active]; s = s.matmul(s.t())
        return a, s


class EGCNODetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, decoder_hidden_dim: int, decoder_dropout: float) -> None:
        super().__init__(); self.encoder = EGCNOEncoder(input_dim, hidden_dim); self.decoder = ReconstructionDecoder(hidden_dim, input_dim, decoder_hidden_dim, decoder_dropout)
    def forward(self, window: WindowData, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        adjs = [s.adj_norm.to(device) for s in window.steps]; nodes = [s.x.to(device) for s in window.steps]
        embs = self.encoder(adjs, nodes); attrs = []; structs = []
        for emb, step, adj in zip(embs, window.steps, adjs):
            a, s = self.decoder(emb, adj, step.active_indices.to(device)); attrs.append(a); structs.append(s)
        return attrs, structs


def reconstruction_score(pred_attr: torch.Tensor, pred_struct: torch.Tensor, gt_attr: torch.Tensor, gt_struct: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ae = torch.sqrt(torch.sum((gt_attr - pred_attr) ** 2, dim=1)); se = torch.sqrt(torch.sum((gt_struct - pred_struct) ** 2, dim=1)); score = alpha * ae + (1.0 - alpha) * se
    return score, ae, se


def window_loss_and_target_scores(model: EGCNODetector, window: WindowData, device: torch.device, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pa, ps = model(window, device); losses = []; target = None
    for i, (a, s, step) in enumerate(zip(pa, ps, window.steps)):
        score, ae, se = reconstruction_score(a, s, step.x_local.to(device), step.adj_local.to(device), alpha); losses.append(score.mean())
        if i == len(window.steps) - 1: target = (score, ae, se)
    assert target is not None
    return torch.stack(losses).mean(), target[0], target[1], target[2]


def prepare_split(rows: pd.DataFrame, sequence_length: int) -> tuple[pd.DataFrame, list[list[int]]]:
    rows = rows.sort_values(["global_graph_index", "graph_index"], kind="stable").reset_index(drop=True)
    return rows, build_history_indices(rows, sequence_length)

def train_epoch(model: EGCNODetector, optimizer: torch.optim.Optimizer, rows: pd.DataFrame, histories: list[list[int]], loader: GraphLoader, device: torch.device, alpha: float, batch_size: int, seed: int, epoch: int) -> float:
    model.train(); indices = np.arange(len(rows)); rng = np.random.default_rng(seed + epoch); rng.shuffle(indices); total = 0.0; samples = 0
    for start in tqdm(range(0, len(indices), batch_size), desc=f"Train epoch {epoch:03d}", leave=False):
        batch = indices[start:start + batch_size]; optimizer.zero_grad(set_to_none=True)
        for idx in batch:
            idx = int(idx); window = load_window(rows, histories[idx], loader); loss, _, _, _ = window_loss_and_target_scores(model, window, device, alpha)
            (loss / len(batch)).backward(); total += float(loss.detach().cpu()); samples += 1
            del window, loss
        optimizer.step()
    return total / max(samples, 1)


@torch.no_grad()
def evaluate_loss(model: EGCNODetector, rows: pd.DataFrame, histories: list[list[int]], loader: GraphLoader, device: torch.device, alpha: float, desc: str) -> float:
    model.eval(); total = 0.0
    for idx in tqdm(range(len(rows)), desc=desc, leave=False):
        window = load_window(rows, histories[idx], loader); loss, _, _, _ = window_loss_and_target_scores(model, window, device, alpha); total += float(loss.cpu())
    return total / max(len(rows), 1)


@torch.no_grad()
def collect_validation_scores(model: EGCNODetector, rows: pd.DataFrame, histories: list[list[int]], loader: GraphLoader, device: torch.device, alpha: float) -> np.ndarray:
    model.eval(); parts = []
    for idx in tqdm(range(len(rows)), desc="Scoring validation"):
        window = load_window(rows, histories[idx], loader); _, score, _, _ = window_loss_and_target_scores(model, window, device, alpha); parts.append(score.cpu().numpy().astype(np.float64))
    if not parts: raise RuntimeError("No validation scores")
    return np.concatenate(parts)


def strict_fpr_threshold(scores: np.ndarray, target_fpr: float) -> tuple[float, float, int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0: raise RuntimeError("No validation scores")
    if not 0.0 < target_fpr < 1.0: raise ValueError("validation-fpr-target must be in (0,1)")
    allowed = max(int(math.ceil(target_fpr * scores.size)) - 1, 0); desc = np.sort(scores)[::-1]
    threshold = float(desc[0] if allowed == 0 else desc[min(allowed, desc.size - 1)])
    fp = int(np.sum(scores > threshold)); fpr = fp / scores.size
    if not fpr < target_fpr:
        threshold = float(np.nextafter(threshold, np.inf)); fp = int(np.sum(scores > threshold)); fpr = fp / scores.size
    if not fpr < target_fpr: raise RuntimeError(f"Could not obtain strict validation FPR < {target_fpr}")
    return threshold, fpr, fp


def snapshot_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, graph_type: str) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64); scores = np.asarray(scores, dtype=np.float64); pred = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel(); fpr = float(fp / (fp + tn)) if fp + tn else float("nan")
    if graph_type == "benign": return {"FPR": fpr}
    tpr = float(tp / (tp + fn)) if tp + fn else float("nan")
    precision = float(precision_score(labels, pred, zero_division=0)); f1 = float(f1_score(labels, pred, zero_division=0)); auc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    return {"FPR": fpr, "TPR": tpr, "PRECISION": precision, "F1": f1, "AUC": auc}


@torch.no_grad()
def evaluate_split(model: EGCNODetector, rows: pd.DataFrame, histories: list[list[int]], loader: GraphLoader, device: torch.device, alpha: float, threshold: float, split: str, run_dir: Path) -> pd.DataFrame:
    model.eval(); values: dict[tuple[str, str], list[float]] = {}; score_parts = []; label_parts = []; graph_parts = []
    for idx in tqdm(range(len(rows)), desc=f"Scoring {split}"):
        row = rows.iloc[idx]; window = load_window(rows, histories[idx], loader); _, score, _, _ = window_loss_and_target_scores(model, window, device, alpha)
        scores = score.cpu().numpy().astype(np.float64); labels = window.steps[-1].labels_local.cpu().numpy().astype(np.int64); gt = str(row["graph_type"])
        for metric, value in snapshot_metrics(labels, scores, threshold, gt).items(): values.setdefault((gt, metric), []).append(value)
        score_parts.append(scores); label_parts.append(labels); graph_parts.append(np.full(labels.size, int(row["global_graph_index"]), dtype=np.int64))
    np.savez_compressed(run_dir / f"{split}_node_scores.npz", scores=np.concatenate(score_parts), labels=np.concatenate(label_parts), global_graph_index=np.concatenate(graph_parts))
    output = []
    for gt, order in (("benign", ["FPR"]), ("mixed", ["FPR", "TPR", "PRECISION", "F1", "AUC"])):
        count = int((rows["graph_type"] == gt).sum())
        for metric in order:
            arr = np.asarray(values.get((gt, metric), []), dtype=np.float64)
            if arr.size == 0: continue
            valid = arr[np.isfinite(arr)]
            output.append({"split": split, "graph_type": gt, "metric": metric, "mean": float(valid.mean()) if valid.size else float("nan"), "std": float(valid.std(ddof=0)) if valid.size else float("nan"), "snapshots": count, "valid_snapshots": int(valid.size)})
    frame = pd.DataFrame(output); print(f"\nSnapshot-averaged node metrics - {split}:"); print(frame.to_string(index=False)); return frame


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    if not 0.0 <= args.alpha <= 1.0: raise ValueError("alpha must be in [0,1]")
    project = args.project_root.expanduser().resolve(); timeline_path = project / args.timeline; graph_root = project / args.graph_root; output_root = project / args.output_root / f"egcn_o_{args.representation}"
    if not timeline_path.is_file(): raise FileNotFoundError(timeline_path)
    if not graph_root.is_dir(): raise FileNotFoundError(graph_root)
    output_root.mkdir(parents=True, exist_ok=True); prefix = "smoke" if args.smoke_test else "run"; run_dir = output_root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80); print(f"EGCN-O - UNRAVELED - {args.representation.upper()}"); print("=" * 80); print("Project root:", project); print("Graph root:", graph_root); print("Run directory:", run_dir); print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0)); print("Visible GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

    timeline = load_timeline(timeline_path); rows_by_split = {s: timeline[timeline["split"] == s].copy().reset_index(drop=True) for s in EXPECTED_SPLIT_COUNTS}
    if args.smoke_test:
        rows_by_split["train"] = smoke_subset(rows_by_split["train"], "train", 16); rows_by_split["validation"] = smoke_subset(rows_by_split["validation"], "validation", 8); rows_by_split["test_week5"] = smoke_subset(rows_by_split["test_week5"], "test_week5", 16); rows_by_split["test_week6"] = smoke_subset(rows_by_split["test_week6"], "test_week6", 16); args.epochs = min(args.epochs, 2); args.patience = min(args.patience, 2)
    print("\nGraphs used:")
    for split, rows in rows_by_split.items(): print(f"{split:12s}: {len(rows):5d}", rows["graph_type"].value_counts().to_dict())

    fmin, fscale, scaler_nodes = fit_minmax_scaler(rows_by_split["train"], graph_root, args.representation); np.savez_compressed(run_dir / "feature_scaler.npz", minimum=fmin, scale=fscale, training_nodes=scaler_nodes, kind="training_minmax")
    loader = GraphLoader(graph_root, args.representation, fmin, fscale); split_data = {}
    for split, rows in rows_by_split.items():
        ordered, histories = prepare_split(rows, args.sequence_length); split_data[split] = (ordered, histories); lengths = np.asarray([len(h) for h in histories]); print(f"{split:12s} temporal windows: min={lengths.min()} mean={lengths.mean():.3f} max={lengths.max()}")

    input_dim = expected_feature_dim(args.representation); model = EGCNODetector(input_dim, args.hidden_dim, args.decoder_hidden_dim, args.decoder_dropout).to(device); parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nArchitecture:"); print("Input dim:", input_dim); print("Encoder: EGCN-O, 2 GRCU layers, RReLU"); print("Hidden dim:", args.hidden_dim); print("Decoder hidden dim:", args.decoder_hidden_dim); print("Alpha:", args.alpha); print("Sequence length:", args.sequence_length); print("Trainable parameters:", parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    config = vars(args).copy(); config.update({"method": "EGCN-O", "task": "node anomaly detection", "project_root": str(project), "timeline_path": str(timeline_path), "graph_root": str(graph_root), "run_dir": str(run_dir), "device": str(device), "input_dim": input_dim, "parameters": parameters, "feature_scaler": "training_minmax", "temporal_window_policy": "causal consecutive history plus current target, up to 4 snapshots, no day/capture crossing", "architecture_reference": "pub_iot EGCN-O matrix-GRU weight evolution from previous weights only; two 64-dim GRCU layers; RReLU; reconstruction decoder", "evaluation": "target-snapshot node scores; per-snapshot mean/std"}); save_json(run_dir / "config.json", config)

    train_rows, train_hist = split_data["train"]; val_rows, val_hist = split_data["validation"]; history_rows = []; best_val = float("inf"); best_epoch = -1; no_improve = 0; checkpoint = run_dir / "best_checkpoint.pt"
    print("\nTraining started")
    for epoch in range(1, args.epochs + 1):
        started = time.time(); tr = train_epoch(model, optimizer, train_rows, train_hist, loader, device, args.alpha, args.batch_size, args.seed, epoch); va = evaluate_loss(model, val_rows, val_hist, loader, device, args.alpha, f"Validation epoch {epoch:03d}"); elapsed = time.time() - started
        history_rows.append({"epoch": epoch, "train_loss": tr, "validation_loss": va, "elapsed_seconds": elapsed}); pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False); print(f"Epoch {epoch:03d} | train={tr:.8f} | val={va:.8f} | time={elapsed:.1f}s")
        if va < best_val - args.min_delta:
            best_val = float(va); best_epoch = epoch; no_improve = 0; torch.save({"epoch": epoch, "validation_loss": best_val, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config}, checkpoint); print(f"  -> Best validation loss: {best_val:.8f}")
        else:
            no_improve += 1
            if no_improve >= args.patience: print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}."); break
        gc.collect();
        if device.type == "cuda": torch.cuda.empty_cache()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(ckpt["model_state_dict"]); print("\nBest epoch:", best_epoch); print("Best validation loss:", best_val)
    val_scores = collect_validation_scores(model, val_rows, val_hist, loader, device, args.alpha); threshold, actual_fpr, actual_fp = strict_fpr_threshold(val_scores, args.validation_fpr_target); thresholds = {"criterion": "validation_fpr_below_target", "target_fpr": float(args.validation_fpr_target), "actual_validation_fpr": float(actual_fpr), "actual_validation_fp": int(actual_fp), "validation_nodes": int(val_scores.size), "node_score": float(threshold)}; save_json(run_dir / "thresholds.json", thresholds); np.savez_compressed(run_dir / "validation_node_scores.npz", scores=val_scores); print("\nThreshold:"); print(json.dumps(thresholds, indent=2))
    frames = []
    for split in TEST_SPLITS:
        rows, histories = split_data[split]; frames.append(evaluate_split(model, rows, histories, loader, device, args.alpha, threshold, split, run_dir))
    metrics = pd.concat(frames, ignore_index=True); metrics.to_csv(run_dir / "metrics.csv", index=False)
    summary = {"method": "EGCN-O", "task": "node anomaly detection", "representation": "TDG" if args.representation == "tdg" else "e-TDG", "feature_dim": input_dim, "best_epoch": int(best_epoch), "best_validation_loss": float(best_val), "thresholds": thresholds, "parameters": int(parameters), "alpha": float(args.alpha), "sequence_length": int(args.sequence_length), "evaluation": "node-level metrics computed per target snapshot and reported as mean and standard deviation", "metrics": metrics.to_dict(orient="records")}; save_json(run_dir / "summary.json", summary)
    print("\n" + "=" * 80); print("EXPERIMENT COMPLETED"); print("=" * 80); print(metrics.to_string(index=False)); print("\nResults saved in:"); print(run_dir)


if __name__ == "__main__":
    main()
