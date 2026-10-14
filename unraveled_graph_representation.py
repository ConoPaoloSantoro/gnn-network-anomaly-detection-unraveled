import os
import re
import json
import time
import glob
import pickle
import argparse
import functools
from math import ceil
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd


REP_LIST = ["TDG", "eTDG", "ALL"]


# =========================
# Graph construction
# =========================

def create_tdg(snapshot_mat, snapshot_features, label, stage_idx=None):
    """Create TDG graph.

    Nodes are (ip, port) pairs.
    Edges are communications observed in the snapshot.
    Node features are the average of incident flow features.
    Node labels are assigned by majority vote over incident flows:
    0 = benign, 1 = malicious.
    """
    node_id_counter = 0
    graph_dict = {}
    node_ids_dict = {}
    node_reverse_dict = {}

    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]

        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)

        if src_node not in node_ids_dict:
            node_ids_dict[src_node] = node_id_counter
            node_reverse_dict[node_id_counter] = src_node
            node_id_counter += 1

        if dst_node not in node_ids_dict:
            node_ids_dict[dst_node] = node_id_counter
            node_reverse_dict[node_id_counter] = dst_node
            node_id_counter += 1

        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]

        if label and stage_idx is not None:
            edge_stage = flow[stage_idx]
            edge_label = 0 if edge_stage == "Benign" else 1
        else:
            edge_label = 0

        graph_dict.setdefault(src_node_id, []).append(
            (dst_node_id, snapshot_features[i], edge_label)
        )
        graph_dict.setdefault(dst_node_id, []).append(
            (src_node_id, snapshot_features[i], edge_label)
        )

    total_nodes = len(node_ids_dict)
    print(f"Total nodes: {total_nodes}")

    flow_features_matrix = np.zeros((total_nodes, snapshot_features.shape[1]), dtype=float)
    node_features_matrix = np.zeros_like(flow_features_matrix)
    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=np.uint8)
    labels_matrix = np.zeros((total_nodes,), dtype=np.uint8)
    edges_list = []

    node_list1 = []
    node_list2 = []

    for _, src_node_id in node_ids_dict.items():
        edge_number = 0
        benign_edge_number = 0
        edge_features_sum = np.zeros((snapshot_features.shape[1],), dtype=float)

        for dst_node_id, edge_features, edge_label in graph_dict[src_node_id]:
            edge_features_sum += np.array(edge_features, dtype=float)
            adj_matrix[src_node_id][dst_node_id] = 1
            edges_list.append((src_node_id, dst_node_id))
            benign_edge_number += 1 if edge_label == 0 else 0
            edge_number += 1
            node_list1.append(src_node_id)
            node_list2.append(dst_node_id)

        if edge_number > 0:
            node_features = edge_features_sum / edge_number
        else:
            node_features = edge_features_sum

        flow_features_matrix[src_node_id] = node_features
        labels_matrix[src_node_id] = 0 if benign_edge_number > edge_number / 2 else 1
        node_list1.append(src_node_id)
        node_list2.append(src_node_id)

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = flow_features_matrix[i]

    node_array1 = np.array(node_list1, dtype=np.uint32)
    node_array2 = np.array(node_list2, dtype=np.uint32)

    return (
        node_features_matrix,
        adj_matrix,
        labels_matrix,
        (node_array1, node_array2),
        node_reverse_dict,
        edges_list,
        node_ids_dict,
    )


def to_global_graph_dict(global_dict, cc_dict):
    for key, value in cc_dict.items():
        global_dict[key] = value


def to_global_graph_dict_degree(global_dict, cc_node_degree):
    for node, degree in cc_node_degree:
        global_dict[node] = degree


def is_barycenter(barycenter_dict, barycenter_nodes, nodes):
    for node in nodes:
        barycenter_dict[node] = 1 if node in barycenter_nodes else 0


def is_center(center_dict, center_nodes, nodes):
    for node in nodes:
        center_dict[node] = 1 if node in center_nodes else 0


def to_node_betweenness(global_dict, edge_betweenness):
    for edge, betweenness in edge_betweenness.items():
        src_node, dst_node = edge
        global_dict[src_node] = betweenness
        global_dict[dst_node] = betweenness


def create_extendend_graph(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict,
                           flow_features_matrix, adj_matrix, labels_matrix):
    """Create eTDG by adding structural node features to the TDG."""
    total_nodes = len(node_ids_dict.keys())
    structural_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1] + structural_features_matrix.shape[1]),
        dtype=float,
    )

    graph = nx.Graph(edges_list)

    if graph.number_of_nodes() == 0:
        for i in range(node_features_matrix.shape[0]):
            node_features_matrix[i] = np.concatenate(
                [flow_features_matrix[i], structural_features_matrix[i]]
            )
        return node_features_matrix, adj_matrix, labels_matrix, edges_list

    graph_degree_centrality = nx.degree_centrality(graph)
    graph_betweenness_centrality = nx.betweenness_centrality(graph)
    graph_closeness_centrality = nx.closeness_centrality(graph)

    cc_degree_centrality = {}
    cc_betweenness_centrality = {}
    cc_closeness_centrality = {}
    second_order_centrality = {}
    edge_betweenness_centrality = {}
    eccentricity = {}
    barycenter = {}
    radius = {}
    center = {}
    size = {}
    degree = {}
    degree_two_hops = {}

    connected_components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    for cc in connected_components:
        if cc.number_of_nodes() == 1:
            only_node = list(cc.nodes())[0]
            cc_eccentricity = {only_node: 0}
            cc_radius = 0
            cc_degree_cent = {only_node: 0}
            cc_betw_cent = {only_node: 0}
            cc_close_cent = {only_node: 0}
            cc_second_order = {only_node: 0}
            cc_edge_betw = {}
            cc_degree = [(only_node, 0)]
            cc_barycenter = [only_node]
            cc_center = [only_node]
        else:
            cc_eccentricity = nx.eccentricity(cc)
            cc_radius = nx.radius(cc, e=cc_eccentricity)
            cc_degree_cent = nx.degree_centrality(cc)
            cc_betw_cent = nx.betweenness_centrality(cc)
            cc_close_cent = nx.closeness_centrality(cc)
            try:
                cc_second_order = nx.second_order_centrality(cc)
            except Exception:
                cc_second_order = {node: 0 for node in cc.nodes()}
            cc_edge_betw = nx.edge_betweenness_centrality(cc)
            cc_degree = cc.degree()
            cc_barycenter = nx.barycenter(cc)
            cc_center = nx.center(cc, e=cc_eccentricity)

        to_global_graph_dict(cc_degree_centrality, cc_degree_cent)
        to_global_graph_dict(cc_betweenness_centrality, cc_betw_cent)
        to_global_graph_dict(cc_closeness_centrality, cc_close_cent)
        to_global_graph_dict(second_order_centrality, cc_second_order)
        to_global_graph_dict(eccentricity, cc_eccentricity)
        to_node_betweenness(edge_betweenness_centrality, cc_edge_betw)
        to_global_graph_dict_degree(degree, cc_degree)
        is_barycenter(barycenter, cc_barycenter, cc.nodes)
        is_center(center, cc_center, cc.nodes)

        for node in cc.nodes:
            degree_two_hops[node] = len(nx.ego_graph(cc, node, radius=2, undirected=True).nodes())
            size[node] = cc.number_of_nodes()
            radius[node] = cc_radius

    for _, node_id in node_ids_dict.items():
        structural_features_matrix[node_id][0] = graph_degree_centrality.get(node_id, 0)
        structural_features_matrix[node_id][1] = graph_betweenness_centrality.get(node_id, 0)
        structural_features_matrix[node_id][2] = graph_closeness_centrality.get(node_id, 0)
        structural_features_matrix[node_id][3] = cc_degree_centrality.get(node_id, 0)
        structural_features_matrix[node_id][4] = cc_betweenness_centrality.get(node_id, 0)
        structural_features_matrix[node_id][5] = cc_closeness_centrality.get(node_id, 0)
        structural_features_matrix[node_id][6] = second_order_centrality.get(node_id, 0)
        structural_features_matrix[node_id][7] = edge_betweenness_centrality.get(node_id, 0)
        structural_features_matrix[node_id][8] = eccentricity.get(node_id, 0)
        structural_features_matrix[node_id][9] = barycenter.get(node_id, 0)
        structural_features_matrix[node_id][10] = radius.get(node_id, 0)
        structural_features_matrix[node_id][11] = center.get(node_id, 0)
        structural_features_matrix[node_id][12] = size.get(node_id, 0)
        structural_features_matrix[node_id][13] = degree.get(node_id, 0)
        structural_features_matrix[node_id][14] = degree_two_hops.get(node_id, 0)

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structural_features_matrix[i]]
        )

    for i in range(snapshot_mat.shape[0]):
        flow = snapshot_mat[i]
        src_ip = flow[2]
        src_port = flow[5]
        dst_ip = flow[6]
        dst_port = flow[9]
        src_node = (src_ip, src_port)
        dst_node = (dst_ip, dst_port)
        src_node_id = node_ids_dict[src_node]
        dst_node_id = node_ids_dict[dst_node]
        if adj_matrix[src_node_id][dst_node_id] != 1 and log_file is not None:
            log_file.write(f"Missing edge ({src_node_id}, {dst_node_id})\n")

    return node_features_matrix, adj_matrix, labels_matrix, edges_list


def create_extendend_graph_approx(snapshot_mat, snapshot_features, log_file, edges_list, node_ids_dict,
                                  flow_features_matrix, adj_matrix, labels_matrix, approx_ratio=0.5):
    """Create a very lightweight approximate eTDG for very large graphs."""
    total_nodes = len(node_ids_dict.keys())
    structural_features_matrix = np.zeros((total_nodes, 15), dtype=float)
    node_features_matrix = np.zeros(
        (total_nodes, flow_features_matrix.shape[1] + structural_features_matrix.shape[1]),
        dtype=float,
    )

    graph = nx.Graph(edges_list)

    if graph.number_of_nodes() == 0:
        for i in range(node_features_matrix.shape[0]):
            node_features_matrix[i] = np.concatenate(
                [flow_features_matrix[i], structural_features_matrix[i]]
            )
        return node_features_matrix, adj_matrix, labels_matrix, edges_list

    graph_degree_centrality = nx.degree_centrality(graph)
    degree = dict(graph.degree())

    size = {}
    for cc in nx.connected_components(graph):
        cc_nodes = list(cc)
        cc_size = len(cc_nodes)
        for node in cc_nodes:
            size[node] = cc_size

    for _, node_id in node_ids_dict.items():
        structural_features_matrix[node_id][0] = graph_degree_centrality.get(node_id, 0)
        structural_features_matrix[node_id][1] = 0
        structural_features_matrix[node_id][2] = 0
        structural_features_matrix[node_id][3] = graph_degree_centrality.get(node_id, 0)
        structural_features_matrix[node_id][4] = 0
        structural_features_matrix[node_id][5] = 0
        structural_features_matrix[node_id][6] = 0
        structural_features_matrix[node_id][7] = 0
        structural_features_matrix[node_id][8] = 0
        structural_features_matrix[node_id][9] = 0
        structural_features_matrix[node_id][10] = 0
        structural_features_matrix[node_id][11] = 0
        structural_features_matrix[node_id][12] = size.get(node_id, 0)
        structural_features_matrix[node_id][13] = degree.get(node_id, 0)
        structural_features_matrix[node_id][14] = 0

    for i in range(node_features_matrix.shape[0]):
        node_features_matrix[i] = np.concatenate(
            [flow_features_matrix[i], structural_features_matrix[i]]
        )

    return node_features_matrix, adj_matrix, labels_matrix, edges_list


# =========================
# Robust CSV loader
# =========================

def load_labeled_csv_robust(csv_path):
    """
    Load labeled Unraveled CSVs even when some rows contain extra commas
    in the textual fields near the end of the row.

    Strategy:
    - keep the prefix columns exactly as defined by the header
    - keep the last 4 label columns:
      Activity, Stage, DefenderResponse, Signature
    - collapse the problematic middle text block into the expected 5 fields:
      requested_server_name, client_fingerprint, server_fingerprint,
      user_agent, content_type
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n")
        columns = header.split(",")

        expected_cols = len(columns)

        req_idx = columns.index("requested_server_name")
        activity_idx = columns.index("Activity")

        prefix_len = req_idx
        middle_len = activity_idx - req_idx
        suffix_len = expected_cols - activity_idx

        rows = []
        repaired_rows = 0
        skipped_rows = 0

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            parts = line.split(",")

            if len(parts) == expected_cols:
                rows.append(parts)
                continue

            if len(parts) > expected_cols:
                prefix = parts[:prefix_len]
                suffix = parts[-suffix_len:]
                middle_raw = parts[prefix_len:-suffix_len]

                if len(middle_raw) >= middle_len:
                    rebuilt_middle = middle_raw[:middle_len - 1] + [",".join(middle_raw[middle_len - 1:])]
                else:
                    rebuilt_middle = middle_raw + ["nan"] * (middle_len - len(middle_raw))

                rebuilt = prefix + rebuilt_middle + suffix

                if len(rebuilt) == expected_cols:
                    rows.append(rebuilt)
                    repaired_rows += 1
                    continue

            skipped_rows += 1

    df = pd.DataFrame(rows, columns=columns)
    print(f"[load_labeled_csv_robust] loaded={len(df)} repaired={repaired_rows} skipped={skipped_rows}")
    return df


# =========================
# Resume helpers
# =========================

def get_existing_graph_indices(base_dir):
    """
    Return the set of snapshot indices already saved for TDG/eTDG
    across all class folders.
    """
    existing = {
        "tdg_graph": set(),
        "etdg_graph": set(),
    }

    for graph_type in ["tdg_graph", "etdg_graph"]:
        for cls in ["full_benign", "mixed", "full_malicious"]:
            graph_dir = os.path.join(base_dir, graph_type, cls)
            if not os.path.isdir(graph_dir):
                continue

            for fname in os.listdir(graph_dir):
                m = re.match(r"graph_(\d+)\.pkl$", fname)
                if m:
                    existing[graph_type].add(int(m.group(1)))

    return existing


def get_resume_snapshot_index(base_dir, graph_repre):
    """
    Compute the first snapshot index to process.

    Rules:
    - if TDG only: resume after the largest TDG snapshot already present
    - if eTDG only: resume after the largest eTDG snapshot already present
    - if ALL: only consider snapshots that have BOTH TDG and eTDG already present
    """
    existing = get_existing_graph_indices(base_dir)

    if graph_repre == "TDG":
        done = existing["tdg_graph"]
    elif graph_repre == "eTDG":
        done = existing["etdg_graph"]
    else:  # ALL
        done = existing["tdg_graph"].intersection(existing["etdg_graph"])

    if not done:
        return 0

    return max(done) + 1


def get_saved_class_for_snapshot(capture_out_dir, graph_type, snapshot_idx):
    """
    Return the class folder name where graph_{snapshot_idx}.pkl is stored,
    or None if not found.
    """
    for cls in ["full_benign", "mixed", "full_malicious"]:
        p = os.path.join(capture_out_dir, graph_type, cls, f"graph_{snapshot_idx}.pkl")
        if os.path.isfile(p):
            return cls
    return None


def class_to_type_value(class_name):
    if class_name == "full_benign":
        return 1.0
    if class_name == "mixed":
        return 2.0
    if class_name == "full_malicious":
        return 3.0
    return None


def build_complete_stats_from_saved_graphs(
    capture_name,
    capture_out_dir,
    graph_repre,
    features,
    time_intervals_list,
    partial_stats,
):
    """
    Rebuild a complete stats dictionary for the capture using:
    - saved graph files already present on disk
    - partial_stats from newly processed snapshots in this run

    This guarantees a coherent stats.json even after resume.
    """
    stats = {capture_name: {}}

    if graph_repre in ["TDG", "ALL"]:
        stats[capture_name]["tdg_graph"] = {}
    if graph_repre in ["eTDG", "ALL"]:
        stats[capture_name]["etdg_graph"] = {}

    for interval_number, time_interval in enumerate(time_intervals_list):
        start_ts, end_ts = time_interval
        snapshot_data = features.loc[
            (features["bidirectional_first_seen_ms"] >= start_ts)
            & (features["bidirectional_first_seen_ms"] < end_ts)
        ]

        if snapshot_data.shape[0] == 0:
            continue

        tdg_class = get_saved_class_for_snapshot(capture_out_dir, "tdg_graph", interval_number)
        etdg_class = get_saved_class_for_snapshot(capture_out_dir, "etdg_graph", interval_number)

        if graph_repre in ["TDG", "ALL"] and tdg_class is not None:
            tdg_path = os.path.join(
                capture_out_dir,
                "tdg_graph",
                tdg_class,
                f"graph_{interval_number}.pkl",
            )
            with open(tdg_path, "rb") as f:
                tdg_graph = pickle.load(f)

            number_nodes = float(tdg_graph["adj"].shape[0])
            number_edges = float(int(np.sum(tdg_graph["adj"])))
            number_flows = float(snapshot_data.shape[0])
            graph_type_value = class_to_type_value(tdg_class)

            stats[capture_name]["tdg_graph"][f"graph_{interval_number}"] = {
                "nodes": number_nodes,
                "edges": number_edges,
                "flows": number_flows,
                "type": graph_type_value,
            }

        if graph_repre in ["eTDG", "ALL"] and etdg_class is not None:
            etdg_path = os.path.join(
                capture_out_dir,
                "etdg_graph",
                etdg_class,
                f"graph_{interval_number}.pkl",
            )
            with open(etdg_path, "rb") as f:
                etdg_graph = pickle.load(f)

            number_nodes = float(etdg_graph["adj"].shape[0])
            number_edges = float(int(np.sum(etdg_graph["adj"])))
            number_flows = float(snapshot_data.shape[0])
            graph_type_value = class_to_type_value(etdg_class)

            delta_time = None
            if (
                "etdg_graph" in partial_stats
                and f"graph_{interval_number}" in partial_stats["etdg_graph"]
                and "delta_time" in partial_stats["etdg_graph"][f"graph_{interval_number}"]
            ):
                delta_time = float(partial_stats["etdg_graph"][f"graph_{interval_number}"]["delta_time"])

            stats[capture_name]["etdg_graph"][f"graph_{interval_number}"] = {
                "nodes": number_nodes,
                "edges": number_edges,
                "flows": number_flows,
                "type": graph_type_value,
                "delta_time": delta_time,
            }

    return stats


# =========================
# I/O and orchestration
# =========================

def save_file(out_path, capture_name, graph_type, graph, full_benign, mixed, full_malicious, interval_number):
    if full_benign:
        save_file_path = os.path.join(out_path, capture_name, graph_type, "full_benign", f"graph_{interval_number}.pkl")
    elif mixed:
        save_file_path = os.path.join(out_path, capture_name, graph_type, "mixed", f"graph_{interval_number}.pkl")
    elif full_malicious:
        save_file_path = os.path.join(out_path, capture_name, graph_type, "full_malicious", f"graph_{interval_number}.pkl")
    else:
        return

    with open(save_file_path, "wb") as handle:
        pickle.dump(graph, handle, protocol=4)


def create_graph(out_path=None, capture_name=None, graph_type="tdg_graph", snapshot_data=None,
                 snapshot_features=None, interval_number=None, stats=None, log_file=None,
                 full_benign=False, mixed=False, full_malicious=False, also_etdg=False,
                 etdg_mode="exact", approx_node_threshold=2000, approx_ratio=0.5):
    label = "Stage" in snapshot_data.columns
    stage_idx = snapshot_data.columns.get_loc("Stage") if label else None

    if graph_type not in stats:
        stats[graph_type] = {}

    if graph_type == "tdg_graph":
        features_matrix, adj_matrix, labels_matrix, edge_list, node_id_to_port_ip, edges_list, node_ids_dict = create_tdg(
            snapshot_mat=snapshot_data.to_numpy(),
            snapshot_features=snapshot_features,
            label=label,
            stage_idx=stage_idx,
        )

        graph = {
            "node_features": features_matrix,
            "adj": adj_matrix,
            "node_labels": labels_matrix,
            "edge_list": edge_list,
            "node_id_to_port_ip": node_id_to_port_ip,
        }

        save_file(
            out_path=out_path,
            capture_name=capture_name,
            graph_type="tdg_graph",
            graph=graph,
            full_benign=full_benign,
            full_malicious=full_malicious,
            mixed=mixed,
            interval_number=interval_number,
        )

        number_nodes = adj_matrix.shape[0]
        number_edges = int(np.sum(adj_matrix))
        number_flows = snapshot_data.shape[0]

        stats[graph_type][f"graph_{interval_number}"] = {
            "nodes": float(number_nodes),
            "edges": float(number_edges),
            "flows": float(number_flows),
            "type": 1.0 if full_benign else 2.0 if mixed else 3.0,
        }

        if also_etdg in ["ALL", "eTDG"]:
            etdg_log_file = log_file if log_file is not None else None

            start = time.time()
            n_nodes = adj_matrix.shape[0]

            if etdg_mode == "approx" and n_nodes >= approx_node_threshold:
                print(f"[APPROX eTDG] capture={capture_name} snapshot={interval_number} nodes={n_nodes}")
                features_matrix_ex, _, _, _ = create_extendend_graph_approx(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=etdg_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix,
                    approx_ratio=approx_ratio,
                )
            else:
                print(f"[EXACT eTDG] capture={capture_name} snapshot={interval_number} nodes={n_nodes}")
                features_matrix_ex, _, _, _ = create_extendend_graph(
                    snapshot_mat=snapshot_data.to_numpy(),
                    snapshot_features=snapshot_features,
                    log_file=etdg_log_file,
                    edges_list=edges_list,
                    node_ids_dict=node_ids_dict,
                    labels_matrix=labels_matrix,
                    adj_matrix=adj_matrix,
                    flow_features_matrix=features_matrix,
                )

            delta = time.time() - start

            etdg_graph = {
                "node_features": features_matrix_ex,
                "adj": adj_matrix,
                "node_labels": labels_matrix,
                "edge_list": edge_list,
                "node_id_to_port_ip": node_id_to_port_ip,
            }

            save_file(
                out_path=out_path,
                capture_name=capture_name,
                graph_type="etdg_graph",
                graph=etdg_graph,
                full_benign=full_benign,
                full_malicious=full_malicious,
                mixed=mixed,
                interval_number=interval_number,
            )

            if "etdg_graph" not in stats:
                stats["etdg_graph"] = {}

            stats["etdg_graph"][f"graph_{interval_number}"] = {
                "nodes": float(number_nodes),
                "edges": float(number_edges),
                "flows": float(number_flows),
                "type": 1.0 if full_benign else 2.0 if mixed else 3.0,
                "delta_time": float(delta),
            }


def generate_graph_from_snapshot(out_path, log_file_path, capture_name, features, graph_type,
                                 etdg_mode, approx_node_threshold, approx_ratio, time_indx):
    time_interval, interval_number = time_indx
    stats = {}

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "a")

    start_ts, end_ts = time_interval
    snapshot_data = features.loc[
        (features["bidirectional_first_seen_ms"] >= start_ts)
        & (features["bidirectional_first_seen_ms"] < end_ts)
    ]

    number_flows = snapshot_data.shape[0]
    print(f"Snapshot {interval_number} - flows: {number_flows}")

    if number_flows == 0:
        log_file.close()
        return stats

    label_col = "Stage" if "Stage" in snapshot_data.columns else None

    if label_col is not None:
        flow_type = snapshot_data[label_col].fillna("Benign")
        benign_flows = int((flow_type == "Benign").sum())
        malicious_flows = int((flow_type != "Benign").sum())
    else:
        benign_flows = number_flows
        malicious_flows = 0

    full_benign = malicious_flows == 0 and benign_flows == number_flows
    full_malicious = malicious_flows == number_flows and benign_flows == 0
    mixed = malicious_flows != 0 and benign_flows != 0

    print(f"Benign: {benign_flows} - Malicious: {malicious_flows}")

    columns_to_drop = [
        "id", "expiration_id", "src_ip", "src_mac", "src_oui", "src_port",
        "dst_ip", "dst_mac", "dst_oui", "dst_port", "protocol", "ip_version",
        "vlan_id", "tunnel_id", "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms", "src2dst_first_seen_ms", "src2dst_last_seen_ms",
        "dst2src_first_seen_ms", "dst2src_last_seen_ms",
        "Activity", "Stage", "DefenderResponse", "Signature"
    ]

    valid_drop_cols = [c for c in columns_to_drop if c in snapshot_data.columns]

    # Il loader robusto restituisce inizialmente colonne di tipo stringa.
    # Selezioniamo le feature previste e le convertiamo esplicitamente in numerico.
    non_numeric_feature_cols = {
        "application_name",
        "application_category_name",
        "requested_server_name",
        "client_fingerprint",
        "server_fingerprint",
        "user_agent",
        "content_type",
        "application_is_guessed",
    }

    feature_cols = [
        col
        for col in snapshot_data.columns
        if col not in valid_drop_cols
        and col not in non_numeric_feature_cols
    ]

    snapshot_features_df = snapshot_data[feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    snapshot_features_df = (
        snapshot_features_df
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    if snapshot_features_df.shape[1] == 0:
        raise RuntimeError("No numeric flow features were selected")

    print(f"Flow features selected: {snapshot_features_df.shape[1]}")
    print("Flow feature columns:", list(snapshot_features_df.columns))

    snapshot_features = snapshot_features_df.to_numpy(dtype=float)

    create_graph(
        out_path=out_path,
        capture_name=capture_name,
        graph_type="tdg_graph",
        snapshot_data=snapshot_data,
        snapshot_features=snapshot_features,
        interval_number=interval_number,
        stats=stats,
        log_file=log_file,
        full_benign=full_benign,
        full_malicious=full_malicious,
        mixed=mixed,
        also_etdg=graph_type,
        etdg_mode=etdg_mode,
        approx_node_threshold=approx_node_threshold,
        approx_ratio=approx_ratio,
    )

    log_file.close()
    return stats


def read_single_file(data_path, out_path, graph_type, snapshot_interval,
                     etdg_mode, approx_node_threshold, approx_ratio, result_rules_path):
    capture_name = os.path.splitext(os.path.basename(result_rules_path))[0]
    print(f"Considering capture {capture_name}")

    capture_out_dir = os.path.join(out_path, capture_name)
    stats_file = os.path.join(capture_out_dir, "stats.json")

    # Skip fully completed captures
    if os.path.isfile(stats_file):
        print(f"[SKIP COMPLETE] {capture_name} already has stats.json")
        with open(stats_file, "r") as f:
            return json.load(f)

    partial_stats = {
        "tdg_graph": {},
        "etdg_graph": {},
    }

    if graph_type in ["eTDG", "ALL"]:
        os.makedirs(os.path.join(capture_out_dir, "etdg_graph", "full_benign"), exist_ok=True)
        os.makedirs(os.path.join(capture_out_dir, "etdg_graph", "mixed"), exist_ok=True)
        os.makedirs(os.path.join(capture_out_dir, "etdg_graph", "full_malicious"), exist_ok=True)

    if graph_type in ["TDG", "ALL"]:
        os.makedirs(os.path.join(capture_out_dir, "tdg_graph", "full_benign"), exist_ok=True)
        os.makedirs(os.path.join(capture_out_dir, "tdg_graph", "mixed"), exist_ok=True)
        os.makedirs(os.path.join(capture_out_dir, "tdg_graph", "full_malicious"), exist_ok=True)

    log_file_path = os.path.join(capture_out_dir, f"{capture_name}.txt")

    features = load_labeled_csv_robust(result_rules_path)
    print(f"Loaded {capture_name}: {features.shape}")

    features["bidirectional_first_seen_ms"] = pd.to_numeric(
        features["bidirectional_first_seen_ms"], errors="coerce"
    )
    features = features.dropna(subset=["bidirectional_first_seen_ms"]).copy()
    features["bidirectional_first_seen_ms"] = features["bidirectional_first_seen_ms"].astype(np.int64)

    features.sort_values("bidirectional_first_seen_ms", inplace=True)
    features.reset_index(drop=True, inplace=True)

    start_ts = features.iloc[0]["bidirectional_first_seen_ms"]
    end_ts = features.iloc[-1]["bidirectional_first_seen_ms"]
    total_snapshots = ceil((end_ts - start_ts) / snapshot_interval) if start_ts < end_ts else 1

    print(f"Number of snapshots: {total_snapshots}")

    time_intervals_list = []
    start_snap_ts = start_ts
    for _ in range(total_snapshots):
        end_snap_ts = start_snap_ts + snapshot_interval
        time_intervals_list.append((start_snap_ts, end_snap_ts))
        start_snap_ts = end_snap_ts

    resume_from = get_resume_snapshot_index(capture_out_dir, graph_type)
    print(f"[RESUME] {capture_name} starting from snapshot {resume_from}")

    if resume_from < total_snapshots:
        f = functools.partial(
            generate_graph_from_snapshot,
            out_path,
            log_file_path,
            capture_name,
            features,
            graph_type,
            etdg_mode,
            approx_node_threshold,
            approx_ratio,
        )

        time_intervals = [[time_intervals_list[idx], idx] for idx in range(resume_from, total_snapshots)]

        with Pool(1) as p:
            stats_snapshots_list = p.map(f, time_intervals)

        for snapshot in stats_snapshots_list:
            for snapshot_representation in snapshot.keys():
                if snapshot_representation not in partial_stats:
                    partial_stats[snapshot_representation] = {}
                for graph_name, graph_stats in snapshot[snapshot_representation].items():
                    partial_stats[snapshot_representation][graph_name] = graph_stats
    else:
        print(f"[NOTHING TO DO] {capture_name}: all graph snapshots already present")

    complete_stats = build_complete_stats_from_saved_graphs(
        capture_name=capture_name,
        capture_out_dir=capture_out_dir,
        graph_repre=graph_type,
        features=features,
        time_intervals_list=time_intervals_list,
        partial_stats=partial_stats,
    )

    with open(stats_file, "w") as stats_writer:
        json.dump(complete_stats, stats_writer, indent=2)

    return complete_stats


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing labeled Unraveled CSV flows")
    parser.add_argument("--out_path", type=str, required=True, help="Path where graphs will be saved")
    parser.add_argument("--graph_repre", type=str, default="ALL", choices=REP_LIST, help="TDG, eTDG, or ALL")
    parser.add_argument("--dataset", type=str, default="UNRAVELED", help="Dataset name")
    parser.add_argument("--snapshot_interval", type=int, default=150000, help="Snapshot time in ms")
    parser.add_argument("--etdg_mode", type=str, default="exact", choices=["exact", "approx"],
                        help="How to compute eTDG structural features")
    parser.add_argument("--approx_node_threshold", type=int, default=2000,
                        help="Use approximate eTDG when number of nodes >= threshold")
    parser.add_argument("--approx_ratio", type=float, default=0.5,
                        help="Sampling ratio for approximate centrality computation")
    parser.add_argument("--compute_stats", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    result_rules_paths = glob.glob(os.path.join(args.data_path, "*.csv"))

    result_rules_paths = [
        p for p in result_rules_paths
        if os.path.basename(p).endswith("_Flow_labeled.csv")
    ]

    result_rules_paths = sorted(result_rules_paths, key=os.path.getsize)

    print("Files to process:")
    for path in result_rules_paths:
        print(path)

    stats = {}

    if not args.compute_stats:
        f = functools.partial(
            read_single_file,
            args.data_path,
            args.out_path,
            args.graph_repre,
            args.snapshot_interval,
            args.etdg_mode,
            args.approx_node_threshold,
            args.approx_ratio,
        )
        stats_list = [f(path) for path in result_rules_paths]

        for capture in stats_list:
            for capture_name in capture.keys():
                stats[capture_name] = capture[capture_name]

        with open(os.path.join(args.out_path, "stats_all_captures.json"), "w") as fp:
            json.dump(stats, fp, indent=2)
    else:
        cnt_dict = {}
        for result_rules_path in result_rules_paths:
            capture_name = os.path.splitext(os.path.basename(result_rules_path))[0]
            print(f"Considering capture {capture_name}")
            stats[capture_name] = {}

            capture_dir = os.path.join(args.out_path, capture_name)
            if os.path.isdir(capture_dir):
                for graph_type_dir in os.listdir(capture_dir):
                    graph_dir = os.path.join(capture_dir, graph_type_dir)
                    if os.path.isdir(graph_dir):
                        stats[capture_name][graph_type_dir] = {}
                        cnt_dict.setdefault(graph_type_dir, {})

                        for class_graph in os.listdir(graph_dir):
                            class_dir = os.path.join(graph_dir, class_graph)
                            if os.path.isdir(class_dir):
                                stats[capture_name][graph_type_dir][class_graph] = 0
                                cnt_dict[graph_type_dir].setdefault(class_graph, 0)

                                for file in os.listdir(class_dir):
                                    if file.endswith(".pkl") and "full_graph" not in file:
                                        stats[capture_name][graph_type_dir][class_graph] += 1
                                        cnt_dict[graph_type_dir][class_graph] += 1

        print(f"Counters: {cnt_dict}")
        with open(os.path.join(args.out_path, f"{args.dataset}_dataset_stats.json"), "w") as outfile:
            json.dump(stats, outfile, indent=2)


if __name__ == "__main__":
    main()