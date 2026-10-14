# Network Anomaly Detection with Graph Neural Networks on UNRAVELED

**Master's thesis — MSc in Computer Engineering (Cybersecurity), University of Salerno, October 2026**

This project evaluates **existing graph-based anomaly-detection methods** on the **UNRAVELED** cybersecurity dataset. The objective is to compare temporal graph representations and established learning approaches for anomalous network traffic, not to introduce a new neural architecture.

## Research overview

Network traffic is aggregated into non-overlapping **150-second windows** and represented by temporal directed graphs. Each graph node represents a network endpoint (IP address, port); edges capture observed communication.

Two representations are compared:

| Representation | Features per node | Description |
|---|---:|---|
| **TDG** | 57 | Traffic-derived features |
| **e-TDG** | 72 | The same 57 traffic features plus 15 graph-structural features |

The topology and node labels are matched between TDG and e-TDG.

## Methods

**Six principal anomaly-detection models**, each evaluated on TDG and e-TDG:

- **Static:** DOMINANT, OC-GNN
- **Dynamic:** EGCN-H, EGCN-O, LSTM-A, GRU-A

Additional **GCN, GraphSAGE and GAT** baselines are treated separately.

The experiments follow a **one-class learning** protocol: the training and validation sets contain benign traffic, and the anomaly threshold is calibrated on benign validation data.

## Dataset and split

The definitive manifest contains **21,314 temporal graph snapshots**, split as follows:

| Subset | Benign | Mixed | Total |
|---|---:|---:|---:|
| Train | 4,411 | 0 | 4,411 |
| Validation | 1,327 | 0 | 1,327 |
| Test — Week 5 | 396 | 3,365 | 3,761 |
| Test — Week 6 | 1,266 | 2,839 | 4,105 |
| Unused | 7,646 | 64 | 7,710 |
| **Total** | **15,046** | **6,268** | **21,314** |

The snapshot manifest is named `graph_timeline_merged_150000_final_split.csv`.

**Threshold caveat:** selecting a threshold with an empirical validation FPR below 1% does **not** guarantee a test FPR below 1%.

Evaluation considers false-positive rate, true-positive rate, precision, F1 score, **AUROC and AUPRC**, with separate analyses for test Weeks 5 and 6.

## Code organization

The prepared thesis code includes:

- `unraveled_graph_representation.py` — graph construction
- `scripts/` — preprocessing, split generation, validation and structural statistics
- `anomaly_detection/methods/` — DOMINANT, OC-GNN, EGCN-H, EGCN-O, LSTM-A and GRU-A
- `anomaly_detection/train_gnn_baseline*.py` — auxiliary GNN baselines
- `anomaly_detection/plot_roc_*.py` — evaluation plots

**Repository publication status:** The project description is available here. The source code and final snapshot manifest are prepared for upload; links to files will work once that upload is completed.

## Data availability and reproducibility

The repository is intentionally not intended to distribute raw UNRAVELED PCAP data, generated graph binaries, trained model weights or intermediate experimental outputs. The complete experiments require separately obtained input data and locally generated graph datasets. Some scripts contain experiment-specific local paths and may require configuration before running in a different environment.

## References

- Myneni et al., *Unraveled — A semi-synthetic dataset for Advanced Persistent Threats*, Computer Networks, 2023.
- *Graph Neural Networks for IoT Security: A Comparative Study*, Internet of Things, 2026.

