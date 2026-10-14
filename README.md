# GNN Evaluation on the Unraveled Dataset

This repository contains the code used for the experimental evaluation of graph-based anomaly-detection methods on the Unraveled cybersecurity dataset.

The project does not propose a new anomaly-detection algorithm or a new GNN architecture. Its purpose is to evaluate existing graph-based methods and graph representations on Unraveled under a one-class anomaly-detection setting.

**Master's thesis project — University of Salerno, October 2026.**

> **Data availability:** Raw UNRAVELED traffic, generated graph files, and trained model weights are not included. Some statistical CSV files from the supplied archive were excluded because their split labels and counts do not match the definitive experimental manifest. The source code and final snapshot manifest are retained.

## Experimental Setting

Network traffic is represented as temporal directed graphs using non-overlapping snapshots of 150,000 ms, corresponding to 2 minutes and 30 seconds.

A graph node represents a transport-layer endpoint identified by an (IP address, port) pair.

Observed communication between endpoints defines graph edges.

Node traffic features are obtained by aggregating the features of incident flows.

For the final binary node labels:

- Stage = Benign corresponds to label 0.
- Any non-Benign Stage corresponds to label 1.

## Graph Representations

Two graph representations are evaluated.

### TDG

The final TDG representation contains:

- the communication graph;
- 57 traffic-derived node features;
- node features obtained from incident-flow attributes.

Therefore, TDG has 57 node features.

### e-TDG

The final e-TDG representation uses the same communication topology as the corresponding TDG and adds 15 structural node features to the 57 traffic features.

Therefore, e-TDG has 72 node features:

57 traffic features + 15 structural features = 72 features.

The 15 structural features used in the final implementation are:

1. Graph degree centrality
2. Graph betweenness
3. Graph closeness
4. Component degree centrality
5. Component betweenness
6. Component closeness
7. Second-order centrality
8. Node-assigned edge betweenness
9. Eccentricity
10. Barycenter indicator
11. Radius
12. Center indicator
13. Component size
14. Degree
15. 2-hop ego node count

The e-TDG representation is an enriched alternative representation and is not a mandatory processing stage after TDG.

## Definitive Dataset Split

The exact snapshot manifest used in the final experiments is stored in:

`graph_timeline_merged_150000_final_split.csv`

The definitive split contains 21,314 graph snapshots.

| Split | Benign | Mixed | Total | Purpose |
| --- | ---: | ---: | ---: | --- |
| Train | 4,411 | 0 | 4,411 | One-class training |
| Validation | 1,327 | 0 | 1,327 | Validation and threshold calibration |
| Test Week 5 | 396 | 3,365 | 3,761 | Final evaluation |
| Test Week 6 | 1,266 | 2,839 | 4,105 | Final evaluation |
| Unused | 7,646 | 64 | 7,710 | Not used in final runs |
| Total | 15,046 | 6,268 | 21,314 | Complete timeline |

Training and validation contain benign snapshots only.

The final experiments therefore follow a one-class classification protocol.

The operating threshold is selected using benign validation data so that the empirical validation false-positive rate is below 1%.

This validation calibration condition does not imply that the false-positive rate must remain below 1% on unseen Week 5 or Week 6 test snapshots.

## Main Anomaly-Detection Methods

Six existing graph-based anomaly-detection methods are evaluated.

### Static Methods

- DOMINANT
- OC-GNN

### Dynamic Methods

- EGCN-H
- EGCN-O
- LSTM-A
- GRU-A

Each method is evaluated using both TDG and e-TDG.

This produces 12 definitive experiments:

6 methods x 2 graph representations = 12 experiments.

The corresponding training scripts are:

- `anomaly_detection/methods/dominant/train_dominant_unraveled.py`
- `anomaly_detection/methods/ocgnn/train_ocgnn_unraveled.py`
- `anomaly_detection/methods/egcn_h/train_egcn_h_unraveled.py`
- `anomaly_detection/methods/egcn_o/train_egcn_o_unraveled.py`
- `anomaly_detection/methods/lstm_a/train_lstm_a_unraveled.py`
- `anomaly_detection/methods/gru_a/train_gru_a_unraveled.py`

The methodological selection of the six methods and the TDG/e-TDG representations follows the comparative graph-based IoT anomaly-detection study used as the reference work for this project.

## Auxiliary GNN Baselines

Additional GNN baseline experiments are available in:

- `anomaly_detection/train_gnn_baseline.py`
- `anomaly_detection/train_gnn_baseline_tdg.py`

These auxiliary GCN, GraphSAGE and GAT experiments are separate from the six main comparative methods.

## Core Processing Pipeline

The principal scripts used for the definitive experiment are:

- `scripts/prepare_merged_net101x_csvs_150000.py`
- `scripts/build_graph_timeline_merged_150000.py`
- `scripts/recompute_final_split_structural_stats.py`
- `scripts/run_full_exact_graph_generation.sh`
- `scripts/validate_exact_graph_dataset.py`

The final graph-generation implementation is:

`unraveled_graph_representation.py`

## Exact Graph Generation

The final graph-generation pipeline uses a snapshot duration of 150,000 ms.

The complete generation process can be launched with:

`bash scripts/run_full_exact_graph_generation.sh`

The generated graph dataset is stored locally under:

`unraveled_graphs_150000_merged_captures_exact_57_72/`

This directory is intentionally excluded from Git because it contains large generated graph files.

The definitive dataset contains corresponding TDG and e-TDG graphs for the same temporal snapshots.

## Graph Validation

The generated graph dataset can be checked with:

`python scripts/validate_exact_graph_dataset.py`

The final validation procedure checks properties including:

- TDG feature dimension equal to 57;
- e-TDG feature dimension equal to 72;
- matching TDG/e-TDG topology;
- matching TDG/e-TDG node mappings;
- matching node labels;
- correspondence between the 57 TDG traffic features and the first 57 e-TDG features;
- absence of NaN and infinite feature values;
- binary node labels.

## Running the Main Methods

Each anomaly-detection method has its own training script under `anomaly_detection/methods/`.

The scripts operate on the definitive graph dataset and use:

`graph_timeline_merged_150000_final_split.csv`

as the final experimental split.

Available command-line parameters for each method can be inspected with:

`python <training_script> --help`

This should be used when selecting TDG or e-TDG and other experiment parameters.

## Evaluation Protocol

Final evaluation is performed at node level.

Week 5 and Week 6 are evaluated separately in order to preserve the temporal evolution of the attacks in Unraveled.

The reported metrics are:

- False Positive Rate (FPR)
- True Positive Rate (TPR)
- Precision
- F1 score
- Area Under the ROC Curve (AUC)

FPR, TPR, Precision and F1 describe performance at the selected validation-calibrated operating threshold.

AUC describes the ranking quality of the continuous anomaly scores across possible thresholds.

Metrics are computed for individual graph snapshots and then aggregated across the corresponding test snapshots.

## Repository Policy

Large datasets and generated artifacts are intentionally excluded from version control.

The repository does not include:

- raw Unraveled data;
- generated graph pickle files;
- experiment output directories;
- model checkpoints;
- log files;
- temporary comparison outputs;
- presentation artifacts;
- backup scripts;
- intermediate local analysis files.

The definitive split manifest is versioned because it documents exactly which graph snapshots belong to training, validation, Week 5 test and Week 6 test sets.

## References

The experimental dataset is based on:

Myneni et al., "Unraveled - A semi-synthetic dataset for Advanced Persistent Threats", Computer Networks, 2023.

The graph representations and principal anomaly-detection methods follow the methodological reference study:

"Graph Neural Networks for IoT Security: A Comparative Study", Internet of Things, 2026.
