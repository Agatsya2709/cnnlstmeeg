# Curated Repository Manifest

This public repository contains the active implementation and the audited reporting evidence only.

| Path | Purpose |
| --- | --- |
| `models/baselines.py` | CNN, LSTM, and CNN-LSTM model definitions |
| `models/hybrid_cnn_lstm_gnn.py` | GCN, GAT, and graph-enhanced hybrid definitions |
| `models/graph_utils.py` | Coordinate parsing, RBF k-NN graph, and normalization |
| `train_compare.py` | Shared data loading, split, optimization, evaluation, and plotting workflow |
| `scripts/MIND_Get_EDF.py` | EEGBCI EDF acquisition helper |
| `scripts/prepare_dataset.py` | Filtering, event extraction, epoching, and CSV export |
| `scripts/audit_results.py` | Integrity and recomputation checks |
| `scripts/aggregate_multiseed_results.py` | Mean and sample-standard-deviation aggregation |
| `scripts/export_separate_graphs.py` | Standalone figure export |
| `run_clean_audit_pack.ps1` | Exact reported three-seed run configuration |
| `assets/electrode_positions.txt` | Bundled electrode coordinate metadata |
| `results/reported/` | Reported CSV/JSON evidence for seeds 11, 23, and 37 |

Excluded by design: raw EDF recordings, generated trial matrices, model checkpoints, TensorBoard logs, legacy single-split outputs, downloaded source repositories, and zip archives.
