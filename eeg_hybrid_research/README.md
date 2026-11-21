# Experiment Package

This directory contains the active implementation used for the committed pilot-study results. It is intentionally standalone: the spatial coordinate file formerly referenced from an external source tree is bundled at `assets/electrode_positions.txt`.

## Model inventory

| Identifier | Implementation | Role |
| --- | --- | --- |
| `cnn` | `models/baselines.py::CNNBaseline` | Temporal-spatial convolution baseline |
| `lstm` | `models/baselines.py::LSTMBaseline` | Sequence baseline |
| `cnn_lstm` | `models/baselines.py::CNNLSTMBaseline` | Convolutional-recurrent baseline |
| `gcn_hybrid` | `models/hybrid_cnn_lstm_gnn.py::HybridCNNLSTMGCN` | CNN + LSTM + GCN fusion |
| `gat_hybrid` | `models/hybrid_cnn_lstm_gnn.py::HybridCNNLSTMGAT` | CNN + LSTM + multi-head GAT fusion |
| `graph_enhanced_cnn` | `models/hybrid_cnn_lstm_gnn.py::GraphEnhancedCNNGAT` | Compact CNN with graph-attention branch |

`CompactCNNGAT` is an exploratory architecture in the source code and is not part of the six-model reported experiment.

## Reproduce the reported configuration

```powershell
py -3.7 -m pip install -r requirements_py37.txt
.\run_clean_audit_pack.ps1 -Seeds "11,23,37"
py -3.7 scripts/aggregate_multiseed_results.py --results-root . --results-template "results_clean_all_models_s12_ff_seed{seed}" --seeds "11,23,37" --out-prefix results_clean_all_models_s12_ff_multiseed
```

The runner uses:

- binary `fists_feet` trials from participants 1 and 2;
- 14 epochs, batch size 32, AdamW learning rate `1e-3`, and weight decay `5e-4`;
- internal validation fraction `0.2`, early-stopping patience `8`;
- label smoothing `0.02`, augmentation standard deviation `0.01`, and gradient clipping `1.0`;
- Gaussian electrode-distance graph with `k=10` and `sigma=0.65`.

The generated output folders are ignored by Git. The committed audit evidence is under `results/reported/`.

## Dataset preparation

```powershell
py -3.7 scripts/MIND_Get_EDF.py --subjects 1,2 --runs 6,10,14 --out-dir data/raw_edf
py -3.7 scripts/prepare_dataset.py --raw-dir data/raw_edf --out-dir data/processed/tune_s12_ff --subjects 1,2 --runs 6,10,14 --task fists_feet --tmin 0.5 --tmax 3.5
```

Raw downloads and processed trial matrices are deliberately excluded from Git. The committed `data/dataset_meta.json` and `data/channel_names.csv` describe the reported input shape and channel order.

## Audit and dashboard

```powershell
py -3.7 scripts/audit_results.py --results-dir results_clean_all_models_s12_ff_seed23 --val-size 0.2
py -3.7 -m streamlit run ui_dashboard.py
```

The dashboard reads locally generated output folders; it is not a replacement for the CSV/JSON audit record.

Read [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) and [docs/RESULTS.md](docs/RESULTS.md) before citing the reported metrics.
