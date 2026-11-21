# Graph-Inductive Deep Learning for EEG Motor Imagery

Reproducible PyTorch benchmark for binary EEG motor-imagery classification using convolutional, recurrent, and graph-based neural networks. This repository is a curated implementation and evidence package for the accompanying student research project; it intentionally excludes raw EEG recordings, checkpoints, copied upstream repositories, and obsolete reporting artifacts.

## What is included

- Six active models: `cnn`, `lstm`, `cnn_lstm`, `gcn_hybrid`, `gat_hybrid`, and `graph_enhanced_cnn`.
- EDF download, preprocessing, epoching, train/test splitting, training, audit, and aggregation scripts.
- A spatial electrode graph built from 64-channel coordinates.
- The exact per-seed CSV/JSON evidence behind the reported multiseed summary.
- Paper-ready figures generated from the audited experiment.
- A Streamlit dashboard for locally generated results.

## Reported experiment

The committed results are from a binary fists-versus-feet motor-imagery pilot experiment using two participants, 90 selected trials, an outer random stratified 80/20 split, and seeds `11`, `23`, and `37`. The outer test set was not used to fit preprocessing statistics. The validation partition was created after scaling the outer training partition; this validation-only limitation is documented and should be corrected with train-fold-only scaling in future work.

| Model | Test accuracy, mean +/- sample SD | Macro F1, mean +/- sample SD |
| --- | ---: | ---: |
| CNN | 87.04% +/- 8.49% | 86.89% +/- 8.62% |
| GCN hybrid | 83.33% +/- 5.56% | 83.22% +/- 5.69% |
| CNN-LSTM | 75.93% +/- 8.49% | 74.81% +/- 9.10% |
| Graph-enhanced CNN | 53.70% +/- 6.42% | 41.51% +/- 14.16% |
| GAT hybrid | 48.15% +/- 11.56% | 45.44% +/- 14.02% |
| LSTM | 33.33% +/- 9.62% | 32.88% +/- 10.07% |

The best observed model in this small pilot is the CNN baseline. These results do **not** establish that graph models outperform convolutional models. The GCN result is competitive but the sample is too small for a strong comparative claim.

## Repository layout

```text
eeg_hybrid_research/
  assets/                 # Electrode coordinates and audited figures
  data/                   # Non-sensitive metadata only
  docs/                   # Protocol, results, and project map
  models/                 # All six model definitions and graph utilities
  results/reported/       # Committed per-seed evidence and aggregates
  scripts/                # Data preparation, audit, plot export, aggregation
  train_compare.py        # Main experiment entry point
  run_clean_audit_pack.ps1# Exact reported experiment configuration
  ui_dashboard.py         # Local Streamlit results dashboard
```

## Quick start

Tested with Python 3.7. From the repository root:

```powershell
cd eeg_hybrid_research
py -3.7 -m pip install -r requirements_py37.txt
.\run_clean_audit_pack.ps1 -Seeds "11,23,37"
```

The runner expects the prepared dataset at `data/processed/tune_s12_ff`. To rebuild it, download the required PhysioNet EDF recordings and preprocess them:

```powershell
py -3.7 scripts/MIND_Get_EDF.py --subjects 1,2 --runs 6,10,14 --out-dir data/raw_edf
py -3.7 scripts/prepare_dataset.py --raw-dir data/raw_edf --out-dir data/processed/tune_s12_ff --subjects 1,2 --runs 6,10,14 --task fists_feet --tmin 0.5 --tmax 3.5
.\run_clean_audit_pack.ps1 -Seeds "11,23,37"
```

See [the project README](eeg_hybrid_research/README.md) for detailed commands, graph settings, result locations, and limitations.

## Graph protocol

The reported runs override the training defaults and use a Gaussian/RBF spatial graph with `k=10` nearest neighbours and `sigma=0.65`. Directed k-NN selections are union-symmetrized, self-loops are added, and symmetric degree normalization is applied. The graph is constructed once per run from `assets/electrode_positions.txt` and shared by the graph-aware model variants.

## Responsible use and limitations

This is a small research/educational benchmark, not a diagnostic system or a clinical decision tool. The split is trial-random rather than subject-independent, so it does not demonstrate cross-subject generalization. Use the committed metrics only with this experimental scope and their stated variability.

## Upstream data and inspiration

The implementation uses the PhysioNet EEG Motor Movement/Imagery Dataset through MNE's EEGBCI access path. Earlier development was informed by the public `EEG-DL` and `EEG-Motor-Imagery-Classification-CNNs-TensorFlow` repositories; neither codebase is vendored here.
