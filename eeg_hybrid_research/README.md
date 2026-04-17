# EEG CNN + LSTM + GNN Hybrid Research Pipeline (Python 3.7)

This project integrates ideas from:

- `EEG-DL` ([source extracted in `../sources/EEG-DL-master`])
- `EEG-Motor-Imagery-Classification-CNNs-TensorFlow` ([source extracted in `../sources/EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master`])

and builds a reproducible benchmark with:

1. `cnn` baseline
2. `lstm` baseline
3. `hybrid` model (CNN + LSTM + GNN)

It keeps the original EEG-DL dataset filename convention:

- `training_set.csv`
- `training_label.csv`
- `test_set.csv`
- `test_label.csv`

## Is adding GNN useful?

Usually yes when channel relationships are meaningful. EEG sensors have spatial structure, and GNN can model cross-channel dependencies that pure CNN/LSTM may miss. In practice, gains depend on:

- dataset size and noise
- graph quality (electrode-distance graph vs correlation graph)
- task setting (2-class generally easier than 4-class)

This code builds the graph from `electrode_positions.txt` first (from the referenced repos), with fallback to correlation-based graph.

## Important integrity note

Metrics are generated from actual training/evaluation only. No fake accuracy values are produced.

## Quick start (few commands)

From this folder:

```powershell
py -3.7 -m pip install --upgrade pip
py -3.7 -m pip install -r requirements_py37.txt
py -3.7 scripts/MIND_Get_EDF.py --subjects 1-30 --runs 4,6,8,10,12,14 --out-dir data/raw_edf
py -3.7 scripts/prepare_dataset.py --raw-dir data/raw_edf --out-dir data/processed/EEG-Motor-Movement-Imagery-Dataset --subjects 1-30 --runs 4,6,8,10,12,14 --task left_right
py -3.7 train_compare.py --data-dir data/processed/EEG-Motor-Movement-Imagery-Dataset --results-dir results --models all --epochs 35
py -3.7 -m streamlit run ui_dashboard.py
```

Or run all core steps:

```powershell
.\run_pipeline.ps1
```

## Task modes

- `left_right`: binary motor imagery from runs 4/8/12 (best chance for >70%)
- `fists_feet`: binary motor imagery from runs 6/10/14
- `four_class`: full 4-class setting (harder)

Switch task via:

```powershell
py -3.7 scripts/prepare_dataset.py --task four_class ...
```

## Outputs

Saved under `results/`:

- `summary.csv`
- `model_comparison.png`
- `<model>_history.csv`
- `<model>_curves.png`
- `<model>_confusion_matrix.png`
- `<model>_best.pt`

