# Reported Experiment Protocol

## Task and split

The reported pilot experiment classifies fists-versus-feet motor imagery from EEGBCI runs 6, 10, and 14 for participants 1 and 2. Ninety selected trials were split once with a stratified random outer 80/20 train/test split (72/18 trials). The same outer split and each listed seed were used for all six models. Seeds 11, 23, and 37 were aggregated with the sample standard deviation (`ddof=1`).

This is not a subject-independent evaluation: both participants can occur in each outer partition. It should therefore be described as a within-cohort pilot benchmark, not as evidence of cross-subject clinical generalization.

## Preprocessing

The data-preparation script downloads EDF recordings through MNE/EEGBCI, extracts task events, epochs from 0.5 to 3.5 seconds, and serializes the trial matrices. The scaler is fitted on the 72-trial outer-training partition. An internal validation split is then created from that partition. Consequently, validation samples influenced the scaler fit; outer-test samples did not. This validation-only leakage limitation is retained for transparency.

## Graph construction

`models/graph_utils.py` creates a Gaussian/RBF weighted graph from the bundled electrode coordinates:

`w_ij = exp(-d_ij^2 / (2 sigma^2))`.

For the reported experiment, each electrode retains ten nearest neighbours (`k=10`) and `sigma=0.65`. The directed k-NN mask is union-symmetrized; self-loops are added before symmetric degree normalization. The resulting spatial graph is supplied to `gcn_hybrid`, `gat_hybrid`, and `graph_enhanced_cnn`.

## Optimization

The exact runner is `run_clean_audit_pack.ps1`. It invokes `train_compare.py` with 14 epochs, batch size 32, learning rate `0.001`, AdamW weight decay `0.0005`, validation fraction 0.2, patience 8, label smoothing 0.02, noise augmentation standard deviation 0.01, and gradient clipping at 1.0. The compact graph-attention options are dropout 0.35 and DropEdge 0.10.

## Integrity checks

Each run writes `integrity_report.json`, `run_info.json`, model histories, and `summary.csv`. The audit script recomputes metrics from prediction artifacts where available and records class-distribution and prediction-agreement diagnostics. The exact raw and aggregated result tables committed under `results/reported/` are the reporting source of truth.
