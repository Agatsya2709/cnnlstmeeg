# Reported Results

`results/reported/multiseed_summary.csv` is the canonical aggregate table. It summarizes three runs (seeds 11, 23, and 37) of the protocol described in `EXPERIMENT_PROTOCOL.md`.

| Model | Accuracy mean | Accuracy sample SD | Macro F1 mean | Balanced accuracy mean |
| --- | ---: | ---: | ---: | ---: |
| CNN | 0.8704 | 0.0849 | 0.8689 | 0.8704 |
| GCN hybrid | 0.8333 | 0.0556 | 0.8322 | 0.8333 |
| CNN-LSTM | 0.7593 | 0.0849 | 0.7481 | 0.7593 |
| Graph-enhanced CNN | 0.5370 | 0.0642 | 0.4151 | 0.5370 |
| GAT hybrid | 0.4815 | 0.1156 | 0.4544 | 0.4815 |
| LSTM | 0.3333 | 0.0962 | 0.3288 | 0.3333 |

The accompanying figures in `assets/figures/` are derived from this audited run series. Do not combine them with legacy single-split figures or checkpoints from earlier exploratory experiments.

The outcomes support a limited conclusion: the CNN baseline performed best in this small pilot, while the GCN hybrid was competitive. The evidence does not support a claim that graph attention improved classification accuracy.
