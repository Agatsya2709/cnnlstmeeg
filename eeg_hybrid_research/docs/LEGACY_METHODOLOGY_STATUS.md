# Legacy Benchmark Status

This repository is the auditable record of the original six-model benchmark, not the corrected follow-up study.

## What The Legacy Protocol Did

- Selected 90 fists-versus-feet trials from EEGBCI participants 1 and 2.
- Applied an 8--30 Hz band-pass filter, extracted 0.5--3.5 s post-cue epochs, and resampled each epoch to 128 time points over 64 channels.
- Created a stratified 72/18 outer trial split with nine test trials per class.
- Fitted `StandardScaler` on the full 72-trial outer-training partition and transformed the 18-trial outer-test partition.
- Created the seed-specific internal 80/20 training/validation split after scaling.

Thus, the outer test set was not used for preprocessing, checkpoint selection, or training. However, validation trials influenced the scaler statistics. This is validation-set information leakage and prevents the experiment from being described as a leakage-controlled model-selection protocol.

## What Has Been Verified

`results/reported/multiseed_raw.csv` contains the model-level values for seeds 11, 23, and 37. `scripts/verify_legacy_results.py` recalculates each mean and sample standard deviation and compares it to `multiseed_summary.csv`. The committed summary values match the raw table.

Because the same 18 test trials are reused across seeds, the reported standard deviations quantify training-run variation conditional on that test set. They are not confidence intervals; one changed test prediction equals 5.56 percentage points.

## Corrected Follow-Up

The separate [singaleegstudy repository](https://github.com/Agatsya2709/singaleegstudy) contains the replacement preprocessing protocol, graph topology ablation code, EEGNet reference model, prediction-level exports, and a preliminary seed-11 evidence package. Its results must remain separate until the planned multi-seed rerun is complete.
