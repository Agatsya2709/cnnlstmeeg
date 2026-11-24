# Dataset Provenance Verification

The committed metadata previously inherited a broader run list from an earlier preparation configuration. The audited 90-trial fists-versus-feet dataset was verified directly against the local source EDF inventory and uses only the following six files:

| Participant | Runs | Selected trials per run | Total |
| --- | --- | ---: | ---: |
| 1 | 6, 10, 14 | 15 | 45 |
| 2 | 6, 10, 14 | 15 | 45 |

For participant 1, each selected run contributed seven both-fists and eight both-feet trials. For participant 2, each selected run contributed eight both-fists and seven both-feet trials. The combined dataset therefore contains 45 trials per class and 90 trials in total.

The extractor applied the configured 8--30 Hz band-pass filter, the 0.5--3.5 s post-cue epoch window, and resampling to 128 time points. Every contributing recording produced a `(15, 64, 128)` epoch array with the same 64-channel order. The published `data/dataset_meta.json` has been corrected to list runs 6, 10, and 14 only.

This verification establishes input provenance and consistency. It does not change the legacy scaler limitation or the interpretation of the reported three-seed results.
