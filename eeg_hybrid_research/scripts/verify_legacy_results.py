#!/usr/bin/env python
"""Verify the published legacy summary against committed per-seed source values."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reported"
RAW_PATH = RESULTS / "multiseed_raw.csv"
SUMMARY_PATH = RESULTS / "multiseed_summary.csv"
METRICS = {
    "test_acc": "test_acc",
    "test_f1_macro": "f1",
    "test_bal_acc": "bal_acc",
}


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def main():
    with RAW_PATH.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        summary_by_model = {row["model"]: row for row in csv.DictReader(handle)}

    models = sorted({row["model"] for row in raw_rows})
    for model in models:
        source = [row for row in raw_rows if row["model"] == model]
        summary = summary_by_model[model]
        for raw_metric, summary_metric in METRICS.items():
            values = [float(row[raw_metric]) for row in source]
            expected_mean = float(summary["{}_mean".format(summary_metric)])
            expected_std = float(summary["{}_std".format(summary_metric)])
            if not math.isclose(mean(values), expected_mean, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError("{} {} mean does not match".format(model, raw_metric))
            if not math.isclose(sample_std(values), expected_std, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError("{} {} sample standard deviation does not match".format(model, raw_metric))
    print("Verified {} models against committed raw seed values.".format(len(models)))


if __name__ == "__main__":
    main()
