"""
Check submission.csv against sample_submission.csv before shipping it.

Round 1 is graded from a ZIP plus a deck, so a malformed CSV is a silent way to
lose marks. This asserts the shape the organisers handed us: same columns in
the same order, same athlete_id set in the same order, integer dtypes, and
values inside the ranges observed in train_labels.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    sub = pd.read_csv(ROOT / "submission.csv")
    sample = pd.read_csv(ROOT / "data" / "test" / "sample_submission.csv")
    labels = pd.read_csv(ROOT / "data" / "train" / "train_labels.csv")

    problems = []

    if list(sub.columns) != list(sample.columns):
        problems.append("columns " + str(list(sub.columns))
                        + " != " + str(list(sample.columns)))
    if len(sub) != len(sample):
        problems.append("row count " + str(len(sub)) + " != " + str(len(sample)))
    if not sub["athlete_id"].equals(sample["athlete_id"]):
        problems.append("athlete_id column does not match sample order/contents")
    if sub.isna().any().any():
        problems.append("contains NaN: "
                        + str(sub.isna().sum()[sub.isna().sum() > 0].to_dict()))

    for c in sub.columns:
        if not pd.api.types.is_integer_dtype(sub[c]):
            problems.append(c + " is " + str(sub[c].dtype) + ", expected integer")

    inj = labels[labels["injured_in_risk_window"] == 1]
    ranges = {
        "injured_in_risk_window": (0, 1),
        "onset_day_offset": (int(inj["onset_day_offset"].min()),
                             int(inj["onset_day_offset"].max())),
        "recovery_duration": (int(inj["recovery_duration"].min()),
                              int(inj["recovery_duration"].max())),
    }
    for c, (lo, hi) in ranges.items():
        actual = (int(sub[c].min()), int(sub[c].max()))
        if actual[0] < lo or actual[1] > hi:
            problems.append(c + " range " + str(actual) + " outside train range "
                            + str((lo, hi)))

    print("submission.csv  rows=" + str(len(sub)) + "  cols=" + str(list(sub.columns)))
    for c, (lo, hi) in ranges.items():
        print("  " + c.ljust(24) + " min=" + str(int(sub[c].min())).rjust(3)
              + "  max=" + str(int(sub[c].max())).rjust(3)
              + "  mean=" + format(sub[c].mean(), ".3f")
              + "   (train range " + str(lo) + ".." + str(hi) + ")")
    rate = sub["injured_in_risk_window"].mean()
    print("  predicted positive rate " + format(rate, ".4f")
          + "  (train " + format(labels["injured_in_risk_window"].mean(), ".4f") + ")")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("\nOK - submission matches sample_submission contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
