"""
The one check this project cannot ship without.

Train carries 30 days of data that test does not, and those days ARE the risk
window the labels describe. If a feature ever reads them, CV looks great and
the leaderboard collapses. `features.assert_no_leakage` guards that -- but a
guard nobody has seen fail is not evidence, so test 2 deliberately removes the
clip and requires the guard to raise.

Run:  python src/test_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

SOURCES = [("dailyActivity_merged.csv", "ActivityDate"),
           ("sleepDay_merged.csv", "SleepDay"),
           ("training_sessions.csv", "date"),
           ("weightLogInfo_merged.csv", "Date")]


def test_guard_passes_on_clipped_data():
    F.assert_no_leakage("train")
    F.assert_no_leakage("test")


def test_guard_fires_when_clip_removed():
    """Negative control: without the clip the guard must raise."""
    orig = F._clip_window
    F._clip_window = lambda df, col: df
    try:
        F.assert_no_leakage("train")
    except AssertionError as e:
        print("  guard fired as expected: " + str(e))
        return
    finally:
        F._clip_window = orig
    raise AssertionError("guard did NOT fire on unclipped train data")


def test_train_and_test_windows_identical():
    for fname, col in SOURCES:
        span = {}
        for split in ("train", "test"):
            d = pd.read_csv(F.DATA / split / fname, parse_dates=[col], usecols=[col])
            c = F._clip_window(d, col)
            span[split] = (c[col].min(), c[col].max())
        assert span["train"] == span["test"], (
            fname + " windows differ: " + str(span))
        print("  " + fname.ljust(28) + str(span["train"][0].date())
              + " .. " + str(span["train"][1].date()) + "  (both splits)")


def test_raw_train_really_overruns_test():
    a = pd.read_csv(F.DATA / "train" / "dailyActivity_merged.csv",
                    parse_dates=["ActivityDate"], usecols=["ActivityDate"])
    b = pd.read_csv(F.DATA / "test" / "dailyActivity_merged.csv",
                    parse_dates=["ActivityDate"], usecols=["ActivityDate"])
    extra = (a["ActivityDate"].max() - b["ActivityDate"].max()).days
    assert extra == 30, "expected 30 extra train days, got " + str(extra)
    print("  raw train ends " + str(a["ActivityDate"].max().date())
          + ", test ends " + str(b["ActivityDate"].max().date())
          + " -> " + str(extra) + " unusable days")


if __name__ == "__main__":
    for fn in [test_guard_passes_on_clipped_data,
               test_guard_fires_when_clip_removed,
               test_train_and_test_windows_identical,
               test_raw_train_really_overruns_test]:
        print("\n== " + fn.__name__)
        fn()
    print("\nALL LEAKAGE TESTS PASSED")
