"""
EDA figures for the PlayHack ML Track Round-1 deck.

Round 1 requires a PPT covering EDA, key insights, methodology, models,
evaluation and results. This script produces the figures for the EDA and
results sections into reports/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(REPORTS / name)
    plt.close(fig)
    print("  wrote reports/" + name)


def main():
    tr = F.get_features("train")
    labels = pd.read_csv(F.DATA / "train" / "train_labels.csv")
    df = tr.merge(labels, left_on="Id", right_on="athlete_id")
    y = df["injured_in_risk_window"]

    # 1. the data-split diagram: why only 30 of the 60 train days are usable
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.barh(["train (labels)", "test"], [30, 30], left=0, color="#4C78A8",
            label="observation window (features)")
    ax.barh(["train (labels)"], [30], left=30, color="#E45756",
            label="risk window (labels; absent from test)")
    ax.set_xlabel("day index from 2026-01-05")
    ax.set_title("Train has 60 days, test has 30 - features must stop at day 30")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "01_data_split.png")

    # 2. label distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    y.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#4C78A8", "#E45756"])
    axes[0].set_title("injured_in_risk_window\n(" + format(y.mean() * 100, ".1f") + "% positive)")
    axes[0].set_xlabel("label")
    inj = df[y == 1]
    axes[1].hist(inj["onset_day_offset"], bins=30, color="#E45756")
    axes[1].set_title("onset_day_offset (near-uniform 1-30)")
    axes[2].hist(inj["recovery_duration"], bins=16, color="#F58518")
    axes[2].set_title("recovery_duration (5-20, right-skewed)")
    save(fig, "02_label_distributions.png")

    # 3. injury rate by categorical driver
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col in zip(axes, ["sport", "position", "gender"]):
        rate = df.groupby(col, observed=True)["injured_in_risk_window"].agg(["mean", "size"])
        rate = rate[rate["size"] >= 20].sort_values("mean")
        ax.barh(rate.index.astype(str), rate["mean"], color="#4C78A8")
        ax.axvline(y.mean(), color="#E45756", ls="--", lw=1, label="base rate")
        ax.set_title("injury rate by " + col)
        ax.set_xlabel("P(injured)")
        ax.legend(fontsize=7)
    save(fig, "03_injury_rate_by_group.png")

    # 4. prior injuries: the strongest single metadata signal
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    pr = df.groupby("prior_season_injury_count")["injured_in_risk_window"].agg(["mean", "size"])
    pr = pr[pr["size"] >= 15]
    axes[0].plot(pr.index, pr["mean"], "o-", color="#E45756")
    axes[0].axhline(y.mean(), color="grey", ls="--", lw=1)
    axes[0].set_xlabel("prior_season_injury_count")
    axes[0].set_ylabel("P(injured)")
    axes[0].set_title("Prior injuries vs risk")
    ages = pd.cut(df["age"], bins=[15, 20, 23, 26, 29, 50])
    ar = df.groupby(ages, observed=True)["injured_in_risk_window"].mean()
    axes[1].bar([str(i) for i in ar.index], ar.values, color="#4C78A8")
    axes[1].axhline(y.mean(), color="#E45756", ls="--", lw=1)
    axes[1].set_title("Risk by age band")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "04_metadata_drivers.png")

    # 5. workload features: injured vs not
    cands = ["sesh_acwr", "load_acwr", "sesh_monotony", "load_strain",
             "resthr_delta_last_first", "sleep_mean", "rest_days", "hr_mean"]
    cands = [c for c in cands if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes.ravel(), cands):
        d0 = df.loc[y == 0, col].dropna()
        d1 = df.loc[y == 1, col].dropna()
        lo, hi = np.nanpercentile(pd.concat([d0, d1]), [1, 99])
        bins = np.linspace(lo, hi, 35)
        ax.hist(d0, bins=bins, alpha=0.6, density=True, label="healthy", color="#4C78A8")
        ax.hist(d1, bins=bins, alpha=0.6, density=True, label="injured", color="#E45756")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(cands):]:
        ax.axis("off")
    fig.suptitle("Workload / recovery features: injured vs healthy", y=1.01)
    save(fig, "05_workload_distributions.png")

    # 6. correlation of top numeric features with the label
    num = df.select_dtypes(include=[np.number]).drop(
        columns=["Id", "athlete_id", "onset_day_offset", "recovery_duration"],
        errors="ignore")
    corr = num.corrwith(y).dropna().sort_values(key=np.abs, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = ["#E45756" if v > 0 else "#4C78A8" for v in corr.values[::-1]]
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors)
    ax.set_title("Top 20 |correlation| with injured_in_risk_window")
    ax.set_xlabel("Pearson r")
    save(fig, "06_top_correlations.png")

    print("EDA figures done.")


if __name__ == "__main__":
    main()
