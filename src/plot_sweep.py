"""
Deck figures for the threshold sweep (reports/threshold_sweep.csv).

Three panels, each answering one question a judge will ask:

  07_threshold_sweep.png   Why does the threshold sit at 0.046 and not at the
                           F1 optimum? -- the score components vs threshold,
                           showing the skill cliff.
  08_penalty_cliff.png     What does a miss actually cost? -- missed injuries
                           and the share of Task B error that is pure penalty.
  09_weight_sensitivity.png Is this robust to the unpublished Task A / Task B
                           weighting? -- winner per weighting, with our choice
                           overlaid.
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

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WEIGHTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

INK, MUTED = "#1a1a2e", "#8b8b9e"
BLUE, ORANGE, GREEN, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def main():
    df = pd.read_csv(REPORTS / "threshold_sweep.csv").sort_values("thr")
    M = json.loads((REPORTS / "metrics.json").read_text())
    ship = M["threshold_oof"]
    f1_thr = float(df.loc[df.f1.idxmax(), "thr"])

    # ---------------------------------------------------------- figure 1 ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    ax.plot(df.thr, df.f1, color=BLUE, lw=2.2, label="Task A: F1")
    ax.plot(df.thr, df.skill_onset, color=GREEN, lw=2.2, label="Task B: onset skill")
    ax.plot(df.thr, df.skill_recovery, color=ORANGE, lw=2.2, label="Task B: recovery skill")
    ax.plot(df.thr, df.mean3, color=INK, lw=2.8, ls="--", label="combined (equal weight)")

    ax.axvline(ship, color=RED, lw=1.6, ls=":")
    ax.axvline(f1_thr, color=MUTED, lw=1.6, ls=":")
    ax.annotate("our threshold %.3f\nflags 99.4%%, 0 missed" % ship,
                xy=(ship, 0.70), xytext=(0.16, 0.80), color=RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    combined_at_f1 = float(df.loc[(df.thr - f1_thr).abs().idxmin(), "mean3"])
    ax.annotate("F1-optimal %.2f\nmisses 502 of 1050 injuries\nboth skills = 0" % f1_thr,
                xy=(f1_thr, combined_at_f1), xytext=(0.50, 0.40), color=MUTED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.axvspan(0, 0.068, color=GREEN, alpha=0.07)
    ax.text(0.005, 0.045, "zero-miss zone", fontsize=8.5, color=GREEN, style="italic")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 0.92)
    ax.set_title("Optimising F1 alone scores ZERO on both Task B components",
                 fontsize=12.5, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax2.plot(df.thr, 100 * df.flagged, color=INK, lw=2, label="% athletes flagged")
    ax2.plot(df.thr, 100 * df.recall, color=GREEN, lw=2, ls="--", label="recall on injured")
    ax2.axhline(98.7, color=ORANGE, lw=1.2, ls=":")
    ax2.text(0.62, 99.6, "recall needed for any recovery skill (98.7%)",
             fontsize=8, color=ORANGE)
    ax2.axvline(ship, color=RED, lw=1.6, ls=":")
    ax2.set_xlabel("classification threshold")
    ax2.set_ylabel("%")
    ax2.set_xlim(0, 1)
    ax2.legend(loc="center right", frameon=False, fontsize=9)
    fig.suptitle("Threshold sweep — 102 thresholds, out-of-fold on 3,000 athletes",
                 fontsize=9, color=MUTED, y=0.985)
    fig.tight_layout()
    fig.savefig(REPORTS / "07_threshold_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- figure 2 ----
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.fill_between(df.thr, 0, df.missed, color=RED, alpha=0.16)
    ax.plot(df.thr, df.missed, color=RED, lw=2.2)
    ax.set_xlabel("classification threshold")
    ax.set_ylabel("injuries missed (of 1,050)", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1100)

    axb = ax.twinx()
    axb.plot(df.thr, 100 * df.penalty_share_of_onset_err, color=INK, lw=2, ls="--")
    axb.set_ylabel("% of Task B error that is pure penalty", color=INK)
    axb.set_ylim(0, 100)
    axb.spines["top"].set_visible(False)

    for t, lbl, dx, dy in [(ship, "ours", 0.08, 250), (f1_thr, "F1-optimal", 0.06, 200)]:
        r = df.iloc[(df.thr - t).abs().argmin()]
        ax.axvline(t, color=MUTED, lw=1.3, ls=":")
        ax.annotate("%s — %d missed" % (lbl, r.missed), xy=(t, r.missed),
                    xytext=(t + dx, r.missed + dy), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.set_title("Each miss costs a flat 30 — against baselines of only 7.6 and 3.2",
                 fontsize=12.5, pad=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "08_penalty_cliff.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- figure 3 ----
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ship_row = df.iloc[(df.thr - ship).abs().argmin()]
    best_thr, best_sc, ship_sc = [], [], []
    for w in WEIGHTS:
        col = "score_%d_%d" % (round(w * 100), round((1 - w) * 100))
        r = df.loc[df[col].idxmax()]
        best_thr.append(r.thr)
        best_sc.append(r[col])
        ship_sc.append(ship_row[col])

    x = np.arange(len(WEIGHTS))
    ax.bar(x - 0.19, best_sc, 0.38, label="best possible threshold", color=MUTED, alpha=0.55)
    ax.bar(x + 0.19, ship_sc, 0.38, label="our shipped threshold (0.046)", color=BLUE)
    for i, (bt, bs, ss) in enumerate(zip(best_thr, best_sc, ship_sc)):
        ax.text(i - 0.19, bs + 0.030, "thr %.2f" % bt, ha="center", fontsize=8, color=MUTED)
        gap = ss - bs
        ax.text(i + 0.19, ss + 0.008, ("%+.4f" % gap) if gap < -0.0001 else "optimal",
                ha="center", fontsize=8, fontweight="bold",
                color=RED if gap < -0.005 else GREEN)
    ax.axvspan(5.5, 6.5, color=RED, alpha=0.06)
    ax.annotate("only here does\nF1-tuning win", xy=(6, 0.52), xytext=(6, 0.575),
                ha="center", fontsize=8.5, color=RED, style="italic",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(["A=%d%%\nB=%d%%" % (round(w * 100), round((1 - w) * 100))
                        for w in WEIGHTS], fontsize=9)
    ax.set_ylabel("combined score")
    ax.set_ylim(0, 0.60)
    ax.set_title("Robust to the unpublished Task A / Task B weighting\n"
                 "our threshold is within 0.001 of optimal in 6 of 7 scenarios",
                 fontsize=12.5, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "09_weight_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------- figure 4: slide-optimised ----
    # 07 is a two-panel figure built for a document; shrunk into a slide it is
    # unreadable. This is the same data as one panel, sized and weighted for
    # projection: fewer elements, bigger type, wider aspect.
    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    ax.plot(df.thr, df.f1, color=BLUE, lw=3.2, label="Task A: F1")
    ax.plot(df.thr, df.skill_onset, color=GREEN, lw=3.2, label="Task B: onset skill")
    ax.plot(df.thr, df.skill_recovery, color=ORANGE, lw=3.2, label="Task B: recovery skill")
    ax.plot(df.thr, df.mean3, color=INK, lw=3.6, ls="--", label="combined")

    ax.axvspan(0, 0.068, color=GREEN, alpha=0.10)
    ax.axvline(ship, color=RED, lw=2.2, ls=":")
    ax.axvline(f1_thr, color=MUTED, lw=2.2, ls=":")
    ax.text(0.075, 0.845, "our threshold 0.046\n99.4% flagged · 0 missed",
            fontsize=13, color=RED, fontweight="bold", va="top")
    ax.text(f1_thr + 0.03, 0.48,
            "F1-optimal 0.38\nmisses 502 of 1,050\nboth skills → 0.000",
            fontsize=13, color=MUTED, va="top")
    ax.annotate("", xy=(0.19, 0.02), xytext=(0.19, 0.30),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2, ls=":"))
    ax.text(0.195, 0.325, "recovery skill dead past here", fontsize=11.5,
            color=ORANGE, style="italic")

    ax.set_xlabel("classification threshold", fontsize=14)
    ax.set_ylabel("score", fontsize=14)
    ax.tick_params(labelsize=12.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.95)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.set_title("Optimising F1 alone scores zero on both Task B components",
                 fontsize=16, pad=14)
    fig.tight_layout()
    fig.savefig(REPORTS / "10_sweep_slide.png", bbox_inches="tight")
    plt.close(fig)

    for f in ["07_threshold_sweep.png", "08_penalty_cliff.png",
              "09_weight_sensitivity.png", "10_sweep_slide.png"]:
        print("wrote reports/%s  (%.0f KB)" % (f, (REPORTS / f).stat().st_size / 1024))


if __name__ == "__main__":
    main()
