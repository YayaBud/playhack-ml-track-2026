"""
Full out-of-fold threshold sweep against the official metric.

Purpose: test, rather than assume, that flagging ~99% of athletes is optimal.
The 24% -> 99% move was driven by a coarse 0.05-step sweep with no
confusion-matrix accounting and no sensitivity analysis. This does the job
properly.

Inputs are all OUT-OF-FOLD (written from inside the CV loops in final.py /
build_final_models.py), so every athlete is scored by models that never trained
on them. Nothing is refit here -- this only re-thresholds existing predictions.

  reports/oof_blend.npy            classifier probability, 3000 athletes
  reports/oof_onset_day_offset.npy onset prediction, 1050 injured athletes
  reports/final_oof_recovery.npy   recovery prediction, 1050 injured athletes

Scoring (problem statement PDF page 6):
  Task A  F1 over all athletes.
  Task B  evaluated ONLY over athletes truly injured in the risk window.
          A hit (truly injured, flagged) contributes |pred - true|.
          A miss (truly injured, not flagged) contributes PENALTY = 30 to BOTH
          timing heads. A false positive is not in this population and so
          contributes nothing to Task B -- it only costs Task A precision.
          skill = max(0, 1 - MAE_model / MAE_baseline), baseline = training mean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import PENALTY, ONSET_RANGE, RECOVERY_RANGE

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Task A weight -> Task B weight = 1 - w. Task B score = mean(skill_on, skill_rc).
WEIGHTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def sweep(y, p, on_true, on_pred, rc_true, rc_pred, grid):
    inj = y == 1
    n_inj = int(inj.sum())
    base_on = float(np.abs(on_true - on_true.mean()).mean())
    base_rc = float(np.abs(rc_true - rc_true.mean()).mean())
    err_on = np.abs(on_pred - on_true)
    err_rc = np.abs(rc_pred - rc_true)

    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & inj).sum())
        fp = int(((pred == 1) & ~inj).sum())
        tn = int(((pred == 0) & ~inj).sum())
        fn = int(((pred == 0) & inj).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        hit = pred[inj] == 1
        mae_on = float(np.where(hit, err_on, PENALTY).mean())
        mae_rc = float(np.where(hit, err_rc, PENALTY).mean())
        s_on = max(0.0, 1 - mae_on / base_on)
        s_rc = max(0.0, 1 - mae_rc / base_rc)

        # how much of the Task B error is penalty vs genuine prediction error
        pen_mass = PENALTY * fn
        hit_mass = float(err_on[hit].sum())
        rows.append(dict(
            thr=float(t), flagged=float(pred.mean()), tp=tp, fp=fp, tn=tn, fn=fn,
            precision=prec, recall=rec, f1=f1, missed=fn,
            mae_onset=mae_on, mae_recovery=mae_rc,
            skill_onset=s_on, skill_recovery=s_rc,
            taskB=(s_on + s_rc) / 2,
            penalty_total=pen_mass,
            penalty_share_of_onset_err=pen_mass / (pen_mass + hit_mass) if (pen_mass + hit_mass) else 0.0,
            mean3=(f1 + s_on + s_rc) / 3,
        ))
    df = pd.DataFrame(rows)
    for w in WEIGHTS:
        df["score_%d_%d" % (round(w * 100), round((1 - w) * 100))] = (
            w * df["f1"] + (1 - w) * df["taskB"])
    return df, base_on, base_rc, n_inj


def main():
    lab = pd.read_csv(ROOT / "data" / "train" / "train_labels.csv")
    y = lab["injured_in_risk_window"].to_numpy(int)
    inj = y == 1
    on_true = lab.loc[inj, "onset_day_offset"].to_numpy(float)
    rc_true = lab.loc[inj, "recovery_duration"].to_numpy(float)

    p = np.load(REPORTS / "oof_blend.npy")
    on_pred = np.clip(np.load(REPORTS / "oof_onset_day_offset.npy"), *ONSET_RANGE)
    rc_pred = np.clip(np.load(REPORTS / "final_oof_recovery.npy"), *RECOVERY_RANGE)
    assert len(p) == len(y) and len(on_pred) == inj.sum() == len(rc_pred)

    grid = np.round(np.arange(0.0, 1.0001, 0.01), 4)
    shipped_thr = json.loads((REPORTS / "metrics.json").read_text())["threshold_oof"]
    grid = np.unique(np.concatenate([grid, [shipped_thr]]))

    df, base_on, base_rc, n_inj = sweep(y, p, on_true, on_pred, rc_true, rc_pred, grid)
    df.to_csv(REPORTS / "threshold_sweep.csv", index=False)

    print("OOF sweep: n=%d, injured=%d (%.1f%%), grid=%d thresholds (0.01 steps)"
          % (len(y), n_inj, 100 * y.mean(), len(grid)))
    print("Task B baselines (training MEAN): onset %.4f  recovery %.4f   PENALTY=%.0f"
          % (base_on, base_rc, PENALTY))
    print("Break-even recall for nonzero skill: onset %.4f, recovery %.4f\n"
          % ((PENALTY - base_on) / (PENALTY - 2.63), (PENALTY - base_rc) / (PENALTY - 2.89)))

    show = df[df.thr.isin(np.round(np.arange(0, 1.001, 0.05), 4))]
    print("=== sweep (every 0.05) ===")
    print("  thr  flag%   TP   FP   FN  prec   rec    F1   |missed| mae_on skl_on skl_rc  taskB  mean3")
    for _, r in show.iterrows():
        print("  %.2f %5.1f%% %4d %4d %4d %.3f %.3f %.4f | %4d  %6.3f %.4f %.4f %.4f %.4f" % (
            r.thr, 100 * r.flagged, r.tp, r.fp, r.fn, r.precision, r.recall, r.f1,
            r.missed, r.mae_onset, r.skill_onset, r.skill_recovery, r.taskB, r.mean3))

    print("\n=== fine detail 0.00-0.20 (0.01 steps) — where the action is ===")
    fine = df[(df.thr <= 0.20)]
    print("  thr  flag%   FN  recall    F1   skl_on skl_rc  taskB  mean3  pen_share")
    for _, r in fine.iterrows():
        mark = "  <-- SHIPPED" if abs(r.thr - shipped_thr) < 1e-9 else ""
        print("  %.4f %5.1f%% %4d  %.4f %.4f  %.4f %.4f %.4f %.4f  %5.1f%%%s" % (
            r.thr, 100 * r.flagged, r.fn, r.recall, r.f1, r.skill_onset,
            r.skill_recovery, r.taskB, r.mean3,
            100 * r.penalty_share_of_onset_err, mark))

    print("\n=== best threshold per Task A / Task B weighting ===")
    print("  weighting     best_thr  flag%    FN  recall  prec     F1   skl_on skl_rc  score")
    winners = {}
    for w in WEIGHTS:
        col = "score_%d_%d" % (round(w * 100), round((1 - w) * 100))
        r = df.loc[df[col].idxmax()]
        winners[w] = r
        print("  A=%2d%% B=%2d%%   %.4f %5.1f%% %4d  %.4f  %.3f  %.4f %.4f %.4f  %.4f" % (
            round(w * 100), round((1 - w) * 100), r.thr, 100 * r.flagged, r.fn,
            r.recall, r.precision, r.f1, r.skill_onset, r.skill_recovery, r[col]))

    print("\n=== where the SHIPPED threshold (%.4f) ranks under each weighting ===" % shipped_thr)
    ship = df.loc[(df.thr - shipped_thr).abs().idxmin()]
    print("  shipped: flag %.1f%%, FN=%d, recall %.4f, prec %.3f, F1 %.4f, "
          "skill %.4f/%.4f" % (100 * ship.flagged, ship.fn, ship.recall,
                               ship.precision, ship.f1, ship.skill_onset, ship.skill_recovery))
    print("  weighting     shipped   best     gap     rank / %d" % len(df))
    for w in WEIGHTS:
        col = "score_%d_%d" % (round(w * 100), round((1 - w) * 100))
        s, b = ship[col], df[col].max()
        rank = int((df[col] > s).sum()) + 1
        print("  A=%2d%% B=%2d%%   %.4f  %.4f  %+.4f   %d" % (
            round(w * 100), round((1 - w) * 100), s, b, s - b, rank))

    print("\n=== the F1-optimal threshold, for reference ===")
    f1r = df.loc[df.f1.idxmax()]
    print("  thr %.2f: flag %.1f%%, FN=%d (%.1f%% of injuries missed), F1 %.4f,"
          % (f1r.thr, 100 * f1r.flagged, f1r.fn, 100 * f1r.fn / n_inj, f1r.f1))
    print("  skill %.4f/%.4f -> under every weighting above it scores:" % (
        f1r.skill_onset, f1r.skill_recovery))
    for w in WEIGHTS:
        col = "score_%d_%d" % (round(w * 100), round((1 - w) * 100))
        print("    A=%2d%%: %.4f  (best %.4f, gap %+.4f)" % (
            round(w * 100), f1r[col], df[col].max(), f1r[col] - df[col].max()))

    print("\nwrote reports/threshold_sweep.csv (%d rows, all columns)" % len(df))


if __name__ == "__main__":
    main()
